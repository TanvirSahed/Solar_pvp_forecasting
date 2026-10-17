"""Reusable building blocks. Models are compositions of these - swap, rewire or fuse them freely."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """Reversible instance norm (Kim et al. 2022). Normalises each channel over time; `denorm` inverts it
    for a chosen channel. Note: PV inputs are already physically scaled, so RevIN is optional here."""

    def __init__(self, n_ch: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(n_ch)) if affine else None
        self.b = nn.Parameter(torch.zeros(n_ch)) if affine else None

    def norm(self, x):  # [b, l, c]
        self.mu = x.mean(1, keepdim=True).detach()
        self.sd = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps).detach()
        x = (x - self.mu) / self.sd
        return x * self.w + self.b if self.w is not None else x

    def denorm(self, y, ch: int):  # y: [b, h] for channel ch
        if self.w is not None:
            y = (y - self.b[ch]) / (self.w[ch] + self.eps)
        return y * self.sd[:, 0, ch:ch + 1] + self.mu[:, 0, ch:ch + 1]


class SeriesDecomp(nn.Module):
    """Moving-average trend / seasonal split (Autoformer, DLinear, PowerMamba). x: [b, l, c]."""

    def __init__(self, kernel: int = 25):
        super().__init__()
        self.k = kernel

    def forward(self, x):
        pad_l, pad_r = (self.k - 1) // 2, self.k // 2
        xp = torch.cat([x[:, :1].expand(-1, pad_l, -1), x, x[:, -1:].expand(-1, pad_r, -1)], 1)
        trend = F.avg_pool1d(xp.transpose(1, 2), self.k, stride=1).transpose(1, 2)
        return x - trend, trend


class PatchEmbed(nn.Module):
    """[b, l, c] -> [b*c, n_patch, d] (channel-independent patches, PatchTST style)."""

    def __init__(self, patch_len: int, stride: int, d_model: int, lookback: int, dropout: float = 0.0):
        super().__init__()
        self.p, self.s = patch_len, stride
        self.n_patch = (lookback - patch_len) // stride + 2  # +1 for end padding (PatchTST)
        self.proj = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patch, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, l, c = x.shape
        x = x.permute(0, 2, 1).reshape(b * c, l)
        x = torch.cat([x, x[:, -1:].expand(-1, self.s)], 1)  # replicate-pad the end by one stride
        x = x.unfold(-1, self.p, self.s)                      # [b*c, n_patch, p]
        return self.drop(self.proj(x) + self.pos), c


def mlp(d_in: int, d_hidden: int, d_out: int, n_layers: int = 2, dropout: float = 0.1, act=nn.GELU):
    layers, d = [], d_in
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d, d_hidden), act(), nn.Dropout(dropout)]
        d = d_hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class HorizonHead(nn.Module):
    """Global representation z [b, d] (+ known-future covariates x_fut [b, H, c_f]) -> forecast [b, H(, Q)].

    use_fut=True adds a per-step shared MLP over [proj(z)_h, x_fut_h] so every backbone can exploit future
    clear-sky / calendar / NWP information in the same way (fair comparison, one switch to ablate)."""

    def __init__(self, d_in: int, horizon: int, n_fut: int = 0, use_fut: bool = True, d_step: int = 32,
                 n_quantiles: int = 0, dropout: float = 0.1):
        super().__init__()
        self.H, self.use_fut = horizon, use_fut and n_fut > 0
        self.q = n_quantiles
        out = max(1, n_quantiles)
        if self.use_fut:
            self.to_steps = nn.Linear(d_in, horizon * d_step)
            self.step_mlp = mlp(d_step + n_fut, 2 * d_step, out, 2, dropout)
        else:
            self.lin = nn.Linear(d_in, horizon * out)

    def forward(self, z, x_fut=None):
        b = z.shape[0]
        if self.use_fut:
            s = self.to_steps(z).view(b, self.H, -1)
            y = self.step_mlp(torch.cat([s, x_fut], -1))
        else:
            y = self.lin(z).view(b, self.H, -1)
        return y if self.q else y.squeeze(-1)


class GradReverse(torch.autograd.Function):
    """Gradient reversal layer for DANN (Ganin et al. 2016, ref/foundational/F14)."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam: float = 1.0):
    return GradReverse.apply(x, lam)


class FourierKAN(nn.Module):
    """Kolmogorov-Arnold layer with Fourier basis: y_j = sum_i sum_k a_ijk cos(k x_i) + b_ijk sin(k x_i) + bias_j.
    (FourierKAN variant used by RFKAN, R16). Grid size = number of harmonics."""

    def __init__(self, d_in: int, d_out: int, grid: int = 4, base: bool = True):
        super().__init__()
        self.k = nn.Parameter(torch.arange(1, grid + 1).float(), requires_grad=False)
        self.coef = nn.Parameter(torch.randn(2, d_out, d_in, grid) / (math.sqrt(d_in) * math.sqrt(grid)))
        self.bias = nn.Parameter(torch.zeros(d_out))
        self.base = nn.Linear(d_in, d_out, bias=False) if base else None  # residual SiLU path (KAN paper)

    def forward(self, x):
        kx = x.unsqueeze(-1) * self.k                                   # [..., d_in, g]
        y = torch.einsum("...ig,oig->...o", torch.cos(kx), self.coef[0]) + \
            torch.einsum("...ig,oig->...o", torch.sin(kx), self.coef[1]) + self.bias
        return y + self.base(F.silu(x)) if self.base is not None else y
