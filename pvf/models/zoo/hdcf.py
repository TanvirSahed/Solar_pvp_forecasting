"""HDCF-Mamba (Shi et al. 2026, Energies 19(5):1315 -> ref/R06_HDCF-Mamba.pdf), re-implemented from Sec. 3.

    E    = X W_v + b_v + PE + TimeEmb(X_mark)                         (Eq. 2, DataEmbedding)
    Z    = Mamba(E)                       single block, global branch (Sec. 3.5)
    M_i  = Upsample_L(AvgPool_{k=7,s_i}(E)),  s_i in {2,4,8}          (Eq. 3-4)
    M    = TimeBlock(stack_i M_i)  [b,3,L,d] -> Inception(6x Conv2d 3x3) x2, residual + LayerNorm (Eq. 6)
    M_f  = ReLU(LN(reshape(M) W_proj))                               (Eq. 7-8)
    H    = concat_time(M_f, Z)  [b, 2L, d]                            (Eq. 11, temporal concatenation)
    Y    = Linear_d->1( Linear_2L->T(H) )                             (Eq. 12-15, two-step linear decoder)

Deviation (switchable): use_fut=True appends the known-future covariates to each decoded step before the last
projection (the paper has no future covariates; set use_fut=false for the faithful version).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import ForecastModel, register
from ..layers.mamba import MambaLayer


def sinusoidal_pe(L, d):
    pe = torch.zeros(L, d)
    pos = torch.arange(L).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class Inception2d(nn.Module):
    """n parallel 3x3 Conv2d kernels, averaged (TimesNet 'Inception_Block_V1' style as described in Sec. 3.4.2)."""

    def __init__(self, c_in, c_out, n_kernels=6):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv2d(c_in, c_out, 3, padding=1) for _ in range(n_kernels))

    def forward(self, x):
        return torch.stack([c(x) for c in self.convs], -1).mean(-1)


class MSFEM(nn.Module):
    def __init__(self, d_model, d_ff=32, kernel=7, strides=(2, 4, 8), n_kernels=6):
        super().__init__()
        self.k, self.strides = kernel, strides
        S = len(strides)
        self.block = nn.Sequential(Inception2d(S, d_ff, n_kernels), nn.GELU(), Inception2d(d_ff, S, n_kernels))
        self.ln = nn.LayerNorm([S, d_model])
        self.proj = nn.Linear(S * d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, E):  # [b, L, d]
        b, L, d = E.shape
        x = E.transpose(1, 2)
        Ms = [F.interpolate(F.avg_pool1d(x, self.k, s, padding=self.k // 2), size=L, mode="linear",
                            align_corners=False).transpose(1, 2) for s in self.strides]
        M = torch.stack(Ms, 1)                                            # [b, S, L, d]
        M = self.ln((M + self.block(M)).permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        M = M.permute(0, 2, 1, 3).reshape(b, L, -1)                       # [b, L, S*d]
        return F.relu(self.ln2(self.proj(M)))


@register("hdcf_mamba")
class HDCFMamba(ForecastModel):
    def __init__(self, spec, d_model=64, d_state=16, expand=2, d_ff=32, strides=(2, 4, 8), dropout=0.1,
                 use_fut=True, backend="auto", **kw):
        super().__init__(spec, **kw)
        self.cal_idx = [i for i, n in enumerate(spec.hist_names) if n.startswith(("hod", "doy"))]
        self.val_idx = [i for i in range(spec.n_hist) if i not in self.cal_idx]
        self.value = nn.Linear(len(self.val_idx), d_model)
        self.temporal = nn.Linear(len(self.cal_idx), d_model)
        self.register_buffer("pe", sinusoidal_pe(spec.lookback, d_model), persistent=False)
        self.drop = nn.Dropout(dropout)
        self.mamba = MambaLayer(d_model, d_state, expand=expand, dropout=dropout, backend=backend)
        self.msfem = MSFEM(d_model, d_ff, strides=tuple(strides))
        self.time_map = nn.Linear(2 * spec.lookback, spec.horizon)
        self.use_fut = use_fut
        self.out = nn.Linear(d_model + (spec.n_fut if use_fut else 0), max(1, self.n_q))

    def forward(self, batch):
        xh = batch["x_hist"]
        E = self.drop(self.value(xh[..., self.val_idx]) + self.temporal(xh[..., self.cal_idx]) + self.pe)
        H = torch.cat([self.msfem(E), self.mamba(E)], 1)                  # [b, 2L, d]
        Hs = self.time_map(H.transpose(1, 2)).transpose(1, 2)              # [b, T, d]
        if self.use_fut:
            Hs = torch.cat([Hs, batch["x_fut"]], -1)
        y = self.out(Hs)
        return {"pred": y if self.n_q else y.squeeze(-1)}
