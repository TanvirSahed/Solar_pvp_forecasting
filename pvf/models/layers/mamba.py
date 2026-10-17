"""Mamba (selective SSM) mixer in pure PyTorch, with optional CUDA kernels.

Backends (chosen by `backend=`):
  "cuda"  - official `mamba_ssm` fused kernel (Linux + nvcc; use the Docker image)
  "pscan" - Blelloch parallel scan from `mambapy` (pure PyTorch, works on Windows, fast enough for L<=1k)
  "seq"   - reference sequential scan (slow, simplest to read/modify)
  "auto"  - cuda > pscan > seq

Everything here is intentionally explicit so you can modify the SSM itself (e.g. inject a physics state,
condition delta on irradiance, tie B/C across nodes ...).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:  # pragma: no cover - optional
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _cuda_scan
except Exception:  # noqa: BLE001
    _cuda_scan = None
try:
    from mambapy.pscan import pscan as _pscan
except Exception:  # noqa: BLE001
    _pscan = None


def selective_scan_seq(x, delta, A, B, C, D):
    """x, delta: [b, l, d]; A: [d, n]; B, C: [b, l, n]; D: [d] -> y [b, l, d]"""
    dA = torch.exp(delta.unsqueeze(-1) * A)                    # b l d n
    dBx = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
    h = torch.zeros_like(dA[:, 0])
    ys = []
    for t in range(x.shape[1]):
        h = dA[:, t] * h + dBx[:, t]
        ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
    return torch.stack(ys, 1) + x * D


def selective_scan_pscan(x, delta, A, B, C, D):
    L = x.shape[1]
    Lp = 1 << (L - 1).bit_length()                              # mambapy pscan wants a power of two
    dA = torch.exp(delta.unsqueeze(-1) * A)
    dBx = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
    if Lp != L:  # pad at the END with identity steps: earlier states are unaffected
        dA = F.pad(dA, (0, 0, 0, 0, 0, Lp - L), value=1.0)
        dBx = F.pad(dBx, (0, 0, 0, 0, 0, Lp - L))
    h = _pscan(dA, dBx)[:, :L]
    return (h * C.unsqueeze(2)).sum(-1) + x * D


class MambaMixer(nn.Module):
    """Mamba-1 block mixer (Gu & Dao 2023, ref/foundational/F04). Input/output [b, l, d_model]."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dt_rank: int | str = "auto", backend: str = "auto", dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.backend = self._pick(backend)

    @staticmethod
    def _pick(backend: str) -> str:
        if backend == "auto":
            return "cuda" if (_cuda_scan is not None and torch.cuda.is_available()) else ("pscan" if _pscan else "seq")
        if backend == "cuda" and _cuda_scan is None:
            raise ImportError("mamba_ssm not installed (Linux/Docker only)")
        if backend == "pscan" and _pscan is None:
            raise ImportError("pip install mambapy")
        return backend

    def ssm(self, x):  # x: [b, l, d_inner] (after conv + SiLU)
        A = -torch.exp(self.A_log.float())
        dbc = self.x_proj(x)
        dt, B, C = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        if self.backend == "cuda":
            y = _cuda_scan(x.transpose(1, 2).contiguous(), delta.transpose(1, 2).contiguous(), A,
                           B.transpose(1, 2).contiguous(), C.transpose(1, 2).contiguous(), self.D.float())
            return y.transpose(1, 2)
        with torch.autocast(device_type=x.device.type, enabled=False):  # scans are unstable in fp16
            args = (x.float(), delta.float(), A, B.float(), C.float(), self.D.float())
            return selective_scan_pscan(*args) if self.backend == "pscan" else selective_scan_seq(*args)

    def forward(self, u):
        L = u.shape[1]
        x, z = self.in_proj(u).chunk(2, dim=-1)
        x = rearrange(self.conv1d(rearrange(x, "b l d -> b d l"))[..., :L], "b d l -> b l d")
        y = self.ssm(F.silu(x)).to(u.dtype)
        return self.out_proj(y * F.silu(z))


class MambaLayer(nn.Module):
    """Pre-norm residual Mamba layer; bidirectional=True runs a second mixer on the reversed sequence
    (Bi-Mamba / S-Mamba style, needed when the 'sequence' is a set of variates or graph nodes)."""

    def __init__(self, d_model: int, d_state=16, d_conv=4, expand=2, dropout=0.1, bidirectional=False,
                 backend="auto", ffn: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = MambaMixer(d_model, d_state, d_conv, expand, backend=backend)
        self.bwd = MambaMixer(d_model, d_state, d_conv, expand, backend=backend) if bidirectional else None
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4 * d_model, d_model)) if ffn else None

    def forward(self, x):
        h = self.norm(x)
        y = self.fwd(h)
        if self.bwd is not None:
            y = y + self.bwd(h.flip(1)).flip(1)
        x = x + self.drop(y)
        if self.ffn is not None:
            x = x + self.drop(self.ffn(x))
        return x
