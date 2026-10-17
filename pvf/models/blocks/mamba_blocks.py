"""Mamba (selective state space) blocks in pure PyTorch, with optional CUDA kernels.

Symbols inside the scan follow the Mamba paper (ref/foundational/F04): for every step t
    state_t = exp(delta_t * A) * state_{t-1} + delta_t * B_t * x_t        y_t = C_t . state_t + D * x_t
A (decay), B (input gate), C (readout) and delta (step size) are input-dependent except A and D.

scan_backend: "cuda"       official mamba_ssm fused kernel (Linux + nvcc; Docker image)
              "parallel"   Blelloch parallel scan from mambapy (pure PyTorch, works on Windows)
              "sequential" step-by-step loop (slow, easiest to read and modify)
              "auto"       cuda > parallel > sequential
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:  # pragma: no cover - optional, Linux only
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _cuda_selective_scan
except Exception:  # noqa: BLE001
    _cuda_selective_scan = None
try:
    from mambapy.pscan import pscan as _parallel_scan
except Exception:  # noqa: BLE001
    _parallel_scan = None


def selective_scan_sequential(x, delta, A, B, C, D):
    """x, delta: [batch, time, channels]; A: [channels, state]; B, C: [batch, time, state]; D: [channels]."""
    decay = torch.exp(delta.unsqueeze(-1) * A)                              # [batch, time, channels, state]
    state_input = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
    state = torch.zeros_like(decay[:, 0])
    outputs = []
    for t in range(x.shape[1]):
        state = decay[:, t] * state + state_input[:, t]
        outputs.append((state * C[:, t].unsqueeze(1)).sum(-1))
    return torch.stack(outputs, 1) + x * D


def _scan_chunk(x, delta, A, B, C, initial_state):
    """Parallel scan of one time chunk that starts from `initial_state` [batch, channels, state]."""
    time = x.shape[1]
    padded_time = 1 << (time - 1).bit_length()                              # mambapy wants a power of two
    decay = torch.exp(delta.unsqueeze(-1) * A)                              # [batch, time, channels, state]
    state_input = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
    if padded_time != time:  # pad at the END with identity steps: earlier states are unaffected
        decay_padded = F.pad(decay, (0, 0, 0, 0, 0, padded_time - time), value=1.0)
        state_input = F.pad(state_input, (0, 0, 0, 0, 0, padded_time - time))
    else:
        decay_padded = decay
    states = _parallel_scan(decay_padded, state_input)[:, :time]            # as if the chunk started from 0
    states = states + torch.cumprod(decay, dim=1) * initial_state.unsqueeze(1)   # carry the previous chunk in
    return (states * C.unsqueeze(2)).sum(-1), states[:, -1]


def selective_scan_parallel(x, delta, A, B, C, D, chunk_steps: int = 64):
    """Chunked parallel scan with gradient checkpointing: memory grows with chunk_steps, not sequence length
    (a 192-step graph model drops from ~14 GB to a few GB)."""
    from torch.utils.checkpoint import checkpoint

    state = x.new_zeros(x.shape[0], x.shape[2], A.shape[1])
    outputs = []
    for start in range(0, x.shape[1], chunk_steps):
        piece = slice(start, start + chunk_steps)
        arguments = (x[:, piece], delta[:, piece], A, B[:, piece], C[:, piece], state)
        if torch.is_grad_enabled():
            y, state = checkpoint(_scan_chunk, *arguments, use_reentrant=False)
        else:
            y, state = _scan_chunk(*arguments)
        outputs.append(y)
    return torch.cat(outputs, 1) + x * D


def _choose_backend(scan_backend: str) -> str:
    if scan_backend == "auto":
        if _cuda_selective_scan is not None and torch.cuda.is_available():
            return "cuda"
        return "parallel" if _parallel_scan is not None else "sequential"
    if scan_backend == "cuda" and _cuda_selective_scan is None:
        raise ImportError("mamba_ssm is not installed (Linux/Docker only)")
    if scan_backend == "parallel" and _parallel_scan is None:
        raise ImportError("pip install mambapy")
    return scan_backend


class MambaMixer(nn.Module):
    """One Mamba-1 mixer: [batch, time, model_size] -> same shape."""

    def __init__(self, model_size: int, state_size: int = 16, conv_kernel_size: int = 4, expansion_factor: int = 2,
                 scan_backend: str = "auto", min_step_size=1e-3, max_step_size=1e-1):
        super().__init__()
        self.inner_size = int(expansion_factor * model_size)
        self.step_rank = math.ceil(model_size / 16)
        self.state_size = state_size
        self.input_projection = nn.Linear(model_size, 2 * self.inner_size, bias=False)
        self.local_convolution = nn.Conv1d(self.inner_size, self.inner_size, conv_kernel_size,
                                           groups=self.inner_size, padding=conv_kernel_size - 1)
        self.selection_projection = nn.Linear(self.inner_size, self.step_rank + 2 * state_size, bias=False)
        self.step_size_projection = nn.Linear(self.step_rank, self.inner_size, bias=True)
        step = torch.exp(torch.rand(self.inner_size) * (math.log(max_step_size) - math.log(min_step_size))
                         + math.log(min_step_size))
        with torch.no_grad():
            self.step_size_projection.bias.copy_(step + torch.log(-torch.expm1(-step)))   # inverse softplus
        self.log_decay = nn.Parameter(torch.log(torch.arange(1, state_size + 1, dtype=torch.float32)
                                                .repeat(self.inner_size, 1)))         # A = -exp(log_decay)
        self.skip_gain = nn.Parameter(torch.ones(self.inner_size))                    # D
        self.output_projection = nn.Linear(self.inner_size, model_size, bias=False)
        self.scan_backend = _choose_backend(scan_backend)

    def selective_scan(self, x):                                  # x: [batch, time, inner_size]
        A = -torch.exp(self.log_decay.float())
        step_low_rank, B, C = torch.split(self.selection_projection(x), [self.step_rank, self.state_size,
                                                                         self.state_size], dim=-1)
        delta = F.softplus(self.step_size_projection(step_low_rank))
        if self.scan_backend == "cuda":
            y = _cuda_selective_scan(x.transpose(1, 2).contiguous(), delta.transpose(1, 2).contiguous(), A,
                                     B.transpose(1, 2).contiguous(), C.transpose(1, 2).contiguous(),
                                     self.skip_gain.float())
            return y.transpose(1, 2)
        with torch.autocast(device_type=x.device.type, enabled=False):              # scans are unstable in fp16
            arguments = (x.float(), delta.float(), A, B.float(), C.float(), self.skip_gain.float())
            if self.scan_backend == "parallel":
                return selective_scan_parallel(*arguments)
            return selective_scan_sequential(*arguments)

    def forward(self, sequence):
        time = sequence.shape[1]
        x, gate = self.input_projection(sequence).chunk(2, dim=-1)
        x = rearrange(self.local_convolution(rearrange(x, "b t c -> b c t"))[..., :time], "b c t -> b t c")
        y = self.selective_scan(F.silu(x)).to(sequence.dtype)
        return self.output_projection(y * F.silu(gate))


class MambaLayer(nn.Module):
    """Pre-norm residual Mamba layer. bidirectional=True adds a mixer over the reversed sequence
    (needed when the "sequence" is a set of variables or graph nodes, which has no natural order)."""

    def __init__(self, model_size: int, state_size: int = 16, conv_kernel_size: int = 4, expansion_factor: int = 2,
                 dropout: float = 0.1, bidirectional: bool = False, use_feedforward: bool = False,
                 scan_backend: str = "auto"):
        super().__init__()
        self.norm = nn.LayerNorm(model_size)
        self.forward_mixer = MambaMixer(model_size, state_size, conv_kernel_size, expansion_factor, scan_backend)
        self.backward_mixer = (MambaMixer(model_size, state_size, conv_kernel_size, expansion_factor, scan_backend)
                               if bidirectional else None)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = nn.Sequential(
            nn.LayerNorm(model_size), nn.Linear(model_size, 4 * model_size), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * model_size, model_size)) if use_feedforward else None

    def forward(self, sequence):
        normalized = self.norm(sequence)
        mixed = self.forward_mixer(normalized)
        if self.backward_mixer is not None:
            mixed = mixed + self.backward_mixer(normalized.flip(1)).flip(1)
        sequence = sequence + self.dropout(mixed)
        if self.feedforward is not None:
            sequence = sequence + self.dropout(self.feedforward(sequence))
        return sequence
