"""Dense graph layers for small node sets (4 inverters, or a handful of variables).

Dense [N, N] adjacency is used on purpose: with N <= ~50 it is faster than sparse message passing, needs no
torch_geometric, and makes it trivial to edit/learn/mix adjacency matrices.

Adjacency modes (selection report, model 12):
    none      - identity (no message passing)      -> "Concat-Mamba" ablation
    physics   - fixed A_physics (Digital-Twin / electrical topology)
    learned   - adaptive, A = softmax(relu(E1 E2^T)) (Graph WaveNet / MTGNN)
    mixed     - A_t = alpha_t * A_physics + (1 - alpha_t) * A_learned, alpha_t from a context gate
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(A: torch.Tensor, add_self: bool = True) -> torch.Tensor:
    """Symmetric GCN normalisation D^-1/2 (A+I) D^-1/2; works for [N,N] or [b,N,N]."""
    if add_self:
        A = A + torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    d = A.sum(-1).clamp_min(1e-6).pow(-0.5)
    return d.unsqueeze(-1) * A * d.unsqueeze(-2)


class AdjacencyLearner(nn.Module):
    def __init__(self, n_nodes: int, mode: str = "learned", A_physics=None, emb_dim: int = 16, d_ctx: int = 0):
        super().__init__()
        self.mode, self.N = mode, n_nodes
        A0 = torch.as_tensor(A_physics, dtype=torch.float32) if A_physics is not None else torch.ones(n_nodes, n_nodes)
        self.register_buffer("A_phys", A0 / A0.sum(-1, keepdim=True).clamp_min(1e-6))
        if mode in ("learned", "mixed"):
            self.E1 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
            self.E2 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        if mode == "mixed":
            # alpha_t depends on context (e.g. pooled node states, DT residual, weather regime)
            self.gate = nn.Linear(max(d_ctx, 1), 1)

    def forward(self, ctx: torch.Tensor | None = None, batch: int = 1):
        if self.mode == "none":
            return torch.eye(self.N, device=self.A_phys.device).expand(batch, -1, -1)
        if self.mode == "physics":
            return self.A_phys.expand(batch, -1, -1)
        A_l = F.softmax(F.relu(self.E1 @ self.E2.T), dim=-1)
        if self.mode == "learned":
            return A_l.expand(batch, -1, -1)
        c = ctx if ctx is not None else torch.zeros(batch, 1, device=A_l.device)
        alpha = torch.sigmoid(self.gate(c)).view(-1, 1, 1)
        self.last_alpha = alpha.detach()  # inspect how much the model trusts physics vs data
        return alpha * self.A_phys + (1 - alpha) * A_l


class DenseGCN(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, A):  # x [b, N, d], A [b, N, N]
        return self.drop(F.gelu(normalize_adj(A) @ self.lin(x)))


class DenseGAT(nn.Module):
    """Multi-head GAT (v2 scoring by default, Brody et al. 2022) restricted/biased by adjacency A.
    A acts as a soft prior: logits += log(A + eps); A=all-ones -> plain full attention."""

    def __init__(self, d_in: int, d_out: int, heads: int = 4, dropout: float = 0.1, v2: bool = True):
        super().__init__()
        assert d_out % heads == 0
        self.h, self.dh, self.v2 = heads, d_out // heads, v2
        self.W = nn.Linear(d_in, d_out, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.out = nn.Linear(d_out, d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, A):
        b, N, _ = x.shape
        h = self.W(x).view(b, N, self.h, self.dh)                       # b N H dh
        if self.v2:
            e = F.leaky_relu(h.unsqueeze(2) + h.unsqueeze(1), 0.2)       # b N N H dh
            logits = (e * self.a_src).sum(-1)                            # b N N H
        else:
            logits = F.leaky_relu((h * self.a_src).sum(-1).unsqueeze(2) + (h * self.a_dst).sum(-1).unsqueeze(1), 0.2)
        logits = logits + torch.log(A.unsqueeze(-1) + 1e-6)
        att = self.drop(torch.softmax(logits, dim=2))                    # normalise over neighbours j
        self.last_att = att.detach()
        y = torch.einsum("bijh,bjhd->bihd", att, h).reshape(b, N, -1)
        return self.out(F.gelu(y))
