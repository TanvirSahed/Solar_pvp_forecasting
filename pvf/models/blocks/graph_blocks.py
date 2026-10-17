"""Dense graph blocks for small node sets (variables, or 4 inverters).

Dense [nodes, nodes] adjacency on purpose: with <= ~50 nodes it is faster than sparse message passing, needs no
torch_geometric, and makes adjacency trivial to edit, learn or mix.

adjacency_mode (selection report, model 12):
    "none"      identity, no message passing          -> the "Concat-Mamba" ablation
    "physics"   fixed physics_adjacency (Digital Twin / electrical topology)
    "learned"   softmax(relu(E1 E2^T)) from node embeddings (Graph WaveNet / MTGNN)
    "mixed"     alpha * physics + (1 - alpha) * learned, alpha predicted from a context vector
Convention: adjacency[i, j] = 1 means node j sends information to node i.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(adjacency: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    """Symmetric GCN normalisation D^-1/2 (A + I) D^-1/2 for [nodes, nodes] or [batch, nodes, nodes]."""
    if add_self_loops:
        adjacency = adjacency + torch.eye(adjacency.shape[-1], device=adjacency.device, dtype=adjacency.dtype)
    inverse_sqrt_degree = adjacency.sum(-1).clamp_min(1e-6).pow(-0.5)
    return inverse_sqrt_degree.unsqueeze(-1) * adjacency * inverse_sqrt_degree.unsqueeze(-2)


class AdjacencyBuilder(nn.Module):
    def __init__(self, n_nodes: int, adjacency_mode: str = "learned", physics_adjacency=None,
                 node_embedding_size: int = 16, context_size: int = 0):
        super().__init__()
        if adjacency_mode not in ("none", "physics", "learned", "mixed"):
            raise ValueError(f"unknown adjacency_mode {adjacency_mode!r}")
        self.adjacency_mode, self.n_nodes = adjacency_mode, n_nodes
        physics = (torch.as_tensor(physics_adjacency, dtype=torch.float32) if physics_adjacency is not None
                   else torch.ones(n_nodes, n_nodes))
        self.register_buffer("physics_adjacency", physics / physics.sum(-1, keepdim=True).clamp_min(1e-6))
        if adjacency_mode in ("learned", "mixed"):
            self.source_embedding = nn.Parameter(torch.randn(n_nodes, node_embedding_size) * 0.1)
            self.target_embedding = nn.Parameter(torch.randn(n_nodes, node_embedding_size) * 0.1)
        if adjacency_mode == "mixed":
            self.physics_trust_gate = nn.Linear(max(context_size, 1), 1)

    def forward(self, context: torch.Tensor | None = None, batch_size: int = 1):     # -> [batch, nodes, nodes]
        if self.adjacency_mode == "none":
            return torch.eye(self.n_nodes, device=self.physics_adjacency.device).expand(batch_size, -1, -1)
        if self.adjacency_mode == "physics":
            return self.physics_adjacency.expand(batch_size, -1, -1)
        learned = F.softmax(F.relu(self.source_embedding @ self.target_embedding.T), dim=-1)
        if self.adjacency_mode == "learned":
            return learned.expand(batch_size, -1, -1)
        context = context if context is not None else torch.zeros(batch_size, 1, device=learned.device)
        physics_trust = torch.sigmoid(self.physics_trust_gate(context)).view(-1, 1, 1)
        self.last_physics_trust = physics_trust.detach()    # inspect: how much the model trusts physics vs data
        return physics_trust * self.physics_adjacency + (1 - physics_trust) * learned


class GraphConvolution(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_states, adjacency):      # node_states [batch, nodes, size], adjacency [batch, nodes, nodes]
        return self.dropout(F.gelu(normalize_adjacency(adjacency) @ self.linear(node_states)))


class GraphAttention(nn.Module):
    """Multi-head graph attention (GATv2 scoring, Brody et al. 2022). The adjacency is a soft prior:
    attention logits += log(adjacency), so an all-ones adjacency means plain attention over all nodes."""

    def __init__(self, input_size: int, output_size: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if output_size % n_heads:
            raise ValueError("output_size must be divisible by n_heads")
        self.n_heads, self.head_size = n_heads, output_size // n_heads
        self.projection = nn.Linear(input_size, output_size, bias=False)
        self.attention_vector = nn.Parameter(torch.randn(n_heads, self.head_size) * 0.1)
        self.output = nn.Linear(output_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_states, adjacency):
        batch, n_nodes, _ = node_states.shape
        heads = self.projection(node_states).view(batch, n_nodes, self.n_heads, self.head_size)
        pair_features = F.leaky_relu(heads.unsqueeze(2) + heads.unsqueeze(1), 0.2)   # [b, i, j, heads, head_size]
        logits = (pair_features * self.attention_vector).sum(-1) + torch.log(adjacency.unsqueeze(-1) + 1e-6)
        attention = self.dropout(torch.softmax(logits, dim=2))                        # over senders j
        self.last_attention = attention.detach()
        messages = torch.einsum("bijh,bjhd->bihd", attention, heads).reshape(batch, n_nodes, -1)
        return self.output(F.gelu(messages))
