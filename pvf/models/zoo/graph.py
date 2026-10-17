"""Graph + temporal hybrids (selection report rows 11-12).

graph_mamba  - "DT-informed GAT/GCN + Mamba" (SET Lab research core). Per-node temporal Mamba encoder ->
               graph message passing with adjacency A_t = alpha_t*A_physics + (1-alpha_t)*A_learned -> readout.
               The report's mandatory ablation chain is just config:
                   Concat-Mamba          : model=mamba
                   GCN+Mamba             : graph_mamba  conv=gcn adj=learned
                   GAT+Mamba             : graph_mamba  conv=gat adj=learned
                   DT-informed GAT+Mamba : graph_mamba  conv=gat adj=mixed   (A_physics from the twin)

NODES. The report assumes 4 inverter nodes (Inv1..Inv4). The current datasets hold ONE inverter per site,
so by default nodes are the physical variables {P, G, T, Gcs} with a physics prior
(G->P, T->P, Gcs->P, Gcs->G, G->T). When per-inverter data arrive, override `build_nodes` to return
[b, N_inv, L, F] and set readout="sum" to forecast each inverter and the plant total.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..base import ForecastModel, register
from ..layers.common import HorizonHead
from ..layers.graph import AdjacencyLearner, DenseGAT, DenseGCN
from ..layers.mamba import MambaLayer

VAR_NODES = ["P", "G", "T", "Gcs"]
# A[i, j] = 1 if node j informs node i (receiver i)
VAR_PHYSICS = {("P", "G"), ("P", "T"), ("P", "Gcs"), ("G", "Gcs"), ("T", "G")}


def variable_physics_adjacency(nodes):
    A = torch.eye(len(nodes))
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if (a, b) in VAR_PHYSICS:
                A[i, j] = 1.0
    return A


class _GRUTemporal(nn.Module):
    def __init__(self, d_in, d, n_layers, dropout):
        super().__init__()
        self.rnn = nn.GRU(d_in, d, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)

    def forward(self, x):
        return self.rnn(x)[0][:, -1]


class _MambaTemporal(nn.Module):
    def __init__(self, d_in, d, n_layers, dropout, d_state=16, backend="auto"):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.layers = nn.Sequential(*[MambaLayer(d, d_state, dropout=dropout, backend=backend) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(self.layers(self.inp(x)))[:, -1]


@register("graph_mamba")
class GraphMamba(ForecastModel):
    def __init__(self, spec, d_model=64, temporal="mamba", t_layers=2, conv="gat", g_layers=2, heads=4,
                 adj="mixed", A_physics=None, nodes=None, readout="target", dropout=0.1, use_fut=True,
                 backend="auto", **kw):
        super().__init__(spec, **kw)
        self.nodes = list(nodes or VAR_NODES)
        self.node_idx = [spec.hist_names.index(n) for n in self.nodes]
        self.cal_idx = [i for i, n in enumerate(spec.hist_names) if n.startswith(("hod", "doy"))]
        N, d_in = len(self.nodes), 1 + len(self.cal_idx)
        self.node_emb = nn.Parameter(torch.randn(1, N, d_model) * 0.02)
        self.temporal = (_MambaTemporal(d_in, d_model, t_layers, dropout, backend=backend) if temporal == "mamba"
                         else _GRUTemporal(d_in, d_model, t_layers, dropout))
        A0 = A_physics if A_physics is not None else variable_physics_adjacency(self.nodes)
        self.adj = AdjacencyLearner(N, adj, A0, d_ctx=d_model)
        Conv = {"gcn": lambda: DenseGCN(d_model, d_model, dropout),
                "gat": lambda: DenseGAT(d_model, d_model, heads, dropout)}[conv]
        self.convs = nn.ModuleList(Conv() for _ in range(g_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(g_layers))
        self.readout = readout
        self.t_node = self.nodes.index(spec.target) if spec.target in self.nodes else 0
        d_z = d_model * (N if readout == "concat" else 1)
        self.head = HorizonHead(d_z, spec.horizon, spec.n_fut, use_fut, n_quantiles=self.n_q, dropout=dropout)

    def build_nodes(self, batch) -> torch.Tensor:
        """-> [b, N, L, F]. Variable nodes: each node gets its own series + shared calendar features."""
        xh = batch["x_hist"]
        cal = xh[..., self.cal_idx]
        return torch.stack([torch.cat([xh[..., i:i + 1], cal], -1) for i in self.node_idx], 1)

    def forward(self, batch):
        x = self.build_nodes(batch)
        b, N, L, F = x.shape
        h = self.temporal(x.reshape(b * N, L, F)).view(b, N, -1) + self.node_emb
        A = self.adj(ctx=h.mean(1), batch=b)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + conv(h, A))
        z = {"target": lambda: h[:, self.t_node], "mean": lambda: h.mean(1), "concat": lambda: h.flatten(1)}[self.readout]()
        out = {"pred": self.head(z, batch["x_fut"]), "adj": A.detach()}
        if hasattr(self.adj, "last_alpha"):
            out["alpha"] = self.adj.last_alpha
        return out
