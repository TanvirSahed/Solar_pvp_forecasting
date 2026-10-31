"""Graph + Mamba hybrid: "DT-informed GAT/GCN + Mamba" (selection report row 12, the SET Lab research core).

    node_series [batch, nodes, lookback, F]  (built by the BuildVariableNodes adapter step)
      -> shared Mamba (or GRU) temporal encoder per node -> node states [batch, nodes, size]
      -> graph layers (GCN or GAT) with adjacency = trust * physics + (1 - trust) * learned
      -> read out the target node -> forecast head

The report's mandatory ablation chain is just hyperparameters:
    Concat-Mamba            model "mamba_temporal"
    GCN + Mamba             graph_layer="gcn", adjacency_mode="learned"
    GAT + Mamba             graph_layer="gat", adjacency_mode="learned"
    DT-informed GAT + Mamba graph_layer="gat", adjacency_mode="mixed"   (physics_adjacency from the twin)

NODES: the report assumes 4 inverter nodes, but the data have one inverter per site, so nodes default to the
physical variables (node_feature_names=None -> every measured history feature). For inverters, pass input_adapter with a BuildInverterNodes step and readout="sum".
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

from ...model_inputs.adapter_steps import BuildVariableNodes
from ...model_inputs.input_adapter import InputAdapter
from ..base import ForecastModel, register_model
from ..blocks.common_blocks import ForecastHead
from ..blocks.graph_blocks import AdjacencyBuilder, GraphAttention, GraphConvolution
from .state_space import MambaTemporalEncoder

# (receiver, sender): which variable physically informs which
VARIABLE_PHYSICS_EDGES = {
    ("ac_power", "irradiance"), ("ac_power", "module_temperature"), ("ac_power", "clear_sky_irradiance"),
    ("irradiance", "clear_sky_irradiance"), ("module_temperature", "irradiance"),
}


def variable_physics_adjacency(node_names) -> torch.Tensor:
    adjacency = torch.eye(len(node_names))
    for receiver_index, receiver in enumerate(node_names):
        for sender_index, sender in enumerate(node_names):
            if (receiver, sender) in VARIABLE_PHYSICS_EDGES:
                adjacency[receiver_index, sender_index] = 1.0
    return adjacency


class GRUTemporalEncoder(nn.Module):
    def __init__(self, n_input_features, model_size, n_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(n_input_features, model_size, n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)

    def forward(self, sequence):
        # shortcut: bypass Windows cuDNN RNN shutdown crashes; retry after upgrading the pinned runtime.
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled and sys.platform != "win32"):
            return self.gru(sequence)[0][:, -1]


@register_model("graph_mamba")
class GraphMamba(ForecastModel):
    def __init__(self, data_spec, node_feature_names=None, model_size=64,
                 temporal_encoder="mamba", n_temporal_layers=2, graph_layer="gat", n_graph_layers=2, n_heads=4,
                 adjacency_mode="mixed", physics_adjacency=None, readout="target", dropout=0.1,
                 use_future_features=True, scan_backend="auto", quantile_levels=None, input_adapter=None):
        node_builder = BuildVariableNodes(data_spec, node_feature_names)
        super().__init__(data_spec, input_adapter or InputAdapter([node_builder], "GraphMamba"), quantile_levels)
        self.node_names = node_builder.node_names
        n_nodes = node_builder.n_nodes
        if temporal_encoder == "mamba":
            self.temporal_encoder = MambaTemporalEncoder(node_builder.node_input_size, model_size, n_temporal_layers,
                                                         dropout=dropout, scan_backend=scan_backend)
        else:
            self.temporal_encoder = GRUTemporalEncoder(node_builder.node_input_size, model_size, n_temporal_layers,
                                                       dropout)
        self.node_identity = nn.Parameter(torch.randn(1, n_nodes, model_size) * 0.02)
        physics = physics_adjacency if physics_adjacency is not None else variable_physics_adjacency(self.node_names)
        self.adjacency_builder = AdjacencyBuilder(n_nodes, adjacency_mode, physics, context_size=model_size)
        make_graph_layer = {"gcn": lambda: GraphConvolution(model_size, model_size, dropout),
                            "gat": lambda: GraphAttention(model_size, model_size, n_heads, dropout)}[graph_layer]
        self.graph_layers = nn.ModuleList(make_graph_layer() for _ in range(n_graph_layers))
        self.graph_norms = nn.ModuleList(nn.LayerNorm(model_size) for _ in range(n_graph_layers))
        self.readout = readout
        self.target_node_index = (self.node_names.index(data_spec.target_feature_name)
                                  if data_spec.target_feature_name in self.node_names else 0)
        representation_size = model_size * (n_nodes if readout == "concat" else 1)
        self.forecast_head = ForecastHead(representation_size, data_spec.horizon_steps, data_spec.n_future_features,
                                          use_future_features, n_outputs_per_step=self.n_outputs_per_step,
                                          dropout=dropout)

    def encode(self, model_input):
        node_series = model_input["node_series"]                                  # [batch, nodes, time, F]
        batch, n_nodes, time, n_features = node_series.shape
        node_states = self.temporal_encoder(node_series.reshape(batch * n_nodes, time, n_features))
        node_states = node_states.view(batch, n_nodes, -1) + self.node_identity
        adjacency = self.adjacency_builder(context=node_states.mean(1), batch_size=batch)
        for graph_layer, norm in zip(self.graph_layers, self.graph_norms):
            node_states = norm(node_states + graph_layer(node_states, adjacency))
        self.last_adjacency = adjacency.detach()
        if self.readout == "target":
            return node_states[:, self.target_node_index]
        if self.readout == "mean":
            return node_states.mean(1)
        return node_states.flatten(1)                                             # "concat"

    def forecast(self, model_input):
        representation = self.encode(model_input)
        output = {"forecast": self.shape_output(self.forecast_head(representation, model_input["future_features"])),
                  "adjacency": self.last_adjacency}
        if hasattr(self.adjacency_builder, "last_physics_trust"):
            output["physics_trust"] = self.adjacency_builder.last_physics_trust
        return output
