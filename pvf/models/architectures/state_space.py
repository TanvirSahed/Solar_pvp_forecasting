"""State-space (Mamba) family (selection report rows 5-6 and the "Concat-Mamba" ablation root).

mamba_temporal  Mamba stack over time steps; all features of a step concatenated ("Concat-Mamba").
s_mamba         S-Mamba (F06): variable tokens (iTransformer-style) mixed by a bidirectional Mamba.
power_mamba     PowerMamba (R19, Sec. IV): RevIN -> trend/seasonal decomposition -> linear time projection
                -> Mamba over time tokens || inverse Mamba over variable tokens -> concat[x_e, x_m, x_im, x_m + x_im]
                -> linear to horizon. External forecasts (Sec. IV-E): the AppendFutureToHistory adapter step
                appends the known future to the sequence.
hdcf_mamba      see hdcf_mamba.py
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ...model_inputs.adapter_steps import AppendFutureToHistory
from ...model_inputs.input_adapter import InputAdapter
from ..base import ForecastModel, register_model
from ..blocks.common_blocks import ForecastHead, ReversibleInstanceNorm, SeriesDecomposition
from ..blocks.mamba_blocks import MambaLayer


class MambaTemporalEncoder(nn.Module):
    """[batch, time, n_features] -> [batch, model_size] (state after the last step). Reusable in hybrids."""

    def __init__(self, n_input_features, model_size=64, n_layers=3, state_size=16, expansion_factor=2, dropout=0.1,
                 scan_backend="auto"):
        super().__init__()
        self.input_projection = nn.Linear(n_input_features, model_size)
        self.mamba_layers = nn.Sequential(*[MambaLayer(model_size, state_size, expansion_factor=expansion_factor,
                                                       dropout=dropout, scan_backend=scan_backend)
                                            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(model_size)
        self.output_size = model_size

    def forward(self, sequence):
        return self.norm(self.mamba_layers(self.input_projection(sequence)))[:, -1]


@register_model("mamba_temporal")
class MambaTemporal(ForecastModel):
    def __init__(self, data_spec, model_size=64, n_layers=3, state_size=16, expansion_factor=2, dropout=0.1,
                 use_future_features=True, scan_backend="auto", quantile_levels=None, input_adapter=None):
        super().__init__(data_spec, input_adapter, quantile_levels)
        self.temporal_encoder = MambaTemporalEncoder(data_spec.n_history_features, model_size, n_layers, state_size,
                                                     expansion_factor, dropout, scan_backend)
        self.forecast_head = ForecastHead(model_size, data_spec.horizon_steps, data_spec.n_future_features,
                                          use_future_features, n_outputs_per_step=self.n_outputs_per_step,
                                          dropout=dropout)

    def encode(self, model_input):
        return self.temporal_encoder(model_input["history_features"])

    def forecast(self, model_input):
        representation = self.encode(model_input)
        return {"forecast": self.shape_output(self.forecast_head(representation, model_input["future_features"]))}


@register_model("s_mamba")
class SMamba(ForecastModel):
    def __init__(self, data_spec, model_size=128, n_layers=2, state_size=16, dropout=0.1, scan_backend="auto",
                 quantile_levels=None, input_adapter=None):
        super().__init__(data_spec, input_adapter, quantile_levels)
        self.history_token_embedding = nn.Linear(data_spec.lookback_steps, model_size)
        self.future_token_embedding = nn.Linear(data_spec.horizon_steps, model_size)
        self.variable_mixer = nn.Sequential(*[MambaLayer(model_size, state_size, dropout=dropout, bidirectional=True,
                                                         use_feedforward=True, scan_backend=scan_backend)
                                              for _ in range(n_layers)])
        self.norm = nn.LayerNorm(model_size)
        self.forecast_head = nn.Linear(model_size, data_spec.horizon_steps * self.n_outputs_per_step)

    def encode(self, model_input):
        tokens = torch.cat([self.history_token_embedding(model_input["history_features"].transpose(1, 2)),
                            self.future_token_embedding(model_input["future_features"].transpose(1, 2))], 1)
        return self.norm(self.variable_mixer(tokens))[:, self.data_spec.target_feature_index]

    def forecast(self, model_input):
        return {"forecast": self.shape_output(self.forecast_head(self.encode(model_input)))}


@register_model("power_mamba")
class PowerMamba(ForecastModel):
    def __init__(self, data_spec, embedding_length=64, state_size=16, moving_average_kernel=25, dropout=0.1,
                 use_reversible_norm=True, scan_backend="auto", quantile_levels=None, input_adapter=None):
        super().__init__(data_spec, input_adapter or InputAdapter([AppendFutureToHistory()], "PowerMamba"),
                         quantile_levels)
        n_channels = data_spec.n_history_features + data_spec.n_future_features
        sequence_length = data_spec.lookback_steps + data_spec.horizon_steps
        self.reversible_norm = ReversibleInstanceNorm(n_channels) if use_reversible_norm else None
        self.decomposition = SeriesDecomposition(moving_average_kernel)
        self.time_projection = nn.Linear(2 * sequence_length, embedding_length)   # [trend; seasonal] -> E, per channel
        self.time_dropout, self.variable_dropout = nn.Dropout(dropout), nn.Dropout(dropout)
        self.time_token_mamba = MambaLayer(n_channels, state_size, dropout=0.0, scan_backend=scan_backend)
        self.variable_token_mamba = MambaLayer(embedding_length, state_size, dropout=0.0, scan_backend=scan_backend)
        self.forecast_head = nn.Linear(4 * embedding_length, data_spec.horizon_steps * self.n_outputs_per_step)

    def encode(self, model_input):                                     # -> [batch, 4 * embedding_length, channels]
        sequence = model_input["extended_sequence"]
        if self.reversible_norm is not None:
            sequence = self.reversible_norm.normalize(sequence)
        seasonal, trend = self.decomposition(sequence)
        embedded = self.time_projection(torch.cat([trend, seasonal], 1).transpose(1, 2)).transpose(1, 2)  # [b, E, C]
        over_time = self.time_token_mamba(self.time_dropout(embedded))
        over_variables = self.variable_token_mamba(self.variable_dropout(embedded.transpose(1, 2))).transpose(1, 2)
        return torch.cat([embedded, over_time, over_variables, over_time + over_variables], 1)

    def forecast(self, model_input):
        target_channel = self.data_spec.target_feature_index
        forecast = self.shape_output(self.forecast_head(self.encode(model_input)[..., target_channel]))
        if self.reversible_norm is not None and not self.n_quantiles:
            forecast = self.reversible_norm.denormalize(forecast, target_channel)
        return {"forecast": forecast}
