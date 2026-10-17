"""Reference models every new architecture must beat (selection report row 1).

smart_persistence  clear-sky-scaled persistence, no parameters
deep_mlp           flattened history (+ future features) -> 256-128-64 -> forecast
lstm, gru          recurrent encoder, last hidden state -> forecast head
tcn                causal dilated convolutions (Bai et al. 2018): 4 blocks, dilations 1/2/4/8 -> receptive field 61
dlinear            linear maps on trend + seasonal parts of the target history (Zeng et al. 2023)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...model_inputs.adapter_steps import SelectTargetHistory
from ...model_inputs.input_adapter import InputAdapter
from ..base import ForecastModel, register_model
from ..blocks.common_blocks import ForecastHead, SeriesDecomposition


@register_model("smart_persistence")
class SmartPersistence(ForecastModel):
    is_trainable = False

    def forecast(self, model_input):
        return {"forecast": model_input["smart_persistence"]}


@register_model("deep_mlp")
class DeepMLP(ForecastModel):
    def __init__(self, data_spec, hidden_sizes=(256, 128, 64), dropout=0.2, use_future_features=True,
                 quantile_levels=None, input_adapter=None):
        super().__init__(data_spec, input_adapter, quantile_levels)
        self.use_future_features = use_future_features
        input_size = data_spec.lookback_steps * data_spec.n_history_features
        if use_future_features:
            input_size += data_spec.horizon_steps * data_spec.n_future_features
        layers = []
        for hidden_size in hidden_sizes:
            layers += [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            input_size = hidden_size
        self.encoder = nn.Sequential(*layers)
        self.forecast_head = nn.Linear(input_size, data_spec.horizon_steps * self.n_outputs_per_step)

    def encode(self, model_input):
        flat_inputs = [model_input["history_features"].flatten(1)]
        if self.use_future_features:
            flat_inputs.append(model_input["future_features"].flatten(1))
        return self.encoder(torch.cat(flat_inputs, 1))                       # [batch, hidden_sizes[-1]]

    def forecast(self, model_input):
        return {"forecast": self.shape_output(self.forecast_head(self.encode(model_input)))}


class _RecurrentForecaster(ForecastModel):
    recurrent_layer_class = nn.LSTM

    def __init__(self, data_spec, hidden_size=64, n_layers=2, dropout=0.1, bidirectional=False, pooling="last",
                 use_future_features=True, quantile_levels=None, input_adapter=None):
        """bidirectional: also read the lookback window backwards (never the horizon) - R20.
        pooling: "last" hidden state, or "attention" = learned softmax-weighted average over time (R20)."""
        super().__init__(data_spec, input_adapter, quantile_levels)
        self.recurrent_encoder = self.recurrent_layer_class(data_spec.n_history_features, hidden_size, n_layers,
                                                            batch_first=True, dropout=dropout,
                                                            bidirectional=bidirectional)
        state_size = hidden_size * (2 if bidirectional else 1)
        if pooling not in ("last", "attention"):
            raise ValueError("pooling must be 'last' or 'attention'")
        self.pooling = pooling
        self.attention_scorer = nn.Linear(state_size, 1) if pooling == "attention" else None
        self.forecast_head = ForecastHead(state_size, data_spec.horizon_steps, data_spec.n_future_features,
                                          use_future_features, n_outputs_per_step=self.n_outputs_per_step,
                                          dropout=dropout)

    def encode(self, model_input):
        hidden_states, _ = self.recurrent_encoder(model_input["history_features"])   # [batch, time, state_size]
        if self.pooling == "last":
            return hidden_states[:, -1]
        weights = torch.softmax(self.attention_scorer(hidden_states), dim=1)            # [batch, time, 1]
        return (weights * hidden_states).sum(1)

    def forecast(self, model_input):
        representation = self.encode(model_input)
        return {"forecast": self.shape_output(self.forecast_head(representation, model_input["future_features"]))}


@register_model("lstm")
class LSTMForecaster(_RecurrentForecaster):
    recurrent_layer_class = nn.LSTM


@register_model("gru")
class GRUForecaster(_RecurrentForecaster):
    recurrent_layer_class = nn.GRU


class CausalConvolutionBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        weight_norm = nn.utils.parametrizations.weight_norm
        self.first_convolution = weight_norm(nn.Conv1d(input_channels, output_channels, kernel_size, dilation=dilation))
        self.second_convolution = weight_norm(nn.Conv1d(output_channels, output_channels, kernel_size, dilation=dilation))
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels else nn.Identity()

    def forward(self, x):                                                   # [batch, channels, time]
        y = self.dropout(F.relu(self.first_convolution(F.pad(x, (self.left_padding, 0)))))
        y = self.dropout(F.relu(self.second_convolution(F.pad(y, (self.left_padding, 0)))))
        return F.relu(y + self.residual(x))


@register_model("tcn")
class TemporalConvolutionNetwork(ForecastModel):
    def __init__(self, data_spec, channels=32, dilations=(1, 2, 4, 8), kernel_size=3, dropout=0.1,
                 use_future_features=True, quantile_levels=None, input_adapter=None):
        super().__init__(data_spec, input_adapter, quantile_levels)
        blocks, input_channels = [], data_spec.n_history_features
        for dilation in dilations:
            blocks.append(CausalConvolutionBlock(input_channels, channels, kernel_size, dilation, dropout))
            input_channels = channels
        self.convolution_encoder = nn.Sequential(*blocks)
        self.forecast_head = ForecastHead(channels, data_spec.horizon_steps, data_spec.n_future_features,
                                          use_future_features, n_outputs_per_step=self.n_outputs_per_step,
                                          dropout=dropout)

    def encode(self, model_input):
        features_over_time = self.convolution_encoder(model_input["history_features"].transpose(1, 2))
        return features_over_time[..., -1]                                  # [batch, channels] at the last step

    def forecast(self, model_input):
        representation = self.encode(model_input)
        return {"forecast": self.shape_output(self.forecast_head(representation, model_input["future_features"]))}


@register_model("dlinear")
class DLinear(ForecastModel):
    def __init__(self, data_spec, moving_average_kernel=25, use_future_features=True, input_adapter=None):
        super().__init__(data_spec, input_adapter or InputAdapter([SelectTargetHistory(data_spec)], "DLinear"))
        self.decomposition = SeriesDecomposition(moving_average_kernel)
        self.seasonal_map = nn.Linear(data_spec.lookback_steps, data_spec.horizon_steps)
        self.trend_map = nn.Linear(data_spec.lookback_steps, data_spec.horizon_steps)
        self.future_feature_map = nn.Linear(data_spec.n_future_features, 1) if use_future_features else None

    def forecast(self, model_input):
        seasonal, trend = self.decomposition(model_input["target_history"].unsqueeze(-1))
        forecast = self.seasonal_map(seasonal[..., 0]) + self.trend_map(trend[..., 0])
        if self.future_feature_map is not None:
            forecast = forecast + self.future_feature_map(model_input["future_features"]).squeeze(-1)
        return {"forecast": forecast}
