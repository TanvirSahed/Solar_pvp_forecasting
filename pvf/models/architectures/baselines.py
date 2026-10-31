"""Reference models every new architecture must beat (selection report row 1).

smart_persistence  clear-sky-scaled persistence, no parameters
lstm, gru          recurrent encoder (optionally bidirectional, attention pooling) -> forecast head
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

from ..base import ForecastModel, register_model
from ..blocks.common_blocks import ForecastHead


@register_model("smart_persistence")
class SmartPersistence(ForecastModel):
    is_trainable = False

    def forecast(self, model_input):
        return {"forecast": model_input["smart_persistence"]}


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
        # shortcut: bypass Windows cuDNN RNN shutdown crashes; retry after upgrading the pinned runtime.
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled and sys.platform != "win32"):
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
