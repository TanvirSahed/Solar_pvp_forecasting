"""Worked fusion example: iTransformer's variable encoder + HDCF-Mamba's temporal encoder, joined by a learned gate.

Recipe - repeat it to fuse any two models:
  1. Take each model's ENCODER (what its encode() calls): here ITransformerEncoder (transformers.py, relations
     BETWEEN variables) and HDCFEncoder (hdcf_mamba.py, global + local patterns OVER TIME).
  2. Merge their input adapters: list every adapter step either model needs, so model_input holds all tensors.
  3. Bring both representations to one shape [batch, fusion_size] (HDCF returns a sequence -> pool it over time).
  4. Fuse them (here: a per-feature gate) and reuse a ForecastHead.
The gate values (output "fusion_gate") show how much each encoder is trusted; 1 = variable encoder only.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ...model_inputs.adapter_steps import SplitCalendarFeatures
from ...model_inputs.input_adapter import InputAdapter
from ..base import ForecastModel, register_model
from ..blocks.common_blocks import ForecastHead
from .hdcf_mamba import HDCFEncoder
from .transformers import ITransformerEncoder


@register_model("fusion_itransformer_hdcf")
class ITransformerHDCFFusion(ForecastModel):
    def __init__(self, data_spec, fusion_size=64, n_attention_heads=4, n_transformer_layers=2, state_size=16,
                 dropout=0.1, use_future_features=True, scan_backend="auto", quantile_levels=None, input_adapter=None):
        # step 2: iTransformer needs the raw ForecastBatch (always kept); HDCF needs the calendar split
        calendar_split = SplitCalendarFeatures(data_spec)
        super().__init__(data_spec, input_adapter or InputAdapter([calendar_split], "ITransformerHDCFFusion"),
                         quantile_levels)
        # step 1: the two encoders, unchanged
        self.variable_encoder = ITransformerEncoder(data_spec, model_size=fusion_size, n_heads=n_attention_heads,
                                                    n_layers=n_transformer_layers, feedforward_size=2 * fusion_size,
                                                    dropout=dropout)
        self.temporal_encoder = HDCFEncoder(len(calendar_split.measurement_indices),
                                            len(calendar_split.calendar_indices), data_spec.lookback_steps,
                                            model_size=fusion_size, state_size=state_size, dropout=dropout,
                                            scan_backend=scan_backend)
        # step 3: learned weighted average over HDCF's 2 * lookback time positions -> one vector
        self.temporal_pooling = nn.Linear(2 * data_spec.lookback_steps, 1)
        # step 4: gate in [0, 1] per feature, decided from both representations
        self.fusion_gate = nn.Sequential(nn.Linear(2 * fusion_size, fusion_size), nn.Sigmoid())
        self.forecast_head = ForecastHead(fusion_size, data_spec.horizon_steps, data_spec.n_future_features,
                                          use_future_features, n_outputs_per_step=self.n_outputs_per_step,
                                          dropout=dropout)

    def encode(self, model_input):
        variable_representation = self.variable_encoder(model_input["history_features"],
                                                        model_input["future_features"])          # [batch, size]
        temporal_sequence = self.temporal_encoder(model_input["measurement_history"],
                                                  model_input["calendar_history"])                # [batch, 2L, size]
        temporal_representation = self.temporal_pooling(temporal_sequence.transpose(1, 2)).squeeze(-1)
        gate = self.fusion_gate(torch.cat([variable_representation, temporal_representation], -1))
        self.last_fusion_gate = gate.detach()
        return gate * variable_representation + (1 - gate) * temporal_representation

    def forecast(self, model_input):
        representation = self.encode(model_input)
        return {"forecast": self.shape_output(self.forecast_head(representation, model_input["future_features"])),
                "fusion_gate": self.last_fusion_gate}
