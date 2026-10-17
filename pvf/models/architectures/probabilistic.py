"""Probabilistic and domain-adaptive models (selection report rows 13-14).

mtsa_qrn  Wang et al. 2026 (Sustainability 18(2):739 -> ref/R08_MTSA-QRN_CQR.pdf), Sec. 3.1-3.4:
          temporal path        multi-scale dilated causal convolutions -> softmax scale weights (Eq. 9-10) -> self-attention
          meteorological path  graph attention over variable series
          weather-adaptive     fusion weights from the clearness-index regime (SummariseWeatherRegime adapter step)
          output               non-crossing quantiles; calibrate with evaluation.conformal.conformalized_quantile_regression
dt_dann   Zhao 2024 (Sci. Rep. -> ref/R18_DigitalTwin_Transformer_PV.pdf): causal-CNN feature generator (replaces the
          positional embedding) -> transformer -> forecast; a domain classifier behind gradient reversal makes the
          features domain-invariant. Source = labelled training data, target = unlabelled batches from
          DataConfig.domain_adaptation_months (the trainer passes them as model_input["target_domain_input"]).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...model_inputs.adapter_steps import SelectVariableSeries, SummariseWeatherRegime
from ...model_inputs.input_adapter import InputAdapter
from ..base import ForecastModel, register_model
from ..blocks.common_blocks import ForecastHead, make_mlp, make_transformer_encoder, reverse_gradient
from ..blocks.graph_blocks import GraphAttention

DEFAULT_QUANTILE_LEVELS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


class NonCrossingQuantiles(nn.Module):
    """Raw head outputs [..., n_quantiles] -> lowest quantile + cumulative positive increments (always ordered)."""

    def forward(self, raw):
        return torch.cat([raw[..., :1], raw[..., :1] + torch.cumsum(F.softplus(raw[..., 1:]), -1)], -1)


@register_model("mtsa_qrn")
class MTSAQRN(ForecastModel):
    def __init__(self, data_spec, model_size=64, dilations=(1, 2, 4, 8), kernel_size=3, n_heads=4, n_graph_heads=4,
                 dropout=0.1, quantile_levels=DEFAULT_QUANTILE_LEVELS, use_future_features=True, input_adapter=None):
        adapter = InputAdapter([SelectVariableSeries(data_spec),
                                SummariseWeatherRegime(data_spec)], "MTSAQRN")
        super().__init__(data_spec, input_adapter or adapter, quantile_levels)
        # temporal path
        self.input_projection = nn.Linear(data_spec.n_history_features, model_size)
        self.kernel_size, self.dilations = kernel_size, tuple(dilations)
        self.scale_convolutions = nn.ModuleList(nn.Conv1d(model_size, model_size, kernel_size, dilation=d)
                                                for d in self.dilations)
        self.scale_scorer = nn.Linear(model_size, 1)
        self.temporal_refinement = nn.TransformerEncoderLayer(model_size, n_heads, 2 * model_size, dropout,
                                                              batch_first=True, norm_first=True)
        # meteorological path
        self.variable_embedding = nn.Linear(data_spec.lookback_steps, model_size)
        self.variable_attention = GraphAttention(model_size, model_size, n_graph_heads, dropout)
        # weather-adaptive fusion
        self.regime_to_path_weights = make_mlp(3, 16, 2, n_layers=2, dropout=0.0)
        # output
        self.forecast_head = ForecastHead(model_size, data_spec.horizon_steps, data_spec.n_future_features,
                                          use_future_features, n_outputs_per_step=self.n_outputs_per_step,
                                          dropout=dropout)
        self.non_crossing = NonCrossingQuantiles()

    def encode(self, model_input):
        projected = self.input_projection(model_input["history_features"]).transpose(1, 2)    # [batch, size, time]
        scales = torch.stack([F.gelu(convolution(F.pad(projected, ((self.kernel_size - 1) * d, 0)))).transpose(1, 2)
                              for convolution, d in zip(self.scale_convolutions, self.dilations)], 1)
        scale_weights = torch.softmax(self.scale_scorer(scales.mean(2)), 1)                  # Eq. 9
        temporal = self.temporal_refinement((scale_weights.unsqueeze(-1) * scales).sum(1))[:, -1]   # Eq. 10
        variables = self.variable_embedding(model_input["variable_series"])                 # [batch, nodes, size]
        all_connected = torch.ones(variables.shape[0], variables.shape[1], variables.shape[1], device=variables.device)
        meteorological = (variables + self.variable_attention(variables, all_connected))[:, 0]   # target node
        path_weights = torch.softmax(self.regime_to_path_weights(model_input["weather_regime"]), -1)
        self.last_scale_weights, self.last_path_weights = scale_weights.detach(), path_weights.detach()
        return path_weights[:, :1] * temporal + path_weights[:, 1:] * meteorological

    def forecast(self, model_input):
        raw = self.forecast_head(self.encode(model_input), model_input["future_features"])
        forecast = self.non_crossing(raw) if self.n_quantiles else raw.squeeze(-1)
        return {"forecast": forecast, "scale_weights": self.last_scale_weights, "path_weights": self.last_path_weights}


class CausalConvolutionFeatureGenerator(nn.Module):
    def __init__(self, n_input_features, model_size, n_layers=2, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.convolutions = nn.ModuleList(nn.Conv1d(n_input_features if i == 0 else model_size, model_size, kernel_size)
                                          for i in range(n_layers))

    def forward(self, sequence):                                            # [batch, time, features] -> [b, t, size]
        hidden = sequence.transpose(1, 2)
        for convolution in self.convolutions:
            hidden = F.gelu(convolution(F.pad(hidden, (self.kernel_size - 1, 0))))
        return hidden.transpose(1, 2)


@register_model("dt_dann")
class DigitalTwinTransformerDANN(ForecastModel):
    def __init__(self, data_spec, model_size=64, n_heads=4, n_layers=2, feedforward_size=128, dropout=0.1,
                 domain_loss_weight=0.1, use_future_features=True, quantile_levels=None, input_adapter=None):
        super().__init__(data_spec, input_adapter, quantile_levels)
        self.feature_generator = CausalConvolutionFeatureGenerator(data_spec.n_history_features, model_size)
        self.transformer = make_transformer_encoder(model_size, n_heads, n_layers, feedforward_size, dropout)
        self.forecast_head = ForecastHead(model_size, data_spec.horizon_steps, data_spec.n_future_features,
                                          use_future_features, n_outputs_per_step=self.n_outputs_per_step,
                                          dropout=dropout)
        self.domain_classifier = make_mlp(model_size, model_size, 1, n_layers=2, dropout=dropout)
        self.domain_loss_weight, self.gradient_reversal_strength = domain_loss_weight, 0.0

    def on_epoch_start(self, epoch: int, n_epochs: int):
        progress = epoch / max(1, n_epochs)
        self.gradient_reversal_strength = 2.0 / (1.0 + math.exp(-10 * progress)) - 1.0   # Ganin et al. schedule

    def encode(self, model_input):
        return self.transformer(self.feature_generator(model_input["history_features"]))[:, -1]

    def forecast(self, model_input):
        source_features = self.encode(model_input)
        output = {"forecast": self.shape_output(self.forecast_head(source_features, model_input["future_features"]))}
        target_domain = model_input.get("target_domain_input")
        if self.training and target_domain is not None and self.domain_loss_weight > 0:
            target_features = self.encode(target_domain)
            features = reverse_gradient(torch.cat([source_features, target_features]), self.gradient_reversal_strength)
            domain_logits = self.domain_classifier(features).squeeze(-1)
            domain_labels = torch.cat([torch.zeros(len(source_features)), torch.ones(len(target_features))])
            output["auxiliary_loss"] = self.domain_loss_weight * F.binary_cross_entropy_with_logits(
                domain_logits, domain_labels.to(domain_logits))
        return output
