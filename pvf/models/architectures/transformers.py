"""Transformer family (selection report rows 2-4).

patch_tst      Nie et al. 2023 (ref/foundational/F01): every channel cut into patches independently; the head reads
               the target channel ("target") or all channels ("all_channels", a fix for the channel-independence
               limitation noted in the report).
i_transformer  Liu et al. 2024 (F02): each variable's whole series is ONE token; attention runs across variables.
               Known-future series (clear sky, calendar) become extra variable tokens.
time_xer       Wang et al. 2024 (F03): target history -> patch tokens + one global token; other variables -> variable
               tokens; the global token cross-attends to the variable tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..base import ForecastModel, register_model
from ..blocks.common_blocks import ForecastHead, PatchEmbedding, ReversibleInstanceNorm, make_transformer_encoder


@register_model("patch_tst")
class PatchTST(ForecastModel):
    def __init__(self, data_spec, patch_length=16, patch_stride=8, model_size=128, n_heads=8, n_layers=3,
                 feedforward_size=256, dropout=0.2, head_reads="target", use_reversible_norm=False,
                 use_future_features=True, quantile_levels=None, input_adapter=None):
        super().__init__(data_spec, input_adapter, quantile_levels)
        self.head_reads = head_reads
        self.reversible_norm = ReversibleInstanceNorm(data_spec.n_history_features) if use_reversible_norm else None
        self.patch_embedding = PatchEmbedding(patch_length, patch_stride, model_size, data_spec.lookback_steps, dropout)
        self.patch_encoder = make_transformer_encoder(model_size, n_heads, n_layers, feedforward_size, dropout)
        channel_representation_size = self.patch_embedding.n_patches * model_size
        if head_reads == "target":
            representation_size = channel_representation_size
        elif head_reads == "all_channels":
            self.channel_compression = nn.Linear(channel_representation_size, 32)
            representation_size = 32 * data_spec.n_history_features
        else:
            raise ValueError("head_reads must be 'target' or 'all_channels'")
        self.forecast_head = ForecastHead(representation_size, data_spec.horizon_steps, data_spec.n_future_features,
                                          use_future_features, n_outputs_per_step=self.n_outputs_per_step,
                                          dropout=dropout)

    def encode(self, model_input):
        history = model_input["history_features"]
        if self.reversible_norm is not None:
            history = self.reversible_norm.normalize(history)
        patch_states = self.patch_encoder(self.patch_embedding(history))          # [batch*channels, patches, size]
        per_channel = patch_states.flatten(1).view(history.shape[0], history.shape[2], -1)
        if self.head_reads == "target":
            return per_channel[:, self.data_spec.target_feature_index]
        return self.channel_compression(per_channel).flatten(1)

    def forecast(self, model_input):
        forecast = self.shape_output(self.forecast_head(self.encode(model_input), model_input["future_features"]))
        if self.reversible_norm is not None and not self.n_quantiles:
            forecast = self.reversible_norm.denormalize(forecast, self.data_spec.target_feature_index)
        return {"forecast": forecast}


class ITransformerEncoder(nn.Module):
    """Variable tokens -> transformer across variables -> the target variable's token. Reusable in fusions."""

    def __init__(self, data_spec, model_size=128, n_heads=8, n_layers=3, feedforward_size=256, dropout=0.1,
                 use_future_tokens=True):
        super().__init__()
        self.target_index = data_spec.target_feature_index
        self.history_token_embedding = nn.Sequential(nn.Linear(data_spec.lookback_steps, model_size), nn.Dropout(dropout))
        self.future_token_embedding = (nn.Sequential(nn.Linear(data_spec.horizon_steps, model_size), nn.Dropout(dropout))
                                       if use_future_tokens else None)
        self.variable_identity = nn.Parameter(
            torch.randn(1, data_spec.n_history_features + data_spec.n_future_features, model_size) * 0.02)
        self.variable_mixer = make_transformer_encoder(model_size, n_heads, n_layers, feedforward_size, dropout)
        self.norm = nn.LayerNorm(model_size)
        self.output_size = model_size

    def forward(self, history_features, future_features):            # -> [batch, model_size]
        tokens = self.history_token_embedding(history_features.transpose(1, 2))      # [batch, n_history, size]
        if self.future_token_embedding is not None:
            tokens = torch.cat([tokens, self.future_token_embedding(future_features.transpose(1, 2))], 1)
        tokens = tokens + self.variable_identity[:, :tokens.shape[1]]
        return self.norm(self.variable_mixer(tokens))[:, self.target_index]


@register_model("i_transformer")
class ITransformer(ForecastModel):
    def __init__(self, data_spec, model_size=128, n_heads=8, n_layers=3, feedforward_size=256, dropout=0.1,
                 use_future_tokens=True, quantile_levels=None, input_adapter=None):
        super().__init__(data_spec, input_adapter, quantile_levels)
        self.variable_encoder = ITransformerEncoder(data_spec, model_size, n_heads, n_layers, feedforward_size,
                                                    dropout, use_future_tokens)
        self.forecast_head = nn.Linear(model_size, data_spec.horizon_steps * self.n_outputs_per_step)

    def encode(self, model_input):
        return self.variable_encoder(model_input["history_features"], model_input["future_features"])

    def forecast(self, model_input):
        return {"forecast": self.shape_output(self.forecast_head(self.encode(model_input)))}


class TimeXerLayer(nn.Module):
    def __init__(self, model_size, n_heads, feedforward_size, dropout):
        super().__init__()
        self.patch_self_attention = nn.MultiheadAttention(model_size, n_heads, dropout, batch_first=True)
        self.global_cross_attention = nn.MultiheadAttention(model_size, n_heads, dropout, batch_first=True)
        self.norm_self, self.norm_cross, self.norm_feedforward = (nn.LayerNorm(model_size) for _ in range(3))
        self.feedforward = nn.Sequential(nn.Linear(model_size, feedforward_size), nn.GELU(), nn.Dropout(dropout),
                                         nn.Linear(feedforward_size, model_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, target_tokens, variable_tokens):  # target_tokens: patches + global token (last)
        normalized = self.norm_self(target_tokens)
        target_tokens = target_tokens + self.dropout(self.patch_self_attention(normalized, normalized, normalized,
                                                                               need_weights=False)[0])
        global_token = target_tokens[:, -1:]
        query = self.norm_cross(global_token)
        global_token = global_token + self.dropout(self.global_cross_attention(query, variable_tokens, variable_tokens,
                                                                               need_weights=False)[0])
        target_tokens = torch.cat([target_tokens[:, :-1], global_token], 1)
        return target_tokens + self.dropout(self.feedforward(self.norm_feedforward(target_tokens)))


@register_model("time_xer")
class TimeXer(ForecastModel):
    def __init__(self, data_spec, patch_length=16, model_size=128, n_heads=8, n_layers=2, feedforward_size=256,
                 dropout=0.1, quantile_levels=None, input_adapter=None):
        super().__init__(data_spec, input_adapter, quantile_levels)
        self.patch_length = patch_length
        n_patches = -(-data_spec.lookback_steps // patch_length)                  # ceil: short series get left-padded
        self.left_padding = n_patches * patch_length - data_spec.lookback_steps
        self.patch_embedding = nn.Linear(patch_length, model_size)
        self.patch_position = nn.Parameter(torch.randn(1, n_patches, model_size) * 0.02)
        self.global_token = nn.Parameter(torch.randn(1, 1, model_size) * 0.02)
        self.other_variable_indices = [i for i in range(data_spec.n_history_features)
                                       if i != data_spec.target_feature_index]
        self.history_variable_embedding = nn.Linear(data_spec.lookback_steps, model_size)
        self.future_variable_embedding = nn.Linear(data_spec.horizon_steps, model_size)
        self.layers = nn.ModuleList(TimeXerLayer(model_size, n_heads, feedforward_size, dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(model_size)
        self.dropout = nn.Dropout(dropout)
        self.forecast_head = nn.Linear((n_patches + 1) * model_size, data_spec.horizon_steps * self.n_outputs_per_step)

    def encode(self, model_input):
        history = model_input["history_features"]
        target_series = history[..., self.data_spec.target_feature_index]
        if self.left_padding:  # repeat the first value so non-overlapping patches tile the lookback exactly
            target_series = torch.cat([target_series[:, :1].expand(-1, self.left_padding), target_series], 1)
        patches = self.patch_embedding(target_series.unfold(-1, self.patch_length, self.patch_length))
        target_tokens = torch.cat([self.dropout(patches + self.patch_position),
                                   self.global_token.expand(history.shape[0], -1, -1)], 1)
        variable_tokens = torch.cat([
            self.history_variable_embedding(history[..., self.other_variable_indices].transpose(1, 2)),
            self.future_variable_embedding(model_input["future_features"].transpose(1, 2))], 1)
        for layer in self.layers:
            target_tokens = layer(target_tokens, variable_tokens)
        return self.norm(target_tokens).flatten(1)                     # [batch, (patches + 1) * model_size]

    def forecast(self, model_input):
        return {"forecast": self.shape_output(self.forecast_head(self.encode(model_input)))}
