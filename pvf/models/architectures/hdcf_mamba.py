"""HDCF-Mamba (Shi et al. 2026, Energies 19(5):1315 -> ref/R06_HDCF-Mamba.pdf), from Sec. 3.

    embedded     = measured values W_v + positional encoding + calendar embedding      (Eq. 2, DataEmbedding)
    global       = Mamba(embedded)                    single block                     (Sec. 3.5)
    local        = MSFEM(embedded): AvgPool(k=7, stride s) -> upsample, s in {2,4,8}    (Eq. 3-4)
                   -> stacked [batch, 3, time, size] -> TimeBlock (2 Inception Conv2d)  (Eq. 6)
                   -> reshape + linear + LayerNorm + ReLU                               (Eq. 7-8)
    fused        = concatenate local and global ALONG TIME -> [batch, 2 * lookback, size] (Eq. 11)
    forecast     = Linear(size -> 1)( Linear(2 * lookback -> horizon)(fused) )           (Eq. 12-15)

Deviation (switchable): use_future_features=True appends the known-future features to each decoded step before
the last projection. The paper has no future covariates; set it False for the faithful version.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...model_inputs.adapter_steps import SplitCalendarFeatures
from ...model_inputs.input_adapter import InputAdapter
from ..base import ForecastModel, register_model
from ..blocks.common_blocks import sinusoidal_positional_encoding
from ..blocks.mamba_blocks import MambaLayer


class InceptionConvolution2d(nn.Module):
    """n parallel 3x3 Conv2d kernels, averaged (TimesNet Inception block, as described in Sec. 3.4.2)."""

    def __init__(self, input_channels, output_channels, n_kernels=6):
        super().__init__()
        self.convolutions = nn.ModuleList(nn.Conv2d(input_channels, output_channels, 3, padding=1)
                                          for _ in range(n_kernels))

    def forward(self, x):
        return torch.stack([convolution(x) for convolution in self.convolutions], -1).mean(-1)


class MultiScaleLocalExtractor(nn.Module):
    """MSFEM: local fluctuations at several time resolutions. [batch, time, size] -> same shape."""

    def __init__(self, model_size, timeblock_size=32, pool_kernel=7, pool_strides=(2, 4, 8), n_kernels=6):
        super().__init__()
        self.pool_kernel, self.pool_strides = pool_kernel, tuple(pool_strides)
        n_scales = len(self.pool_strides)
        self.timeblock = nn.Sequential(InceptionConvolution2d(n_scales, timeblock_size, n_kernels), nn.GELU(),
                                       InceptionConvolution2d(timeblock_size, n_scales, n_kernels))
        self.timeblock_norm = nn.LayerNorm([n_scales, model_size])
        self.scale_projection = nn.Linear(n_scales * model_size, model_size)
        self.output_norm = nn.LayerNorm(model_size)

    def forward(self, embedded):
        batch, time, _ = embedded.shape
        channels_first = embedded.transpose(1, 2)
        scales = [F.interpolate(F.avg_pool1d(channels_first, self.pool_kernel, stride, padding=self.pool_kernel // 2),
                                size=time, mode="linear", align_corners=False).transpose(1, 2)
                  for stride in self.pool_strides]
        stacked = torch.stack(scales, 1)                                              # [batch, scales, time, size]
        stacked = stacked + self.timeblock(stacked)
        stacked = self.timeblock_norm(stacked.permute(0, 2, 1, 3))                    # [batch, time, scales, size]
        return F.relu(self.output_norm(self.scale_projection(stacked.reshape(batch, time, -1))))


class HDCFEncoder(nn.Module):
    """Embedding + global Mamba branch + local MSFEM branch, fused along time -> [batch, 2 * lookback, size].
    Reusable in fusions (see fusion_example.py)."""

    def __init__(self, n_measured_features, n_calendar_features, lookback_steps, model_size=64, state_size=16,
                 expansion_factor=2, timeblock_size=32, pool_strides=(2, 4, 8), dropout=0.1, scan_backend="auto"):
        super().__init__()
        self.value_embedding = nn.Linear(n_measured_features, model_size)
        self.calendar_embedding = nn.Linear(n_calendar_features, model_size) if n_calendar_features else None
        self.register_buffer("positional_encoding", sinusoidal_positional_encoding(lookback_steps, model_size),
                             persistent=False)
        self.dropout = nn.Dropout(dropout)
        self.global_branch = MambaLayer(model_size, state_size, expansion_factor=expansion_factor, dropout=dropout,
                                        scan_backend=scan_backend)
        self.local_branch = MultiScaleLocalExtractor(model_size, timeblock_size, pool_strides=pool_strides)
        self.output_size = model_size

    def forward(self, measurement_history, calendar_history):
        embedded = self.value_embedding(measurement_history) + self.positional_encoding
        if self.calendar_embedding is not None:
            embedded = embedded + self.calendar_embedding(calendar_history)
        embedded = self.dropout(embedded)
        return torch.cat([self.local_branch(embedded), self.global_branch(embedded)], 1)


@register_model("hdcf_mamba")
class HDCFMamba(ForecastModel):
    def __init__(self, data_spec, model_size=64, state_size=16, expansion_factor=2, timeblock_size=32,
                 pool_strides=(2, 4, 8), dropout=0.1, use_future_features=True, scan_backend="auto",
                 quantile_levels=None, input_adapter=None):
        calendar_split = SplitCalendarFeatures(data_spec)
        super().__init__(data_spec, input_adapter or InputAdapter([calendar_split], "HDCFMamba"), quantile_levels)
        self.encoder = HDCFEncoder(len(calendar_split.measurement_indices), len(calendar_split.calendar_indices),
                                   data_spec.lookback_steps, model_size, state_size, expansion_factor, timeblock_size,
                                   pool_strides, dropout, scan_backend)
        self.time_map = nn.Linear(2 * data_spec.lookback_steps, data_spec.horizon_steps)    # decoder step 1
        self.use_future_features = use_future_features
        output_input_size = model_size + (data_spec.n_future_features if use_future_features else 0)
        self.output_projection = nn.Linear(output_input_size, self.n_outputs_per_step)       # decoder step 2

    def encode(self, model_input):
        return self.encoder(model_input["measurement_history"], model_input["calendar_history"])

    def forecast(self, model_input):
        per_step = self.time_map(self.encode(model_input).transpose(1, 2)).transpose(1, 2)  # [batch, horizon, size]
        if self.use_future_features:
            per_step = torch.cat([per_step, model_input["future_features"]], -1)
        return {"forecast": self.shape_output(self.output_projection(per_step))}
