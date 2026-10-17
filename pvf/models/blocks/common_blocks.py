"""Reusable neural building blocks. Architectures are compositions of these - swap, rewire or fuse them freely."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(input_size: int, hidden_size: int, output_size: int, n_layers: int = 2, dropout: float = 0.1,
             activation=nn.GELU) -> nn.Sequential:
    layers, size = [], input_size
    for _ in range(n_layers - 1):
        layers += [nn.Linear(size, hidden_size), activation(), nn.Dropout(dropout)]
        size = hidden_size
    layers.append(nn.Linear(size, output_size))
    return nn.Sequential(*layers)


def make_transformer_encoder(model_size, n_heads, n_layers, feedforward_size, dropout) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(model_size, n_heads, feedforward_size, dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)


def sinusoidal_positional_encoding(length: int, model_size: int) -> torch.Tensor:
    encoding = torch.zeros(length, model_size)
    position = torch.arange(length).unsqueeze(1).float()
    frequency = torch.exp(torch.arange(0, model_size, 2).float() * (-math.log(10000.0) / model_size))
    encoding[:, 0::2] = torch.sin(position * frequency)
    encoding[:, 1::2] = torch.cos(position * frequency[: encoding[:, 1::2].shape[1]])
    return encoding


class ForecastHead(nn.Module):
    """Representation [batch, representation_size] (+ future features [batch, horizon, n_future]) -> forecast.

    use_future_features=True: the representation is expanded to one vector per horizon step and a shared MLP
    reads [step vector, that step's future features]. Gives every encoder the same access to future clear sky /
    calendar / NWP information - one switch to ablate it."""

    def __init__(self, representation_size: int, horizon_steps: int, n_future_features: int = 0,
                 use_future_features: bool = True, step_size: int = 32, n_outputs_per_step: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.use_future_features = use_future_features and n_future_features > 0
        if self.use_future_features:
            self.expand_to_steps = nn.Linear(representation_size, horizon_steps * step_size)
            self.step_mlp = make_mlp(step_size + n_future_features, 2 * step_size, n_outputs_per_step, 2, dropout)
        else:
            self.linear = nn.Linear(representation_size, horizon_steps * n_outputs_per_step)

    def forward(self, representation, future_features=None):     # -> [batch, horizon, n_outputs_per_step]
        batch = representation.shape[0]
        if self.use_future_features:
            step_vectors = self.expand_to_steps(representation).view(batch, self.horizon_steps, -1)
            return self.step_mlp(torch.cat([step_vectors, future_features], -1))
        return self.linear(representation).view(batch, self.horizon_steps, -1)


class ReversibleInstanceNorm(nn.Module):
    """RevIN (Kim et al. 2022): normalise each channel over time, invert for the forecast channel afterwards.
    Optional here because inputs are already physically scaled."""

    def __init__(self, n_channels: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_channels)) if affine else None
        self.bias = nn.Parameter(torch.zeros(n_channels)) if affine else None

    def normalize(self, x):                                       # x: [batch, time, channels]
        self.mean = x.mean(1, keepdim=True).detach()
        self.std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps).detach()
        x = (x - self.mean) / self.std
        return x * self.weight + self.bias if self.weight is not None else x

    def denormalize(self, y, channel: int):                       # y: [batch, horizon] of one channel
        if self.weight is not None:
            y = (y - self.bias[channel]) / (self.weight[channel] + self.eps)
        return y * self.std[:, 0, channel:channel + 1] + self.mean[:, 0, channel:channel + 1]


class SeriesDecomposition(nn.Module):
    """Moving-average trend + seasonal remainder (PowerMamba). x: [batch, time, channels]."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):                                         # -> (seasonal, trend)
        pad_left, pad_right = (self.kernel_size - 1) // 2, self.kernel_size // 2
        padded = torch.cat([x[:, :1].expand(-1, pad_left, -1), x, x[:, -1:].expand(-1, pad_right, -1)], 1)
        trend = F.avg_pool1d(padded.transpose(1, 2), self.kernel_size, stride=1).transpose(1, 2)
        return x - trend, trend


class PatchEmbedding(nn.Module):
    """[batch, time, channels] -> [batch * channels, n_patches, model_size] (channel-independent, PatchTST)."""

    def __init__(self, patch_length: int, stride: int, model_size: int, lookback_steps: int, dropout: float = 0.0):
        super().__init__()
        self.patch_length, self.stride = patch_length, stride
        self.n_patches = (lookback_steps - patch_length) // stride + 2   # +1 patch from end padding (PatchTST)
        self.projection = nn.Linear(patch_length, model_size)
        self.position = nn.Parameter(torch.randn(1, self.n_patches, model_size) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, time, channels = x.shape
        series = x.permute(0, 2, 1).reshape(batch * channels, time)
        series = torch.cat([series, series[:, -1:].expand(-1, self.stride)], 1)   # replicate-pad one stride
        patches = series.unfold(-1, self.patch_length, self.stride)              # [b*c, n_patches, patch_length]
        return self.dropout(self.projection(patches) + self.position)


class FourierKANLayer(nn.Module):
    """Kolmogorov-Arnold layer with a Fourier basis (RFKAN, R16):
    y_j = sum_i sum_k a_ijk cos(k x_i) + b_ijk sin(k x_i) + bias_j  (+ a SiLU base path as in the KAN paper)."""

    def __init__(self, input_size: int, output_size: int, n_harmonics: int = 4, use_base_path: bool = True):
        super().__init__()
        self.harmonics = nn.Parameter(torch.arange(1, n_harmonics + 1).float(), requires_grad=False)
        self.coefficients = nn.Parameter(
            torch.randn(2, output_size, input_size, n_harmonics) / (math.sqrt(input_size) * math.sqrt(n_harmonics)))
        self.bias = nn.Parameter(torch.zeros(output_size))
        self.base_path = nn.Linear(input_size, output_size, bias=False) if use_base_path else None

    def forward(self, x):
        angles = x.unsqueeze(-1) * self.harmonics                      # [..., input_size, n_harmonics]
        y = (torch.einsum("...ik,oik->...o", torch.cos(angles), self.coefficients[0])
             + torch.einsum("...ik,oik->...o", torch.sin(angles), self.coefficients[1]) + self.bias)
        return y + self.base_path(F.silu(x)) if self.base_path is not None else y


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(context, x, strength):
        context.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(context, gradient):
        return -context.strength * gradient, None


def reverse_gradient(x, strength: float = 1.0):
    """Identity forward, gradient * -strength backward (DANN, Ganin et al. 2016)."""
    return _GradientReversal.apply(x, strength)
