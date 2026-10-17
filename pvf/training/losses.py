"""Loss functions, selectable by name. Every loss is masked: night and missing targets are ignored.

Signature: loss(forecast, target, target_mask, quantile_levels=None, **loss_options) -> scalar
    forecast [batch, horizon] (point) or [batch, horizon, n_quantiles]; target, target_mask [batch, horizon]
Add a loss: write the per-element function and put it in LOSSES.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(per_element, target_mask):
    return (per_element * target_mask).sum() / target_mask.sum().clamp_min(1.0)


def mean_squared_error(forecast, target, target_mask, quantile_levels=None, **_):
    return _masked_mean((forecast - target) ** 2, target_mask)


def mean_absolute_error(forecast, target, target_mask, quantile_levels=None, **_):
    return _masked_mean((forecast - target).abs(), target_mask)


def huber(forecast, target, target_mask, quantile_levels=None, delta=0.05, **_):
    """Squared for errors below delta (5 % of rated power), absolute above: robust to outliers."""
    return _masked_mean(F.huber_loss(forecast, target, reduction="none", delta=delta), target_mask)


def pinball(forecast, target, target_mask, quantile_levels, **_):
    """Quantile loss averaged over quantile levels (forecast [batch, horizon, n_quantiles])."""
    levels = torch.tensor(quantile_levels, device=forecast.device, dtype=forecast.dtype)
    error = target.unsqueeze(-1) - forecast
    return _masked_mean(torch.maximum(levels * error, (levels - 1) * error).mean(-1), target_mask)


LOSSES = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "huber": huber,
    "pinball": pinball,
}


def get_loss(name: str):
    if name not in LOSSES:
        raise KeyError(f"unknown loss {name!r}. Available: {list(LOSSES)}")
    return LOSSES[name]
