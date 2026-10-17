"""Standard metrics for every trained model, so runs are directly comparable.

Inputs are normalised power (P / rated power) over scored targets (target_mask == 1: daytime and measured), so
MAE/RMSE on them ARE normalised errors; *_pct = percent of rated power. Arrays: [samples, horizon(, quantiles)].
"""
from __future__ import annotations

import numpy as np

METRIC_DESCRIPTIONS = {
    "nmae_pct": "mean absolute error, % of rated power",
    "nrmse_pct": "root mean squared error, % of rated power",
    "nbias_pct": "mean error (forecast - actual), % of rated power; > 0 = over-forecast",
    "p95_abs_error_pct": "95th percentile absolute error, % of rated power (large-miss risk)",
    "mae_kw": "mean absolute error in kW",
    "rmse_kw": "root mean squared error in kW",
    "r2": "coefficient of determination",
    "rmse_skill_vs_persistence": "1 - RMSE / RMSE(smart persistence); > 0 beats the physics reference",
    "mae_skill_vs_persistence": "1 - MAE / MAE(smart persistence)",
    "ramp_mae_pct": "error of step-to-step changes during ramps (|dP| > ramp threshold), % of rated power",
    "n_scored": "number of scored (sample, horizon) points",
    "pinball": "mean quantile (pinball) loss",
    "crps": "continuous ranked probability score (from quantiles)",
    "picp80": "share of actuals inside the 10-90 % interval (ideal 0.80)",
    "pinaw80": "mean 10-90 % interval width / actual range",
    "picp90": "share of actuals inside the 5-95 % interval (ideal 0.90)",
    "pinaw90": "mean 5-95 % interval width / actual range",
}


def _scored_only(values, target_mask):
    return np.where(target_mask > 0, values, np.nan)


def point_forecast_metrics(forecast, target, target_mask, smart_persistence, rated_power_kw=1.0,
                           ramp_threshold_fraction=0.1) -> dict:
    error = _scored_only(forecast - target, target_mask)
    absolute_error = np.abs(error)
    persistence_error = _scored_only(smart_persistence - target, target_mask)
    scored_target = _scored_only(target, target_mask)
    rmse = np.sqrt(np.nanmean(error ** 2))
    metrics = {
        "nmae_pct": 100 * np.nanmean(absolute_error),
        "nrmse_pct": 100 * rmse,
        "nbias_pct": 100 * np.nanmean(error),
        "p95_abs_error_pct": 100 * np.nanpercentile(absolute_error, 95),
        "mae_kw": rated_power_kw * np.nanmean(absolute_error),
        "rmse_kw": rated_power_kw * rmse,
        "r2": 1 - np.nansum(error ** 2) / np.nansum((scored_target - np.nanmean(scored_target)) ** 2),
        "rmse_skill_vs_persistence": 1 - rmse / np.sqrt(np.nanmean(persistence_error ** 2)),
        "mae_skill_vs_persistence": 1 - np.nanmean(absolute_error) / np.nanmean(np.abs(persistence_error)),
    }
    target_change, forecast_change = np.diff(target, axis=1), np.diff(forecast, axis=1)
    is_ramp = (target_mask[:, 1:] * target_mask[:, :-1] > 0) & (np.abs(target_change) > ramp_threshold_fraction)
    metrics["ramp_mae_pct"] = 100 * float(np.abs(forecast_change - target_change)[is_ramp].mean()) if is_ramp.any() else np.nan
    metrics["n_scored"] = int(target_mask.sum())
    return {name: float(value) for name, value in metrics.items()}


def per_horizon_metrics(forecast, target, target_mask, smart_persistence) -> dict:
    """Metric curves over the horizon (one value per step ahead) - where in the day-ahead range a model wins or fails."""
    error = _scored_only(forecast - target, target_mask)
    persistence_error = _scored_only(smart_persistence - target, target_mask)
    rmse = np.sqrt(np.nanmean(error ** 2, 0))
    return {"nmae_pct": (100 * np.nanmean(np.abs(error), 0)).tolist(),
            "nrmse_pct": (100 * rmse).tolist(),
            "rmse_skill_vs_persistence": (1 - rmse / np.sqrt(np.nanmean(persistence_error ** 2, 0))).tolist()}


def quantile_forecast_metrics(quantile_forecast, target, target_mask, quantile_levels) -> dict:
    levels = list(quantile_levels)
    error = target[..., None] - quantile_forecast
    pinball_per_point = np.maximum(np.asarray(levels) * error, (np.asarray(levels) - 1) * error).mean(-1)
    pinball = float(np.nanmean(_scored_only(pinball_per_point, target_mask)))
    metrics = {"pinball": pinball, "crps": 2.0 * pinball}            # CRPS ~= 2 x mean pinball on a dense grid
    scored_target = _scored_only(target, target_mask)
    target_range = np.nanmax(scored_target) - np.nanmin(scored_target)
    for lower, upper, name in ((0.1, 0.9, "80"), (0.05, 0.95, "90")):
        if lower in levels and upper in levels:
            low, high = quantile_forecast[..., levels.index(lower)], quantile_forecast[..., levels.index(upper)]
            inside = _scored_only(((target >= low) & (target <= high)).astype(float), target_mask)
            metrics[f"picp{name}"] = float(np.nanmean(inside))
            metrics[f"pinaw{name}"] = float(np.nanmean(_scored_only(high - low, target_mask)) / target_range)
    return metrics


def evaluate_predictions(predictions: dict, quantile_levels, rated_power_kw, ramp_threshold_fraction) -> dict:
    """predictions from training.predict -> {"overall": {...}, "per_horizon": {...}}. Quantile models are scored on
    their median for the point metrics, plus probabilistic metrics."""
    forecast = predictions["forecast"]
    overall = {}
    if quantile_levels:
        overall.update(quantile_forecast_metrics(forecast, predictions["target"], predictions["target_mask"],
                                                 quantile_levels))
        levels = list(quantile_levels)
        forecast = forecast[..., levels.index(0.5)] if 0.5 in levels else forecast.mean(-1)
    forecast = np.clip(forecast, 0.0, None)                          # PV power cannot be negative
    overall.update(point_forecast_metrics(forecast, predictions["target"], predictions["target_mask"],
                                          predictions["smart_persistence"], rated_power_kw, ramp_threshold_fraction))
    return {"overall": overall, "per_horizon": per_horizon_metrics(forecast, predictions["target"],
                                                                   predictions["target_mask"],
                                                                   predictions["smart_persistence"])}
