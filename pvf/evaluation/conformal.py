"""Prediction intervals with guaranteed coverage, added after training to any model.

split_conformal_intervals          symmetric intervals around any point forecaster (per horizon step)
conformalized_quantile_regression  widen/narrow a quantile model's interval (Romano et al. 2019, F15; mtsa_qrn, R08)
adaptive_conformal_intervals       online intervals that adapt under drift (Gibbs & Candes 2021, F16; R07)
Arrays are [samples, horizon]. Calibration and test samples must not overlap in time.
"""
from __future__ import annotations

import numpy as np


def _conformal_quantile(scores, miscoverage):
    """Per horizon step: the finite-sample-corrected (1 - miscoverage) quantile of the calibration scores."""
    n_scores = np.sum(np.isfinite(scores), 0)
    level = np.minimum(1.0, np.ceil((n_scores + 1) * (1 - miscoverage)) / np.maximum(n_scores, 1))
    return np.array([np.nanquantile(scores[:, h], level[h]) for h in range(scores.shape[1])])


def split_conformal_intervals(calibration_forecast, calibration_target, calibration_mask, test_forecast,
                              miscoverage=0.1):
    scores = np.where(calibration_mask > 0, np.abs(calibration_target - calibration_forecast), np.nan)
    half_width = _conformal_quantile(scores, miscoverage)
    return test_forecast - half_width, test_forecast + half_width


def conformalized_quantile_regression(calibration_lower, calibration_upper, calibration_target, calibration_mask,
                                      test_lower, test_upper, miscoverage=0.1):
    scores = np.where(calibration_mask > 0, np.maximum(calibration_lower - calibration_target,
                                                       calibration_target - calibration_upper), np.nan)
    correction = _conformal_quantile(scores, miscoverage)
    return test_lower - correction, test_upper + correction


def adaptive_conformal_intervals(forecast, target, target_mask, origin_index, miscoverage=0.1, adaptation_rate=0.005,
                                 window=500, initial_scores=None):
    """Online intervals over time-ordered samples (rows sorted by origin_index). The outcome of origin i at horizon
    step h is only known at origin_index[i] + h + 1, so it enters the history only from then on (no look-ahead).
    Returns lower, upper and the miscoverage level used at each point, all [samples, horizon]."""
    n_samples, horizon = forecast.shape
    lower, upper = np.full((n_samples, horizon), np.nan), np.full((n_samples, horizon), np.nan)
    level_used = np.full((n_samples, horizon), miscoverage)
    for h in range(horizon):
        current_level = miscoverage
        past_scores = ([] if initial_scores is None
                       else list(initial_scores[:, h][np.isfinite(initial_scores[:, h])][-window:]))
        not_yet_revealed: list[tuple[int, int]] = []          # (time the outcome is known, row)
        for i in range(n_samples):
            while not_yet_revealed and not_yet_revealed[0][0] <= origin_index[i]:
                _, row = not_yet_revealed.pop(0)
                if np.isfinite(lower[row, h]):
                    missed = float(not (lower[row, h] <= target[row, h] <= upper[row, h]))
                    current_level += adaptation_rate * (miscoverage - missed)
                past_scores.append(abs(target[row, h] - forecast[row, h]))
            level_used[i, h] = current_level
            if len(past_scores) >= 20:
                half_width = np.quantile(past_scores[-window:], min(1.0, max(0.0, 1 - current_level)))
                lower[i, h], upper[i, h] = forecast[i, h] - half_width, forecast[i, h] + half_width
            if target_mask[i, h] > 0:
                not_yet_revealed.append((int(origin_index[i]) + h + 1, i))
    return lower, upper, level_used
