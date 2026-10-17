"""Distribution-free calibration layers (post-hoc, model-agnostic).

split_conformal - symmetric intervals around ANY point forecaster (per horizon).
cqr             - Conformalized Quantile Regression (Romano et al. 2019, F15) for quantile models (MTSA-QRN, R08).
aci             - Adaptive Conformal Inference (Gibbs & Candes 2021, F16) for online / DT replay under drift (R07).
All arrays: [N, H]; calibration and test sets must be chronologically disjoint.
"""
from __future__ import annotations

import numpy as np


def _q(scores, alpha):
    n = np.sum(np.isfinite(scores), 0)
    level = np.minimum(1.0, np.ceil((n + 1) * (1 - alpha)) / np.maximum(n, 1))
    return np.array([np.nanquantile(scores[:, h], level[h]) for h in range(scores.shape[1])])


def split_conformal(pred_cal, y_cal, mask_cal, pred_test, alpha=0.1):
    s = np.where(mask_cal > 0, np.abs(y_cal - pred_cal), np.nan)
    q = _q(s, alpha)
    return pred_test - q, pred_test + q


def cqr(lo_cal, hi_cal, y_cal, mask_cal, lo_test, hi_test, alpha=0.1):
    s = np.where(mask_cal > 0, np.maximum(lo_cal - y_cal, y_cal - hi_cal), np.nan)
    q = _q(s, alpha)
    return lo_test - q, hi_test + q


def aci(pred, y, mask, t0, alpha=0.1, gamma=0.005, window=500, init_scores=None):
    """Online ACI over time-ordered samples (rows sorted by origin index t0).
    The outcome of origin i at horizon h (step h+1) is only revealed at t0[i]+h+1, so residuals and coverage
    feedback enter the history only once an origin >= that time is being forecast (no look-ahead).
    Returns lo, hi [N, H] and the alpha_t trajectory [N, H]."""
    N, H = pred.shape
    lo, hi, at = np.full((N, H), np.nan), np.full((N, H), np.nan), np.full((N, H), alpha)
    for h in range(H):
        a = alpha
        hist = list(init_scores[:, h][np.isfinite(init_scores[:, h])][-window:]) if init_scores is not None else []
        pending: list[tuple[int, int]] = []          # (reveal_time, row)
        for i in range(N):
            while pending and pending[0][0] <= t0[i]:
                _, j = pending.pop(0)
                if np.isfinite(lo[j, h]):
                    a = a + gamma * (alpha - float(not (lo[j, h] <= y[j, h] <= hi[j, h])))
                hist.append(abs(y[j, h] - pred[j, h]))
            at[i, h] = a
            if len(hist) >= 20:
                q = np.quantile(hist[-window:], min(1.0, max(0.0, 1 - a)))
                lo[i, h], hi[i, h] = pred[i, h] - q, pred[i, h] + q
            if mask[i, h] > 0:
                pending.append((int(t0[i]) + h + 1, i))
    return lo, hi, at
