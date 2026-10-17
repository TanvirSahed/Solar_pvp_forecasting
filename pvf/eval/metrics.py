"""Metrics on normalised power (y = P/P_rated) over valid daytime targets (mask==1).

Since y is already divided by the AC rating, MAE/RMSE on y ARE nMAE/nRMSE (fraction of capacity).
Multiply by rated_kw for kW. All functions accept numpy arrays shaped [N, H] (or [N, H, Q]).
"""
from __future__ import annotations

import numpy as np


def _m(a, mask):
    return np.where(mask > 0, a, np.nan)


def point_metrics(pred, y, mask, sp=None, rated_kw=1.0, ramp_theta=0.1) -> dict:
    e = _m(pred - y, mask)
    ae = np.abs(e)
    out = {
        "nMAE_%": 100 * np.nanmean(ae),
        "nRMSE_%": 100 * np.sqrt(np.nanmean(e ** 2)),
        "nBias_%": 100 * np.nanmean(e),
        "P95_abs_err_%": 100 * np.nanpercentile(ae, 95),
        "MAE_kW": rated_kw * np.nanmean(ae),
        "RMSE_kW": rated_kw * np.sqrt(np.nanmean(e ** 2)),
    }
    yy = _m(y, mask)
    out["R2"] = 1 - np.nansum(e ** 2) / np.nansum((yy - np.nanmean(yy)) ** 2)
    if sp is not None:
        es = _m(sp - y, mask)
        out["skill_vs_SP_rmse"] = 1 - np.sqrt(np.nanmean(e ** 2)) / np.sqrt(np.nanmean(es ** 2))
        out["skill_vs_SP_mae"] = 1 - np.nanmean(ae) / np.nanmean(np.abs(es))
    # ramp tracking along the horizon: |dy| > theta (theta * P_rated; site default in sites.py)
    dy, dp = np.diff(y, axis=1), np.diff(pred, axis=1)
    m2 = (mask[:, 1:] * mask[:, :-1]) > 0
    ramp = m2 & (np.abs(dy) > ramp_theta)
    out["ramp_MAE_%"] = 100 * float(np.abs(dp - dy)[ramp].mean()) if ramp.any() else np.nan
    out["n"] = int(mask.sum())
    return {k: float(v) for k, v in out.items()}


def per_horizon(pred, y, mask, sp=None) -> dict:
    e = _m(pred - y, mask)
    d = {"nMAE_%": 100 * np.nanmean(np.abs(e), 0), "nRMSE_%": 100 * np.sqrt(np.nanmean(e ** 2, 0))}
    if sp is not None:
        es = _m(sp - y, mask)
        d["skill_vs_SP_rmse"] = 1 - np.sqrt(np.nanmean(e ** 2, 0)) / np.sqrt(np.nanmean(es ** 2, 0))
    return {k: v.tolist() for k, v in d.items()}


def pinball(q_pred, y, mask, quantiles) -> float:
    q = np.asarray(quantiles)[None, None, :]
    d = y[..., None] - q_pred
    loss = np.maximum(q * d, (q - 1) * d).mean(-1)
    return float(np.nanmean(_m(loss, mask)))


def crps_from_quantiles(q_pred, y, mask, quantiles) -> float:
    """CRPS ~= 2 * mean pinball over a dense-enough quantile grid."""
    return 2.0 * pinball(q_pred, y, mask, quantiles)


def interval_metrics(lo, hi, y, mask, nominal: float) -> dict:
    inside = _m(((y >= lo) & (y <= hi)).astype(float), mask)
    width = _m(hi - lo, mask)
    yr = np.nanmax(_m(y, mask)) - np.nanmin(_m(y, mask))
    return {f"PICP{int(nominal*100)}": float(np.nanmean(inside)), f"PINAW{int(nominal*100)}": float(np.nanmean(width) / yr)}


def prob_metrics(q_pred, y, mask, quantiles) -> dict:
    qs = list(quantiles)
    out = {"pinball": pinball(q_pred, y, mask, qs), "CRPS": crps_from_quantiles(q_pred, y, mask, qs)}
    for lo_q in (0.05, 0.1):
        hi_q = round(1 - lo_q, 4)
        if lo_q in qs and hi_q in qs:
            out.update(interval_metrics(q_pred[..., qs.index(lo_q)], q_pred[..., qs.index(hi_q)], y, mask, hi_q - lo_q))
    return out
