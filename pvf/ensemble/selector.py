"""Round-3 adaptive selection / fusion on top of frozen forecasters (selection report row 15, Sec. 4).

The report requires RL selectors (DQN R24, PPO R25, DDPG R35) to be compared against
static-best, weighted ensemble, stacking and a contextual-bandit baseline. This module provides those baselines,
operating on saved predictions (runs/<site>/<model>/<ts>/pred_<split>.npz), so RL agents can be dropped in
with the same interface:   weights = selector.step(context_t, preds_t)  ->  update(errors once revealed).

Causality: like ACI, the error of origin i at horizon h is only revealed at t0[i]+h+1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_preds(run_dirs: list[str | Path], split: str):
    """Align several runs on common origins. Returns dict(P [N,H,K], y, mask, sp, t0, names)."""
    rs = [np.load(Path(d) / f"pred_{split}.npz") for d in run_dirs]
    common = rs[0]["t0"]
    for r in rs[1:]:
        common = np.intersect1d(common, r["t0"])
    idx = [np.searchsorted(r["t0"], common) for r in rs]
    P = np.stack([r["pred"][i] if r["pred"].ndim == 2 else np.median(r["pred"][i], -1) for r, i in zip(rs, idx)], -1)
    r0, i0 = rs[0], idx[0]
    return {"P": P, "y": r0["y"][i0], "mask": r0["mask"][i0], "sp": r0["sp"][i0], "t0": common,
            "names": [Path(d).parent.name for d in run_dirs]}


def static_best(val, test):
    k = int(np.argmin([np.nanmean(np.abs(val["P"][..., j] - val["y"])[val["mask"] > 0]) for j in range(val["P"].shape[-1])]))
    return test["P"][..., k], k


def rolling_inverse_error(d, window=96, power=1.0):
    """Per-horizon weights proportional to 1/MAE^power over the last `window` revealed errors."""
    P, y, m, t0 = d["P"], d["y"], d["mask"], d["t0"]
    N, H, K = P.shape
    out = np.empty((N, H))
    for h in range(H):
        hist, pend = [], []
        for i in range(N):
            while pend and pend[0][0] <= t0[i]:
                hist.append(pend.pop(0)[1])
            if hist:
                mae = np.mean(hist[-window:], 0) + 1e-6
                w = mae ** -power
                w /= w.sum()
            else:
                w = np.full(K, 1 / K)
            out[i, h] = P[i, h] @ w
            if m[i, h] > 0:
                pend.append((t0[i] + h + 1, np.abs(P[i, h] - y[i, h])))
    return out


class LinUCB:
    """Disjoint LinUCB contextual bandit: arm = forecaster, reward = -|error|. Context e.g. [1, kt, kt_std, h/H]."""

    def __init__(self, n_arms, d_ctx, alpha=0.5):
        self.A = np.stack([np.eye(d_ctx)] * n_arms)
        self.b = np.zeros((n_arms, d_ctx))
        self.alpha = alpha

    def select(self, x):
        Ainv = np.linalg.inv(self.A)
        theta = np.einsum("kij,kj->ki", Ainv, self.b)
        ucb = theta @ x + self.alpha * np.sqrt(np.einsum("i,kij,j->k", x, Ainv, x))
        return int(np.argmax(ucb))

    def update(self, arm, x, reward):
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x


def linucb_select(d, context, alpha=0.5):
    """context: [N, d_ctx] features available at the origin. One bandit per horizon."""
    P, y, m, t0 = d["P"], d["y"], d["mask"], d["t0"]
    N, H, K = P.shape
    out = np.empty((N, H))
    for h in range(H):
        bandit, pend = LinUCB(K, context.shape[1], alpha), []
        for i in range(N):
            while pend and pend[0][0] <= t0[i]:
                _, arm, x, r = pend.pop(0)
                bandit.update(arm, x, r)
            arm = bandit.select(context[i])
            out[i, h] = P[i, h, arm]
            if m[i, h] > 0:
                pend.append((t0[i] + h + 1, arm, context[i], -abs(P[i, h, arm] - y[i, h])))
    return out
