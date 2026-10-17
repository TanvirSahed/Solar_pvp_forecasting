"""Sliding forecast windows. Every model sees the same origins, features and targets.

A ForecastBatch (what the DataLoader yields) is a dict:
    history_features   [batch, lookback, n_history_features]   values at t-lookback+1 .. t
    future_features    [batch, horizon, n_future_features]     known-in-advance values at t+1 .. t+horizon
    target             [batch, horizon]                        ac_power at t+1 .. t+horizon (P / rated power)
    target_mask        [batch, horizon]                        1 = daytime and measured -> scored and trained on
    smart_persistence  [batch, horizon]                        physics reference forecast (P / rated power)
    origin_index       [batch]                                 forecast origin t, index into DataSpec.time_index
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .site_record import compute_smart_persistence

PROTECTED_FROM_TRAINING = ("validation", "test", "holdout")


class ForecastWindows:
    """Feature arrays for the whole site record; slices them into windows on request."""

    def __init__(self, history_table, future_table, ac_power_fraction, clear_sky_wm2, lookback_steps, horizon_steps,
                 daytime_clear_sky_threshold_wm2=10.0):
        self.history_values = history_table.to_numpy(np.float32, copy=True)   # writable (pandas 3 returns read-only views)
        self.future_values = future_table.to_numpy(np.float32, copy=True)
        self.target_values = ac_power_fraction.astype(np.float32)
        self.clear_sky_wm2 = clear_sky_wm2
        self.is_daytime = clear_sky_wm2 > daytime_clear_sky_threshold_wm2
        self.lookback_steps, self.horizon_steps = lookback_steps, horizon_steps

    def valid_origins(self, split_roles: np.ndarray, split_name: str) -> np.ndarray:
        """Origins t whose horizon lies entirely in `split_name`, whose inputs are finite, with >= 1 daytime target.
        Training origins additionally may not see evaluation timesteps in their input window (purge)."""
        lookback, horizon = self.lookback_steps, self.horizon_steps
        n_steps = len(self.target_values)
        origins = np.arange(lookback - 1, n_steps - horizon)

        def count_until(flags):  # count_until(f)[i] = number of True flags before index i
            return np.concatenate([[0], np.cumsum(flags)])

        outside_split = count_until(split_roles != split_name)
        origins = origins[outside_split[origins + horizon + 1] - outside_split[origins + 1] == 0]
        if split_name == "train":
            protected = count_until(np.isin(split_roles, PROTECTED_FROM_TRAINING))
            origins = origins[protected[origins + 1] - protected[origins - lookback + 1] == 0]
        finite_history = count_until(np.isfinite(self.history_values).all(1))
        origins = origins[finite_history[origins + 1] - finite_history[origins - lookback + 1] == lookback]
        target_steps = origins[:, None] + np.arange(1, horizon + 1)[None, :]
        future_is_finite = np.isfinite(self.future_values).all(1)[target_steps].all(1)
        has_daytime_target = (np.isfinite(self.target_values) & self.is_daytime)[target_steps].any(1)
        return origins[future_is_finite & has_daytime_target]


class ForecastWindowDataset(Dataset):
    def __init__(self, windows: ForecastWindows, origins: np.ndarray):
        self.windows, self.origins = windows, np.asarray(origins)
        persistence = compute_smart_persistence(windows.target_values.astype(np.float64), windows.clear_sky_wm2,
                                                self.origins, windows.horizon_steps)
        self.smart_persistence = np.clip(np.nan_to_num(persistence, nan=0.0), 0.0, 1.2).astype(np.float32)

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, i: int) -> dict:
        w, t = self.windows, int(self.origins[i])
        target = w.target_values[t + 1:t + w.horizon_steps + 1]
        target_mask = (np.isfinite(target) & w.is_daytime[t + 1:t + w.horizon_steps + 1]).astype(np.float32)
        return {
            "history_features": torch.from_numpy(w.history_values[t - w.lookback_steps + 1:t + 1]),
            "future_features": torch.from_numpy(w.future_values[t + 1:t + w.horizon_steps + 1]),
            "target": torch.from_numpy(np.nan_to_num(target, nan=0.0)),
            "target_mask": torch.from_numpy(target_mask),
            "smart_persistence": torch.from_numpy(self.smart_persistence[i]),
            "origin_index": torch.tensor(t, dtype=torch.long),
        }
