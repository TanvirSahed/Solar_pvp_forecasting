"""Which split (train / validation / test / holdout) every timestep belongs to.

mode "blocked" (default): the calendar is cut into `block_days` blocks assigned cyclically by `block_pattern`
    (default 4 x train, 1 x validation, 1 x test -> ~67/17/17), so every season is in every split.
    Sample rules (windows.py): all targets of a sample lie in one block, and training input windows never contain
    validation/test/holdout timesteps (purge). Neighbouring blocks are correlated, so blocked scores are slightly
    optimistic -> confirm final claims on untouched `holdout_months`.
mode "months": explicit month lists, e.g. train 1-8, validation 9, test 10-12 (pure chronological test).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "validation", "test", "holdout")
EVALUATION_SPLIT_NAMES = ("validation", "test", "holdout")


@dataclass
class SplitConfig:
    mode: str = "blocked"                       # "blocked" | "months"
    block_days: int = 7
    block_pattern: list = field(default_factory=lambda: ["train"] * 4 + ["validation", "test"])
    train_months: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    validation_months: list = field(default_factory=lambda: [9])
    test_months: list = field(default_factory=lambda: [10, 11, 12])
    holdout_months: list = field(default_factory=list)   # e.g. [12]: untouched final test, used by both modes


def assign_split_roles(time_index: pd.DatetimeIndex, split: SplitConfig) -> np.ndarray:
    roles = np.full(len(time_index), "none", dtype=object)
    if split.mode == "blocked":
        unknown = set(split.block_pattern) - set(SPLIT_NAMES)
        if unknown:
            raise ValueError(f"block_pattern has unknown split names {unknown}; use {SPLIT_NAMES}")
        day_number = (time_index.normalize() - time_index[0].normalize()).days.to_numpy()
        block_number = day_number // int(split.block_days)
        roles[:] = np.array(list(split.block_pattern), dtype=object)[block_number % len(split.block_pattern)]
    elif split.mode == "months":
        for role, months in (("train", split.train_months), ("validation", split.validation_months),
                             ("test", split.test_months)):
            roles[np.isin(time_index.month, list(months))] = role
    else:
        raise ValueError(f"unknown split mode {split.mode!r}; use 'blocked' or 'months'")
    if split.holdout_months:
        roles[np.isin(time_index.month, list(split.holdout_months))] = "holdout"
    return roles
