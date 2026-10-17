"""Data pipeline: DataConfig -> site record -> named features -> train/validation/test windows.

    data_config = DataConfig(site="nist", history_feature_names=[...], lookback_steps=96)
    data_bundle = build_data_bundle(data_config)
    data_bundle.datasets["train"][0]      # one ForecastBatch sample (see windows.py)
    data_bundle.spec                      # DataSpec: everything a model needs to know about its inputs
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .features import CALENDAR_FEATURE_NAMES, build_feature_table, check_future_features
from .site_record import load_site_record
from .sites import get_site
from .splits import EVALUATION_SPLIT_NAMES, SplitConfig, assign_split_roles
from .windows import ForecastWindowDataset, ForecastWindows

TARGET_FEATURE_NAME = "ac_power"
DEFAULT_HISTORY_FEATURE_NAMES = ["ac_power", "irradiance", "module_temperature", "clear_sky_irradiance",
                                 *CALENDAR_FEATURE_NAMES]
DEFAULT_FUTURE_FEATURE_NAMES = ["clear_sky_irradiance", "ac_power_yesterday", *CALENDAR_FEATURE_NAMES]


@dataclass
class DataConfig:
    site: str = "ssp"                                   # "ssp" | "nist"
    resolution: str = "30min"
    history_feature_names: list = field(default_factory=lambda: list(DEFAULT_HISTORY_FEATURE_NAMES))
    future_feature_names: list = field(default_factory=lambda: list(DEFAULT_FUTURE_FEATURE_NAMES))
    lookback_steps: int = 128                           # 128 x 30 min = 64 h of history
    horizon_steps: int = 48                             # 48 x 30 min = 0.5 .. 24 h ahead, predicted jointly
    daytime_clear_sky_threshold_wm2: float = 10.0       # targets count only when clear sky > this
    split: SplitConfig = field(default_factory=SplitConfig)
    domain_adaptation_months: list | None = None        # unlabelled target-domain inputs (dt_dann), e.g. [10, 11, 12]


@dataclass
class DataSpec:
    """Shape and meaning of a ForecastBatch. Passed to every model constructor."""
    site: str
    resolution: str
    lookback_steps: int
    horizon_steps: int
    history_feature_names: list
    future_feature_names: list
    rated_power_kw: float
    steps_per_day: int
    ramp_threshold_fraction: float
    time_index: pd.DatetimeIndex = field(repr=False)
    target_feature_name: str = TARGET_FEATURE_NAME

    @property
    def n_history_features(self) -> int:
        return len(self.history_feature_names)

    @property
    def n_future_features(self) -> int:
        return len(self.future_feature_names)

    @property
    def target_feature_index(self) -> int:
        return self.history_feature_names.index(self.target_feature_name)

    def history_index(self, feature_name: str) -> int:
        if feature_name not in self.history_feature_names:
            raise KeyError(f"this model needs history feature {feature_name!r}; add it to history_feature_names")
        return self.history_feature_names.index(feature_name)


@dataclass
class DataBundle:
    spec: DataSpec
    datasets: dict          # split name -> ForecastWindowDataset
    split_roles: np.ndarray
    site_record: pd.DataFrame


def build_data_bundle(config: DataConfig) -> DataBundle:
    if TARGET_FEATURE_NAME not in config.history_feature_names:
        print(f"note: {TARGET_FEATURE_NAME!r} added to history_feature_names (every model needs the past target)")
        config.history_feature_names = [TARGET_FEATURE_NAME, *config.history_feature_names]
    check_future_features(config.future_feature_names)   # leaking the future must stop the run, never be skipped
    site = get_site(config.site)
    site_record = load_site_record(config.site, config.resolution)
    steps_per_day = int(pd.Timedelta("1D") / pd.Timedelta(config.resolution))
    if "ac_power_yesterday" in config.future_feature_names and config.horizon_steps > steps_per_day:
        raise ValueError("ac_power_yesterday is only known in advance for horizons up to one day")
    windows = ForecastWindows(
        build_feature_table(site_record, config.history_feature_names),
        build_feature_table(site_record, config.future_feature_names),
        build_feature_table(site_record, [TARGET_FEATURE_NAME])[TARGET_FEATURE_NAME].to_numpy(),
        site_record["clear_sky_irradiance_wm2"].to_numpy(),
        config.lookback_steps, config.horizon_steps, config.daytime_clear_sky_threshold_wm2)
    split_roles = assign_split_roles(site_record.index, config.split)
    origins = {name: windows.valid_origins(split_roles, name) for name in ("train", *EVALUATION_SPLIT_NAMES)}
    if len(origins["validation"]) == 0:  # e.g. mode "months" without validation months
        n_validation = max(1, len(origins["train"]) // 7)
        print(f"note: no validation samples in this split -> using the last {n_validation} training samples")
        origins["validation"] = origins["train"][-n_validation:]
        origins["train"] = origins["train"][origins["train"] + config.horizon_steps < origins["validation"][0]]
    if len(origins["train"]) == 0:
        raise ValueError("no training samples: lookback + horizon may be longer than the training blocks "
                         "(raise split.block_days or lower lookback_steps)")
    if config.domain_adaptation_months:
        adaptation_roles = np.where(np.isin(site_record.index.month, config.domain_adaptation_months), "adapt", "-")
        origins["domain_adaptation"] = windows.valid_origins(adaptation_roles, "adapt")
    spec = DataSpec(config.site, config.resolution, config.lookback_steps, config.horizon_steps,
                    list(config.history_feature_names), list(config.future_feature_names), site.rated_power_kw,
                    steps_per_day, site.ramp_threshold_fraction, site_record.index)
    datasets = {name: ForecastWindowDataset(windows, o) for name, o in origins.items() if len(o)}
    return DataBundle(spec, datasets, split_roles, site_record)


def make_data_loader(dataset, batch_size: int, shuffle: bool, seed: int = 0, num_workers: int = 0) -> DataLoader:
    import torch

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                      generator=torch.Generator().manual_seed(seed), persistent_workers=num_workers > 0)


def describe_data_bundle(bundle: DataBundle) -> pd.DataFrame:
    rows = []
    for name, dataset in bundle.datasets.items():
        times = bundle.spec.time_index[dataset.origins]
        rows.append({"split": name, "samples": len(dataset), "first_origin": f"{times[0]:%Y-%m-%d}",
                     "last_origin": f"{times[-1]:%Y-%m-%d}", "months": sorted(set(times.month))})
    return pd.DataFrame(rows)


def sample_batch(bundle: DataBundle, split: str = "train", batch_size: int = 4, seed: int = 0) -> dict:
    """A few random samples as one ForecastBatch - for adapter traces, plots and quick model calls."""
    import torch

    dataset = bundle.datasets[split]
    indices = np.random.default_rng(seed).choice(len(dataset), size=min(batch_size, len(dataset)), replace=False)
    samples = [dataset[int(i)] for i in indices]
    return {key: torch.stack([s[key] for s in samples]) for key in samples[0]}
