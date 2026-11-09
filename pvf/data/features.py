"""Named, configurable model features.

Every feature is one column computed from the site record and already scaled to roughly [-1, 1]
(physical scaling, nothing is fitted -> no leakage). Choose features by name in DataConfig:

    history_feature_names = ["ac_power", "irradiance", "module_temperature", "clear_sky_irradiance", ...]
    future_feature_names  = ["clear_sky_irradiance", "ac_power_yesterday", "hour_of_day_sin", ...]

A future feature is read at the forecast target times, so it must be KNOWN IN ADVANCE (clear sky, calendar,
yesterday's power). Measured weather at target times is only allowed as an explicit "oracle" (perfect forecast).
To add a feature: write one function below and add it to FEATURES.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .nwp import nwp_column

CALENDAR_FEATURE_NAMES = ["hour_of_day_sin", "hour_of_day_cos", "day_of_year_sin", "day_of_year_cos"]


@dataclass(frozen=True)
class Feature:
    name: str
    description: str
    known_in_advance: bool
    build: Callable[[pd.DataFrame, float], np.ndarray]   # (site_record, rated_power_kw) -> values


def _hour_fraction(record):
    return (record.index.hour + record.index.minute / 60.0) / 24.0


def _day_fraction(record):
    return (record.index.dayofyear - 1) / 365.0


def _steps_per_day(record):
    return int(pd.Timedelta("1D") / pd.Timedelta(record.attrs["resolution"]))


def _clearness_index(record, _):
    is_daylight = record["clear_sky_irradiance_wm2"] > 50.0
    ratio = record["irradiance_wm2"] / record["clear_sky_irradiance_wm2"].clip(lower=1.0)
    return np.where(is_daylight, ratio.clip(0.0, 1.5), 0.0)


FEATURES: dict[str, Feature] = {f.name: f for f in [
    Feature("ac_power", "AC power / rated power", False,
            lambda r, rated: r["ac_power_kw"].to_numpy() / rated),
    Feature("irradiance", "measured irradiance / 1000 W/m2", False,
            lambda r, _: r["irradiance_wm2"].to_numpy() / 1000.0),
    Feature("module_temperature", "(module temperature - 25 C) / 25 C", False,
            lambda r, _: (r["module_temperature_c"].to_numpy() - 25.0) / 25.0),
    Feature("clear_sky_irradiance", "Ineichen clear-sky GHI / 1000 W/m2", True,
            lambda r, _: r["clear_sky_irradiance_wm2"].to_numpy() / 1000.0),
    Feature("clearness_index", "irradiance / clear-sky irradiance (0 at night)", False, _clearness_index),
    Feature("ac_power_yesterday", "AC power 24 h earlier / rated power", True,
            lambda r, rated: r["ac_power_kw"].shift(_steps_per_day(r)).to_numpy() / rated),
    Feature("hour_of_day_sin", "sin(2 pi hour/24)", True, lambda r, _: np.sin(2 * np.pi * _hour_fraction(r))),
    Feature("hour_of_day_cos", "cos(2 pi hour/24)", True, lambda r, _: np.cos(2 * np.pi * _hour_fraction(r))),
    Feature("day_of_year_sin", "sin(2 pi day/365)", True, lambda r, _: np.sin(2 * np.pi * _day_fraction(r))),
    Feature("day_of_year_cos", "cos(2 pi day/365)", True, lambda r, _: np.cos(2 * np.pi * _day_fraction(r))),
    # Archived NWP forecasts (pvf/data/nwp.py, ECMWF IFS by default). "day1" = issued ~24 h before the target time,
    # "_day2" = ~48 h before (strictly earlier than every origin of a 24 h horizon).
    Feature("nwp_irradiance", "NWP global irradiance forecast (issued ~24 h ahead) / 1000 W/m2", True,
            lambda r, _: nwp_column(r, "shortwave_radiation", 1) / 1000.0),
    Feature("nwp_cloud_cover", "NWP total cloud cover forecast (issued ~24 h ahead) / 100 %", True,
            lambda r, _: nwp_column(r, "cloud_cover", 1) / 100.0),
    Feature("nwp_temperature", "(NWP 2 m air temperature forecast (issued ~24 h ahead) - 25 C) / 25 C", True,
            lambda r, _: (nwp_column(r, "temperature_2m", 1) - 25.0) / 25.0),
    Feature("nwp_irradiance_day2", "NWP global irradiance forecast issued ~48 h ahead / 1000 W/m2", True,
            lambda r, _: nwp_column(r, "shortwave_radiation", 2) / 1000.0),
    Feature("nwp_cloud_cover_day2", "NWP total cloud cover forecast issued ~48 h ahead / 100 %", True,
            lambda r, _: nwp_column(r, "cloud_cover", 2) / 100.0),
    Feature("nwp_temperature_day2", "(NWP 2 m air temperature forecast issued ~48 h ahead - 25 C) / 25 C", True,
            lambda r, _: (nwp_column(r, "temperature_2m", 2) - 25.0) / 25.0),
    # Perfect-forecast stand-ins for NWP. Upper bound only - never report as a real forecast.
    Feature("irradiance_oracle", "measured irradiance used as a perfect forecast", True,
            lambda r, _: r["irradiance_wm2"].to_numpy() / 1000.0),
    Feature("module_temperature_oracle", "measured module temperature used as a perfect forecast", True,
            lambda r, _: (r["module_temperature_c"].to_numpy() - 25.0) / 25.0),
]}


def build_feature_table(site_record: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """One float32 column per requested feature, same index as the site record."""
    unknown = [n for n in feature_names if n not in FEATURES]
    if unknown:
        raise KeyError(f"unknown features {unknown}. Available: {list(FEATURES)}")
    rated_power_kw = float(site_record.attrs["rated_power_kw"])
    columns = {n: FEATURES[n].build(site_record, rated_power_kw) for n in dict.fromkeys(feature_names)}
    return pd.DataFrame(columns, index=site_record.index).astype("float32")


def check_future_features(future_feature_names: list[str]) -> None:
    leaking = [n for n in future_feature_names if not FEATURES[n].known_in_advance]
    if leaking:
        raise ValueError(f"{leaking} are not known in advance and would leak the future. "
                         f"Use their '_oracle' version if a perfect-forecast upper bound is intended.")


def describe_features() -> pd.DataFrame:
    return pd.DataFrame([(f.name, f.known_in_advance, f.description) for f in FEATURES.values()],
                        columns=["feature", "usable_as_future_feature", "description"])
