"""Site record: ac_power_kw, irradiance_wm2, module_temperature_c and clear_sky_irradiance_wm2 on a uniform grid
(the same processing as the processed datasets: interval means, Ineichen clear sky from pvlib).

Index: tz-aware local time, label = interval START, value = interval MEAN.
Clear sky is evaluated at the interval midpoint so it matches interval-mean measurements.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .sites import DATA_DIR, Site, get_site

CACHE_DIR = DATA_DIR / "cache"
SITE_RECORD_VERSION = "2"  # bump whenever the processing below changes (invalidates caches)


def _read_source_column(path, column, unit_factor, timezone) -> pd.Series:
    frame = pd.read_parquet(path, columns=["datetime", column])
    series = frame.set_index("datetime")[column].astype("float64") * unit_factor
    series = series[~series.index.duplicated(keep="first")].sort_index()
    # ambiguous (DST fall-back) / nonexistent (spring-forward) local times are at night -> drop them
    series.index = series.index.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    return series[series.index.notna()]


def clear_sky_irradiance(site: Site, index: pd.DatetimeIndex, resolution: str) -> pd.Series:
    import pvlib

    location = pvlib.location.Location(site.latitude, site.longitude, tz=site.timezone, altitude=site.altitude_m)
    interval_midpoints = index + pd.Timedelta(resolution) / 2
    ghi = location.get_clearsky(interval_midpoints, model="ineichen")["ghi"]
    return pd.Series(ghi.to_numpy(), index=index).clip(lower=0.0)


def load_site_record(site_name: str, resolution: str = "30min", use_cache: bool = True) -> pd.DataFrame:
    site = get_site(site_name)
    # cache key: processing version + input file sizes -> a copied or stale cache is never reused by mistake
    source_sizes = "".join(str(path.stat().st_size) for path, _, _ in site.source_columns.values() if path.exists())
    cache_key = hashlib.md5((SITE_RECORD_VERSION + source_sizes).encode()).hexdigest()[:8]
    cache_file = CACHE_DIR / f"{site.name}_{resolution}_{cache_key}.parquet"
    if use_cache and cache_file.exists():
        record = pd.read_parquet(cache_file)
    else:
        record = pd.DataFrame({
            name: _read_source_column(path, column, factor, site.timezone)
            .resample(resolution, label="left", closed="left").mean()
            for name, (path, column, factor) in site.source_columns.items()
        })
        main_year = int(pd.Series(record.index.year).mode()[0])   # SSP file has a few 2026 stragglers
        record = record[record.index.year == main_year]
        full_index = pd.date_range(record.index.min().normalize(), record.index.max().normalize() + pd.Timedelta("1D"),
                                   freq=resolution, tz=site.timezone, inclusive="left")
        record = record.reindex(full_index)
        record["clear_sky_irradiance_wm2"] = clear_sky_irradiance(site, record.index, resolution)
        is_night = record["clear_sky_irradiance_wm2"] <= 0.0
        for column in ("ac_power_kw", "irradiance_wm2"):
            record[column] = record[column].clip(lower=0.0)
            record.loc[is_night & record[column].isna(), column] = 0.0   # missing at night == no generation
        record.index.name = "time"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        record.to_parquet(cache_file)
    record.attrs.update(site=site.name, resolution=resolution, rated_power_kw=site.rated_power_kw)
    return record


def compute_smart_persistence(ac_power, clear_sky, origin_indices, horizon_steps, min_clear_sky=1.0) -> np.ndarray:
    """s(t, h) = P(t) * Gcs(t+h) / Gcs(t), and 0 when Gcs(t) <= min_clear_sky. Returns [n_origins, horizon]."""
    steps_ahead = np.arange(1, horizon_steps + 1)
    clear_sky_now = clear_sky[origin_indices][:, None]
    clear_sky_ahead = clear_sky[origin_indices[:, None] + steps_ahead[None, :]]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(clear_sky_now > min_clear_sky,
                        ac_power[origin_indices][:, None] * clear_sky_ahead / clear_sky_now, 0.0)
