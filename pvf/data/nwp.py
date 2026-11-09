"""Archived numerical weather prediction (NWP) forecasts for a site, as known-in-advance inputs.

Source: Open-Meteo Previous Runs API (free, no key; non-commercial use). `<variable>_previous_dayN` is the value the
model predicted N x 24 h before the valid time, so it was available before the forecast origin:
    day1  issued ~24 h before each target time. Safe for lead times up to ~18 h (a run is published a few hours
          after its start time); slightly optimistic for the last ~6 h of a 24 h horizon.
    day2  issued ~48 h before each target time. Strictly earlier than every forecast origin of a 24 h horizon.

Files: Data/nwp/<site>_<model>_<year>.parquet (hourly, UTC), downloaded once on first use, then read offline.
Alignment to the site record (30-min intervals labelled by their start, values = interval means):
    shortwave_radiation is the mean over the PRECEDING hour -> shifted to its interval start, then held for both
    half-hours of that hour; cloud_cover and temperature_2m are instantaneous -> interpolated to interval midpoints.
"""
from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd

from .sites import DATA_DIR, get_site

NWP_DIR = DATA_DIR / "nwp"
NWP_MODEL = "ecmwf_ifs025"            # chosen on the training weeks: see results/nwp_s3.md
NWP_VARIABLES = ("shortwave_radiation", "cloud_cover", "temperature_2m")
NWP_OFFSETS_DAYS = (1, 2)
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
_aligned_cache: dict = {}


def nwp_file(site_name: str, model: str, year: int):
    return NWP_DIR / f"{site_name}_{model}_{year}.parquet"


def download_nwp(site_name: str, year: int, model: str = NWP_MODEL) -> pd.DataFrame:
    """Hourly UTC table with one column per (variable, offset), e.g. shortwave_radiation_previous_day1."""
    site = get_site(site_name)
    columns = [f"{v}_previous_day{n}" for v in NWP_VARIABLES for n in NWP_OFFSETS_DAYS]
    query = (f"{PREVIOUS_RUNS_URL}?latitude={site.latitude}&longitude={site.longitude}"
             f"&start_date={year - 1}-12-31&end_date={year + 1}-01-01&hourly={','.join(columns)}"
             f"&models={model}&timezone=GMT")
    print(f"downloading NWP {model} for {site_name} {year} (one time)")
    with urllib.request.urlopen(query, timeout=120) as response:
        payload = json.load(response)
    hourly = payload["hourly"]
    table = pd.DataFrame({c: pd.to_numeric(pd.Series(hourly[c]), errors="coerce") for c in columns})
    table.index = pd.DatetimeIndex(pd.to_datetime(hourly["time"]), name="time_utc").tz_localize("UTC")
    table.attrs.update(model=model, grid_latitude=payload.get("latitude"), grid_longitude=payload.get("longitude"))
    NWP_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(nwp_file(site_name, model, year))
    return table


def load_nwp(site_name: str, year: int, model: str = NWP_MODEL) -> pd.DataFrame:
    path = nwp_file(site_name, model, year)
    return pd.read_parquet(path) if path.exists() else download_nwp(site_name, year, model)


def align_to_record(hourly: pd.DataFrame, record_index: pd.DatetimeIndex, resolution: str) -> pd.DataFrame:
    """Hourly UTC NWP -> the site record's grid (interval start labels, interval-mean semantics)."""
    step = pd.Timedelta(resolution)
    aligned = {}
    for column in hourly.columns:
        series = hourly[column].tz_convert(record_index.tz)
        if column.startswith("shortwave_radiation"):              # mean over the preceding hour
            interval_start = series.copy()
            interval_start.index = series.index - pd.Timedelta("1h")
            aligned[column] = interval_start.reindex(record_index, method="ffill", limit=int(pd.Timedelta("1h") / step))
        else:                                                      # instantaneous: value at the interval midpoint
            midpoints = record_index + step / 2
            union = series.index.union(midpoints)
            values = series.reindex(union).interpolate("time", limit_area="inside").reindex(midpoints)
            aligned[column] = pd.Series(values.to_numpy(), index=record_index)
    return pd.DataFrame(aligned, index=record_index)


def nwp_column(record: pd.DataFrame, variable: str, offset_days: int = 1, model: str | None = None) -> np.ndarray:
    """One NWP variable on the record's grid. If it cannot be loaded, prints a note and returns zeros (the run goes
    on without weather information instead of failing)."""
    model = model or NWP_MODEL
    site_name, resolution = record.attrs.get("site"), record.attrs.get("resolution", "30min")
    key = (site_name, model, resolution, len(record), record.index[0])
    if key not in _aligned_cache:
        try:
            years = sorted(set(record.index.year))
            hourly = pd.concat([load_nwp(site_name, y, model) for y in years])
            hourly = hourly[~hourly.index.duplicated(keep="first")].sort_index()
            aligned = align_to_record(hourly, record.index, resolution)
            aligned = aligned.interpolate(limit=4, limit_area="inside")    # bridge gaps of up to 2 h
            missing = aligned.isna().mean()
            if missing.max() > 0:
                print(f"note: NWP {model} missing share per column: {missing[missing > 0].round(4).to_dict()}")
            _aligned_cache[key] = aligned
        except Exception as error:  # noqa: BLE001 - no network / no file: continue without NWP, loudly
            print(f"note: NWP {model} unavailable for {site_name} ({type(error).__name__}: {error}) -> zeros")
            _aligned_cache[key] = None
    aligned = _aligned_cache[key]
    column = f"{variable}_previous_day{offset_days}"
    if aligned is None or column not in aligned:
        return np.zeros(len(record))
    return aligned[column].to_numpy()
