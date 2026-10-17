"""The two PV sites (provenance of the processed data: Resources/Journal_*.pdf, Sec. II and Table I).

Timezone note (verified from the data):
  * NIST timestamps are LOCAL CLOCK TIME WITH DST (America/New_York): the mean irradiance peak moves from
    ~12:05 in January to ~12:54 in June. Fixed GMT-5 would shift clear sky by 1 h from March to November.
  * SSP timestamps are fixed GMT+6 (Bangladesh has no DST).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"


@dataclass(frozen=True)
class Site:
    name: str
    latitude: float
    longitude: float
    altitude_m: float
    timezone: str                    # zone used to localise the naive timestamps
    rated_power_kw: float            # AC rating; ac_power feature = P / rated_power_kw
    # site-record column -> (parquet file, column in that file, factor to convert to the column's unit)
    source_columns: dict = field(default_factory=dict)
    ramp_threshold_fraction: float = 0.10   # |dP| > fraction * rated power counts as a ramp


SITES: dict[str, Site] = {
    "nist": Site(
        name="nist", latitude=39.13, longitude=-77.21, altitude_m=138.0, timezone="America/New_York",
        rated_power_kw=260.0,
        source_columns={
            "ac_power_kw": (DATA_DIR / "features_nist_ground_2017/ac_power.parquet", "ac_power", 1e-3),  # W -> kW
            "irradiance_wm2": (DATA_DIR / "features_nist_ground_2017/ground_radiation_wm2.parquet",
                               "ground_radiation_wm2", 1.0),
            "module_temperature_c": (DATA_DIR / "features_nist_ground_2017/pv_module_temperature.parquet",
                                     "pv_module_temperature", 1.0),
        },
        ramp_threshold_fraction=0.15,
    ),
    "ssp": Site(
        name="ssp", latitude=24.39, longitude=89.75, altitude_m=15.0, timezone="Etc/GMT-6", rated_power_kw=2330.0,
        source_columns={
            "ac_power_kw": (DATA_DIR / "features_ssp_2025/dataset.parquet", "ac_power", 1.0),                  # 5-min
            "irradiance_wm2": (DATA_DIR / "features_ssp_2025/dataset.parquet", "roof_radiation_wm2", 1.0),     # 30-min
            "module_temperature_c": (DATA_DIR / "features_ssp_2025/dataset.parquet", "pv_module_temperature", 1.0),
        },
        ramp_threshold_fraction=0.10,
    ),
}


def get_site(name: str) -> Site:
    try:
        return SITES[name.lower()]
    except KeyError as error:
        raise KeyError(f"unknown site {name!r}; known: {list(SITES)}") from error
