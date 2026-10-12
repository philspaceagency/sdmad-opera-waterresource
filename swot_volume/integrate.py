"""Join SWOT levels with Sentinel-1 water areas and compute level, area and storage volume per date."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import hypsometry as hy

RESULT_COLUMNS = ["date", "wse_m", "wse_u_m", "s1_area_km2", "s1_date", "lag_days", "curve_area_km2", "volume_mcm", "volume_basis"]


def s1_daily_area(areas, site=None, min_valid_frac=0.9):
    """Sentinel-1 `areas.csv` (from map_water.py) -> one water area per date (mean of the passes that day; scenes whose observed area is
    < min_valid_frac of the largest are dropped). Accepts a path or DataFrame."""
    df = pd.read_csv(areas) if isinstance(areas, str) else areas.copy()
    if site is not None:
        df = df[df["site"] == site]
    if df.empty:
        raise ValueError("no Sentinel-1 areas for that site")
    df = df[df["valid_km2"] >= min_valid_frac * df["valid_km2"].max()]
    out = df.groupby("date", as_index=False)["water_km2"].mean().rename(columns={"water_km2": "s1_area_km2"})
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)


def pair(swot, s1, max_lag_days=6):
    """For each SWOT observation the nearest Sentinel-1 date within `max_lag_days` (Sentinel-1 revisit is 6-12 days; a reservoir changes
    slowly). Returns the SWOT rows plus s1_area_km2, s1_date, lag_days; unpaired SWOT rows keep NaN."""
    sw = swot.copy()
    sw["_d"] = pd.to_datetime(sw["date"])
    a = pd.merge_asof(sw.sort_values("_d"), s1.rename(columns={"date": "s1_date"}).assign(_s=lambda d: d["s1_date"]).sort_values("_s"),
                      left_on="_d", right_on="_s", direction="nearest", tolerance=pd.Timedelta(days=max_lag_days))
    a["lag_days"] = (a["s1_date"] - a["_d"]).dt.days
    return a.drop(columns=["_d", "_s"]).reset_index(drop=True)


def compute(swot, s1, max_lag_days=6, method="isotonic", h_ref=None, v_ref=None, curve=None):
    """Level, area and volume per SWOT observation.

    1. pair SWOT levels with Sentinel-1 areas (<= max_lag_days apart);
    2. fit A(h) on the pairs (or use a given `curve`);
    3. volume = v_ref + integral of A dh from h_ref. With no (h_ref, v_ref) the volume is relative to the lowest level of the
       pairs and `volume_basis` says 'relative to lowest observed level'; with them it is absolute ('absolute').
    Returns (table, curve). Every SWOT level gets a volume, also those without a Sentinel-1 date (volume comes from the curve)."""
    p = pair(swot, s1, max_lag_days)
    ok = p.dropna(subset=["s1_area_km2"])
    if curve is None:
        curve = hy.fit_curve(ok["wse_m"], ok["s1_area_km2"], method=method)
    absolute = v_ref is not None
    if absolute and h_ref is None:
        raise ValueError("v_ref needs h_ref, the level at which that volume holds")
    vol = hy.volume_series(curve, p["wse_m"].to_numpy(), h_ref=h_ref, v_ref=v_ref or 0.0)
    p["curve_area_km2"] = curve.area(p["wse_m"].to_numpy())
    p["volume_mcm"] = vol
    p["volume_basis"] = "absolute" if absolute else "relative to lowest observed level"
    return p[RESULT_COLUMNS], curve


def rmse_of_curve(table):
    """RMSE (km2) between the fitted curve area and the Sentinel-1 area on paired dates: a check of the area-elevation fit."""
    t = table.dropna(subset=["s1_area_km2"])
    return float(np.sqrt(np.mean((t["curve_area_km2"] - t["s1_area_km2"]) ** 2))) if len(t) else float("nan")
