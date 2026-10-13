"""SWOT lake heights over a reservoir (PO.DAAC product SWOT_L2_HR_LakeSP).

The product is a set of shapefiles per 'tile' of a swath: `Obs` (lakes observed, matched to the prior lake database), `Prior` (the
same lakes, whole-lake) and `Unassigned`. For a reservoir the useful fields of `Obs` / `Prior` are

    wse, wse_u      water surface elevation and its uncertainty (m, above the geoid EGM2008)
    area_total      lake area (m2) including the part SWOT could not see (dark water, layover) estimated
    area_detct      detected area (m2)
    partial_f       1 if the lake is only partly inside the swath
    xovr_cal_q      crossover calibration quality (0 good ... 3 bad)
    quality_f       overall lake quality (0 good, 1 suspect, 2 degraded, 3 bad)
    dark_frac       fraction of the lake that is dark water
    ice_clim_f / ice_dyn_f   ice flags (irrelevant in the Philippines)
    time_str        UTC observation time, 'time' seconds since 2000-01-01

Field names follow the product description in the PO.DAAC SWOT tutorial; if a release renames one, `NAMES` is the one place to change it.
The network parts need `earthaccess` (free NASA Earthdata login: `earthaccess.login()` or EARTHDATA_USERNAME / EARTHDATA_PASSWORD) and
`geopandas`; `observations_from_frames` needs neither the network nor earthaccess, so the rest of the chain is testable offline.
"""
from __future__ import annotations

import glob
import os
import re
import zipfile

import numpy as np
import pandas as pd

SHORT_NAME = "SWOT_L2_HR_LakeSP_2.0"
NAMES = {"wse": "wse", "wse_u": "wse_u", "area": "area_total", "area_detct": "area_detct", "partial": "partial_f",
         "xovr_q": "xovr_cal_q", "quality": "quality_f", "dark": "dark_frac", "time": "time_str", "lake_id": "lake_id"}
OUT_COLUMNS = ["date", "time_utc", "lake_id", "wse_m", "wse_u_m", "swot_area_km2", "partial", "quality", "xovr_q", "dark_frac"]


CONTINENTS = {"AF": "Africa", "EU": "Europe and Middle East", "SI": "Siberia", "AS": "Central and Southeast Asia", "AU": "Australia and Oceania",
              "SA": "South America", "NA": "North America and Caribbean", "AR": "North American Arctic", "GR": "Greenland"}
PHILIPPINES_CONTINENT = "AS"
# LakeSP / RiverSP granules are continent-level swaths named ..._<Obs|Prior|Unassigned>_<cycle>_<PPP pass 001-584>_<CC continent>_<start>_<end>_...
_GRANULE = re.compile(r"_(Obs|Prior|Unassigned)_(\d{3})_(\d{3})_([A-Z]{2})(?:_|\.)")


def parse_granule_name(name):
    """'SWOT_L2_HR_LakeSP_Obs_010_013_AS_2024...zip' -> {'kind': 'Obs', 'cycle': 10, 'pass': 13, 'continent': 'AS'}, or None if it does not match."""
    m = _GRANULE.search(os.path.basename(str(name)))
    return None if not m else {"kind": m.group(1), "cycle": int(m.group(2)), "pass": int(m.group(3)), "continent": m.group(4)}


def wanted(name, kind="Obs", continent=PHILIPPINES_CONTINENT, passes=None):
    """True if a granule file name is of `kind`, on `continent` (None = any) and, when given, one of the pass numbers `passes`."""
    g = parse_granule_name(name)
    return bool(g) and g["kind"] == kind and (continent is None or g["continent"] == continent) and (not passes or g["pass"] in set(passes))


def search(bbox, start, end, short_name=SHORT_NAME):
    """Granule results over bbox=(west, south, east, north) and [start, end] (ISO dates). Needs earthaccess."""
    import earthaccess
    earthaccess.login()
    return earthaccess.search_data(short_name=short_name, bounding_box=tuple(bbox), temporal=(start, end))


def download(results, out_dir, kind="Obs", continent=PHILIPPINES_CONTINENT, passes=None):
    """Download the granules (only the `kind` shapefile zips: 'Obs', 'Prior' or 'Unassigned', on `continent` 'AS' = Central and Southeast
    Asia, which holds the Philippines; None = any; optionally only the pass numbers in `passes`) and unzip them. Returns the .shp paths."""
    import earthaccess
    os.makedirs(out_dir, exist_ok=True)
    links = [l for r in results for l in r.data_links() if wanted(l, kind, continent, passes)]
    files = earthaccess.download(links, out_dir) if links else []
    shps = []
    for f in files:
        f = str(f)
        if f.endswith(".zip"):
            d = f[:-4]
            if not os.path.isdir(d):
                with zipfile.ZipFile(f) as z:
                    z.extractall(d)
            shps += glob.glob(os.path.join(d, "*.shp"))
        elif f.endswith(".shp"):
            shps.append(f)
    return sorted(shps)


def read_shapefiles(paths, reservoir=None, bbox=None):
    """Read the shapefiles into one GeoDataFrame, keeping lakes that intersect `reservoir` (a shapely geometry in lon/lat) or `bbox`."""
    import geopandas as gpd
    frames = []
    for p in paths:
        g = gpd.read_file(p)
        if len(g):
            g = g.to_crs("EPSG:4326") if g.crs is not None else g.set_crs("EPSG:4326")
            frames.append(g)
    if not frames:
        return gpd.GeoDataFrame()
    g = pd.concat(frames, ignore_index=True)
    if reservoir is not None:
        g = g[g.intersects(reservoir)]
    elif bbox is not None:
        from shapely.geometry import box
        g = g[g.intersects(box(*bbox))]
    return g


def observations_from_frames(frames, max_quality=1, max_xovr=1, allow_partial=False, min_area_km2=None):
    """One row per SWOT lake observation of the reservoir, filtered on quality.

    `frames`: a DataFrame / GeoDataFrame (or list of them) with the product fields. When several polygons of one pass belong to the
    reservoir (a reservoir split across tiles), they are combined: area summed, level the area-weighted mean. Returns OUT_COLUMNS,
    sorted by time. `max_quality` / `max_xovr` keep flags <= the value (0 = only 'good'); partial lakes are dropped unless `allow_partial`.
    """
    n = NAMES
    df = pd.concat(frames, ignore_index=True) if isinstance(frames, (list, tuple)) else frames.copy()
    if df.empty:
        return pd.DataFrame(columns=OUT_COLUMNS)
    t = pd.to_datetime(df[n["time"]], errors="coerce", utc=True)
    df = df.assign(_t=t)
    df = df[np.isfinite(pd.to_numeric(df[n["wse"]], errors="coerce")) & df["_t"].notna()]          # fill values are -1e12 etc.
    df = df[pd.to_numeric(df[n["wse"]]) > -1e6]
    if n["quality"] in df:
        df = df[pd.to_numeric(df[n["quality"]], errors="coerce") <= max_quality]
    if n["xovr_q"] in df:
        df = df[pd.to_numeric(df[n["xovr_q"]], errors="coerce").fillna(0) <= max_xovr]
    if n["partial"] in df and not allow_partial:
        df = df[pd.to_numeric(df[n["partial"]], errors="coerce").fillna(0) == 0]
    rows = []
    for tt, g in df.groupby(df["_t"].dt.floor("min")):
        a = pd.to_numeric(g[n["area"]], errors="coerce").clip(lower=0).fillna(0).to_numpy() / 1e6
        w = a if a.sum() > 0 else np.ones(len(g))
        h = pd.to_numeric(g[n["wse"]]).to_numpy()
        u = pd.to_numeric(g[n["wse_u"]], errors="coerce").to_numpy() if n["wse_u"] in g else np.full(len(g), np.nan)
        rows.append({"date": tt.strftime("%Y-%m-%d"), "time_utc": tt.isoformat(), "lake_id": ",".join(sorted(set(g[n["lake_id"]].astype(str)))) if n["lake_id"] in g else "",
                     "wse_m": float(np.average(h, weights=w)), "wse_u_m": float(np.sqrt(np.nansum((w / w.sum() * u) ** 2))) if np.isfinite(u).any() else np.nan,
                     "swot_area_km2": float(a.sum()), "partial": int(pd.to_numeric(g[n["partial"]], errors="coerce").fillna(0).max()) if n["partial"] in g else 0,
                     "quality": int(pd.to_numeric(g[n["quality"]], errors="coerce").max()) if n["quality"] in g else 0,
                     "xovr_q": int(pd.to_numeric(g[n["xovr_q"]], errors="coerce").max()) if n["xovr_q"] in g else 0,
                     "dark_frac": float(pd.to_numeric(g[n["dark"]], errors="coerce").mean()) if n["dark"] in g else np.nan})
    out = pd.DataFrame(rows, columns=OUT_COLUMNS).sort_values("time_utc").reset_index(drop=True)
    if min_area_km2:
        out = out[out["swot_area_km2"] >= min_area_km2].reset_index(drop=True)
    return out
