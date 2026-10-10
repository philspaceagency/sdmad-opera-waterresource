"""Nationwide mapping: cover a country with tiles, map every tile, keep an index of the results.

    boundary = national_boundary(vector="philippines.geojson")      # or None: Earth Engine's country boundaries
    tiles    = make_tiles(boundary)                                  # ~20 km tiles, each clipped to the land (+ a 1 km margin)
    tiles, screen = prescreen(tiles, jrc.GeeJrc())                   # optional: drop tiles with (almost) no surface water
    result   = run_nationwide(model, boundary, cfg, ...)             # = prescreen + batch.map_sites over the tiles + index

Each tile is a `Site` (kind "tile") with its own UTM zone, so the Philippines' three zones (50-52N) need no special care, and the land
clip keeps the open sea out of the water areas. Tiles go through `batch.map_sites`, so a nationwide run is resumable and one
failing tile does not stop the others. Per-tile products: `<out>/<tile>/frequency.tif` (% of valid observations with water),
`n_obs.tif`, optional per-date masks, and `<out>/areas.csv`; `<out>/index.geojson` lists every tile with its statistics.

What is NOT produced: one national raster. At 10 m the Philippines is ~3 x 10^9 pixels per layer, so the tiles are the product;
a virtual mosaic per UTM zone is one `gdalbuildvrt` away (or load the folder in QGIS).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

from . import batch, jrc, sites as sites_mod

TILE_DEG = 0.18                 # about 20 km at the Philippines' latitudes
MIN_TILE_LAND_KM2 = 1e-3        # 10 pixels: below this a tile has no land, only rounding noise
LSIB_ASSET = "USDOS/LSIB_SIMPLE/2017"


# ---------------------------------------------------------------------------------------- boundary
def national_boundary(vector=None, country="Philippines", ee_module=None):
    """The land to map, as a shapely (Multi)Polygon in lon/lat.

    vector : a polygon file (country outline, or provinces: all features are merged). Recommended: it is exact and offline.
    None   : Earth Engine's country boundaries (`USDOS/LSIB_SIMPLE/2017`, property `country_na`). This route needs `ee` to be
             initialised and the asset / property names to be current; if it fails, pass `vector`."""
    from shapely.ops import unary_union
    if vector:
        feats = sites_mod._features(vector)
        polys = [g for _, g in feats if g.geom_type in ("Polygon", "MultiPolygon")]
        if not polys:
            raise ValueError(f"{vector}: no polygon features (a country / province outline is needed)")
        return unary_union(polys)
    try:
        import ee
        ee = ee_module or ee
        fc = ee.FeatureCollection(LSIB_ASSET).filter(ee.Filter.eq("country_na", country))
        gj = fc.geometry().getInfo()
        from shapely.geometry import shape
        geom = shape(gj)
        if geom.is_empty:
            raise ValueError(f"no feature with country_na == {country!r}")
        return geom
    except Exception as e:                                              # noqa: BLE001
        raise RuntimeError(f"could not get the {country} boundary from Earth Engine ({e.__class__.__name__}: {e}); "
                           "pass a polygon file with national_boundary(vector=...)") from e


# ---------------------------------------------------------------------------------------- tiles
def make_tiles(boundary, tile_deg=TILE_DEG, margin_km=1.0, min_land_km2=1.0, bbox=None):
    """Tiles of `tile_deg` degrees on a fixed grid (so the same tile always has the same name and box, which is what makes runs
    resumable and comparable) that touch the land. Each tile's `geometry` is the land inside it plus `margin_km` (shorelines and
    river mouths keep their edge pixels); tiles with less than `min_land_km2` of land are dropped. `bbox` restricts the area."""
    from shapely.geometry import box as sbox
    from shapely.prepared import prep
    land = boundary.buffer(margin_km / 111.0)                          # a degree buffer is plenty for a 1 km margin
    pl = prep(land)
    w, s, e, n = bbox if bbox else land.bounds
    i0, i1 = int(math.floor(w / tile_deg)), int(math.floor(e / tile_deg))
    j0, j1 = int(math.floor(s / tile_deg)), int(math.floor(n / tile_deg))
    tiles = []
    for j in range(j0, j1 + 1):
        for i in range(i0, i1 + 1):
            tb = (i * tile_deg, j * tile_deg, (i + 1) * tile_deg, (j + 1) * tile_deg)
            poly = sbox(*tb)
            if not pl.intersects(poly):
                continue
            clip = land.intersection(poly)
            if clip.is_empty:
                continue
            km2 = clip.area * (111.320 * math.cos(math.radians(0.5 * (tb[1] + tb[3])))) * 110.574
            if km2 < max(min_land_km2, MIN_TILE_LAND_KM2):   # also drops the float-noise 'tiles' a land edge on a tile edge leaves
                continue
            tiles.append(sites_mod.Site(f"tile_{j:04d}_{i:04d}", tb, None, "tile", f"land {km2:.0f} km2",
                                        f"tile {j},{i}", clip))
    return tiles


def prescreen(tiles, jrc_source, min_water_frac=0.001, occurrence_min=50.0, res_m=100.0, log=print):
    """Drop the tiles with (almost) no surface water, using JRC occurrence on a coarse grid inside each tile's land outline.
    A tile is kept when at least `min_water_frac` of its land is water for >= `occurrence_min` % of 1984-2021. Cheap (one small
    Earth Engine request per tile) and saves most of the Sentinel-1 reading for the dry hills. Returns (kept tiles, DataFrame of
    every tile's fraction). Narrow streams can be missed: lower `min_water_frac` or skip this step for a complete census."""
    from types import SimpleNamespace
    from affine import Affine
    from . import aoi
    rows, kept = [], []
    for k, t in enumerate(tiles, start=1):
        grid = jrc.grid_for_bbox(t.bbox, t.crs, res_m)
        occ = jrc_source.occurrence(grid)
        gb = SimpleNamespace(transform=Affine(grid.res, 0, grid.x0, 0, -grid.res, grid.y1), crs=grid.crs, shape=(grid.ny, grid.nx))
        inside = aoi.aoi_mask(t, gb)
        inside = np.ones(occ.shape, bool) if inside is None else inside
        frac = float((np.nan_to_num(occ, nan=0.0)[inside] >= occurrence_min).mean()) if inside.any() else 0.0
        rows.append({"site": t.name, "water_frac": frac, "kept": frac >= min_water_frac})
        if frac >= min_water_frac:
            kept.append(t)
        if k % 100 == 0:
            log(f"  prescreen {k}/{len(tiles)}: {len(kept)} kept")
    df = pd.DataFrame(rows)
    log(f"prescreen: {len(kept)} of {len(tiles)} tiles hold surface water (>= {min_water_frac:.2%} of their land)")
    return kept, df


# ---------------------------------------------------------------------------------------- index and totals
def write_index(tiles, result, out_dir, screen=None):
    """`<out>/index.geojson`: one feature per tile (its box) with status, dates processed, mean / max fused water area (km2), last
    date and (if a prescreen ran) the JRC water fraction. Returns the path."""
    a = result.areas
    failed = {f["site"]: f["error"] for f in result.failures}
    frac = dict(zip(screen["site"], screen["water_frac"])) if screen is not None else {}
    feats = []
    for t in tiles:
        r = a[a["site"] == t.name]
        w, s, e, n = t.bbox
        props = {"tile": t.name, "status": "failed" if t.name in failed else ("ok" if len(r) else "no_data"),
                 "n_dates": int(len(r)), "mean_fused_km2": round(float(r["fused_km2"].mean()), 4) if len(r) else None,
                 "max_fused_km2": round(float(r["fused_km2"].max()), 4) if len(r) else None,
                 "last_date": str(r["date"].max()) if len(r) else None, "jrc_water_frac": round(frac[t.name], 5) if t.name in frac else None,
                 "error": failed.get(t.name)}
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}})
    path = os.path.join(out_dir, "index.geojson")
    with open(path, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh)
    return path


def monthly_national_area(areas):
    """National water area per month (km2): each tile's mean over that month's passes, summed over tiles. `n_tiles` says how many
    tiles were observed: a month with few tiles is an undercount, not a drop in water."""
    if areas.empty:
        return pd.DataFrame(columns=["month", "n_tiles", "fused_km2", "otsu_km2"])
    d = areas.assign(month=pd.to_datetime(areas["date"]).dt.to_period("M").astype(str))
    per_tile = d.groupby(["month", "site"])[["fused_km2", "otsu_km2"]].mean().reset_index()
    out = per_tile.groupby("month").agg(n_tiles=("site", "nunique"), fused_km2=("fused_km2", "sum"), otsu_km2=("otsu_km2", "sum"))
    return out.reset_index()


# ---------------------------------------------------------------------------------------- run
def run_nationwide(model, boundary, cfg: batch.MapConfig, tile_deg=TILE_DEG, margin_km=1.0, jrc_source=None, min_water_frac=0.001,
                   bbox=None, limit_tiles=None, log=print):
    """Tile `boundary`, optionally pre-screen with JRC (`jrc_source` = jrc.GeeJrc(); None skips it), map every tile with
    `batch.map_sites`, write the index. Returns (MapResult, tiles mapped, index path).

    `bbox` restricts the run to a lon/lat box (try a region first); `limit_tiles` caps the number of tiles (a quick test)."""
    tiles = make_tiles(boundary, tile_deg, margin_km, bbox=bbox)
    log(f"{len(tiles)} tile(s) of {tile_deg} deg touch the land")
    screen = None
    if jrc_source is not None:
        tiles, screen = prescreen(tiles, jrc_source, min_water_frac, log=log)
    if limit_tiles:
        tiles = tiles[:limit_tiles]
        log(f"limited to the first {len(tiles)} tile(s)")
    os.makedirs(cfg.out_dir, exist_ok=True)
    result = batch.map_sites(tiles, model, cfg, log=log)
    path = write_index(tiles, result, cfg.out_dir, screen)
    log(f"index: {path}")
    return result, tiles, path
