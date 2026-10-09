"""Per-scene Otsu water mapping: linear VV/VH -> refined Lee -> dB -> edge+binary Otsu -> classes.

`map_water` is pure NumPy (no network). `run_area` ties it to `acquire` and writes one GeoTIFF per date plus a
thresholds table, which doubles as the pseudo-label set for the TerraMind stage (`terramind_seg`).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from . import acquire, otsu, preprocess as pp, speckle


@dataclass
class WaterConfig:
    bbox: tuple = acquire.ANGAT_BBOX
    crs: str = acquire.ANGAT_CRS
    resolution_m: float = 10.0
    start: str = "2024-01-01"
    end: str = "2024-12-31"
    min_orbit_cover: float = 0.9
    # speckle
    speckle_filter: str = "refined_lee"          # 'refined_lee' | 'lee' | 'none'
    enl: float = None                             # None = estimated from the scene
    # thresholding
    band: str = "vv"                              # water is detected on 'vv' or 'vh'
    tile_px: int = 256
    max_gap_db: float = 3.0
    rule: str = "hysteresis"                      # 'hysteresis' | 'mean'
    min_area_px: int = 200                        # 2 ha at 10 m
    fill_holes_px: int = 200
    edge_kw: dict = field(default_factory=dict)   # sigma, low_q, high_q, buffer_px
    out_dir: str = "outputs/angat"
    cache_dir: str = "cache"


@dataclass
class WaterMap:
    classes: np.ndarray          # uint8: 0 land, 1 water, 2 uncertain, 255 nodata
    water: np.ndarray            # bool, cleaned
    db: np.ndarray               # filtered dB of the detection band
    thresholds: otsu.Thresholds
    vh_db: np.ndarray = None     # filtered VH dB (model input) when VH was provided

    def summary(self, pixel_area_m2=100.0):
        th = self.thresholds
        return {"mode": th.mode, "t_edge": _f(th.t_edge), "t_binary": _f(th.t_binary), "t_low": _f(th.t_low),
                "t_high": _f(th.t_high), "gap_db": _f(th.gap_db), "n_tiles": th.n_tiles,
                "water_px": int(self.water.sum()), "water_km2": float(self.water.sum() * pixel_area_m2 / 1e6),
                "uncertain_px": int((self.classes == otsu.CLASS_UNCERTAIN).sum())}


def _f(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


def map_water(vv, vh=None, cfg: WaterConfig = None) -> WaterMap:
    """Water map of one scene. `vv`, `vh`: linear power (h, w); `vh` only needed when cfg.band == 'vh'."""
    cfg = cfg or WaterConfig()
    band = vv if cfg.band == "vv" else vh
    if band is None:
        raise ValueError(f"band {cfg.band!r} requested but not provided")
    filt = speckle.apply_filter(band, cfg.speckle_filter, **({"enl": cfg.enl} if cfg.speckle_filter != "none" else {}))
    db = pp.to_db(filt)
    valid = np.isfinite(db)
    th = otsu.combined_otsu(db, valid, tile=cfg.tile_px, max_gap_db=cfg.max_gap_db, edge_kw=dict(cfg.edge_kw))
    classes = otsu.classify(db, th, valid, rule=cfg.rule)
    water = otsu.clean_mask(classes == otsu.CLASS_WATER, cfg.min_area_px, cfg.fill_holes_px)
    classes = np.where((classes == otsu.CLASS_WATER) & ~water, otsu.CLASS_LAND, classes).astype(np.uint8)
    classes[water] = otsu.CLASS_WATER
    vh_db = pp.to_db(speckle.apply_filter(vh, cfg.speckle_filter, **({"enl": cfg.enl} if cfg.speckle_filter != "none" else {}))) \
        if vh is not None else None
    return WaterMap(classes, water, db, th, vh_db)


# ---------------------------------------------------------------------------------------- full run
def _write_tif(path, arr, geobox, dtype, nodata=None):
    import rasterio
    from rasterio.transform import Affine
    h, w = arr.shape
    t = geobox.transform
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1, dtype=dtype, crs=str(geobox.crs),
                       transform=Affine(t.a, t.b, t.c, t.d, t.e, t.f), compress="deflate", nodata=nodata,
                       tiled=True, blockxsize=256, blockysize=256) as dst:
        dst.write(arr.astype(dtype), 1)


def run_area(cfg: WaterConfig, log=print):
    """Search -> one orbit -> read -> per-date Otsu water map. Writes <out_dir>/<date>_classes.tif, <date>_db.tif
    (filtered VV dB), <date>_vhdb.tif and thresholds.json. Returns the list of per-date summaries."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    acquire.set_stac_cache(cfg.cache_dir)
    items = acquire.search_s1(cfg.bbox, cfg.start, cfg.end)
    key, items = acquire.choose_orbit(items, cfg.bbox, cfg.min_orbit_cover)
    if not items:
        raise RuntimeError("no Sentinel-1 RTC scene fully covers the area in the requested window")
    log(f"orbit {key}: {len(items)} items on {len(acquire.group_by_day(items))} days")
    gb = acquire.make_geobox(cfg.bbox, cfg.crs, cfg.resolution_m)
    rows = []
    lon = 0.5 * (cfg.bbox[0] + cfg.bbox[2])
    by_day = acquire.group_by_day(items)
    for day in sorted(by_day):
        loaded = acquire.load_s1(by_day[day], gb, "nearest" if cfg.resolution_m <= 10 else "average", log=log)
        if loaded is None:
            continue
        times, vv, vh = loaded
        for k in range(len(times)):
            wm = map_water(vv[k], vh[k], cfg)
            stamp = str(acquire.solar_dates([times[k]], lon)[0])
            _write_tif(os.path.join(cfg.out_dir, f"{stamp}_classes.tif"), wm.classes, gb, "uint8", nodata=255)
            _write_tif(os.path.join(cfg.out_dir, f"{stamp}_db.tif"), wm.db, gb, "float32", nodata=float("nan"))
            _write_tif(os.path.join(cfg.out_dir, f"{stamp}_vhdb.tif"), wm.vh_db, gb, "float32", nodata=float("nan"))
            row = {"date": stamp, **wm.summary(cfg.resolution_m ** 2)}
            rows.append(row)
            log(f"  {stamp}: {row['mode']:<11} t_edge={row['t_edge']} t_bin={row['t_binary']} water={row['water_km2']:.2f} km2")
    with open(os.path.join(cfg.out_dir, "thresholds.json"), "w") as fh:
        json.dump({"config": {k: v for k, v in asdict(cfg).items()}, "scenes": rows}, fh, indent=2, default=str)
    return rows
