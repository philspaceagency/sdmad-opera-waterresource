"""Inference for one Sentinel-1 scene: features -> TerraMind probability -> water mask, and export.

Works for ANY date with an S1 RTC pass, including dates after JRC's coverage (the model needs no JRC at inference). It does need
the site's terrain (DEM, land cover), which does not change and is fetched once (`terrain.site_terrain`).
`infer_scene` is pure NumPy/torch; `iter_window` and `write_result` touch the network / disk.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import acquire, features, scene as sc

FLAG_LAND, FLAG_WATER, FLAG_TERRAIN = 0, 1, 2
MAX_SLOPE_DEG = 25.0


@dataclass
class SceneResult:
    vv_db: np.ndarray
    vh_db: np.ndarray
    sdwi: np.ndarray
    prob: np.ndarray             # TerraMind water probability
    water: np.ndarray            # bool: prob >= threshold, minus the pixels the terrain rule removed
    flags: np.ndarray            # uint8: 0 land, 1 water, 2 model said water but the slope is too steep (removed)
    orbit: str                   # 'ascending' | 'descending'

    @property
    def valid(self):
        return np.isfinite(self.vv_db) & np.isfinite(self.vh_db)

    def area_km2(self, res_m=10.0, mask=None):
        """Areas in km2. `mask` (bool, same grid) restricts the count to an outline (see aoi.aoi_mask)."""
        a = res_m ** 2 / 1e6
        m = np.ones(self.water.shape, bool) if mask is None else mask
        return {"water": float((self.water & m).sum() * a), "terrain_suppressed": float(((self.flags == FLAG_TERRAIN) & m).sum() * a)}


def infer_scene(vv_lin, vh_lin, orbit_state, terrain, model, scene_cfg=None, tile=224, overlap=64, batch=4, device=None,
                threshold=0.5, max_slope_deg=MAX_SLOPE_DEG, use_landcover=True):
    """Features -> probability -> water.

    The model decides. `max_slope_deg` is a last safeguard, not the method: water the model predicts on terrain steeper than this
    (open water is flat; 25 deg is well above the DEM slope at a steep shoreline) is removed and flagged FLAG_TERRAIN so it can be
    reviewed. None disables it. `use_landcover=False` blanks the land-cover channels."""
    f = sc.prepare(vv_lin, vh_lin, scene_cfg)
    x = features.compose(f.vv_db, f.vh_db, f.sdwi, terrain, orbit_state, drop_context=not use_landcover)
    prob = model.predict_prob(x, tile=tile, overlap=overlap, batch=batch, device=device)
    prob = np.where(f.valid, prob, 0.0).astype(np.float32)
    wet = prob >= threshold
    steep = wet & (terrain.slope > max_slope_deg) if max_slope_deg else np.zeros(wet.shape, bool)
    water = wet & ~steep
    flags = np.where(steep, FLAG_TERRAIN, np.where(water, FLAG_WATER, FLAG_LAND)).astype(np.uint8)
    return SceneResult(f.vv_db, f.vh_db, f.sdwi, prob, water, flags, orbit_state)


def load_model(path, device=None):
    """(model, meta, device) from a checkpoint written by training: the complete `terramind_water.pt` loads offline; the small
    `terramind_water_decoder.pt` first rebuilds the pretrained encoder from Hugging Face. device None = GPU if there is one."""
    import torch
    from . import terramind_seg as tm
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: train a model first (notebooks/ai_water_training.ipynb) or point to a checkpoint")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = tm.load_checkpoint(str(path), device=device)
    return model, meta, device


def iter_window(site, start, end, scene_cfg=None, cache_dir="cache", log=print, skip_days=None):
    """(geobox, generator of (day, orbit state, vv_lin, vh_lin)) for a site over any date window (Planetary Computer), ascending
    and descending passes. `skip_days`: keys ('YYYY-MM-DD:orbit') already processed, not read again."""
    cfg = scene_cfg or sc.SceneConfig()
    return acquire.iter_scenes(site.bbox, site.crs, cfg.resolution_m, start, end, cfg.min_orbit_cover, cache_dir, log, skip_days)


ALL_OUTPUTS = ("water", "flags", "prob")


def write_result(out_dir, stamp, res: SceneResult, geobox, which=ALL_OUTPUTS, mask=None):
    """GeoTIFFs <stamp>_<orbit>_<name>.tif for the names in `which`: water (0/1), flags (uint8), prob (float32).
    `mask` (bool, same grid): everything outside it is written as land."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    inside = np.ones(res.water.shape, bool) if mask is None else mask
    arrays = {
        "water": ((res.water & inside).astype(np.uint8), "uint8", None),
        "flags": (np.where(inside, res.flags, 0).astype(np.uint8), "uint8", None),
        "prob": (np.where(inside, res.prob, 0.0).astype(np.float32), "float32", None),
    }
    for name in which:
        if name not in arrays:
            continue
        arr, dtype, nodata = arrays[name]
        p = os.path.join(out_dir, f"{stamp}_{res.orbit}_{name}.tif")
        sc.write_tif(p, arr, geobox, dtype, nodata)
        paths[name] = p
    return paths
