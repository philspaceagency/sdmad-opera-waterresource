"""Inference for one Sentinel-1 scene: Otsu water map + TerraMind probability + fusion, and export.

Works for ANY date with an S1 RTC pass, including dates after JRC's coverage (the model needs no JRC at inference).
`infer_scene` is pure NumPy/torch; `iter_window` and `write_result` touch the network / disk.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import acquire, otsu, pipeline


@dataclass
class SceneResult:
    vv_db: np.ndarray
    vh_db: np.ndarray
    classes: np.ndarray          # Otsu classes (otsu.CLASS_*)
    otsu_water: np.ndarray       # bool, Otsu water after cleaning
    prob: np.ndarray             # TerraMind water probability (None if no model)
    fused_water: np.ndarray      # bool
    flags: np.ndarray            # uint8: 0 land, 1 water, 2 model-only, 3 Otsu-only (terramind_seg.FUSE_*)
    thresholds: otsu.Thresholds

    def area_km2(self, res_m=10.0):
        a = res_m ** 2 / 1e6
        return {"otsu": float(self.otsu_water.sum() * a), "fused": float(self.fused_water.sum() * a),
                "model_only_candidates": float((self.flags == 2).sum() * a),
                "otsu_only_candidates": float((self.flags == 3).sum() * a)}


def infer_scene(vv_lin, vh_lin, model=None, water_cfg=None, tile=224, overlap=64, batch=4, device=None,
                sure_conflict=0.9, accept=0.5):
    """Otsu on the scene, then (if `model` is given) TerraMind probability and the fusion of both.
    Without a model `fused_water` is the Otsu water."""
    cfg = water_cfg or pipeline.WaterConfig()
    wm = pipeline.map_water(vv_lin, vh_lin, cfg)
    if model is None:
        flags = wm.water.astype(np.uint8)
        return SceneResult(wm.db, wm.vh_db, wm.classes, wm.water, None, wm.water, flags, wm.thresholds)
    from . import terramind_seg as tm
    prob = tm.predict_scene(model, wm.db, wm.vh_db, tile=tile, overlap=overlap, batch=batch, device=device)
    fused, flags = tm.fuse(prob, wm.classes, sure_conflict=sure_conflict, accept=accept)
    return SceneResult(wm.db, wm.vh_db, wm.classes, wm.water, prob, fused, flags, wm.thresholds)


def iter_window(site, start, end, water_cfg=None, cache_dir="cache", log=print):
    """(geobox, generator of (day, vv_lin, vh_lin)) for a site over any date window (Planetary Computer)."""
    cfg = water_cfg or pipeline.WaterConfig()
    return acquire.iter_scenes(site.bbox, site.crs, cfg.resolution_m, start, end, cfg.min_orbit_cover, cache_dir, log)


def write_result(out_dir, stamp, res: SceneResult, geobox):
    """GeoTIFFs: <stamp>_otsu_classes, _fused_water (uint8), _flags (uint8), _prob (float32, if a model was used)."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    def w(name, arr, dtype, nodata=None):
        p = os.path.join(out_dir, f"{stamp}_{name}.tif")
        pipeline._write_tif(p, arr, geobox, dtype, nodata)
        paths[name] = p
    w("otsu_classes", res.classes, "uint8", 255)
    w("fused_water", res.fused_water.astype(np.uint8), "uint8")
    w("flags", res.flags, "uint8")
    if res.prob is not None:
        w("prob", res.prob, "float32")
    return paths
