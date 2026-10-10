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

    def area_km2(self, res_m=10.0, mask=None):
        """Areas in km2. `mask` (bool, same grid) restricts the count to an outline (see aoi.aoi_mask)."""
        a = res_m ** 2 / 1e6
        m = np.ones(self.fused_water.shape, bool) if mask is None else mask
        return {"otsu": float((self.otsu_water & m).sum() * a), "fused": float((self.fused_water & m).sum() * a),
                "model_only_candidates": float(((self.flags == 2) & m).sum() * a),
                "otsu_only_candidates": float(((self.flags == 3) & m).sum() * a)}


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


def iter_window(site, start, end, water_cfg=None, cache_dir="cache", log=print, skip_days=None):
    """(geobox, generator of (day, vv_lin, vh_lin)) for a site over any date window (Planetary Computer). `skip_days`: days
    ('YYYY-MM-DD') already processed, not read again."""
    cfg = water_cfg or pipeline.WaterConfig()
    return acquire.iter_scenes(site.bbox, site.crs, cfg.resolution_m, start, end, cfg.min_orbit_cover, cache_dir, log, skip_days)


ALL_OUTPUTS = ("otsu_classes", "fused_water", "flags", "prob")


def write_result(out_dir, stamp, res: SceneResult, geobox, which=ALL_OUTPUTS, mask=None):
    """GeoTIFFs <stamp>_<name>.tif for the names in `which`: otsu_classes (uint8, 255 = no data), fused_water (0/1), flags (uint8),
    prob (float32, only if a model was used). `mask` (bool, same grid): everything outside it is written as land / no data."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    inside = np.ones(res.fused_water.shape, bool) if mask is None else mask
    arrays = {
        "otsu_classes": (np.where(inside, res.classes, 255).astype(np.uint8), "uint8", 255),
        "fused_water": ((res.fused_water & inside).astype(np.uint8), "uint8", None),
        "flags": (np.where(inside, res.flags, 0).astype(np.uint8), "uint8", None),
    }
    if res.prob is not None:
        arrays["prob"] = (np.where(inside, res.prob, 0.0).astype(np.float32), "float32", None)
    for name in which:
        if name not in arrays:
            continue
        arr, dtype, nodata = arrays[name]
        p = os.path.join(out_dir, f"{stamp}_{name}.tif")
        pipeline._write_tif(p, arr, geobox, dtype, nodata)
        paths[name] = p
    return paths
