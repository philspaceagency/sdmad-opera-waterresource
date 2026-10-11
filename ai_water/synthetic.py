"""Synthetic Sentinel-1-like scenes for tests and for checking the filters offline.

A lake (smooth irregular shoreline) on land with a few rough/bright patches; backscatter in linear power with
multiplicative gamma speckle of `looks` equivalent looks. Water is dark (~ -20 dB VV), land ~ -8 dB.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def make_scene(shape=(512, 512), water_frac=0.15, looks=4, seed=0, vv_water=-20.0, vv_land=-8.0, vh_offset=-7.0):
    """Returns dict(vv, vh: linear float32; truth: bool water; vv_clean: linear noise-free)."""
    rng = np.random.default_rng(seed)
    h, w = shape
    field = ndi.gaussian_filter(rng.standard_normal(shape), sigma=min(h, w) / 12.0)
    field = (field - field.mean()) / field.std()
    thr = np.quantile(field, water_frac)
    truth = field < thr

    land_tex = ndi.gaussian_filter(rng.standard_normal(shape), 6.0)
    land_tex = 1.5 * (land_tex - land_tex.mean()) / land_tex.std()
    vv_db = np.where(truth, vv_water, vv_land + land_tex)
    # wind-roughened tail on the water (slightly brighter, smooth)
    vv_db = np.where(truth, vv_db + 1.0 * ndi.gaussian_filter(rng.standard_normal(shape), 20.0) * 3, vv_db)
    vv_clean = (10.0 ** (vv_db / 10.0)).astype(np.float32)
    vh_clean = (10.0 ** ((vv_db + vh_offset) / 10.0)).astype(np.float32)

    def speckle(x):
        return (x * rng.gamma(looks, 1.0 / looks, size=x.shape)).astype(np.float32)

    return {"vv": speckle(vv_clean), "vh": speckle(vh_clean), "truth": truth, "vv_clean": vv_clean}


def iou(pred, truth):
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    u = (pred | truth).sum()
    return float((pred & truth).sum() / u) if u else 1.0


def jrc_month_from_truth(truth, cloud=None, block=3):
    """What JRC would show for `truth`: majority vote on block x block cells (Landsat 30 m = 3 S1 pixels at 10 m);
    2 water, 1 not water, 0 no data (`cloud` mask)."""
    from . import jrc
    h, w = truth.shape
    hb, wb = h // block, w // block
    blk = truth[:hb * block, :wb * block].reshape(hb, block, wb, block).mean((1, 3)) >= 0.5
    out = np.full(truth.shape, jrc.NOT_WATER, np.uint8)
    out[:hb * block, :wb * block] = np.where(np.kron(blk, np.ones((block, block), bool)), jrc.WATER, jrc.NOT_WATER)
    if cloud is not None:
        out[cloud] = jrc.NODATA
    return out


class SyntheticJrc:
    """Offline stand-in for jrc.GeeJrc: `truth_by_month` = {"2019-05": bool water mask}. Months not listed are no data."""

    def __init__(self, truth_by_month, cloud=None):
        self.truth_by_month, self.cloud, self.calls = truth_by_month, cloud, []

    def month(self, month, grid):
        import pandas as pd
        key = pd.Timestamp(month).strftime("%Y-%m")
        self.calls.append(key)
        t = self.truth_by_month.get(key)
        return np.zeros((grid.ny, grid.nx), np.uint8) if t is None else jrc_month_from_truth(t, self.cloud)

    def occurrence(self, grid):
        return np.zeros((grid.ny, grid.nx), np.float32)


def make_terrain_scene(shape=(384, 384), seed=0, res_m=10.0, orbit="descending", looks=4, slope_deg=30.0):
    """A reservoir in a valley with a hill on each side, seen from `orbit`. Returns dict(vv, vh linear, truth bool water, dem (m)).

    The flank that faces away from the sensor is in radar shadow: as dark as water (-22 dB), which is the false-water trap.
    The flank that faces the sensor is bright (foreshortening). Descending passes look west (the EAST flank is in shadow),
    ascending passes look east (the WEST flank is)."""
    rng = np.random.default_rng(seed)
    h, w = shape
    xc = w // 2
    half = 30 + ndi.gaussian_filter(rng.standard_normal(h), 12.0) * 40          # lake half-width per row, wobbly shore
    half = np.clip(half, 20, 60)
    cols = np.arange(w)[None, :]
    dist = np.abs(cols - xc) - half[:, None]                                    # < 0 inside the lake
    truth = dist < 0
    dem = 100.0 + np.clip(dist, 0, None) * res_m * np.tan(np.radians(slope_deg))
    dem = ndi.gaussian_filter(dem, 1.0).astype(np.float32)
    east = (cols > xc) & ~truth
    shadow_side = east if orbit == "descending" else (~east & ~truth & (cols < xc))
    steep = np.clip(dist, 0, None) > 3
    land_tex = ndi.gaussian_filter(rng.standard_normal(shape), 4.0)
    land_tex = 1.2 * (land_tex - land_tex.mean()) / land_tex.std()
    vv_db = np.where(truth, -20.0, -8.0 + land_tex)
    vv_db = np.where(shadow_side & steep, -22.0 + 0.5 * land_tex, vv_db)
    vv_db = np.where(~shadow_side & ~truth & steep, -5.0 + 0.5 * land_tex, vv_db)
    vv = 10.0 ** (vv_db / 10.0)
    vh = 10.0 ** ((vv_db - 7.0) / 10.0)

    def speckle(x):
        return (x * rng.gamma(looks, 1.0 / looks, size=x.shape)).astype(np.float32)
    return {"vv": speckle(vv), "vh": speckle(vh), "truth": truth, "dem": dem, "shadow": shadow_side & steep}


class SyntheticTerrain:
    """Offline stand-in for terrain.GeeTerrain: serves a fixed DEM (and optional WorldCover raster) for any grid of that shape."""

    def __init__(self, dem, worldcover=None):
        self._dem = dem
        self._wc = worldcover if worldcover is not None else np.full(dem.shape, 10, np.uint8)
        self.calls = 0

    def dem(self, grid):
        self.calls += 1
        return self._dem.copy()

    def landcover(self, grid):
        return self._wc.copy()
