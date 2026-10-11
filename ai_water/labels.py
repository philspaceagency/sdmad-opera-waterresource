"""Training labels from the date-matched JRC monthly map, corrected with terrain and checked against the radar.

JRC (Landsat, 30 m, one map per calendar month) is independent of the radar, so radar shadow and wet soil cannot leak into the
labels the way they did when a Sentinel-1 threshold also had a say. JRC supplies the label; the radar is only used to audit it.

  JRC month (this S1 date's month)                label   source
  water                                            1       JRC_WATER
  not water                                        0       JRC_LAND
  not water, but dark in this pass (VV < dark_db)  0       HARD_NEGATIVE  (shadow / wet soil / smooth bare ground: exactly
                                                                           what the model must learn to call land)
  water on a slope > max_water_slope               ignore  TERRAIN        (open water is flat: a JRC error, e.g. Landsat
                                                                           topographic shadow)
  within `edge_px` of the JRC shoreline            ignore  (JRC is 30 m: its shoreline is uncertain by about a Landsat
                                                           pixel; the model sharpens it from the 10 m radar)
  JRC no data / radar no data                      ignore

Scenes whose JRC month does not fit the pass (reservoir at another level, Landsat gaps) show little radar contrast between
the JRC-water and JRC-land pixels and are rejected by `scene_quality`.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import jrc
from .features import IGNORE

SRC_NONE, SRC_JRC_WATER, SRC_JRC_LAND, SRC_HARD_NEGATIVE, SRC_TERRAIN = 0, 1, 2, 3, 4
SOURCE_NAMES = {SRC_NONE: "none", SRC_JRC_WATER: "jrc_water", SRC_JRC_LAND: "jrc_land", SRC_HARD_NEGATIVE: "hard_negative",
                SRC_TERRAIN: "terrain_ignored"}


def _disk(r):
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def jrc_labels(jrc_class, vv_db, slope, edge_px=3, max_water_slope=15.0, dark_db=-18.0):
    """Returns (labels uint8 [0 land, 1 water, 255 ignore], source uint8, stats dict).

    jrc_class : JRC monthly water class (band `water`) on the S1 grid (0 no data, 1 not water, 2 water)
    vv_db     : filtered VV in dB of this scene (NaN = no data)
    slope     : terrain slope in degrees on the same grid
    edge_px   : width (S1 pixels) of the ignored band around the JRC shoreline; 3 px = 30 m = one Landsat pixel
    """
    if not (jrc_class.shape == vv_db.shape == slope.shape):
        raise ValueError(f"grid mismatch: JRC {jrc_class.shape}, radar {vv_db.shape}, terrain {slope.shape}")
    jw, jl = jrc_class == jrc.WATER, jrc_class == jrc.NOT_WATER
    seen = np.isfinite(vv_db)
    if edge_px > 0:
        k = _disk(int(edge_px))
        band = ndi.binary_dilation(jw, k) & ~ndi.binary_erosion(jw, k, border_value=1)
    else:
        band = np.zeros(jw.shape, bool)
    core = ~band & seen

    lab = np.full(jrc_class.shape, IGNORE, np.uint8)
    src = np.zeros(jrc_class.shape, np.uint8)
    steep = core & jw & (slope > max_water_slope)
    water = core & jw & ~steep
    land = core & jl
    hard = land & (vv_db < dark_db)
    lab[water], src[water] = 1, SRC_JRC_WATER
    lab[land], src[land] = 0, SRC_JRC_LAND
    src[hard] = SRC_HARD_NEGATIVE
    src[steep] = SRC_TERRAIN

    def med(m):
        return float(np.median(vv_db[m])) if m.any() else float("nan")

    stats = {
        "labelled_px": int((lab != IGNORE).sum()),
        "water_px": int(water.sum()),
        "land_px": int(land.sum()),
        "hard_negative_px": int(hard.sum()),
        "terrain_ignored_px": int(steep.sum()),
        "jrc_nodata_frac": float((jrc_class == jrc.NODATA).mean()),
        # how much darker the JRC-water pixels are than the ordinary JRC-land pixels (dark land, i.e. hard negatives, left out so
        # that a hillside in shadow does not hide the contrast) in this pass, dB: small = the JRC month does not fit
        "contrast_db": med(land & ~hard) - med(water),
    }
    return lab, src, stats


def scene_quality(stats, min_contrast_db=5.0, min_water_px=2000, max_nodata=0.4):
    """Is this scene usable for training? Needs a real water body, enough JRC coverage and radar contrast between JRC water and
    JRC land. Low contrast usually means a JRC month that does not match the S1 date (the reservoir moved) or a wind-roughened
    surface that no label could be learned from."""
    c = stats["contrast_db"]
    return bool(stats["water_px"] >= min_water_px and stats["jrc_nodata_frac"] <= max_nodata
                and np.isfinite(c) and c >= min_contrast_db)
