"""Training labels from a date-matched JRC month AND the Sentinel-1 Otsu map of the same date.

Neither source is trusted alone. JRC (Landsat, 30 m, monthly) is independent of SAR but coarse in space and time;
Otsu is 10 m and same-day but carries SAR-specific errors (radar shadow, wind-roughened water). Labels are given
where the two are consistent, and the source of every labelled pixel is recorded so it can be audited.

  JRC month (this S1 date's month)   Otsu (this date)        label   source
  water                              water                   1       AGREE
  not water                          land                    0       AGREE
  water / not water                  uncertain band          1 / 0   JRC_RESOLVED  (JRC decides the band)
  water                              land                    ignore  (conflict: counted)
  not water                          water                   ignore  (conflict: counted) -- or 0 / NEVER_WATER
                                                             when `never_water_negatives` and occurrence == 0
  within `edge_px` of the JRC shoreline   Otsu certain       Otsu    OTSU_EDGE  (JRC is 30 m: its shoreline is
                                                                      uncertain by about a Landsat pixel, so the
                                                                      10 m Otsu edge decides there)
  JRC no data                        any                     ignore

`never_water_negatives` is OFF by default: pixels never seen as water in 38 years but wet in S1 include radar
shadow (should be land) and flooded paddies / new floods (are water). Turn it on only where shadow is the problem.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import jrc, otsu
from .pseudo_labels import IGNORE

SRC_NONE, SRC_AGREE, SRC_JRC_RESOLVED, SRC_OTSU_EDGE, SRC_NEVER_WATER = 0, 1, 2, 3, 4
SOURCE_NAMES = {SRC_NONE: "none", SRC_AGREE: "agree", SRC_JRC_RESOLVED: "jrc_resolved", SRC_OTSU_EDGE: "otsu_edge",
                SRC_NEVER_WATER: "never_water"}


def _disk(r):
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def jrc_otsu_labels(classes, jrc_class, occurrence=None, edge_px=3, resolve_uncertain=True, edge_policy="otsu",
                    never_water_negatives=False):
    """Returns (labels uint8 [0 land, 1 water, 255 ignore], source uint8, stats dict).

    classes   : Otsu classes (otsu.CLASS_*), same grid
    jrc_class : JRC monthly waterClass on the same grid (0 no data, 1 not water, 2 water)
    occurrence: JRC occurrence 0-100 (NaN = unobserved), only used by `never_water_negatives`
    edge_px   : width (S1 pixels) of the ignore/Otsu band around the JRC shoreline; 3 px = 30 m = one Landsat pixel
    """
    if classes.shape != jrc_class.shape:
        raise ValueError(f"grid mismatch: Otsu {classes.shape} vs JRC {jrc_class.shape}")
    if edge_policy not in ("otsu", "ignore"):
        raise ValueError("edge_policy must be 'otsu' or 'ignore'")
    jw, jl = jrc_class == jrc.WATER, jrc_class == jrc.NOT_WATER
    ow, ol, ou = (classes == otsu.CLASS_WATER), (classes == otsu.CLASS_LAND), (classes == otsu.CLASS_UNCERTAIN)

    if edge_px > 0:
        k = _disk(int(edge_px))
        band = ndi.binary_dilation(jw, k) & ~ndi.binary_erosion(jw, k, border_value=1)
    else:
        band = np.zeros(jw.shape, bool)
    core = ~band

    lab = np.full(classes.shape, IGNORE, np.uint8)
    src = np.zeros(classes.shape, np.uint8)

    def put(mask, value, source):
        lab[mask] = value
        src[mask] = source

    put(core & jw & ow, 1, SRC_AGREE)
    put(core & jl & ol, 0, SRC_AGREE)
    if resolve_uncertain:
        put(core & jw & ou, 1, SRC_JRC_RESOLVED)
        put(core & jl & ou, 0, SRC_JRC_RESOLVED)
    if edge_policy == "otsu":
        put(band & (jw | jl) & ow, 1, SRC_OTSU_EDGE)
        put(band & (jw | jl) & ol, 0, SRC_OTSU_EDGE)
    c_water_vs_land = core & jw & ol          # JRC water, Otsu land
    c_land_vs_water = core & jl & ow          # JRC not water, Otsu water
    if never_water_negatives and occurrence is not None:
        never = c_land_vs_water & (np.nan_to_num(occurrence, nan=-1) == 0)
        put(never, 0, SRC_NEVER_WATER)
    else:
        never = np.zeros(jw.shape, bool)

    n_agree = int((src == SRC_AGREE).sum())
    n_conf = int(c_water_vs_land.sum() + (c_land_vs_water & ~never).sum())
    stats = {
        "labelled_px": int((lab != IGNORE).sum()),
        "water_px": int((lab == 1).sum()),
        "agree_px": n_agree,
        "jrc_resolved_px": int((src == SRC_JRC_RESOLVED).sum()),
        "otsu_edge_px": int((src == SRC_OTSU_EDGE).sum()),
        "never_water_px": int(never.sum()),
        "conflict_jrc_water_otsu_land_px": int(c_water_vs_land.sum()),
        "conflict_jrc_land_otsu_water_px": int((c_land_vs_water & ~never).sum()),
        "jrc_nodata_frac": float((jrc_class == jrc.NODATA).mean()),
        # share of the cross-checked pixels (both sources decided) on which they agree
        "agreement": float(n_agree / max(1, n_agree + n_conf)),
    }
    return lab, src, stats


def scene_quality(stats, min_agreement=0.9, min_water_px=2000, max_nodata=0.4):
    """Is this scene usable for training? Needs a real water body, enough JRC coverage and JRC/Otsu consistency.
    Low agreement usually means a JRC month that does not match the S1 date (reservoir moved) or a bad Otsu scene."""
    return bool(stats["water_px"] >= min_water_px and stats["jrc_nodata_frac"] <= max_nodata
                and stats["agreement"] >= min_agreement)
