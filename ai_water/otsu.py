"""Otsu thresholding for SAR water mapping: binary-Otsu, edge-Otsu and their combination.

All functions take backscatter in dB (use `preprocess.to_db` after speckle filtering). Water is the LOW tail.

  * `binary_otsu`        - Otsu on the whole valid histogram. Fails (threshold lands inside the land mode)
                           when water is a tiny part of the scene, i.e. when the histogram is not bimodal.
  * `tiled_binary_otsu`  - binary-Otsu per tile, keeping only tiles that are clearly bimodal, then the median of
                           the tile thresholds. Robust to scenes that are mostly land.
  * `edge_otsu`          - Donchyts et al. (2016): Canny edges -> buffer -> Otsu on the pixels inside the
                           buffer. The buffer holds roughly half water / half land pixels along every
                           shoreline, so the histogram is bimodal even when the lake is small.
  * `combined_otsu`      - uses both, and reports how far they agree. Where they disagree, pixels between the
                           two thresholds are 'uncertain' and are only accepted as water when connected to a
                           pixel both methods call water (hysteresis).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

DB_RANGE = (-30.0, 5.0)
MIN_CONTRAST_DB = 4.0     # open water is >= ~5 dB darker than land in VV; a split closer than this is texture, not water


# ---------------------------------------------------------------------------------------- Otsu core
@dataclass
class OtsuResult:
    threshold: float
    bcv: float              # between-class variance / total variance, 0..1 (higher = more bimodal)
    ashman_d: float         # Ashman's D of the two classes; > 2 means cleanly separable
    water_fraction: float   # share of the pixels below the threshold
    n: int
    contrast_db: float = 0.0   # mean(land class) - mean(water class), dB

    @property
    def bimodal(self):
        return self.ashman_d >= 2.0 and self.bcv >= 0.5

    def valid(self, min_contrast_db=MIN_CONTRAST_DB):
        """Finite, bimodal, and the two classes differ by a physically plausible backscatter contrast."""
        return bool(np.isfinite(self.threshold) and self.bimodal and self.contrast_db >= min_contrast_db)


_NONE = OtsuResult(np.nan, 0.0, 0.0, 0.0, 0)


def otsu_threshold(values, nbins=256, value_range=DB_RANGE):
    """Otsu threshold of a 1-D sample. Returns OtsuResult (threshold NaN if the sample is degenerate)."""
    v = np.asarray(values, np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size < 50:
        return _NONE
    lo, hi = value_range
    v = v[(v >= lo) & (v <= hi)]
    if v.size < 50 or np.ptp(v) < 1e-3:
        return _NONE
    hist, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = hist / hist.sum()
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu_cum = np.cumsum(p * centers)
    mu_t = mu_cum[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = mu_cum / w0
        mu1 = (mu_t - mu_cum) / w1
        sigma_b = w0 * w1 * (mu0 - mu1) ** 2
    sigma_b = np.where(np.isfinite(sigma_b), sigma_b, 0.0)
    k = int(np.argmax(sigma_b))
    var_t = float(np.sum(p * (centers - mu_t) ** 2))
    below, above = v[v <= centers[k]], v[v > centers[k]]
    if below.size < 2 or above.size < 2:
        return _NONE
    t = float(0.5 * (centers[k] + centers[min(k + 1, nbins - 1)]))
    d = float(np.sqrt(2.0) * abs(below.mean() - above.mean()) / np.sqrt(below.var() + above.var() + 1e-12))
    return OtsuResult(t, float(sigma_b[k] / max(var_t, 1e-12)), d, float(below.size / v.size), int(v.size),
                      float(above.mean() - below.mean()))


def binary_otsu(db, valid=None, **kw):
    """Plain Otsu over every valid pixel of the image."""
    db = np.asarray(db, np.float32)
    ok = np.isfinite(db) if valid is None else (valid & np.isfinite(db))
    return otsu_threshold(db[ok], **kw)


def tiled_binary_otsu(db, valid=None, tile=256, min_d=2.0, min_bcv=0.6, min_water=0.05, max_water=0.95, **kw):
    """Binary-Otsu on tiles; keep the bimodal ones (Ashman D, BCV and a sensible water share) and take the median
    threshold. Returns (OtsuResult of the median — bcv/D/water averaged over the kept tiles, n_tiles_used)."""
    db = np.asarray(db, np.float32)
    ok = np.isfinite(db) if valid is None else (valid & np.isfinite(db))
    h, w = db.shape
    kept = []
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            r = otsu_threshold(db[y:y + tile, x:x + tile][ok[y:y + tile, x:x + tile]], **kw)
            if np.isfinite(r.threshold) and r.ashman_d >= min_d and r.bcv >= min_bcv \
                    and r.contrast_db >= MIN_CONTRAST_DB and min_water <= r.water_fraction <= max_water:
                kept.append(r)
    if not kept:
        return _NONE, 0
    t = float(np.median([r.threshold for r in kept]))
    return OtsuResult(t, float(np.mean([r.bcv for r in kept])), float(np.mean([r.ashman_d for r in kept])),
                      float(np.mean([r.water_fraction for r in kept])), int(sum(r.n for r in kept)),
                      float(np.mean([r.contrast_db for r in kept]))), len(kept)


# ---------------------------------------------------------------------------------------- edge Otsu
def _disk(radius):
    r = int(max(1, radius))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def edge_buffer(db, valid=None, sigma=1.0, low_q=0.80, high_q=0.97, buffer_px=3):
    """Boolean mask of the pixels within `buffer_px` of a Canny edge of the (dB) image. Edges are found on the
    image rescaled to 0-1 between its 1st and 99th percentile; thresholds are gradient-magnitude quantiles."""
    from skimage.feature import canny
    db = np.asarray(db, np.float32)
    ok = np.isfinite(db) if valid is None else (valid & np.isfinite(db))
    if not ok.any():
        return np.zeros(db.shape, bool)
    lo, hi = np.percentile(db[ok], [1, 99])
    img = np.clip((np.where(ok, db, np.median(db[ok])) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    edges = canny(img, sigma=sigma, low_threshold=low_q, high_threshold=high_q, use_quantiles=True)
    edges &= ok
    return ndi.binary_dilation(edges, structure=_disk(buffer_px)) & ok


def edge_otsu(db, valid=None, min_pixels=800, **kw):
    """Edge-Otsu (Donchyts et al. 2016). Returns (OtsuResult, edge-buffer mask); NaN threshold if the image has too
    few edge pixels for a stable histogram (no shoreline in the scene)."""
    db = np.asarray(db, np.float32)
    buf_kw = {k: kw.pop(k) for k in ("sigma", "low_q", "high_q", "buffer_px") if k in kw}
    buf = edge_buffer(db, valid, **buf_kw)
    if buf.sum() < min_pixels:
        return _NONE, buf
    r = otsu_threshold(db[buf], **kw)
    return (r if r.valid() else _NONE), buf     # edges of land texture give a weak, low-contrast split: reject


# ---------------------------------------------------------------------------------------- combination
@dataclass
class Thresholds:
    t_edge: float
    t_binary: float
    t_low: float            # below this: certain water
    t_high: float           # above this: certain land; between low and high = uncertain
    mode: str               # 'both' | 'edge_only' | 'binary_only' | 'none'
    edge: OtsuResult = field(repr=False, default=None)
    binary: OtsuResult = field(repr=False, default=None)
    n_tiles: int = 0
    gap_db: float = 0.0

    @property
    def usable(self):
        return self.mode != "none"

    @property
    def t_mean(self):
        return 0.5 * (self.t_low + self.t_high)


def combined_otsu(db, valid=None, tile=256, max_gap_db=3.0, edge_kw=None, binary_kw=None):
    """Edge-Otsu + binary-Otsu.

    * binary-Otsu is the tiled, bimodal-tiles-only version, with the global one as fallback when no tile qualifies
      but the whole scene is clearly bimodal.
    * If both exist and are within `max_gap_db` (dB), they bracket an uncertain band [t_low, t_high].
    * If they disagree by more than that, the binary one is distrusted (it is the one that breaks when water is
      scarce) and the edge threshold is used alone, with a narrow band around it.
    * If only one exists, it is used alone. If none, `mode == 'none'` (no water, or no usable contrast).
    """
    edge_kw, binary_kw = dict(edge_kw or {}), dict(binary_kw or {})
    e, _ = edge_otsu(db, valid, **edge_kw)
    b, n_tiles = tiled_binary_otsu(db, valid, tile=tile, **binary_kw)
    if not np.isfinite(b.threshold):
        g = binary_otsu(db, valid, **binary_kw)
        if g.valid():
            b = g
    te, tb = e.threshold, b.threshold
    if np.isfinite(te) and np.isfinite(tb):
        gap = abs(te - tb)
        if gap <= max_gap_db:
            return Thresholds(te, tb, min(te, tb), max(te, tb), "both", e, b, n_tiles, gap)
        return Thresholds(te, tb, te - 0.5, te + 0.5, "edge_only", e, b, n_tiles, gap)
    if np.isfinite(te):
        return Thresholds(te, np.nan, te - 0.5, te + 0.5, "edge_only", e, b, n_tiles)
    if np.isfinite(tb):
        return Thresholds(np.nan, tb, tb - 0.5, tb + 0.5, "binary_only", e, b, n_tiles)
    return Thresholds(np.nan, np.nan, np.nan, np.nan, "none", e, b, n_tiles)


# ---------------------------------------------------------------------------------------- masks
CLASS_LAND, CLASS_WATER, CLASS_UNCERTAIN, CLASS_NODATA = 0, 1, 2, 255


def classify(db, th: Thresholds, valid=None, rule="hysteresis"):
    """Per-pixel classes from thresholds.

      CLASS_WATER     db < t_low (both methods agree), or in the uncertain band AND connected to such water
      CLASS_UNCERTAIN uncertain band pixels with no connection to certain water
      CLASS_LAND      db > t_high
      CLASS_NODATA    invalid / NaN

    rule 'hysteresis' (default) as above; 'mean' = db < mean threshold with no uncertain class.
    """
    from skimage.filters import apply_hysteresis_threshold
    db = np.asarray(db, np.float32)
    ok = np.isfinite(db) if valid is None else (valid & np.isfinite(db))
    out = np.full(db.shape, CLASS_NODATA, np.uint8)
    if not th.usable:
        out[ok] = CLASS_LAND
        return out
    filled = np.where(ok, db, np.float32(1e3))
    if rule == "mean":
        out[ok] = np.where(filled[ok] < th.t_mean, CLASS_WATER, CLASS_LAND)
        return out
    if rule != "hysteresis":
        raise ValueError("rule must be 'hysteresis' or 'mean'")
    out[ok] = CLASS_LAND
    band = ok & (db < th.t_high)
    sure = ok & (db < th.t_low)
    if th.t_high > th.t_low:
        grown = apply_hysteresis_threshold(-filled, -th.t_high, -th.t_low) & ok
    else:
        grown = sure
    out[band] = CLASS_UNCERTAIN
    out[grown] = CLASS_WATER
    out[sure] = CLASS_WATER
    return out


def clean_mask(water, min_area_px=20, fill_holes_px=20):
    """Remove water blobs smaller than min_area_px and fill land holes smaller than fill_holes_px (4-connected)."""
    water = np.asarray(water, bool)
    lab, n = ndi.label(water)
    if n:
        sizes = ndi.sum(water, lab, index=np.arange(1, n + 1))
        water = np.isin(lab, 1 + np.flatnonzero(sizes >= min_area_px))
    land = ~water
    lab, n = ndi.label(land)
    if n:
        sizes = ndi.sum(land, lab, index=np.arange(1, n + 1))
        water = water | np.isin(lab, 1 + np.flatnonzero(sizes < fill_holes_px))
    return water
