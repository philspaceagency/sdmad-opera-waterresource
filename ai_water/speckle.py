"""SAR speckle filters (NumPy / SciPy, testable offline).

All filters work on LINEAR power (gamma0 / sigma0), never on dB: speckle is multiplicative in linear
power, and averaging in dB biases the mean low. Convert to dB only after filtering (`preprocess.to_db`).

  * `lee`          - classic Lee (1980) MMSE filter on a square window.
  * `refined_lee`  - Lee (1981) refined filter: 7x7 window, edge direction found from a 3x3 grid of
                     sub-window means, statistics taken on the half-window on the homogeneous side of
                     the edge. Smooths water and land hard but keeps the shoreline sharp, which is what
                     edge-based thresholding needs.
  * `multitemporal_mean` - mean of co-registered, same-orbit acquisitions (Quegan-style, in linear power).

NaN pixels (border noise, no data) are ignored: they are filled with the scene mean for the filtering
and restored to NaN afterwards.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage as ndi

_EPS = 1e-12


def _fill(x):
    """(filled copy, nan mask). NaNs are replaced by the scene's nan-mean so they do not leak into windows."""
    x = np.asarray(x, np.float32)
    bad = ~np.isfinite(x)
    if not bad.any():
        return x, bad
    fill = np.float32(np.nanmean(np.where(bad, np.nan, x))) if (~bad).any() else np.float32(0.0)
    return np.where(bad, fill, x), bad


def _box_stats(x, size):
    """Local mean and variance on a size x size window (reflect at the borders)."""
    m = ndi.uniform_filter(x, size=size, mode="reflect")
    m2 = ndi.uniform_filter(x * x, size=size, mode="reflect")
    return m, np.maximum(m2 - m * m, 0.0)


def lee(x, size=7, enl=None):
    """Classic Lee filter. `enl` = equivalent number of looks; None estimates the noise coefficient of variation
    from the scene's smoothest windows (mean of the lowest 5 % of local CV^2)."""
    x, bad = _fill(x)
    m, v = _box_stats(x, size)
    cv2 = v / (m * m + _EPS)
    cu2 = _noise_cv2(cv2, enl)
    out = _lee_mmse(x, m, cv2, cu2)
    out[bad] = np.nan
    return out.astype(np.float32)


def _noise_cv2(cv2, enl):
    if enl is not None:
        return 1.0 / float(enl)
    return float(np.mean(np.sort(cv2.ravel())[: max(1, cv2.size // 20)]))


def _lee_mmse(x, m, cv2, cu2):
    """Lee MMSE estimate: m + w (x - m), w = 1 - cu2/cv2 clipped to [0, 1]."""
    w = np.clip(1.0 - cu2 / np.maximum(cv2, _EPS), 0.0, 1.0)
    return m + w * (x - m)


# ---------------------------------------------------------------------------------------- refined Lee
def _half_plane_masks():
    """Eight 7x7 half-window masks (centre row/column/diagonal included), in the order
    top, bottom, left, right, top-left, bottom-right, top-right, bottom-left."""
    r, c = np.mgrid[0:7, 0:7]
    return np.stack([
        r <= 3, r >= 3, c <= 3, c >= 3,
        (r + c) <= 6, (r + c) >= 6,
        (r - c) <= 0, (r - c) >= 0,
    ]).astype(np.float32)


_MASKS = _half_plane_masks()
# (side A, side B) mask indices for each gradient axis, and the sub-window pair (row-major 3x3 index) it uses
_AXES = (
    ((1, 7), (0, 1)),      # vertical      : m1 (top)          vs m7 (bottom)       -> masks top / bottom
    ((6, 2), (7, 6)),      # anti-diagonal : m6 (bottom-left)  vs m2 (top-right)    -> masks bottom-left / top-right
    ((3, 5), (2, 3)),      # horizontal    : m3 (left)         vs m5 (right)        -> masks left / right
    ((0, 8), (4, 5)),      # diagonal      : m0 (top-left)     vs m8 (bottom-right) -> masks top-left / bottom-right
)


def _masked_stats(x, mask):
    """Mean and variance of x over a 7x7 kernel restricted to `mask` (normalised)."""
    k = mask / mask.sum()
    m = ndi.correlate(x, k, mode="reflect")
    m2 = ndi.correlate(x * x, k, mode="reflect")
    return m, np.maximum(m2 - m * m, 0.0)


def refined_lee(x, enl=None):
    """Refined Lee speckle filter (Lee 1981; 7x7 window, 3x3 sub-windows).

    1. Means/variances of the nine 3x3 sub-windows of the 7x7 window are read at offsets (-2, 0, +2).
    2. The edge axis is the one with the largest sub-window mean gradient (vertical, horizontal or one
       of the two diagonals).
    3. The half-window on the side of the edge whose mean is closer to the centre sub-window is used.
    4. Lee MMSE with the half-window mean/variance. Noise CV^2 comes from `enl` (1/ENL) or, when None,
       from the mean of the 5 smallest normalised variances of the nine sub-windows (as in the usual
       Earth Engine refined-Lee implementation), computed per pixel.
    """
    x, bad = _fill(x)
    m3, v3 = _box_stats(x, 3)

    # nine sub-window means / variances at offsets (-2, 0, +2) in row and column (edge-padded at the border)
    sm, sv = _subwindows(m3, v3, (-2, 0, 2))

    centre = sm[4]
    grads = np.stack([np.abs(sm[a] - sm[b]) for (a, b), _ in _AXES])          # (4, h, w)
    axis = np.argmax(grads, axis=0)                                            # (h, w)

    # for each axis, which side is homogeneous with the centre
    mask_idx = np.empty(x.shape, np.int8)
    for ax, ((a, b), (ma, mb)) in enumerate(_AXES):
        pick_a = np.abs(sm[a] - centre) <= np.abs(sm[b] - centre)
        mask_idx = np.where(axis == ax, np.where(pick_a, ma, mb), mask_idx)

    # statistics for the eight half-windows, then the chosen one per pixel
    stats = [_masked_stats(x, m) for m in _MASKS]
    mean8 = np.stack([s[0] for s in stats])
    var8 = np.stack([s[1] for s in stats])
    idx = mask_idx[None].astype(np.intp)
    mean_sel = np.take_along_axis(mean8, idx, axis=0)[0]
    var_sel = np.take_along_axis(var8, idx, axis=0)[0]

    if enl is not None:
        cu2 = 1.0 / float(enl)
    else:
        nv = np.sort(sv / (sm * sm + _EPS), axis=0)[:5]
        cu2 = nv.mean(axis=0)
    cv2 = var_sel / (mean_sel * mean_sel + _EPS)
    out = _lee_mmse(x, mean_sel, cv2, cu2)
    out[bad] = np.nan
    return out.astype(np.float32)


def _subwindows(m3, v3, offs):
    """(9, h, w) stacks of the 3x3 sub-window means / variances shifted by (dy, dx) in offs x offs, row-major."""
    pad = max(abs(o) for o in offs)
    mp = np.pad(m3, pad, mode="edge")
    vp = np.pad(v3, pad, mode="edge")
    h, w = m3.shape
    sm, sv = [], []
    for dy in offs:
        for dx in offs:
            sm.append(mp[pad + dy:pad + dy + h, pad + dx:pad + dx + w])
            sv.append(vp[pad + dy:pad + dy + h, pad + dx:pad + dx + w])
    return np.stack(sm), np.stack(sv)


# ---------------------------------------------------------------------------------------- multi-temporal
def multitemporal_mean(stack, min_obs=2):
    """Mean over time of a (T, h, w) stack of linear power from ONE orbit (same geometry); pixels with fewer than
    `min_obs` valid observations are NaN. Reduces speckle by ~sqrt(T) without blurring space, but averages
    over water-level change: use it only for a stable-extent (e.g. dry-season) reference, never for an event."""
    stack = np.asarray(stack, np.float32)
    n = np.isfinite(stack).sum(0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
    return np.where(n >= min_obs, mean, np.nan).astype(np.float32)


FILTERS = {"refined_lee": refined_lee, "lee": lee, "none": lambda x, **_: np.asarray(x, np.float32)}


def apply_filter(x, name="refined_lee", **kw):
    if name not in FILTERS:
        raise ValueError(f"speckle filter must be one of {sorted(FILTERS)}")
    return FILTERS[name](x, **kw)
