"""Area-elevation curve A(h) of a reservoir and the volume above a reference level.

SWOT gives the level h, Sentinel-1 the area A on (nearly) the same day. Pairs (h, A) over the reservoir's range of levels trace its
hypsometric curve. A is fitted as a monotone function of h (isotonic regression through the pairs, then linear interpolation; a
power law A = a (h - h0)^b is also available), and the storage between two levels is  V = integral of A dh.

Absolute storage needs the volume at one level (`v_ref` at `h_ref`, e.g. dead storage at the minimum operating level from the dam
operator). Without it, volumes are relative to the lowest level in the pairs ('storage above the lowest observed level') and say so.
"""
from __future__ import annotations

import numpy as np


class AreaCurve:
    """Monotone A(h): `h` ascending (m), `a` area (km2, non-decreasing). Outside the sampled range it is held/extrapolated per `extrapolate`."""

    def __init__(self, h, a, extrapolate="linear"):
        self.h = np.asarray(h, float)
        self.a = np.asarray(a, float)
        self.extrapolate = extrapolate

    def area(self, h):
        h = np.asarray(h, float)
        a = np.interp(h, self.h, self.a)
        if self.extrapolate == "linear" and len(self.h) > 1:
            s_lo = (self.a[1] - self.a[0]) / max(self.h[1] - self.h[0], 1e-9)
            s_hi = (self.a[-1] - self.a[-2]) / max(self.h[-1] - self.h[-2], 1e-9)
            a = np.where(h < self.h[0], np.maximum(self.a[0] + s_lo * (h - self.h[0]), 0.0), a)
            a = np.where(h > self.h[-1], self.a[-1] + s_hi * (h - self.h[-1]), a)
        return a

    def volume_between(self, h0, h1, n=200):
        """Million m3 between levels h0 <= h1 (km2 x m = 1e6 m3). Trapezoid integral of A(h) over n steps."""
        h0, h1 = np.broadcast_arrays(np.asarray(h0, float), np.asarray(h1, float))
        out = np.empty(h0.shape)
        for idx in np.ndindex(h0.shape):
            hs = np.linspace(h0[idx], h1[idx], n)
            ar = self.area(hs)
            out[idx] = float(np.sum((ar[1:] + ar[:-1]) / 2 * np.diff(hs)))
        return out


def _pava(y, w):
    """Pool-adjacent-violators: the non-decreasing least-squares fit to y (sorted by x) with weights w."""
    y = list(map(float, y)); w = list(map(float, w)); idx = [[i] for i in range(len(y))]
    i = 0
    while i < len(y) - 1:
        if y[i] > y[i + 1]:
            tw = w[i] + w[i + 1]
            y[i:i + 2] = [(y[i] * w[i] + y[i + 1] * w[i + 1]) / tw]
            w[i:i + 2] = [tw]
            idx[i:i + 2] = [idx[i] + idx[i + 1]]
            i = max(i - 1, 0)
        else:
            i += 1
    out = np.empty(sum(len(g) for g in idx))
    for v, g in zip(y, idx):
        out[g] = v
    return out


def fit_curve(h, a, method="isotonic", bin_m=0.25):
    """Fit A(h) from paired observations. Levels closer than `bin_m` are averaged first (so the curve is not driven by clusters).
    method 'isotonic' (default, no shape assumption) or 'power' (A = c (h - h0)^b, h0 fitted below the lowest level)."""
    h = np.asarray(h, float); a = np.asarray(a, float)
    ok = np.isfinite(h) & np.isfinite(a)
    h, a = h[ok], a[ok]
    if len(h) < 3:
        raise ValueError(f"need at least 3 (level, area) pairs to fit an area-elevation curve, have {len(h)}")
    key = np.round(h / bin_m).astype(int)
    hb = np.array([h[key == k].mean() for k in np.unique(key)])
    ab = np.array([a[key == k].mean() for k in np.unique(key)])
    wb = np.array([(key == k).sum() for k in np.unique(key)], float)
    order = np.argsort(hb)
    hb, ab, wb = hb[order], ab[order], wb[order]
    if len(hb) < 2:
        raise ValueError("levels span less than one bin; widen the date range or lower bin_m")
    if method == "isotonic":
        return AreaCurve(hb, _pava(ab, wb))
    if method == "power":
        best = None
        for h0 in np.linspace(hb[0] - 5 * (hb[-1] - hb[0]), hb[0] - 1e-3, 80):
            x = np.log(hb - h0); y = np.log(np.clip(ab, 1e-6, None))
            b, lc = np.polyfit(x, y, 1, w=np.sqrt(wb))
            r = float(np.sum(wb * (y - (b * x + lc)) ** 2))
            if best is None or r < best[0]:
                best = (r, h0, b, np.exp(lc))
        _, h0, b, c = best
        hs = np.linspace(hb[0], hb[-1], 100)
        return AreaCurve(hs, c * (hs - h0) ** b)
    raise ValueError("method must be 'isotonic' or 'power'")


def volume_series(curve, levels, h_ref=None, v_ref=0.0):
    """Storage (million m3) at each level: v_ref + integral of A dh from h_ref (default: the curve's lowest sampled level)."""
    h_ref = curve.h[0] if h_ref is None else h_ref
    return v_ref + curve.volume_between(np.full(np.shape(levels), h_ref), np.asarray(levels, float))
