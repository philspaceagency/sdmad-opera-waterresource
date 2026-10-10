"""JRC Global Surface Water from Google Earth Engine, read onto the Sentinel-1 grid and matched to the S1 date.

  * `JRC/GSW1_4/MonthlyHistory`  band `water` per calendar month: 0 no data, 1 not water, 2 water (Landsat, 30 m).
  * `JRC/GSW1_4/GlobalSurfaceWater` occurrence 0-100 % over 1984-2021.

Temporal matching: the JRC product is monthly, so the finest possible match to a Sentinel-1 acquisition is the
CALENDAR MONTH of the pass. `matched_month` fetches exactly that month; only if it is mostly cloud/no-data does it
look at neighbouring months, up to `max_offset` (default 0 = never), and it reports the offset used so training
code can discard or down-weight inexact matches. GSW1_4 ends in Dec 2021 (check the current version of the asset).

Earth Engine is reached through `ee.data.computePixels` on the S1 grid itself (same pattern as ai-irrigation's
weather.py), so no resampling step is needed on our side. The fetcher is a small class so tests (and other
sources, e.g. a local raster) can stand in for it: anything with `.month(month, grid)` and `.occurrence(grid)`.
"""
from __future__ import annotations

import re
import time as _time
from dataclasses import dataclass

import numpy as np
import pandas as pd

MONTHLY_ASSET = "JRC/GSW1_4/MonthlyHistory"
OCCURRENCE_ASSET = "JRC/GSW1_4/GlobalSurfaceWater"
JRC_FIRST = pd.Timestamp("1984-03-01")
JRC_LAST = pd.Timestamp("2021-12-01")          # last monthly image in GSW1_4
NODATA, NOT_WATER, WATER = 0, 1, 2
# Band names are read from Earth Engine and matched against these (first hit wins), so a wrong guess fails with the list of
# real bands instead of a bare EEException. GSW1_4 MonthlyHistory has `water`; `waterClass` is the YearlyHistory name.
MONTHLY_BANDS = ("water", "waterClass")
OCCURRENCE_BANDS = ("occurrence",)
_TRANSIENT = re.compile(r"\b(quota|too many|rate limit|timed out|timeout|deadline|unavailable|internal error|50[0234]|429|"
                        r"connection|reset by peer|temporar\w+)\b", re.I)


def _is_transient(e):
    """Worth retrying: quota, timeouts, 5xx, connection trouble. A request Earth Engine rejects (missing band, bad asset id,
    malformed expression) fails identically every time and is raised at once."""
    return bool(_TRANSIENT.search(f"{e.__class__.__name__} {e}"))


@dataclass(frozen=True)
class Grid:
    """Target raster grid: metric CRS, square pixels, upper-left corner (x0, y1)."""
    crs: str
    res: float
    x0: float
    y1: float
    nx: int
    ny: int


def grid_from_geobox(gb):
    t = gb.transform
    return Grid(str(gb.crs), float(t.a), float(t.c), float(t.f), int(gb.shape[1]), int(gb.shape[0]))


def grid_for_bbox(bbox, crs, res_m):
    """Grid of `res_m` pixels covering a lon/lat (west, south, east, north) box in a metric CRS (pyproj only, no odc)."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tr.transform([bbox[0], bbox[2], bbox[0], bbox[2]], [bbox[1], bbox[1], bbox[3], bbox[3]])
    x0, x1 = np.floor(min(xs) / res_m) * res_m, np.ceil(max(xs) / res_m) * res_m
    y0, y1 = np.floor(min(ys) / res_m) * res_m, np.ceil(max(ys) / res_m) * res_m
    return Grid(crs, float(res_m), float(x0), float(y1), int((x1 - x0) / res_m), int((y1 - y0) / res_m))


def init_gee(project=None, high_volume=True, authenticate=False):
    """ee.Initialize (same as ai-irrigation's weather.init_gee). Authenticate once with `earthengine authenticate`."""
    import ee
    if authenticate:
        ee.Authenticate()
    kw = {"project": project} if project else {}
    if high_volume:
        kw["opt_url"] = "https://earthengine-highvolume.googleapis.com"
    ee.Initialize(**kw)
    return ee


class GeeJrc:
    """JRC GSW from Earth Engine, resampled by EE (nearest) onto the requested grid, in row blocks that stay under
    the computePixels size limit."""

    def __init__(self, ee_module=None, monthly_asset=MONTHLY_ASSET, occurrence_asset=OCCURRENCE_ASSET,
                 max_request_mb=30, tries=5):
        self._ee = ee_module
        self.monthly_asset, self.occurrence_asset = monthly_asset, occurrence_asset
        self.max_request_mb, self.tries = max_request_mb, tries
        self._bands = {}

    @property
    def ee(self):
        if self._ee is None:
            import ee
            self._ee = ee
        return self._ee

    # -- public
    def month(self, month, grid):
        """(ny, nx) uint8 water class (0 no data, 1 not water, 2 water) for the calendar month containing `month`."""
        ee = self.ee
        band = self._band(self.monthly_asset, ee.ImageCollection(self.monthly_asset).first(), MONTHLY_BANDS)
        start = pd.Timestamp(month).to_period("M").start_time
        end = start + pd.offsets.MonthBegin(1)
        col = ee.ImageCollection(self.monthly_asset).filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        img = col.select(band).mosaic().unmask(0).toUint8()
        return self._compute(img, grid, band, np.uint8)

    def occurrence(self, grid):
        """(ny, nx) float32 occurrence 0-100; NaN where JRC has no observation."""
        ee = self.ee
        band = self._band(self.occurrence_asset, ee.Image(self.occurrence_asset), OCCURRENCE_BANDS)
        img = ee.Image(self.occurrence_asset).select(band).unmask(-1).toFloat()
        a = self._compute(img, grid, band, np.float32)
        a[a < 0] = np.nan
        return a

    def _band(self, asset, image, candidates):
        """The band of `asset` to use: the first of `candidates` that exists. Asks Earth Engine once per asset (cached)."""
        if asset not in self._bands:
            names = list(image.bandNames().getInfo())
            hit = next((c for c in candidates if c in names), None)
            if hit is None:
                raise RuntimeError(f"{asset} has bands {names}, none of the expected {list(candidates)}: "
                                   "set GeeJrc's asset (or the *_BANDS constants in jrc.py) to match.")
            self._bands[asset] = hit
        return self._bands[asset]

    # -- request plumbing
    def _compute(self, image, grid, name, dtype):
        itemsize = np.dtype(dtype).itemsize
        rows = max(1, int(self.max_request_mb * 1e6 // max(1, grid.nx * itemsize)))
        out = np.empty((grid.ny, grid.nx), dtype)
        for r0 in range(0, grid.ny, rows):
            r1 = min(grid.ny, r0 + rows)
            out[r0:r1] = self._block(image, grid, name, r0, r1)
        return out

    def _block(self, image, grid, name, r0, r1):
        req = {"expression": image.select([name]), "fileFormat": "NUMPY_NDARRAY",
               "grid": {"dimensions": {"width": grid.nx, "height": r1 - r0},
                        "affineTransform": {"scaleX": grid.res, "shearX": 0, "translateX": grid.x0,
                                            "shearY": 0, "scaleY": -grid.res, "translateY": grid.y1 - r0 * grid.res},
                        "crsCode": grid.crs}}
        last = None
        for k in range(self.tries):
            try:
                return np.asarray(self.ee.data.computePixels(req)[name])
            except Exception as e:
                if not _is_transient(e):                            # rejected request: retrying cannot help
                    raise RuntimeError(f"Earth Engine rejected the request: {e!r}") from e
                last = e                                            # quota, timeout, 5xx: back off and retry
                if k < self.tries - 1:
                    _time.sleep(min(30, 2 ** k))
        raise RuntimeError(f"Earth Engine request failed after {self.tries} tries: {last!r}")


@dataclass
class MatchedMonth:
    water_class: np.ndarray      # (ny, nx) uint8: 0 no data, 1 not water, 2 water
    month: pd.Timestamp          # the JRC month actually used
    offset_months: int           # 0 = the month of the S1 pass
    valid_frac: float            # share of the AOI with a JRC observation that month


def matched_month(fetcher, grid, day, max_offset=0, min_valid=0.6, cache=None):
    """The JRC monthly map for the month of the Sentinel-1 date `day`.

    Tries offset 0 first, then -1, +1, -2, +2 ... up to `max_offset` months; the first month whose valid share is at
    least `min_valid` wins. Returns None if the date is outside JRC coverage or no allowed month is good enough
    (the scene is then not used for JRC-labelled training). `cache` (dict) shares months between S1 dates."""
    base = pd.Timestamp(day).to_period("M").start_time
    order = [0] + [s * k for k in range(1, max_offset + 1) for s in (-1, 1)]
    for off in order:
        m = base + pd.DateOffset(months=off)
        if m < JRC_FIRST or m > JRC_LAST:
            continue
        key = (grid, m)
        if cache is not None and key in cache:
            arr = cache[key]
        else:
            arr = fetcher.month(m, grid)
            if cache is not None:
                cache[key] = arr
        valid = float((arr > NODATA).mean())
        if valid >= min_valid:
            return MatchedMonth(arr, m, off, valid)
    return None
