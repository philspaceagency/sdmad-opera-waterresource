"""Terrain and land-cover layers for a site grid, from Google Earth Engine (same `computePixels` path as the JRC labels).

    DEM          COPERNICUS/DEM/GLO30 (band DEM), 30 m, resampled bilinearly onto the 10 m Sentinel-1 grid
    land cover   ESA/WorldCover/v200 (band Map), 10 m, 2021 -- context only, see features.py

`site_terrain(grid, source)` turns them into a `features.Terrain` (relief, slope, aspect, land-cover group) and keeps the result
on disk per grid, so a site's terrain is fetched once however many times the dataset or the maps are rebuilt. Anything with
`.dem(grid)` and `.landcover(grid)` can stand in for `GeeTerrain` (tests, or a local raster).
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from . import features
from .jrc import GeePixels

DEM_ASSET = "COPERNICUS/DEM/GLO30"
DEM_BANDS = ("DEM", "elevation")                 # first one the asset has wins; a miss lists the real bands
LANDCOVER_ASSET = "ESA/WorldCover/v200"
LANDCOVER_BANDS = ("Map",)


class GeeTerrain(GeePixels):
    def __init__(self, ee_module=None, dem_asset=DEM_ASSET, landcover_asset=LANDCOVER_ASSET, max_request_mb=30, tries=5):
        super().__init__(ee_module, max_request_mb, tries)
        self.dem_asset, self.landcover_asset = dem_asset, landcover_asset

    def dem(self, grid):
        """(ny, nx) float32 elevation in metres (0 where the DEM has no tile, i.e. the open sea)."""
        ee = self.ee
        col = ee.ImageCollection(self.dem_asset)
        band = self._band(self.dem_asset, col.first(), DEM_BANDS)
        img = col.select(band).map(lambda i: i.resample("bilinear")).mosaic().unmask(0).toFloat()
        return self._compute(img, grid, band, np.float32)

    def landcover(self, grid):
        """(ny, nx) uint8 WorldCover class values (0 = no data)."""
        ee = self.ee
        col = ee.ImageCollection(self.landcover_asset)
        band = self._band(self.landcover_asset, col.first(), LANDCOVER_BANDS)
        img = col.select(band).mosaic().unmask(0).toUint8()
        return self._compute(img, grid, band, np.uint8)


def _cache_file(cache_dir, grid):
    key = hashlib.md5(repr(grid).encode()).hexdigest()[:16]
    return os.path.join(cache_dir, "terrain", f"terrain_{key}.npz")


def site_terrain(grid, source, cache_dir=None, use_landcover=True):
    """The `features.Terrain` of `grid` (a jrc.Grid). Fetched from `source` once and cached in `<cache_dir>/terrain/`.
    `use_landcover=False` leaves the land-cover layer empty (the model's land-cover channels are then all zero)."""
    path = _cache_file(cache_dir, grid) if cache_dir else None
    if path and os.path.exists(path):
        t = features.Terrain.load(path)
        if t.shape == (grid.ny, grid.nx):
            return t
    dem = source.dem(grid)
    lc = features.landcover_groups(source.landcover(grid)) if use_landcover else None
    t = features.terrain_from_dem(dem, grid.res, lc)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        t.save(path + ".part.npz")
        os.replace(path + ".part.npz", path)
    return t
