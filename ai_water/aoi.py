"""Clip results to the outline of a water body (or to the land, for nationwide tiles)."""
from __future__ import annotations

import numpy as np


def aoi_mask(site, geobox):
    """Boolean (h, w) mask of the pixels of `geobox` inside `site.geometry` (a lon/lat outline), or None when the site has no
    outline (then the whole box counts). Pixels the outline touches are included, so a one-pixel river is not lost."""
    geom = getattr(site, "geometry", None)
    if geom is None:
        return None
    from pyproj import Transformer
    from rasterio.features import rasterize
    from .sites import project
    to_site = Transformer.from_crs("EPSG:4326", str(geobox.crs), always_xy=True)
    g = project(geom, to_site)
    m = rasterize([(g, 1)], out_shape=tuple(geobox.shape), transform=geobox.transform, fill=0, all_touched=True, dtype="uint8")
    return m.astype(bool)
