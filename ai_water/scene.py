"""One Sentinel-1 scene -> the filtered features the model reads: refined Lee in linear power, then VV / VH in dB and SDWI.

Pure NumPy (no network). `write_tif` is the GeoTIFF writer shared by the dataset, the inference outputs and the composites.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import features, preprocess as pp, speckle


VH_FLOOR_DB = -45.0


@dataclass
class SceneConfig:
    resolution_m: float = 10.0
    min_orbit_cover: float = 0.9                 # a pass counts only if its frames cover this share of the area
    speckle_filter: str = "refined_lee"          # 'refined_lee' | 'lee' | 'none'
    enl: float = None                            # equivalent number of looks; None = estimated from the scene


@dataclass
class S1Features:
    vv_db: np.ndarray
    vh_db: np.ndarray
    sdwi: np.ndarray

    @property
    def valid(self):
        return np.isfinite(self.vv_db) & np.isfinite(self.vh_db)


def prepare(vv_lin, vh_lin, cfg: SceneConfig = None) -> S1Features:
    """Linear VV / VH -> speckle filter -> (VV dB, VH dB, SDWI). Non-positive values and border noise are NaN in all three. VV
    decides what is a valid pixel; VH has a lower floor (-45 dB) because open water and radar shadow can legitimately be darker
    than -30 dB in the cross-polarised band, and those are exactly the pixels the model must see."""
    cfg = cfg or SceneConfig()
    kw = {"enl": cfg.enl} if cfg.speckle_filter != "none" else {}
    vv = speckle.apply_filter(vv_lin, cfg.speckle_filter, **kw)
    vh = speckle.apply_filter(vh_lin, cfg.speckle_filter, **kw)
    vv_db, vh_db = pp.to_db(vv), pp.to_db(vh, floor_db=VH_FLOOR_DB)
    bad = ~(np.isfinite(vv_db) & np.isfinite(vh_db))
    vv_db[bad] = np.nan
    vh_db[bad] = np.nan
    sd = features.sdwi(vv, vh)
    sd[bad] = np.nan
    return S1Features(vv_db, vh_db, sd)


def write_tif(path, arr, geobox, dtype, nodata=None):
    import rasterio
    from rasterio.transform import Affine
    h, w = arr.shape
    t = geobox.transform
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1, dtype=dtype, crs=str(geobox.crs),
                       transform=Affine(t.a, t.b, t.c, t.d, t.e, t.f), compress="deflate", nodata=nodata,
                       tiled=True, blockxsize=256, blockysize=256) as dst:
        dst.write(arr.astype(dtype), 1)
