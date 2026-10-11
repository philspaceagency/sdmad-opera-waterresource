"""Earth Engine terrain / land-cover requests (mocked) and the per-grid cache."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from ai_water import features as ft, jrc, synthetic as syn, terrain

G = jrc.Grid("EPSG:32651", 10.0, 300000.0, 1650000.0, 64, 48)


def fake_ee(dem_bands=("DEM", "EDM", "FLM"), lc_bands=("Map",), seen=None):
    ee = MagicMock()
    bands = {terrain.DEM_ASSET: list(dem_bands), terrain.LANDCOVER_ASSET: list(lc_bands)}
    ee.ImageCollection.side_effect = lambda asset: MagicMock(**{"first.return_value.bandNames.return_value.getInfo.return_value": bands[asset]},
                                                           _asset=asset)

    def compute(req):
        if seen is not None:
            seen.append(req)
        g = req["grid"]["dimensions"]
        return {"DEM": np.full((g["height"], g["width"]), 250.0, np.float32),
                "Map": np.full((g["height"], g["width"]), 10, np.uint8)}
    ee.data.computePixels.side_effect = compute
    return ee


def test_dem_and_landcover_are_read_with_the_real_band_names():
    seen = []
    src = terrain.GeeTerrain(ee_module=fake_ee(seen=seen))
    dem, lc = src.dem(G), src.landcover(G)
    assert dem.shape == (48, 64) and dem.dtype == np.float32 and (dem == 250).all()
    assert lc.shape == (48, 64) and lc.dtype == np.uint8 and (lc == 10).all()
    assert len(seen) == 2 and seen[0]["grid"]["crsCode"] == "EPSG:32651"


def test_unknown_band_names_fail_with_the_list_of_real_ones():
    with pytest.raises(RuntimeError, match=r"has bands \['elev'\]"):
        terrain.GeeTerrain(ee_module=fake_ee(dem_bands=("elev",))).dem(G)


def test_site_terrain_builds_the_layers_and_caches_per_grid(tmp_path):
    dem = np.tile(np.arange(64, dtype=np.float32) * 5.0, (48, 1))
    src = syn.SyntheticTerrain(dem, np.full(dem.shape, 80, np.uint8))                 # WorldCover 80 = permanent water
    t = terrain.site_terrain(G, src, str(tmp_path))
    assert t.shape == (48, 64) and t.slope[5:-5, 5:-5].mean() == pytest.approx(np.degrees(np.arctan(0.5)), abs=0.2)
    assert (t.landcover == 0).all()                                                    # water is not a land-cover group
    t2 = terrain.site_terrain(G, src, str(tmp_path))
    assert src.calls == 1 and np.allclose(t2.slope, t.slope, atol=0.1)                 # second call came from the cache
    other = jrc.Grid("EPSG:32651", 10.0, 300500.0, 1650000.0, 64, 48)
    terrain.site_terrain(other, src, str(tmp_path))
    assert src.calls == 2


def test_landcover_can_be_switched_off():
    src = syn.SyntheticTerrain(np.zeros((48, 64), np.float32), np.full((48, 64), 10, np.uint8))
    assert (terrain.site_terrain(G, src, None, use_landcover=False).landcover == 0).all()
