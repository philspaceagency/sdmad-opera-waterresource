"""Frequency figure and the GeoPackage output."""
import numpy as np
import pytest
from affine import Affine
from types import SimpleNamespace

from ai_water import plots

GB = SimpleNamespace(transform=Affine(10, 0, 300000, 0, -10, 1650000), crs="EPSG:32651")


def test_classes_follow_the_legend():
    f = np.array([[0, 1, 5, 6, 20, 21, 50, 51, 80, 81, 100, 255]], np.uint8)
    assert plots.class_index(f).tolist() == [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, -1]]


def test_figure_is_written_and_cropped_to_the_observed_area(tmp_path):
    from PIL import Image
    f = np.full((300, 400), 255, np.uint8)
    f[100:200, 120:300] = 0
    f[130:170, 150:250] = 70
    p = plots.frequency_figure(str(tmp_path / "f.png"), f, GB, "Some Dam", "subtitle", "footer", margin_px=10)
    im = Image.open(p)
    assert im.size[0] > 800 and im.size[1] > 500
    with pytest.raises(ValueError):
        plots.frequency_figure(str(tmp_path / "g.png"), np.full((20, 20), 255, np.uint8), GB, "x")


def test_gpkg_holds_water_as_1_and_non_water_as_0(tmp_path):
    gpd = pytest.importorskip("geopandas")
    pytest.importorskip("pyogrio")
    from ai_water import infer
    water = np.zeros((40, 50), bool)
    water[10:20, 10:30] = True
    valid = np.ones_like(water)
    valid[:, 45:] = False
    infer.write_gpkg(str(tmp_path / "w.gpkg"), water, valid, GB)
    g = gpd.read_file(tmp_path / "w.gpkg", layer="surfacewater")
    assert sorted(g["water"]) == [0, 1] and g.crs.to_string() == "EPSG:32651"
    w = g[g["water"] == 1].geometry.iloc[0]
    n = g[g["water"] == 0].geometry.iloc[0]
    assert w.area == pytest.approx(10 * 20 * 100) and n.area == pytest.approx((40 * 45 - 200) * 100)    # 10 m pixels; invalid columns excluded
