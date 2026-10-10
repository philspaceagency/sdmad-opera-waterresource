"""Export helpers and the synthetic JRC stand-in (no torch needed)."""
import numpy as np
import pytest

from ai_water import jrc, synthetic as syn


def test_synthetic_jrc_is_a_30m_view_of_the_truth_with_gaps():
    t = syn.make_scene(shape=(96, 96), water_frac=0.2, seed=0)["truth"]
    cloud = np.zeros_like(t)
    cloud[:, :10] = True
    m = syn.jrc_month_from_truth(t, cloud)
    assert set(np.unique(m)) <= {jrc.NODATA, jrc.NOT_WATER, jrc.WATER} and (m[:, :10] == jrc.NODATA).all()
    assert ((m == jrc.WATER) == t)[:, 10:].mean() > 0.95        # 30 m majority vote differs from 10 m only at the shore


def test_synthetic_jrc_source_matches_the_gee_interface():
    t = syn.make_scene(shape=(96, 96), water_frac=0.2, seed=0)["truth"]
    src = syn.SyntheticJrc({"2019-05": t})
    g = jrc.Grid("EPSG:32651", 10.0, 0.0, 960.0, 96, 96)
    assert (src.month("2019-05-20", g) > 0).all() and (src.month("2019-06-01", g) == 0).all()
    assert src.occurrence(g).shape == (96, 96)
    assert jrc.matched_month(src, g, "2019-05-20").offset_months == 0


def test_write_result_makes_geotiffs(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine
    from ai_water import infer, otsu
    h = w = 32
    res = infer.SceneResult(np.zeros((h, w), np.float32), np.zeros((h, w), np.float32),
                            np.full((h, w), otsu.CLASS_LAND, np.uint8), np.zeros((h, w), bool),
                            np.full((h, w), 0.2, np.float32), np.zeros((h, w), bool), np.zeros((h, w), np.uint8), None)

    class GB:
        transform = Affine(10, 0, 300000, 0, -10, 1650000)
        crs = "EPSG:32651"
    paths = infer.write_result(str(tmp_path), "2024-05-01", res, GB)
    assert set(paths) == {"otsu_classes", "fused_water", "flags", "prob"}
    with rasterio.open(paths["prob"]) as r:
        assert r.shape == (h, w) and r.crs.to_string() == "EPSG:32651" and r.transform.a == 10
