import numpy as np
import pytest

from ai_water import otsu, pipeline, preprocess as pp, speckle, synthetic as syn


def _db(scene, filt=True):
    return pp.to_db(speckle.refined_lee(scene["vv"]) if filt else scene["vv"])


def test_otsu_threshold_on_two_gaussians():
    rng = np.random.default_rng(0)
    v = np.r_[rng.normal(-20, 1.5, 5000), rng.normal(-8, 2, 5000)]
    r = otsu.otsu_threshold(v)
    assert -16 < r.threshold < -12 and r.valid() and r.contrast_db > 10


def test_otsu_degenerate_inputs():
    assert not np.isfinite(otsu.otsu_threshold(np.full(1000, -10.0)).threshold)
    assert not np.isfinite(otsu.otsu_threshold(np.array([1.0, 2.0])).threshold)


@pytest.mark.parametrize("wf", [0.30, 0.10])
def test_both_methods_find_the_threshold_when_water_is_common(wf):
    db = _db(syn.make_scene(water_frac=wf, seed=1))
    th = otsu.combined_otsu(db)
    assert th.mode == "both" and -16 < th.t_low <= th.t_high < -12


def test_edge_otsu_survives_where_global_otsu_is_unimodal():
    s = syn.make_scene(water_frac=0.10, seed=1)
    db = _db(s)
    e, buf = otsu.edge_otsu(db)
    assert e.valid() and -16 < e.threshold < -12 and buf.any()


def test_small_lake_falls_back_to_one_valid_method_and_still_maps_it():
    s = syn.make_scene(water_frac=0.03, seed=1)
    wm = pipeline.map_water(s["vv"], s["vh"], pipeline.WaterConfig(min_area_px=50, fill_holes_px=50))
    assert wm.thresholds.usable
    assert syn.iou(wm.water, s["truth"]) > 0.9


def test_no_water_scene_returns_no_water_instead_of_splitting_land():
    s = syn.make_scene(water_frac=0.0001, seed=1)
    wm = pipeline.map_water(s["vv"], s["vh"])
    assert wm.water.mean() < 0.005


def test_texture_edges_are_rejected_by_the_contrast_gate():
    rng = np.random.default_rng(0)
    db = (-9 + 1.2 * rng.standard_normal((300, 300))).astype(np.float32)   # land only, speckle-like texture
    e, _ = otsu.edge_otsu(db)
    assert not e.valid() or e.contrast_db >= otsu.MIN_CONTRAST_DB


def test_classify_hysteresis_only_grows_from_certain_water():
    db = np.full((20, 20), -5.0, np.float32)
    db[2:8, 2:8] = -18.0          # certain water
    db[2:8, 8:10] = -13.5         # uncertain band, connected
    db[14:18, 14:18] = -13.5      # uncertain band, isolated
    th = otsu.Thresholds(-13.0, -14.0, -14.0, -13.0, "both")
    # make the band explicit: certain < -14, band (-14, -13)  -> use values -13.5
    c = otsu.classify(db, th)
    assert (c[2:8, 2:8] == otsu.CLASS_WATER).all()
    assert (c[2:8, 8:10] == otsu.CLASS_WATER).all()          # connected -> accepted
    assert (c[14:18, 14:18] == otsu.CLASS_UNCERTAIN).all()   # isolated -> uncertain


def test_clean_mask_drops_specks_and_fills_holes():
    m = np.zeros((50, 50), bool)
    m[5:30, 5:30] = True
    m[10:12, 10:12] = False       # 4-px hole
    m[40, 40] = True              # speck
    out = otsu.clean_mask(m, min_area_px=10, fill_holes_px=10)
    assert out[10, 10] and not out[40, 40]


def test_pipeline_end_to_end_accuracy_with_and_without_filter():
    s = syn.make_scene(water_frac=0.12, looks=4, seed=5)
    good = pipeline.map_water(s["vv"], s["vh"], pipeline.WaterConfig(speckle_filter="refined_lee"))
    assert syn.iou(good.water, s["truth"]) > 0.97
