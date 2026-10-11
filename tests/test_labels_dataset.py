"""JRC + terrain labels, the dataset build, and the patch set (all offline; Planetary Computer / Earth Engine are stand-ins)."""
import json

import numpy as np
import pytest

from ai_water import dataset, features as ft, jrc, labels as lb, scene, sites, synthetic as syn

GRID = jrc.Grid("EPSG:32651", 10.0, 300000.0, 1650000.0, 384, 384)
SHAPE = (384, 384)


def scene_at(orbit="descending", seed=3):
    s = syn.make_terrain_scene(SHAPE, seed=seed, orbit=orbit)
    f = scene.prepare(s["vv"], s["vh"])
    return s, f, ft.terrain_from_dem(s["dem"], 10.0)


def test_labels_follow_jrc_and_leave_the_shoreline_to_the_model():
    s, f, t = scene_at()
    jr = syn.jrc_month_from_truth(s["truth"])
    lab, src, st = lb.jrc_labels(jr, f.vv_db, t.slope)
    m = lab != ft.IGNORE
    assert m.mean() > 0.9 and (lab[m] == s["truth"][m]).mean() > 0.995
    edge = np.abs(np.diff((jr == jrc.WATER).astype(int), axis=1)) > 0                 # JRC shoreline columns
    r, c = np.nonzero(edge)
    assert (lab[r, c] == ft.IGNORE).all() and (lab[r, np.clip(c + 1, 0, 383)] == ft.IGNORE).all()
    assert st["contrast_db"] > 8 and st["water_px"] > 5000


def test_radar_shadow_is_labelled_land_and_counted_as_a_hard_negative():
    """The point of the redesign: JRC says land on the shadowed flank, so the dark radar there teaches 'land'."""
    s, f, t = scene_at("descending")
    lab, src, st = lb.jrc_labels(syn.jrc_month_from_truth(s["truth"]), f.vv_db, t.slope)
    core = s["shadow"] & (lab != ft.IGNORE)
    assert core.sum() > 3000 and (lab[core] == 0).all()
    assert (src[core] == lb.SRC_HARD_NEGATIVE).mean() > 0.9 and st["hard_negative_px"] > 3000


def test_jrc_water_on_a_steep_slope_is_not_trusted():
    s, f, t = scene_at()
    jr = syn.jrc_month_from_truth(s["truth"])
    jr[s["shadow"]] = jrc.WATER                                      # e.g. Landsat topographic shadow read as water
    lab, src, st = lb.jrc_labels(jr, f.vv_db, t.slope, edge_px=0)
    steep = s["shadow"] & (t.slope > 15)
    assert steep.sum() > 1000 and (lab[steep] == ft.IGNORE).all() and (src[steep] == lb.SRC_TERRAIN).all()
    assert st["terrain_ignored_px"] >= int(steep.sum())


def test_jrc_and_radar_nodata_are_never_labelled():
    s, f, t = scene_at()
    jr = syn.jrc_month_from_truth(s["truth"])
    cloud = np.zeros(SHAPE, bool)
    cloud[:60] = True
    jr[cloud] = jrc.NODATA
    vv = f.vv_db.copy()
    vv[:, :20] = np.nan
    lab, _, st = lb.jrc_labels(jr, vv, t.slope)
    assert (lab[cloud] == ft.IGNORE).all() and (lab[:, :20] == ft.IGNORE).all()
    assert st["jrc_nodata_frac"] == pytest.approx(cloud.mean())


def test_grid_mismatch_is_an_error():
    with pytest.raises(ValueError, match="grid mismatch"):
        lb.jrc_labels(np.ones((10, 10), np.uint8), np.zeros((10, 11), np.float32), np.zeros((10, 10), np.float32))


def test_a_jrc_month_that_does_not_fit_the_pass_has_no_contrast():
    s, f, t = scene_at()
    right = lb.jrc_labels(syn.jrc_month_from_truth(s["truth"]), f.vv_db, t.slope)[2]
    wrong = lb.jrc_labels(syn.jrc_month_from_truth(np.roll(s["truth"], 150, axis=1)), f.vv_db, t.slope)[2]
    assert lb.scene_quality(right) and not lb.scene_quality(wrong)
    assert not lb.scene_quality({**right, "water_px": 100}) and not lb.scene_quality({**right, "jrc_nodata_frac": 0.7})


# ------------------------------------------------------------------ dataset
def stream(items):
    for day, orbit, seed in items:
        s = syn.make_terrain_scene(SHAPE, seed=3, orbit=orbit) if seed is None else syn.make_scene(SHAPE, 0.2, seed=seed)
        yield day, orbit, s["vv"], s["vh"]


def build(tmp_path, items, jrc_truth):
    s0 = syn.make_terrain_scene(SHAPE, seed=3)
    cfg = dataset.DatasetConfig(out_dir=str(tmp_path), cache_dir=str(tmp_path / "cache"))
    terr = syn.SyntheticTerrain(s0["dem"])
    site = sites.Site("testsite", (121.0, 15.0, 121.1, 15.1))
    metas = dataset.build_site(site, cfg, stream(items), GRID, syn.SyntheticJrc(jrc_truth), terr, log=lambda *_: None)
    return cfg, site, metas, terr


def test_build_site_uses_both_orbits_rejects_bad_months_and_loads_patches(tmp_path):
    truth = syn.make_terrain_scene(SHAPE, seed=3)["truth"]
    items = [("2019-05-12", "descending", None), ("2019-05-18", "ascending", None),
             ("2019-06-05", "descending", None),             # JRC June shows a lake somewhere else  -> low contrast
             ("2019-08-02", "descending", None)]             # no JRC data in August                  -> no_jrc_month
    cfg, site, metas, terr = build(tmp_path, items, {"2019-05": truth, "2019-06": np.roll(truth, 150, axis=1)})
    assert [m["usable"] for m in metas] == [True, True, False, False]
    # (the shifted 'lake' falls on steep hillside, where JRC water is not trusted: either gate may be what stops it)
    assert metas[2]["reject_reason"] in ("low_contrast", "few_water_px") and metas[3]["reject_reason"] == "no_jrc_month"
    d = tmp_path / "testsite"
    assert (d / "2019-05-12_descending.npz").exists() and (d / "2019-05-18_ascending.npz").exists() and (d / "terrain.npz").exists()
    assert not (d / "2019-06-05_descending.npz").exists()
    z = np.load(d / "2019-05-12_descending.npz")
    assert set(z.files) == {"vv_db", "vh_db", "sdwi", "labels", "source"}
    man = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert [m["orbit"] for m in man] == ["descending", "ascending", "descending", "descending"]
    assert dataset.done_scenes(str(tmp_path), "testsite") == {"2019-05-12:descending", "2019-05-18:ascending",
                                                              "2019-06-05:descending", "2019-08-02:descending"}
    assert dataset.manifest_table(str(tmp_path))["orbit"].tolist() == ["descending", "ascending", "descending", "descending"]

    ps = dataset.load_patches(str(tmp_path), size=128, stride=64, keep_empty=1.0)
    assert len(ps) == len(ps.metas) > 0 and ps.s1.shape[1:] == (3, 128, 128)
    X, Y = ps.batch(range(len(ps)))
    assert X.shape == (len(ps), ft.N_CHANNELS, 128, 128) and Y.shape == (len(ps), 128, 128)
    assert (Y == 1).any() and (Y == 0).any() and np.isfinite(X).all()
    # the range-slope channel flips between the two geometries for the same ground
    i_d = next(i for i, m in enumerate(ps.metas) if m["orbit"] == "descending" and m["corner"] == (128, 128))
    i_a = next(i for i, m in enumerate(ps.metas) if m["orbit"] == "ascending" and m["corner"] == (128, 128))
    assert np.corrcoef(X[i_d, 7].ravel(), X[i_a, 7].ravel())[0, 1] < -0.5
    assert terr.calls == 1


def test_terrain_is_fetched_once_per_grid_and_reused_from_the_cache(tmp_path):
    truth = syn.make_terrain_scene(SHAPE, seed=3)["truth"]
    cfg, site, _, terr = build(tmp_path, [("2019-05-12", "descending", None)], {"2019-05": truth})
    dataset.build_site(site, cfg, stream([("2019-05-18", "ascending", None)]), GRID, syn.SyntheticJrc({"2019-05": truth}), terr,
                       log=lambda *_: None)
    assert terr.calls == 1


def test_patches_with_hard_negatives_survive_the_empty_patch_filter(tmp_path):
    truth = syn.make_terrain_scene(SHAPE, seed=3)["truth"]
    build(tmp_path, [("2019-05-12", "descending", None)], {"2019-05": truth})
    strict = dataset.load_patches(str(tmp_path), size=64, stride=64, min_water=0.9, min_hard=0.01, keep_empty=0.0)
    assert len(strict) > 0 and all(m["hard_frac"] >= 0.01 or m["water_frac"] >= 0.9 for m in strict.metas)
    blind = dataset.load_patches(str(tmp_path), size=64, stride=64, min_water=0.9, min_hard=2.0, keep_empty=0.0)
    assert len(blind) < len(strict)


def test_split_by_holdout_site_and_by_block():
    metas = [{"site": s, "corner": (0, c), "shape": [384, 384]} for s in ("angat", "magat") for c in (0, 128, 256)]
    tr, va = dataset.split_patches(metas, 128, holdout_sites=["Magat"])
    assert {metas[i]["site"] for i in va} == {"magat"} and {metas[i]["site"] for i in tr} == {"angat"}
    tr, va = dataset.split_patches(metas, 128)
    assert not set(tr) & set(va) and all(metas[i]["corner"][1] + 128 <= 288 for i in tr)


def test_manifest_table_explains_an_empty_or_fully_rejected_manifest(tmp_path):
    with pytest.raises(RuntimeError, match="does not exist"):
        dataset.manifest_table(str(tmp_path))
    (tmp_path / "manifest.jsonl").write_text("")
    with pytest.raises(RuntimeError, match="is empty"):
        dataset.manifest_table(str(tmp_path))
    (tmp_path / "manifest.jsonl").write_text(json.dumps({"site": "a", "date": "2019-05-01", "orbit": "ascending", "usable": False,
                                                         "reject_reason": "low_contrast"}) + "\n")
    with pytest.raises(RuntimeError, match="low_contrast"):
        dataset.manifest_table(str(tmp_path))
