"""JRC date matching, tiered labels, dataset build/load and the Earth Engine request plumbing (all offline)."""
import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from angat_water import dataset, jrc, labels as lb, otsu, pipeline, pseudo_labels as pl, sites, synthetic as syn

S = 3  # Landsat 30 m = 3 Sentinel-1 pixels of 10 m


def fake_jrc_month(truth, cloud=None):
    """What JRC would show for `truth`: majority vote on 3x3 blocks (30 m), 2 = water, 1 = not water, 0 = no data."""
    h, w = truth.shape
    hb, wb = h // S, w // S
    blk = truth[:hb * S, :wb * S].reshape(hb, S, wb, S).mean((1, 3)) >= 0.5
    cls = np.where(np.kron(blk, np.ones((S, S), bool)), jrc.WATER, jrc.NOT_WATER).astype(np.uint8)
    out = np.full(truth.shape, jrc.NOT_WATER, np.uint8)
    out[:hb * S, :wb * S] = cls
    if cloud is not None:
        out[cloud] = jrc.NODATA
    return out


class FakeJrc:
    def __init__(self, truth_by_month, cloud=None):
        self.truth_by_month, self.cloud, self.calls = truth_by_month, cloud, []

    def month(self, month, grid):
        self.calls.append(pd.Timestamp(month).strftime("%Y-%m"))
        t = self.truth_by_month.get(pd.Timestamp(month).strftime("%Y-%m"))
        if t is None:
            return np.zeros((grid.ny, grid.nx), np.uint8)
        return fake_jrc_month(t, self.cloud)

    def occurrence(self, grid):
        return np.zeros((grid.ny, grid.nx), np.float32)


GRID = jrc.Grid("EPSG:32651", 10.0, 300000.0, 1650000.0, 384, 384)


def _scene(seed=0, water=0.15):
    s = syn.make_scene(shape=(384, 384), water_frac=water, looks=4, seed=seed)
    return s


# ------------------------------------------------------------------ date matching
def test_matched_month_uses_the_calendar_month_of_the_pass_and_caches():
    t = _scene()["truth"]
    src = FakeJrc({"2019-07": t})
    cache = {}
    m = jrc.matched_month(src, GRID, "2019-07-23", cache=cache)
    assert m.month == pd.Timestamp("2019-07-01") and m.offset_months == 0 and m.valid_frac == 1.0
    jrc.matched_month(src, GRID, "2019-07-05", cache=cache)
    assert src.calls == ["2019-07"]                      # second S1 date of the month reuses the fetched month


def test_matched_month_never_borrows_a_neighbour_unless_allowed():
    t = _scene()["truth"]
    src = FakeJrc({"2019-08": t})                        # July is empty (cloud), August is fine
    assert jrc.matched_month(src, GRID, "2019-07-23") is None
    m = jrc.matched_month(src, GRID, "2019-07-23", max_offset=1)
    assert m.offset_months == 1 and m.month == pd.Timestamp("2019-08-01")


def test_matched_month_outside_jrc_coverage_is_none():
    src = FakeJrc({})
    assert jrc.matched_month(src, GRID, "2023-05-10") is None
    assert jrc.matched_month(src, GRID, "1983-01-10") is None
    assert src.calls == []                               # no pointless Earth Engine request


# ------------------------------------------------------------------ labels
def _label_purity(lab, truth):
    m = lab != pl.IGNORE
    return float((lab[m] == truth[m]).mean()), int(m.sum())


def test_labels_are_nearly_pure_and_more_pure_than_otsu_alone_when_otsu_has_shadow():
    s = _scene(seed=2)
    truth = s["truth"]
    vv = s["vv"].copy()
    shadow = np.zeros_like(truth)
    shadow[300:360, 20:120] = True                       # steep-slope shadow on land: dark in S1, land for JRC
    shadow &= ~truth
    vv[shadow] = 10 ** (-21 / 10) * np.random.default_rng(0).gamma(4, .25, int(shadow.sum()))
    wm = pipeline.map_water(vv, s["vh"], pipeline.WaterConfig())
    assert (wm.water & shadow).sum() > 1000              # Otsu is fooled
    otsu_only = pl.labels_from_classes(wm.classes)
    lab, src, st = lb.jrc_otsu_labels(wm.classes, fake_jrc_month(truth))
    p_otsu, _ = _label_purity(otsu_only, truth)
    p_new, n = _label_purity(lab, truth)
    assert p_new > p_otsu and p_new > 0.995 and n > 0.9 * truth.size
    assert (lab[shadow] != 1).all()                      # no shadow pixel is ever taught as water
    assert st["conflict_jrc_land_otsu_water_px"] > 1000  # and the conflict is reported


def test_never_water_negatives_turn_shadow_into_land_labels():
    s = _scene(seed=2)
    truth = s["truth"]
    classes = np.where(truth, otsu.CLASS_WATER, otsu.CLASS_LAND).astype(np.uint8)
    shadow = np.zeros_like(truth)
    shadow[300:340, 20:100] = True
    shadow &= ~truth
    classes[shadow] = otsu.CLASS_WATER
    occ = np.zeros(truth.shape, np.float32)
    occ[truth] = 90
    off, _, _ = lb.jrc_otsu_labels(classes, fake_jrc_month(truth), occ, never_water_negatives=False)
    on, src, st = lb.jrc_otsu_labels(classes, fake_jrc_month(truth), occ, never_water_negatives=True)
    assert (off[shadow] == pl.IGNORE).all()
    assert (on[shadow] == 0).all() and (src[shadow] == lb.SRC_NEVER_WATER).all() and st["never_water_px"] > 0


def test_uncertain_band_is_decided_by_jrc():
    truth = np.zeros((60, 60), bool)
    truth[:, :30] = True
    classes = np.full(truth.shape, otsu.CLASS_UNCERTAIN, np.uint8)
    lab, src, _ = lb.jrc_otsu_labels(classes, fake_jrc_month(truth), edge_px=0)
    assert (lab[:, :30] == 1).all() and (lab[:, 30:] == 0).all() and (src == lb.SRC_JRC_RESOLVED).all()
    lab2, _, _ = lb.jrc_otsu_labels(classes, fake_jrc_month(truth), edge_px=0, resolve_uncertain=False)
    assert (lab2 == pl.IGNORE).all()


def test_shoreline_band_takes_the_10m_otsu_label_not_the_30m_jrc_one():
    truth = np.zeros((60, 60), bool)
    truth[:, :31] = True                                 # shoreline NOT on a 30 m block edge
    jr = fake_jrc_month(truth)                           # JRC puts it at column 30 or 33
    classes = np.where(truth, otsu.CLASS_WATER, otsu.CLASS_LAND).astype(np.uint8)
    lab, src, _ = lb.jrc_otsu_labels(classes, jr, edge_px=3)
    assert (lab[:, 30] == 1).all() and (lab[:, 31] == 0).all()           # follows Otsu at 10 m
    assert (src[:, 27:33] == lb.SRC_OTSU_EDGE).all() and (src[:, 33] == lb.SRC_AGREE).all()
    ign, _, _ = lb.jrc_otsu_labels(classes, jr, edge_px=3, edge_policy="ignore")
    assert (ign[:, 29:33] == pl.IGNORE).all()


def test_jrc_nodata_is_never_labelled():
    s = _scene(seed=1)
    cloud = np.zeros_like(s["truth"])
    cloud[:, :100] = True
    classes = np.where(s["truth"], otsu.CLASS_WATER, otsu.CLASS_LAND).astype(np.uint8)
    lab, _, st = lb.jrc_otsu_labels(classes, fake_jrc_month(s["truth"], cloud))
    assert (lab[cloud] == pl.IGNORE).all() and st["jrc_nodata_frac"] == pytest.approx(cloud.mean())


def test_a_jrc_month_from_the_wrong_water_level_drops_agreement():
    truth = _scene(seed=3, water=0.25)["truth"]
    lower = _scene(seed=3, water=0.10)["truth"]          # reservoir much lower in the JRC month than on the S1 date
    classes = np.where(truth, otsu.CLASS_WATER, otsu.CLASS_LAND).astype(np.uint8)
    _, _, good = lb.jrc_otsu_labels(classes, fake_jrc_month(truth))
    _, _, bad = lb.jrc_otsu_labels(classes, fake_jrc_month(lower))
    assert good["agreement"] > 0.99 and bad["agreement"] < 0.9
    assert lb.scene_quality(good) and not lb.scene_quality(bad)


# ------------------------------------------------------------------ dataset
def _scene_stream(truths_by_day, seed0=0):
    for k, (day, truth_water) in enumerate(truths_by_day):
        s = syn.make_scene(shape=(384, 384), water_frac=truth_water, looks=4, seed=seed0 + k)
        yield day, s["vv"], s["vh"], s["truth"]


def test_build_site_writes_usable_scenes_rejects_mismatches_and_loads_patches(tmp_path):
    cfg = dataset.DatasetConfig(out_dir=str(tmp_path), water=pipeline.WaterConfig(min_area_px=50, fill_holes_px=50))
    good = syn.make_scene(shape=(384, 384), water_frac=0.2, looks=4, seed=11)
    wrong = syn.make_scene(shape=(384, 384), water_frac=0.2, looks=4, seed=12)
    other = syn.make_scene(shape=(384, 384), water_frac=0.06, looks=4, seed=13)   # JRC month shows another extent
    src = FakeJrc({"2019-05": good["truth"], "2019-06": other["truth"], "2019-07": other["truth"]})
    scenes = [("2019-05-12", good["vv"], good["vh"]),        # JRC May = this scene          -> usable
              ("2019-06-05", wrong["vv"], wrong["vh"]),      # JRC June is another level      -> low agreement
              ("2019-08-02", good["vv"], good["vh"])]        # no JRC data in August          -> rejected
    site = sites.Site("testsite", (121.0, 15.0, 121.1, 15.1))
    metas = dataset.build_site(site, cfg, iter(scenes), GRID, src, log=lambda *_: None)
    assert [m["usable"] for m in metas] == [True, False, False]
    assert metas[1]["reject_reason"] == "low_agreement" and metas[2]["reject_reason"] == "no_jrc_month"
    assert (tmp_path / "testsite" / "2019-05-12.npz").exists() and not (tmp_path / "testsite" / "2019-06-05.npz").exists()
    man = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert len(man) == 3 and man[0]["jrc_month"] == "2019-05" and man[0]["month_offset"] == 0

    X, Y, pm = dataset.load_patches(str(tmp_path), size=128, stride=64, keep_empty=1.0)
    assert X.shape[1:] == (2, 128, 128) and len(pm) == len(Y) > 0 and {m["site"] for m in pm} == {"testsite"}
    pur, _ = _label_purity(Y, np.zeros(Y.shape, np.uint8))          # sanity: labels exist and contain both classes
    assert (Y == 1).any() and (Y == 0).any()


def test_split_by_holdout_site_and_by_block():
    metas = [{"site": s, "corner": (0, c), "shape": [384, 384]} for s in ("angat", "magat") for c in (0, 128, 256)]
    tr, va = dataset.split_patches(metas, 128, holdout_sites=["Magat"])
    assert {metas[i]["site"] for i in va} == {"magat"} and {metas[i]["site"] for i in tr} == {"angat"}
    tr, va = dataset.split_patches(metas, 128)
    assert not set(tr) & set(va) and all(metas[i]["corner"][1] + 128 <= 288 for i in tr)


# ------------------------------------------------------------------ Earth Engine request plumbing
def test_gee_blocks_cover_the_grid_with_correct_offsets():
    ee = MagicMock()
    sent = []

    def compute(req):
        g = req["grid"]
        sent.append((g["dimensions"]["height"], g["affineTransform"]["translateY"], g["crsCode"]))
        return {"waterClass": np.full((g["dimensions"]["height"], g["dimensions"]["width"]), 2, np.uint8)}
    ee.data.computePixels.side_effect = compute
    g = jrc.Grid("EPSG:32651", 10.0, 300000.0, 1650000.0, 1000, 1000)
    arr = jrc.GeeJrc(ee_module=ee, max_request_mb=0.4).month("2019-07-15", g)     # 400 rows of 1000 B per block
    assert arr.shape == (1000, 1000) and (arr == 2).all()
    assert [h for h, _, _ in sent] == [400, 400, 200]
    assert [y for _, y, _ in sent] == [1650000.0, 1650000.0 - 4000, 1650000.0 - 8000]
    assert {c for _, _, c in sent} == {"EPSG:32651"}
    ee.ImageCollection.assert_called_with(jrc.MONTHLY_ASSET)
    ee.ImageCollection.return_value.filterDate.assert_called_with("2019-07-01", "2019-08-01")   # the S1 month only


def test_gee_occurrence_marks_unobserved_as_nan():
    ee = MagicMock()
    ee.data.computePixels.side_effect = lambda req: {"occurrence": np.array([[-1, 0, 100]], np.float32)}
    g = jrc.Grid("EPSG:32651", 10.0, 0.0, 10.0, 3, 1)
    occ = jrc.GeeJrc(ee_module=ee).occurrence(g)
    assert np.isnan(occ[0, 0]) and occ[0, 1] == 0 and occ[0, 2] == 100


def test_gee_retries_then_raises():
    ee = MagicMock()
    ee.data.computePixels.side_effect = RuntimeError("quota")
    g = jrc.Grid("EPSG:32651", 10.0, 0.0, 10.0, 3, 1)
    with pytest.raises(RuntimeError, match="failed after 2 tries"):
        jrc.GeeJrc(ee_module=ee, tries=2).month("2019-07-01", g)


def test_site_registry():
    assert sites.get("Angat").name == "angat" and sites.get("magat").bbox[0] < sites.get("magat").bbox[2]
    with pytest.raises(KeyError):
        sites.get("nowhere")


# ------------------------------------------------------------------ failure messages (empty / all-rejected manifests)
def test_manifest_table_explains_an_empty_or_missing_manifest(tmp_path):
    with pytest.raises(RuntimeError, match="does not exist"):
        dataset.manifest_table(str(tmp_path))
    (tmp_path / "manifest.jsonl").write_text("")
    with pytest.raises(RuntimeError, match="is empty.*no scenes"):
        dataset.manifest_table(str(tmp_path))


def test_manifest_table_explains_when_every_scene_was_rejected(tmp_path):
    rows = [{"site": "angat", "date": "2019-05-12", "usable": False, "reject_reason": "low_agreement",
             "labels": {"agreement": 0.7, "water_px": 10, "jrc_nodata_frac": 0.0}},
            {"site": "angat", "date": "2019-06-12", "usable": False, "reject_reason": "no_jrc_month"}]
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(RuntimeError, match=r"all 2 scenes were rejected.*low_agreement.*max_month_offset"):
        dataset.manifest_table(str(tmp_path))


def test_manifest_table_has_the_columns_the_notebook_groups_by(tmp_path):
    cfg = dataset.DatasetConfig(out_dir=str(tmp_path), water=pipeline.WaterConfig(min_area_px=50, fill_holes_px=50))
    sc = syn.make_scene(shape=(384, 384), water_frac=0.2, looks=4, seed=11)
    dataset.build_site(sites.Site("testsite", (121.0, 15.0, 121.1, 15.1)), cfg, iter([("2019-05-12", sc["vv"], sc["vh"])]),
                       GRID, FakeJrc({"2019-05": sc["truth"]}), log=lambda *_: None)
    tbl = dataset.manifest_table(str(tmp_path))
    assert list(tbl.groupby("site")["usable"].sum()) == [1] and {"agreement", "water_km2", "reject_reason"} <= set(tbl)


def test_build_site_warns_when_no_scene_was_read(tmp_path):
    logs = []
    cfg = dataset.DatasetConfig(out_dir=str(tmp_path))
    out = dataset.build_site(sites.Site("testsite", (121.0, 15.0, 121.1, 15.1)), cfg, iter([]), GRID, FakeJrc({}), log=logs.append)
    assert out == [] and any("WARNING" in m and "no Sentinel-1 scene was read" in m for m in logs)
