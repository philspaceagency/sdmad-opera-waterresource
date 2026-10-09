"""Offline tests of the pure parts of acquisition (orbit choice, solar day) and of the pseudo-label builder."""
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("shapely")
from shapely.geometry import box, mapping  # noqa: E402

from angat_water import acquire, otsu, pseudo_labels as pl  # noqa: E402

BBOX = acquire.ANGAT_BBOX


def _item(orbit, state, when, footprint=(120.5, 14.0, 121.8, 15.5)):
    return SimpleNamespace(properties={"sat:relative_orbit": orbit, "sat:orbit_state": state},
                           geometry=mapping(box(*footprint)), bbox=list(footprint),
                           datetime=datetime(*when, tzinfo=timezone.utc))


def test_choose_orbit_picks_the_orbit_with_most_full_cover_days():
    a = [_item(61, "ascending", (2024, 1, d, 10)) for d in (1, 13, 25)]
    b = [_item(112, "descending", (2024, 1, d, 22)) for d in (4, 16)]
    partial = [_item(61, "ascending", (2024, 2, d, 10), footprint=(121.2, 14.0, 121.8, 15.5)) for d in (6, 18)]
    key, items = acquire.choose_orbit(a + b + partial, BBOX, min_cover=0.9)
    assert key == (61, "ascending") and len(items) == 5  # partial scenes belong to the orbit but are not counted


def test_choose_orbit_none_when_nothing_covers():
    far = [_item(61, "ascending", (2024, 1, 1, 10), footprint=(100, 0, 101, 1))]
    assert acquire.choose_orbit(far, BBOX) == (None, [])


def test_solar_day_rolls_over_for_evening_utc_pass_in_the_philippines():
    d = acquire.solar_dates([np.datetime64("2024-01-01T22:00:00")], 121.1)[0]
    assert str(d) == "2024-01-02"


def test_scene_ok_requires_agreement_and_a_real_water_body():
    good = {"mode": "both", "gap_db": 0.3, "water_px": 50000, "uncertain_px": 100}
    assert pl.scene_ok(good)
    assert not pl.scene_ok({**good, "mode": "edge_only"})
    assert not pl.scene_ok({**good, "gap_db": 2.5})
    assert not pl.scene_ok({**good, "water_px": 500})


def test_patches_ignore_uncertain_and_split_by_block():
    classes = np.zeros((512, 512), np.uint8)
    classes[100:300, 100:300] = otsu.CLASS_WATER
    classes[50:60, 50:60] = otsu.CLASS_UNCERTAIN
    y = pl.labels_from_classes(classes)
    assert y[55, 55] == pl.IGNORE and y[150, 150] == 1 and y[0, 0] == 0
    x = pl.normalise_s1(np.full((512, 512), -12.6, np.float32), np.full((512, 512), -20.3, np.float32))
    X, Y, corners = pl.extract_patches(x, y, size=128, stride=64, min_valid=0.95)
    assert X.shape[1:] == (2, 128, 128) and len(corners) == len(Y) > 0
    tr, va = pl.spatial_split(corners, (512, 512), size=128, val_frac=0.25)
    assert tr and va and not set(tr) & set(va)
    assert max(corners[i][1] + 128 for i in tr) <= 384 <= min(corners[i][1] for i in va) + 128


# ------------------------------------------------------------------ iter_scenes / iter_s1 wiring (I/O mocked)
def _patch_io(monkeypatch, items):
    calls = {}
    monkeypatch.setattr(acquire, "set_stac_cache", lambda d: calls.setdefault("cache", d))
    monkeypatch.setattr(acquire, "search_s1", lambda bbox, s, e: calls.setdefault("window", (s, e)) and items)
    monkeypatch.setattr(acquire, "make_geobox", lambda bbox, crs, res: SimpleNamespace(
        shape=(4, 5), transform=SimpleNamespace(a=res, c=300000.0, f=1650000.0), crs=crs))

    def load(its, gb, resampling, log=None):
        calls.setdefault("loaded", []).append(sorted(str(i.datetime.date()) for i in its))
        t = np.array([np.datetime64(i.datetime.replace(tzinfo=None), "ns") for i in its])
        return t, np.ones((len(its), 4, 5), np.float32), np.full((len(its), 4, 5), 0.1, np.float32)
    monkeypatch.setattr(acquire, "load_s1", load)
    return calls


def test_iter_scenes_streams_one_orbit_lazily_with_solar_days(monkeypatch):
    items = [_item(61, "ascending", (2019, 3, 5, 10)), _item(61, "ascending", (2019, 3, 17, 22)),
             _item(112, "descending", (2019, 3, 8, 22))]            # a second, less frequent orbit is not used
    calls = _patch_io(monkeypatch, items)
    gb, gen = acquire.iter_scenes(BBOX, "EPSG:32651", 10.0, "2019-03-01", "2019-03-31", cache_dir="c", log=lambda *_: None)
    assert "loaded" not in calls                                    # nothing is read until iterated
    out = list(gen)
    assert [str(d) for d, _, _ in out] == ["2019-03-05", "2019-03-18"]    # 22:00 UTC pass -> next local day
    assert out[0][1].shape == (4, 5) and calls["window"] == ("2019-03-01", "2019-03-31") and calls["cache"] == "c"
    assert all("2019-03-08" not in c for c in calls["loaded"])


def test_iter_scenes_without_a_covering_pass_raises(monkeypatch):
    _patch_io(monkeypatch, [])
    with pytest.raises(RuntimeError, match="no Sentinel-1 RTC pass"):
        acquire.iter_scenes(BBOX, "EPSG:32651", 10.0, "2019-03-01", "2019-03-31", log=lambda *_: None)


def test_dataset_iter_s1_clamps_the_window_to_jrc_coverage(monkeypatch):
    from angat_water import dataset, sites
    calls = _patch_io(monkeypatch, [_item(61, "ascending", (2021, 12, 20, 10))])
    cfg = dataset.DatasetConfig(start="2016-01-01", end="2023-12-31")
    grid, gen = dataset.iter_s1(sites.get("angat"), cfg, log=lambda *_: None)
    assert calls["window"] == ("2016-01-01", "2021-12-31")        # end clamped to JRC's last month, start kept
    assert grid.nx == 5 and grid.ny == 4 and grid.res == 10.0
    assert len(list(gen)) == 1


def test_iter_scenes_reports_unreadable_days_and_the_final_tally(monkeypatch):
    items = [_item(61, "ascending", (2019, 3, 5, 10)), _item(61, "ascending", (2019, 3, 17, 10))]
    _patch_io(monkeypatch, items)
    good = acquire.load_s1

    def flaky(its, gb, resampling, log=None):
        return None if its[0].datetime.day == 5 else good(its, gb, resampling, log)
    monkeypatch.setattr(acquire, "load_s1", flaky)
    logs = []
    _, gen = acquire.iter_scenes(BBOX, "EPSG:32651", 10.0, "2019-03-01", "2019-03-31", log=logs.append)
    assert len(list(gen)) == 1
    assert any("2019-03-05" in m and "skipped" in m for m in logs)
    assert any("read 1 scene(s) from 2 day(s), 1 day(s) unreadable" in m for m in logs)


def test_iter_scenes_resumes_by_not_reading_days_already_done(monkeypatch):
    items = [_item(61, "ascending", (2019, 3, 5, 10)), _item(61, "ascending", (2019, 3, 17, 22))]
    calls = _patch_io(monkeypatch, items)
    logs = []
    _, gen = acquire.iter_scenes(BBOX, "EPSG:32651", 10.0, "2019-03-01", "2019-03-31", log=logs.append,
                                 skip_days={"2019-03-05"})
    out = list(gen)
    assert [str(d) for d, _, _ in out] == ["2019-03-18"]
    assert calls["loaded"] == [["2019-03-17"]]                      # the finished day was never read
    assert any("1 already done (resumed)" in m for m in logs)
