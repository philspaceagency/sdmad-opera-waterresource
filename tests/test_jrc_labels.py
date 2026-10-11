"""JRC date matching and the Earth Engine request plumbing (all offline)."""
import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ai_water import jrc, synthetic as syn

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


# ------------------------------------------------------------------ Earth Engine request plumbing
def fake_ee(monthly_bands=("water",), occurrence_bands=("occurrence",), compute=None):
    """A mocked `ee` whose band names behave like the real assets (MonthlyHistory: `water`; GlobalSurfaceWater: `occurrence`)."""
    ee = MagicMock()
    ee.ImageCollection.return_value.first.return_value.bandNames.return_value.getInfo.return_value = list(monthly_bands)
    ee.Image.return_value.bandNames.return_value.getInfo.return_value = list(occurrence_bands)
    if compute is not None:
        ee.data.computePixels.side_effect = compute
    return ee


def test_gee_blocks_cover_the_grid_with_correct_offsets():
    sent = []

    def compute(req):
        g = req["grid"]
        sent.append((g["dimensions"]["height"], g["affineTransform"]["translateY"], g["crsCode"]))
        return {"water": np.full((g["dimensions"]["height"], g["dimensions"]["width"]), 2, np.uint8)}
    ee = fake_ee(compute=compute)
    g = jrc.Grid("EPSG:32651", 10.0, 300000.0, 1650000.0, 1000, 1000)
    arr = jrc.GeeJrc(ee_module=ee, max_request_mb=0.4).month("2019-07-15", g)     # 400 rows of 1000 B per block
    assert arr.shape == (1000, 1000) and (arr == 2).all()
    assert [h for h, _, _ in sent] == [400, 400, 200]
    assert [y for _, y, _ in sent] == [1650000.0, 1650000.0 - 4000, 1650000.0 - 8000]
    assert {c for _, _, c in sent} == {"EPSG:32651"}
    ee.ImageCollection.assert_called_with(jrc.MONTHLY_ASSET)
    ee.ImageCollection.return_value.filterDate.assert_called_with("2019-07-01", "2019-08-01")   # the S1 month only
    ee.ImageCollection.return_value.filterDate.return_value.select.assert_called_with("water")  # the real band name


def test_gee_monthly_band_is_read_from_the_asset_and_waterclass_is_a_fallback():
    seen = []

    def compute(req):
        seen.append(req["expression"])
        return {"waterClass": np.ones((1, 3), np.uint8)}
    ee = fake_ee(monthly_bands=("waterClass",), compute=compute)
    g = jrc.Grid("EPSG:32651", 10.0, 0.0, 10.0, 3, 1)
    jrc.GeeJrc(ee_module=ee).month("2019-07-01", g)
    ee.ImageCollection.return_value.filterDate.return_value.select.assert_called_with("waterClass")


def test_gee_unknown_bands_fail_with_the_list_of_real_ones():
    ee = fake_ee(monthly_bands=("something_else", "other"))
    g = jrc.Grid("EPSG:32651", 10.0, 0.0, 10.0, 3, 1)
    with pytest.raises(RuntimeError, match=r"has bands \['something_else', 'other'\].*expected"):
        jrc.GeeJrc(ee_module=ee).month("2019-07-01", g)
    ee.data.computePixels.assert_not_called()


def test_gee_band_lookup_is_asked_once_per_asset():
    ee = fake_ee(compute=lambda req: {"water": np.ones((1, 3), np.uint8)})
    g = jrc.Grid("EPSG:32651", 10.0, 0.0, 10.0, 3, 1)
    src = jrc.GeeJrc(ee_module=ee)
    src.month("2019-05-01", g)
    src.month("2019-06-01", g)
    assert ee.ImageCollection.return_value.first.return_value.bandNames.return_value.getInfo.call_count == 1


def test_gee_occurrence_marks_unobserved_as_nan():
    ee = fake_ee(compute=lambda req: {"occurrence": np.array([[-1, 0, 100]], np.float32)})
    g = jrc.Grid("EPSG:32651", 10.0, 0.0, 10.0, 3, 1)
    occ = jrc.GeeJrc(ee_module=ee).occurrence(g)
    assert np.isnan(occ[0, 0]) and occ[0, 1] == 0 and occ[0, 2] == 100


def test_gee_transient_errors_are_retried_then_raised(monkeypatch):
    monkeypatch.setattr(jrc._time, "sleep", lambda s: None)
    ee = fake_ee()
    ee.data.computePixels.side_effect = RuntimeError("Too many concurrent requests: quota")
    g = jrc.Grid("EPSG:32651", 10.0, 0.0, 10.0, 3, 1)
    with pytest.raises(RuntimeError, match="failed after 3 tries"):
        jrc.GeeJrc(ee_module=ee, tries=3).month("2019-07-01", g)
    assert ee.data.computePixels.call_count == 3


def test_gee_rejected_requests_are_not_retried(monkeypatch):
    monkeypatch.setattr(jrc._time, "sleep", lambda s: pytest.fail("must not back off on a request EE rejects"))
    ee = fake_ee()
    ee.data.computePixels.side_effect = RuntimeError("Image.select: Band pattern 'x' did not match any bands. Available bands: [water]")
    g = jrc.Grid("EPSG:32651", 10.0, 0.0, 10.0, 3, 1)
    with pytest.raises(RuntimeError, match="rejected the request.*did not match any bands"):
        jrc.GeeJrc(ee_module=ee, tries=5).month("2019-07-01", g)
    assert ee.data.computePixels.call_count == 1


def test_transient_classifier():
    t = jrc._is_transient
    assert t(RuntimeError("Earth Engine memory capacity exceeded")) is False
    assert all(t(RuntimeError(m)) for m in ("HTTP 503 Service Unavailable", "Read timed out", "Too many requests", "quota exceeded",
                                            "Connection reset by peer", "429"))
    assert not t(RuntimeError("ImageCollection.mosaic: Error in map(ID=2018_01): Image.select: Band pattern 'waterClass' "
                              "did not match any bands. Available bands: [water]"))
