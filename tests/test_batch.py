"""AOI clipping and multi-site batch processing (Planetary Computer replaced by stand-ins)."""
import csv
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("pyproj")
from affine import Affine  # noqa: E402
from pyproj import Transformer  # noqa: E402
from shapely.geometry import box as sbox  # noqa: E402

from ai_water import aoi, batch, infer, sites, synthetic as syn  # noqa: E402

X0, Y1, RES, N = 300000.0, 1650000.0, 10.0, 256


def geobox():
    return SimpleNamespace(transform=Affine(RES, 0, X0, 0, -RES, Y1), crs="EPSG:32651", shape=(N, N))


def lonlat_box(x0, y0, x1, y1):
    t = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
    (w, s), (e, n) = t.transform(x0, y0), t.transform(x1, y1)
    return w, s, e, n


def site(name, geometry=None, kind="reservoir"):
    bb = lonlat_box(X0, Y1 - N * RES, X0 + N * RES, Y1)
    return sites.Site(name, bb, "EPSG:32651", kind, label=name.title(), geometry=geometry)


# ------------------------------------------------------------------ aoi
def test_aoi_mask_clips_to_the_outline_and_is_none_without_one():
    assert aoi.aoi_mask(site("a"), geobox()) is None
    half = sbox(*lonlat_box(X0, Y1 - N * RES, X0 + N * RES / 2, Y1))        # the western half of the grid
    m = aoi.aoi_mask(site("a", half), geobox())
    # (a UTM rectangle is slightly rotated in lon/lat, so allow a sliver at the edge of the outline)
    assert m.shape == (N, N) and m[:, : N // 2 - 6].mean() > 0.99 and m[:, N // 2 + 6:].sum() == 0
    assert abs(m.mean() - 0.5) < 0.03


# ------------------------------------------------------------------ a stand-in for Planetary Computer
class FakeS1:
    """Per-site list of (day, scene dict). `reads` records every pass actually read; sites in `broken` raise."""

    def __init__(self, plan, broken=()):
        self.plan, self.broken, self.reads = plan, set(broken), []

    def install(self, monkeypatch, limit=None):
        def iter_window(st, start, end, water_cfg=None, cache_dir=None, log=print, skip_days=None):
            if st.name in self.broken:
                raise RuntimeError(f"no Sentinel-1 RTC pass covers {st.name}")
            items = self.plan[st.name][:limit] if limit else self.plan[st.name]

            def gen():
                for day, sc in items:
                    if skip_days and day in skip_days:
                        continue
                    self.reads.append((st.name, day))
                    yield day, sc["vv"], sc["vh"]
            return geobox(), gen()
        monkeypatch.setattr(infer, "iter_window", iter_window)


def scenes(seed, fracs=(0.2, 0.2, 0.12)):
    sc = [syn.make_scene((N, N), water_frac=f, looks=4, seed=seed + (0 if i < 2 else 7)) for i, f in enumerate(fracs)]
    return [("2024-01-10", sc[0]), ("2024-01-22", sc[1]), ("2024-02-03", sc[2])]


def cfg(tmp_path, **kw):
    return batch.MapConfig(start="2024-01-01", end="2024-03-01", out_dir=str(tmp_path), cache_dir=str(tmp_path / "c"), **kw)


# ------------------------------------------------------------------ map_sites
def test_two_sites_are_processed_and_areas_are_recorded(tmp_path, monkeypatch):
    fake = FakeS1({"alpha": scenes(1), "beta": scenes(30)})
    fake.install(monkeypatch)
    res = batch.map_sites([site("alpha"), site("beta", kind="lake")], None, cfg(tmp_path), log=lambda *_: None)
    assert res.ok and len(res.areas) == 6 and set(res.areas["site"]) == {"alpha", "beta"}
    a = res.areas
    assert (a["fused_km2"] > 0).all() and (a["valid_km2"] > 6.0).all()                    # 256 x 256 px x 100 m2 = 6.55 km2
    assert a.loc[a.site == "beta", "kind"].iloc[0] == "lake" and a.loc[a.site == "alpha", "label"].iloc[0] == "Alpha"
    assert (tmp_path / "alpha" / "2024-01-10_fused_water.tif").exists() and (tmp_path / "beta" / "frequency.tif").exists()
    assert not (tmp_path / "alpha" / "2024-01-10_prob.tif").exists()                      # only what `write` asks for


def test_area_is_counted_inside_the_outline_only(tmp_path, monkeypatch):
    FakeS1({"alpha": scenes(1)}).install(monkeypatch)
    full = batch.map_sites([site("alpha")], None, cfg(tmp_path / "full"), log=lambda *_: None).areas
    half = sbox(*lonlat_box(X0, Y1 - N * RES, X0 + N * RES / 2, Y1))
    clip = batch.map_sites([site("alpha", half)], None, cfg(tmp_path / "clip"), log=lambda *_: None).areas
    assert (clip["fused_km2"] < full["fused_km2"]).all() and (clip["valid_km2"] < 0.55 * full["valid_km2"]).all()
    with rasterio.open(tmp_path / "clip" / "alpha" / "2024-01-10_fused_water.tif") as r:
        assert r.read(1)[:, N // 2 + 3:].sum() == 0                                        # nothing outside the outline


def test_composite_frequency_is_the_share_of_dates_with_water(tmp_path, monkeypatch):
    plan = scenes(1)                                  # dates 1 and 2 are the same scene A; date 3 is another lake B
    FakeS1({"alpha": plan}).install(monkeypatch)
    batch.map_sites([site("alpha")], None, cfg(tmp_path), log=lambda *_: None)
    from scipy import ndimage as ndi
    a, b = plan[0][1]["truth"], plan[2][1]["truth"]
    core = lambda m: ndi.binary_erosion(m, iterations=4)
    with rasterio.open(tmp_path / "alpha" / "frequency.tif") as r:
        f = r.read(1)
    only_a, both, neither = core(a & ~b), core(a & b), core(~a & ~b)
    assert only_a.sum() > 100 and neither.sum() > 100
    assert (f[only_a] == 67).mean() > 0.97                                                 # water on 2 of 3 dates
    assert (f[neither] == 0).mean() > 0.97
    if both.sum() > 100:
        assert (f[both] == 100).mean() > 0.97
    with rasterio.open(tmp_path / "alpha" / "n_obs.tif") as r:
        assert r.read(1).max() == 3


def test_an_interrupted_run_resumes_without_rereading_and_keeps_the_composite(tmp_path, monkeypatch):
    fake = FakeS1({"alpha": scenes(1)})
    fake.install(monkeypatch, limit=2)                                                     # the first run only got 2 passes
    batch.map_sites([site("alpha")], None, cfg(tmp_path), log=lambda *_: None)
    assert [d for _, d in fake.reads] == ["2024-01-10", "2024-01-22"]
    fake.reads.clear()
    fake.install(monkeypatch)                                                              # second run sees all 3
    res = batch.map_sites([site("alpha")], None, cfg(tmp_path), log=lambda *_: None)
    assert [d for _, d in fake.reads] == ["2024-02-03"]                                    # only the new pass was read
    assert list(res.areas["date"]) == ["2024-01-10", "2024-01-22", "2024-02-03"]
    with rasterio.open(tmp_path / "alpha" / "n_obs.tif") as r:
        assert r.read(1).max() == 3                                                        # composite includes all three dates
    fake.reads.clear()
    batch.map_sites([site("alpha")], None, cfg(tmp_path), log=lambda *_: None)
    assert fake.reads == []                                                                # nothing left to do


def test_resume_false_reprocesses_everything(tmp_path, monkeypatch):
    fake = FakeS1({"alpha": scenes(1)})
    fake.install(monkeypatch)
    batch.map_sites([site("alpha")], None, cfg(tmp_path), log=lambda *_: None)
    n = len(fake.reads)
    batch.map_sites([site("alpha")], None, cfg(tmp_path, resume=False), log=lambda *_: None)
    assert len(fake.reads) == 2 * n
    assert len(batch.read_areas(str(tmp_path))) == 3                                       # duplicates collapse on read


def test_one_failing_site_does_not_stop_the_others(tmp_path, monkeypatch):
    FakeS1({"alpha": scenes(1), "bad": [], "gamma": scenes(40)}, broken={"bad"}).install(monkeypatch)
    logs = []
    res = batch.map_sites([site("alpha"), site("bad"), site("gamma")], None, cfg(tmp_path), log=logs.append)
    assert not res.ok and [f["site"] for f in res.failures] == ["bad"] and "no Sentinel-1 RTC pass" in res.failures[0]["error"]
    assert set(res.areas["site"]) == {"alpha", "gamma"}
    rows = list(csv.DictReader(open(tmp_path / "failures.csv")))
    assert len(rows) == 1 and rows[0]["site"] == "bad"
    assert any("FAILED [bad]" in m for m in logs)
    with pytest.raises(RuntimeError, match="no Sentinel-1 RTC pass"):
        batch.map_sites([site("bad")], None, cfg(tmp_path / "strict", stop_on_error=True), log=lambda *_: None)


def test_composite_can_be_turned_off(tmp_path, monkeypatch):
    FakeS1({"alpha": scenes(1)}).install(monkeypatch)
    batch.map_sites([site("alpha")], None, cfg(tmp_path, composite=False, write=()), log=lambda *_: None)
    assert not (tmp_path / "alpha" / "frequency.tif").exists() and not list((tmp_path / "alpha").glob("*_fused_water.tif"))
    assert len(batch.read_areas(str(tmp_path))) == 3
