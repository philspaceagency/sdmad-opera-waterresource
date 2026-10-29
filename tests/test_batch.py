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

from ai_water import aoi, batch, features as ft, infer, sites, synthetic as syn  # noqa: E402

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


# ------------------------------------------------------------------ stand-ins for Planetary Computer, Earth Engine and the model
class FakeModel:
    """Water where the standardised VV is darker than -15 dB. Needs no torch: infer_scene only calls `predict_prob`."""

    def predict_prob(self, x, **kw):
        return np.where(x[0] < (-15.0 + 10.93) / 4.391, 0.9, 0.1).astype(np.float32)


class FakeS1:
    """Per-site list of (day, orbit, scene dict). `reads` records every pass actually read; sites in `broken` raise."""

    def __init__(self, plan, broken=()):
        self.plan, self.broken, self.reads = plan, set(broken), []

    def install(self, monkeypatch, limit=None):
        def iter_window(st, start, end, scene_cfg=None, cache_dir=None, log=print, skip_days=None):
            if st.name in self.broken:
                raise RuntimeError(f"no Sentinel-1 RTC pass covers {st.name}")
            items = self.plan[st.name][:limit] if limit else self.plan[st.name]

            def gen():
                for day, orbit, sc in items:
                    if skip_days and f"{day}:{orbit}" in skip_days:
                        continue
                    self.reads.append((st.name, day))
                    yield day, orbit, sc["vv"], sc["vh"]
            return geobox(), gen()
        monkeypatch.setattr(infer, "iter_window", iter_window)


FLAT = syn.SyntheticTerrain(np.full((N, N), 100.0, np.float32))


def scenes(seed, fracs=(0.2, 0.2, 0.12), orbit="descending"):
    sc = [syn.make_scene((N, N), water_frac=f, looks=4, seed=seed + (0 if i < 2 else 7)) for i, f in enumerate(fracs)]
    return [("2024-01-10", orbit, sc[0]), ("2024-01-22", orbit, sc[1]), ("2024-02-03", orbit, sc[2])]


def cfg(tmp_path, **kw):
    kw.setdefault("terrain_source", FLAT)
    return batch.MapConfig(start="2024-01-01", end="2024-03-01", out_dir=str(tmp_path), cache_dir=str(tmp_path / "c"), **kw)


# ------------------------------------------------------------------ map_sites
def test_two_sites_are_processed_and_areas_are_recorded(tmp_path, monkeypatch):
    fake = FakeS1({"alpha": scenes(1), "beta": scenes(30)})
    fake.install(monkeypatch)
    res = batch.map_sites([site("alpha"), site("beta", kind="lake")], FakeModel(), cfg(tmp_path), log=lambda *_: None)
    assert res.ok and len(res.areas) == 6 and set(res.areas["site"]) == {"alpha", "beta"}
    a = res.areas
    assert (a["water_km2"] > 0).all() and (a["valid_km2"] > 6.0).all()                    # 256 x 256 px x 100 m2 = 6.55 km2
    assert a.loc[a.site == "beta", "kind"].iloc[0] == "lake" and a.loc[a.site == "alpha", "label"].iloc[0] == "Alpha"
    assert (tmp_path / "alpha" / "philsa_multimodal-surfacewater-weekly_20240110_v0.1.0_r01_surfacewater.tif").exists() and (tmp_path / "beta" / "frequency.tif").exists()
    assert (tmp_path / "alpha" / "philsa_multimodal-surfacewater-weekly_20240110_v0.1.0_r01_flags.tif").exists() and not list((tmp_path / "alpha").glob("*_surfacewaterprob.tif"))                      # only what `write` asks for


def test_area_is_counted_inside_the_outline_only(tmp_path, monkeypatch):
    FakeS1({"alpha": scenes(1)}).install(monkeypatch)
    full = batch.map_sites([site("alpha")], FakeModel(), cfg(tmp_path / "full"), log=lambda *_: None).areas
    half = sbox(*lonlat_box(X0, Y1 - N * RES, X0 + N * RES / 2, Y1))
    clip = batch.map_sites([site("alpha", half)], FakeModel(), cfg(tmp_path / "clip"), log=lambda *_: None).areas
    assert (clip["water_km2"] < full["water_km2"]).all() and (clip["valid_km2"] < 0.55 * full["valid_km2"]).all()
    with rasterio.open(tmp_path / "clip" / "alpha" / "philsa_multimodal-surfacewater-weekly_20240110_v0.1.0_r01_surfacewater.tif") as r:
        assert r.read(1)[:, N // 2 + 3:].sum() == 0                                        # nothing outside the outline


def test_composite_frequency_is_the_share_of_dates_with_water(tmp_path, monkeypatch):
    plan = scenes(1)                                  # dates 1 and 2 are the same scene A; date 3 is another lake B
    FakeS1({"alpha": plan}).install(monkeypatch)
    batch.map_sites([site("alpha")], FakeModel(), cfg(tmp_path), log=lambda *_: None)
    from scipy import ndimage as ndi
    a, b = plan[0][2]["truth"], plan[2][2]["truth"]
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
    batch.map_sites([site("alpha")], FakeModel(), cfg(tmp_path), log=lambda *_: None)
    assert [d for _, d in fake.reads] == ["2024-01-10", "2024-01-22"]
    fake.reads.clear()
    fake.install(monkeypatch)                                                              # second run sees all 3
    res = batch.map_sites([site("alpha")], FakeModel(), cfg(tmp_path), log=lambda *_: None)
    assert [d for _, d in fake.reads] == ["2024-02-03"]                                    # only the new pass was read
    assert list(res.areas["date"]) == ["2024-01-10", "2024-01-22", "2024-02-03"]
    with rasterio.open(tmp_path / "alpha" / "n_obs.tif") as r:
        assert r.read(1).max() == 3                                                        # composite includes all three dates
    fake.reads.clear()
    batch.map_sites([site("alpha")], FakeModel(), cfg(tmp_path), log=lambda *_: None)
    assert fake.reads == []                                                                # nothing left to do


def test_resume_false_reprocesses_everything(tmp_path, monkeypatch):
    fake = FakeS1({"alpha": scenes(1)})
    fake.install(monkeypatch)
    batch.map_sites([site("alpha")], FakeModel(), cfg(tmp_path), log=lambda *_: None)
    n = len(fake.reads)
    batch.map_sites([site("alpha")], FakeModel(), cfg(tmp_path, resume=False), log=lambda *_: None)
    assert len(fake.reads) == 2 * n
    assert len(batch.read_areas(str(tmp_path))) == 3                                       # duplicates collapse on read


def test_one_failing_site_does_not_stop_the_others(tmp_path, monkeypatch):
    FakeS1({"alpha": scenes(1), "bad": [], "gamma": scenes(40)}, broken={"bad"}).install(monkeypatch)
    logs = []
    res = batch.map_sites([site("alpha"), site("bad"), site("gamma")], FakeModel(), cfg(tmp_path), log=logs.append)
    assert not res.ok and [f["site"] for f in res.failures] == ["bad"] and "no Sentinel-1 RTC pass" in res.failures[0]["error"]
    assert set(res.areas["site"]) == {"alpha", "gamma"}
    rows = list(csv.DictReader(open(tmp_path / "failures.csv")))
    assert len(rows) == 1 and rows[0]["site"] == "bad"
    assert any("FAILED [bad]" in m for m in logs)
    with pytest.raises(RuntimeError, match="no Sentinel-1 RTC pass"):
        batch.map_sites([site("bad")], FakeModel(), cfg(tmp_path / "strict", stop_on_error=True), log=lambda *_: None)


def test_composite_can_be_turned_off(tmp_path, monkeypatch):
    FakeS1({"alpha": scenes(1)}).install(monkeypatch)
    batch.map_sites([site("alpha")], FakeModel(), cfg(tmp_path, composite=False, write=()), log=lambda *_: None)
    assert not (tmp_path / "alpha" / "frequency.tif").exists() and not list((tmp_path / "alpha").glob("*_surfacewater.tif"))
    assert len(batch.read_areas(str(tmp_path))) == 3


# ------------------------------------------------------------------ ascending + descending
def test_two_pass_directions_cancel_radar_shadow_in_the_composite(tmp_path, monkeypatch):
    """Shadow is dark on one flank per geometry; the lake is dark in both. min(asc, desc) keeps only the lake."""
    d = syn.make_terrain_scene((N, N), seed=5, orbit="descending")
    a = syn.make_terrain_scene((N, N), seed=5, orbit="ascending")
    plan = [("2024-01-10", "descending", d), ("2024-01-16", "ascending", a), ("2024-01-22", "descending", d),
            ("2024-01-28", "ascending", a)]
    FakeS1({"alpha": plan}).install(monkeypatch)
    terr = syn.SyntheticTerrain(d["dem"])
    res = batch.map_sites([site("alpha")], FakeModel(), cfg(tmp_path, terrain_source=terr, max_slope_deg=None, write=()),
                          log=lambda *_: None)
    assert set(res.areas["orbit"]) == {"ascending", "descending"} and len(res.areas) == 4
    from scipy import ndimage as ndi

    def read(name):
        with rasterio.open(tmp_path / "alpha" / name) as r:
            return r.read(1)
    comb, fd, fa = read("frequency.tif"), read("frequency_descending.tif"), read("frequency_ascending.tif")
    lake = ndi.binary_erosion(d["truth"], iterations=4)
    shadow_d, shadow_a = ndi.binary_erosion(d["shadow"], iterations=3), ndi.binary_erosion(a["shadow"], iterations=3)
    assert (comb[lake] == 100).mean() > 0.97                                              # the lake is water in both geometries
    assert (fd[shadow_d] == 100).mean() > 0.9 and (fa[shadow_d] == 0).mean() > 0.9        # the descending shadow looks like water ...
    assert (comb[shadow_d] == 0).mean() > 0.9 and (comb[shadow_a] == 0).mean() > 0.9      # ... but not in the composite
    with rasterio.open(tmp_path / "alpha" / "n_obs.tif") as r:
        assert r.read(1).max() == 4


def test_a_single_direction_gives_one_composite_and_no_per_direction_files(tmp_path, monkeypatch):
    FakeS1({"alpha": scenes(1)}).install(monkeypatch)
    batch.map_sites([site("alpha")], FakeModel(), cfg(tmp_path), log=lambda *_: None)
    assert (tmp_path / "alpha" / "frequency.tif").exists() and not list((tmp_path / "alpha").glob("frequency_*.tif"))


def test_the_terrain_rule_is_reported_in_the_areas(tmp_path, monkeypatch):
    d = syn.make_terrain_scene((N, N), seed=5, orbit="descending")
    FakeS1({"alpha": [("2024-01-10", "descending", d)]}).install(monkeypatch)
    res = batch.map_sites([site("alpha")], FakeModel(), cfg(tmp_path, terrain_source=syn.SyntheticTerrain(d["dem"])),
                          log=lambda *_: None)
    row = res.areas.iloc[0]
    assert row["terrain_suppressed_km2"] > 0.5 and row["water_km2"] > 0.5                 # shadow removed, lake kept


def test_the_frequency_picture_and_preview_are_written_with_the_product_names(tmp_path, monkeypatch):
    FakeS1({"alpha": scenes(1)}).install(monkeypatch)
    batch.map_sites([site("alpha")], FakeModel(), cfg(tmp_path, write=("water", "flags", "preview")), log=lambda *_: None)
    d = tmp_path / "alpha"
    assert (d / "philsa_multimodal-surfacewater-weekly_20240203_v0.1.0_r01_preview.png").exists()
    assert (d / "philsa_multimodal-surfacewaterfreq-frequency_202401-202402_v0.1.0_r01_surfacewaterfreq.png").exists()


def test_two_directions_on_the_same_day_are_warned_about(tmp_path, monkeypatch):
    d = syn.make_terrain_scene((N, N), seed=5, orbit="descending")
    a = syn.make_terrain_scene((N, N), seed=5, orbit="ascending")
    FakeS1({"alpha": [("2024-01-10", "descending", d), ("2024-01-10", "ascending", a)]}).install(monkeypatch)
    logs = []
    batch.map_sites([site("alpha")], FakeModel(), cfg(tmp_path, terrain_source=syn.SyntheticTerrain(d["dem"])), log=logs.append)
    assert any("same day" in m for m in logs)
