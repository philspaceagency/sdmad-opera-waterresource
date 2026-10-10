"""Nationwide mapping: tiles over a boundary, pre-screen, run, index, monthly totals (network replaced by stand-ins)."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("pyproj")
from affine import Affine  # noqa: E402
from shapely.geometry import box as sbox, mapping  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from ai_water import batch, infer, jrc, nationwide, sites, synthetic as syn  # noqa: E402
from test_batch import N, RES, X0, Y1, geobox, lonlat_box  # noqa: E402


# ------------------------------------------------------------------ boundary
def test_boundary_from_a_polygon_file_merges_features_and_refuses_points(tmp_path):
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "a"}, "geometry": mapping(sbox(121.0, 14.0, 121.3, 14.3))},
        {"type": "Feature", "properties": {"name": "b"}, "geometry": mapping(sbox(121.3, 14.0, 121.6, 14.3))}]}
    p = tmp_path / "country.geojson"
    p.write_text(json.dumps(gj))
    b = nationwide.national_boundary(vector=str(p))
    assert abs(b.area - 0.18) < 1e-9 and b.bounds == (121.0, 14.0, 121.6, 14.3)
    pts = tmp_path / "pts.geojson"
    pts.write_text(json.dumps({"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [121, 14]}}))
    with pytest.raises(ValueError, match="no polygon features"):
        nationwide.national_boundary(vector=str(pts))


def test_boundary_without_a_file_explains_how_to_fix_a_failed_earth_engine_lookup():
    ee = SimpleNamespace(FeatureCollection=lambda *_: (_ for _ in ()).throw(RuntimeError("not initialized")),
                         Filter=SimpleNamespace(eq=lambda *a: None))
    with pytest.raises(RuntimeError, match=r"pass a polygon file with national_boundary\(vector=\.\.\.\)"):
        nationwide.national_boundary(ee_module=ee)


# ------------------------------------------------------------------ tiles
ISLAND = sbox(121.05, 14.02, 121.50, 14.40)


def test_tiles_sit_on_a_fixed_grid_and_their_land_clips_add_up_to_the_land():
    tiles = nationwide.make_tiles(ISLAND, tile_deg=0.18, margin_km=0.0, min_land_km2=0.0)
    names = {t.name for t in tiles}
    assert len(names) == len(tiles) > 6
    t = next(t for t in tiles if t.name == "tile_0078_0673")                  # floor(14.05/0.18)=78, floor(121.14/0.18)=673
    assert t.bbox == pytest.approx((673 * 0.18, 78 * 0.18, 674 * 0.18, 79 * 0.18)) and t.kind == "tile" and t.crs == "EPSG:32651"
    assert abs(unary_union([t.geometry for t in tiles]).symmetric_difference(ISLAND).area) < 1e-9     # nothing lost or doubled
    assert all(tl.geometry.within(sbox(*tl.bbox).buffer(1e-9)) for tl in tiles)
    inner = next(t for t in tiles if sbox(*t.bbox).within(ISLAND))
    assert inner.geometry.equals(sbox(*inner.bbox))                              # a tile fully on land is not clipped
    assert [t.name for t in nationwide.make_tiles(ISLAND, 0.18, 0.0, 0.0)] == [t.name for t in tiles]   # deterministic


def test_slivers_of_land_do_not_become_tiles_and_a_bbox_restricts_the_run():
    spill = unary_union([ISLAND, sbox(121.50, 14.10, 121.5004, 14.12)])         # a 40 m x 2 km sliver into the next tile
    n_all = len(nationwide.make_tiles(spill, 0.18, 0.0, min_land_km2=0.0))
    n_min = len(nationwide.make_tiles(spill, 0.18, 0.0, min_land_km2=1.0))
    assert n_min == n_all - 1
    part = nationwide.make_tiles(ISLAND, 0.18, 0.0, 0.0, bbox=(121.05, 14.02, 121.20, 14.20))
    assert 0 < len(part) < len(nationwide.make_tiles(ISLAND, 0.18, 0.0, 0.0))


def test_margin_keeps_the_shoreline_inside_the_tile_geometry():
    tiles = nationwide.make_tiles(ISLAND, 0.18, margin_km=2.0, min_land_km2=0.0)
    assert unary_union([t.geometry for t in tiles]).area > ISLAND.area


# ------------------------------------------------------------------ prescreen
class OccAlternating:
    def __init__(self):
        self.n = 0

    def occurrence(self, grid):
        self.n += 1
        occ = np.zeros((grid.ny, grid.nx), np.float32)
        if self.n % 2 == 0:
            occ[: grid.ny // 5] = 100.0                                            # a fifth of the tile is permanent water
        return occ


ALIGNED = sbox(121.14, 14.04, 121.50, 14.40)          # exactly 2 x 2 tiles of 0.18 deg, all of it land


def test_prescreen_keeps_only_tiles_with_surface_water():
    tiles = nationwide.make_tiles(ALIGNED, 0.18, 0.0)
    assert len(tiles) == 4 and all(t.geometry.equals(sbox(*t.bbox)) or t.geometry.symmetric_difference(sbox(*t.bbox)).area < 1e-9 for t in tiles)
    kept, df = nationwide.prescreen(tiles, OccAlternating(), min_water_frac=0.02, log=lambda *_: None)
    assert [t.name for t in kept] == [t.name for t in tiles[1::2]]
    assert len(df) == len(tiles) and df["kept"].sum() == len(kept) and (df.loc[df.kept, "water_frac"] > 0.05).all()


def test_prescreen_counts_water_inside_the_land_outline_only():
    class RightHalfWater:
        def occurrence(self, grid):
            occ = np.zeros((grid.ny, grid.nx), np.float32)
            occ[:, int(grid.nx * 0.65):] = 100.0                                      # the sea: well clear of the land half
            return occ
    bb = (121.0, 14.0, 121.18, 14.18)
    left = sbox(121.0, 14.0, 121.09, 14.18)                                       # land = the left half; the 'water' is sea to the right
    t = sites.Site("coast", bb, None, "tile", geometry=left)
    kept, df = nationwide.prescreen([t], RightHalfWater(), min_water_frac=0.01, log=lambda *_: None)
    assert kept == [] and df.loc[0, "water_frac"] < 0.03
    kept, _ = nationwide.prescreen([sites.Site("inland", bb, None, "tile")], RightHalfWater(), min_water_frac=0.01, log=lambda *_: None)
    assert len(kept) == 1


# ------------------------------------------------------------------ run
def _stub_iter_window(monkeypatch, broken=()):
    reads = []

    def iter_window(st, start, end, water_cfg=None, cache_dir=None, log=print, skip_days=None):
        if st.name in broken:
            raise RuntimeError("no Sentinel-1 RTC pass covers this tile")
        g = jrc.grid_for_bbox(st.bbox, st.crs, RES)                                # a N x N scene at this tile's north-west corner
        gb = SimpleNamespace(transform=Affine(RES, 0, g.x0, 0, -RES, g.y1), crs=st.crs, shape=(N, N))
        sc = [syn.make_scene((N, N), water_frac=f, looks=4, seed=3 + k) for k, f in enumerate((0.18, 0.1))]

        def gen():
            for day, s in zip(("2024-01-10", "2024-02-03"), sc):
                if skip_days and day in skip_days:
                    continue
                reads.append((st.name, day))
                yield day, s["vv"], s["vh"]
        return gb, gen()
    monkeypatch.setattr(infer, "iter_window", iter_window)
    return reads


def test_run_nationwide_maps_tiles_isolates_failures_and_writes_the_index(tmp_path, monkeypatch):
    w, s, e, n = lonlat_box(X0, Y1 - N * RES, X0 + N * RES, Y1)
    cx, cy = 0.5 * (w + e), 0.5 * (s + n)
    boundary = sbox(cx - 0.6, cy - 0.6, cx + 0.6, cy + 0.6)                        # a big 'country'
    inner = (cx - 0.1, cy - 0.1, cx + 0.1, cy + 0.1)                               # run only the tiles around its middle (all land)
    bad = nationwide.make_tiles(boundary, 0.18, 0.0, bbox=inner)[1].name
    _stub_iter_window(monkeypatch, broken={bad})
    cfg = batch.MapConfig(start="2024-01-01", end="2024-03-01", out_dir=str(tmp_path), cache_dir=str(tmp_path / "c"), write=())
    res, tiles, idx = nationwide.run_nationwide(None, boundary, cfg, margin_km=0.0, bbox=inner, limit_tiles=3, log=lambda *_: None)
    assert len(tiles) == 3 and [f["site"] for f in res.failures] == [bad]
    assert set(res.areas["site"]) == {t.name for t in tiles} - {bad} and (res.areas["fused_km2"] > 0).all()
    gj = json.load(open(idx))
    status = {f["properties"]["tile"]: f["properties"] for f in gj["features"]}
    assert status[bad]["status"] == "failed" and "no Sentinel-1 RTC pass" in status[bad]["error"]
    ok = [p for t, p in status.items() if t != bad]
    assert all(p["status"] == "ok" and p["n_dates"] == 2 and p["mean_fused_km2"] > 0 for p in ok)
    assert all((tmp_path / t / "frequency.tif").exists() for t in status if t != bad)


def test_run_nationwide_with_prescreen_only_maps_wet_tiles(tmp_path, monkeypatch):
    w, s, e, n = lonlat_box(X0, Y1 - N * RES, X0 + N * RES, Y1)
    cx, cy = 0.5 * (w + e), 0.5 * (s + n)
    boundary = sbox(cx - 0.2, cy - 0.2, cx + 0.2, cy + 0.2)
    reads = _stub_iter_window(monkeypatch)
    cfg = batch.MapConfig(start="2024-01-01", end="2024-03-01", out_dir=str(tmp_path), cache_dir=str(tmp_path / "c"), write=())
    res, tiles, idx = nationwide.run_nationwide(None, boundary, cfg, margin_km=0.0, jrc_source=OccAlternating(), min_water_frac=0.02,
                                                log=lambda *_: None)
    total = len(nationwide.make_tiles(boundary, 0.18, 0.0))
    assert 0 < len(tiles) < total and {t for t, _ in reads} == {t.name for t in tiles}
    assert all(f["properties"]["jrc_water_frac"] is not None for f in json.load(open(idx))["features"])


# ------------------------------------------------------------------ totals
def test_monthly_national_area_averages_passes_within_a_tile_then_sums_tiles():
    a = pd.DataFrame([
        {"site": "t1", "date": "2024-01-05", "fused_km2": 10.0, "otsu_km2": 9.0},
        {"site": "t1", "date": "2024-01-17", "fused_km2": 14.0, "otsu_km2": 13.0},       # t1 January mean 12
        {"site": "t2", "date": "2024-01-09", "fused_km2": 5.0, "otsu_km2": 4.0},
        {"site": "t1", "date": "2024-02-02", "fused_km2": 11.0, "otsu_km2": 10.0}])
    m = nationwide.monthly_national_area(a).set_index("month")
    assert m.loc["2024-01", "fused_km2"] == 17.0 and m.loc["2024-01", "n_tiles"] == 2
    assert m.loc["2024-02", "fused_km2"] == 11.0 and m.loc["2024-02", "n_tiles"] == 1          # undercount is visible, not hidden
    assert nationwide.monthly_national_area(a.iloc[0:0]).empty
