import json
import sys
import warnings

import pytest

pytest.importorskip("pyproj")
from shapely.geometry import LineString, Point, box as sbox, mapping  # noqa: E402

from ai_water import sites  # noqa: E402

CORNERS = "((14.872088697247067, 121.14302981543655),(15.053187085370627, 121.26233447214564))"


# ------------------------------------------------------------------ geometry helpers
def test_corners_in_the_lat_lon_format_of_a_map_give_a_lon_lat_box_in_either_order():
    bb = sites.parse_corners(CORNERS)
    assert bb == pytest.approx((121.14302981543655, 14.872088697247067, 121.26233447214564, 15.053187085370627))
    swapped = "((15.053187085370627, 121.26233447214564),(14.872088697247067, 121.14302981543655))"
    assert sites.parse_corners(swapped) == pytest.approx(bb)


def test_swapped_lat_lon_is_caught():
    with pytest.raises(ValueError, match="out of range.*swapped"):
        sites.parse_corners("((121.14, 14.87),(121.26, 15.05))")
    with pytest.raises(ValueError, match="two \\(lat, lon\\) pairs"):
        sites.parse_corners("(14.87, 121.14)")


def test_small_boxes_are_expanded_about_their_centre_and_big_ones_are_left_alone():
    small = (121.3232, 14.0725, 121.3367, 14.0851)                       # Sampaloc Lake: 1.4 km
    bb, (wk, hk), grown = sites.expand_bbox(small, 5.0)
    assert grown and 1.3 < wk < 1.6 and 1.3 < hk < 1.5
    s = sites.box_site("Sampaloc Lake", small, "lake")
    w, h = s.size_km()
    assert w == pytest.approx(5.0, abs=0.02) and h == pytest.approx(5.0, abs=0.02) and "expanded" in s.note
    assert s.center == pytest.approx(((small[0] + small[2]) / 2, (small[1] + small[3]) / 2))
    big = (121.14, 14.87, 121.26, 15.05)
    assert sites.expand_bbox(big, 5.0)[2] is False and sites.box_site("x", big).bbox == big


def test_utm_zone_follows_the_location_so_a_nationwide_run_crosses_zones_correctly():
    assert sites.utm_crs(121.1, 14.9) == "EPSG:32651" and sites.utm_crs(119.5, 10.0) == "EPSG:32650"
    assert sites.utm_crs(126.5, 7.0) == "EPSG:32652" and sites.utm_crs(150.0, -30.0) == "EPSG:32756"
    assert sites.box_site("palawan", (118.5, 9.5, 118.7, 9.7), min_size_km=0).crs == "EPSG:32650"


def test_site_box_must_be_west_south_east_north():
    with pytest.raises(ValueError, match="west < east"):
        sites.Site("bad", (121.3, 14.9, 121.1, 15.0))


def test_a_point_gets_a_box_from_its_radius_or_the_default_for_its_kind():
    s = sites.point_site("pt", 14.9, 121.2, radius_km=3.0, kind="lake")
    assert s.size_km()[0] == pytest.approx(6.0, abs=0.05) and s.size_km()[1] == pytest.approx(6.0, abs=0.05)
    assert sites.point_site("r", 14.9, 121.2, kind="river").size_km()[0] == pytest.approx(2 * sites.DEFAULT_RADIUS_KM["river"], abs=0.05)


# ------------------------------------------------------------------ CSV
def write(tmp_path, text, name="s.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_csv_accepts_corners_nsew_columns_and_points_and_ignores_comments(tmp_path):
    p = write(tmp_path, '﻿# a comment\nname,kind,corners,south,west,north,east,lat,lon,radius_km\n'
                        f'Dam A,reservoir,"{CORNERS}",,,,,,,\n'
                        'Lake B,lake,,14.0,121.0,14.1,121.1,,,\n'
                        'River C,river,,,,,,13.7,123.1,4\n')
    a, b, c = sites.sites_from_csv(p)
    assert (a.name, a.kind) == ("dam_a", "reservoir") and a.label == "Dam A"
    assert b.bbox[:2] == pytest.approx((121.0, 14.0)) and b.size_km()[0] >= 5.0                  # expanded
    assert c.kind == "river" and c.size_km()[0] == pytest.approx(8.0, abs=0.1)


def test_csv_errors_name_the_row_and_the_problem(tmp_path):
    with pytest.raises(ValueError, match=r"row 2 \(X\): give `corners`"):
        sites.sites_from_csv(write(tmp_path, "name,kind\nX,lake\n"))
    with pytest.raises(ValueError, match="row 3.*duplicate"):
        sites.sites_from_csv(write(tmp_path, f'name,corners\nA,"{CORNERS}"\nA,"{CORNERS}"\n'))
    with pytest.raises(ValueError, match="row 2.*`name` is empty"):
        sites.sites_from_csv(write(tmp_path, f'name,corners\n,"{CORNERS}"\n'))
    with pytest.raises(ValueError, match="row 2 \\(Z\\).*out of range"):
        sites.sites_from_csv(write(tmp_path, 'name,lat,lon\nZ,121.1,14.9\n'))


def test_a_site_outside_the_philippines_warns_but_loads(tmp_path):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = sites.sites_from_csv(write(tmp_path, 'name,lat,lon\nElsewhere,48.8,2.3\n'))
    assert len(s) == 1 and any("outside the Philippines" in str(x.message) for x in w)


def test_the_six_training_sites_load_as_supplied():
    ss = sites.load_training_sites(register_them=False)
    assert [s.name for s in ss] == ["angat_dam", "magat_dam", "la_mesa_dam", "sampaloc_lake", "libmanan_river", "cagayan_river"]
    assert [s.kind for s in ss] == ["reservoir", "reservoir", "reservoir", "lake", "river", "river"]
    assert ss[0].bbox == sites.default_site().bbox == pytest.approx((121.14302981543655, 14.872088697247067, 121.26233447214564, 15.053187085370627))
    assert ss[1].bbox == pytest.approx((121.30825389251115, 16.714838338457948, 121.52111399871818, 16.85568413745271))
    assert all(min(s.size_km()) >= 5.0 - 0.02 for s in ss) and all(s.crs == "EPSG:32651" for s in ss)
    assert "expanded" in next(s for s in ss if s.name == "sampaloc_lake").note


def test_training_sites_file_that_is_missing_or_empty_explains_itself(tmp_path):
    with pytest.raises(FileNotFoundError, match="create it"):
        sites.load_training_sites(tmp_path / "none.csv")
    with pytest.raises(ValueError, match="has no sites yet"):
        sites.load_training_sites(write(tmp_path, "# only comments\nname,kind,corners\n"))


# ------------------------------------------------------------------ vector files
def geojson(tmp_path, features, name="v.geojson"):
    p = tmp_path / name
    p.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": pr, "geometry": mapping(g)} for pr, g in features]}))
    return p


def test_polygon_features_become_sites_with_a_buffered_box_and_outline(tmp_path):
    dam = sbox(121.20, 14.90, 121.24, 14.95)
    p = geojson(tmp_path, [({"name": "My Dam"}, dam)])
    (s,) = sites.sites_from_vector(p, buffer_km=1.0)
    assert (s.name, s.label, s.kind) == ("my_dam", "My Dam", "reservoir")
    w, so, e, n = s.bbox
    assert w < 121.20 - 0.008 and e > 121.24 + 0.008 and so < 14.90 - 0.008 and n > 14.95 + 0.008      # ~1 km margin all round
    assert s.geometry is not None and s.geometry.contains(dam) and s.geometry.area > dam.area
    s0 = sites.sites_from_vector(p, buffer_km=0.0)[0]
    assert s0.bbox == pytest.approx(dam.bounds) and s0.geometry.equals(dam)


def test_points_lines_names_and_duplicates_in_a_vector_file(tmp_path):
    p = geojson(tmp_path, [({"name": "Reach"}, LineString([(121.0, 14.0), (121.1, 14.05)])),
                           ({"NAME": "Pond"}, Point(121.5, 14.5)), ({}, Point(121.6, 14.6)),
                           ({"name": "Reach"}, LineString([(122.0, 14.0), (122.1, 14.05)]))])
    r, pond, unnamed, r2 = sites.sites_from_vector(p, radius_km=2.0)
    assert r.kind == "river" and pond.size_km()[0] == pytest.approx(4.0, abs=0.1) and unnamed.name == "aoi_3"
    assert (r.name, r2.name) == ("reach_1", "reach_2")                                               # repeated names are made unique
    q = geojson(tmp_path, [({"id": 7, "dam": "Ambuklao"}, sbox(120.8, 16.4, 120.9, 16.5))], "q.geojson")
    assert sites.sites_from_vector(q)[0].label == "Ambuklao" and sites.sites_from_vector(q, name_col="id")[0].label == "7"


def test_non_geojson_vector_files_explain_the_geopandas_requirement(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "geopandas", None)
    f = tmp_path / "dams.gpkg"
    f.write_bytes(b"x")
    with pytest.raises(ImportError, match="needs geopandas"):
        sites.sites_from_vector(f)


def test_geojson_in_a_projected_crs_is_refused_rather_than_misread(tmp_path):
    p = tmp_path / "utm.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:32651"}},
                             "features": [{"type": "Feature", "properties": {}, "geometry": mapping(sbox(3e5, 1.6e6, 3.1e5, 1.7e6))}]}))
    with pytest.raises(ValueError, match="EPSG:4326"):
        sites.sites_from_vector(p)


def test_load_sites_dispatches_on_the_extension(tmp_path):
    csv_p = write(tmp_path, f'name,corners\nA,"{CORNERS}"\n')
    assert sites.load_sites(csv_p)[0].name == "a"
    assert sites.load_sites(geojson(tmp_path, [({"name": "V"}, sbox(121.2, 14.9, 121.3, 15.0))]))[0].name == "v"
    with pytest.raises(FileNotFoundError):
        sites.load_sites(tmp_path / "missing.csv")


# ------------------------------------------------------------------ registry
def test_default_site_is_angat_dam_and_lookup_is_forgiving():
    d = sites.default_site()
    assert d.name == "angat_dam" and d.label == "Angat Dam" and d.kind == "reservoir"
    assert sites.get("angat") is d and sites.get("Angat Dam") is d and sites.get("ANGAT_DAM") is d
    with pytest.raises(KeyError, match="known:.*angat_dam"):
        sites.get("nowhere")


def test_register_accepts_the_same_box_twice_but_not_a_different_one():
    a = sites.box_site("Test Registry Lake", (121.0, 14.0, 121.1, 14.1), "lake")
    sites.register(a)
    sites.register(sites.box_site("Test Registry Lake", (121.0, 14.0, 121.1, 14.1), "lake"))
    assert sites.get("test registry lake") == a
    with pytest.raises(ValueError, match="different box"):
        sites.register(sites.box_site("Test Registry Lake", (121.0, 14.0, 121.2, 14.2), "lake"))
    sites.SITES.pop("test_registry_lake")


# ------------------------------------------------------------------ choosing the area
def test_resolve_aoi_defaults_to_angat_and_takes_a_single_dam_file(tmp_path):
    assert [s.name for s in sites.resolve_aoi()] == ["angat_dam"]
    p = geojson(tmp_path, [({"name": "Pantabangan Dam"}, sbox(121.1, 15.7, 121.2, 15.9))])
    (s,) = sites.resolve_aoi(p)
    assert s.label == "Pantabangan Dam" and s.geometry is not None


def test_resolve_aoi_does_not_silently_map_the_first_of_many(tmp_path):
    p = geojson(tmp_path, [({"name": "A"}, sbox(121.0, 14.0, 121.1, 14.1)), ({"name": "B"}, sbox(121.5, 14.5, 121.6, 14.6)),
                           ({"name": "C"}, sbox(122.0, 15.0, 122.1, 15.1))])
    with pytest.raises(ValueError, match=r"3 features.*multiple=True.*names"):
        sites.resolve_aoi(p)
    assert [s.name for s in sites.resolve_aoi(p, multiple=True)] == ["a", "b", "c"]
    assert [s.name for s in sites.resolve_aoi(p, names=["C", "a"])] == ["a", "c"]
    with pytest.raises(KeyError, match=r"not in .*\['zzz'\]"):
        sites.resolve_aoi(p, names="Zzz")
    assert [s.name for s in sites.resolve_aoi(write(tmp_path, f'name,corners\nQ,"{CORNERS}"\nR,"{CORNERS.replace("14.8", "13.8")}"\n'), multiple=True)] == ["q", "r"]
