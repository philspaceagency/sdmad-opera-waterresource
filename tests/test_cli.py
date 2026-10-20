"""scripts/map_water.py: argument handling, with Planetary Computer replaced by a stand-in."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("pyproj")
from shapely.geometry import box as sbox, mapping  # noqa: E402

from ai_water import infer, jrc, terrain  # noqa: E402
from test_batch import FLAT, FakeModel, FakeS1, N, RES, X0, Y1, lonlat_box, scenes  # noqa: E402

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "map_water.py"


def run_cli_real(monkeypatch, *argv):
    spec = importlib.util.spec_from_file_location("map_water_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", ["map_water.py", *argv])
    mod.main()


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(infer, "load_model", lambda path, device=None: (FakeModel(), {"sites": ["stand-in"]}, "cpu"))
    monkeypatch.setattr(jrc, "init_gee", lambda project=None, **kw: None)
    monkeypatch.setattr(terrain, "GeeTerrain", lambda *a, **k: FLAT)
    spec = importlib.util.spec_from_file_location("map_water_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", ["map_water.py", *argv])
    mod.main()


class AnySite(FakeS1):
    """Every site gets the same three scenes (the CLI picks site names itself)."""

    def __init__(self):
        super().__init__({})
        self.sc = scenes(1)

    @property
    def plan(self):
        class P(dict):
            def __missing__(s, k):
                return self.sc
        return P()

    @plan.setter
    def plan(self, v):
        pass


def geojson(tmp_path, feats, name="v.geojson"):
    p = tmp_path / name
    p.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": pr, "geometry": mapping(g)} for pr, g in feats]}))
    return str(p)


def test_default_area_is_angat_dam(tmp_path, monkeypatch, capsys):
    fake = AnySite()
    fake.install(monkeypatch)
    run_cli(monkeypatch, "--start", "2024-01-01", "--end", "2024-03-01", "--out", str(tmp_path))
    assert {s for s, _ in fake.reads} == {"angat_dam"} and (tmp_path / "areas.csv").exists()
    assert "mapping: Angat Dam" in capsys.readouterr().out and (tmp_path / "angat_dam" / "frequency.tif").exists()


def test_a_dam_vector_file_replaces_the_default(tmp_path, monkeypatch):
    fake = AnySite()
    fake.install(monkeypatch)
    f = geojson(tmp_path, [({"name": "Pantabangan Dam"}, sbox(121.10, 15.78, 121.16, 15.86))])
    run_cli(monkeypatch, "--start", "2024-01-01", "--end", "2024-03-01", "--out", str(tmp_path / "o"), "--aoi", f)
    assert {s for s, _ in fake.reads} == {"pantabangan_dam"}


def test_many_bodies_need_the_multiple_flag_and_names_can_pick(tmp_path, monkeypatch):
    f = geojson(tmp_path, [({"name": "A Dam"}, sbox(121.0, 14.0, 121.1, 14.1)), ({"name": "B Dam"}, sbox(121.5, 14.5, 121.6, 14.6))])
    common = ["--start", "2024-01-01", "--end", "2024-03-01", "--aoi", f]
    fake = AnySite()
    fake.install(monkeypatch)
    with pytest.raises(SystemExit, match=r"error: .*2 features.*multiple=True"):
        run_cli(monkeypatch, *common, "--out", str(tmp_path / "x"))
    run_cli(monkeypatch, *common, "--multiple", "--out", str(tmp_path / "m"))
    assert {s for s, _ in fake.reads} == {"a_dam", "b_dam"}
    fake.reads.clear()
    run_cli(monkeypatch, *common, "--names", "B Dam", "--out", str(tmp_path / "n"))
    assert {s for s, _ in fake.reads} == {"b_dam"}


def test_a_missing_model_is_explained(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    with pytest.raises(FileNotFoundError, match="train a model first"):
        monkeypatch.undo()
        monkeypatch.setattr(jrc, "init_gee", lambda project=None, **kw: None)
        run_cli_real(monkeypatch, "--model", str(tmp_path / "none.pt"), "--start", "2024-01-01", "--end", "2024-03-01", "--out", str(tmp_path))


def test_a_nationwide_flag_no_longer_exists_on_map_water(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, "--nationwide", "--start", "2024-01-01", "--end", "2024-03-01", "--out", str(tmp_path))
