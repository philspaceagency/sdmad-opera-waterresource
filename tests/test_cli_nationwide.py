"""scripts/map_nationwide.py: argument handling, with Planetary Computer replaced by a stand-in."""
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

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "map_nationwide.py"


def run_cli_real(monkeypatch, *argv):
    spec = importlib.util.spec_from_file_location("map_nationwide_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", ["map_nationwide.py", *argv])
    mod.main()


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(infer, "load_model", lambda path, device=None: (FakeModel(), {"sites": ["stand-in"]}, "cpu"))
    monkeypatch.setattr(jrc, "init_gee", lambda project=None, **kw: None)
    monkeypatch.setattr(terrain, "GeeTerrain", lambda *a, **k: FLAT)
    spec = importlib.util.spec_from_file_location("map_nationwide_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", ["map_nationwide.py", *argv])
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


def test_nationwide_from_a_boundary_file_with_a_region_and_a_cap(tmp_path, monkeypatch, capsys):
    w, s, e, n = lonlat_box(X0, Y1 - N * RES, X0 + N * RES, Y1)
    cx, cy = 0.5 * (w + e), 0.5 * (s + n)
    boundary = geojson(tmp_path, [({"name": "Isla"}, sbox(cx - 0.6, cy - 0.6, cx + 0.6, cy + 0.6))], "isla.geojson")
    AnySite().install(monkeypatch)
    out = tmp_path / "nat"
    run_cli(monkeypatch, "--boundary", boundary, "--bbox", str(cx - 0.1), str(cy - 0.1), str(cx + 0.1), str(cy + 0.1),
            "--limit-tiles", "2", "--start", "2024-01-01", "--end", "2024-03-01", "--out", str(out))
    idx = json.load(open(out / "index.geojson"))["features"]
    assert len(idx) == 2 and all(f["properties"]["status"] == "ok" for f in idx)
    assert not list(out.glob("tile_*/*_surfacewater.tif"))                     # nationwide writes no per-date masks by default
    assert "n_tiles" in capsys.readouterr().out
