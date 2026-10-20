"""Nationwide surface-water mapping: ~20 km tiles over a country outline, each mapped like a single site.

    python scripts/map_nationwide.py --boundary philippines.geojson --prescreen --gee-project my-project \
        --start 2024-01-01 --end 2024-03-31 --bbox 120.5 14.0 121.5 15.0       # --bbox / --limit-tiles: try a region first

Omit --boundary to use Earth Engine's country boundary (--country). --prescreen drops tiles with (almost) no surface water using JRC.
Output goes to --out (default outputs/nationwide): per tile <tile>/frequency.tif, n_obs.tif (and per-date masks with --write), plus
areas.csv, failures.csv and index.geojson (every tile with its status and statistics). Re-running resumes; --restart reprocesses.
Individual water bodies or a list of them are mapped with scripts/map_water.py or notebooks/ai_water_inference.ipynb instead.
Earth Engine is needed (DEM, land cover): authenticate once with `earthengine authenticate` and pass --gee-project.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_water import batch, infer, jrc, nationwide, scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/terramind_water_decoder.pt", help="checkpoint (small or full)")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="outputs/nationwide")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--gee-project", default=None)
    g = ap.add_argument_group("coverage")
    g.add_argument("--boundary", default=None, help="polygon file of the country / provinces (default: Earth Engine's LSIB)")
    g.add_argument("--country", default="Philippines")
    g.add_argument("--tile-deg", type=float, default=nationwide.TILE_DEG)
    g.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), default=None, help="restrict to a lon/lat box")
    g.add_argument("--limit-tiles", type=int, default=None, help="map only the first N tiles (a quick test)")
    g.add_argument("--prescreen", action="store_true", help="skip tiles without surface water (JRC)")
    g.add_argument("--min-water-frac", type=float, default=0.001)
    o = ap.add_argument_group("outputs")
    o.add_argument("--write", nargs="*", default=[], choices=list(infer.ALL_OUTPUTS), help="per-scene GeoTIFFs (default: none)")
    o.add_argument("--threshold", type=float, default=0.5, help="water probability threshold (default 0.5)")
    o.add_argument("--max-slope", type=float, default=infer.MAX_SLOPE_DEG,
                   help="terrain safeguard: water on slopes steeper than this (deg) is removed and flagged; 0 = off")
    o.add_argument("--no-landcover", action="store_true", help="blank the land-cover channels")
    o.add_argument("--restart", action="store_true", help="reprocess scenes already done")
    o.add_argument("--stop-on-error", action="store_true")
    a = ap.parse_args()

    model, meta, device = infer.load_model(a.model)
    print(f"model {a.model} on {device}: trained on {meta.get('sites')} {meta.get('start')}..{meta.get('end')}")
    try:
        jrc.init_gee(a.gee_project)
    except Exception as e:                                          # noqa: BLE001
        sys.exit(f"error: Earth Engine could not be initialised ({e.__class__.__name__}: {e}). It supplies the DEM and land cover: "
                 "run `earthengine authenticate` once and pass --gee-project <your project>.")
    cfg = batch.MapConfig(start=a.start, end=a.end, out_dir=a.out, write=tuple(a.write), composite=True, resume=not a.restart,
                          stop_on_error=a.stop_on_error, cache_dir=a.cache, device=None, threshold=a.threshold,
                          max_slope_deg=a.max_slope or None, use_landcover=not a.no_landcover, scene=scene.SceneConfig())
    try:
        boundary = nationwide.national_boundary(a.boundary, a.country)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        sys.exit(f"error: {e}")
    res, tiles, idx = nationwide.run_nationwide(model, boundary, cfg, tile_deg=a.tile_deg,
                                                jrc_source=jrc.GeeJrc() if a.prescreen else None,
                                                min_water_frac=a.min_water_frac, bbox=a.bbox, limit_tiles=a.limit_tiles)
    print(nationwide.monthly_national_area(res.areas).to_string(index=False))
    if res.failures:
        print(f"{len(res.failures)} failure(s): see {os.path.join(a.out, 'failures.csv')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
