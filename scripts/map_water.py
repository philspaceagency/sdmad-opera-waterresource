"""Map surface water with a trained model: one body (default Angat Dam), many bodies, or the whole country.

    # Angat Dam (the default area)
    python scripts/map_water.py --model models/terramind_water_decoder.pt --start 2024-01-01 --end 2024-03-31

    # another dam: give its outline (GeoJSON; GeoPackage / Shapefile / KML need geopandas)
    python scripts/map_water.py --model ... --aoi my_dam.geojson --start ... --end ...

    # many bodies: a file with one feature per body (or a sites CSV); --names picks some of them
    python scripts/map_water.py --model ... --aoi dams.geojson --multiple --start ... --end ...
    python scripts/map_water.py --model ... --aoi dams.geojson --names "Ambuklao Dam" "Binga Dam" ...

    # nationwide: ~20 km tiles over a country outline (omit --boundary to use Earth Engine's), optionally only the tiles with water
    python scripts/map_water.py --model ... --nationwide --boundary philippines.geojson --prescreen --gee-project my-project \
        --start 2024-01-01 --end 2024-03-31 --bbox 120.5 14.0 121.5 15.0       # --bbox / --limit-tiles: try a region first

Output goes to --out (default outputs/maps): per site <site>/frequency.tif, n_obs.tif and per-date masks, plus areas.csv, failures.csv
and (nationwide) index.geojson. Re-running resumes; --restart reprocesses. --otsu-only runs without a model (thresholds only).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_water import batch, infer, jrc, nationwide, pipeline, sites  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/terramind_water_decoder.pt", help="checkpoint (small or full); see --otsu-only")
    ap.add_argument("--otsu-only", action="store_true", help="no model: Otsu thresholds only")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="outputs/maps")
    ap.add_argument("--cache", default="cache")
    g = ap.add_argument_group("area (default: Angat Dam)")
    g.add_argument("--aoi", default=None, help="vector file of the water body / bodies, or a sites CSV")
    g.add_argument("--names", nargs="*", default=None, help="only these features of --aoi")
    g.add_argument("--multiple", action="store_true", help="map every feature of --aoi")
    g.add_argument("--name-col", default=None, help="property that holds the feature's name")
    g.add_argument("--buffer-km", type=float, default=1.0, help="margin around each outline (default 1 km)")
    g.add_argument("--kind", default=None, help="reservoir | lake | river ... (default: guessed from the geometry)")
    n = ap.add_argument_group("nationwide")
    n.add_argument("--nationwide", action="store_true")
    n.add_argument("--boundary", default=None, help="polygon file of the country / provinces (default: Earth Engine's LSIB)")
    n.add_argument("--country", default="Philippines")
    n.add_argument("--tile-deg", type=float, default=nationwide.TILE_DEG)
    n.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), default=None, help="restrict to a lon/lat box")
    n.add_argument("--limit-tiles", type=int, default=None)
    n.add_argument("--prescreen", action="store_true", help="skip tiles without surface water (JRC, needs Earth Engine)")
    n.add_argument("--min-water-frac", type=float, default=0.001)
    ap.add_argument("--gee-project", default=None)
    o = ap.add_argument_group("outputs")
    o.add_argument("--write", nargs="*", default=None, choices=list(infer.ALL_OUTPUTS),
                   help="per-scene GeoTIFFs (default: fused_water flags; nationwide: none)")
    o.add_argument("--no-composite", action="store_true", help="skip frequency.tif / n_obs.tif")
    o.add_argument("--restart", action="store_true", help="reprocess dates already done")
    o.add_argument("--stop-on-error", action="store_true")
    a = ap.parse_args()

    model = None
    if not a.otsu_only:
        model, meta, device = infer.load_model(a.model)
        print(f"model {a.model} on {device}: trained on {meta.get('sites')} {meta.get('start')}..{meta.get('end')}")
    write = tuple(a.write) if a.write is not None else (() if a.nationwide else ("fused_water", "flags"))
    cfg = batch.MapConfig(start=a.start, end=a.end, out_dir=a.out, write=write, composite=not a.no_composite, resume=not a.restart,
                          stop_on_error=a.stop_on_error, cache_dir=a.cache, device=None, water=pipeline.WaterConfig(cache_dir=a.cache))

    if a.nationwide:
        source = None
        if a.prescreen or not a.boundary:
            jrc.init_gee(a.gee_project)
            source = jrc.GeeJrc() if a.prescreen else None
        try:
            boundary = nationwide.national_boundary(a.boundary, a.country)
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            sys.exit(f"error: {e}")
        res, tiles, idx = nationwide.run_nationwide(model, boundary, cfg, tile_deg=a.tile_deg, jrc_source=source,
                                                    min_water_frac=a.min_water_frac, bbox=a.bbox, limit_tiles=a.limit_tiles)
        print(nationwide.monthly_national_area(res.areas).to_string(index=False))
    else:
        kw = {k: v for k, v in dict(name_col=a.name_col, kind=a.kind).items() if v}
        try:
            chosen = sites.resolve_aoi(a.aoi, a.names, a.multiple,
                                       **({"buffer_km": a.buffer_km, **kw} if a.aoi and not a.aoi.lower().endswith(".csv") else {}))
        except (ValueError, KeyError, FileNotFoundError, ImportError) as e:
            sys.exit(f"error: {e.args[0] if e.args else e}")
        print("mapping:", ", ".join(f"{s.label} ({s.kind})" for s in chosen))
        res = batch.map_sites(chosen, model, cfg)
    print(res.areas.groupby("site")[["fused_km2"]].agg(["count", "mean", "max"]).round(3).to_string() if len(res.areas) else "no scenes mapped")
    if res.failures:
        print(f"{len(res.failures)} failure(s): see {os.path.join(a.out, 'failures.csv')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
