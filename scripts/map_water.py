"""Map surface water with a trained model: one body (default Angat Dam) or a list of bodies.

    # Angat Dam (the default area)
    python scripts/map_water.py --model models/terramind_water_decoder.pt --start 2024-01-01 --end 2024-03-31

    # another dam: give its outline (GeoJSON; GeoPackage / Shapefile / KML need geopandas)
    python scripts/map_water.py --model ... --aoi my_dam.geojson --start ... --end ...

    # many bodies: a file with one feature per body (or a sites CSV); --names picks some of them
    python scripts/map_water.py --model ... --aoi dams.geojson --multiple --start ... --end ...
    python scripts/map_water.py --model ... --aoi dams.geojson --names "Ambuklao Dam" "Binga Dam" ...

Output goes to --out (default outputs/maps): per site <site>/ the named per-date products (ai_water/naming.py), frequency.tif / .png, n_obs.tif, plus areas.csv, failures.csv.
The whole country is a separate job: scripts/map_nationwide.py. Re-running resumes; --restart reprocesses.
Earth Engine is needed for the terrain (DEM, land cover): authenticate once with `earthengine authenticate` and pass --gee-project.
Both Sentinel-1 pass directions are used when the archive has them; frequency.tif is the minimum over the two directions.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_water import batch, infer, jrc, scene, sites  # noqa: E402


def _has_geopandas():
    try:
        import geopandas  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/terramind_water_decoder.pt", help="checkpoint (small or full)")
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
    ap.add_argument("--gee-project", default=None)
    o = ap.add_argument_group("outputs")
    o.add_argument("--write", nargs="*", default=None, choices=list(infer.ALL_OUTPUTS),
                   help="per-scene outputs (default: water flags preview, and gpkg when geopandas is installed)")
    o.add_argument("--threshold", type=float, default=0.5, help="water probability threshold (default 0.5)")
    o.add_argument("--max-slope", type=float, default=infer.MAX_SLOPE_DEG,
                   help="terrain safeguard: water on slopes steeper than this (deg) is removed and flagged; 0 = off")
    o.add_argument("--no-landcover", action="store_true", help="blank the land-cover channels")
    o.add_argument("--no-composite", action="store_true", help="skip frequency.tif / n_obs.tif")
    o.add_argument("--restart", action="store_true", help="reprocess dates already done")
    o.add_argument("--stop-on-error", action="store_true")
    a = ap.parse_args()

    model, meta, device = infer.load_model(a.model)
    print(f"model {a.model} on {device}: trained on {meta.get('sites')} {meta.get('start')}..{meta.get('end')}")
    try:
        jrc.init_gee(a.gee_project)
    except Exception as e:                                          # noqa: BLE001
        sys.exit(f"error: Earth Engine could not be initialised ({e.__class__.__name__}: {e}). It supplies the DEM and land cover: "
                 "run `earthengine authenticate` once and pass --gee-project <your project>.")
    write = tuple(a.write) if a.write is not None else ("water", "flags", "preview") + (("gpkg",) if _has_geopandas() else ())
    cfg = batch.MapConfig(start=a.start, end=a.end, out_dir=a.out, write=write, composite=not a.no_composite, resume=not a.restart,
                          stop_on_error=a.stop_on_error, cache_dir=a.cache, device=None, threshold=a.threshold,
                          max_slope_deg=a.max_slope or None, use_landcover=not a.no_landcover, scene=scene.SceneConfig())

    kw = {k: v for k, v in dict(name_col=a.name_col, kind=a.kind).items() if v}
    try:
        chosen = sites.resolve_aoi(a.aoi, a.names, a.multiple,
                                   **({"buffer_km": a.buffer_km, **kw} if a.aoi and not a.aoi.lower().endswith(".csv") else {}))
    except (ValueError, KeyError, FileNotFoundError, ImportError) as e:
        sys.exit(f"error: {e.args[0] if e.args else e}")
    print("mapping:", ", ".join(f"{s.label} ({s.kind})" for s in chosen))
    res = batch.map_sites(chosen, model, cfg)
    print(res.areas.groupby("site")[["water_km2"]].agg(["count", "mean", "max"]).round(3).to_string() if len(res.areas) else "no scenes mapped")
    if res.failures:
        print(f"{len(res.failures)} failure(s): see {os.path.join(a.out, 'failures.csv')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
