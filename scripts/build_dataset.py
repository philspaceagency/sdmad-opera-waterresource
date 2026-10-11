"""Build the JRC-labelled Sentinel-1 water dataset (needs Planetary Computer + Google Earth Engine).

    earthengine authenticate                                  # once
    python scripts/build_dataset.py --gee-project my-gcp-project --start 2018-01-01 --end 2021-12-31

By default every site in data/training_sites.csv is used (3 dams, 1 lake, 2 river reaches); pick some with --sites NAME ..., use
another file (CSV, or a vector file of outlines) with --sites-file, or add a box with --site-bbox:

    python scripts/build_dataset.py --sites "Angat Dam" "Magat Dam" ...
    python scripts/build_dataset.py --site-bbox pampanga_river 120.70 15.10 120.90 15.30 --kind river ...

Each S1 date (ascending and descending passes alike) is labelled with the JRC Monthly History map of the same calendar month,
corrected with the terrain (see ai_water/labels.py); the DEM and land cover come from Earth Engine too. The manifest
(<out>/manifest.jsonl) lists every scene, usable or rejected with the reason. Re-running resumes: scenes already in the manifest are
skipped without being read again (--restart starts over).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_water import dataset, jrc, sites, terrain  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites-file", default=None, help="sites CSV or vector file (default: data/training_sites.csv)")
    ap.add_argument("--sites", nargs="*", default=None, help="only these sites of the file (by name)")
    ap.add_argument("--site-bbox", nargs=5, action="append", default=[], metavar=("NAME", "W", "S", "E", "N"), help="add a lon/lat box")
    ap.add_argument("--kind", default="other", help="kind for --site-bbox sites (reservoir|lake|river|...)")
    ap.add_argument("--start", default="2017-03-01")
    ap.add_argument("--end", default="2021-12-31")
    ap.add_argument("--out", default="outputs/dataset")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--gee-project", default=None)
    ap.add_argument("--restart", action="store_true", help="delete the manifest and rebuild from scratch (default: resume)")
    ap.add_argument("--max-month-offset", type=int, default=0,
                    help="0 = JRC month must be the S1 month; 1 allows the neighbouring month if that one is cloudy")
    ap.add_argument("--edge-px", type=int, default=3)
    ap.add_argument("--max-water-slope", type=float, default=15.0, help="JRC water on a steeper slope (deg) is ignored")
    ap.add_argument("--dark-db", type=float, default=-18.0, help="JRC land darker than this in the radar = hard negative")
    ap.add_argument("--min-contrast-db", type=float, default=5.0, help="scenes with less JRC-water vs JRC-land contrast are rejected")
    ap.add_argument("--no-landcover", action="store_true", help="leave the land-cover layer empty")
    a = ap.parse_args()

    found = sites.load_sites(a.sites_file) if a.sites_file else sites.load_training_sites(register_them=False)
    sites.register(found)
    chosen = [sites.get(n) for n in a.sites] if a.sites else list(found)
    for name, w, s, e, n in a.site_bbox:
        chosen.append(sites.custom(name, (float(w), float(s), float(e), float(n)), kind=a.kind))
    print("sites:", ", ".join(f"{s.label} ({s.kind}, {s.size_km()[0]:.0f}x{s.size_km()[1]:.0f} km)" for s in chosen))

    cfg = dataset.DatasetConfig(out_dir=a.out, start=a.start, end=a.end, cache_dir=a.cache,
                                max_month_offset=a.max_month_offset, edge_px=a.edge_px,
                                max_water_slope=a.max_water_slope, dark_db=a.dark_db, min_contrast_db=a.min_contrast_db,
                                use_landcover=not a.no_landcover)
    os.makedirs(a.out, exist_ok=True)
    manifest = os.path.join(a.out, "manifest.jsonl")
    if a.restart and os.path.exists(manifest):
        os.remove(manifest)
        print("--restart: manifest deleted")
    jrc.init_gee(a.gee_project)
    source, terr = jrc.GeeJrc(), terrain.GeeTerrain()
    for site in chosen:
        done = dataset.done_scenes(a.out, site.name)
        if done:
            print(f"[{site.name}] resuming: {len(done)} scene(s) already in the manifest")
        grid, scenes = dataset.iter_s1(site, cfg, skip_days=done)
        metas = dataset.build_site(site, cfg, scenes, grid, source, terr)
        used = sum(1 for m in metas if m.get("usable"))
        print(f"[{site.name}] {used}/{len(metas)} new scenes usable")


if __name__ == "__main__":
    main()
