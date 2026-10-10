"""Build the JRC-labelled Sentinel-1 water dataset (needs Planetary Computer + Google Earth Engine).

    earthengine authenticate                                  # once
    python scripts/build_dataset.py --gee-project my-gcp-project --start 2018-01-01 --end 2021-12-31

By default every site in data/training_sites.csv is used (3 dams, 1 lake, 2 river reaches); pick some with --sites NAME ..., use
another file (CSV, or a vector file of outlines) with --sites-file, or add a box with --site-bbox:

    python scripts/build_dataset.py --sites "Angat Dam" "Magat Dam" ...
    python scripts/build_dataset.py --site-bbox pampanga_river 120.70 15.10 120.90 15.30 --kind river ...

Each S1 date is labelled with the JRC Monthly History map of the same calendar month (see ai_water/labels.py). The manifest
(<out>/manifest.jsonl) lists every scene, usable or rejected with the reason. Re-running resumes: scenes already in the manifest are
skipped without being read again (--restart starts over).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_water import dataset, jrc, pipeline, sites  # noqa: E402


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
    ap.add_argument("--never-water-negatives", action="store_true",
                    help="label JRC-never-water pixels that S1 calls water as LAND (radar shadow); off by default")
    ap.add_argument("--min-agreement", type=float, default=0.9)
    a = ap.parse_args()

    found = sites.load_sites(a.sites_file) if a.sites_file else sites.load_training_sites(register_them=False)
    sites.register(found)
    chosen = [sites.get(n) for n in a.sites] if a.sites else list(found)
    for name, w, s, e, n in a.site_bbox:
        chosen.append(sites.custom(name, (float(w), float(s), float(e), float(n)), kind=a.kind))
    print("sites:", ", ".join(f"{s.label} ({s.kind}, {s.size_km()[0]:.0f}x{s.size_km()[1]:.0f} km)" for s in chosen))

    cfg = dataset.DatasetConfig(out_dir=a.out, start=a.start, end=a.end, cache_dir=a.cache,
                                max_month_offset=a.max_month_offset, edge_px=a.edge_px,
                                never_water_negatives=a.never_water_negatives, min_agreement=a.min_agreement,
                                water=pipeline.WaterConfig(cache_dir=a.cache))
    os.makedirs(a.out, exist_ok=True)
    manifest = os.path.join(a.out, "manifest.jsonl")
    if a.restart and os.path.exists(manifest):
        os.remove(manifest)
        print("--restart: manifest deleted")
    jrc.init_gee(a.gee_project)
    source = jrc.GeeJrc()
    for site in chosen:
        cfg.water.bbox, cfg.water.crs = site.bbox, site.crs
        done = dataset.done_days(a.out, site.name)
        if done:
            print(f"[{site.name}] resuming: {len(done)} scene(s) already in the manifest")
        grid, scenes = dataset.iter_s1(site, cfg, skip_days=done)
        metas = dataset.build_site(site, cfg, scenes, grid, source)
        used = sum(1 for m in metas if m.get("usable"))
        print(f"[{site.name}] {used}/{len(metas)} new scenes usable")


if __name__ == "__main__":
    main()
