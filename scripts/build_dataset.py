"""Build the JRC-labelled Sentinel-1 water dataset (needs Planetary Computer + Google Earth Engine).

    earthengine authenticate                                  # once
    python scripts/build_dataset.py --sites angat magat --start 2018-01-01 --end 2021-12-31 \
        --gee-project my-gcp-project --out outputs/dataset

    # a river reach or any other extra site
    python scripts/build_dataset.py --site-bbox pampanga_river 120.70 15.10 120.90 15.30 --kind river ...

Each S1 date is labelled with the JRC Monthly History map of the same calendar month (see angat_water/labels.py). The
manifest (<out>/manifest.jsonl) lists every scene, usable or rejected with the reason.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from angat_water import dataset, jrc, pipeline, sites  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*", default=["angat"], help=f"registered sites: {sorted(sites.SITES)}")
    ap.add_argument("--site-bbox", nargs=5, action="append", default=[], metavar=("NAME", "W", "S", "E", "N"))
    ap.add_argument("--kind", default="other", help="kind for --site-bbox sites (reservoir|river|lake|paddy|...)")
    ap.add_argument("--start", default="2017-03-01")
    ap.add_argument("--end", default="2021-12-31")
    ap.add_argument("--out", default="outputs/dataset")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--gee-project", default=None)
    ap.add_argument("--max-month-offset", type=int, default=0,
                    help="0 = JRC month must be the S1 month; 1 allows the neighbouring month if that one is cloudy")
    ap.add_argument("--edge-px", type=int, default=3)
    ap.add_argument("--never-water-negatives", action="store_true",
                    help="label JRC-never-water pixels that S1 calls water as LAND (radar shadow); off by default")
    ap.add_argument("--min-agreement", type=float, default=0.9)
    a = ap.parse_args()

    for name, w, s, e, n in a.site_bbox:
        sites.custom(name, (w, s, e, n), kind=a.kind)
        a.sites.append(name)

    cfg = dataset.DatasetConfig(out_dir=a.out, start=a.start, end=a.end, cache_dir=a.cache,
                                max_month_offset=a.max_month_offset, edge_px=a.edge_px,
                                never_water_negatives=a.never_water_negatives, min_agreement=a.min_agreement,
                                water=pipeline.WaterConfig(cache_dir=a.cache))
    jrc.init_gee(a.gee_project)
    source = jrc.GeeJrc()
    os.makedirs(a.out, exist_ok=True)
    for name in a.sites:
        site = sites.get(name)
        cfg.water.bbox, cfg.water.crs = site.bbox, site.crs
        grid, scenes = dataset.iter_s1(site, cfg)
        metas = dataset.build_site(site, cfg, scenes, grid, source)
        used = sum(1 for m in metas if m.get("usable"))
        print(f"[{site.name}] {used}/{len(metas)} scenes usable")


if __name__ == "__main__":
    main()
