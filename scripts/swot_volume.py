"""SWOT level + Sentinel-1 area -> level, area and storage volume of a reservoir (default: Angat Dam).

    python scripts/swot_volume.py --areas outputs/maps/areas.csv --start 2023-08-01 --end 2025-06-30
    python scripts/swot_volume.py --areas ... --swot-dir cache/swot          # reuse downloaded shapefiles, no login
    python scripts/swot_volume.py ... --h-ref 180 --v-ref 120                # absolute: 120 million m3 at 180 m (from the dam operator)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import glob  # noqa: E402

try:                                              # inside sdmad-opera-waterresource; standalone (wateraltimeter) it falls back to Angat
    from ai_water import sites  # noqa: E402
except ImportError:
    sites = None
from swot_volume import integrate, swot  # noqa: E402

ANGAT_NAME = "angat_dam"
ANGAT_BBOX = (121.14302981543655, 14.872088697247067, 121.26233447214564, 15.053187085370627)   # west, south, east, north


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--areas", required=True, help="areas.csv written by map_water.py")
    ap.add_argument("--site", default=ANGAT_NAME)
    ap.add_argument("--aoi", help="vector file with the reservoir outline, to pick SWOT lakes (default: the site box)")
    ap.add_argument("--start", default="2023-08-01")
    ap.add_argument("--end", default=__import__("datetime").date.today().isoformat())
    ap.add_argument("--swot-dir", default="cache/swot", help="where SWOT shapefiles are downloaded / already are")
    ap.add_argument("--max-lag-days", type=int, default=6)
    ap.add_argument("--max-quality", type=int, default=1, help="keep quality_f <= this (0 good, 1 suspect)")
    ap.add_argument("--lake-ids", nargs="*", help="keep only these SWOT lake_id values (see the lake summary)")
    ap.add_argument("--min-area-km2", type=float, default=1.0)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--method", default="isotonic", choices=["isotonic", "power"])
    ap.add_argument("--h-ref", type=float, help="level (m, EGM2008) at which --v-ref holds")
    ap.add_argument("--v-ref", type=float, help="storage (million m3) at --h-ref: makes the volume absolute")
    ap.add_argument("--out", default="outputs/swot_volume")
    a = ap.parse_args()

    site = sites.get(a.site) if sites else None
    bbox = site.bbox if site else ANGAT_BBOX
    site_name = site.name if site else a.site
    outline = None
    if a.aoi:
        if sites is None:
            sys.exit("--aoi needs the ai_water package (sdmad-opera-waterresource)")
        outline = sites.sites_from_vector(a.aoi)[0].geometry
    print("searching SWOT L2_HR_LakeSP ...")
    try:
        paths = swot.download(swot.search(bbox, a.start, a.end), a.swot_dir, kind="Obs")      # files already in swot-dir are not fetched again
    except Exception as e:                                                                       # offline / no login: use what is on disk
        print("search failed (", e, ") - using shapefiles already in", a.swot_dir)
        paths = sorted(glob.glob(os.path.join(a.swot_dir, "**", "*_Obs_*.shp"), recursive=True))
    g = swot.read_shapefiles(paths, reservoir=outline, bbox=bbox)
    print(swot.funnel(g, a.max_quality, 1, a.allow_partial).to_string(index=False))
    print(swot.lake_summary(g).to_string(index=False))
    obs = swot.observations_from_frames(g, max_quality=a.max_quality, allow_partial=a.allow_partial, lake_ids=a.lake_ids, min_area_km2=a.min_area_km2)
    obs = obs[(obs["date"] >= a.start) & (obs["date"] <= a.end)]
    if obs.empty:
        sys.exit("no usable SWOT observations of the reservoir in that window")
    table, curve = integrate.compute(obs, integrate.s1_daily_area(a.areas, site=site_name), a.max_lag_days, a.method, a.h_ref, a.v_ref)
    os.makedirs(a.out, exist_ok=True)
    table.to_csv(os.path.join(a.out, "level_area_volume.csv"), index=False)
    print(table.to_string(index=False))
    print(f"\n{len(obs)} SWOT passes, {table['s1_area_km2'].notna().sum()} paired with Sentinel-1; area-curve RMSE {integrate.rmse_of_curve(table):.2f} km2")


if __name__ == "__main__":
    main()
