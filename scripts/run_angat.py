"""Run the Otsu (edge + binary) water mapping over Angat Dam and its reservoir.

    python scripts/run_angat.py --start 2024-01-01 --end 2024-12-31 --out outputs/angat
    python scripts/run_angat.py --demo        # offline synthetic scene, no network
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_water import pipeline, synthetic  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--out", default="outputs/angat")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), help="override the Angat bbox")
    ap.add_argument("--band", choices=("vv", "vh"), default="vv")
    ap.add_argument("--speckle", choices=("refined_lee", "lee", "none"), default="refined_lee")
    ap.add_argument("--demo", action="store_true", help="synthetic scene, offline")
    a = ap.parse_args()

    cfg = pipeline.WaterConfig(start=a.start, end=a.end, out_dir=a.out, cache_dir=a.cache, band=a.band,
                               speckle_filter=a.speckle, **({"bbox": tuple(a.bbox)} if a.bbox else {}))
    if a.demo:
        s = synthetic.make_scene(water_frac=0.08, looks=4, seed=3)
        wm = pipeline.map_water(s["vv"], s["vh"], cfg)
        print(wm.summary(), "IoU vs truth:", round(synthetic.iou(wm.water, s["truth"]), 3))
        return
    pipeline.run_area(cfg)


if __name__ == "__main__":
    main()
