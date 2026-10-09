"""Fine-tune TerraMind, then fuse model + Otsu for every scene.

Two label sources:

  JRC-matched dataset (recommended; several sites; see scripts/build_dataset.py)
      python scripts/train_terramind.py --dataset outputs/dataset --holdout-sites magat

  Otsu-only pseudo-labels of one site (output of scripts/run_angat.py)
      python scripts/train_terramind.py --out outputs/angat --backbone terramind_v1_base

Validation: whole held-out sites (--holdout-sites; the honest generalisation test) or, without them, an east-strip block
of every scene so overlapping patches never straddle train/val.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from angat_water import pseudo_labels as pl, terramind_seg as tm  # noqa: E402


def _read(path):
    import rasterio
    with rasterio.open(path) as r:
        return r.read(1), r.profile


def train_on_dataset(a):
    from angat_water import dataset
    X, Y, metas = dataset.load_patches(a.dataset, size=a.patch, stride=a.patch // 2, keep_empty=a.keep_empty)
    if not len(X):
        raise SystemExit("no usable patches: check manifest.jsonl for the rejection reasons")
    tr, va = dataset.split_patches(metas, a.patch, a.holdout_sites)
    sites = sorted({m["site"] for m in metas})
    print(f"{len(X)} patches from {sites}: {len(tr)} train / {len(va)} val; water share of labelled "
          f"{(Y == 1).sum() / max(1, (Y != pl.IGNORE).sum()):.3f}")
    if not va:
        print("WARNING: no validation patches; the best epoch is chosen by training loss (optimistic).")
    model = tm.TerraMindWater(a.backbone, pretrained=not a.no_pretrained, unfreeze_last=a.unfreeze_last)
    tm.train(model, X, Y, tr, va, epochs=a.epochs, batch=a.batch, lr=a.lr)
    os.makedirs(a.out, exist_ok=True)
    tm.save_checkpoint(model, os.path.join(a.out, "terramind_water.pt"), {"dataset": a.dataset, "sites": sites})
    if va:
        per = {}
        for s in sorted({metas[i]["site"] for i in va}):
            idx = [i for i in va if metas[i]["site"] == s]
            import torch
            per[s] = tm.evaluate(model, torch.from_numpy(X), torch.from_numpy(Y.astype(np.int64)), idx)
        print("validation IoU(water) per site:", {k: round(v, 4) for k, v in per.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/angat", help="run_angat output (Otsu-only mode) / where to save the model")
    ap.add_argument("--dataset", default=None, help="JRC-matched dataset from build_dataset.py (replaces --out labels)")
    ap.add_argument("--holdout-sites", nargs="*", default=[], help="sites kept out of training and used for validation")
    ap.add_argument("--keep-empty", type=float, default=0.2, help="share of near-waterless patches kept")
    ap.add_argument("--backbone", default="terramind_v1_base")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (offline smoke test only)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--unfreeze-last", type=int, default=0, help="also fine-tune the last N transformer blocks")
    ap.add_argument("--patch", type=int, default=224)
    a = ap.parse_args()

    if a.dataset:
        return train_on_dataset(a)
    meta = json.load(open(os.path.join(a.out, "thresholds.json")))
    good = [r for r in meta["scenes"] if pl.scene_ok(r)]
    print(f"{len(good)}/{len(meta['scenes'])} scenes qualify as pseudo-label sources")
    Xs, Ys, shape = [], [], None
    for r in good:
        cls, _ = _read(os.path.join(a.out, f"{r['date']}_classes.tif"))
        vv, _ = _read(os.path.join(a.out, f"{r['date']}_db.tif"))
        vh, _ = _read(os.path.join(a.out, f"{r['date']}_vhdb.tif"))
        shape = cls.shape
        X, Y, C = pl.extract_patches(pl.normalise_s1(vv, vh), pl.labels_from_classes(cls), size=a.patch,
                                     stride=a.patch // 2, min_valid=0.9)
        Xs.append(X)
        Ys.append(Y)
        r["_corners"] = C
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    corners = [c for r in good for c in r["_corners"]]
    tr, va = pl.spatial_split(corners, shape, size=a.patch)
    print(f"{len(X)} patches: {len(tr)} train / {len(va)} val; water share {(Y == 1).mean():.3f}")
    if not va:
        print("WARNING: no validation patches (scene too narrow for the east-strip hold-out): the best epoch is "
              "chosen by training loss, so the reported fit is optimistic. Use a larger AOI or lower --patch.")

    model = tm.TerraMindWater(a.backbone, pretrained=not a.no_pretrained, unfreeze_last=a.unfreeze_last)
    tm.train(model, X, Y, tr, va, epochs=a.epochs, batch=a.batch, lr=a.lr)
    tm.save_checkpoint(model, os.path.join(a.out, "terramind_water.pt"), {"labels": "otsu"})

    import rasterio
    for r in meta["scenes"]:
        cls, prof = _read(os.path.join(a.out, f"{r['date']}_classes.tif"))
        vv, _ = _read(os.path.join(a.out, f"{r['date']}_db.tif"))
        vh, _ = _read(os.path.join(a.out, f"{r['date']}_vhdb.tif"))
        prob = tm.predict_scene(model, vv, vh)
        water, flags = tm.fuse(prob, cls)
        prof.update(dtype="uint8", count=1, nodata=None)
        with rasterio.open(os.path.join(a.out, f"{r['date']}_fused_flags.tif"), "w", **prof) as dst:
            dst.write(flags, 1)
        print(f"{r['date']}: fused water {water.sum() * 100 / 1e6:.2f} km2  "
              f"fm_only {int((flags == tm.FUSE_FM_ONLY).sum())} px  otsu_only {int((flags == tm.FUSE_OTSU_ONLY).sum())} px")


if __name__ == "__main__":
    main()
