"""Fine-tune TerraMind on the JRC-labelled dataset (scripts/build_dataset.py).

    python scripts/train_terramind.py --dataset outputs/dataset --holdout-sites "La Mesa Dam"

The model reads VV, VH, SDWI, terrain (relief, slope, aspect, radar-look slope) and land-cover groups (ai_water/features.py).
Validation: whole held-out sites (--holdout-sites; the honest generalisation test) or, without them, an east-strip block
of every scene so overlapping patches never straddle train/val.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_water import dataset, terramind_seg as tm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset from build_dataset.py")
    ap.add_argument("--out", default="outputs/model", help="where to save the checkpoints")
    ap.add_argument("--holdout-sites", nargs="*", default=[], help="sites kept out of training and used for validation")
    ap.add_argument("--keep-empty", type=float, default=0.2, help="share of near-waterless patches kept")
    ap.add_argument("--max-per-scene", type=int, default=32, help="random patches kept per scene (bounds memory)")
    ap.add_argument("--backbone", default="terramind_v1_base")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (offline smoke test only)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--unfreeze-last", type=int, default=0, help="also fine-tune the last N transformer blocks")
    ap.add_argument("--patch", type=int, default=224)
    a = ap.parse_args()

    ps = dataset.load_patches(a.dataset, size=a.patch, stride=a.patch // 2, keep_empty=a.keep_empty, max_per_scene=a.max_per_scene)
    if not len(ps):
        raise SystemExit("no usable patches: check manifest.jsonl for the rejection reasons")
    tr, va = dataset.split_patches(ps.metas, a.patch, a.holdout_sites)
    sites = sorted({m["site"] for m in ps.metas})
    orbits = sorted({m["orbit"] for m in ps.metas})
    lab = ps.Y != tm.IGNORE
    print(f"{len(ps)} patches from {sites} ({', '.join(orbits)}): {len(tr)} train / {len(va)} val; "
          f"water share of labelled {(ps.Y == 1).sum() / max(1, lab.sum()):.3f}")
    if not va:
        print("WARNING: no validation patches; the best epoch is chosen by training loss (optimistic).")
    model = tm.TerraMindWater(a.backbone, pretrained=not a.no_pretrained, unfreeze_last=a.unfreeze_last)
    tm.train(model, ps, tr, va, epochs=a.epochs, batch=a.batch, lr=a.lr)
    os.makedirs(a.out, exist_ok=True)
    meta = {"dataset": a.dataset, "sites": sites, "orbits": orbits, "backbone": a.backbone,
            "pretrained_encoder": not a.no_pretrained, "holdout": a.holdout_sites, "epochs": a.epochs}
    tm.save_checkpoint(model, os.path.join(a.out, "terramind_water.pt"), meta)                           # complete (large)
    tm.save_checkpoint(model, os.path.join(a.out, "terramind_water_decoder.pt"), meta, trainable_only=True)   # small: fine for git
    for f in ("terramind_water.pt", "terramind_water_decoder.pt"):
        print(f"saved {os.path.join(a.out, f)}  ({os.path.getsize(os.path.join(a.out, f)) / 1e6:.1f} MB)")
    if va:
        per = {s: tm.evaluate(model, ps, [i for i in va if ps.metas[i]["site"] == s])
               for s in sorted({ps.metas[i]["site"] for i in va})}
        print("validation IoU(water) per site:", {k: round(v, 4) for k, v in per.items()})


if __name__ == "__main__":
    main()
