"""Build and load the JRC-labelled Sentinel-1 water dataset.

For every Sentinel-1 date of a site:
    S1 RTC (VV, VH) -> refined Lee -> dB -> edge+binary Otsu classes          (pipeline.map_water)
    JRC monthly map of the SAME calendar month, on the S1 grid               (jrc.matched_month)
    -> tiered labels + per-pixel source + agreement statistics                (labels.jrc_otsu_labels)
and one `<out>/<site>/<date>.npz` plus a line in `<out>/manifest.jsonl`. Rejected scenes are in the manifest too, with
the reason, so the dataset can be audited (and the thresholds for rejection revisited) without re-reading anything.

Only dates inside JRC's coverage (GSW1_4: up to Dec 2021) and Sentinel-1 RTC's (from 2017) can be labelled this way.
A model trained on them is applied to any later date: the labels need JRC, the model does not.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import acquire, jrc, labels as lb, pipeline, pseudo_labels as pl, sites as sites_mod


@dataclass
class DatasetConfig:
    out_dir: str = "outputs/dataset"
    start: str = "2017-03-01"                 # Sentinel-1 RTC on Planetary Computer starts in 2017
    end: str = "2021-12-31"                   # JRC GSW1_4 monthly history ends here
    max_month_offset: int = 0                 # 0 = JRC month must be the S1 month
    min_jrc_valid: float = 0.6                # share of the AOI observed by Landsat that month
    edge_px: int = 3
    resolve_uncertain: bool = True
    edge_policy: str = "otsu"
    never_water_negatives: bool = False
    min_agreement: float = 0.9
    min_water_px: int = 2000
    max_jrc_nodata: float = 0.4
    water: pipeline.WaterConfig = field(default_factory=pipeline.WaterConfig)
    cache_dir: str = "cache"

    def label_kw(self):
        return dict(edge_px=self.edge_px, resolve_uncertain=self.resolve_uncertain, edge_policy=self.edge_policy,
                    never_water_negatives=self.never_water_negatives)


# ---------------------------------------------------------------------------------------- one scene (pure)
def make_scene(vv_lin, vh_lin, day, matched, occurrence, cfg: DatasetConfig):
    """Labels for one S1 scene. Returns (arrays dict, meta dict). `matched` is a jrc.MatchedMonth."""
    wm = pipeline.map_water(vv_lin, vh_lin, cfg.water)
    lab, src, stats = lb.jrc_otsu_labels(wm.classes, matched.water_class, occurrence, **cfg.label_kw())
    ok = lb.scene_quality(stats, cfg.min_agreement, cfg.min_water_px, cfg.max_jrc_nodata)
    reason = None
    if not ok:
        # the root cause first: no JRC coverage > JRC and Otsu disagree (wrong month / bad scene) > no water body
        reason = ("jrc_nodata" if stats["jrc_nodata_frac"] > cfg.max_jrc_nodata else
                  "low_agreement" if stats["agreement"] < cfg.min_agreement else "few_water_px")
    meta = {"date": str(pd.Timestamp(day).date()), "jrc_month": matched.month.strftime("%Y-%m"),
            "month_offset": matched.offset_months, "jrc_valid_frac": round(matched.valid_frac, 4),
            "otsu": wm.summary(cfg.water.resolution_m ** 2), "labels": stats, "usable": ok, "reject_reason": reason,
            "shape": list(lab.shape)}
    arrays = {"vv_db": wm.db.astype(np.float32), "vh_db": wm.vh_db.astype(np.float32), "labels": lab, "source": src}
    return arrays, meta


# ---------------------------------------------------------------------------------------- one site
def iter_s1(site, cfg: DatasetConfig, log=print, skip_days=None):
    """(jrc grid, generator of (solar day, vv_lin, vh_lin)) from Planetary Computer for the site, restricted to the dataset
    window and to JRC's coverage. Only one orbit; days whose JRC month is outside coverage are not even read."""
    start = max(pd.Timestamp(cfg.start), jrc.JRC_FIRST)
    end = min(pd.Timestamp(cfg.end), jrc.JRC_LAST + pd.offsets.MonthEnd(0))
    log(f"[{site.name}] {start.date()}..{end.date()}")
    gb, gen = acquire.iter_scenes(site.bbox, site.crs, cfg.water.resolution_m, start, end,
                                  cfg.water.min_orbit_cover, cfg.cache_dir, log, skip_days)
    return jrc.grid_from_geobox(gb), gen


def build_site(site, cfg: DatasetConfig, scenes, grid, jrc_source, log=print):
    """Label every scene of `scenes` (iterable of (day, vv_lin, vh_lin)) and write npz + manifest lines.
    `jrc_source` has .month(month, grid) and .occurrence(grid) (jrc.GeeJrc or a stand-in). Returns the metas."""
    os.makedirs(os.path.join(cfg.out_dir, site.name), exist_ok=True)
    occ = jrc_source.occurrence(grid) if cfg.never_water_negatives else None
    cache, metas = {}, []
    with open(os.path.join(cfg.out_dir, "manifest.jsonl"), "a") as man:
        for day, vv, vh in scenes:
            mm = jrc.matched_month(jrc_source, grid, day, cfg.max_month_offset, cfg.min_jrc_valid, cache)
            if mm is None:
                meta = {"site": site.name, "date": str(pd.Timestamp(day).date()), "usable": False,
                        "reject_reason": "no_jrc_month"}
                log(f"  {meta['date']}: no usable JRC month")
            else:
                arrays, meta = make_scene(vv, vh, day, mm, occ, cfg)
                meta.update(site=site.name, kind=site.kind)
                if meta["usable"]:
                    np.savez_compressed(os.path.join(cfg.out_dir, site.name, f"{meta['date']}.npz"), **arrays)
                st = meta["labels"]
                log(f"  {meta['date']} (JRC {meta['jrc_month']}, off {meta['month_offset']:+d}): "
                    f"agreement {st['agreement']:.3f}, water {st['water_px']} px -> "
                    f"{'USE' if meta['usable'] else 'reject: ' + meta['reject_reason']}")
            man.write(json.dumps(meta, default=str) + "\n")
            man.flush()
            metas.append(meta)
    if not metas:
        earlier = len(done_days(cfg.out_dir, site.name))
        if earlier:
            log(f"[{site.name}] nothing new to label: all {earlier} scene(s) are already in the manifest (resumed run).")
        else:
            log(f"WARNING [{site.name}]: no Sentinel-1 scene was read, so nothing was labelled. Look above for 'unreadable' "
                "lines or an error from Planetary Computer (token / rate limit), and check the dates and the site box.")
    return metas


# ---------------------------------------------------------------------------------------- loading
def done_days(out_dir, site_name):
    """{'YYYY-MM-DD'} of the scenes of `site_name` already in the manifest (labelled or rejected): pass it as `skip_days` to
    resume an interrupted build without re-reading those passes. Empty if there is no manifest yet."""
    path = os.path.join(out_dir, "manifest.jsonl")
    if not os.path.exists(path):
        return set()
    return {m["date"] for m in read_manifest(out_dir) if m.get("site") == site_name and "date" in m}


def read_manifest(out_dir):
    with open(os.path.join(out_dir, "manifest.jsonl")) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def manifest_table(out_dir, res_m=10.0):
    """The manifest as a pandas DataFrame (one row per scene: site, date, JRC month, offset, agreement, water km2, JRC no-data,
    usable, reject_reason). Raises RuntimeError with the likely cause if there is nothing to show or nothing is usable."""
    path = os.path.join(out_dir, "manifest.jsonl")
    if not os.path.exists(path):
        raise RuntimeError(f"{path} does not exist: run the dataset-building step first (and check it finished).")
    man = read_manifest(out_dir)
    if not man:
        raise RuntimeError(
            f"{path} is empty: the dataset step labelled no scenes. Typical causes: no Sentinel-1 scene could be read "
            "(Planetary Computer token / rate limit: set PC_SDK_SUBSCRIPTION_KEY), the dataset cell stopped with an error "
            "earlier and was not re-run, or the dates / site box contain no pass. Read the output of the dataset cell.")
    tbl = pd.DataFrame([{
        "site": m.get("site"), "date": m["date"], "jrc_month": m.get("jrc_month"), "offset": m.get("month_offset"),
        "agreement": m.get("labels", {}).get("agreement"),
        "water_km2": (m["labels"]["water_px"] * res_m ** 2 / 1e6) if "labels" in m else None,
        "jrc_nodata": m.get("labels", {}).get("jrc_nodata_frac"), "usable": bool(m.get("usable")),
        "reject_reason": m.get("reject_reason")} for m in man])
    if not tbl["usable"].any():
        reasons = tbl["reject_reason"].value_counts().to_dict()
        raise RuntimeError(
            f"all {len(tbl)} scenes were rejected {reasons}. low_agreement: the JRC month does not match the pass "
            "(try max_month_offset=1) or Otsu failed on the scene; no_jrc_month: Landsat cloud gaps that month; "
            "jrc_nodata: too little JRC coverage; few_water_px: no water body in the scene. Lower min_agreement only "
            "after looking at the scenes.")
    return tbl


def load_patches(out_dir, size=224, stride=112, sites=None, min_labelled=0.5, min_water=0.005, keep_empty=0.2,
                 max_month_offset=0, seed=0, max_per_scene=None, dtype=np.float32):
    """Training patches from the usable scenes. Returns (X (n,2,s,s) float32, Y (n,s,s) uint8, meta list).

    Patches with less than `min_water` water share (of labelled pixels) are mostly land and would swamp the rare water
    class (rivers, small tanks): only a random `keep_empty` fraction of them is kept. `max_month_offset` drops scenes
    whose JRC month was not the S1 month.

    Memory: a 2000 x 2000 px scene gives hundreds of overlapping 224 px patches (about 0.4 MB each as float32). All patches of
    all scenes are held in RAM, so for a real run set `max_per_scene` (random subsample, e.g. 32) and `dtype=np.float16`
    (halves it; the training loop casts each batch back to float32)."""
    rng = np.random.default_rng(seed)
    Xs, Ys, metas = [], [], []
    for m in read_manifest(out_dir):
        if not m.get("usable") or (sites and m["site"] not in sites) or abs(m.get("month_offset", 0)) > max_month_offset:
            continue
        z = np.load(os.path.join(out_dir, m["site"], f"{m['date']}.npz"))
        x = pl.normalise_s1(z["vv_db"], z["vh_db"])
        X, Y, C = pl.extract_patches(x, z["labels"], size=size, stride=stride, min_valid=min_labelled)
        keep = []
        for i in range(len(C)):
            lab = Y[i][Y[i] != pl.IGNORE]
            wf = float((lab == 1).mean()) if lab.size else 0.0
            if wf < min_water and rng.random() > keep_empty:
                continue
            keep.append((i, wf))
        if max_per_scene and len(keep) > max_per_scene:
            keep = [keep[j] for j in sorted(rng.choice(len(keep), max_per_scene, replace=False))]
        for i, wf in keep:
            Xs.append(X[i].astype(dtype, copy=False))
            Ys.append(Y[i])
            metas.append({"site": m["site"], "date": m["date"], "corner": C[i], "shape": m["shape"], "water_frac": wf})
    if not Xs:
        return np.zeros((0, 2, size, size), dtype), np.zeros((0, size, size), np.uint8), []
    return np.stack(Xs), np.stack(Ys), metas


def split_patches(metas, size=224, holdout_sites=(), val_frac=0.25):
    """(train idx, val idx). With `holdout_sites` the validation set is those whole sites (the honest generalisation
    test: Magat never seen in training). Otherwise an east-strip block of every scene, so overlapping neighbours
    never straddle the split."""
    if holdout_sites:
        hs = {sites_mod.slug(s) for s in holdout_sites}                  # 'Magat Dam', 'magat dam' and 'magat_dam' all match
        val = [i for i, m in enumerate(metas) if m["site"] in hs]
        train = [i for i, m in enumerate(metas) if m["site"] not in hs]
        return train, val
    train, val = [], []
    for i, m in enumerate(metas):
        cut = int(m["shape"][1] * (1.0 - val_frac))
        c = m["corner"][1]
        if c >= cut:
            val.append(i)
        elif c + size <= cut:
            train.append(i)
    return train, val
