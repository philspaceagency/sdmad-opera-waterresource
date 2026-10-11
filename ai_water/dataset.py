"""Build and load the JRC-labelled Sentinel-1 water dataset.

For every Sentinel-1 date of a site (ascending and descending passes alike):
    S1 RTC (VV, VH) -> refined Lee -> VV dB, VH dB, SDWI                  (scene.prepare)
    JRC monthly map of the SAME calendar month, on the S1 grid            (jrc.matched_month)
    DEM -> relief, slope, aspect; WorldCover -> land-cover groups         (terrain.site_terrain, once per site)
    -> labels from JRC, checked against the radar and the terrain         (labels.jrc_labels)
and one `<out>/<site>/<date>_<orbit>.npz` plus a line in `<out>/manifest.jsonl`. Rejected scenes are in the manifest too, with
the reason, so the dataset can be audited without re-reading anything. `<out>/<site>/terrain.npz` holds the site's terrain.

Only dates inside JRC's coverage (GSW1_4: up to Dec 2021) and Sentinel-1 RTC's (from 2017) can be labelled this way.
A model trained on them is applied to any later date: the labels need JRC, the model does not.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import acquire, features, jrc, labels as lb, scene as sc, sites as sites_mod, terrain as tr
from .features import IGNORE


@dataclass
class DatasetConfig:
    out_dir: str = "outputs/dataset"
    start: str = "2017-03-01"                 # Sentinel-1 RTC on Planetary Computer starts in 2017
    end: str = "2021-12-31"                   # JRC GSW1_4 monthly history ends here
    max_month_offset: int = 0                 # 0 = JRC month must be the S1 month
    min_jrc_valid: float = 0.6                # share of the AOI observed by Landsat that month
    edge_px: int = 3
    max_water_slope: float = 15.0             # JRC water on a steeper slope (deg) is not trusted
    dark_db: float = -18.0                    # JRC land darker than this in the radar = hard negative
    min_contrast_db: float = 5.0
    min_water_px: int = 2000
    max_jrc_nodata: float = 0.4
    use_landcover: bool = True
    scene: sc.SceneConfig = field(default_factory=sc.SceneConfig)
    cache_dir: str = "cache"

    def label_kw(self):
        return dict(edge_px=self.edge_px, max_water_slope=self.max_water_slope, dark_db=self.dark_db)


# ---------------------------------------------------------------------------------------- one scene (pure)
def make_scene(vv_lin, vh_lin, day, orbit_state, matched, terrain: features.Terrain, cfg: DatasetConfig):
    """Features and labels for one S1 scene. Returns (arrays dict, meta dict). `matched` is a jrc.MatchedMonth."""
    f = sc.prepare(vv_lin, vh_lin, cfg.scene)
    lab, src, stats = lb.jrc_labels(matched.water_class, f.vv_db, terrain.slope, **cfg.label_kw())
    ok = lb.scene_quality(stats, cfg.min_contrast_db, cfg.min_water_px, cfg.max_jrc_nodata)
    reason = None
    if not ok:
        # the root cause first: no JRC coverage > no water body > JRC does not fit the pass
        reason = ("jrc_nodata" if stats["jrc_nodata_frac"] > cfg.max_jrc_nodata else
                  "few_water_px" if stats["water_px"] < cfg.min_water_px else "low_contrast")
    meta = {"date": str(pd.Timestamp(day).date()), "orbit": orbit_state, "jrc_month": matched.month.strftime("%Y-%m"),
            "month_offset": matched.offset_months, "jrc_valid_frac": round(matched.valid_frac, 4),
            "labels": stats, "usable": ok, "reject_reason": reason, "shape": list(lab.shape)}
    arrays = {"vv_db": f.vv_db.astype(np.float32), "vh_db": f.vh_db.astype(np.float32), "sdwi": f.sdwi.astype(np.float32),
              "labels": lab, "source": src}
    return arrays, meta


# ---------------------------------------------------------------------------------------- one site
def iter_s1(site, cfg: DatasetConfig, log=print, skip_days=None):
    """(jrc grid, generator of (solar day, orbit state, vv_lin, vh_lin)) from Planetary Computer for the site, restricted to the
    dataset window and to JRC's coverage. One relative orbit per pass direction; days whose JRC month is outside coverage are
    not even read. `skip_days`: keys from `done_scenes`."""
    start = max(pd.Timestamp(cfg.start), jrc.JRC_FIRST)
    end = min(pd.Timestamp(cfg.end), jrc.JRC_LAST + pd.offsets.MonthEnd(0))
    log(f"[{site.name}] {start.date()}..{end.date()}")
    gb, gen = acquire.iter_scenes(site.bbox, site.crs, cfg.scene.resolution_m, start, end,
                                  cfg.scene.min_orbit_cover, cfg.cache_dir, log, skip_days)
    return jrc.grid_from_geobox(gb), gen


def build_site(site, cfg: DatasetConfig, scenes, grid, jrc_source, terrain_source, log=print):
    """Label every scene of `scenes` (iterable of (day, orbit state, vv_lin, vh_lin)) and write npz + manifest lines.
    `jrc_source` has .month(month, grid); `terrain_source` has .dem(grid) and .landcover(grid) (jrc.GeeJrc / terrain.GeeTerrain or
    stand-ins). Returns the metas."""
    sdir = os.path.join(cfg.out_dir, site.name)
    os.makedirs(sdir, exist_ok=True)
    terrain = tr.site_terrain(grid, terrain_source, cfg.cache_dir, cfg.use_landcover)
    terrain.save(os.path.join(sdir, "terrain.npz"))
    cache, metas = {}, []
    with open(os.path.join(cfg.out_dir, "manifest.jsonl"), "a") as man:
        for day, state, vv, vh in scenes:
            mm = jrc.matched_month(jrc_source, grid, day, cfg.max_month_offset, cfg.min_jrc_valid, cache)
            if mm is None:
                meta = {"site": site.name, "date": str(pd.Timestamp(day).date()), "orbit": state, "usable": False,
                        "reject_reason": "no_jrc_month"}
                log(f"  {meta['date']} {state}: no usable JRC month")
            else:
                arrays, meta = make_scene(vv, vh, day, state, mm, terrain, cfg)
                meta.update(site=site.name, kind=site.kind)
                if meta["usable"]:
                    np.savez_compressed(os.path.join(sdir, f"{meta['date']}_{state}.npz"), **arrays)
                st = meta["labels"]
                log(f"  {meta['date']} {state} (JRC {meta['jrc_month']}, off {meta['month_offset']:+d}): contrast "
                    f"{st['contrast_db']:.1f} dB, water {st['water_px']} px, hard negatives {st['hard_negative_px']} px -> "
                    f"{'USE' if meta['usable'] else 'reject: ' + meta['reject_reason']}")
            man.write(json.dumps(meta, default=str) + "\n")
            man.flush()
            metas.append(meta)
    if not metas:
        earlier = len(done_scenes(cfg.out_dir, site.name))
        if earlier:
            log(f"[{site.name}] nothing new to label: all {earlier} scene(s) are already in the manifest (resumed run).")
        else:
            log(f"WARNING [{site.name}]: no Sentinel-1 scene was read, so nothing was labelled. Look above for 'unreadable' "
                "lines or an error from Planetary Computer (token / rate limit), and check the dates and the site box.")
    return metas


# ---------------------------------------------------------------------------------------- loading
def done_scenes(out_dir, site_name):
    """{'YYYY-MM-DD:orbit'} of the scenes of `site_name` already in the manifest (labelled or rejected): pass it as `skip_days` to
    resume an interrupted build without re-reading those passes. Empty if there is no manifest yet."""
    path = os.path.join(out_dir, "manifest.jsonl")
    if not os.path.exists(path):
        return set()
    return {acquire.scene_key(m["date"], m.get("orbit")) for m in read_manifest(out_dir)
            if m.get("site") == site_name and "date" in m}


def read_manifest(out_dir):
    with open(os.path.join(out_dir, "manifest.jsonl")) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def manifest_table(out_dir, res_m=10.0):
    """The manifest as a pandas DataFrame (one row per scene: site, date, orbit, JRC month, offset, radar contrast, water km2,
    hard negatives, JRC no-data, usable, reject_reason). Raises RuntimeError with the likely cause if there is nothing to show or
    nothing is usable."""
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
        "site": m.get("site"), "date": m["date"], "orbit": m.get("orbit"), "jrc_month": m.get("jrc_month"),
        "offset": m.get("month_offset"), "contrast_db": m.get("labels", {}).get("contrast_db"),
        "water_km2": (m["labels"]["water_px"] * res_m ** 2 / 1e6) if "labels" in m else None,
        "hard_neg_km2": (m["labels"]["hard_negative_px"] * res_m ** 2 / 1e6) if "labels" in m else None,
        "jrc_nodata": m.get("labels", {}).get("jrc_nodata_frac"), "usable": bool(m.get("usable")),
        "reject_reason": m.get("reject_reason")} for m in man])
    if not tbl["usable"].any():
        reasons = tbl["reject_reason"].value_counts().to_dict()
        raise RuntimeError(
            f"all {len(tbl)} scenes were rejected {reasons}. low_contrast: the JRC month does not match the pass "
            "(try max_month_offset=1) or the water was too rough for the radar to tell it from land; no_jrc_month: Landsat "
            "cloud gaps that month; jrc_nodata: too little JRC coverage; few_water_px: no water body in the scene. Lower "
            "min_contrast_db only after looking at the scenes.")
    return tbl


class PatchSet:
    """Training patches. The S1 channels (VV, VH, SDWI) are stored per patch (float16, they differ per scene); the terrain and
    land-cover channels are cut from the site's terrain raster when a batch is asked for, so they cost no memory per patch.
    `batch(idx)` -> (X (B, N_CHANNELS, s, s) float32, Y (B, s, s) uint8)."""

    def __init__(self, s1, Y, metas, terrains):
        self.s1, self.Y, self.metas, self.terrains = s1, Y, metas, terrains
        self.size = s1.shape[-1] if len(s1) else 0

    def __len__(self):
        return len(self.Y)

    def batch(self, idx):
        xs = []
        for i in idx:
            m = self.metas[i]
            r, c = m["corner"]
            t = self.terrains[m["site"]].crop(r, c, self.size)
            xs.append(np.concatenate([self.s1[i].astype(np.float32), features.terrain_channels(t, m["orbit"])]))
        return np.stack(xs), self.Y[list(idx)]


def load_patches(out_dir, size=224, stride=112, sites=None, min_labelled=0.5, min_water=0.005, min_hard=0.01, keep_empty=0.2,
                 max_month_offset=0, seed=0, max_per_scene=None, dtype=np.float16):
    """Training patches from the usable scenes, as a `PatchSet`.

    Patches with less than `min_water` water share (of labelled pixels) AND less than `min_hard` hard-negative share (JRC land
    that is dark in the radar: shadow, wet soil) are mostly plain land and would swamp the rare classes: only a random
    `keep_empty` fraction of them is kept. `max_month_offset` drops scenes whose JRC month was not the S1 month.

    Memory: a 2000 x 2000 px scene gives hundreds of overlapping 224 px patches. Only the 3 S1 channels are held (0.3 MB each as
    float16), so for a real run set `max_per_scene` (random subsample, e.g. 32)."""
    rng = np.random.default_rng(seed)
    Xs, Ys, metas, terrains = [], [], [], {}
    for m in read_manifest(out_dir):
        if not m.get("usable") or (sites and m["site"] not in sites) or abs(m.get("month_offset", 0)) > max_month_offset:
            continue
        z = np.load(os.path.join(out_dir, m["site"], f"{m['date']}_{m['orbit']}.npz"))
        if m["site"] not in terrains:
            terrains[m["site"]] = features.Terrain.load(os.path.join(out_dir, m["site"], "terrain.npz"))
        x = features.s1_channels(z["vv_db"], z["vh_db"], z["sdwi"])
        y, src = z["labels"], z["source"]
        keep = []
        for r, c in features.extract_patches(x, y, size, stride, min_labelled):
            yy = y[r:r + size, c:c + size]
            lab = yy != IGNORE
            wf = float((yy[lab] == 1).mean()) if lab.any() else 0.0
            hf = float((src[r:r + size, c:c + size] == lb.SRC_HARD_NEGATIVE).mean())
            if wf < min_water and hf < min_hard and rng.random() > keep_empty:
                continue
            keep.append(((r, c), wf, hf))
        if max_per_scene and len(keep) > max_per_scene:
            keep = [keep[j] for j in sorted(rng.choice(len(keep), max_per_scene, replace=False))]
        for (r, c), wf, hf in keep:
            Xs.append(x[:, r:r + size, c:c + size].astype(dtype))
            Ys.append(y[r:r + size, c:c + size])
            metas.append({"site": m["site"], "date": m["date"], "orbit": m["orbit"], "corner": (r, c), "shape": m["shape"],
                          "water_frac": wf, "hard_frac": hf})
    if not Xs:
        return PatchSet(np.zeros((0, 3, size, size), dtype), np.zeros((0, size, size), np.uint8), [], terrains)
    return PatchSet(np.stack(Xs), np.stack(Ys), metas, terrains)


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
