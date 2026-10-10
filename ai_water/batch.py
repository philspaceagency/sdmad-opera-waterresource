"""Map many water bodies in one call: resumable, and one site failing never stops the others.

For every site and every Sentinel-1 date in a window: Otsu + TerraMind + fusion (`infer.infer_scene`), restricted to the site's
outline when it has one (`aoi.aoi_mask`). Writes under `<out_dir>/<site>/`:

    <date>_fused_water.tif, <date>_flags.tif ...   per scene (what `MapConfig.write` lists; () for none)
    frequency.tif   % of the valid observations on which each pixel was water (255 = never observed)
    n_obs.tif       number of valid observations per pixel
    state.npz       the running composite and the dates done, so an interrupted run resumes without re-reading any pass

and, for all sites together, `<out_dir>/areas.csv` (one row per site and date: Otsu / fused water area in km2, the model-only /
Otsu-only candidate areas, the observed area, the Otsu mode) and `<out_dir>/failures.csv`.
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import aoi, infer, pipeline

AREA_COLUMNS = ["site", "label", "kind", "date", "otsu_km2", "fused_km2", "model_only_km2", "otsu_only_km2", "valid_km2", "otsu_mode"]


@dataclass
class MapConfig:
    start: str = "2024-01-01"
    end: str = "2024-12-31"
    out_dir: str = "outputs/maps"
    write: tuple = ("fused_water", "flags")      # per-scene GeoTIFFs: any of otsu_classes, fused_water, flags, prob; () = none
    composite: bool = True                       # frequency.tif / n_obs.tif per site
    resume: bool = True                          # skip dates already done
    stop_on_error: bool = False                  # False: record the failure and go on with the next site
    tile: int = 224                              # TerraMind tile (multiple of 16) and overlap
    overlap: int = 56
    device: str = None
    cache_dir: str = "cache"
    water: pipeline.WaterConfig = field(default_factory=pipeline.WaterConfig)


@dataclass
class MapResult:
    areas: pd.DataFrame
    failures: list
    out_dir: str

    @property
    def ok(self):
        return not self.failures


# ---------------------------------------------------------------------------------------- bookkeeping
def read_areas(out_dir):
    """areas.csv as a DataFrame, one row per (site, date) (a re-processed date keeps its latest row); empty frame if none."""
    path = os.path.join(out_dir, "areas.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=AREA_COLUMNS)
    df = pd.read_csv(path)
    return df.drop_duplicates(["site", "date"], keep="last").reset_index(drop=True)


def _append_row(path, columns, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def _save_state(path, water_count, valid_count, days):
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, water_count=water_count, valid_count=valid_count, days=np.array(sorted(days), dtype="U10"))
    os.replace(tmp, path)


def _load_state(path, shape):
    if not os.path.exists(path):
        return None
    z = np.load(path)
    if z["water_count"].shape != tuple(shape):
        return None                                   # the grid changed (other box): start this site again
    return z["water_count"].copy(), z["valid_count"].copy(), {str(d) for d in z["days"]}


# ---------------------------------------------------------------------------------------- one site
def map_site(site, model, cfg: MapConfig, log=print):
    """Process one site; returns its area rows (new ones only). Raises on failure (see `map_sites` for the isolated version)."""
    sdir = os.path.join(cfg.out_dir, site.name)
    os.makedirs(sdir, exist_ok=True)
    state_path = os.path.join(sdir, "state.npz")

    done = set()
    if cfg.resume:
        if cfg.composite and os.path.exists(state_path):
            done = {str(d) for d in np.load(state_path)["days"]}
        elif not cfg.composite:
            prev = read_areas(cfg.out_dir)
            done = set(prev.loc[prev["site"] == site.name, "date"].astype(str))
    gb, scenes = infer.iter_window(site, cfg.start, cfg.end, cfg.water, cache_dir=cfg.cache_dir, log=log, skip_days=done)
    inside = aoi.aoi_mask(site, gb)

    state = _load_state(state_path, gb.shape) if (cfg.resume and cfg.composite) else None
    if state is None:
        wc, vc, days = np.zeros(gb.shape, np.uint16), np.zeros(gb.shape, np.uint16), set()
        if done and cfg.composite:
            log(f"[{site.name}] state does not match this grid: composite restarts, earlier per-scene outputs are kept")
    else:
        wc, vc, days = state
    res_m, rows = cfg.water.resolution_m, []
    for day, vv, vh in scenes:
        stamp = str(pd.Timestamp(str(day)).date())
        res = infer.infer_scene(vv, vh, model, cfg.water, tile=cfg.tile, overlap=cfg.overlap, device=cfg.device)
        a = res.area_km2(res_m, inside)
        valid = np.isfinite(res.vv_db) if inside is None else (np.isfinite(res.vv_db) & inside)
        if cfg.write:
            infer.write_result(sdir, stamp, res, gb, which=cfg.write, mask=inside)
        if cfg.composite:
            wc += (res.fused_water & valid).astype(np.uint16)
            vc += valid.astype(np.uint16)
            days.add(stamp)
            _save_state(state_path, wc, vc, days)
        row = {"site": site.name, "label": site.label, "kind": site.kind, "date": stamp, "otsu_km2": a["otsu"],
               "fused_km2": a["fused"], "model_only_km2": a["model_only_candidates"], "otsu_only_km2": a["otsu_only_candidates"],
               "valid_km2": float(valid.sum() * res_m ** 2 / 1e6), "otsu_mode": res.thresholds.mode}
        _append_row(os.path.join(cfg.out_dir, "areas.csv"), AREA_COLUMNS, row)
        rows.append(row)
        log(f"  [{site.name}] {stamp}: Otsu {a['otsu']:.2f} km2 | fused {a['fused']:.2f} km2 | mode {res.thresholds.mode}")
    if cfg.composite and int(vc.max()) > 0:
        freq = np.where(vc > 0, np.round(100.0 * wc / np.maximum(vc, 1)), 255).astype(np.uint8)
        pipeline._write_tif(os.path.join(sdir, "frequency.tif"), freq, gb, "uint8", 255)
        pipeline._write_tif(os.path.join(sdir, "n_obs.tif"), vc, gb, "uint16", 0)
    return rows


# ---------------------------------------------------------------------------------------- many sites
def map_sites(sites, model, cfg: MapConfig, log=print):
    """Process every site in `sites`. A site that fails (no pass covers it, a read error ...) is recorded in
    `<out_dir>/failures.csv` and the run goes on, unless `cfg.stop_on_error`. Returns a MapResult (areas of ALL sites incl.
    earlier runs, and this run's failures)."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    failures, t0 = [], time.time()
    for i, site in enumerate(sites, start=1):
        log(f"[{i}/{len(sites)}] {site.label} ({site.kind})")
        try:
            map_site(site, model, cfg, log)
        except Exception as e:                                      # noqa: BLE001 - isolate the site, report it
            if cfg.stop_on_error:
                raise
            msg = f"{e.__class__.__name__}: {e}"
            log(f"  FAILED [{site.name}]: {msg}")
            failures.append({"site": site.name, "error": msg, "when": pd.Timestamp.now().isoformat(timespec="seconds")})
            _append_row(os.path.join(cfg.out_dir, "failures.csv"), ["site", "error", "when"], failures[-1])
    log(f"done {len(sites)} site(s) in {(time.time() - t0) / 60:.1f} min, {len(failures)} failed")
    return MapResult(read_areas(cfg.out_dir), failures, cfg.out_dir)
