"""Map many water bodies in one call: resumable, and one site failing never stops the others.

For every site and every Sentinel-1 date in a window, ascending and descending passes alike: features (VV, VH, SDWI, terrain, land
cover) -> TerraMind -> water (`infer.infer_scene`), restricted to the site's outline when it has one (`aoi.aoi_mask`). Writes under
`<out_dir>/<site>/`:

    philsa_multimodal-surfacewater-weekly_<YYYYMMDD>_v0.1.0_r01_{surfacewater,flags}.tif, _surfacewater.gpkg and _preview.png   per scene (what
                    `MapConfig.write` lists; () for none); see naming.py
    philsa_multimodal-surfacewaterfreq-frequency_<YYYYMM>-<YYYYMM>_v0.1.0_r01_surfacewaterfreq.png   figure of frequency.tif (one per site, both directions combined; plots.py)
    frequency.tif   % of the valid observations on which each pixel was water (255 = never observed). With both pass directions
                    it is the pixel-wise MINIMUM of the two directions' frequencies: radar shadow sits on different slopes in
                    the two geometries, so a pixel that is 'water' in only one of them is shadow, not water
    frequency_ascending.tif, frequency_descending.tif   the per-direction frequencies, when both directions were observed
    n_obs.tif       number of valid observations per pixel (both directions)
    state.npz       the running composite and the scenes done, so an interrupted run resumes without re-reading any pass

and, for all sites together, `<out_dir>/areas.csv` (one row per site, date and orbit: water area in km2, the area the terrain rule
removed, the observed area) and `<out_dir>/failures.csv`.
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import acquire, aoi, infer, jrc, naming, plots, scene as sc, terrain as tr

AREA_COLUMNS = ["site", "label", "kind", "date", "orbit", "water_km2", "terrain_suppressed_km2", "valid_km2"]
STATES = acquire.ORBIT_STATES


@dataclass
class MapConfig:
    start: str = "2024-01-01"
    end: str = "2024-12-31"
    out_dir: str = "outputs/maps"
    write: tuple = ("water", "flags", "preview")   # per-scene outputs: any of water, flags, preview, gpkg, prob; () = none
    composite: bool = True                       # frequency*.tif / n_obs.tif per site
    resume: bool = True                          # skip scenes already done
    stop_on_error: bool = False                  # False: record the failure and go on with the next site
    tile: int = 224                              # TerraMind tile (multiple of 16) and overlap
    overlap: int = 56
    threshold: float = 0.5                       # probability at which a pixel is water
    max_slope_deg: float = infer.MAX_SLOPE_DEG   # terrain safeguard, None = off (see infer.infer_scene)
    use_landcover: bool = True
    version: str = naming.PRODUCT_VERSION        # product version and revision in the output file names
    revision: str = naming.REVISION
    device: str = None
    cache_dir: str = "cache"
    scene: sc.SceneConfig = field(default_factory=sc.SceneConfig)
    terrain_source: object = None                # .dem(grid) / .landcover(grid); None = Earth Engine (terrain.GeeTerrain)


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
    """areas.csv as a DataFrame, one row per (site, date, orbit) (a re-processed scene keeps its latest row); empty frame if none."""
    path = os.path.join(out_dir, "areas.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=AREA_COLUMNS)
    df = pd.read_csv(path)
    return df.drop_duplicates(["site", "date", "orbit"], keep="last").reset_index(drop=True)


def _append_row(path, columns, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def _save_state(path, water_count, valid_count, days):
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, water_count=water_count, valid_count=valid_count, days=np.array(sorted(days), dtype="U24"))
    os.replace(tmp, path)


def _load_state(path, shape):
    if not os.path.exists(path):
        return None
    z = np.load(path)
    if z["water_count"].shape != (len(STATES),) + tuple(shape):
        return None                                   # the grid changed (other box): start this site again
    return z["water_count"].copy(), z["valid_count"].copy(), {str(d) for d in z["days"]}


def composite_frequency(water_count, valid_count):
    """(combined, per-state list) water frequency in % (uint8, 255 = never observed) from counts shaped (states, h, w).
    Combined = the minimum over the pass directions that observed the pixel (see module docstring)."""
    per = []
    for k in range(water_count.shape[0]):
        v = valid_count[k]
        per.append(np.where(v > 0, 100.0 * water_count[k] / np.maximum(v, 1), np.nan))
    stack = np.stack(per)
    with np.errstate(all="ignore"):
        comb = np.nanmin(np.where(np.isnan(stack), np.inf, stack), axis=0)
    comb = np.where(np.isinf(comb), 255, np.round(comb)).astype(np.uint8)
    return comb, [np.where(np.isnan(p), 255, np.round(p)).astype(np.uint8) for p in per]


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
            p = prev[prev["site"] == site.name]
            done = {acquire.scene_key(d, o) for d, o in zip(p["date"], p["orbit"])}
    gb, scenes = infer.iter_window(site, cfg.start, cfg.end, cfg.scene, cache_dir=cfg.cache_dir, log=log, skip_days=done)
    inside = aoi.aoi_mask(site, gb)
    terrain = tr.site_terrain(jrc.grid_from_geobox(gb), cfg.terrain_source or tr.GeeTerrain(), cfg.cache_dir, cfg.use_landcover)

    state = _load_state(state_path, gb.shape) if (cfg.resume and cfg.composite) else None
    if state is None:
        wc = np.zeros((len(STATES),) + tuple(gb.shape), np.uint16)
        vc, days = np.zeros_like(wc), set()
        if done and cfg.composite:
            log(f"[{site.name}] state does not match this grid: composite restarts, earlier per-scene outputs are kept")
    else:
        wc, vc, days = state
    res_m, rows = cfg.scene.resolution_m, []
    for day, orbit, vv, vh in scenes:
        stamp = str(pd.Timestamp(str(day)).date())
        res = infer.infer_scene(vv, vh, orbit, terrain, model, cfg.scene, tile=cfg.tile, overlap=cfg.overlap, device=cfg.device,
                                threshold=cfg.threshold, max_slope_deg=cfg.max_slope_deg, use_landcover=cfg.use_landcover)
        a = res.area_km2(res_m, inside)
        valid = res.valid if inside is None else (res.valid & inside)
        if cfg.write:
            if any(k.startswith(stamp + ":") and k != acquire.scene_key(stamp, orbit) for k in days):
                log(f"  WARNING [{site.name}] {stamp}: another pass direction was mapped the same day; the file names carry no "
                    "direction, so this scene replaces its outputs")
            infer.write_result(sdir, stamp, res, gb, which=cfg.write, mask=inside, version=cfg.version, revision=cfg.revision)
        if cfg.composite:
            k = STATES.index(orbit)
            wc[k] += (res.water & valid).astype(np.uint16)
            vc[k] += valid.astype(np.uint16)
            days.add(acquire.scene_key(stamp, orbit))
            _save_state(state_path, wc, vc, days)
        row = {"site": site.name, "label": site.label, "kind": site.kind, "date": stamp, "orbit": orbit,
               "water_km2": a["water"], "terrain_suppressed_km2": a["terrain_suppressed"],
               "valid_km2": float(valid.sum() * res_m ** 2 / 1e6)}
        _append_row(os.path.join(cfg.out_dir, "areas.csv"), AREA_COLUMNS, row)
        rows.append(row)
        log(f"  [{site.name}] {stamp} {orbit}: water {a['water']:.2f} km2 (terrain rule removed {a['terrain_suppressed']:.2f} km2)")
    if cfg.composite and int(vc.max()) > 0:
        comb, per = composite_frequency(wc, vc)
        sc.write_tif(os.path.join(sdir, "frequency.tif"), comb, gb, "uint8", 255)
        sc.write_tif(os.path.join(sdir, "n_obs.tif"), vc.sum(0), gb, "uint16", 0)
        first, last = min(d.split(":")[0] for d in days), max(d.split(":")[0] for d in days)
        states = sorted({d.split(":")[1] for d in days})
        plots.frequency_figure(os.path.join(sdir, naming.frequency_name(first, last, cfg.version, cfg.revision)), comb, gb, site.label,
                               f"Sentinel-1 {' + '.join(states)} \u00b7 {first} to {last} \u00b7 {len(days)} passes \u00b7 {site.kind}",
                               f"ai_water v{cfg.version} {cfg.revision}")
        if (vc.reshape(len(STATES), -1).max(1) > 0).all():
            for state_name, p in zip(STATES, per):
                sc.write_tif(os.path.join(sdir, f"frequency_{state_name}.tif"), p, gb, "uint8", 255)
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
