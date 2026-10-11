"""Sentinel-1 acquisition from Microsoft Planetary Computer (`sentinel-1-rtc`).

Follows the pattern of Leads-DigiSaka-System/ai-irrigation (irrigation_pipeline/stac_data.py, sar.py):
  * STAC search with retry / rate-limit back-off and an on-disk cache of the search result;
  * asset URLs signed at read time (so a cached search stays valid), token requests serialised;
  * odc-stac load onto an explicit geobox, frames of one pass fused ('solar_day');
  * ONE relative orbit per pass direction (the best ascending and the best descending one), so the viewing geometry is identical
    from date to date within a direction, while radar shadow, which moves with the look direction, is not the same in both;
  * UTC -> local solar day for grouping.
Differences: terrain-corrected gamma0 is read at the NATIVE 10 m (not averaged to 100 m): the refined-Lee window
and the shoreline both need the full resolution; speckle is handled by `speckle.py` instead of multilooking.
RTC values are linear power.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time as _time

import numpy as np
import pandas as pd

MPC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S1_COLLECTION = "sentinel-1-rtc"

# Angat Dam (14.91 N, 121.15 E) and its reservoir, Norzagaray / Dona Remedios Trinidad, Bulacan: the default area.
# Corners supplied by the project (lat, lon): (14.8721, 121.1430) and (15.0532, 121.2623).
ANGAT_BBOX = (121.14302981543655, 14.872088697247067, 121.26233447214564, 15.053187085370627)   # (west, south, east, north)
ANGAT_CRS = "EPSG:32651"                         # UTM 51N

_CATALOG = None
_CACHE_DIR = None
_RATE_WAIT = (20, 40, 80, 160)
_SIGN_LOCK = threading.Lock()


def set_stac_cache(directory):
    """Keep STAC search results on disk: a rerun costs no request. Windows that reach the last 10 days are reused for
    24 h only (new scenes keep appearing)."""
    global _CACHE_DIR
    _CACHE_DIR = directory
    if directory:
        os.makedirs(os.path.join(directory, "stac"), exist_ok=True)


def _open_catalog(retries=4, fresh=False):
    """Planetary Computer STAC client with retry/backoff. PC_SDK_SUBSCRIPTION_KEY (optional) raises rate limits."""
    global _CATALOG
    if _CATALOG is not None and not fresh:
        return _CATALOG
    import pystac_client
    from pystac_client.stac_api_io import StacApiIO
    from urllib3 import Retry
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=None)
    key = os.environ.get("PC_SDK_SUBSCRIPTION_KEY")
    headers = {"Ocp-Apim-Subscription-Key": key} if key else None
    last = None
    for i in range(retries):
        try:
            _CATALOG = pystac_client.Client.open(MPC_STAC_URL, stac_io=StacApiIO(max_retries=retry, headers=headers))
            return _CATALOG
        except Exception as e:
            last = e
            if i < retries - 1:
                _time.sleep(2 ** i)
    raise last


def _is_rate_limit(e):
    text = f"{e.__class__.__name__} {e}".lower()
    return "rate limit" in text or "429" in text or "too many requests" in text


class SigningError(RuntimeError):
    """The Planetary Computer would not issue a read token (HTTP 403 / 429 / 5xx)."""


def _sign_url(href, tries=5):
    import planetary_computer as pc
    last = None
    for k in range(tries):
        try:
            with _SIGN_LOCK:
                return pc.sign(href)
        except Exception as e:
            last = e
            if k < tries - 1:
                _time.sleep(min(30, 2 ** (k + 1)))
    raise SigningError(
        f"Planetary Computer refused a read token after {tries} tries: {last!r}. A 403/429 usually means anonymous "
        "access is throttled from this network: set PC_SDK_SUBSCRIPTION_KEY "
        "(https://planetarycomputer.developer.azure-api.net) or retry later.") from last


def _cache_path(query):
    if not _CACHE_DIR:
        return None
    key = hashlib.md5(json.dumps(query, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return os.path.join(_CACHE_DIR, "stac", f"{query['collections'][0]}_{key}.json")


def _cache_fresh(path, query):
    if not os.path.exists(path):
        return False
    end = str(query.get("datetime", "")).split("/")[-1]
    try:
        recent = pd.Timestamp(end) >= pd.Timestamp.now().normalize() - pd.Timedelta(days=10)
    except ValueError:
        recent = True
    return not recent or (_time.time() - os.path.getmtime(path)) < 86400


def _search(retries=5, **query):
    """STAC search -> list of unsigned items, from the disk cache when possible."""
    import pystac
    path = _cache_path(query)
    if path and _cache_fresh(path, query):
        with open(path) as fh:
            return [pystac.Item.from_dict(d) for d in json.load(fh)]
    items = []
    for i in range(retries):
        try:
            items = list(_open_catalog(fresh=i > 0).search(limit=1000, **query).items())
            if items:
                break
        except Exception as e:
            if i == retries - 1:
                raise
            _time.sleep(_RATE_WAIT[min(i, len(_RATE_WAIT) - 1)] if _is_rate_limit(e) else 2 ** i)
            continue
        if i < retries - 1:
            _time.sleep(2 ** i)
    if items and path:
        tmp = path + ".part"
        with open(tmp, "w") as fh:
            json.dump([it.to_dict() for it in items], fh)
        os.replace(tmp, path)
    return items


def search_s1(bbox, start, end):
    """Sentinel-1 RTC IW items with VV and VH over bbox in [start, end] (unsigned)."""
    items = _search(collections=[S1_COLLECTION], bbox=list(bbox), datetime=f"{start}/{end}")
    return [i for i in items
            if i.properties.get("sar:instrument_mode", "IW") == "IW" and "vv" in i.assets and "vh" in i.assets]


# ------------------------------------------------------------------ orbit / date selection (pure, testable offline)
def orbit_key(item):
    p = item.properties
    return (p.get("sat:relative_orbit"), p.get("sat:orbit_state"))


def _footprint(item):
    from shapely.geometry import box, shape
    return shape(item.geometry) if item.geometry else box(*item.bbox)


def solar_dates(times, lon):
    """UTC timestamps -> local solar day (datetime64[D]); the Philippines is ~ +8 h, so a 22:00 UTC pass is next day."""
    t = pd.DatetimeIndex(np.asarray(times)) + pd.Timedelta(hours=float(lon) / 15.0)
    return t.normalize().values.astype("datetime64[D]")


def item_day(item):
    b = item.bbox
    return solar_dates([np.datetime64(item.datetime.replace(tzinfo=None))], 0.5 * (b[0] + b[2]))[0]


def group_by_day(items):
    out = {}
    for it in items:
        out.setdefault(item_day(it), []).append(it)
    return out


ORBIT_STATES = ("ascending", "descending")


def choose_orbits(items, bbox, min_cover=0.9, states=ORBIT_STATES):
    """For each pass direction in `states`, the relative orbit that covers `bbox` on the most days, counting a day only when
    that day's frames cover at least `min_cover` of the box (the whole water body must be in one image). Returns
    {state: (relative orbit, [items of that orbit])} with only the directions that have at least one such day. Ascending and
    descending passes see radar shadow on opposite slopes, so both are kept when the archive has them."""
    from shapely.geometry import box
    from shapely.ops import unary_union
    area = box(*bbox)
    cand = {}
    for it in items:
        if _footprint(it).intersects(area) and orbit_key(it)[1] in states:
            cand.setdefault(orbit_key(it), []).append(it)
    out = {}
    for state in states:
        best, best_score = None, 0
        for key, its in sorted(((k, v) for k, v in cand.items() if k[1] == state), key=lambda kv: str(kv[0])):
            score = sum(1 for g in group_by_day(its).values()
                        if unary_union([_footprint(i) for i in g]).intersection(area).area / area.area >= min_cover)
            if score > best_score:
                best, best_score = key, score
        if best is not None:
            out[state] = (best[0], sorted(cand[best], key=lambda i: i.datetime))
    return out


# ------------------------------------------------------------------ reading
def make_geobox(bbox, crs=ANGAT_CRS, resolution_m=10.0):
    """Geobox on a `resolution_m` lattice covering the lon/lat bbox in a metric CRS."""
    from affine import Affine
    from odc.geo.geobox import GeoBox
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tr.transform([bbox[0], bbox[2], bbox[0], bbox[2]], [bbox[1], bbox[1], bbox[3], bbox[3]])
    x0 = np.floor(min(xs) / resolution_m) * resolution_m
    x1 = np.ceil(max(xs) / resolution_m) * resolution_m
    y0 = np.floor(min(ys) / resolution_m) * resolution_m
    y1 = np.ceil(max(ys) / resolution_m) * resolution_m
    w, h = int((x1 - x0) / resolution_m), int((y1 - y0) / resolution_m)
    return GeoBox((h, w), Affine(resolution_m, 0, x0, 0, -resolution_m, y1), crs)


def load_s1(items, geobox, resampling="nearest", pool=4, log=print):
    """Items -> (times (T,) datetime64[ns] UTC, VV (T,h,w), VH (T,h,w)) linear power float32 on `geobox`; frames of the
    same pass fused (solar_day). None if nothing could be read. `resampling`: 'nearest' for a 10 m grid, 'average' to
    multilook onto a coarser one."""
    import odc.stac
    from rasterio.errors import RasterioError
    if not items:
        return None
    last = None
    for attempt in range(4):
        try:
            ds = odc.stac.load(items, bands=["vv", "vh"], geobox=geobox, groupby="solar_day", resampling=resampling,
                               chunks=None, pool=pool, patch_url=_sign_url)
            break
        except (RasterioError, AssertionError) as e:       # some COGs open without a CRS -> retry
            last = e
            if log:
                log(f"    read retry {attempt + 1}/3 ({e.__class__.__name__})")
            _time.sleep(2 ** attempt)
    else:
        raise last
    if ds.sizes.get("time", 0) == 0:
        return None
    return (np.asarray(ds["time"].values, "datetime64[ns]"),
            np.asarray(ds["vv"].values, np.float32), np.asarray(ds["vh"].values, np.float32))


def load_local_stack(vv_paths, vh_paths=None):
    """Offline alternative: read co-registered VV (and VH) GeoTIFFs (linear power) -> (vv (T,h,w), vh or None, profile)."""
    import rasterio
    vv, vh, profile = [], [], None
    for i, p in enumerate(vv_paths):
        with rasterio.open(p) as r:
            vv.append(r.read(1).astype(np.float32))
            profile = profile or r.profile
        if vh_paths:
            with rasterio.open(vh_paths[i]) as r:
                vh.append(r.read(1).astype(np.float32))
    return np.stack(vv), (np.stack(vh) if vh else None), profile


def iter_scenes(bbox, crs, resolution_m, start, end, min_cover=0.9, cache_dir=None, log=print, skip_days=None,
                orbits=ORBIT_STATES):
    """(geobox, generator of (local solar day, orbit state, vv_lin (h, w), vh_lin (h, w))) for every Sentinel-1 RTC date of the
    best relative orbit of each pass direction in `orbits` that covers the bbox in [start, end], in date order. Scenes are read
    lazily, one acquisition day at a time. `skip_days` (a set of 'YYYY-MM-DD:state' keys, see `scene_key`) are not read at all:
    this is how an interrupted run resumes."""
    set_stac_cache(cache_dir)
    items = search_s1(bbox, str(start)[:10], str(end)[:10])
    chosen = choose_orbits(items, bbox, min_cover, orbits)
    if not chosen:
        raise RuntimeError(f"no Sentinel-1 RTC pass covers {tuple(bbox)} in {str(start)[:10]}..{str(end)[:10]}")
    plan = []
    for state, (rel, its) in chosen.items():
        days = group_by_day(its)
        log(f"{state} orbit {rel}: {len(its)} items on {len(days)} days")
        plan += [(day, state, g) for day, g in days.items()]
    plan.sort(key=lambda p: (p[0], p[1]))
    gb = make_geobox(bbox, crs, resolution_m)
    lon = 0.5 * (bbox[0] + bbox[2])

    def gen():
        n_scenes, n_skipped, n_done = 0, 0, 0
        for day, state, its in plan:
            if skip_days and scene_key(day, state) in skip_days:
                n_done += 1
                continue
            loaded = load_s1(its, gb, "nearest" if resolution_m <= 10 else "average", log=log)
            if loaded is None:
                n_skipped += 1
                log(f"  {day} {state}: no pixels could be read from {len(its)} item(s), skipped")
                continue
            times, vv, vh = loaded
            for k in range(len(times)):
                n_scenes += 1
                yield solar_dates([times[k]], lon)[0], state, vv[k], vh[k]
        log(f"read {n_scenes} scene(s) from {len(plan)} acquisition day(s)" + (f", {n_skipped} unreadable" if n_skipped else "")
            + (f", {n_done} already done (resumed)" if n_done else ""))
    return gb, gen()


def scene_key(day, state):
    """Identifier of one acquisition in the manifests and resume sets: 'YYYY-MM-DD:ascending'."""
    return f"{str(pd.Timestamp(str(day)).date())}:{state}"
