"""Water bodies to map: the default area, a CSV / vector-file loader, and the geometry helpers around them.

A `Site` is one area of interest: a lon/lat box (what Sentinel-1 is searched and read over), its UTM zone (chosen from the
box centre, so a nationwide run that crosses zones works), and optionally the outline of the water body (to clip results to).

Where sites come from
  * the built-in default: Angat Dam (`default_site()`);
  * a CSV (`sites_from_csv`): the training sites live in `data/training_sites.csv`;
  * a vector file (`sites_from_vector`): GeoJSON natively, GeoPackage / Shapefile / KML through geopandas. One site per
    feature: polygons (a dam / reservoir outline), lines (a river reach) or points (with a radius).

CSV columns (case-insensitive; lines starting with # are ignored). `name` is required, plus ONE way of giving the place:
  corners             "((lat1, lon1), (lat2, lon2))"  two opposite corners, as copied from a map
  south,west,north,east   decimal degrees
  lat,lon[,radius_km]     a point and a radius (default by kind)
  kind (optional)     reservoir | lake | river | other

Boxes smaller than `min_size_km` (default 5 km) on a side are expanded around their centre: Sentinel-1 patches are 2.2 km wide
and need context, so a 1.4 km lake would otherwise give no training patches. The expansion is recorded in `Site.note`.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from . import acquire

MIN_SIZE_KM = 5.0
DEFAULT_RADIUS_KM = {"reservoir": 8.0, "dam": 8.0, "lake": 6.0, "river": 5.0, "other": 5.0}
KM_PER_DEG_LAT = 110.574
PHILIPPINES = (116.0, 4.0, 127.5, 22.0)          # west, south, east, north: a sanity box, not a boundary
TRAINING_SITES_CSV = Path(__file__).resolve().parent.parent / "data" / "training_sites.csv"


def slug(name):
    """'Angat Dam' -> 'angat_dam'."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_") or "site"


def utm_crs(lon, lat):
    """EPSG code of the UTM zone containing (lon, lat), e.g. 'EPSG:32651' for Luzon, 'EPSG:32650' for west Palawan."""
    zone = int((lon + 180.0) // 6.0) % 60 + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


@dataclass(frozen=True)
class Site:
    name: str                       # id, lower-case: 'angat_dam'
    bbox: tuple                     # (west, south, east, north) EPSG:4326
    crs: str = None                 # metric CRS; None = the UTM zone of the box centre
    kind: str = "reservoir"         # reservoir | lake | river | tile | other
    note: str = ""
    label: str = ""                 # display name: 'Angat Dam'
    geometry: object = field(default=None, compare=False, hash=False, repr=False)   # shapely outline (EPSG:4326) to clip to

    def __post_init__(self):
        w, s, e, n = self.bbox
        if not (w < e and s < n):
            raise ValueError(f"site {self.name!r}: bbox must be (west, south, east, north) with west < east and south < north, got {self.bbox}")
        if self.crs is None:
            object.__setattr__(self, "crs", utm_crs(0.5 * (w + e), 0.5 * (s + n)))
        if not self.label:
            object.__setattr__(self, "label", self.name)

    @property
    def center(self):
        return 0.5 * (self.bbox[0] + self.bbox[2]), 0.5 * (self.bbox[1] + self.bbox[3])

    def size_km(self):
        """(width, height) of the box in km."""
        w, s, e, n = self.bbox
        return (e - w) * 111.320 * math.cos(math.radians(0.5 * (s + n))), (n - s) * KM_PER_DEG_LAT


# ---------------------------------------------------------------------------------------- geometry helpers
def _check_lonlat(lat, lon, what):
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"{what}: latitude {lat} / longitude {lon} out of range (are lat and lon swapped?)")


def parse_corners(text):
    """'((14.87, 121.14), (15.05, 121.26))' -> bbox (west, south, east, north). Two opposite corners as (lat, lon), any order."""
    nums = [float(v) for v in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(text))]
    if len(nums) != 4:
        raise ValueError(f"corners must hold two (lat, lon) pairs, got {text!r}")
    (lat1, lon1), (lat2, lon2) = (nums[0], nums[1]), (nums[2], nums[3])
    _check_lonlat(lat1, lon1, "corner 1")
    _check_lonlat(lat2, lon2, "corner 2")
    return (min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))


def expand_bbox(bbox, min_size_km=MIN_SIZE_KM):
    """Grow `bbox` about its centre so both sides are at least `min_size_km`. Returns (bbox, (orig_w_km, orig_h_km), expanded)."""
    w, s, e, n = bbox
    lat = 0.5 * (s + n)
    kx = 111.320 * math.cos(math.radians(lat))
    wk, hk = (e - w) * kx, (n - s) * KM_PER_DEG_LAT
    cx, cy = 0.5 * (w + e), 0.5 * (s + n)
    nw, nh = max(wk, min_size_km), max(hk, min_size_km)
    if nw == wk and nh == hk:
        return tuple(bbox), (wk, hk), False
    return (cx - 0.5 * nw / kx, cy - 0.5 * nh / KM_PER_DEG_LAT, cx + 0.5 * nw / kx, cy + 0.5 * nh / KM_PER_DEG_LAT), (wk, hk), True


def _warn_if_outside_philippines(name, bbox):
    w, s, e, n = bbox
    if not (PHILIPPINES[0] <= w and e <= PHILIPPINES[2] and PHILIPPINES[1] <= s and n <= PHILIPPINES[3]):
        warnings.warn(f"site {name!r} lies outside the Philippines {PHILIPPINES}: check latitude / longitude order", stacklevel=3)


def box_site(label, bbox, kind="reservoir", min_size_km=MIN_SIZE_KM, geometry=None, crs=None):
    """Site from a lon/lat box (expanded to `min_size_km`); `label` is the display name, its slug the id."""
    nb, (wk, hk), grown = expand_bbox(bbox, min_size_km) if min_size_km else (tuple(bbox), (0, 0), False)
    note = f"box expanded from {wk:.1f} x {hk:.1f} km to at least {min_size_km:g} km" if grown else ""
    _warn_if_outside_philippines(label, nb)
    return Site(slug(label), nb, crs, kind, note, str(label), geometry)


def point_site(label, lat, lon, radius_km=None, kind="reservoir", crs=None):
    """Site from a point and a radius (default by kind)."""
    _check_lonlat(lat, lon, f"site {label!r}")
    r = float(radius_km if radius_km else DEFAULT_RADIUS_KM.get(kind, 5.0))
    dlat, dlon = r / KM_PER_DEG_LAT, r / (111.320 * math.cos(math.radians(lat)))
    return box_site(label, (lon - dlon, lat - dlat, lon + dlon, lat + dlat), kind, min_size_km=0, crs=crs)


def project(geom, transformer):
    """Apply a pyproj Transformer to every vertex of a shapely geometry."""
    import numpy as np
    import shapely
    if hasattr(shapely, "transform"):                                     # shapely >= 2
        return shapely.transform(geom, lambda xy: np.column_stack(transformer.transform(xy[:, 0], xy[:, 1])))
    from shapely.ops import transform
    return transform(transformer.transform, geom)


def _buffer_km(geom, km):
    """Buffer a lon/lat geometry by `km`, done in its UTM zone so the distance is metres, not degrees."""
    if not km:
        return geom
    from pyproj import Transformer
    c = geom.centroid
    crs = utm_crs(c.x, c.y)
    fwd, back = Transformer.from_crs("EPSG:4326", crs, always_xy=True), Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return project(project(geom, fwd).buffer(km * 1000.0), back)


# ---------------------------------------------------------------------------------------- CSV
_ALIASES = {"latitude": "lat", "longitude": "lon", "lng": "lon", "long": "lon", "label": "name", "type": "kind",
            "min_lat": "south", "min_lon": "west", "max_lat": "north", "max_lon": "east", "radius": "radius_km"}


def sites_from_csv(path, min_size_km=MIN_SIZE_KM):
    """Sites from a CSV (format in the module docstring). Raises ValueError naming the row and the problem."""
    text = Path(path).read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return []
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    out, seen = [], set()
    for i, raw in enumerate(rows, start=2):
        r = {_ALIASES.get(k.strip().lower(), k.strip().lower()): (v or "").strip() for k, v in raw.items() if k}
        where = f"{path} row {i}"
        name = r.get("name")
        if not name:
            raise ValueError(f"{where}: `name` is empty")
        kind = (r.get("kind") or "reservoir").lower()
        try:
            if r.get("corners"):
                site = box_site(name, parse_corners(r["corners"]), kind, min_size_km)
            elif all(r.get(k) for k in ("south", "west", "north", "east")):
                s_, w_, n_, e_ = (float(r[k]) for k in ("south", "west", "north", "east"))
                _check_lonlat(s_, w_, name)
                _check_lonlat(n_, e_, name)
                site = box_site(name, (w_, s_, e_, n_), kind, min_size_km)
            elif r.get("lat") and r.get("lon"):
                site = point_site(name, float(r["lat"]), float(r["lon"]), float(r["radius_km"]) if r.get("radius_km") else None, kind)
            else:
                raise ValueError("give `corners`, or south/west/north/east, or lat/lon")
        except ValueError as e:
            raise ValueError(f"{where} ({name}): {e}") from None
        if site.name in seen:
            raise ValueError(f"{where}: duplicate site name {name!r}")
        seen.add(site.name)
        out.append(site)
    return out


# ---------------------------------------------------------------------------------------- vector files
_NAME_KEYS = ("name", "Name", "NAME", "dam", "Dam", "DAM", "dam_name", "Dam_Name", "label", "Label", "id", "ID", "fid")


def _features(path):
    """[(properties dict, shapely geometry in EPSG:4326)] from a vector file."""
    from shapely.geometry import shape
    ext = Path(path).suffix.lower()
    if ext in (".geojson", ".json"):
        with open(path, encoding="utf-8") as fh:
            gj = json.load(fh)
        feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj] if gj.get("type") == "Feature" else \
            [{"properties": {}, "geometry": gj}]
        crs_name = str(((gj.get("crs") or {}).get("properties") or {}).get("name", "")).upper()
        if crs_name and not any(t in crs_name for t in ("4326", "CRS84")):
            raise ValueError(f"{path}: GeoJSON is in {crs_name}; re-save it as EPSG:4326 (lon/lat) or use a GeoPackage / Shapefile")
        return [(f.get("properties") or {}, shape(f["geometry"])) for f in feats if f.get("geometry")]
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(f"reading {ext} files needs geopandas (pip install geopandas); GeoJSON works without it") from None
    gdf = gpd.read_file(path)
    if gdf.crs is not None:
        gdf = gdf.to_crs("EPSG:4326")
    return [({k: v for k, v in row.items() if k != "geometry"}, row.geometry) for _, row in gdf.iterrows() if row.geometry is not None]


def sites_from_vector(path, name_col=None, buffer_km=1.0, kind=None, radius_km=None, min_size_km=0.0, prefix="aoi"):
    """One Site per feature of a vector file.

    Polygons / lines: the box is the feature's bounds + `buffer_km`, and the buffered outline is kept to clip results to.
    Points: a box of `radius_km` (default by `kind`).  `name_col` picks the property used as the name (else the first of
    name / dam / label / id, else `<prefix>_<n>`). The file must be lon/lat (GeoJSON) or carry its CRS (GPKG, SHP, KML)."""
    sites = []
    for i, (props, geom) in enumerate(_features(path), start=1):
        raw = props.get(name_col) if name_col else next((props[k] for k in _NAME_KEYS if props.get(k) not in (None, "")), None)
        label = str(raw) if raw not in (None, "") else f"{prefix}_{i}"
        k = (kind or props.get("kind") or props.get("type") or ("river" if geom.geom_type.endswith("LineString") else "reservoir"))
        if geom.geom_type in ("Point", "MultiPoint"):
            c = geom.centroid
            sites.append(point_site(label, c.y, c.x, radius_km, str(k).lower()))
            continue
        outline = _buffer_km(geom, buffer_km)
        w, s, e, n = outline.bounds
        sites.append(box_site(label, (w, s, e, n), str(k).lower(), min_size_km, geometry=outline))
    names = [s.name for s in sites]
    if len(set(names)) != len(names):                                   # disambiguate repeated names
        counts = {}
        fixed = []
        for s in sites:
            counts[s.name] = counts.get(s.name, 0) + 1
            fixed.append(s if names.count(s.name) == 1 else Site(f"{s.name}_{counts[s.name]}", s.bbox, s.crs, s.kind, s.note, s.label, s.geometry))
        sites = fixed
    return sites


def load_sites(path, **kw):
    """CSV -> `sites_from_csv`; anything else -> `sites_from_vector`."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    return sites_from_csv(path, **{k: v for k, v in kw.items() if k == "min_size_km"}) if str(path).lower().endswith(".csv") \
        else sites_from_vector(path, **kw)


# ---------------------------------------------------------------------------------------- registry
ANGAT_DAM = box_site("Angat Dam", acquire.ANGAT_BBOX, kind="reservoir", min_size_km=0)
SITES = {ANGAT_DAM.name: ANGAT_DAM}
DEFAULT_SITE = ANGAT_DAM.name
_ALIASES_IDS = {"angat": "angat_dam"}


def default_site():
    """The default area of interest: Angat Dam."""
    return SITES[DEFAULT_SITE]


def register(sites):
    """Add sites to the registry (so `get(name)` finds them). Re-registering the same box is fine; a different box under an
    existing name is an error (it would silently change an earlier result)."""
    for s in ([sites] if isinstance(sites, Site) else sites):
        old = SITES.get(s.name)
        if old is not None and old.bbox != s.bbox:
            raise ValueError(f"site {s.name!r} is already registered with a different box")
        SITES[s.name] = s
    return sites


def get(name):
    key = _ALIASES_IDS.get(slug(name), slug(name))
    try:
        return SITES[key]
    except KeyError:
        raise KeyError(f"unknown site {name!r}; known: {sorted(SITES)} (load more with sites.load_sites(path) / sites.register)") from None


def custom(name, bbox, kind="other", crs=None, min_size_km=0):
    """Register an ad-hoc site from a (west, south, east, north) box."""
    s = box_site(name, tuple(float(v) for v in bbox), kind, min_size_km, crs=crs)
    SITES[s.name] = s
    return s


def load_training_sites(path=None, register_them=True):
    """The sites to train on, from `data/training_sites.csv` (or `path`). Raises with the expected format when there are none."""
    path = Path(path) if path else TRAINING_SITES_CSV
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: create it (see ai_water/sites.py for the columns) with the dams / lakes / rivers to train on")
    sites = sites_from_csv(path)
    if not sites:
        raise ValueError(f"{path} has no sites yet: add one row per water body (name, kind, corners)")
    if register_them:
        register(sites)
    return sites


def resolve_aoi(path=None, names=None, multiple=False, **kw):
    """The sites to map. Nothing given -> Angat Dam, the default area.

    path     : a vector file (the dam / reservoir outline, or many of them) or a sites CSV; extra keywords go to the loader
               (`name_col`, `buffer_km`, `kind`, `radius_km` for vector files)
    names    : keep only the features with these names (matched like `slug`)
    multiple : a file with several features is processed whole only when asked: otherwise it must hold exactly one body (or `names`
               must pick one), so a wrong file never silently maps the first feature only."""
    if not path:
        return [default_site()]
    found = load_sites(path, **kw)
    if not found:
        raise ValueError(f"{path} holds no sites")
    if names:
        want = {slug(n) for n in ([names] if isinstance(names, str) else names)}
        pick = [s for s in found if s.name in want]
        if len(pick) != len(want):
            raise KeyError(f"not in {path}: {sorted(want - {s.name for s in pick})}; it has {[s.name for s in found]}")
        return pick
    if len(found) > 1 and not multiple:
        raise ValueError(f"{path} has {len(found)} features ({[s.name for s in found][:6]}...): set multiple=True to map all of them, "
                         "or names=[...] to pick")
    return found
