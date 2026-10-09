"""Registry of water bodies to map. Each site gets its own bbox and its own orbit choice.

Bounding boxes are APPROXIMATE (dam + reservoir + some margin); check them on a map and edit. Add rivers, lakes or
look-alike surfaces (paddies, fishponds, steep shadow terrain) as extra sites: variety is what lets the model
generalise, and held-out sites are the honest test (`dataset.split_by_site`).
"""
from __future__ import annotations

from dataclasses import dataclass

from . import acquire


@dataclass(frozen=True)
class Site:
    name: str
    bbox: tuple              # (west, south, east, north) EPSG:4326
    crs: str = "EPSG:32651"  # metric CRS (UTM 51N covers Luzon east of 120 E)
    kind: str = "reservoir"  # reservoir | river | lake | wetland | paddy | other
    note: str = ""


SITES = {
    "angat": Site("angat", acquire.ANGAT_BBOX, kind="reservoir", note="Angat Dam, Bulacan (approx. box)"),
    "magat": Site("magat", (121.30, 16.72, 121.52, 16.95), kind="reservoir",
                  note="Magat Dam, Isabela/Ifugao (approx. box; check the reservoir tail to the north-east)"),
}


def get(name):
    try:
        return SITES[name.lower()]
    except KeyError:
        raise KeyError(f"unknown site {name!r}; known: {sorted(SITES)} (register more in sites.SITES or --site-bbox)")


def custom(name, bbox, kind="other", crs="EPSG:32651"):
    """Register an ad-hoc site (e.g. a river reach) for this process."""
    s = Site(name.lower(), tuple(float(v) for v in bbox), crs=crs, kind=kind)
    SITES[s.name] = s
    return s
