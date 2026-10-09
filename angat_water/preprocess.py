"""Sentinel-1 helpers: linear <-> dB, border-noise masking."""
from __future__ import annotations

import numpy as np


def to_db(lin, floor_db=-30.0, ceil_db=10.0):
    """Linear power -> dB. Non-positive, non-finite, border-noise (<= floor_db) and implausibly bright (> ceil_db)
    pixels -> NaN."""
    lin = np.asarray(lin, np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.where(lin > 0, lin, np.nan))
    db[~np.isfinite(db) | (db <= floor_db) | (db > ceil_db)] = np.nan
    return db.astype(np.float32)


def from_db(db):
    return (10.0 ** (np.asarray(db, np.float32) / 10.0)).astype(np.float32)


def valid_mask(*linear_bands, min_db=-30.0):
    """True where every band is finite and above the border-noise floor."""
    ok = None
    for b in linear_bands:
        m = np.isfinite(b) & (b > 0) & (10.0 * np.log10(np.where(b > 0, b, 1.0)) > min_db)
        ok = m if ok is None else ok & m
    return ok
