"""Turn Otsu water maps into training patches for the foundation model (no torch needed).

Label values: 0 land, 1 water, IGNORE (255) = uncertain band / nodata. Uncertain Otsu pixels are NOT trained on:
the model is free to decide them from context, which is where it can improve on the thresholds.

Scenes where the two Otsu variants did not agree (mode != 'both', or a large gap) are dropped: a pseudo-label set is
only as good as the weakest scene in it.
"""
from __future__ import annotations

import numpy as np

from . import otsu

IGNORE = 255

# TerraMind per-modality input normalisation of (VV, VH) in dB, as in terratorch 1.2.x
# (terratorch/models/backbones/terramind/model/terramind_register.py). Planetary Computer's `sentinel-1-rtc` is
# terrain-corrected gamma0, i.e. the S1RTC modality: S1GRD statistics would shift the input by ~2 dB.
S1_STATS = {
    "S1RTC": (np.array([-10.93, -17.329], np.float32), np.array([4.391, 4.459], np.float32)),
    "S1GRD": (np.array([-12.599, -20.293], np.float32), np.array([5.195, 5.890], np.float32)),
}


def labels_from_classes(classes):
    """Otsu class raster -> training labels (0 land, 1 water, IGNORE otherwise)."""
    lab = np.full(classes.shape, IGNORE, np.uint8)
    lab[classes == otsu.CLASS_LAND] = 0
    lab[classes == otsu.CLASS_WATER] = 1
    return lab


def scene_ok(summary, max_gap_db=1.5, min_water_px=2000, max_uncertain_ratio=0.10):
    """Keep a scene as pseudo-label source only when edge- and binary-Otsu agree and it shows a real water body."""
    if summary.get("mode") != "both":
        return False
    if (summary.get("gap_db") or 0.0) > max_gap_db or summary.get("water_px", 0) < min_water_px:
        return False
    return summary.get("uncertain_px", 0) / max(1, summary["water_px"]) <= max_uncertain_ratio   # uncertain << water body


def normalise_s1(vv_db, vh_db, modality="S1RTC"):
    """(2, h, w) float32 standardised with the TerraMind statistics of `modality`; NaN -> 0 (i.e. the mean)."""
    mean, std = S1_STATS[modality]
    x = np.stack([vv_db, vh_db]).astype(np.float32)
    x = (x - mean[:, None, None]) / std[:, None, None]
    return np.nan_to_num(x, nan=0.0)


def extract_patches(x, y, size=224, stride=112, min_valid=0.9, min_water=0.0, max_water=1.0):
    """Sliding-window patches of image x (2, h, w) and labels y (h, w). Returns (X (n,2,s,s), Y (n,s,s), corners).
    Patches need >= min_valid labelled pixels; water share (of labelled) inside [min_water, max_water]."""
    _, h, w = x.shape
    X, Y, C = [], [], []
    for r in range(0, h - size + 1, stride):
        for c in range(0, w - size + 1, stride):
            yy = y[r:r + size, c:c + size]
            lab = yy != IGNORE
            if lab.mean() < min_valid:
                continue
            wf = (yy[lab] == 1).mean() if lab.any() else 0.0
            if not (min_water <= wf <= max_water):
                continue
            X.append(x[:, r:r + size, c:c + size])
            Y.append(yy)
            C.append((r, c))
    if not X:
        return np.zeros((0, 2, size, size), np.float32), np.zeros((0, size, size), np.uint8), []
    return np.stack(X), np.stack(Y), C


def spatial_split(corners, shape, size=224, val_frac=0.25):
    """Train/validation indices by BLOCK (east strip held out), never by random patch: overlapping neighbours of a
    validation patch must not sit in training."""
    h, w = shape
    cut = int(w * (1.0 - val_frac))
    val = [i for i, (r, c) in enumerate(corners) if c >= cut]
    train = [i for i, (r, c) in enumerate(corners) if c + size <= cut]
    return train, val
