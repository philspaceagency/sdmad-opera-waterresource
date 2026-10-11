"""The model's input channels: Sentinel-1 (VV, VH, SDWI), terrain (relief, slope, aspect, range slope) and land cover.

    channel  0  VV dB            }  the two S1RTC channels the pretrained TerraMind encoder was trained on
             1  VH dB            }
             2  SDWI             ln(10 * VV * VH) - 8 on speckle-filtered linear power (Dual-Polarized Water Index)
             3  relief           height above the lowest DEM point within ~1.5 km (m / 100): NOT the absolute altitude, which
                                 would only fingerprint the six training sites
             4  slope            degrees / 30
             5  aspect sin       }  direction the slope faces (compass, clockwise from north), fading to 0 on flat ground
             6  aspect cos       }
             7  range slope      slope component along the radar look direction (deg / 45). > 0 = faces away from the sensor
                                 (radar shadow when steep), < 0 = faces the sensor (foreshortening / layover). Depends on the
                                 pass direction, so it is computed per scene
             8..13 land cover    one-hot of 6 groups (tree, shrub/grass, crop, built, bare, wetland); water is NOT a group

Land cover is context only: it is dropped at random while training (`CONTEXT_DROPOUT`) so the model cannot lean on it.
Only channels 0-1 go through the pretrained encoder; the rest reach the decoder through the pixel stem (late fusion).
Everything here is pure NumPy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

IGNORE = 255

# TerraMind per-modality input normalisation of (VV, VH) in dB, as in terratorch 1.2.x
# (terratorch/models/backbones/terramind/model/terramind_register.py). Planetary Computer's `sentinel-1-rtc` is
# terrain-corrected gamma0, i.e. the S1RTC modality: S1GRD statistics would shift the input by ~2 dB.
S1_STATS = {
    "S1RTC": (np.array([-10.93, -17.329], np.float32), np.array([4.391, 4.459], np.float32)),
    "S1GRD": (np.array([-12.599, -20.293], np.float32), np.array([5.195, 5.890], np.float32)),
}
SDWI_MEAN, SDWI_STD = -12.0, 3.0

LC_GROUPS = ("tree", "shrub_grass", "crop", "built", "bare", "wetland")
# ESA WorldCover class value -> group (1-based); 80 permanent water and anything else -> 0 (all-zero: unknown)
WORLDCOVER_GROUP = {10: 1, 20: 2, 30: 2, 40: 3, 50: 4, 60: 5, 70: 5, 100: 5, 90: 6, 95: 6}

CHANNELS = (["vv_db", "vh_db", "sdwi", "relief", "slope", "aspect_sin", "aspect_cos", "range_slope"]
            + [f"lc_{g}" for g in LC_GROUPS])
N_CHANNELS = len(CHANNELS)
N_ENCODER = 2                                    # channels fed to the pretrained encoder
S1_CH = slice(0, 3)
TERRAIN_CH = slice(3, 8)
LC_CH = slice(8, 8 + len(LC_GROUPS))
CONTEXT_DROPOUT = 0.5                            # share of training samples whose land-cover channels are zeroed

# nominal look azimuth of Sentinel-1 (right-looking): heading ~350 deg ascending, ~190 deg descending
LOOK_AZIMUTH = {"ascending": 80.0, "descending": 280.0}


# ---------------------------------------------------------------------------------------- Sentinel-1
def sdwi(vv_lin, vh_lin):
    """Sentinel-1 Dual-Polarized Water Index, SDWI = ln(10 * VV * VH) - 8 (Jia et al. 2019), on (filtered) LINEAR power.
    NaN where either band is not positive."""
    vv, vh = np.asarray(vv_lin, np.float32), np.asarray(vh_lin, np.float32)
    ok = (vv > 0) & (vh > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(10.0 * np.where(ok, vv, np.nan) * np.where(ok, vh, np.nan)) - 8.0
    return out.astype(np.float32)


# ---------------------------------------------------------------------------------------- terrain
@dataclass
class Terrain:
    """Static per-site layers on the site's grid: relief (m), slope and aspect (degrees, aspect = compass azimuth the slope
    faces, 0-360, NaN never), land-cover group (uint8, 0 = none/water, 1..6 = LC_GROUPS)."""
    relief: np.ndarray
    slope: np.ndarray
    aspect: np.ndarray
    landcover: np.ndarray

    @property
    def shape(self):
        return self.slope.shape

    def save(self, path):
        np.savez_compressed(path, relief=self.relief.astype(np.float16), slope=self.slope.astype(np.float16),
                            aspect=self.aspect.astype(np.float16), landcover=self.landcover.astype(np.uint8))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["relief"].astype(np.float32), z["slope"].astype(np.float32), z["aspect"].astype(np.float32),
                   z["landcover"].astype(np.uint8))

    def crop(self, r, c, size):
        s = (slice(r, r + size), slice(c, c + size))
        return Terrain(self.relief[s], self.slope[s], self.aspect[s], self.landcover[s])


def terrain_from_dem(dem, res_m, landcover=None, relief_window_m=1500.0):
    """Relief, slope and aspect from a DEM (m) on a regular grid with square `res_m` pixels, rows running north -> south.
    NaN cells are filled with the DEM median first. `landcover` is the group raster (see `landcover_groups`)."""
    dem = np.asarray(dem, np.float32)
    if not np.isfinite(dem).all():
        fill = np.nanmedian(dem) if np.isfinite(dem).any() else 0.0
        dem = np.where(np.isfinite(dem), dem, fill).astype(np.float32)
    drow, dcol = np.gradient(dem, res_m)                 # d/d(south), d/d(east)
    slope = np.degrees(np.arctan(np.hypot(drow, dcol))).astype(np.float32)
    # the slope FACES the direction of steepest descent: east = -dz/deast, north = +dz/dsouth
    aspect = (np.degrees(np.arctan2(-dcol, drow)) % 360.0).astype(np.float32)
    k = max(3, 2 * int(relief_window_m / res_m / 2) + 1)
    relief = (dem - ndi.minimum_filter(dem, size=k, mode="nearest")).astype(np.float32)
    lc = np.zeros(dem.shape, np.uint8) if landcover is None else np.asarray(landcover, np.uint8)
    return Terrain(relief, slope, aspect, lc)


def landcover_groups(worldcover):
    """ESA WorldCover classes -> group raster (0 = unknown / permanent water, 1..6 = LC_GROUPS). Water is deliberately
    NOT a group: the land-cover layer must not hand the model the answer."""
    wc = np.asarray(worldcover)
    out = np.zeros(wc.shape, np.uint8)
    for value, group in WORLDCOVER_GROUP.items():
        out[wc == value] = group
    return out


def range_slope(slope, aspect, orbit_state):
    """Terrain slope along the radar look direction, degrees. Positive = the slope faces away from the sensor (steep ones are in
    radar shadow), negative = it faces the sensor (foreshortening / layover). `orbit_state` 'ascending' | 'descending'."""
    if orbit_state not in LOOK_AZIMUTH:
        raise ValueError(f"orbit_state must be 'ascending' or 'descending', not {orbit_state!r}")
    look = np.radians(LOOK_AZIMUTH[orbit_state])
    return np.degrees(np.arctan(np.tan(np.radians(slope)) * np.cos(np.radians(aspect) - look))).astype(np.float32)


# ---------------------------------------------------------------------------------------- stacking
def s1_channels(vv_db, vh_db, sdwi_):
    """(3, h, w) standardised S1 channels: VV, VH (TerraMind S1RTC statistics) and SDWI. NaN -> 0 (the mean)."""
    mean, std = S1_STATS["S1RTC"]
    x = np.stack([(vv_db - mean[0]) / std[0], (vh_db - mean[1]) / std[1], (sdwi_ - SDWI_MEAN) / SDWI_STD]).astype(np.float32)
    return np.nan_to_num(x, nan=0.0)


def terrain_channels(terrain: Terrain, orbit_state, drop_context=False):
    """(N_CHANNELS - 3, h, w) standardised terrain and land-cover channels for a scene seen in `orbit_state`."""
    x = np.zeros((N_CHANNELS - 3,) + terrain.shape, np.float32)
    x[0] = np.clip(terrain.relief / 100.0, 0.0, 5.0)
    x[1] = np.clip(terrain.slope / 30.0, 0.0, 3.0)
    fade = np.clip(terrain.slope / 5.0, 0.0, 1.0)             # aspect means nothing on flat ground
    x[2] = np.sin(np.radians(terrain.aspect)) * fade
    x[3] = np.cos(np.radians(terrain.aspect)) * fade
    x[4] = np.clip(range_slope(terrain.slope, terrain.aspect, orbit_state) / 45.0, -2.0, 2.0)
    if not drop_context:
        for g in range(1, len(LC_GROUPS) + 1):
            x[5 + g - 1] = terrain.landcover == g
    return x


def compose(vv_db, vh_db, sdwi_, terrain: Terrain, orbit_state, drop_context=False):
    """Standardised (N_CHANNELS, h, w) float32 input of the model for one scene."""
    if terrain.shape != vv_db.shape:
        raise ValueError(f"terrain grid {terrain.shape} does not match the scene {vv_db.shape}")
    return np.concatenate([s1_channels(vv_db, vh_db, sdwi_), terrain_channels(terrain, orbit_state, drop_context)])


def extract_patches(x, y, size=224, stride=112, min_valid=0.9):
    """Sliding-window patches of x (C, h, w) and labels y (h, w). Returns (corners, kept) where `kept` are the corners whose
    patch has >= min_valid labelled pixels. Patches are cut later (`slice`), so nothing is copied here."""
    _, h, w = x.shape
    out = []
    for r in range(0, h - size + 1, stride):
        for c in range(0, w - size + 1, stride):
            if (y[r:r + size, c:c + size] != IGNORE).mean() >= min_valid:
                out.append((r, c))
    return out


def spatial_split(corners, shape, size=224, val_frac=0.25):
    """Train/validation indices by BLOCK (east strip held out), never by random patch: overlapping neighbours of a
    validation patch must not sit in training."""
    h, w = shape
    cut = int(w * (1.0 - val_frac))
    val = [i for i, (r, c) in enumerate(corners) if c >= cut]
    train = [i for i, (r, c) in enumerate(corners) if c + size <= cut]
    return train, val
