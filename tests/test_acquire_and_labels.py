"""Offline tests of the pure parts of acquisition (orbit choice, solar day) and of the pseudo-label builder."""
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("shapely")
from shapely.geometry import box, mapping  # noqa: E402

from angat_water import acquire, otsu, pseudo_labels as pl  # noqa: E402

BBOX = acquire.ANGAT_BBOX


def _item(orbit, state, when, footprint=(120.5, 14.0, 121.8, 15.5)):
    return SimpleNamespace(properties={"sat:relative_orbit": orbit, "sat:orbit_state": state},
                           geometry=mapping(box(*footprint)), bbox=list(footprint),
                           datetime=datetime(*when, tzinfo=timezone.utc))


def test_choose_orbit_picks_the_orbit_with_most_full_cover_days():
    a = [_item(61, "ascending", (2024, 1, d, 10)) for d in (1, 13, 25)]
    b = [_item(112, "descending", (2024, 1, d, 22)) for d in (4, 16)]
    partial = [_item(61, "ascending", (2024, 2, d, 10), footprint=(121.2, 14.0, 121.8, 15.5)) for d in (6, 18)]
    key, items = acquire.choose_orbit(a + b + partial, BBOX, min_cover=0.9)
    assert key == (61, "ascending") and len(items) == 5  # partial scenes belong to the orbit but are not counted


def test_choose_orbit_none_when_nothing_covers():
    far = [_item(61, "ascending", (2024, 1, 1, 10), footprint=(100, 0, 101, 1))]
    assert acquire.choose_orbit(far, BBOX) == (None, [])


def test_solar_day_rolls_over_for_evening_utc_pass_in_the_philippines():
    d = acquire.solar_dates([np.datetime64("2024-01-01T22:00:00")], 121.1)[0]
    assert str(d) == "2024-01-02"


def test_scene_ok_requires_agreement_and_a_real_water_body():
    good = {"mode": "both", "gap_db": 0.3, "water_px": 50000, "uncertain_px": 100}
    assert pl.scene_ok(good)
    assert not pl.scene_ok({**good, "mode": "edge_only"})
    assert not pl.scene_ok({**good, "gap_db": 2.5})
    assert not pl.scene_ok({**good, "water_px": 500})


def test_patches_ignore_uncertain_and_split_by_block():
    classes = np.zeros((512, 512), np.uint8)
    classes[100:300, 100:300] = otsu.CLASS_WATER
    classes[50:60, 50:60] = otsu.CLASS_UNCERTAIN
    y = pl.labels_from_classes(classes)
    assert y[55, 55] == pl.IGNORE and y[150, 150] == 1 and y[0, 0] == 0
    x = pl.normalise_s1(np.full((512, 512), -12.6, np.float32), np.full((512, 512), -20.3, np.float32))
    X, Y, corners = pl.extract_patches(x, y, size=128, stride=64, min_valid=0.95)
    assert X.shape[1:] == (2, 128, 128) and len(corners) == len(Y) > 0
    tr, va = pl.spatial_split(corners, (512, 512), size=128, val_frac=0.25)
    assert tr and va and not set(tr) & set(va)
    assert max(corners[i][1] + 128 for i in tr) <= 384 <= min(corners[i][1] for i in va) + 128
