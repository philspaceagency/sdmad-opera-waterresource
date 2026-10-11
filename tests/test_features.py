"""SDWI, terrain derivatives, land-cover groups and the channel stack (pure NumPy)."""
import numpy as np
import pytest

from ai_water import features as ft, synthetic as syn


def test_sdwi_is_ln_10_vv_vh_minus_8_on_linear_power():
    vv, vh = np.array([0.1, 0.01], np.float32), np.array([0.02, 0.001], np.float32)
    assert np.allclose(ft.sdwi(vv, vh), np.log(10 * vv * vh) - 8, atol=1e-5)
    s = ft.sdwi(np.array([0.1, -1.0, np.nan]), np.array([0.1, 0.1, 0.1]))
    assert np.isfinite(s[0]) and np.isnan(s[1]) and np.isnan(s[2])


def test_plane_rising_to_the_east_faces_west_with_the_right_slope():
    res = 10.0
    dem = np.tile(np.arange(64, dtype=np.float32) * res * 0.5, (64, 1))        # 0.5 m per m eastwards
    t = ft.terrain_from_dem(dem, res)
    assert np.allclose(t.slope[8:-8, 8:-8], np.degrees(np.arctan(0.5)), atol=0.1)
    assert np.allclose(t.aspect[8:-8, 8:-8], 270.0, atol=0.5)                  # downslope = west


def test_plane_rising_to_the_north_faces_south():
    dem = np.tile((np.arange(64, dtype=np.float32) * 5.0)[::-1, None], (1, 64))  # row 0 is north and highest
    t = ft.terrain_from_dem(dem, 10.0)
    assert np.allclose(t.aspect[8:-8, 8:-8], 180.0, atol=0.5)


def test_relief_is_height_above_the_local_low_not_absolute_altitude():
    dem = np.full((200, 200), 500.0, np.float32)
    dem[:, 100:] += 80.0
    assert ft.terrain_from_dem(dem, 10.0).relief.max() == pytest.approx(80.0)
    assert ft.terrain_from_dem(dem + 1000.0, 10.0).relief.max() == pytest.approx(80.0)    # same at any altitude


def test_range_slope_is_positive_on_the_slope_facing_away_from_the_sensor():
    slope, asp = np.full((1, 2), 30.0, np.float32), np.array([[280.0, 100.0]], np.float32)
    d = ft.range_slope(slope, asp, "descending")           # looks west (280): a slope facing 280 faces AWAY
    assert d[0, 0] == pytest.approx(30.0, abs=0.1) and d[0, 1] == pytest.approx(-30.0, abs=0.1)
    a = ft.range_slope(slope, asp, "ascending")
    assert a[0, 0] < 0 < a[0, 1]                          # the opposite geometry swaps which flank is shadowed
    with pytest.raises(ValueError):
        ft.range_slope(slope, asp, "sideways")


def test_landcover_groups_never_encode_water():
    wc = np.array([[10, 20, 30, 40, 50, 60, 80, 90, 95, 100, 0]], np.uint8)
    assert ft.landcover_groups(wc).tolist() == [[1, 2, 2, 3, 4, 5, 0, 6, 6, 5, 0]]      # 80 (water) -> 0


def test_compose_has_the_documented_channels_and_context_can_be_dropped():
    s = syn.make_terrain_scene(shape=(96, 96), seed=1)
    t = ft.terrain_from_dem(s["dem"], 10.0, ft.landcover_groups(np.full((96, 96), 10)))
    from ai_water import scene
    f = scene.prepare(s["vv"], s["vh"])
    x = ft.compose(f.vv_db, f.vh_db, f.sdwi, t, "descending")
    assert x.shape == (ft.N_CHANNELS, 96, 96) and x.dtype == np.float32 and np.isfinite(x).all()
    assert x[ft.LC_CH.start].min() == 1.0 and x[ft.LC_CH.start + 1:].max() == 0.0       # all 'tree'
    z = ft.compose(f.vv_db, f.vh_db, f.sdwi, t, "descending", drop_context=True)
    assert (z[ft.LC_CH] == 0).all() and np.array_equal(z[:ft.LC_CH.start], x[:ft.LC_CH.start])
    # the shadowed flank is the one the range-slope channel marks as facing away from the radar
    sh = s["shadow"]
    assert x[7][sh].mean() > 0.4 and x[7][s["truth"]].mean() == pytest.approx(0.0, abs=0.05)
    with pytest.raises(ValueError):
        ft.compose(f.vv_db[:50], f.vh_db[:50], f.sdwi[:50], t, "descending")


def test_terrain_save_load_and_crop(tmp_path):
    s = syn.make_terrain_scene(shape=(64, 64))
    t = ft.terrain_from_dem(s["dem"], 10.0, np.full((64, 64), 3, np.uint8))
    t.save(tmp_path / "t.npz")
    u = ft.Terrain.load(tmp_path / "t.npz")
    assert np.allclose(u.slope, t.slope, atol=0.1) and (u.landcover == 3).all()
    assert t.crop(8, 16, 32).shape == (32, 32)
