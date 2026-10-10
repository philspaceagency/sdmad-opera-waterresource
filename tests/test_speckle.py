import numpy as np
from scipy import ndimage as ndi

from ai_water import speckle, synthetic as syn


def _cv(a, mask):
    return float(np.nanstd(a[mask]) / np.nanmean(a[mask]))


def test_refined_lee_reduces_speckle_and_keeps_shape():
    s = syn.make_scene(water_frac=0.2, looks=4, seed=0)
    f = speckle.refined_lee(s["vv"])
    assert f.shape == s["vv"].shape and f.dtype == np.float32
    inner = ndi.binary_erosion(s["truth"], iterations=6)
    assert _cv(f, inner) < 0.6 * _cv(s["vv"], inner)


def test_refined_lee_preserves_mean_in_linear_power():
    s = syn.make_scene(water_frac=0.2, looks=4, seed=1)
    f = speckle.refined_lee(s["vv"])
    assert abs(f.mean() / s["vv"].mean() - 1.0) < 0.03


def test_refined_lee_keeps_a_step_edge_sharper_than_boxcar():
    rng = np.random.default_rng(0)
    clean = np.ones((128, 128), np.float32)
    clean[:, 64:] = 0.05
    noisy = clean * rng.gamma(4, 0.25, clean.shape).astype(np.float32)
    ref = speckle.refined_lee(noisy)
    box = ndi.uniform_filter(noisy, 7)
    # one-pixel step profile across the edge, away from the borders
    step = lambda a: float(np.median(a[20:108, 62]) - np.median(a[20:108, 66]))
    assert step(ref) > step(box)


def test_nan_pixels_stay_nan_and_do_not_leak():
    s = syn.make_scene(shape=(128, 128), seed=2)["vv"].copy()
    s[:10, :] = np.nan
    f = speckle.refined_lee(s)
    assert np.isnan(f[:10]).all() and np.isfinite(f[10:]).all()


def test_enl_argument_and_lee_agree_roughly():
    s = syn.make_scene(shape=(128, 128), seed=3)["vv"]
    a, b = speckle.refined_lee(s, enl=4), speckle.lee(s, enl=4)
    assert abs(a.mean() - b.mean()) / b.mean() < 0.05


def test_multitemporal_mean():
    stack = np.stack([np.full((4, 4), v, np.float32) for v in (1.0, 2.0, 3.0)])
    stack[:, 0, 0] = np.nan
    stack[1:, 1, 1] = np.nan
    m = speckle.multitemporal_mean(stack, min_obs=2)
    assert np.isnan(m[0, 0]) and np.isnan(m[1, 1]) and m[2, 2] == 2.0
