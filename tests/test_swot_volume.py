"""swot_volume: SWOT table filtering, area-elevation fit, volume integral, pairing with Sentinel-1 (all offline)."""
import numpy as np
import pandas as pd
import pytest

from swot_volume import hypsometry as hy, integrate, swot


def frames(n=12, bad=True):
    # reservoir with A = 2 (h - 170) km2 for h in 175..205
    rng = np.random.default_rng(0)
    h = np.linspace(175, 205, n)
    df = pd.DataFrame({"lake_id": "4000000001", "wse": h, "wse_u": 0.1, "area_total": 2 * (h - 170) * 1e6, "partial_f": 0, "xovr_cal_q": 0,
                       "quality_f": 0, "dark_frac": 0.1,
                       "time_str": pd.date_range("2024-01-01 02:00", periods=n, freq="12D").strftime("%Y-%m-%dT%H:%M:%SZ")})
    if bad:
        df.loc[0, "quality_f"] = 3
        df.loc[1, "partial_f"] = 1
        df.loc[2, "wse"] = -1e12
    return df


def test_observations_filter_and_fill_values():
    obs = swot.observations_from_frames(frames())
    assert len(obs) == 9 and (obs["wse_m"] > 100).all() and (obs["partial"] == 0).all()
    assert abs(obs["swot_area_km2"].iloc[0] - 2 * (obs["wse_m"].iloc[0] - 170)) < 1e-6


def test_split_polygons_combine():
    f = frames(2, bad=False)
    f.loc[1, "time_str"] = f.loc[0, "time_str"]
    f["area_total"] = [3e6, 1e6]
    f["wse"] = [100.0, 104.0]
    o = swot.observations_from_frames(f)
    assert len(o) == 1 and o["swot_area_km2"][0] == 4.0 and o["wse_m"][0] == 101.0


def test_curve_and_volume_match_analytic():
    h = np.linspace(175, 205, 40)
    c = hy.fit_curve(h, 2 * (h - 170), bin_m=0.1)
    assert np.allclose(c.area([180, 200]), [20, 60], atol=0.3)
    v = hy.volume_series(c, np.array([175.0, 200.0]))           # int 2(h-170) dh = (h-170)^2
    assert v[0] == 0 and abs(v[1] - (30 ** 2 - 5 ** 2)) < 5
    assert abs(hy.volume_series(c, np.array([175.0]), h_ref=175, v_ref=100)[0] - 100) < 1e-9


def test_isotonic_removes_noise_decrease():
    c = hy.fit_curve([1, 2, 3, 4], [10, 12, 11, 14], bin_m=0.5)
    assert np.all(np.diff(c.a) >= 0)


def test_power_fit_and_too_few_pairs():
    h = np.linspace(175, 205, 30)
    c = hy.fit_curve(h, 0.5 * (h - 165) ** 1.5, method="power", bin_m=0.1)
    assert abs(c.area(190.0) - 0.5 * 25 ** 1.5) < 3
    with pytest.raises(ValueError):
        hy.fit_curve([1, 2], [1, 2])


def test_compute_end_to_end():
    obs = swot.observations_from_frames(frames(bad=False))
    s1 = pd.DataFrame({"date": pd.to_datetime(obs["date"]) + pd.Timedelta(days=2), "s1_area_km2": 2 * (obs["wse_m"] - 170) + 0.2})
    t, curve = integrate.compute(obs, s1, max_lag_days=3)
    assert t["lag_days"].eq(2).all() and (t["volume_basis"] == "relative to lowest observed level").all()
    assert t["volume_mcm"].is_monotonic_increasing and integrate.rmse_of_curve(t) < 0.5
    t2, _ = integrate.compute(obs, s1, 3, h_ref=obs["wse_m"].min(), v_ref=50.0)
    assert abs(t2["volume_mcm"].iloc[0] - 50) < 1e-6 and (t2["volume_basis"] == "absolute").all()
    with pytest.raises(ValueError):
        integrate.compute(obs, s1, 3, v_ref=5.0)


def test_s1_daily_area_drops_partial_scenes():
    df = pd.DataFrame({"site": "angat_dam", "date": ["2024-01-01"] * 2 + ["2024-01-13"], "orbit": ["asc", "desc", "asc"],
                       "water_km2": [20.0, 22.0, 5.0], "valid_km2": [100.0, 100.0, 40.0]})
    out = integrate.s1_daily_area(df, site="angat_dam")
    assert len(out) == 1 and out["s1_area_km2"][0] == 21.0


def test_granule_name_parsing_and_filter():
    n = "SWOT_L2_HR_LakeSP_Obs_010_013_AS_20240101T000000_20240101T001000_PIC0_01.zip"
    assert swot.parse_granule_name(n) == {"kind": "Obs", "cycle": 10, "pass": 13, "continent": "AS"}
    assert swot.parse_granule_name("something_else.zip") is None
    assert swot.wanted(n) and not swot.wanted(n, kind="Prior") and not swot.wanted(n.replace("_AS_", "_NA_"))
    assert swot.wanted(n.replace("_AS_", "_NA_"), continent=None) and swot.wanted(n, passes=[13]) and not swot.wanted(n, passes=[14])
