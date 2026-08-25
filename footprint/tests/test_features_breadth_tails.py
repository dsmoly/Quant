"""Features, effective breadth, and tail-risk estimators."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import features as F
from src.breadth import (breadth_report, cov_to_corr, effective_n_mp,
                         effective_n_participation, fundamental_law_ir,
                         residualise_covariance, variance_explained)
from src.loader import Panel, load_panel
from src.tailrisk import (downside_ratio, expected_shortfall, fit_gpd, hill_plot,
                          realized_semivariance, select_threshold, tail_report,
                          value_at_risk)
from src.universe import SURVIVORSHIP_NOTE, liquid_universe, universe_turnover

PANEL = Path(__file__).parent / "fixtures" / "panel"


@pytest.fixture(scope="module")
def panel():
    return load_panel(PANEL)


def _toy_panel(close, volume):
    idx = pd.bdate_range("2020-01-01", periods=len(close))
    c = pd.DataFrame({"A": close}, index=idx)
    v = pd.DataFrame({"A": volume}, index=idx)
    return Panel(close=c, open=c, high=c, low=c, volume=v)


# ------------------------------------------------------------------ features


def test_amihud_is_impact_per_dollar_and_rises_when_liquidity_falls():
    n = 40
    ret = np.r_[np.zeros(1), np.full(n - 1, 0.01)]
    close = 100 * np.cumprod(1 + ret)
    liquid = _toy_panel(close, np.full(n, 1e7))
    illiquid = _toy_panel(close, np.full(n, 1e5))
    a_liq = F.amihud(liquid, window=10).iloc[-1, 0]
    a_ill = F.amihud(illiquid, window=10).iloc[-1, 0]
    assert a_ill > a_liq
    assert a_ill / a_liq == pytest.approx(100.0, rel=1e-6)


def test_daily_impact_is_nan_when_volume_is_zero():
    p = _toy_panel(np.array([10.0, 11.0, 12.0]), np.array([1e6, 0.0, 1e6]))
    assert np.isnan(F.daily_impact(p).iloc[1, 0])


def test_abnormal_lambda_sign_convention():
    """Below-normal impact = liquidity supplied (negative); above = demanded."""
    rng = np.random.default_rng(0)
    n = 120
    ret = rng.normal(scale=0.01, size=n)
    close = 100 * np.cumprod(1 + ret)
    vol = np.full(n, 1e6)
    # Final day: same move, ten times the volume -> impact per dollar collapses.
    vol[-1] = 1e7
    p = _toy_panel(close, vol)
    assert F.abnormal_lambda(p, window=60).iloc[-1, 0] < 0

    vol2 = np.full(n, 1e6)
    vol2[-1] = 1e5                      # same move on a tenth of the volume
    p2 = _toy_panel(close, vol2)
    assert F.abnormal_lambda(p2, window=60).iloc[-1, 0] > 0


def test_abnormal_lambda_baseline_excludes_today():
    """The trailing median must not contain the observation being scored."""
    n = 100
    rng = np.random.default_rng(1)
    close = 100 * np.cumprod(1 + rng.normal(scale=0.01, size=n))
    p = _toy_panel(close, np.full(n, 1e6))
    impact = F.daily_impact(p)
    base_manual = impact["A"].iloc[99 - 60:99].median()
    got = F.abnormal_lambda(p, window=60).iloc[99, 0]
    assert got == pytest.approx(np.log(impact["A"].iloc[99] / base_manual), rel=1e-9)


def test_persistent_imbalance_signs_with_the_run_and_scales_with_volume():
    n = 90
    ret = np.r_[np.zeros(n - 4), np.full(4, 0.01)]      # four up days at the end
    close = 100 * np.cumprod(1 + ret)
    quiet = _toy_panel(close, np.full(n, 1e6))
    heavy_vol = np.full(n, 1e6)
    heavy_vol[-4:] = 3e6
    heavy = _toy_panel(close, heavy_vol)
    q = F.persistent_imbalance(quiet).iloc[-1, 0]
    h = F.persistent_imbalance(heavy).iloc[-1, 0]
    assert q > 0 and h > q                       # same run, heavier volume scores more

    down = _toy_panel(100 * np.cumprod(1 + np.r_[np.zeros(n - 4), np.full(4, -0.01)]),
                      np.full(n, 1e6))
    assert F.persistent_imbalance(down).iloc[-1, 0] < 0


def test_persistent_imbalance_caps_the_run():
    n = 120
    long_run = _toy_panel(100 * np.cumprod(1 + np.r_[np.zeros(20), np.full(n - 20, 0.005)]),
                          np.full(n, 1e6))
    v = F.persistent_imbalance(long_run, max_run=5)
    # Once past the cap the score stops growing with run length.
    tail = v["A"].iloc[-20:]
    assert tail.max() / tail.min() < 1.5


def test_dollar_volume_anomaly_is_zero_when_dollar_volume_is_flat():
    """The feature is about *dollar* volume, so a drifting price on constant share
    volume is a genuine anomaly, not a bug. Holding notional flat zeroes it."""
    n = 100
    close = 100 * np.cumprod(1 + np.full(n, 0.001))
    p = _toy_panel(close, 1e8 / close)                  # constant dollar volume
    assert F.dollar_volume_anomaly(p, window=60).iloc[-1, 0] == pytest.approx(0.0, abs=1e-9)


def test_dollar_volume_anomaly_sees_a_drifting_notional():
    n = 100
    close = 100 * np.cumprod(1 + np.full(n, 0.001))
    p = _toy_panel(close, np.full(n, 1e6))              # notional drifts with price
    got = F.dollar_volume_anomaly(p, window=60).iloc[-1, 0]
    assert got == pytest.approx(0.001 * 30, abs=2e-3)   # ~half a window of drift


def test_features_use_no_future_information(panel):
    """Truncating the panel must not change any feature value already computed."""
    full = F.compute_features(panel)
    cut = len(panel.dates) - 30
    truncated = F.compute_features(panel.slice_dates(end=panel.dates[cut - 1]))
    for name, frame in truncated.items():
        a = full[name].iloc[:cut]
        b = frame.iloc[:cut]
        pd.testing.assert_frame_equal(a, b, check_freq=False)


def test_footprint_score_combines_and_is_finite(panel):
    s = F.footprint_score(panel)
    assert s.shape == panel.close.shape
    assert np.isfinite(s.to_numpy()[-50:]).any()


def test_footprint_score_needs_a_nonzero_sign(panel):
    with pytest.raises(ValueError):
        F.footprint_score(panel, signs={"amihud": 0.0})


def test_cross_sectional_z_is_centred(panel):
    z = F.cross_sectional_z(F.amihud(panel))
    row = z.iloc[-1].dropna()
    assert row.mean() == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------- universe


def test_liquid_universe_is_point_in_time(panel):
    mask = liquid_universe(panel, n=4, window=20)
    assert mask.sum(axis=1).max() <= 4
    # Selection uses lagged volume, so the first window is empty.
    assert not mask.iloc[:5].to_numpy().any()


def test_universe_turnover_is_zero_when_membership_is_constant():
    idx = pd.bdate_range("2020-01-01", periods=10)
    mask = pd.DataFrame(True, index=idx, columns=["A", "B"])
    assert universe_turnover(mask).iloc[1:].sum() == 0.0


def test_survivorship_note_is_specific():
    assert "delist" in SURVIVORSHIP_NOTE.lower()
    assert "upper bound" in SURVIVORSHIP_NOTE.lower()


# -------------------------------------------------------------------- breadth


def test_effective_n_known_answers():
    assert effective_n_participation(np.eye(20)) == pytest.approx(20.0)
    assert effective_n_participation(np.ones((20, 20))) == pytest.approx(1.0)


def test_effective_n_equicorrelation_matches_the_analytic_value():
    n, rho = 25, 0.5
    corr = np.full((n, n), rho) + np.eye(n) * (1 - rho)
    # Eigenvalues: 1+(n-1)rho once, and 1-rho (n-1) times.
    big = 1 + (n - 1) * rho
    small = 1 - rho
    expected = (big + (n - 1) * small) ** 2 / (big ** 2 + (n - 1) * small ** 2)
    assert effective_n_participation(corr) == pytest.approx(expected)


def test_effective_n_is_scale_invariant():
    cov = np.diag([1.0, 4.0, 9.0])
    assert effective_n_participation(cov) == pytest.approx(
        effective_n_participation(cov * 1000))


def test_residualising_removes_the_dominant_factor():
    n = 12
    corr = np.full((n, n), 0.6) + np.eye(n) * 0.4
    resid = residualise_covariance(corr, n_factors=1)
    assert variance_explained(resid)[0] < variance_explained(corr)[0]
    assert effective_n_participation(resid) > effective_n_participation(corr)
    # Still positive semi-definite.
    assert np.linalg.eigvalsh(resid).min() > -1e-9


def test_cov_to_corr_has_unit_diagonal():
    cov = np.array([[4.0, 1.0], [1.0, 9.0]])
    c = cov_to_corr(cov)
    assert np.allclose(np.diag(c), 1.0)
    assert c[0, 1] == pytest.approx(1 / 6)


def test_mp_count_finds_no_factor_in_pure_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 20))
    corr = np.corrcoef(x, rowvar=False)
    assert effective_n_mp(corr, 2000) == 0


def test_mp_count_finds_a_planted_factor():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(2000, 1))
    x = f @ np.ones((1, 20)) + rng.normal(size=(2000, 20))
    assert effective_n_mp(np.corrcoef(x, rowvar=False), 2000) >= 1


def test_fundamental_law_uses_the_breadth_it_is_given():
    assert fundamental_law_ir(0.03, 25) > fundamental_law_ir(0.03, 9)
    assert fundamental_law_ir(0.03, 0) == 0.0


def test_breadth_report_on_a_degenerate_block():
    r = pd.DataFrame({"A": [0.01, 0.02, 0.03]})
    rep = breadth_report(r)
    assert rep["n_names"] == 1


# ------------------------------------------------------------------ tail risk


@pytest.fixture(scope="module")
def fat_returns():
    rng = np.random.default_rng(0)
    return pd.Series(rng.standard_t(3, size=3000) * 0.01)


def test_expected_shortfall_exceeds_var_and_is_positive(fat_returns):
    es = expected_shortfall(fat_returns, 0.05)
    var = value_at_risk(fat_returns, 0.05)
    assert es > var > 0


def test_expected_shortfall_on_a_known_distribution():
    """Uniform losses: ES at 5% is the mean of the top 5%."""
    x = pd.Series(-np.linspace(0.0, 1.0, 10001))       # losses 0..1
    assert expected_shortfall(x, 0.05) == pytest.approx(0.975, abs=2e-3)


def test_semivariance_is_below_total_vol_and_ratio_near_root_half_when_symmetric():
    rng = np.random.default_rng(1)
    sym = pd.Series(rng.normal(scale=0.01, size=20000))
    assert downside_ratio(sym) == pytest.approx(1 / np.sqrt(2), abs=0.02)


def test_semivariance_detects_asymmetry():
    rng = np.random.default_rng(2)
    x = rng.normal(scale=0.01, size=20000)
    x[x < 0] *= 2.0                                   # fatten the left side only
    assert downside_ratio(pd.Series(x)) > 1 / np.sqrt(2) + 0.05
    assert realized_semivariance(pd.Series(x)) > 0


def test_hill_plot_returns_a_curve_not_a_point(fat_returns):
    hp = hill_plot(fat_returns)
    assert hp.k.size > 20
    assert hp.alpha.size == hp.k.size
    assert hp.plateau_lo <= hp.plateau_hi
    assert "tail index in" in hp.summary()


def test_hill_recovers_the_order_of_magnitude_for_a_pareto_tail():
    """Hill lands near the truth but is biased, which is the whole point.

    On 20k exact-Pareto draws with alpha=3 the plateau sits at roughly [3.09,
    3.11] -- close, but not bracketing the true value. That residual bias on
    *ideal* data is precisely why this project reports a range and refuses to
    quote a point estimate on real returns, where the tail is not exactly Pareto
    and the sample is two orders of magnitude smaller.
    """
    rng = np.random.default_rng(3)
    alpha_true = 3.0
    x = pd.Series(-(rng.pareto(alpha_true, size=20000) + 1))
    hp = hill_plot(x)
    mid = 0.5 * (hp.plateau_lo + hp.plateau_hi)
    assert abs(mid - alpha_true) < 0.3
    assert hp.plateau_hi - hp.plateau_lo < 1.0


def test_hill_declines_to_report_on_a_short_sample():
    hp = hill_plot(pd.Series(np.random.default_rng(0).normal(size=20)))
    assert np.isnan(hp.plateau_lo)


def test_gpd_fit_recovers_a_planted_shape():
    rng = np.random.default_rng(4)
    from scipy import stats as st
    xi_true = 0.25
    exceed = st.genpareto.rvs(xi_true, scale=0.01, size=4000, random_state=rng)
    losses = pd.Series(-(exceed + 0.02))
    fit = fit_gpd(losses, threshold=0.02)
    assert fit.converged
    assert fit.shape == pytest.approx(xi_true, abs=0.08)
    assert fit.tail_index == pytest.approx(1 / xi_true, rel=0.4)


def test_gpd_declines_on_too_few_exceedances():
    fit = fit_gpd(pd.Series(-np.linspace(0.0, 0.1, 50)), threshold=0.099)
    assert not fit.converged
    assert np.isnan(fit.shape)


def test_threshold_table_reports_every_candidate(fat_returns):
    t = select_threshold(fat_returns)
    assert len(t) > 10
    assert {"threshold", "shape", "ks_p", "n_exceed", "recommended"} <= set(t.columns)
    assert t["recommended"].sum() <= 1
    assert (t["threshold"].diff().dropna() >= -1e-12).all()   # monotone in quantile


def test_infinite_mean_tail_reports_no_expected_shortfall():
    """xi >= 1 means the GPD has no finite mean; the fit must not invent one."""
    from src.tailrisk import GPDFit
    fit = GPDFit(threshold=0.02, n_exceed=100, shape=1.4, scale=0.01,
                 ad_stat=0.0, ks_p=0.5, converged=True)
    assert np.isnan(fit.expected_shortfall(0.01, 1000))


def test_tail_report_leads_with_the_nonparametric_numbers(fat_returns):
    rep = tail_report(fat_returns)
    assert rep["expected_shortfall"] > 0
    assert rep["expected_shortfall_se"] > 0
    assert len(rep["hill_range"]) == 2
    assert rep["excess_kurtosis"] > 1        # it is a t(3)
