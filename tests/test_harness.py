"""Invariants. Each of these pins a bug that was actually found, or a property
the rest of the package assumes."""
from __future__ import annotations

import numpy as np
import pytest

import qr
from qr.panel import forward_return
from qr.sizing import SizingEngine, SizingParams, apply_caps, effective_n_participation


# ------------------------------------------------------------ alignment

def test_forward_return_has_no_lookahead():
    r = np.arange(20, dtype=float).reshape(10, 2)
    assert np.allclose(forward_return(r, 1)[0], r[1])
    assert np.allclose(forward_return(r, 3)[0], r[1:4].sum(0))
    assert np.isnan(forward_return(r, 1)[-1]).all()
    assert np.isnan(forward_return(r, 3)[-3:]).all()


def test_forward_return_propagates_missing():
    r = np.arange(20, dtype=float).reshape(10, 2)
    r[5, 0] = np.nan
    f = forward_return(r, 3)
    assert np.isnan(f[3, 0]) and np.isfinite(f[3, 1])


def test_a_pure_lag_of_returns_has_ic_one():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1, (300, 40))
    fwd = forward_return(r, 1)
    ic = qr.ic_series(fwd, fwd)
    assert np.nanmean(ic) > 0.999


# ------------------------------------------------------------ transforms

def test_rank_normal_is_monotone_invariant():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (50, 30))
    a = qr.rank_normal(x)
    b = qr.rank_normal(np.exp(3 * x))          # any monotone map
    assert np.allclose(a, b, atol=1e-9)


def test_neutralise_removes_the_factor():
    rng = np.random.default_rng(2)
    f = rng.normal(0, 1, (60, 40))
    x = 3.0 * f + rng.normal(0, 0.1, (60, 40))
    res = qr.neutralise(x, [f])
    ic = qr.ic_series(res, f)
    assert abs(np.nanmean(ic)) < 0.15


def test_masked_cells_are_nan_not_zero():
    x = np.ones((5, 4))
    m = np.ones((5, 4), dtype=bool)
    m[:, 0] = False
    assert np.isnan(qr.zscore(x, m)[:, 0]).all()


# ------------------------------------------------------------ inference

def test_newey_west_widens_se_under_positive_autocorrelation():
    rng = np.random.default_rng(3)
    T = 2000
    e = rng.normal(0, 1, T)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.7 * x[t - 1] + e[t]
    naive = x.std(ddof=1) / np.sqrt(T)
    assert qr.newey_west_se(x, lags=20) > 2 * naive


def test_newey_west_matches_naive_for_iid():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 4000)
    naive = x.std(ddof=1) / np.sqrt(x.size)
    assert 0.8 < qr.newey_west_se(x, lags=5) / naive < 1.25


def test_ic_summary_reports_both_t_stats():
    rng = np.random.default_rng(5)
    ic = 0.02 + 0.05 * rng.normal(0, 1, 500)
    s = qr.ic_summary(ic, horizon=10)
    assert s.lags == 9
    assert np.isfinite(s.t_stat) and np.isfinite(s.t_naive)


def test_lo_factor_equals_sqrt_q_for_iid():
    rng = np.random.default_rng(6)
    x = rng.normal(0, 1, 5000)
    assert abs(qr.lo_annualisation_factor(x, 252) - np.sqrt(252)) / np.sqrt(252) < 0.15


def test_lo_factor_penalises_positive_autocorrelation():
    rng = np.random.default_rng(7)
    T = 20000
    e = rng.normal(0, 1, T)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.2 * x[t - 1] + e[t]
    assert qr.lo_annualisation_factor(x, 12) < np.sqrt(12)


# ------------------------------------------------------------ deflation

def test_expected_max_sharpe_grows_with_trials():
    v = [qr.expected_max_sharpe(n) for n in (2, 10, 100, 1000)]
    assert all(a < b for a, b in zip(v, v[1:]))


def test_deflated_sharpe_falls_as_trials_rise():
    rng = np.random.default_rng(8)
    r = rng.normal(0.02, 1.0, 1500)
    d1 = qr.deflated_sharpe(returns=r, n_trials=1)["dsr"]
    d2 = qr.deflated_sharpe(returns=r, n_trials=500)["dsr"]
    assert d1 > d2


def test_min_trl_is_infinite_when_below_benchmark():
    assert qr.min_track_record_length(0.01, benchmark_sr=0.05) == float("inf")


# ------------------------------------------------------------ decay

def test_decay_recovers_a_known_rate():
    N, T, true_beta = 300, 4000, 0.10
    rng = np.random.default_rng(9)
    phi = np.exp(-true_beta)
    s = np.zeros((T, N))
    s[0] = rng.normal(0, 1, N)
    for t in range(1, T):
        s[t] = phi * s[t - 1] + np.sqrt(1 - phi ** 2) * rng.normal(0, 1, N)
    r = 0.08 * s + np.sqrt(1 - 0.08 ** 2) * rng.normal(0, 1, (T, N))
    hs, ic, se, tt = qr.marginal_ic_curve(s, r, horizons=(1, 2, 3, 5, 8, 12, 20))
    fit = qr.fit_decay(hs, ic, se, tt)
    assert 0.5 * true_beta < fit.beta < 2.0 * true_beta


def test_decay_refuses_when_signal_is_noise():
    rng = np.random.default_rng(10)
    s = rng.normal(0, 1, (600, 100))
    r = rng.normal(0, 1, (600, 100))
    hs, ic, se, tt = qr.marginal_ic_curve(s, r, horizons=(1, 2, 3, 5, 10))
    fit = qr.fit_decay(hs, ic, se, tt)
    assert not np.isfinite(fit.beta)
    assert "clear" in fit.note or "non-positive" in fit.note


# ------------------------------------------------------------ cv

def test_purged_kfold_leaves_a_gap_around_the_test_fold():
    cv = qr.PurgedKFold(n_splits=4, horizon=5, embargo=3)
    for train, test in cv.split(400):
        assert not set(train) & set(test)
        lo, hi = test.min(), test.max()
        before = train[train < lo]
        after = train[train > hi]
        if before.size:
            assert lo - before.max() > 5
        if after.size:
            assert after.min() - hi > 3


def test_walk_forward_never_trains_on_the_future():
    for train, test in qr.walk_forward_splits(500, n_splits=4, horizon=5, embargo=2):
        assert train.max() < test.min()


# ------------------------------------------------------------ combine

def test_shrinkage_zeroes_an_unmeasurable_signal():
    a = np.array([1.0, 1.0, 1.0])
    huge = np.array([1e12, 1e12, 1e12])
    assert np.allclose(qr.shrink_by_tstat(a, huge), 0.0)


def test_shrinkage_at_t_equals_one_halves():
    a = np.array([2.0])
    assert np.allclose(qr.shrink_by_tstat(a, np.array([2.0])), 1.0)


# ------------------------------------------------------------ sizing

def _mkt(N=80, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.uniform(0.6, 1.4, N)
    R = np.outer(rng.normal(0, 0.01, 600), b) + rng.normal(0, 1, (600, N)) * rng.uniform(0.008, 0.03, N)
    cov = np.cov(R, rowvar=False)
    return cov, np.sqrt(np.diag(cov)), N


def test_max_weight_is_actually_enforced():
    cov, sigma, N = _mkt(20)
    a = 0.05 * sigma * np.array([8., 6., 5.] + [-0.05] * (N - 3))
    eng = SizingEngine(SizingParams(max_weight=0.06, use_band=False))
    w, d = eng.size(alpha=a, sigma=sigma, cov=cov, current=np.zeros(N),
                    cost=np.full(N, 2e-4), alpha_se=np.abs(a) / 3)
    assert np.abs(w).max() <= 0.06 + 1e-9


def test_neutral_book_is_neutral_and_directional_is_not():
    cov, sigma, N = _mkt(60, seed=1)
    rng = np.random.default_rng(2)
    a = 0.05 * sigma * np.abs(rng.normal(0, 1, N))      # all-positive alpha
    for neutral, expect_net_zero in ((True, True), (False, False)):
        eng = SizingEngine(SizingParams(neutral=neutral, residual_factors=1 if neutral else 0,
                                        use_band=False))
        w, d = eng.size(alpha=a, sigma=sigma, cov=cov, current=np.zeros(N),
                        cost=np.full(N, 2e-4), alpha_se=np.abs(a) / 3)
        if expect_net_zero:
            assert abs(d.net) < 1e-8
        else:
            assert abs(d.net) > 1e-6


def test_zero_alpha_gives_zero_book():
    cov, sigma, N = _mkt(50, seed=3)
    a = np.zeros(N)
    eng = SizingEngine(SizingParams())
    w, d = eng.size(alpha=a, sigma=sigma, cov=cov, current=np.zeros(N),
                    cost=np.full(N, 2e-4), alpha_se=np.ones(N))
    assert d.gross == 0.0


def test_pure_noise_signal_is_not_relevered_by_vol_targeting():
    """The failure that motivated confidence_scales_risk: shrinkage said the
    signal was worthless and vol targeting divided the shrinkage back out."""
    cov, sigma, N = _mkt(60, seed=4)
    rng = np.random.default_rng(5)
    a = 0.05 * sigma * rng.normal(0, 1, N)
    se = np.abs(a) * 1e9                                  # t ~ 0
    eng = SizingEngine(SizingParams(use_band=False))
    w, d = eng.size(alpha=a, sigma=sigma, cov=cov, current=np.zeros(N),
                    cost=np.full(N, 2e-4), alpha_se=se)
    assert d.gross < 1e-6


def test_reported_vol_matches_the_book_actually_held():
    cov, sigma, N = _mkt(70, seed=6)
    rng = np.random.default_rng(7)
    a = 0.05 * sigma * rng.normal(0, 1, N)
    eng = SizingEngine(SizingParams())
    w, d = eng.size(alpha=a, sigma=sigma, cov=cov, current=np.zeros(N),
                    cost=np.full(N, 2e-4), alpha_se=np.abs(a) / 2)
    from qr.sizing import predicted_vol
    actual = predicted_vol(w, cov) * np.sqrt(252)
    assert abs(actual - d.predicted_vol_annual) < 1e-8


def test_band_scales_with_the_book():
    """The units bug: band was fixed while the book moved with target_vol."""
    cov, sigma, N = _mkt(80, seed=8)
    rng = np.random.default_rng(9)
    a = 0.05 * sigma * rng.normal(0, 1, N)
    turn = {}
    for tv in (0.05, 0.10, 0.20):
        eng = SizingEngine(SizingParams(target_vol=tv))
        w, d = eng.size(alpha=a, sigma=sigma, cov=cov, current=np.zeros(N),
                        cost=np.full(N, 5e-4), alpha_se=np.abs(a) / 2)
        turn[tv] = d.n_crossed_band
    # the fraction of names crossing should be stable, not 0 then all
    vals = list(turn.values())
    assert max(vals) - min(vals) < 0.5 * N


def test_nan_covariance_row_drops_the_name_not_the_scale():
    cov, sigma, N = _mkt(60, seed=10)
    rng = np.random.default_rng(11)
    a = 0.05 * sigma * rng.normal(0, 1, N)
    bad = cov.copy()
    bad[5, :] = np.nan
    bad[:, 5] = np.nan
    eng = SizingEngine(SizingParams(use_band=False))
    w_ok, d_ok = eng.size(alpha=a, sigma=sigma, cov=cov, current=np.zeros(N),
                          cost=np.full(N, 2e-4), alpha_se=np.abs(a) / 2)
    w_bad, d_bad = eng.size(alpha=a, sigma=sigma, cov=bad, current=np.zeros(N),
                            cost=np.full(N, 2e-4), alpha_se=np.abs(a) / 2)
    assert d_bad.dropped_no_cov == 1
    assert w_bad[5] == 0.0
    assert abs(d_bad.gross - d_ok.gross) / max(d_ok.gross, 1e-12) < 0.5


def test_rate_adjustment_moves_less_than_a_jump():
    cov, sigma, N = _mkt(60, seed=12)
    rng = np.random.default_rng(13)
    a = 0.05 * sigma * rng.normal(0, 1, N)
    kw = dict(alpha=a, sigma=sigma, cov=cov, current=np.zeros(N),
              cost=np.full(N, 1e-4), alpha_se=np.abs(a) / 2)
    _, d_rate = SizingEngine(SizingParams(adjustment="rate")).size(**kw)
    _, d_tgt = SizingEngine(SizingParams(adjustment="target")).size(**kw)
    assert d_rate.turnover < d_tgt.turnover


def test_apply_caps_satisfies_both_constraints():
    rng = np.random.default_rng(14)
    w = rng.normal(0, 0.1, 40)
    active = np.ones(40, dtype=bool)
    out = apply_caps(w, active, 0.05, neutral=True)
    assert np.abs(out).max() <= 0.05 + 1e-9
    assert abs(out.sum()) < 1e-6


def test_breadth_is_lower_on_a_correlated_universe():
    n = 40
    tight = np.full((n, n), 0.9)
    np.fill_diagonal(tight, 1.0)
    loose = np.eye(n)
    assert effective_n_participation(tight) < effective_n_participation(loose)
    assert abs(effective_n_participation(loose) - n) < 1e-6


# ------------------------------------------------------------ trial log

def test_trial_log_is_idempotent_on_identical_config(tmp_path):
    log = qr.TrialLog(str(tmp_path / "log.jsonl"))
    for _ in range(5):
        log.record("h", {"a": 1}, {"sharpe": 0.5}, family="f")
    assert log.count() == 1
    log.record("h", {"a": 2}, {"sharpe": 0.6}, family="f")
    assert log.count() == 2
    assert log.count(family="f") == 2
    assert log.count(family="other") == 0


# ------------------------------------------------------------ end to end

def test_evaluate_finds_a_planted_signal_and_rejects_noise():
    rng = np.random.default_rng(15)
    T, N = 1200, 150
    s = rng.normal(0, 1, (T, N))
    # s[t] must predict r[t+1], not r[t] -- the harness measures forward IC
    r = np.zeros((T, N))
    r[1:] = 0.06 * s[:-1] + np.sqrt(1 - 0.06 ** 2) * rng.normal(0, 1, (T - 1, N))
    r[0] = rng.normal(0, 1, N)
    good = qr.evaluate_signal(s, r, name="planted", horizon=1)
    assert good.ic_raw.t_stat > 4
    assert good.verdict in ("SURVIVES", "NOT DEFLATION-PROOF")

    noise = rng.normal(0, 1, (T, N))
    bad = qr.evaluate_signal(noise, r, name="noise", horizon=1)
    assert abs(bad.ic_raw.t_stat) < 3
    assert bad.verdict in ("DEAD", "NON-MONOTONE", "UNMEASURABLE")


def test_evaluate_warns_when_not_neutralised():
    rng = np.random.default_rng(16)
    s = rng.normal(0, 1, (300, 60))
    r = rng.normal(0, 1, (300, 60))
    rep = qr.evaluate_signal(s, r, name="x")
    assert any("NO FACTOR NEUTRALISATION" in n for n in rep.notes)
    assert any("NO TRIAL LOG" in n for n in rep.notes)


def test_evaluate_detects_a_signal_that_is_just_a_factor():
    rng = np.random.default_rng(17)
    T, N = 800, 100
    f = rng.normal(0, 1, (T, N))
    r = np.zeros((T, N))
    r[1:] = 0.06 * f[:-1] + np.sqrt(1 - 0.06 ** 2) * rng.normal(0, 1, (T - 1, N))
    r[0] = rng.normal(0, 1, N)
    disguised = 2.5 * f + rng.normal(0, 0.3, (T, N))
    rep = qr.evaluate_signal(disguised, r, name="disguised", factors=[f],
                             factor_names=["the factor"])
    assert any("neutralisation removed" in n for n in rep.notes)
