"""Sizing engine: closed-form identities and portfolio invariants."""

import numpy as np
import pytest

from src.sizing import (SizingEngine, SizingParams, apply_band, neutralise,
                        predicted_vol, shrink_by_tstat)


def _cov(n, rho=0.3, sd=0.02, seed=0):
    c = np.full((n, n), rho) + np.eye(n) * (1 - rho)
    return c * sd ** 2


# ------------------------------------------------------------- the closed form


def test_target_position_scales_as_one_over_sigma():
    p = SizingParams()
    a = 0.01
    q1 = p.target_position(a, 0.01)
    q2 = p.target_position(a, 0.02)
    # q* ~ 1/((beta + rho(sigma)) * sigma); at small rho this approaches a
    # factor of two, and rho makes the high-vol name smaller still.
    assert q2 < q1
    assert q1 / q2 > 2.0


def test_target_position_is_linear_in_alpha():
    p = SizingParams()
    assert p.target_position(0.02, 0.015) == pytest.approx(
        2 * p.target_position(0.01, 0.015))


def test_higher_phi_means_smaller_positions():
    small = SizingParams(phi=5.0).target_position(0.01, 0.02)
    large = SizingParams(phi=100.0).target_position(0.01, 0.02)
    assert abs(large) < abs(small)


def test_phi_scaling_is_one_over_sqrt_phi_only_when_rho_is_small():
    """q* ~ 1/(sqrt(phi) (beta + rho)) and rho itself carries sqrt(phi).

    So the textbook 1/sqrt(phi) law is an asymptotic statement, valid when the
    gap-closing rate is small against the signal decay. Away from that limit the
    dependence is stronger, and pinning the exact form here stops the two from
    being confused later.
    """
    sigma, alpha = 0.02, 0.01

    def exact(phi):
        p = SizingParams(phi=phi)
        return alpha / ((p.beta + p.rho(sigma)) * p.abs_h2(sigma))

    for phi in (5.0, 100.0):
        assert SizingParams(phi=phi).target_position(alpha, sigma) == pytest.approx(exact(phi))

    # rho << beta: recover 1/sqrt(phi) to within a percent.
    tiny = SizingParams(phi=1e-4, beta=5.0)
    tiny4 = SizingParams(phi=4e-4, beta=5.0)
    ratio = abs(tiny.target_position(alpha, sigma)) / abs(tiny4.target_position(alpha, sigma))
    assert ratio == pytest.approx(2.0, rel=0.01)


def test_faster_decay_discounts_the_target():
    slow = SizingParams(beta=0.05).target_position(0.01, 0.02)
    fast = SizingParams(beta=2.0).target_position(0.01, 0.02)
    assert abs(fast) < abs(slow)


def test_band_is_proportional_to_cost_and_inverse_to_h2():
    p = SizingParams()
    assert p.no_trade_band(0.002, 0.02) == pytest.approx(2 * p.no_trade_band(0.001, 0.02))
    assert p.no_trade_band(0.0, 0.02) == 0.0


def test_bad_parameters_are_rejected():
    for kw in (dict(phi=0.0), dict(k=-1.0), dict(beta=0.0), dict(target_vol=0.0)):
        with pytest.raises(ValueError):
            SizingParams(**kw)


# ------------------------------------------------------------------ shrinkage


def test_shrinkage_matches_t2_over_1_plus_t2():
    a = np.array([0.02])
    se = np.array([0.02])           # t = 1 -> keep a half
    assert shrink_by_tstat(a, se)[0] == pytest.approx(0.01)
    assert shrink_by_tstat(a, np.array([0.02 / 3]))[0] == pytest.approx(0.02 * 9 / 10)


def test_shrinkage_kills_an_unmeasured_signal():
    assert shrink_by_tstat(np.array([0.05]), np.array([np.inf]))[0] == 0.0
    assert shrink_by_tstat(np.array([0.05]), np.array([1e9]))[0] == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------ portfolio operations


def test_neutralise_zeroes_the_active_sum_only():
    w = np.array([1.0, 2.0, 3.0, 99.0])
    active = np.array([True, True, True, False])
    out = neutralise(w, active)
    assert out[active].sum() == pytest.approx(0.0)
    assert out[3] == 0.0


def test_band_holds_inside_and_moves_to_the_edge_outside():
    cur = np.array([0.0, 0.0, 0.0])
    tgt = np.array([0.05, 0.2, -0.2])
    band = np.array([0.1, 0.1, 0.1])
    out = apply_band(cur, tgt, band, to_edge=True)
    assert out[0] == 0.0                       # inside the band: no trade
    assert out[1] == pytest.approx(0.1)        # moved to the near edge, not to 0.2
    assert out[2] == pytest.approx(-0.1)


def test_band_to_target_mode_ignores_the_edge():
    out = apply_band(np.zeros(2), np.array([0.05, 0.3]), np.array([0.1, 0.1]),
                     to_edge=False)
    assert out[0] == 0.0 and out[1] == pytest.approx(0.3)


def test_predicted_vol_matches_the_quadratic_form():
    cov = _cov(3)
    w = np.array([0.3, -0.2, 0.1])
    assert predicted_vol(w, cov) == pytest.approx(np.sqrt(w @ cov @ w))


# --------------------------------------------------------------- the engine


def _size(engine, alpha, se=None, current=None, n=6, cost=0.002):
    sigma = np.full(n, 0.02)
    cov = _cov(n)
    return engine.size(alpha=alpha, sigma=sigma, cov=cov,
                       current=np.zeros(n) if current is None else current,
                       cost=cost, alpha_se=se)


def test_engine_output_is_dollar_neutral():
    e = SizingEngine()
    alpha = np.array([0.02, 0.01, 0.0, -0.01, -0.02, 0.005])
    w, d = _size(e, alpha, se=np.full(6, 0.002))
    assert w.sum() == pytest.approx(0.0, abs=1e-12)
    assert d.net == pytest.approx(0.0, abs=1e-12)


def test_engine_respects_the_gross_cap():
    e = SizingEngine(SizingParams(max_gross=0.5, target_vol=5.0))
    w, d = _size(e, np.linspace(-0.1, 0.1, 6), se=np.full(6, 1e-6))
    assert d.gross <= 0.5 + 1e-9


def test_engine_respects_the_per_name_cap():
    e = SizingEngine(SizingParams(max_weight=0.02, target_vol=5.0, max_gross=99.0))
    w, _ = _size(e, np.linspace(-0.1, 0.1, 6), se=np.full(6, 1e-6))
    assert np.abs(w).max() <= 0.02 + 1e-9


def test_an_unmeasured_signal_is_sized_at_essentially_zero():
    """The property the dumb-signal experiment depends on.

    Shrinkage alone is not enough -- volatility targeting is scale-invariant and
    would divide it straight back out -- so confidence also scales the risk
    budget. This test pins that behaviour.
    """
    e = SizingEngine()
    alpha = np.array([0.02, 0.01, 0.0, -0.01, -0.02, 0.005])
    confident, _ = _size(e, alpha, se=np.full(6, 1e-5))
    unmeasured, _ = _size(e, alpha, se=np.full(6, 1e3))
    assert np.abs(unmeasured).sum() < 0.01 * np.abs(confident).sum()


def test_inactive_names_are_closed_regardless_of_the_band():
    e = SizingEngine()
    n = 6
    active = np.array([True] * 5 + [False])
    w, _ = e.size(alpha=np.linspace(-0.02, 0.02, n), sigma=np.full(n, 0.02),
                  cov=_cov(n), current=np.full(n, 0.05), cost=0.002,
                  alpha_se=np.full(n, 0.001), active=active)
    assert w[-1] == 0.0


def test_no_covariance_means_no_position():
    e = SizingEngine()
    w, d = e.size(alpha=np.ones(4) * 0.01, sigma=np.full(4, 0.02), cov=None,
                  current=np.zeros(4), cost=0.001)
    assert np.abs(w).sum() == pytest.approx(0.0)


def test_nan_covariance_is_not_traded_on():
    e = SizingEngine()
    cov = _cov(4)
    cov[0, 1] = np.nan
    w, _ = e.size(alpha=np.ones(4) * 0.01, sigma=np.full(4, 0.02), cov=cov,
                  current=np.zeros(4), cost=0.001)
    assert np.isfinite(w).all()


def test_band_reduces_turnover_against_an_unchanged_target():
    banded = SizingEngine(SizingParams(use_band=True))
    naked = SizingEngine(SizingParams(use_band=False))
    alpha = np.array([0.02, 0.01, 0.0, -0.01, -0.02, 0.005])
    se = np.full(6, 0.002)
    cur = np.zeros(6)
    w_b, d_b = _size(banded, alpha, se=se, current=cur)
    w_n, d_n = _size(naked, alpha, se=se, current=cur)
    # From flat both must trade; the banded one moves less far.
    assert d_b.turnover <= d_n.turnover + 1e-12


def test_vol_scaling_switch_removes_cross_sectional_vol_dependence():
    on = SizingEngine(SizingParams(vol_scaling=True))
    off = SizingEngine(SizingParams(vol_scaling=False))
    n = 4
    sigma = np.array([0.01, 0.02, 0.03, 0.04])
    alpha = np.array([0.01, 0.01, 0.01, 0.01])
    cov = np.diag(sigma ** 2)
    kw = dict(alpha=alpha, cov=cov, current=np.zeros(n), cost=0.0,
              alpha_se=np.full(n, 1e-6))
    w_on, _ = on.size(sigma=sigma, **kw)
    w_off, _ = off.size(sigma=sigma, **kw)
    # With vol scaling off, equal alphas give equal (pre-neutralisation) targets.
    assert np.std(np.abs(w_off)) < np.std(np.abs(w_on))
