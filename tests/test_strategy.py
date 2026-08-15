"""Checks on the online estimators and the ambiguity policy."""

import numpy as np
import pytest

from src.market import DealerMarket, MarketEvent
from src.strategy import (
    AdaptiveRAMM,
    AmbiguityPolicy,
    EstimatorConfig,
    MarketMaker,
    OnlineReferenceModel,
    fixed_robust_market_maker,
    neutral_market_maker,
)


def _estimator(**kw) -> OnlineReferenceModel:
    kw.setdefault("kappa0", 27.0)
    kw.setdefault("lam0", 2.0)
    kw.setdefault("sigma0", 0.01)
    kw.setdefault("mu0", 0.02)
    kw.setdefault("q_max", 8)
    return OnlineReferenceModel(EstimatorConfig(), **kw)


def test_kappa_estimator_recovers_a_known_fill_decay():
    """Feed Bernoulli fills from a true kappa and check the recursive MLE finds it."""
    rng = np.random.default_rng(0)
    true_kappa = 45.0
    est = _estimator(kappa0=20.0)
    for _ in range(20_000):
        depth = rng.uniform(0.005, 0.06)
        filled = rng.random() < np.exp(-true_kappa * depth)
        est.observe_rfq(depth, filled, 0.25)
    assert est.kappa == pytest.approx(true_kappa, rel=0.12)


def test_kappa_estimator_stays_within_bounds():
    """Degenerate evidence must not drive the estimate out of its admissible range.

    Always-filled evidence pushes kappa to the floor.  Never-filled evidence
    pushes it up, but the score d e^(-kd) / (1 - e^(-kd)) vanishes as kappa grows,
    so it asymptotes rather than pinning the ceiling -- the estimator saturates
    instead of running away, which is the property that matters.
    """
    lo, hi = EstimatorConfig().kappa_bounds
    est = _estimator(kappa0=27.0)
    for _ in range(5_000):
        est.observe_rfq(0.05, True, 0.25)          # always filled -> pushes kappa down
    assert est.kappa == pytest.approx(lo)

    est = _estimator(kappa0=27.0)
    for _ in range(5_000):
        est.observe_rfq(0.05, False, 0.25)         # never filled -> pushes kappa up
    assert 27.0 < est.kappa <= hi


def test_sigma_estimator_recovers_a_known_volatility():
    rng = np.random.default_rng(1)
    true_sigma, dt = 0.04, 0.5
    est = _estimator(sigma0=0.01)
    mid = 100.0
    for _ in range(20_000):
        mid += true_sigma * np.sqrt(dt) * rng.standard_normal()
        est.observe_mid(mid, dt)
    assert est.sigma == pytest.approx(true_sigma, rel=0.25)


def test_mu_estimator_tracks_the_cover_level():
    est = _estimator(mu0=0.02)
    for _ in range(600):
        est.observe_cover(0.031)
    assert est.mu == pytest.approx(0.031, rel=1e-3)


def test_mu_estimator_ignores_missing_cover():
    est = _estimator(mu0=0.02)
    before = est.mu
    est.observe_cover(np.nan)
    est.observe_cover(-1.0)
    assert est.mu == before


def test_tail_weight_responds_to_extreme_inventory():
    est = _estimator(q_max=8)
    for _ in range(3_000):
        est.observe_inventory(0)
    assert est.tail == pytest.approx(0.0, abs=1e-3)
    for _ in range(3_000):
        est.observe_inventory(7)
    assert est.tail == pytest.approx(1.0, rel=1e-2)


def test_policy_is_aggressive_when_the_market_is_supra_competitive():
    pol = AmbiguityPolicy(nash_depth=0.02)
    _, phi_wide = pol(mu=0.02 * 1.30, sigma=0.01, tail=0.0)
    _, phi_at_nash = pol(mu=0.02, sigma=0.01, tail=0.0)
    assert phi_wide == pytest.approx(pol.phi_max)
    assert phi_at_nash == 0.0


def test_policy_deadband_suppresses_small_gaps():
    pol = AmbiguityPolicy(nash_depth=0.02, gap_deadband=0.05)
    _, phi = pol(mu=0.02 * 1.04, sigma=0.01, tail=0.0)
    assert phi == 0.0


def test_policy_is_defensive_on_volatility_and_inventory_tails():
    pol = AmbiguityPolicy(nash_depth=0.02)
    calm, _ = pol(mu=0.02, sigma=0.01, tail=0.0)
    volatile, _ = pol(mu=0.02, sigma=0.03, tail=0.0)
    heavy_tail, _ = pol(mu=0.02, sigma=0.01, tail=0.5)
    assert volatile > calm
    assert heavy_tail > calm


def test_policy_is_defensive_when_the_market_quotes_inside_nash():
    pol = AmbiguityPolicy(nash_depth=0.02)
    at_nash, _ = pol(mu=0.02, sigma=0.01, tail=0.0)
    inside, phi = pol(mu=0.014, sigma=0.01, tail=0.0)
    assert inside > at_nash
    assert phi == 0.0


def test_policy_output_is_clipped():
    pol = AmbiguityPolicy(nash_depth=0.02, phi_alpha_max=9.0, phi_max=5.0)
    phi_alpha, phi = pol(mu=0.2, sigma=10.0, tail=1.0)
    assert 0.0 <= phi_alpha <= 9.0
    assert 0.0 <= phi <= 5.0


def test_gap_metric_signs():
    pol = AmbiguityPolicy(nash_depth=0.02)
    assert pol.gap(0.024) == pytest.approx(0.2)
    assert pol.gap(0.02) == 0.0
    assert pol.gap(0.016) < 0


def _event(depth=0.03, filled=False, q=0, mid=100.0, t=1.0):
    return MarketEvent(time=t, dt=0.25, mid=mid, side="ask", quoted_depth=depth,
                       filled=filled, cover=np.nan if filled else 0.02,
                       inventory=q, wealth=0.0)


def test_market_maker_resolves_on_schedule_only():
    mm = MarketMaker(resolve_every=10, phi_alpha=1.0, adapt_reference=False)
    before = mm.quoter.curves()[0].copy()
    for i in range(9):
        mm.observe(_event(t=i * 0.25))
    assert mm.quoter.curves()[0] == pytest.approx(before, nan_ok=True)
    mm.params.phi_alpha = 12.0                      # takes effect at the next resolve
    mm.observe(_event(t=10.0))
    assert not np.allclose(mm.quoter.curves()[0], before, equal_nan=True)


def test_adaptive_strategy_logs_and_moves_its_ambiguity_vector():
    pol = AmbiguityPolicy(nash_depth=0.02)
    mm = AdaptiveRAMM(policy=pol, mu0=0.02, resolve_every=5, adapt_reference=False)
    for i in range(20):
        ev = _event(filled=False, t=i * 0.25)
        ev.cover = 0.02
        mm.observe(ev)
    tight_phi = mm.phi_log[-1]
    for i in range(400):
        ev = _event(filled=False, t=100 + i * 0.25)
        ev.cover = 0.02 * 1.4                       # the crowd widens out
        mm.observe(ev)
    assert mm.phi_log[-1] > tight_phi
    assert len(mm.phi_log) == len(mm.phi_alpha_log)


def test_neutral_arm_has_no_ambiguity():
    mm = neutral_market_maker()
    assert mm.params.phi_alpha == 0.0 and mm.params.phi == 0.0
    assert mm.name == "neutral"


def test_fixed_robust_arm_keeps_its_vector_constant():
    mm = fixed_robust_market_maker(phi_alpha=3.0, phi=7.0, adapt_reference=False)
    for i in range(500):
        mm.observe(_event(t=i * 0.25))
    assert mm.params.phi_alpha == 3.0 and mm.params.phi == 7.0


def test_strategies_run_end_to_end_and_stay_within_limits():
    for factory in (neutral_market_maker, fixed_robust_market_maker,
                    lambda: AdaptiveRAMM(policy=AmbiguityPolicy())):
        mm = factory()
        market = DealerMarket(q_max=8, rng=np.random.default_rng(17))
        res = market.run(mm, horizon=600.0)
        assert res.n_fills > 0
        assert np.all(np.abs(res.inventory_path) <= 8)
        assert np.isfinite(res.pnl)


def test_quotes_are_finite_and_non_negative_across_inventories():
    mm = AdaptiveRAMM(policy=AmbiguityPolicy(), q_max=6)
    for q in range(-6, 7):
        ask, bid = mm.quote(q)
        for side, val in (("ask", ask), ("bid", bid)):
            if np.isfinite(val):
                assert val >= 0.0, (q, side, val)
        assert np.isfinite(ask) or np.isfinite(bid)
