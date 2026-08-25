"""Checks on the robust directional control, the alpha estimator and the taker."""

import numpy as np
import pytest

from src.directional import (
    AlphaEstimatorConfig,
    BuyAndHoldTrader,
    DirectionalParams,
    FlatTrader,
    OrderFlowAlpha,
    RobustDirectionalTrader,
    TraderConfig,
)
from src.market import DealerMarket, TakerObservation, TrueDynamics


# --------------------------------------------------------------- the control


def test_closed_form_solves_the_hjb_exactly():
    """The quadratic ansatz is a solution, not an approximation."""
    p = DirectionalParams(sigma=0.01, beta=1.0, k=0.02, phi_alpha=40.0, sigma_alpha=0.02)
    q = np.linspace(-20, 20, 41)
    a = np.linspace(-0.2, 0.2, 41)
    Q, A = np.meshgrid(q, a)
    assert np.max(np.abs(p.hjb_residual(Q, A))) < 1e-12


def test_closed_form_solves_the_hjb_across_parameters():
    for sigma in (0.005, 0.02):
        for k in (0.005, 0.05):
            for phi in (5.0, 200.0):
                for beta in (0.25, 4.0):
                    p = DirectionalParams(sigma=sigma, beta=beta, k=k, phi_alpha=phi,
                                          sigma_alpha=0.03)
                    r = p.hjb_residual(np.linspace(-10, 10, 21), np.linspace(-0.1, 0.1, 21))
                    assert np.max(np.abs(r)) < 1e-10, (sigma, k, phi, beta)


def test_value_function_is_concave_in_position():
    p = DirectionalParams()
    assert p.h2 < 0
    assert p.rho > 0
    assert p.h1 > 0


def test_target_is_linear_in_the_signal_and_odd():
    p = DirectionalParams(q_max=1000)
    assert float(p.unclipped_target(0.0)) == 0.0
    assert float(p.unclipped_target(0.04)) == pytest.approx(2 * float(p.unclipped_target(0.02)))
    assert float(p.unclipped_target(-0.02)) == pytest.approx(-float(p.unclipped_target(0.02)))


def test_position_size_falls_as_the_root_of_ambiguity_aversion():
    """q* scales as 1/sqrt(phi_alpha): phi_alpha is the sizing knob."""
    base = DirectionalParams(phi_alpha=40.0, q_max=10_000, beta=1.0)
    quad = DirectionalParams(phi_alpha=160.0, q_max=10_000, beta=1.0)
    ratio = float(base.unclipped_target(0.02)) / float(quad.unclipped_target(0.02))
    # Exactly 2 only if rho were fixed; rho also grows, so the ratio exceeds 2.
    assert ratio > 2.0
    assert float(quad.unclipped_target(0.02)) < float(base.unclipped_target(0.02))


def test_position_size_falls_with_volatility():
    """Volatility targeting falls out of the control rather than being bolted on."""
    calm = DirectionalParams(sigma=0.005, q_max=10_000)
    wild = DirectionalParams(sigma=0.05, q_max=10_000)
    assert float(wild.unclipped_target(0.02)) < float(calm.unclipped_target(0.02))


def test_faster_decaying_signal_is_worth_less():
    slow = DirectionalParams(beta=0.25, q_max=10_000)
    fast = DirectionalParams(beta=8.0, q_max=10_000)
    assert float(fast.unclipped_target(0.02)) < float(slow.unclipped_target(0.02))


def test_target_respects_the_position_limit():
    p = DirectionalParams(q_max=3)
    assert float(p.target_position(100.0)) == 3.0
    assert float(p.target_position(-100.0)) == -3.0


def test_trade_rate_mean_reverts_to_the_target():
    p = DirectionalParams(q_max=10_000)
    target = float(p.target_position(0.03))
    assert p.trade_rate(target, 0.03) == pytest.approx(0.0)
    assert p.trade_rate(target - 1.0, 0.03) > 0        # below target: buy
    assert p.trade_rate(target + 1.0, 0.03) < 0        # above target: sell


def test_marginal_value_vanishes_at_the_target_and_has_the_right_sign():
    p = DirectionalParams(q_max=10_000)
    t = float(p.unclipped_target(0.03))
    assert p.marginal_value(t, 0.03) == pytest.approx(0.0)
    assert p.marginal_value(t - 1.0, 0.03) > 0         # one more unit is worth having
    assert p.marginal_value(t + 1.0, 0.03) < 0


def test_no_trade_band_scales_with_the_spread():
    p = DirectionalParams()
    assert p.no_trade_band(0.0) == 0.0
    assert p.no_trade_band(0.04) == pytest.approx(2 * p.no_trade_band(0.02))
    assert p.no_trade_band(-1.0) == 0.0                # a negative cost is not a subsidy


def test_trade_threshold_is_the_marginal_value_equalling_the_cost():
    """At the band edge, one more unit is worth exactly the half-spread."""
    p = DirectionalParams(q_max=10_000)
    c = 0.02
    b = p.no_trade_band(c)
    t = float(p.unclipped_target(0.05))
    assert abs(p.marginal_value(t - b, 0.05)) == pytest.approx(c)


def test_signal_needed_to_trade_from_flat_is_independent_of_risk_appetite():
    """|alpha| > c (beta + rho) is the condition to trade at all.

    Risk appetite sets the size of the bet, but whether the trade clears the
    spread at all is a question about the signal and its decay.
    """
    c = 0.02
    for phi in (10.0, 40.0, 200.0):
        p = DirectionalParams(phi_alpha=phi, q_max=10_000)
        band = p.no_trade_band(c)
        # Smallest alpha whose target escapes the band.
        alpha_star = band * abs(p.h2) / p.h1
        assert alpha_star == pytest.approx(c * (p.beta + p.rho))


def test_worst_case_drift_leans_against_the_position():
    p = DirectionalParams(phi_alpha=40.0)
    assert p.worst_case_drift(0, 0.02) == pytest.approx(0.02)
    assert p.worst_case_drift(5, 0.02) < 0.02          # long: assume the drift is worse
    assert p.worst_case_drift(-5, 0.02) > 0.02


def test_rejects_invalid_parameters():
    for kw in (dict(sigma=0), dict(k=0), dict(beta=0), dict(phi_alpha=0), dict(q_max=0)):
        with pytest.raises(ValueError):
            DirectionalParams(**kw)


# ------------------------------------------------------------ alpha estimator


def test_alpha_features_decay_between_orders():
    est = OrderFlowAlpha(AlphaEstimatorConfig(decays=(1.0,)))
    est.observe_order(0.0, "buy")
    assert est.z[0] == pytest.approx(1.0)
    est.decay_to(1.0)
    assert est.z[0] == pytest.approx(np.exp(-1.0))


def test_alpha_features_signed_by_order_side():
    est = OrderFlowAlpha(AlphaEstimatorConfig(decays=(1.0,)))
    est.observe_order(0.0, "buy")
    est.observe_order(0.0, "sell")
    assert est.z[0] == pytest.approx(0.0)


def test_estimator_recovers_a_planted_linear_relationship():
    """Feed returns that really are c * z and check the fit finds c."""
    rng = np.random.default_rng(0)
    cfg = AlphaEstimatorConfig(decays=(1.0,), horizon=1.0, rls_forgetting=1.0)
    est = OrderFlowAlpha(cfg)
    true_c = 0.004
    t, mid = 0.0, 100.0
    for _ in range(4000):
        t += 0.25
        est.observe_order(t, "buy" if rng.random() < 0.5 else "sell")
        est.decay_to(t)
        mid += true_c * est.z[0] * 0.25          # the drift the signal claims
        est.observe_mid(t, mid)
    # The raw fit is attenuated by (1 - e^-bh)/(bh); alpha divides it back out.
    atten = (1.0 - np.exp(-1.0)) / 1.0
    assert est.coef[0] == pytest.approx(true_c * atten, rel=0.15)
    est.z[0] = 1.0
    assert est.alpha == pytest.approx(true_c, rel=0.15)


def test_estimator_stays_near_zero_on_unpredictable_returns():
    """Order flow with no relationship to returns must not produce a signal."""
    rng = np.random.default_rng(1)
    est = OrderFlowAlpha(AlphaEstimatorConfig(decays=(0.5, 2.0), horizon=1.0))
    t, mid = 0.0, 100.0
    for _ in range(4000):
        t += 0.25
        est.observe_order(t, "buy" if rng.random() < 0.5 else "sell")
        mid += 0.01 * rng.standard_normal()      # independent of the flow
        est.observe_mid(t, mid)
    assert abs(est.alpha) < 0.05


def test_estimator_output_is_clamped():
    cfg = AlphaEstimatorConfig(decays=(1.0,), max_alpha=0.1)
    est = OrderFlowAlpha(cfg)
    est.coef[0] = 10.0
    est.z[0] = 100.0
    assert est.alpha == pytest.approx(0.1)


def test_effective_decay_falls_back_before_any_fit():
    est = OrderFlowAlpha(AlphaEstimatorConfig(decays=(0.25, 1.0, 4.0)))
    assert est.effective_decay() == pytest.approx(1.0)


# ------------------------------------------------------------------- traders


def _obs(t=1.0, mid=100.0, side="buy", q=0, hs=0.02, dt=0.25):
    return TakerObservation(time=t, dt=dt, mid=mid, order_side=side, half_spread=hs,
                            inventory=q, wealth=0.0)


def test_flat_trader_never_trades_and_earns_exactly_zero():
    market = DealerMarket(rng=np.random.default_rng(0))
    res = market.run_taker(FlatTrader(), 900.0)
    assert res.n_fills == 0
    assert res.pnl == 0.0
    assert np.all(res.inventory_path == 0)


def test_buy_and_hold_takes_one_unit_and_keeps_it():
    market = DealerMarket(rng=np.random.default_rng(0))
    res = market.run_taker(BuyAndHoldTrader(), 900.0)
    assert res.n_fills == 1
    assert res.terminal_inventory == 1


def test_trader_holds_off_until_warmed_up():
    trader = RobustDirectionalTrader(TraderConfig(warmup_updates=10_000))
    for i in range(500):
        assert trader.decide(_obs(t=i * 0.25, q=0)) == 0


def test_trader_does_nothing_inside_the_band():
    """A position inside the band is left alone: closing it would not pay the spread."""
    trader = RobustDirectionalTrader(TraderConfig(warmup_updates=0))
    trader.signal.n_updates = 10_000
    trader.signal.coef[:] = 0.0                  # no signal at all, so the target is flat
    band = trader.params.no_trade_band(0.02)
    assert band > 1.0, "band too narrow for this check"
    assert trader.decide(_obs(q=1)) == 1


def test_trader_trims_a_position_that_sits_outside_the_band():
    """Outside the band it moves, but only to the band edge, not all the way."""
    trader = RobustDirectionalTrader(TraderConfig(warmup_updates=0))
    trader.signal.n_updates = 10_000
    trader.signal.coef[:] = 0.0
    band = trader.params.no_trade_band(0.02)
    far = int(band) + 3
    new_q = trader.decide(_obs(q=far))
    assert new_q < far                            # it trims
    assert new_q == pytest.approx(round(band), abs=1)   # to the edge, not to flat


def test_trader_rejects_an_unknown_signal_mode():
    with pytest.raises(ValueError):
        RobustDirectionalTrader(TraderConfig(signal_mode="nonsense"))


def test_inverted_mode_mirrors_the_live_signal():
    live = RobustDirectionalTrader(TraderConfig(signal_mode="live", warmup_updates=0))
    inv = RobustDirectionalTrader(TraderConfig(signal_mode="inverted", warmup_updates=0))
    for t in (live, inv):
        t.signal.n_updates = 10_000
        t.signal.coef[:] = 0.01
        t.signal.z[:] = 5.0
    assert live._apply_signal_mode(0.05) == pytest.approx(0.05)
    assert inv._apply_signal_mode(0.05) == pytest.approx(-0.05)


def test_taker_pays_the_spread_on_every_unit():
    """Each unit crosses at mid +/- the quoted half-spread; nothing trades at mid."""
    market = DealerMarket(rng=np.random.default_rng(3))
    market.mu = 0.02
    market._execute_taker(2)
    assert market.q == 2
    assert len(market.fills) == 2
    for f in market.fills:
        assert f.depth == pytest.approx(-0.02)          # negative depth: a cost
        assert f.price == pytest.approx(market.S + 0.02, abs=1e-9)


def test_taker_orders_impact_the_price_but_do_not_excite_arrivals():
    """The taker pays for its footprint without double-counting client flow.

    The Hawkes process is calibrated to client arrivals at a branching ratio of
    0.9; adding the taker's own orders to the self-excitation drives it above 1
    and the arrival process becomes explosive.
    """
    market = DealerMarket(rng=np.random.default_rng(4))
    lam0, kap0, alpha0 = market.lam.copy(), market.kap.copy(), market.alpha
    market._execute_taker(1)                            # one buy
    assert market.alpha > alpha0                        # price impact applies
    assert np.all(market.kap > kap0)                    # the book thins
    assert np.allclose(market.lam, lam0)                # arrivals are untouched


def test_client_orders_still_excite_arrivals():
    market = DealerMarket(rng=np.random.default_rng(5))
    lam0 = market.lam.copy()
    market._apply_market_order(0)
    assert np.all(market.lam > lam0)


def test_taker_position_respects_the_limit():
    market = DealerMarket(q_max=3, rng=np.random.default_rng(6))
    market._execute_taker(99)
    assert market.q == 3
    market._execute_taker(-99)
    assert market.q == -3


def test_directional_edge_grows_with_the_size_of_the_signal():
    """The whole thesis in one assertion: the drift on offer has to beat the spread.

    At the papers' own eps the signal is a small fraction of the half-spread and
    the strategy is barely viable; scale the market-order impact up and the same
    strategy, unchanged, becomes profitable.
    """
    def mean_pnl(eps):
        out = []
        for seed in range(8):
            trader = RobustDirectionalTrader(TraderConfig())
            market = DealerMarket(TrueDynamics(eps=eps), rng=np.random.default_rng(seed))
            out.append(market.run_taker(trader, 1200.0).pnl)
        return float(np.mean(out))

    small, large = mean_pnl(0.001), mean_pnl(0.025)
    assert large > small
    assert large > 1.0
    assert abs(small) < 0.5 * large


def test_directional_arm_trades_and_profits_when_the_signal_is_large():
    """Under toxic flow the same strategy trades, and the sign of the P&L is real:
    inverting the signal turns the profit into a loss."""
    live_pnl, inv_pnl, trades = [], [], 0
    for seed in range(8):
        live = RobustDirectionalTrader(TraderConfig(signal_mode="live"))
        m1 = DealerMarket(TrueDynamics(eps=0.025), rng=np.random.default_rng(seed))
        live_pnl.append(m1.run_taker(live, 1200.0).pnl)
        trades += live.n_trades

        inv = RobustDirectionalTrader(TraderConfig(signal_mode="inverted"))
        m2 = DealerMarket(TrueDynamics(eps=0.025), rng=np.random.default_rng(seed))
        inv_pnl.append(m2.run_taker(inv, 1200.0).pnl)

    assert trades > 0
    assert float(np.mean(live_pnl)) > 0
    assert float(np.mean(inv_pnl)) < 0
    # The inverted arm loses more than the live arm makes: both pay the spread.
    assert abs(float(np.mean(inv_pnl))) > float(np.mean(live_pnl))
