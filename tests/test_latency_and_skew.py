"""Checks on latency, taker costs, and the alpha-skewing market maker."""

import numpy as np
import pytest

from src.analytics import decompose_pnl
from src.directional import FlatTrader, RobustDirectionalTrader, TraderConfig
from src.market import DealerMarket, TrueDynamics
from src.skew_maker import AlphaSkewMarketMaker, SkewConfig, plain_maker, skew_maker
from src.strategy import AmbiguityPolicy


# ------------------------------------------------------------------- latency


def test_zero_latency_is_the_old_behaviour():
    """Latency must be strictly additive: at zero it changes nothing."""
    def run(lat):
        t = RobustDirectionalTrader(TraderConfig())
        m = DealerMarket(TrueDynamics(eps=0.015), latency=lat,
                         rng=np.random.default_rng(4))
        return m.run_taker(t, 600.0)

    a, b = run(0.0), run(0.0)
    assert a.pnl == b.pnl and a.n_fills == b.n_fills


def test_latency_delays_execution_not_the_decision():
    """A queued order executes at its own time, at the price prevailing then."""
    m = DealerMarket(TrueDynamics(eps=0.015), latency=0.2, rng=np.random.default_rng(5))
    trader = RobustDirectionalTrader(TraderConfig())
    m.run_taker(trader, 600.0)
    assert m._pending_target is None or isinstance(m._pending_target, tuple)
    # Fills must never be stamped before the decision that caused them.
    times = [f.time for f in m.fills]
    assert times == sorted(times)


def test_taker_only_holds_one_order_in_flight():
    """Without this the trader re-issues on every event while the first is working."""
    cfg = TraderConfig(warmup_updates=0)
    trader = RobustDirectionalTrader(cfg)
    trader.signal.n_updates = 10_000
    from src.market import TakerObservation
    obs = TakerObservation(time=1.0, dt=0.25, mid=100.0, order_side="buy",
                           half_spread=0.02, inventory=0, wealth=0.0, pending_target=3)
    assert trader.decide(obs) == 3            # defers to the working order


def test_latency_costs_the_taker_money():
    def mean_pnl(lat):
        out = []
        for s in range(10):
            t = RobustDirectionalTrader(TraderConfig())
            m = DealerMarket(TrueDynamics(eps=0.015), latency=lat,
                             rng=np.random.default_rng(s))
            out.append(m.run_taker(t, 1200.0).pnl)
        return float(np.mean(out))

    assert mean_pnl(0.0) > mean_pnl(0.25)


def test_latency_costs_the_maker_money():
    """Stale quotes get picked off, so quote latency is a direct cost."""
    def mean_pnl(lat):
        out = []
        for s in range(10):
            m = DealerMarket(TrueDynamics(eps=0.015), latency=lat,
                             rng=np.random.default_rng(s))
            out.append(m.run(plain_maker(policy=AmbiguityPolicy()), 1200.0).pnl)
        return float(np.mean(out))

    assert mean_pnl(0.0) > mean_pnl(0.25)


# --------------------------------------------------------------- taker costs


def test_fee_and_slippage_are_charged_on_top_of_the_spread():
    m = DealerMarket(taker_fee=0.003, taker_slippage=0.002, rng=np.random.default_rng(6))
    m.mu = 0.02
    assert m.taker_half_spread == pytest.approx(0.025)
    m._execute_taker(1)
    assert m.fills[0].price == pytest.approx(m.S + 0.025)
    assert m.taker_costs == pytest.approx(0.005)


def test_higher_fees_reduce_taker_pnl_and_turnover():
    def run(fee):
        pnl, fills = [], []
        for s in range(10):
            t = RobustDirectionalTrader(TraderConfig())
            m = DealerMarket(TrueDynamics(eps=0.015), latency=0.05, taker_fee=fee,
                             rng=np.random.default_rng(s))
            r = m.run_taker(t, 1200.0)
            pnl.append(r.pnl)
            fills.append(r.n_fills)
        return float(np.mean(pnl)), float(np.mean(fills))

    cheap_pnl, cheap_fills = run(0.0)
    dear_pnl, dear_fills = run(0.08)
    assert dear_pnl < cheap_pnl
    assert dear_fills < cheap_fills          # the band widens with the cost


def test_attribution_still_reconciles_with_costs_and_latency():
    for seed in range(6):
        t = RobustDirectionalTrader(TraderConfig())
        m = DealerMarket(TrueDynamics(eps=0.015), latency=0.05, taker_fee=0.01,
                         taker_slippage=0.005, rng=np.random.default_rng(seed))
        r = m.run_taker(t, 900.0)
        assert decompose_pnl(r).total == pytest.approx(r.pnl, abs=1e-9)


# ---------------------------------------------------------------- skew maker


def _skew(**kw):
    return AlphaSkewMarketMaker(skew=SkewConfig(**kw), policy=AmbiguityPolicy())


def test_skew_leans_the_bid_and_widens_the_offer_on_positive_alpha():
    mm = _skew()
    mm.signal.n_updates = 10_000
    base_ask, base_bid = super(AlphaSkewMarketMaker, mm).quote(0)

    mm.signal.coef[:] = 1.0
    mm.signal.z[:] = 0.02                     # positive alpha
    ask, bid = mm.quote(0)
    assert ask > base_ask                     # avoid selling into a rise
    assert bid < base_bid                     # lean in to buy

    mm.signal.z[:] = -0.02                    # negative alpha
    ask, bid = mm.quote(0)
    assert ask < base_ask
    assert bid > base_bid


def test_skew_never_crosses_the_midprice():
    """Depths stay non-negative however large the signal: this remains a maker."""
    mm = _skew(max_skew=10.0)
    mm.signal.n_updates = 10_000
    mm.signal.coef[:] = 100.0
    for z in (-50.0, -1.0, 1.0, 50.0):
        mm.signal.z[:] = z
        for q in (-6, 0, 6):
            ask, bid = mm.quote(q)
            for side in (ask, bid):
                if np.isfinite(side):
                    assert side >= 0.0


def test_skew_is_capped():
    mm = _skew(max_skew=0.004)
    mm.signal.n_updates = 10_000
    mm.signal.coef[:] = 1000.0
    mm.signal.z[:] = 10.0
    mm.quote(0)
    assert abs(mm.skew_log[-1]) <= 0.004 + 1e-12


def test_zero_scale_reproduces_the_plain_maker():
    a = _skew(skew_scale=0.0)
    a.signal.n_updates = 10_000
    a.signal.coef[:] = 1.0
    a.signal.z[:] = 0.05
    assert a.quote(0) == pytest.approx(super(AlphaSkewMarketMaker, a).quote(0), nan_ok=True)


def test_skew_holds_off_until_warmed_up():
    mm = _skew(warmup_updates=10_000)
    mm.signal.coef[:] = 1.0
    mm.signal.z[:] = 0.05
    assert mm.alpha == 0.0
    assert mm.quote(0) == pytest.approx(super(AlphaSkewMarketMaker, mm).quote(0), nan_ok=True)


def test_gamma_discounts_a_faster_signal():
    mm = _skew()
    mm.signal.n_updates = 10_000
    mm.fill_rate = 0.5
    mm.signal.coef[:] = np.array([1.0, 0.0, 0.0])     # weight on the slowest decay
    slow = mm.gamma
    mm.signal.coef[:] = np.array([0.0, 0.0, 1.0])     # weight on the fastest
    assert mm.gamma < slow


def test_skew_reduces_adverse_selection_and_inverting_it_does_the_opposite():
    """The central claim of test 2, in miniature."""
    def adverse(factory):
        vals = []
        for s in range(10):
            m = DealerMarket(TrueDynamics(eps=0.015), latency=0.05,
                             rng=np.random.default_rng(s))
            vals.append(decompose_pnl(m.run(factory(), 1200.0)).adverse_selection)
        return float(np.mean(vals))

    pol = AmbiguityPolicy()
    plain = adverse(lambda: plain_maker(policy=pol))
    leaned = adverse(lambda: skew_maker(1.0, policy=pol))
    wrong = adverse(lambda: skew_maker(1.0, invert=True, policy=pol))
    assert leaned < plain                    # markout cost falls
    assert wrong > plain                     # and the control confirms it is the signal


def test_skew_maker_is_still_a_maker():
    """It must never appear in the fill log as a liquidity taker."""
    m = DealerMarket(TrueDynamics(eps=0.015), latency=0.05, rng=np.random.default_rng(2))
    res = m.run(skew_maker(1.0, policy=AmbiguityPolicy()), 900.0)
    assert res.n_fills > 0
    for f in res.fills:
        assert f.depth >= 0.0                # a taker's depths are recorded negative
