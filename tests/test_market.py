"""Checks on the simulated market: calibration identities and mechanics."""

import numpy as np
import pytest

from src.market import CompetitionParams, DealerMarket, TrueDynamics


class ConstantQuoter:
    """A market maker that always shows the same depths."""

    name = "constant"

    def __init__(self, ask: float = 0.03, bid: float = 0.03):
        self.ask, self.bid = ask, bid
        self.events = []

    def quote(self, q):
        return self.ask, self.bid

    def observe(self, ev):
        self.events.append(ev)


def test_stationary_moments_match_the_papers_reference_model():
    """Table 1 is calibrated so the long-run means are lambda = 2 and kappa = 27."""
    d = TrueDynamics()
    assert d.stationary_lam() == pytest.approx(2.0)
    assert d.stationary_kap() == pytest.approx(27.0)


def test_explosive_hawkes_is_rejected():
    with pytest.raises(ValueError):
        TrueDynamics(eta_lam=6.0, nu_lam=3.0).stationary_lam()


def test_simulated_arrival_rate_matches_the_stationary_intensity():
    """The thinned event stream must reproduce the Hawkes stationary rate.

    Averaged over seeds: the branching ratio (eta + nu) / beta is 0.9 at the
    paper's parameters, which inflates the variance of the event count by roughly
    1 / (1 - 0.9)^2 = 100, so a single path is far too noisy to test against.
    """
    rates = []
    for seed in range(12):
        m = DealerMarket(rng=np.random.default_rng(100 + seed))
        res = m.run(ConstantQuoter(ask=np.inf, bid=np.inf), horizon=4000.0)
        rates.append(res.n_rfq / 4000.0)
    assert float(np.mean(rates)) == pytest.approx(2.0 * TrueDynamics().stationary_lam(), rel=0.05)


def test_simulated_kappa_mean_matches_the_reference_value():
    m = DealerMarket(rng=np.random.default_rng(3))
    seen = []
    for _ in range(4000):
        m._advance(m.rng.exponential(1.0 / m._intensity_bound()))
        if m.rng.random() < 0.5:
            m._apply_market_order(0 if m.rng.random() < 0.5 else 1)
        seen.append(m.kap.mean())
    # Long-run mean is 27 when every candidate is an order; thinning lowers the
    # realised excitation, so only assert it stays in a sane band around it.
    assert 15.0 < float(np.mean(seen)) < 30.0


def test_win_probability_is_bounded_by_the_monopolistic_rate():
    """Assumption 2.2(3) of the mean-field paper."""
    c = CompetitionParams()
    d = np.linspace(0.0, 0.2, 50)
    for mu in (0.005, 0.02, 0.05):
        assert np.all(c.win_probability(d, mu, 27.0) <= np.exp(-27.0 * d) / c.C + 1e-12)


def test_win_probability_monotonicity():
    c = CompetitionParams()
    d = np.linspace(0.0, 0.15, 40)
    p = c.win_probability(d, 0.02, 27.0)
    assert np.all(np.diff(p) < 0)                                   # wider quote, fewer wins
    assert np.all(c.win_probability(d, 0.04, 27.0) > p)             # wider crowd, more wins


def test_no_quote_means_no_fill():
    m = DealerMarket(rng=np.random.default_rng(5))
    res = m.run(ConstantQuoter(ask=np.inf, bid=np.inf), horizon=600.0)
    assert res.n_rfq > 100
    assert res.n_fills == 0
    assert res.pnl == pytest.approx(0.0)
    assert res.terminal_inventory == 0


def test_inventory_never_breaches_the_limit():
    m = DealerMarket(q_max=3, rng=np.random.default_rng(9))
    res = m.run(ConstantQuoter(ask=0.0, bid=0.0), horizon=900.0)   # maximally aggressive
    assert res.n_fills > 50
    assert np.all(np.abs(res.inventory_path) <= 3)


def test_cash_and_inventory_reconcile_with_the_fill_log():
    m = DealerMarket(rng=np.random.default_rng(21))
    m.run(ConstantQuoter(0.02, 0.02), horizon=900.0)
    cash = sum(f.price if f.side == "ask" else -f.price for f in m.fills)
    inv = sum(-1 if f.side == "ask" else 1 for f in m.fills)
    assert m.cash == pytest.approx(cash)
    assert m.q == inv


def test_market_orders_move_the_price_against_a_filled_quote():
    """Adverse selection: buy MOs push alpha up, and a buy MO is what lifts our ask."""
    m = DealerMarket(rng=np.random.default_rng(1))
    m._apply_market_order(0)
    assert m.alpha > 0
    m.reset()
    m._apply_market_order(1)
    assert m.alpha < 0


def test_market_orders_excite_arrivals_and_thin_the_book():
    m = DealerMarket(rng=np.random.default_rng(2))
    lam0, kap0 = m.lam.copy(), m.kap.copy()
    m._apply_market_order(0)
    assert m.lam[0] > lam0[0] and m.lam[1] > lam0[1]     # self- and cross-excitation
    assert m.kap[0] > kap0[0] and m.kap[1] > kap0[1]     # book gets thinner


def test_cover_is_reported_only_on_lost_trades():
    m = DealerMarket(rng=np.random.default_rng(13))
    mm = ConstantQuoter(0.03, 0.03)
    m.run(mm, horizon=600.0)
    assert any(e.filled for e in mm.events) and any(not e.filled for e in mm.events)
    for e in mm.events:
        assert np.isnan(e.cover) if e.filled else e.cover > 0


def test_regime_switching_is_reachable_and_moves_the_population_quote():
    comp = CompetitionParams(mean_comp_seconds=30.0, mean_supra_seconds=30.0,
                             mu_reversion=1.0, supra_multiple=1.5)
    m = DealerMarket(competition=comp, rng=np.random.default_rng(4))
    res = m.run(ConstantQuoter(0.03, 0.03), horizon=3000.0)
    assert 0.2 < res.supra_fraction < 0.8
    assert res.mu_path.max() > comp.nash_depth * 1.2


def test_frozen_regime_stays_frozen():
    comp = CompetitionParams(mean_comp_seconds=1e12, mean_supra_seconds=1e12,
                             supra_multiple=1.0)
    m = DealerMarket(competition=comp, rng=np.random.default_rng(6))
    res = m.run(ConstantQuoter(0.03, 0.03), horizon=2000.0)
    assert res.supra_fraction == 0.0


def test_tighter_quotes_win_more():
    def fills(depth, seed):
        m = DealerMarket(rng=np.random.default_rng(seed))
        return m.run(ConstantQuoter(depth, depth), horizon=1200.0).n_fills

    for seed in (1, 2, 3):
        assert fills(0.01, seed) > fills(0.05, seed)


def test_liquidation_penalty_is_charged_on_terminal_inventory():
    m = DealerMarket(q_max=3, theta=0.1, rng=np.random.default_rng(31))
    res = m.run(ConstantQuoter(0.0, np.inf), horizon=600.0)   # sell only: end up short
    assert res.terminal_inventory < 0
    assert res.pnl == pytest.approx(m.wealth - 0.1 * res.terminal_inventory**2)


def test_run_is_reproducible_under_a_fixed_seed():
    def run():
        m = DealerMarket(rng=np.random.default_rng(99))
        return m.run(ConstantQuoter(0.025, 0.025), horizon=600.0)

    a, b = run(), run()
    assert a.pnl == b.pnl and a.n_fills == b.n_fills
    assert np.array_equal(a.inventory_path, b.inventory_path)
