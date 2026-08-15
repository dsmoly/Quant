"""Checks on the P&L attribution identity and the metric set."""

import numpy as np
import pytest

from src.analytics import Attribution, decompose_pnl, equity_grid, summarise
from src.market import DealerMarket, TrueDynamics
from src.strategy import fixed_robust_market_maker, neutral_market_maker


def _run(seed=0, horizon=900.0, eps=0.001, factory=None):
    factory = factory or (lambda: fixed_robust_market_maker(phi_alpha=6.0, phi=4.0))
    market = DealerMarket(TrueDynamics(eps=eps), rng=np.random.default_rng(seed))
    return market.run(factory(), horizon)


def test_attribution_reconciles_exactly():
    """The four components must sum to the reported P&L, on every path.

    This is the identity P&L = sum(delta) + sum(dq (S_T - S_i)), split at the
    markout horizon. decompose_pnl raises if it fails, so simply running it over
    many paths is the check.
    """
    for seed in range(20):
        res = _run(seed)
        a = decompose_pnl(res)
        assert a.total == pytest.approx(res.pnl, abs=1e-9)
        assert a.n_fills == res.n_fills


def test_attribution_reconciles_under_toxic_flow():
    for seed in range(10):
        res = _run(seed, eps=0.02)
        a = decompose_pnl(res)
        assert a.total == pytest.approx(res.pnl, abs=1e-9)


def test_attribution_is_independent_of_the_markout_horizon():
    """Splitting the position term at a different horizon moves value between the
    adverse-selection and inventory buckets but never changes the total."""
    res = _run(3)
    totals = [decompose_pnl(res, markout_horizon=h).total for h in (1.0, 5.0, 30.0, 120.0)]
    assert totals == pytest.approx([res.pnl] * 4, abs=1e-9)
    a1 = decompose_pnl(res, markout_horizon=1.0)
    a2 = decompose_pnl(res, markout_horizon=120.0)
    assert a1.spread_capture == pytest.approx(a2.spread_capture)
    assert a1.adverse_selection != pytest.approx(a2.adverse_selection)


def test_spread_capture_equals_the_sum_of_quoted_depths():
    res = _run(7)
    a = decompose_pnl(res)
    assert a.spread_capture == pytest.approx(sum(f.depth for f in res.fills))
    assert a.spread_capture > 0


def test_adverse_selection_is_positive_when_flow_is_toxic():
    """Toxic flow must show up as a cost, and a larger one than benign flow."""
    benign = np.mean([decompose_pnl(_run(s, eps=0.001)).adverse_selection for s in range(8)])
    toxic = np.mean([decompose_pnl(_run(s, eps=0.02)).adverse_selection for s in range(8)])
    assert benign > 0
    assert toxic > 3 * benign


def test_capture_ratio_falls_as_flow_gets_more_toxic():
    ratios = []
    for eps in (0.001, 0.006, 0.02):
        ratios.append(np.mean([decompose_pnl(_run(s, eps=eps)).capture_ratio
                               for s in range(8)]))
    assert ratios[0] > ratios[1] > ratios[2]
    assert ratios[0] > 0.85


def test_no_fills_gives_a_zero_attribution():
    class Silent:
        name = "silent"

        def quote(self, q):
            return np.inf, np.inf

        def observe(self, ev):
            pass

    market = DealerMarket(rng=np.random.default_rng(1))
    res = market.run(Silent(), 300.0)
    a = decompose_pnl(res)
    assert a.n_fills == 0
    assert a.spread_capture == 0.0
    assert a.total == pytest.approx(res.pnl)


def test_attribution_raises_when_the_identity_breaks():
    """Guard the guard: a fill log inconsistent with the reported P&L must be caught."""
    res = _run(11)
    res.fills[0].price += 1.0            # a traded price the cash ledger never saw
    with pytest.raises(AssertionError):
        decompose_pnl(res)

    res = _run(11)
    res.pnl += 0.5                       # reported P&L disagrees with the log
    with pytest.raises(AssertionError):
        decompose_pnl(res)


def test_depth_and_fill_midprice_move_together():
    """Perturbing a fill's depth alone cannot break the identity, by construction.

    The midprice at the fill is recovered as ``price - depth`` on an ask, so a
    change in depth shifts spread capture and the fill midprice by the same
    amount in opposite directions, and the total is unmoved. Worth pinning: it
    means the reconciliation check tests the *ledger*, not the depth bookkeeping,
    and a depth error would have to be caught elsewhere.
    """
    res = _run(11)
    before = decompose_pnl(res)
    res.fills[0].depth += 1.0
    after = decompose_pnl(res)
    assert after.total == pytest.approx(before.total)
    assert after.spread_capture == pytest.approx(before.spread_capture + 1.0)
    assert after.adverse_selection == pytest.approx(before.adverse_selection + 1.0)


def test_metrics_are_internally_consistent():
    results = [_run(s) for s in range(12)]
    m = summarise("arm", results, 900.0)
    pnls = np.array([r.pnl for r in results])
    assert m.n_paths == 12
    assert m.mean_pnl == pytest.approx(pnls.mean())
    assert m.median_pnl == pytest.approx(np.median(pnls))
    assert m.std_pnl == pytest.approx(pnls.std(ddof=1))
    assert m.sharpe_episode == pytest.approx(pnls.mean() / pnls.std(ddof=1))
    assert m.worst_path == pytest.approx(pnls.min())
    assert m.best_path == pytest.approx(pnls.max())
    assert m.var_95 <= m.median_pnl
    assert m.cvar_95 <= m.var_95
    assert 0.0 <= m.win_rate <= 1.0
    assert m.pnl_per_hour == pytest.approx(m.mean_pnl * 4.0)


def test_attribution_totals_match_the_reported_mean():
    """The metric-level attribution must reproduce the mean P&L."""
    results = [_run(s) for s in range(10)]
    m = summarise("arm", results, 900.0)
    rebuilt = m.spread_capture - m.adverse_selection + m.inventory_pnl - m.liquidation
    assert rebuilt == pytest.approx(m.mean_pnl, abs=1e-9)


def test_drawdown_is_non_negative_and_bounded():
    results = [_run(s) for s in range(8)]
    m = summarise("arm", results, 900.0)
    assert m.max_drawdown >= m.mean_max_drawdown >= 0
    assert m.max_dd_duration >= 0


def test_sharpe_daily_scaling():
    results = [_run(s) for s in range(8)]
    m = summarise("arm", results, 900.0, trading_day_seconds=6.5 * 3600)
    expected = m.sharpe_episode * np.sqrt(6.5 * 3600 / 900.0)
    assert m.sharpe_daily_equiv == pytest.approx(expected)


def test_equity_grid_shape_and_endpoints():
    results = [_run(s, horizon=600.0) for s in range(5)]
    grid = equity_grid(results, 600.0, n_points=50)
    assert grid.shape == (5, 50)
    assert np.all(grid[:, 0] == 0.0)          # every path starts flat
    for i, r in enumerate(results):
        assert grid[i, -1] == pytest.approx(r.pnl_path[-1], abs=1e-9)


def test_robust_arm_beats_neutral_on_risk():
    """The headline comparison should hold on a modest sample, not just the big run."""
    seeds = range(16)
    neutral = summarise("n", [_run(s, factory=neutral_market_maker) for s in seeds], 900.0)
    robust = summarise("r", [_run(s) for s in seeds], 900.0)
    assert robust.sharpe_episode > neutral.sharpe_episode
    assert robust.inv_rms < neutral.inv_rms


def test_attribution_dataclass_ratios():
    a = Attribution(spread_capture=10.0, adverse_selection=2.0, inventory_pnl=-0.5,
                    liquidation=0.1, total=7.4, n_fills=100, markout_horizon=5.0)
    assert a.gross_edge_per_fill == pytest.approx(0.1)
    assert a.adverse_per_fill == pytest.approx(0.02)
    assert a.capture_ratio == pytest.approx(0.8)
