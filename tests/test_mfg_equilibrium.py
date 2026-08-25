"""Checks against the analytical structure of the mean-field dealer-market paper."""

import numpy as np
import pytest

from src.mfg_equilibrium import (
    IntensityParams,
    MFGConfig,
    best_response_quote,
    monopolistic_quotes,
    solve_mfg,
)


def _fast_config(**kw) -> MFGConfig:
    """Small grid and looser tolerance so the tests stay quick."""
    kw.setdefault("Z", 5)
    kw.setdefault("dt", 0.1)
    kw.setdefault("tol", 1e-7)
    return MFGConfig(**kw)


def test_intensity_satisfies_assumption_2_2():
    p = IntensityParams()
    d = np.linspace(-0.5, 4.0, 60)
    # (1) decreasing in own quote, increasing in the population quote
    assert np.all(np.diff(p.f(d, 0.5)) < 0)
    assert np.all(p.f(d, 1.0) > p.f(d, 0.5))
    # (2) vanishes as the quote widens; (3) dominated by the monopolistic rate
    assert p.f(50.0, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert np.all(p.f(d, 3.0) <= p.monopolistic(d) + 1e-12)


def test_best_response_matches_brute_force_maximisation():
    """The fixed point on the first-order condition finds the true argmax."""
    params = IntensityParams()
    grid = np.linspace(-1.0, 8.0, 200_001)
    for mu in (0.0, 0.5, 1.5):
        for p in (-0.3, 0.0, 0.4, 1.2):
            got = float(best_response_quote(np.array([p]), mu, params, -1.0)[0])
            obj = params.f(grid, mu) * (grid - p)
            want = float(grid[int(np.argmax(obj))])
            assert got == pytest.approx(want, abs=1e-4), (mu, p, got, want)


def test_best_response_is_bracketed_by_the_theoretical_bounds():
    """Since g/g' lies in [1/k_m, 1], the optimum sits in [p + 1/k_m, p + 1]."""
    params = IntensityParams(k_m=3.0)
    p = np.array([-0.2, 0.0, 0.5, 1.0, 2.0])
    d = best_response_quote(p, 0.5, params, -10.0)
    assert np.all(d >= p + 1.0 / params.k_m - 1e-9)
    assert np.all(d <= p + 1.0 + 1e-9)


def test_best_response_respects_the_quote_floor():
    params = IntensityParams()
    d = best_response_quote(np.array([-5.0]), 0.0, params, delta_floor=-1.0)
    assert d[0] == pytest.approx(-1.0)


def test_equilibrium_converges():
    sol = solve_mfg(_fast_config())
    assert sol.residuals[-1] < 1e-6
    assert len(sol.residuals) >= 2
    assert sol.residuals[-1] < sol.residuals[0]


def test_equilibrium_quotes_skew_with_inventory():
    """Ask falls and bid rises with inventory -- the skew of Figures 4 and 6."""
    sol = solve_mfg(_fast_config())
    ask = sol.ask_quote[1:]
    bid = sol.bid_quote[:-1]
    assert np.all(np.diff(ask) < 0), ask
    assert np.all(np.diff(bid) > 0), bid


def test_equilibrium_is_symmetric_between_the_two_sides():
    sol = solve_mfg(_fast_config())
    assert sol.mu_a == pytest.approx(sol.mu_b, rel=1e-6)
    assert sol.ask_quote[1:] == pytest.approx(sol.bid_quote[:-1][::-1], rel=1e-6)
    assert sol.density == pytest.approx(sol.density[::-1], rel=1e-6)


def test_inventory_density_is_a_distribution_peaked_at_flat():
    """Figure 3: the population prefers flat inventory because holding it costs."""
    sol = solve_mfg(_fast_config())
    assert sol.density.sum() == pytest.approx(1.0)
    assert np.all(sol.density >= 0)
    assert int(np.argmax(sol.density)) == sol.q_grid.size // 2


def test_value_function_is_maximised_at_flat_inventory():
    sol = solve_mfg(_fast_config())
    assert int(np.argmax(sol.value)) == sol.q_grid.size // 2


def test_monopolist_quotes_wider_than_nash():
    """Figure 4: competition compresses spreads below the monopolistic level.

    This gap is the band the strategy calls supra-competitive.
    """
    cfg = _fast_config()
    sol = solve_mfg(cfg)
    m_ask, m_bid = monopolistic_quotes(cfg)
    inner = slice(1, -1)
    assert np.all(m_ask[inner] > sol.ask_quote[inner])
    assert np.all(m_bid[inner] > sol.bid_quote[inner])


def test_quotes_are_strategic_complements():
    """Larger k -- a stronger dependence of our fill rate on the population quote --
    raises the equilibrium spread.

    This is the mechanism the whole strategy hangs on.  In eq. (3.1) the parameter
    k governs how much a *wide* population helps us win, so a wider crowd supports
    a wider best response, which widens the crowd further.  Quotes are strategic
    complements, and equilibrium spreads rise with k.  It is the same feedback the
    paper's learning dealers ride into the supra-competitive regime, and the
    reason a market can sit persistently wide of Nash without anyone colluding
    explicitly.
    """
    weak = _fast_config(ask=IntensityParams(k=1.0), bid=IntensityParams(k=1.0))
    strong = _fast_config(ask=IntensityParams(k=3.0), bid=IntensityParams(k=3.0))
    assert solve_mfg(strong).nash_half_spread() > solve_mfg(weak).nash_half_spread()


def test_higher_inventory_cost_sharpens_the_skew():
    cheap = solve_mfg(_fast_config(inventory_cost=0.005))
    dear = solve_mfg(_fast_config(inventory_cost=0.05))
    slope_cheap = cheap.ask_quote[1] - cheap.ask_quote[-1]
    slope_dear = dear.ask_quote[1] - dear.ask_quote[-1]
    assert slope_dear > slope_cheap


def test_quotes_are_barred_at_the_inventory_bounds():
    sol = solve_mfg(_fast_config())
    assert np.isnan(sol.ask_quote[0])       # at -Z we cannot sell any more
    assert np.isnan(sol.bid_quote[-1])      # at +Z we cannot buy any more
