"""Checks against the analytical results in the robust market-making paper."""

import numpy as np
import pytest

from src.robust_quotes import (
    RobustParams,
    RobustQuoter,
    _generator,
    optimal_depths,
    solve_h_stationary,
    solve_h_terminal,
    worst_case_drift,
)

# Parameters of Figure 1 in the paper.
FIG1 = dict(kappa=15.0, lam_buy=2.0, lam_sell=2.0, sigma=0.01, alpha=0.0, theta=0.01, q_max=3)


def test_base_depth_reduces_to_reference_model():
    """With no fill ambiguity the base depth is the classic 1/kappa."""
    p = RobustParams(kappa=15.0, phi=0.0)
    assert p.base_depth() == pytest.approx(1.0 / 15.0)


def test_base_depth_continuous_and_decreasing_in_phi():
    p = RobustParams(kappa=15.0)
    depths = []
    for phi in [0.0, 1e-9, 1e-6, 0.1, 1.0, 6.0, 20.0]:
        p.phi = phi
        depths.append(p.base_depth())
    assert depths[0] == pytest.approx(depths[1], rel=1e-6)
    assert depths[1] == pytest.approx(depths[2], rel=1e-4)
    assert all(a > b for a, b in zip(depths[2:], depths[3:])), depths


def test_xi_limit_matches_ambiguity_neutral_hjb():
    """xi -> lambda / e as phi -> 0, the reduction of the neutral HJB equation."""
    p = RobustParams(kappa=27.0, lam_buy=2.0, lam_sell=3.0, phi=0.0)
    assert p.xi() == pytest.approx((2.0 / np.e, 3.0 / np.e))
    p.phi = 1e-7
    assert p.xi() == pytest.approx((2.0 / np.e, 3.0 / np.e), rel=1e-6)


def test_figure_1_sell_depths():
    """Reproduce Figure 1: ambiguity-neutral sell depths at t=0 with T=10 seconds.

    The published curve reads roughly 0.105, 0.085, 0.072, 0.060, 0.048, 0.025
    for q = -2, -1, 0, 1, 2, 3.
    """
    p = RobustParams(**FIG1)
    h = solve_h_terminal(p, T=10.0, n_steps=8000)
    ask, _ = optimal_depths(p, h[0])
    expected = [0.105, 0.085, 0.072, 0.060, 0.048, 0.025]
    assert ask[1:] == pytest.approx(expected, abs=3e-3)


def test_symmetry_of_depths_proposition_6():
    """delta_+(q) = delta_-(-q) when the dynamics are symmetric."""
    p = RobustParams(**FIG1, phi_alpha=7.0, phi=3.0)
    h = solve_h_terminal(p, T=10.0, n_steps=4000)
    ask, bid = optimal_depths(p, h[0])
    assert ask[1:] == pytest.approx(bid[:-1][::-1], abs=1e-9)


def test_proposition_7_effect_of_drift_ambiguity():
    """Raising phi_alpha widens sell depths for q <= 0 and tightens them for q > 0."""
    base = RobustParams(**FIG1, phi_alpha=0.0)
    more = RobustParams(**FIG1, phi_alpha=20.0)
    a0, _ = optimal_depths(base, solve_h_terminal(base, T=10.0, n_steps=4000)[0])
    a1, _ = optimal_depths(more, solve_h_terminal(more, T=10.0, n_steps=4000)[0])
    q = base.q_grid
    lower = ~np.isnan(a0) & (q <= 0)
    upper = ~np.isnan(a0) & (q > 0)
    assert np.all(a1[lower] >= a0[lower] - 1e-12)
    assert np.all(a1[upper] <= a0[upper] + 1e-12)


def test_drift_ambiguity_widens_total_depth():
    """Figure 5, right panel: the total spread is a buffer against adverse selection."""
    base = RobustParams(**FIG1, phi_alpha=0.0)
    more = RobustParams(**FIG1, phi_alpha=20.0)
    t0 = np.nansum
    a0, b0 = optimal_depths(base, solve_h_terminal(base, T=10.0, n_steps=4000)[0])
    a1, b1 = optimal_depths(more, solve_h_terminal(more, T=10.0, n_steps=4000)[0])
    mid = base.q_grid.size // 2
    assert (a1 + b1)[mid] > (a0 + b0)[mid]
    assert t0(a1 + b1) > t0(a0 + b0)


def test_fill_ambiguity_narrows_total_depth():
    """Figure 7: fearing non-execution, the MM tightens to increase churn."""
    base = RobustParams(**FIG1, phi=0.0)
    more = RobustParams(**FIG1, phi=6.0)
    a0, b0 = optimal_depths(base, solve_h_terminal(base, T=10.0, n_steps=4000)[0])
    a1, b1 = optimal_depths(more, solve_h_terminal(more, T=10.0, n_steps=4000)[0])
    mid = base.q_grid.size // 2
    assert (a1 + b1)[mid] < (a0 + b0)[mid]


def test_worst_case_drift_leans_against_inventory():
    """eta*(q) = alpha - phi_alpha sigma^2 q, eq. (24)."""
    p = RobustParams(**FIG1, phi_alpha=20.0)
    assert worst_case_drift(p, 0) == pytest.approx(p.alpha)
    assert worst_case_drift(p, 3) == pytest.approx(p.alpha - 20.0 * 0.01**2 * 3)
    assert worst_case_drift(p, 3) < worst_case_drift(p, -3)


def test_stationary_solution_matches_long_horizon_integration():
    """The eigenvector shortcut agrees with integrating the ODE from a distant horizon."""
    p = RobustParams(kappa=27.0, lam_buy=2.0, lam_sell=2.0, theta=0.001, q_max=8,
                     phi_alpha=2.0, phi=5.0)
    h_eig = solve_h_stationary(p)
    h_int = solve_h_terminal(p, T=400.0, n_steps=200_000)[0]
    h_int = h_int - h_int[p.q_grid.size // 2]
    assert h_eig == pytest.approx(h_int, abs=1e-6)


def test_stationary_generator_is_tridiagonal_with_positive_offdiagonals():
    p = RobustParams(q_max=4, phi=3.0)
    A, xi_p, xi_m = _generator(p)
    n = A.shape[0]
    assert xi_p > 0 and xi_m > 0
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1:
                assert A[i, j] == 0.0
    assert np.allclose(np.diag(A, -1), xi_p)
    assert np.allclose(np.diag(A, 1), xi_m)


def test_quoter_bars_the_blocked_side_at_inventory_limits():
    q = RobustQuoter(RobustParams(q_max=5))
    ask, bid = q.depths(5)
    assert np.isfinite(ask) and np.isnan(bid)     # long the cap: sell only
    ask, bid = q.depths(-5)
    assert np.isnan(ask) and np.isfinite(bid)     # short the floor: buy only
    with pytest.raises(ValueError):
        q.depths(6)


def test_quoter_skews_monotonically_with_inventory():
    """Long inventory must lower the ask and raise the bid, at every level."""
    q = RobustQuoter(RobustParams(kappa=27.0, q_max=8, phi_alpha=2.0))
    ask, bid = q.curves()
    a = ask[1:]
    b = bid[:-1]
    assert np.all(np.diff(a) < 0), a
    assert np.all(np.diff(b) > 0), b


def test_quoter_caches_and_invalidates():
    q = RobustQuoter(RobustParams(kappa=27.0, q_max=4))
    first = q.curves()[0]
    q.update(phi_alpha=0.0)                       # unchanged value: same solution
    assert q.curves()[0] == pytest.approx(first, nan_ok=True)
    q.update(phi_alpha=10.0)
    assert not np.allclose(q.curves()[0], first, equal_nan=True)
    with pytest.raises(AttributeError):
        q.update(not_a_parameter=1.0)


def test_depths_are_non_negative():
    """The (.)+ in Proposition 3 must actually be enforced."""
    p = RobustParams(kappa=27.0, q_max=8, theta=0.5, phi_alpha=20.0, phi=15.0)
    ask, bid = optimal_depths(p, solve_h_stationary(p))
    assert np.nanmin(ask) >= 0.0
    assert np.nanmin(bid) >= 0.0


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        RobustParams(kappa=0.0)
    with pytest.raises(ValueError):
        RobustParams(phi=-1.0)
