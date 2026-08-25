"""Pareto-kernel entropy estimator: correctness, and the bug it nearly hid."""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

from src.entropy import (EntropyFit, fit_tail_index, gaussian_entropy,
                         kernel_entropy, kozachenko_leonenko, pareto_kernel,
                         rolling_tail_index, rvs_symmetric_stable,
                         spacings_entropy)


def test_pareto_kernel_is_a_density():
    for a, b in ((0.5, 1.0), (2.0, 0.3), (4.0, 2.5)):
        total, _ = integrate.quad(lambda u: pareto_kernel(u, a, b), -np.inf, np.inf)
        assert total == pytest.approx(1.0, rel=1e-6)


def test_pareto_kernel_is_symmetric_and_decreasing():
    u = np.array([0.0, 0.5, 1.0, 5.0])
    k = pareto_kernel(u, 1.5, 1.0)
    assert np.all(np.diff(k) < 0)
    assert pareto_kernel(-2.0, 1.5, 1.0) == pytest.approx(pareto_kernel(2.0, 1.5, 1.0))


def test_entropy_recovers_the_gaussian_closed_form():
    """H = 0.5 log(2 pi e sigma^2) is exact; the estimator must find it."""
    x = np.random.default_rng(0).normal(0.0, 2.0, 4000)
    f = fit_tail_index(x)
    assert f.entropy == pytest.approx(gaussian_entropy(x.std(ddof=1)), abs=0.05)
    assert abs(f.excess_entropy) < 0.05


def test_entropy_rises_with_scale():
    rng = np.random.default_rng(1)
    a = fit_tail_index(rng.normal(0, 1, 2000)).entropy
    b = fit_tail_index(rng.normal(0, 4, 2000)).entropy
    # H(kX) = H(X) + log k
    assert b - a == pytest.approx(np.log(4.0), abs=0.15)


def test_heavy_tails_are_not_reported_as_gaussian():
    """Regression test for a real bug.

    The bandwidth search was originally scaled by the sample standard deviation.
    An alpha-stable law with alpha < 2 has infinite variance, so that statistic
    just tracks the largest draw, and the search range landed in the wrong place:
    alpha=0.8 data came back with alpha_hat pinned at the 6.0 upper bound, i.e.
    the heaviest-tailed sample in the suite was labelled Gaussian. The scale is
    now taken from the interquartile range, which is finite for every alpha.
    """
    x = rvs_symmetric_stable(0.8, 800, rng=np.random.default_rng(3))
    f = fit_tail_index(x)
    assert f.alpha < 1.5, f"alpha_hat={f.alpha} -- the sd-scaling bug is back"
    assert f.regime == "alpha-stable"


def test_alpha_hat_is_monotone_in_the_truth():
    rng = np.random.default_rng(4)
    got = [np.median([fit_tail_index(rvs_symmetric_stable(a, 600, rng=rng)).alpha
                      for _ in range(3)]) for a in (0.8, 1.2, 1.6)]
    assert got[0] < got[1] < got[2]


def test_gaussian_data_is_labelled_gaussian():
    x = np.random.default_rng(5).normal(size=1500)
    assert fit_tail_index(x).regime == "Gaussian"


def test_ties_are_handled_rather_than_refused():
    """Discrete tick sizes produce repeated returns; the fit must survive them."""
    rng = np.random.default_rng(6)
    x = np.round(rng.normal(size=600), 2)          # heavy tie structure
    assert len(np.unique(x)) < x.size
    f = fit_tail_index(x)
    assert f.converged and np.isfinite(f.entropy)
    # The nearest-neighbour estimator, by contrast, is undefined on ties.
    assert np.isnan(kozachenko_leonenko(x))


def test_nearest_neighbour_and_spacings_agree_on_clean_gaussian():
    x = np.random.default_rng(7).normal(size=3000)
    truth = gaussian_entropy(1.0)
    assert kozachenko_leonenko(x) == pytest.approx(truth, abs=0.1)
    assert spacings_entropy(x) == pytest.approx(truth, abs=0.1)


def test_short_samples_are_declined_not_guessed():
    f = fit_tail_index([1.0, 2.0, 3.0])
    assert not f.converged and np.isnan(f.alpha)


def test_rolling_window_is_trailing_by_default():
    """The label at t must not move when data after t changes.

    The original R uses a centred window (w.begin = t-30, w.end = t+30), so its
    regime label at time t is computed from thirty days of future data. That is
    fine for dating regimes in a paper and unusable as a trading input.
    """
    rng = np.random.default_rng(8)
    x = pd.Series(rng.normal(size=400))
    base = rolling_tail_index(x, window=61, step=20)
    tampered = x.copy()
    tampered.iloc[300:] = rng.normal(0, 25, len(tampered) - 300)   # wreck the future
    after = rolling_tail_index(tampered, window=61, step=20)
    common = base.index.intersection(after.index)
    common = common[common < 300]
    np.testing.assert_allclose(base.loc[common, "alpha"].to_numpy(),
                               after.loc[common, "alpha"].to_numpy())


def test_centred_window_does_leak_and_is_opt_in():
    rng = np.random.default_rng(9)
    x = pd.Series(rng.normal(size=400))
    base = rolling_tail_index(x, window=61, step=20, centred=True)
    tampered = x.copy()
    tampered.iloc[300:] = rng.normal(0, 25, len(tampered) - 300)
    after = rolling_tail_index(tampered, window=61, step=20, centred=True)
    common = base.index.intersection(after.index)
    near = common[(common > 260) & (common < 300)]
    assert len(near) > 0
    # Labels just before the tampering move, which is the leak.
    assert not np.allclose(base.loc[near, "alpha"], after.loc[near, "alpha"])


def test_stable_sampler_has_the_right_tail_ordering():
    rng = np.random.default_rng(10)
    q = [np.quantile(np.abs(rvs_symmetric_stable(a, 20000, rng=rng)), 0.999)
         for a in (0.8, 1.4, 2.0)]
    assert q[0] > q[1] > q[2]          # heavier alpha -> fatter extreme quantile
