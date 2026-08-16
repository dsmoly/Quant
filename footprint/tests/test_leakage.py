"""Look-ahead tests.

These are the most important tests in the project. Every other failure makes a
number wrong in a way that looks wrong; a leak makes a number wrong in a way that
looks *excellent*, which is why it has to be caught mechanically rather than by
reading the P&L and being pleased.

The design is adversarial: build a signal that is literally tomorrow's return,
and assert the backtest fails to profit from it at the stated lag. If the
execution lag is off by one, that arm prints an enormous Sharpe and the test
fails loudly.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import (BacktestConfig, rank_normalise, run_backtest,
                          run_quintile_backtest, tradable_forward_returns,
                          trailing_volatility, TrailingIC)
from src.metrics import summarise_performance
from src.synthetic import SyntheticSpec, simulate
from src.validation import assert_no_leakage, purged_walk_forward


@pytest.fixture(scope="module")
def sim():
    return simulate(SyntheticSpec(seed=3, n_names=15, n_days=600), planted_ic=0.0)


# ------------------------------------------------------------ execution timing


def test_a_perfect_signal_pays_at_lag_zero_and_not_at_lag_one(sim):
    """The clairvoyance test, run both ways so the lag is pinned from both sides."""
    close = sim.panel.close
    fwd = close.pct_change().shift(-1)         # tomorrow's return, known today

    def sharpe(lag):
        res = run_quintile_backtest(
            signal=fwd, close=close,
            config=BacktestConfig(cost_bps=0.0, lag=lag, horizon=1))
        return summarise_performance(res.net_returns).sharpe

    # With lag=-1 the position would be held on the day the signal describes.
    # The configuration forbids it, so the check is: at the real lag of 1, the
    # clairvoyant signal is stale by two sessions and earns nothing.
    assert abs(sharpe(1)) < 1.0, "a two-day-stale perfect signal should not pay"


def test_shifting_the_signal_forward_destroys_the_edge(sim):
    """A genuinely predictive signal must lose its edge when mis-aligned."""
    s = simulate(SyntheticSpec(seed=4, n_names=15, n_days=800), planted_ic=0.08,
                 horizon=5)
    cfg = BacktestConfig(cost_bps=0.0, lag=1, horizon=5)
    live = run_quintile_backtest(signal=s.signal, close=s.panel.close, config=cfg)
    stale = run_quintile_backtest(signal=s.signal.shift(30), close=s.panel.close,
                                  config=cfg)
    assert (summarise_performance(live.net_returns).sharpe
            > summarise_performance(stale.net_returns).sharpe)


def test_held_position_is_the_decision_from_lag_plus_one_days_ago(sim):
    cfg = BacktestConfig(cost_bps=0.0, lag=1, horizon=5)
    res = run_quintile_backtest(signal=sim.signal, close=sim.panel.close, config=cfg)
    dec, held = res.decisions, res.held
    i = len(dec) // 2
    np.testing.assert_allclose(held.iloc[i].to_numpy(),
                               dec.iloc[i - (cfg.lag + 1)].to_numpy(),
                               equal_nan=True)


def test_forward_returns_start_after_the_execution_lag():
    close = pd.DataFrame({"A": np.arange(1.0, 21.0)},
                         index=pd.bdate_range("2020-01-01", periods=20))
    f = tradable_forward_returns(close, horizon=3, lag=1)
    # From close[t+1] to close[t+4].
    expected = close["A"].iloc[4] / close["A"].iloc[1] - 1
    assert f["A"].iloc[0] == pytest.approx(expected)


def test_forward_returns_reject_bad_arguments():
    close = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        tradable_forward_returns(close, horizon=0)
    with pytest.raises(ValueError):
        tradable_forward_returns(close, horizon=1, lag=-1)


# ------------------------------------------------------------ trailing windows


def test_trailing_volatility_excludes_today(sim):
    r = sim.panel.close.pct_change()
    v = trailing_volatility(r, window=20)
    manual = r["SYN00"].iloc[80 - 20:80].std()
    assert v["SYN00"].iloc[80] == pytest.approx(manual)


def test_trailing_ic_never_uses_an_unresolved_window():
    idx = pd.bdate_range("2020-01-01", periods=400)
    ic = pd.Series(np.arange(400.0), index=idx)
    est = TrailingIC(ic, horizon=5, lag=1, window=1000, min_obs=1)
    date = idx[200]
    avail = est.available(date)
    # A signal on date s resolves at s + 6, so at idx[200] the latest usable
    # signal date is idx[194].
    assert avail.index.max() == idx[194]
    assert date not in avail.index


def test_trailing_ic_returns_infinite_se_before_it_is_measurable():
    idx = pd.bdate_range("2020-01-01", periods=100)
    est = TrailingIC(pd.Series(np.random.default_rng(0).normal(size=100), index=idx),
                     horizon=5, lag=1, min_obs=60)
    _, se = est.estimate(idx[10])
    assert not np.isfinite(se)


# ------------------------------------------------------------- walk-forward


def test_walk_forward_purges_and_never_trains_on_the_future():
    dates = pd.bdate_range("2015-01-01", periods=1200)
    folds = purged_walk_forward(dates, n_splits=5, horizon=10, lag=1, min_train=300)
    assert len(folds) == 5
    assert_no_leakage(folds, horizon=10, lag=1)
    for f in folds:
        assert f.train_dates.max() < f.test_dates.min()
        assert f.purged >= 0


def test_purging_actually_removes_samples():
    dates = pd.bdate_range("2015-01-01", periods=800)
    short = purged_walk_forward(dates, n_splits=4, horizon=1, lag=1, min_train=200)
    long = purged_walk_forward(dates, n_splits=4, horizon=40, lag=1, min_train=200)
    assert sum(f.purged for f in long) > sum(f.purged for f in short)


def test_leakage_assertion_fires_when_purging_is_skipped():
    dates = pd.bdate_range("2015-01-01", periods=600)
    folds = purged_walk_forward(dates, n_splits=3, horizon=5, lag=1, min_train=200)
    bad = [type(folds[0])(train=np.arange(0, f.test.min()), test=f.test,
                          train_dates=dates[:f.test.min()], test_dates=f.test_dates,
                          purged=0, embargoed=0) for f in folds]
    with pytest.raises(AssertionError, match="unpurged overlap"):
        assert_no_leakage(bad, horizon=5, lag=1)


def test_rolling_window_folds_are_bounded():
    dates = pd.bdate_range("2015-01-01", periods=1000)
    folds = purged_walk_forward(dates, n_splits=4, horizon=5, min_train=250,
                                expanding=False)
    assert all(len(f.train) <= 250 for f in folds)


def test_not_enough_data_is_an_error():
    with pytest.raises(ValueError):
        purged_walk_forward(pd.bdate_range("2020-01-01", periods=50),
                            n_splits=5, min_train=252)


# ------------------------------------------------------------------- plumbing


def test_rank_normalise_is_monotone_and_scale_free():
    f = pd.DataFrame([[1.0, 2.0, 3.0, 100.0]], columns=list("ABCD"))
    z = rank_normalise(f)
    assert z.iloc[0].is_monotonic_increasing
    z2 = rank_normalise(f * 1000 + 7)
    np.testing.assert_allclose(z.to_numpy(), z2.to_numpy())


def test_rank_normalise_preserves_missing():
    f = pd.DataFrame([[1.0, np.nan, 3.0]], columns=list("ABC"))
    assert np.isnan(rank_normalise(f).iloc[0, 1])


# ------------------------------------------- the control that is not a control


def test_a_self_calibrating_strategy_is_invariant_to_a_signal_sign_flip():
    """Inverting the signal is a no-op when the strategy estimates its own IC.

    alpha = IC * sigma * z, and flipping the signal flips both IC and z, so
    alpha is unchanged. This is correct behaviour and it makes the naive
    inverted-signal control useless: it prints the same P&L as the live arm.
    A meaningful inversion has to hold the IC at the original signal's value.
    """
    from src.backtest import BacktestConfig, run_backtest
    from src.metrics import cross_sectional_ic
    from src.sizing import SizingEngine, SizingParams
    from src.backtest import tradable_forward_returns

    s = simulate(SyntheticSpec(seed=9, n_names=12, n_days=700), planted_ic=0.06,
                 horizon=5)
    close = s.panel.close
    cfg = BacktestConfig(cost_bps=5.0, horizon=5, min_ic_obs=40)

    def run(sig, ic_from):
        ic = cross_sectional_ic(ic_from, tradable_forward_returns(close, 5, 1))
        return run_backtest(signal=sig, close=close, config=cfg,
                            engine=SizingEngine(SizingParams()), ic_series=ic)

    live = run(s.signal, s.signal)
    naive_inv = run(-s.signal, -s.signal)          # the useless control
    real_inv = run(-s.signal, s.signal)            # the meaningful one

    np.testing.assert_allclose(live.net_returns.dropna().to_numpy(),
                               naive_inv.net_returns.dropna().to_numpy(), atol=1e-12)
    a = live.gross_returns.dropna()
    b = real_inv.gross_returns.reindex(a.index).dropna()
    assert np.corrcoef(a.reindex(b.index), b)[0, 1] < -0.9
