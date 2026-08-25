"""Control arms.

A backtest number means nothing on its own. These four say what it should be
compared against, and each one fails in a different, diagnostic way:

  flat            holds nothing. Its net return is exactly zero, so any non-zero
                  result from it is a bug in the accounting, not a strategy.

  buy and hold    equal-weight long the universe. This is the benchmark the
                  strategy has to beat to justify existing, and in a
                  survivorship-screened sample it is *inflated* -- see
                  ``universe.SURVIVORSHIP_NOTE``. If a dollar-neutral strategy
                  cannot beat a biased long benchmark, that is worth knowing
                  plainly.

  inverted        the same signal with its sign flipped. If the strategy makes
                  money and the inversion loses a symmetric amount, the edge is
                  in the signal. If both make money, the edge is in the sizing,
                  the universe, or a bug -- and that asymmetry is the single most
                  informative diagnostic in the set.

  shuffled        the signal's *timing* destroyed while its cross-sectional shape
                  is preserved, by permuting the date index. Anything that
                  survives this is not prediction: it is exposure to whatever
                  static tilt the signal encodes (size, volatility, sector), and
                  should be reported as such rather than as alpha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def flat_signal(signal: pd.DataFrame) -> pd.DataFrame:
    """A signal that ranks every name identically, so no position is taken."""
    return pd.DataFrame(0.0, index=signal.index, columns=signal.columns).where(
        signal.notna())


def buy_and_hold_weights(universe: pd.DataFrame) -> pd.DataFrame:
    """Equal weight, long only, renormalised as the universe changes."""
    mask = universe.astype(float)
    n = mask.sum(axis=1).replace(0, np.nan)
    return mask.div(n, axis=0).fillna(0.0)


def invert_signal(signal: pd.DataFrame) -> pd.DataFrame:
    return -signal


def shuffle_signal_timing(signal: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Permute whole cross-sections across dates, preserving each day's shape.

    Rows are moved intact, so the correlation structure *within* a day survives
    and only the date alignment is destroyed. Shuffling within rows instead would
    also destroy the cross-sectional tilt and would test a weaker null.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(signal))
    out = signal.iloc[order].copy()
    out.index = signal.index
    return out


def buy_and_hold_returns(universe: pd.DataFrame, close: pd.DataFrame,
                         lag: int = 1, cost_bps: float = 0.0) -> dict:
    """Return series for the long-only benchmark, on the same execution lag."""
    w = buy_and_hold_weights(universe.reindex_like(close).fillna(False))
    held = w.shift(lag + 1)
    rets = close.pct_change()
    gross = (held * rets).sum(axis=1, skipna=True)
    turnover = held.diff().abs().sum(axis=1)
    costs = turnover * (cost_bps / 1e4)
    return {"gross_returns": gross, "net_returns": gross - costs,
            "turnover": turnover, "costs": costs, "held": held}
