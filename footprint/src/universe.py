"""Point-in-time universe selection, and an honest statement of its bias.

Two different biases are at stake here and they need separating, because one is
fixable in code and the other is not.

Look-ahead in universe selection *is* fixable, and is fixed here: the liquid set
on date t is chosen using trailing dollar volume observed strictly before t, so
the backtest never holds a name because of liquidity it had not yet shown. The
universe is therefore re-selected as the walk-forward advances.

Survivorship is *not* fixable from this data source, and no amount of care in
this module removes it. See ``SURVIVORSHIP_NOTE``.
"""

from __future__ import annotations

import pandas as pd

from .loader import Panel

SURVIVORSHIP_NOTE = """\
SURVIVORSHIP BIAS -- unfixable from a directory of currently-listed tickers.

The universe here is whatever files sit in data/. If those files were obtained by
downloading "the most liquid US names" as of today, then by construction the
sample contains only companies that still exist and are still liquid today. Every
name that was delisted, acquired, bankrupted, or that fell out of the liquid set
is absent -- and those are disproportionately the names that fell hardest.

The direction of the resulting error is not neutral, and it is not the same for
every part of this project:

  * Long-only and buy-and-hold benchmarks are inflated, straightforwardly, since
    the losers were removed from the sample.
  * A cross-sectional dollar-neutral strategy is affected more subtly. It is
    hurt on the short side -- the disaster names it would have shorted are
    missing -- and the tail-risk estimates in ``tailrisk.py`` are biased toward
    optimism for the same reason: the fattest realised left tails belong to
    securities that are no longer in any current listing.
  * The footprint signal is a liquidity-footprint signal, so a universe screened
    on *present* liquidity is exactly the wrong screen: it conditions on the
    outcome variable's cousin. A name that was liquid in 2015 and illiquid by
    2020 is precisely a name whose lambda dynamics carry information, and it is
    the one most likely to be missing.

Magnitude, from the literature rather than from this data: survivorship in US
equity samples has historically been worth on the order of 1-4% per year on
long-only returns. That is comparable to or larger than the entire expected edge
here, so it cannot be waved away as second order.

The only real fix is a point-in-time database with delisted securities and
delisting returns (CRSP, or a vendor equivalent). Until data/ contains delisted
names with their terminal returns, every number this project reports should be
read as an upper bound.
"""


def trailing_dollar_volume(panel: Panel, window: int = 60) -> pd.DataFrame:
    """Median dollar volume over a trailing window, strictly excluding today.

    The shift by one is what makes it point-in-time: the value on date t uses
    sessions up to t-1 only, so selecting on it cannot see t's own trading.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    dv = panel.dollar_volume
    return dv.rolling(window, min_periods=max(2, window // 3)).median().shift(1)


def liquid_universe(panel: Panel, *, n: int = 20, window: int = 60,
                    min_price: float = 5.0) -> pd.DataFrame:
    """Boolean membership mask: is ticker j in the liquid set on date t?

    Ranks on trailing median dollar volume and keeps the top ``n``. A minimum
    price screen removes sub-$5 names, where the tick is a large fraction of the
    spread and the cost model in this project would understate trading costs
    badly.

    Both screens use lagged data only.
    """
    adv = trailing_dollar_volume(panel, window)
    eligible = adv.notna()
    if min_price > 0:
        eligible &= panel.close.shift(1) >= min_price
    scores = adv.where(eligible)
    # rank descending; 1 is the most liquid name that day
    ranks = scores.rank(axis=1, ascending=False, method="first")
    return (ranks <= n) & eligible


def apply_universe(frame: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Blank out everything outside the universe, keeping the frame's shape."""
    return frame.where(mask.reindex_like(frame).fillna(False))


def universe_turnover(mask: pd.DataFrame) -> pd.Series:
    """Names entering or leaving the universe each day, as a fraction of size.

    A universe that churns fast makes the backtest's turnover figures optimistic,
    because entering and leaving names force trades the signal never asked for.
    """
    size = mask.sum(axis=1).replace(0, pd.NA)
    changes = mask.astype(int).diff().abs().sum(axis=1)
    return (changes / size).astype(float)
