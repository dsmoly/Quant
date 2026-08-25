"""Signed order flow, price bars, and how badly trade prices lie.

Sign convention
---------------
``is_buyer_maker == True`` means the resting side was the buyer, so the incoming
aggressive order was a **sell**. Therefore

    sign = -1 if is_buyer_maker else +1

This is the one thing here that is easy to invert and impossible to catch after
the fact: a flipped sign turns the response function upside down, and an upside
down response function still looks like a perfectly plausible curve. It is pinned
by a test that constructs trades with a known aggressor.

Bid-ask bounce
--------------
The response function is ``E[p(t+h) - p(t) | flow at t]``, and what ``p`` is
matters enormously at short horizons. A trade-price series alternates between bid
and ask as the aggressor side alternates, which injects a negative first-order
autocorrelation into returns that has nothing to do with information. At h = 1s
that bounce can be the same order of magnitude as the signal, and because it is
*negative* autocorrelation it makes impact look like it decays faster than it
does -- exactly the direction that would produce a spurious "the signal dies in
seconds" conclusion.

Two defences, both reported:

  1. Where bookTicker is available (futures only), the mid price is used and the
     bounce is absent by construction.
  2. Where a mid *is* available, ``effective_half_spread_bps`` measures the
     bounce directly as the signed distance from mid to trade price. This is the
     number to quote.

  3. Where it is not -- spot, which publishes no bookTicker -- ``roll_spread``
     tries to infer it from the negative serial covariance of trade-price
     returns (Roll 1984):

         s = 2 sqrt(-Cov(r_t, r_{t-1}))

     **and on a real crypto tape this frequently fails outright.** Roll's model
     assumes serially independent order flow. Order flow is not independent: it
     is famously long-memory because large orders are split into many child
     trades. When the impact autocorrelation from that splitting exceeds the
     bounce, the serial covariance turns *positive*, the square root has no real
     value, and the estimator returns nothing. That is verified here on tapes
     with known parameters -- Roll recovers a planted spread exactly when impact
     is switched off, and fails when it is not. So on spot the bounce may not be
     measurable at all, which is a further argument for measuring this question
     on futures where a book is published.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def sign_trades(trades: pd.DataFrame) -> pd.Series:
    """+1 for buyer-initiated, -1 for seller-initiated."""
    if "is_buyer_maker" not in trades.columns:
        raise ValueError("trades frame has no is_buyer_maker column")
    return pd.Series(np.where(trades["is_buyer_maker"].to_numpy(), -1.0, 1.0),
                     index=trades.index, name="sign")


@dataclass
class Bars:
    """Regular-grid bars. ``index`` is the left edge of each interval."""

    freq: str
    seconds: float
    trade_price: pd.Series      # last trade print in the bar
    mid: pd.Series | None       # last mid in the bar, if a book was available
    signed_volume: pd.Series    # sum of sign * qty
    signed_count: pd.Series     # sum of sign
    volume: pd.Series           # sum of |qty|
    n_trades: pd.Series

    @property
    def price(self) -> pd.Series:
        """Mid where we have it, trade price otherwise."""
        return self.mid if self.mid is not None else self.trade_price

    @property
    def has_mid(self) -> bool:
        return self.mid is not None

    def frame(self) -> pd.DataFrame:
        d = {"trade_price": self.trade_price, "signed_volume": self.signed_volume,
             "signed_count": self.signed_count, "volume": self.volume,
             "n_trades": self.n_trades}
        if self.mid is not None:
            d["mid"] = self.mid
        return pd.DataFrame(d)


def _freq_seconds(freq: str) -> float:
    return float(pd.to_timedelta(freq).total_seconds())


def build_bars(trades: pd.DataFrame, freq: str = "1s",
               book: pd.DataFrame | None = None) -> Bars:
    """Aggregate a trade tape (and optionally a book tape) onto a regular grid.

    Bars with no trades are dropped rather than forward-filled. Forward-filling
    would manufacture zero-flow, zero-return observations that drag every
    correlation toward zero and, worse, would make a quiet period look like
    evidence of decay.
    """
    if trades.empty:
        raise ValueError("no trades")
    t = trades.copy()
    t["sign"] = sign_trades(t)
    t["signed_qty"] = t["sign"] * t["qty"]
    g = t.set_index("ts").groupby(pd.Grouper(freq=freq))

    bars = pd.DataFrame({
        "trade_price": g["price"].last(),
        "signed_volume": g["signed_qty"].sum(),
        "signed_count": g["sign"].sum(),
        "volume": g["qty"].sum(),
        "n_trades": g["price"].count(),
    })
    bars = bars[bars["n_trades"] > 0]

    mid = None
    if book is not None and not book.empty:
        bm = book.set_index("ts").groupby(pd.Grouper(freq=freq))["mid"].last()
        mid = bm.reindex(bars.index)
        # A bar with trades but no book update inherits the last known mid; the
        # book is a much denser tape than the trade tape, so this is rare.
        mid = mid.ffill()
        if mid.isna().all():
            mid = None
    return Bars(freq=freq, seconds=_freq_seconds(freq),
                trade_price=bars["trade_price"], mid=mid,
                signed_volume=bars["signed_volume"],
                signed_count=bars["signed_count"], volume=bars["volume"],
                n_trades=bars["n_trades"])


def resample_bars(bars: Bars, freq: str) -> Bars:
    """Coarsen existing bars to a longer interval, summing flow and taking last price."""
    f = bars.frame()
    g = f.groupby(pd.Grouper(freq=freq))
    out = pd.DataFrame({
        "trade_price": g["trade_price"].last(),
        "signed_volume": g["signed_volume"].sum(),
        "signed_count": g["signed_count"].sum(),
        "volume": g["volume"].sum(),
        "n_trades": g["n_trades"].sum(),
    })
    out = out[out["n_trades"] > 0]
    mid = g["mid"].last().reindex(out.index) if "mid" in f.columns else None
    return Bars(freq=freq, seconds=_freq_seconds(freq),
                trade_price=out["trade_price"], mid=mid,
                signed_volume=out["signed_volume"], signed_count=out["signed_count"],
                volume=out["volume"], n_trades=out["n_trades"])


def normalise_flow(x: pd.Series, window: int | None = None) -> pd.Series:
    """Flow in units of its own standard deviation.

    A trailing window keeps it causal when the series is used as a live signal;
    with ``window=None`` the full-sample scale is used, which is appropriate for
    *measuring* a response function (the question is descriptive, not a backtest)
    but would be look-ahead in a strategy.
    """
    if window:
        sd = x.rolling(window, min_periods=max(20, window // 5)).std().shift(1)
    else:
        sd = pd.Series(float(x.std(ddof=1)), index=x.index)
    return (x / sd.where(sd > 0)).replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------------------ bounce diagnostics


def roll_spread(prices: pd.Series) -> float:
    """Roll's effective spread from the negative serial covariance of returns.

    Returns the full spread in the price's own units. When the serial covariance
    is positive the estimator is undefined -- Roll's model cannot describe a
    trending tape -- and NaN is returned rather than a fabricated number.
    """
    r = np.diff(np.log(pd.Series(prices).dropna().to_numpy()))
    if r.size < 32:
        return float("nan")
    cov = float(np.cov(r[1:], r[:-1])[0, 1])
    if cov >= 0:
        # Roll's model cannot describe a tape whose returns are positively
        # autocorrelated; returning nan is the honest answer, not a small number.
        return float("nan")
    spread = float(2.0 * np.sqrt(-cov))
    # Guard against a degenerate series where the "spread" is at float noise.
    return spread if spread > 1e-12 else float("nan")


def first_order_autocorr(prices: pd.Series) -> float:
    r = np.diff(np.log(pd.Series(prices).dropna().to_numpy()))
    if r.size < 32:
        return float("nan")
    return float(np.corrcoef(r[1:], r[:-1])[0, 1])


def align_mid_to_trades(trades: pd.DataFrame, book: pd.DataFrame) -> pd.DataFrame:
    """As-of join the prevailing mid onto each trade (backward, never forward).

    ``direction='backward'`` is load-bearing: the mid attached to a trade must be
    the last one published at or before it. Joining to the nearest quote in
    either direction would let a post-trade quote update -- which already
    reflects the trade -- leak into the pre-trade reference, and the measured
    effective spread would collapse toward zero.
    """
    t = trades[["ts", "price", "qty", "is_buyer_maker"]].sort_values("ts")
    b = book[["ts", "mid"]].sort_values("ts")
    out = pd.merge_asof(t, b, on="ts", direction="backward")
    out["sign"] = sign_trades(out)
    return out.dropna(subset=["mid"])


def effective_half_spread_bps(trades_with_mid: pd.DataFrame) -> float:
    """Signed distance from mid to trade price, in bps -- the direct measurement.

    ``mean( sign * (price - mid) / mid )``. Where a mid is available this is the
    bounce, measured rather than modelled, and it does not care whether order
    flow is autocorrelated. It is the number Roll's estimator is trying to infer
    without a book.
    """
    d = trades_with_mid
    if d.empty:
        return float("nan")
    return float((d["sign"] * (d["price"] - d["mid"]) / d["mid"]).mean() * 1e4)


def bounce_diagnostics(bars: Bars, trades: pd.DataFrame | None = None) -> dict:
    """How much does using trade prices instead of mid distort things?

    ``trades`` is the raw tape, and it matters: **Roll's estimator only works at
    the trade level.** The bounce is an alternation between bid and ask from one
    trade to the next, so aggregating into bars averages it away -- on a tape
    with several trades per second, one-second bars show *positive* return
    autocorrelation from impact rather than the negative autocorrelation Roll
    needs, and the estimator returns nothing at all. Passing the raw trades gives
    the spread estimate; passing only bars gives the bar-level autocorrelations,
    which are still worth reporting but are not a bounce measurement.
    """
    out = {
        "trade_bar_roll_spread_bps": 1e4 * roll_spread(bars.trade_price),
        "trade_ac1": first_order_autocorr(bars.trade_price),
        "has_mid": bars.has_mid,
    }
    if trades is not None and not trades.empty:
        out["tick_roll_spread_bps"] = 1e4 * roll_spread(trades["price"])
        out["tick_ac1"] = first_order_autocorr(trades["price"])
        if "mid" in trades.columns:
            out["effective_half_spread_bps"] = effective_half_spread_bps(trades)
        out["roll_usable"] = bool(np.isfinite(out["tick_roll_spread_bps"]))
    if bars.has_mid:
        out["mid_roll_spread_bps"] = 1e4 * roll_spread(bars.mid)
        out["mid_ac1"] = first_order_autocorr(bars.mid)
        out["ac1_gap"] = out["trade_ac1"] - out["mid_ac1"]
    return out
