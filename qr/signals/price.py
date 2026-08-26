"""Signals computable from OHLCV and a calendar.

These are the ones you can actually build on day one with free or cheap data.
The higher-conviction ideas in ``docs/signal_catalogue.md`` need index
membership, borrow, or options data; start here because the harness is what you
are really testing at first, and a signal you cannot compute cannot test it.
"""

from __future__ import annotations

import numpy as np


def _roll_mean(x, w, min_periods=None):
    """Trailing mean over ``w`` rows ending at t-1 (strictly past)."""
    x = np.asarray(x, dtype=float)
    T, N = x.shape
    mp = min_periods or max(2, w // 2)
    finite = np.isfinite(x)
    v = np.where(finite, x, 0.0)
    cs = np.vstack([np.zeros((1, N)), np.cumsum(v, axis=0)])
    cn = np.vstack([np.zeros((1, N)), np.cumsum(finite, axis=0)])
    out = np.full((T, N), np.nan)
    for t in range(T):
        lo = max(0, t - w)
        s = cs[t] - cs[lo]
        c = cn[t] - cn[lo]
        ok = c >= mp
        out[t, ok] = s[ok] / c[ok]
    return out


def _roll_std(x, w, min_periods=None):
    x = np.asarray(x, dtype=float)
    m1 = _roll_mean(x, w, min_periods)
    m2 = _roll_mean(x ** 2, w, min_periods)
    var = m2 - m1 ** 2
    return np.sqrt(np.clip(var, 0.0, None))


def time_series_momentum(returns, lookback: int = 252, skip: int = 21):
    """Trailing return over ``lookback``, skipping the most recent ``skip``.

    The skip is not decoration: the last month carries short-term reversal that
    runs against momentum, and including it materially weakens the signal in
    equities. Observable at the close of t using returns through t.
    """
    r = np.asarray(returns, dtype=float)
    T, N = r.shape
    v = np.where(np.isfinite(r), r, 0.0)
    c = np.vstack([np.zeros((1, N)), np.cumsum(v, axis=0)])
    out = np.full((T, N), np.nan)
    for t in range(T):
        hi = t + 1 - skip
        lo = hi - lookback
        if lo < 0:
            continue
        out[t] = c[hi] - c[lo]
    return out


def trend_ensemble(returns, lookbacks=(21, 63, 126, 252), skip: int = 0):
    """Average of z-scored trailing returns over several lookbacks.

    Averaging across horizons rather than optimising one is the cheapest real
    robustness available, and the horizons turn out to be genuinely diversifying
    rather than merely a hedge: fast and slow variants of the same family run
    around 0.45 correlated, so the ensemble is also a breadth play.
    """
    parts = []
    for lb in lookbacks:
        m = time_series_momentum(returns, lookback=lb, skip=skip)
        s = _roll_std(np.asarray(returns, dtype=float), max(lb, 21)) * np.sqrt(lb)
        with np.errstate(divide="ignore", invalid="ignore"):
            parts.append(np.where(s > 0, m / s, np.nan))
    P = np.stack(parts, axis=0)
    with np.errstate(invalid="ignore"):
        out = np.nanmean(P, axis=0)
    return np.where(np.isfinite(out), out, np.nan)


def overnight_intraday_split(open_px, close_px, prev_close):
    """Return (overnight, intraday) components of the daily return.

    Mechanism: overnight and intraday returns are compensated differently and
    have opposite short-horizon autocorrelation. Overnight carries most of the
    equity risk premium and news-driven repricing; intraday carries
    liquidity provision and inventory effects.

    Why it may not be arbed: capturing either leg cleanly requires trading at
    the open and the close, which doubles your cost and forces you into the two
    most competitive moments of the day. The effect is well documented and
    still hard to monetise, which is a good sign -- an effect that survives
    documentation is usually surviving on frictions, and frictions are more
    durable than secrecy.
    """
    o = np.asarray(open_px, dtype=float)
    c = np.asarray(close_px, dtype=float)
    pc = np.asarray(prev_close, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        overnight = np.where(pc > 0, o / pc - 1.0, np.nan)
        intraday = np.where(o > 0, c / o - 1.0, np.nan)
    return overnight, intraday


def intraday_reversal(open_px, close_px, prev_close, window: int = 5):
    """Trailing mean of the intraday leg, sign-flipped: fade recent intraday moves.

    Mechanism: intraday moves are disproportionately liquidity-driven, and the
    compensation for absorbing them accrues over the following days. The
    overnight leg is deliberately excluded because it contains the news.

    Kill criteria: dies once costs at 5bps are charged; or the effect is
    entirely in the smallest quintile by dollar volume.
    """
    _, intra = overnight_intraday_split(open_px, close_px, prev_close)
    return -_roll_mean(intra, window, min_periods=max(2, window // 2))


def turn_of_month_flow(returns, day_of_month, equity_mask=None, window: int = 21,
                       days_before: int = 3):
    """Month-end rebalancing pressure, **conditioned on the preceding divergence**.

    Mechanism: balanced funds and pensions rebalance to fixed weights at month
    end. If equities outperformed bonds over the month, they must sell equities
    and buy bonds; the flow is proportional to the divergence, not constant.

    Why it may not be arbed: nearly every published version trades the calendar
    unconditionally, which averages the good months with the ones where the
    flow pointed the other way. Conditioning on the magnitude *and sign* of the
    month's divergence is where the edge concentrates, and the capacity is
    small enough that it is not worth a large fund's operational risk.

    ``day_of_month`` is a (T,) integer array; ``equity_mask`` is a (N,) boolean
    marking which columns are the equity leg (the rest are treated as the bond
    leg). Returns a signal that is zero except in the ``days_before`` window.
    """
    r = np.asarray(returns, dtype=float)
    T, N = r.shape
    dom = np.asarray(day_of_month)
    if equity_mask is None:
        equity_mask = np.ones(N, dtype=bool)
    eq = np.asarray(equity_mask, dtype=bool)

    month_ret = _roll_mean(r, window, min_periods=window // 2) * window
    out = np.zeros((T, N))
    # last `days_before` calendar days of the month, detected by a day-of-month drop
    is_last = np.zeros(T, dtype=bool)
    for t in range(T - 1):
        if dom[t + 1] < dom[t]:
            for j in range(max(0, t - days_before + 1), t + 1):
                is_last[j] = True
    for t in range(T):
        if not is_last[t]:
            continue
        row = month_ret[t]
        if not np.isfinite(row[eq]).any():
            continue
        div = np.nanmean(row[eq]) - (np.nanmean(row[~eq]) if (~eq).any() else 0.0)
        if not np.isfinite(div):
            continue
        # rebalancers sell what outperformed: signal is the negative divergence
        out[t, eq] = -div
        if (~eq).any():
            out[t, ~eq] = div
    return np.where(out == 0.0, np.nan, out)


def dispersion_conditioned_trend(returns, lookbacks=(21, 63, 126, 252),
                                 disp_window: int = 63):
    """Trend, scaled by how much cross-sectional dispersion there is to exploit.

    Mechanism: a cross-sectional signal needs cross-sectional variation. When
    every name moves together (high correlation, low dispersion) there is
    nothing for a relative signal to pick up, and the same forecast produces
    the same trades against a much worse opportunity set.

    Why it may not be arbed: this is a *conditioning* variable, not a signal.
    It does not show up in a standard factor regression because it changes the
    signal's magnitude rather than its direction, so screens built around
    unconditional IC will not find it.

    Kill criteria: the interaction adds nothing once you control for realised
    volatility -- dispersion and volatility are correlated and only one of them
    can be the mechanism.
    """
    tr = trend_ensemble(returns, lookbacks)
    r = np.asarray(returns, dtype=float)
    T = r.shape[0]
    disp = np.full(T, np.nan)
    for t in range(T):
        lo = max(0, t - disp_window)
        block = r[lo:t]
        if block.shape[0] < max(5, disp_window // 3):
            continue
        cs = np.nanstd(block, axis=1)
        if np.isfinite(cs).sum() >= 3:
            disp[t] = float(np.nanmean(cs))
    med = np.nanmedian(disp)
    if not np.isfinite(med) or med <= 0:
        return tr
    scale = np.clip(disp / med, 0.25, 4.0)
    return tr * scale[:, None]


def idiosyncratic_volatility(returns, market_returns=None, window: int = 63):
    """Trailing volatility of the market-residual return.

    Included specifically because its relationship to forward returns is
    **known to be non-monotone** -- humped, with both the lowest and highest
    buckets underperforming the middle for different reasons. It is therefore
    the natural test case for the one regime where a flexible model beats a
    linear one, and a good check that your quantile report is working: if this
    comes back monotone, something in the pipeline is wrong.
    """
    r = np.asarray(returns, dtype=float)
    T, N = r.shape
    if market_returns is None:
        market_returns = np.nanmean(r, axis=1)
    m = np.asarray(market_returns, dtype=float)
    resid = np.full((T, N), np.nan)
    for t in range(T):
        lo = max(0, t - window)
        blk, mk = r[lo:t], m[lo:t]
        if blk.shape[0] < max(10, window // 3):
            continue
        ok_m = np.isfinite(mk)
        if ok_m.sum() < 10:
            continue
        x = mk[ok_m]
        xc = x - x.mean()
        denom = float(xc @ xc)
        if denom <= 0:
            continue
        for i in range(N):
            y = blk[ok_m, i]
            ok = np.isfinite(y)
            if ok.sum() < 10:
                continue
            yy, xx = y[ok], xc[ok]
            b = float(xx @ (yy - yy.mean())) / max(float(xx @ xx), 1e-300)
            e = (yy - yy.mean()) - b * xx
            resid[t, i] = float(np.std(e, ddof=1))
    return resid


def amihud_illiquidity_change(returns, dollar_volume, window: int = 21,
                              baseline: int = 252):
    """Change in Amihud illiquidity (|return| / dollar volume) vs its own baseline.

    Mechanism: the *level* of illiquidity is a well-known risk premium and is
    priced. The *change* is a flow signal -- a name becoming harder to trade is
    a name where someone is working a large order, and the price impact of that
    order is not yet complete.

    Why it may not be arbed: the level is in every commercial factor model and
    the change is not, because the change is noisy at the single-name level and
    only works in aggregate across many names. It also requires clean volume
    data, which is more annoying than it sounds once you handle splits, halts
    and consolidated-tape quirks.

    Kill criteria: subsumed by short-term reversal; or the entire effect is in
    the bottom liquidity decile, where you cannot trade it.
    """
    r = np.abs(np.asarray(returns, dtype=float))
    dv = np.asarray(dollar_volume, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        illiq = np.where(dv > 0, r / dv, np.nan)
    recent = _roll_mean(illiq, window)
    base = _roll_mean(illiq, baseline, min_periods=baseline // 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((base > 0) & np.isfinite(recent), np.log(recent / base), np.nan)
    return out
