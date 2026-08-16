"""The cross-sectional backtest loop.

Execution convention, which is the thing most worth getting right
-----------------------------------------------------------------
A signal computed from the close of day t is executed at the close of day t+1 and
therefore earns the return from close t+1 to close t+2. In code, the position
held on day d is the decision made on day ``d - lag - 1``. There is a test that
asserts this by feeding in a signal built from *future* returns and checking the
backtest fails to profit from it; if the lag were wrong that test would print a
spectacular Sharpe, which is exactly the failure mode it exists to catch.

Why the loop is stateful
------------------------
The no-trade band compares the target against what is currently held, so the book
cannot be vectorised into a single shift: today's position depends on yesterday's,
which depended on the day before. The position the band is measured against is
the *previous decision*, since that is what will actually be on the books when
this decision reaches the market.

Turning a signal into an alpha
------------------------------
The sizing control needs ``alpha`` in return units, not a z-score. The standard
conversion is Grinold's:

    alpha_i = IC * sigma_i * z_i

where z is the cross-sectionally standardised signal and IC is the correlation
the signal actually achieves. Two consequences matter here. First, IC is
*estimated from trailing data*, never assumed -- and only from signal dates whose
forward windows closed before the decision date, or the backtest would be reading
its own future. Second, the standard error of that IC estimate flows straight
into the confidence shrinkage: t = IC/se(IC) is the signal's own t-statistic, so
a signal with no measurable edge is shrunk to zero and sized at zero. That is the
mechanism the dumb-signal experiment is built to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import cross_sectional_ic
from .sizing import SizingEngine, SizingParams, rolling_covariance


# ------------------------------------------------------------------ utilities


def rank_normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank, mapped to standard normal scores.

    Predicting ranks rather than raw returns is the brief, and rank-normalising
    the *signal* is the matching choice on the input side: it makes the sizing
    invariant to the feature's own scale and immune to the one wild print that
    would otherwise dominate a z-score.
    """
    ranks = frame.rank(axis=1, method="average")
    n = frame.notna().sum(axis=1)
    u = ranks.div(n + 1, axis=0)
    out = pd.DataFrame(stats.norm.ppf(u.to_numpy(dtype=float)),
                       index=frame.index, columns=frame.columns)
    return out.where(frame.notna())


def tradable_forward_returns(close: pd.DataFrame, horizon: int,
                             lag: int = 1) -> pd.DataFrame:
    """Return earned from executing a date-t signal, over ``horizon`` days.

    From close t+lag to close t+lag+horizon -- the window a signal observed at t
    can actually be traded into. Using close t to t+horizon instead would credit
    the strategy with a move it could not have captured, and is the single most
    common source of an accidentally spectacular backtest.
    """
    if horizon < 1 or lag < 0:
        raise ValueError("horizon must be >= 1 and lag >= 0")
    entry = close.shift(-lag)
    exit_ = close.shift(-(lag + horizon))
    return exit_ / entry - 1.0


def trailing_volatility(returns: pd.DataFrame, window: int = 60,
                        min_periods: int | None = None) -> pd.DataFrame:
    """Trailing return volatility, shifted so date t uses data strictly before t."""
    mp = min_periods or max(10, window // 3)
    return returns.rolling(window, min_periods=mp).std().shift(1)


class TrailingIC:
    """Estimates a signal's IC and the standard error of that estimate.

    Only ICs whose forward window closed strictly before the decision date are
    used. With a horizon of h and an execution lag of l, a signal generated on
    date s is not fully scored until s + l + h, so a decision on date t may use
    signal dates up to ``t - l - h`` and no later.
    """

    def __init__(self, ic: pd.Series, *, horizon: int, lag: int = 1,
                 window: int = 252, min_obs: int = 60):
        self.ic = pd.Series(ic).dropna().sort_index()
        self.horizon, self.lag = int(horizon), int(lag)
        self.window, self.min_obs = int(window), int(min_obs)

    def available(self, date) -> pd.Series:
        """The ICs legitimately observable by ``date``."""
        if self.ic.empty:
            return self.ic
        idx = self.ic.index
        # Position of the last signal date whose window has closed by `date`.
        cutoff_pos = idx.searchsorted(date, side="right") - 1 - (self.lag + self.horizon)
        if cutoff_pos < 0:
            return self.ic.iloc[:0]
        lo = max(0, cutoff_pos + 1 - self.window)
        return self.ic.iloc[lo:cutoff_pos + 1]

    def estimate(self, date) -> tuple[float, float]:
        """(IC estimate, standard error). Returns (0, inf) before it is measurable.

        Returning an infinite standard error rather than a missing value is
        deliberate: it flows through the shrinkage as a factor of exactly zero,
        so an unmeasured signal is sized at zero instead of being skipped by a
        special case somewhere else.
        """
        w = self.available(date)
        if len(w) < self.min_obs:
            return 0.0, float("inf")
        mean = float(w.mean())
        sd = float(w.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            return mean, float("inf")
        # Overlapping forward windows make consecutive ICs dependent; inflating
        # the standard error by sqrt(h) is the cheap Newey-West-equivalent
        # correction and keeps the shrinkage honest for slow signals.
        se = sd / np.sqrt(len(w)) * np.sqrt(max(self.horizon, 1))
        return mean, float(se)


def rebalance_mask(dates: pd.DatetimeIndex, rule: str) -> np.ndarray:
    """Which dates the strategy is allowed to change its book on."""
    rule = rule.upper()
    if rule in ("D", "DAILY"):
        return np.ones(len(dates), dtype=bool)
    if rule in ("W", "WEEKLY"):
        # Last observed session of each ISO week.
        s = pd.Series(np.arange(len(dates)), index=dates)
        keep = s.groupby([dates.isocalendar().year, dates.isocalendar().week]).max()
        mask = np.zeros(len(dates), dtype=bool)
        mask[keep.to_numpy()] = True
        return mask
    if rule in ("M", "MONTHLY"):
        s = pd.Series(np.arange(len(dates)), index=dates)
        keep = s.groupby([dates.year, dates.month]).max()
        mask = np.zeros(len(dates), dtype=bool)
        mask[keep.to_numpy()] = True
        return mask
    raise ValueError(f"unknown rebalance rule: {rule}")


# -------------------------------------------------------------------- config


@dataclass
class BacktestConfig:
    cost_bps: float = 20.0        # one-way, per unit of weight traded
    rebalance: str = "D"
    lag: int = 1                  # sessions between signal and execution
    horizon: int = 5              # forward horizon the alpha is aimed at
    vol_window: int = 60
    cov_window: int = 120
    ic_window: int = 252
    min_ic_obs: int = 60
    warmup: int = 0               # extra sessions to skip at the start
    sizing: SizingParams = field(default_factory=SizingParams)

    def __post_init__(self) -> None:
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")
        if self.lag < 0:
            raise ValueError("lag must be non-negative; a lag of 0 trades on the "
                             "same close the signal is computed from")


@dataclass
class BacktestResult:
    decisions: pd.DataFrame
    held: pd.DataFrame
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    costs: pd.Series
    diagnostics: pd.DataFrame
    ic_used: pd.Series
    config: BacktestConfig

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.net_returns.fillna(0.0)).cumprod()


# ---------------------------------------------------------------------- core


def run_backtest(*, signal: pd.DataFrame, close: pd.DataFrame,
                 config: BacktestConfig | None = None,
                 universe: pd.DataFrame | None = None,
                 engine: SizingEngine | None = None,
                 ic_series: pd.Series | None = None,
                 fixed_ic: float | None = None) -> BacktestResult:
    """Run the cross-sectional strategy over a panel.

    ``signal`` is a raw feature frame (date x ticker); it is rank-normalised
    internally. ``ic_series`` lets a caller supply ICs measured on a *training*
    period only, which is how the walk-forward avoids scoring a signal with
    knowledge from its own test window. ``fixed_ic`` bypasses estimation entirely
    and is used by the controlled experiments where the true IC is known.
    """
    cfg = config or BacktestConfig()
    engine = engine or SizingEngine(cfg.sizing)

    signal, close = signal.align(close, join="inner")
    dates = close.index
    tickers = list(close.columns)
    returns = close.pct_change()

    z = rank_normalise(signal)
    if universe is not None:
        univ = universe.reindex(index=dates, columns=tickers).fillna(False).to_numpy(bool)
    else:
        univ = np.isfinite(z.to_numpy(dtype=float))

    vol = trailing_volatility(returns, cfg.vol_window)
    covs = rolling_covariance(returns, cfg.cov_window)

    if fixed_ic is None:
        ic_in = (ic_series if ic_series is not None
                 else cross_sectional_ic(signal,
                                         tradable_forward_returns(close, cfg.horizon, cfg.lag)))
        ic_est = TrailingIC(ic_in, horizon=cfg.horizon, lag=cfg.lag,
                            window=cfg.ic_window, min_obs=cfg.min_ic_obs)
    else:
        ic_est = None

    reb = rebalance_mask(dates, cfg.rebalance)
    z_arr = z.to_numpy(dtype=float)
    vol_arr = vol.to_numpy(dtype=float)

    n = len(tickers)
    prev = np.zeros(n)
    decisions = np.full((len(dates), n), np.nan)
    diag_rows, ic_used = [], {}

    for i, date in enumerate(dates):
        if i < cfg.warmup or not reb[i]:
            decisions[i] = prev
            continue
        zi = z_arr[i]
        si = vol_arr[i]
        active = univ[i] & np.isfinite(zi) & np.isfinite(si) & (si > 0)
        if active.sum() < 3:
            decisions[i] = prev
            continue

        if ic_est is None:
            ic_hat, ic_se = float(fixed_ic), abs(float(fixed_ic)) / 3.0 or 1e-9
        else:
            ic_hat, ic_se = ic_est.estimate(date)
        ic_used[date] = ic_hat

        # Grinold: alpha = IC * sigma * z, with the IC's own error carried through
        # so that shrinkage responds to how well the signal is measured.
        alpha = np.where(active, ic_hat * si * zi, 0.0)
        alpha_se = np.where(active, ic_se * si * np.abs(zi), np.inf)

        cov = covs.get(date)
        if cov is None:
            decisions[i] = prev
            continue

        w, d = engine.size(alpha=alpha, sigma=np.where(active, si, np.nan), cov=cov,
                           current=prev, cost=cfg.cost_bps / 1e4,
                           alpha_se=alpha_se, active=active)
        decisions[i] = w
        prev = w
        diag_rows.append({"date": date, **d.__dict__, "ic_hat": ic_hat, "ic_se": ic_se})

    dec = pd.DataFrame(decisions, index=dates, columns=tickers)
    return _assemble(dec, returns, cfg,
                     pd.DataFrame(diag_rows).set_index("date") if diag_rows
                     else pd.DataFrame(),
                     pd.Series(ic_used, dtype=float))


def _assemble(dec: pd.DataFrame, returns: pd.DataFrame, cfg: BacktestConfig,
              diagnostics: pd.DataFrame, ic_used: pd.Series) -> BacktestResult:
    """Common accounting: apply the execution lag, then charge costs on turnover.

    Kept in one place so the engine-driven and quintile backtests cannot drift
    apart in how they book P&L -- a comparison between two arms is worthless if
    they account differently.
    """
    # The position held on day d is the decision made on d - lag - 1.
    held = dec.shift(cfg.lag + 1)
    live = held.notna().any(axis=1)
    gross = (held * returns).sum(axis=1, skipna=True).where(live)
    turnover = held.diff().abs().sum(axis=1).where(live)
    costs = turnover * (cfg.cost_bps / 1e4)
    return BacktestResult(
        decisions=dec, held=held, gross_returns=gross, net_returns=gross - costs,
        turnover=turnover, costs=costs, diagnostics=diagnostics,
        ic_used=ic_used, config=cfg)


def run_quintile_backtest(*, signal: pd.DataFrame, close: pd.DataFrame,
                          config: BacktestConfig | None = None,
                          universe: pd.DataFrame | None = None,
                          quantile: float = 0.2,
                          gross: float = 1.0) -> BacktestResult:
    """The naive construction: equal-weight long the top fifth, short the bottom.

    No volatility targeting, no confidence shrinkage, no no-trade band, no risk
    budgeting -- just the ranking. This exists to be the baseline the sizing
    engine is measured against, so that "the framework earns its place" is a
    measured claim rather than an assumed one. It is also the construction the
    conventional cross-sectional study uses, which makes it the right thing to
    beat.
    """
    cfg = config or BacktestConfig()
    signal, close = signal.align(close, join="inner")
    dates, tickers = close.index, list(close.columns)
    returns = close.pct_change()

    s = signal.copy()
    if universe is not None:
        s = s.where(universe.reindex(index=dates, columns=tickers).fillna(False))

    ranks = s.rank(axis=1, pct=True)
    n_ok = s.notna().sum(axis=1)
    longs = (ranks > 1 - quantile) & (n_ok.to_numpy()[:, None] >= 5)
    shorts = (ranks <= quantile) & (n_ok.to_numpy()[:, None] >= 5)

    nl = longs.sum(axis=1).replace(0, np.nan)
    ns = shorts.sum(axis=1).replace(0, np.nan)
    w = (longs.astype(float).div(nl, axis=0).fillna(0.0)
         - shorts.astype(float).div(ns, axis=0).fillna(0.0)) * (gross / 2.0)

    # Only change the book on rebalance dates; otherwise carry it forward.
    reb = rebalance_mask(dates, cfg.rebalance)
    w = w.where(pd.Series(reb, index=dates), other=np.nan).ffill().fillna(0.0)
    if cfg.warmup:
        w.iloc[:cfg.warmup] = 0.0
    return _assemble(w, returns, cfg, pd.DataFrame(), pd.Series(dtype=float))
