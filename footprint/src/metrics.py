"""Performance and predictive-power metrics.

Two families here, and they answer different questions. The IC family asks
whether a signal ranks names correctly; the P&L family asks whether that ranking
survives costs. A signal can pass the first and fail the second, which is the
usual outcome and the reason ``breakeven_cost_bps`` is reported for every feature
rather than only a headline Sharpe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------- rank agreement


def cross_sectional_ic(signal: pd.DataFrame, forward_returns: pd.DataFrame,
                       *, method: str = "spearman", min_names: int = 5) -> pd.Series:
    """Per-date rank correlation between a signal and forward returns.

    Spearman by default: the brief is to predict cross-sectional *ranks*, and
    rank correlation is robust to the fat-tailed return distribution that makes
    Pearson IC on raw returns a hostage to one or two names per day.
    """
    sig, fwd = signal.align(forward_returns, join="inner")
    out = {}
    for date in sig.index:
        a = sig.loc[date].to_numpy(dtype=float)
        b = fwd.loc[date].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < min_names:
            continue
        x, y = a[ok], b[ok]
        # A constant cross-section has no ranking to score.
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            continue
        r = (stats.spearmanr(x, y).statistic if method == "spearman"
             else stats.pearsonr(x, y).statistic)
        if np.isfinite(r):
            out[date] = float(r)
    return pd.Series(out, name="ic").sort_index()


@dataclass
class ICSummary:
    n_days: int
    mean_ic: float
    std_ic: float
    t_stat: float
    ir: float
    hit_rate: float
    p_value: float

    def as_dict(self) -> dict:
        return asdict(self)


def summarise_ic(ic: pd.Series, periods_per_year: int = 252) -> ICSummary:
    """Mean IC with the t-statistic that decides whether it is real.

    The t-statistic treats daily ICs as independent, which overstates
    significance when the signal is slow-moving and forward windows overlap --
    a 10-day forward return sampled daily has roughly 10x autocorrelation in its
    IC series. ``ic_newey_west_t`` corrects for exactly that and should be
    preferred whenever the horizon exceeds one day.
    """
    ic = pd.Series(ic).dropna()
    n = len(ic)
    if n < 2:
        return ICSummary(n, float("nan"), float("nan"), float("nan"),
                         float("nan"), float("nan"), float("nan"))
    mean, sd = float(ic.mean()), float(ic.std(ddof=1))
    t = mean / (sd / np.sqrt(n)) if sd > 0 else float("nan")
    p = float(2 * (1 - stats.t.cdf(abs(t), df=n - 1))) if np.isfinite(t) else float("nan")
    return ICSummary(
        n_days=n, mean_ic=mean, std_ic=sd, t_stat=float(t),
        ir=float(mean / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        hit_rate=float((ic > 0).mean()), p_value=p,
    )


def ic_newey_west_t(ic: pd.Series, lags: int | None = None) -> float:
    """t-statistic on the mean IC, robust to the overlap induced autocorrelation.

    With a k-day forward return sampled every day, consecutive ICs share k-1 days
    of their window and are mechanically correlated; the naive t-statistic then
    reports significance that is not there. Newey-West with ``lags`` set to the
    overlap length is the standard fix.
    """
    x = pd.Series(ic).dropna().to_numpy(dtype=float)
    n = x.size
    if n < 3:
        return float("nan")
    lags = int(lags if lags is not None else np.floor(4 * (n / 100) ** (2 / 9)))
    lags = max(0, min(lags, n - 2))
    e = x - x.mean()
    gamma0 = float(e @ e / n)
    var = gamma0
    for L in range(1, lags + 1):
        cov = float(e[L:] @ e[:-L] / n)
        var += 2.0 * (1.0 - L / (lags + 1.0)) * cov       # Bartlett kernel
    var = max(var, 1e-30)
    return float(x.mean() / np.sqrt(var / n))


# --------------------------------------------------------------- P&L metrics


@dataclass
class PerfSummary:
    n_days: int
    total_return: float
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    skew: float
    kurtosis: float
    mean_turnover: float
    mean_gross: float
    mean_net: float
    breakeven_cost_bps: float

    def as_dict(self) -> dict:
        return asdict(self)


def max_drawdown(equity: pd.Series) -> float:
    eq = pd.Series(equity).dropna()
    if eq.empty:
        return float("nan")
    peak = eq.cummax()
    return float((eq / peak - 1.0).min())


def breakeven_cost_bps(gross_returns: pd.Series, turnover: pd.Series) -> float:
    """One-way cost, in bps, at which the gross edge is exactly consumed.

    ``mean gross return per day / mean turnover per day``, converted to bps. This
    is the single most useful number for deciding whether a feature is tradable:
    it is directly comparable to a commission schedule, and unlike Sharpe it does
    not improve by trading less unless the edge per trade actually improves.
    """
    g, t = pd.Series(gross_returns).align(pd.Series(turnover), join="inner")
    g, t = g.dropna(), t.reindex(g.index).fillna(0.0)
    mt = float(t.mean())
    if mt <= 0:
        return float("inf") if float(g.mean()) > 0 else float("nan")
    return float(g.mean() / mt * 1e4)


def summarise_performance(returns: pd.Series, *, turnover=None, gross=None,
                          net=None, periods_per_year: int = 252,
                          gross_returns=None) -> PerfSummary:
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 2:
        nan = float("nan")
        return PerfSummary(n, nan, nan, nan, nan, nan, nan, nan, nan, nan,
                           nan, nan, nan, nan)
    equity = (1.0 + r).cumprod()
    mean, sd = float(r.mean()), float(r.std(ddof=1))
    ann_ret = float(mean * periods_per_year)
    ann_vol = float(sd * np.sqrt(periods_per_year))
    mdd = max_drawdown(equity)
    turn = pd.Series(turnover).dropna() if turnover is not None else pd.Series(dtype=float)
    be = (breakeven_cost_bps(gross_returns if gross_returns is not None else r, turn)
          if not turn.empty else float("nan"))
    return PerfSummary(
        n_days=n,
        total_return=float(equity.iloc[-1] - 1.0),
        ann_return=ann_ret,
        ann_vol=ann_vol,
        sharpe=float(ann_ret / ann_vol) if ann_vol > 0 else float("nan"),
        max_drawdown=mdd,
        calmar=float(ann_ret / abs(mdd)) if mdd and mdd < 0 else float("nan"),
        hit_rate=float((r > 0).mean()),
        skew=float(stats.skew(r)) if n > 2 else float("nan"),
        kurtosis=float(stats.kurtosis(r)) if n > 3 else float("nan"),
        mean_turnover=float(turn.mean()) if not turn.empty else float("nan"),
        mean_gross=float(pd.Series(gross).mean()) if gross is not None else float("nan"),
        mean_net=float(pd.Series(net).mean()) if net is not None else float("nan"),
        breakeven_cost_bps=be,
    )


def deflated_sharpe(sharpe: float, n_trials: int, n_obs: int) -> float:
    """Sharpe adjusted for how many variants were tried to find it.

    The expected maximum of ``n_trials`` independent zero-skill Sharpe estimates
    grows like sqrt(2 log n_trials)/sqrt(T). Reporting the raw maximum of a sweep
    without this correction is the most common way a backtest lies, and this
    project runs sweeps, so the correction is available and used.
    """
    if n_trials < 1 or n_obs < 2 or not np.isfinite(sharpe):
        return float("nan")
    euler = 0.5772156649
    if n_trials == 1:
        expected_max = 0.0
    else:
        z1 = stats.norm.ppf(1 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        expected_max = (1 - euler) * z1 + euler * z2
    return float((sharpe * np.sqrt(n_obs) - expected_max) / np.sqrt(n_obs))
