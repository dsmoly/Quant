"""Confidence intervals that respect serial dependence.

Strategy returns are autocorrelated, skewed and fat-tailed. Every one of those
breaks the textbook Sharpe standard error, and they break it in the optimistic
direction. Two tools here:

**Stationary bootstrap** (Politis-Romano 1994). Resamples geometric-length
blocks, so the resampled series preserves short-range dependence. Use it for
any statistic of a return series -- Sharpe, drawdown, hit rate.

**Lo's annualisation factor** (Lo 2002). The sqrt(q) scaling of a Sharpe ratio
is only correct for i.i.d. returns. With positive autocorrelation it
*overstates* the annualised Sharpe -- 10% at rho=0.1, 20% at rho=0.2. Momentum
and trend books have exactly that, so the correction is not optional for the
strategies this harness is aimed at.
"""

from __future__ import annotations

import numpy as np


def stationary_bootstrap(x, n_boot: int = 2000, mean_block: float = 20.0,
                         statistic=None, seed: int | None = 0):
    """Politis-Romano stationary bootstrap.

    Returns the bootstrap distribution of ``statistic`` (default: the mean).
    Block lengths are Geometric(1/mean_block) and the series wraps, which is
    what makes the resampled series stationary.
    """
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    T = v.size
    if T < 5:
        return np.array([])
    if statistic is None:
        statistic = np.mean
    rng = np.random.default_rng(seed)
    p = 1.0 / max(mean_block, 1.0)

    idx = np.empty((n_boot, T), dtype=np.int64)
    starts = rng.integers(0, T, size=(n_boot, T))
    newblock = rng.random((n_boot, T)) < p
    newblock[:, 0] = True
    cur = np.zeros(n_boot, dtype=np.int64)
    for t in range(T):
        cur = np.where(newblock[:, t], starts[:, t], (cur + 1) % T)
        idx[:, t] = cur
    samples = v[idx]
    return np.array([statistic(row) for row in samples])


def lo_annualisation_factor(returns, q: int) -> float:
    """Lo (2002) correction to the sqrt(q) Sharpe scaling.

    Returns the factor to multiply a per-period Sharpe by, in place of sqrt(q):

        q / sqrt(q + 2 * sum_{k=1}^{q-1} (q-k) * rho_k)

    Equals sqrt(q) exactly when the returns are serially uncorrelated.
    """
    v = np.asarray(returns, dtype=float)
    v = v[np.isfinite(v)]
    T = v.size
    q = int(q)
    if T < 10 or q < 2:
        return float(np.sqrt(max(q, 1)))
    e = v - v.mean()
    denom = float(e @ e)
    if denom <= 0:
        return float(np.sqrt(q))
    total = float(q)
    for k in range(1, min(q, T - 1)):
        rho = float(e[k:] @ e[:-k]) / denom
        total += 2.0 * (q - k) * rho
    if total <= 0:
        return float("nan")
    return float(q / np.sqrt(total))


def sharpe_ci(returns, periods_per_year: int = 252, n_boot: int = 2000,
              mean_block: float = 20.0, alpha: float = 0.05,
              use_lo: bool = True, seed: int | None = 0) -> dict:
    """Annualised Sharpe with a block-bootstrap CI and Lo's autocorrelation fix.

    ``naive_annualised`` is what a spreadsheet would report; the difference
    between it and ``annualised`` is the autocorrelation you were about to be
    paid for twice.
    """
    v = np.asarray(returns, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 10:
        return {k: float("nan") for k in
                ("sharpe_per_period", "annualised", "naive_annualised",
                 "lo_factor", "ci_low", "ci_high", "n_obs")}

    def sr(a):
        s = a.std(ddof=1)
        return float(a.mean() / s) if s > 0 else 0.0

    per = sr(v)
    factor = (lo_annualisation_factor(v, periods_per_year) if use_lo
              else float(np.sqrt(periods_per_year)))
    dist = stationary_bootstrap(v, n_boot=n_boot, mean_block=mean_block,
                                statistic=sr, seed=seed)
    lo_q, hi_q = (np.quantile(dist, [alpha / 2, 1 - alpha / 2])
                  if dist.size else (np.nan, np.nan))
    return {
        "sharpe_per_period": per,
        "annualised": per * factor,
        "naive_annualised": per * float(np.sqrt(periods_per_year)),
        "lo_factor": factor,
        "ci_low": float(lo_q * factor),
        "ci_high": float(hi_q * factor),
        "n_obs": int(v.size),
    }
