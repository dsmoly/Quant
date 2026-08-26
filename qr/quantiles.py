"""Quantile sorts: is the relationship monotone, or is one tail carrying it?

The IC is a single number and it hides two things that change what you should
do next:

1. **Non-monotonicity.** A humped relationship (idiosyncratic volatility,
   leverage, size, analyst coverage all have one) produces a near-zero IC while
   carrying real information. If the quantile means are U-shaped, stop ranking
   the feature and give a flexible model the raw version -- that is the one
   regime where a GBM genuinely beats a linear fit.

2. **Tail concentration.** A signal whose entire spread comes from the bottom
   decile is a short-selling strategy with a borrow problem, not a
   cross-sectional signal. The long-short decomposition here tells you which
   side you are actually being paid for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from qr.ic import newey_west_se


@dataclass
class QuantileReport:
    n_quantiles: int
    mean_return: np.ndarray        # (Q,) mean forward return per quantile
    se_return: np.ndarray          # (Q,) Newey-West SE of each
    spread: float                  # top minus bottom
    spread_t: float
    long_leg: float                # top minus cross-sectional mean
    short_leg: float               # cross-sectional mean minus bottom
    monotonicity: float            # Spearman of quantile index vs mean return
    series: np.ndarray             # (T, Q) per-date quantile means

    def __str__(self) -> str:
        legs = f"long {self.long_leg:+.4f} / short {self.short_leg:+.4f}"
        return (f"Q{self.n_quantiles} spread {self.spread:+.4f} (t {self.spread_t:+.2f})  "
                f"monotonicity {self.monotonicity:+.2f}  {legs}")


def quantile_report(signal, fwd, mask=None, n_quantiles: int = 5,
                    horizon: int | None = None, min_names: int = 20) -> QuantileReport:
    """Sort into quantiles per date and report mean forward return of each."""
    s = np.asarray(signal, dtype=float)
    f = np.asarray(fwd, dtype=float)
    if s.shape != f.shape:
        raise ValueError(f"signal shape {s.shape} != fwd shape {f.shape}")
    m = np.isfinite(s) & np.isfinite(f)
    if mask is not None:
        m &= np.asarray(mask, dtype=bool)

    T = s.shape[0]
    Q = int(n_quantiles)
    series = np.full((T, Q), np.nan)
    means = np.full(T, np.nan)
    for t in range(T):
        sel = m[t]
        n = int(sel.sum())
        if n < max(min_names, Q * 2):
            continue
        v, y = s[t, sel], f[t, sel]
        r = stats.rankdata(v, method="average")
        # edges chosen on rank, so ties cannot collapse a bucket
        q = np.clip(((r - 0.5) / n * Q).astype(int), 0, Q - 1)
        for k in range(Q):
            kk = q == k
            if kk.any():
                series[t, k] = float(y[kk].mean())
        means[t] = float(y.mean())

    ok = np.isfinite(series).all(axis=1)
    if ok.sum() < 3:
        nan = np.full(Q, np.nan)
        return QuantileReport(Q, nan, nan, np.nan, np.nan, np.nan, np.nan, np.nan, series)

    mr = series[ok].mean(axis=0)
    lags = None if horizon is None else max(int(horizon) - 1, 0)
    se = np.array([newey_west_se(series[ok, k], lags) for k in range(Q)])
    spread_series = series[ok, Q - 1] - series[ok, 0]
    spread_se = newey_west_se(spread_series, lags)
    cs_mean = float(np.nanmean(means[ok]))
    mono = float(stats.spearmanr(np.arange(Q), mr).statistic) if Q > 2 else np.nan
    return QuantileReport(
        n_quantiles=Q,
        mean_return=mr,
        se_return=se,
        spread=float(spread_series.mean()),
        spread_t=float(spread_series.mean() / spread_se) if spread_se > 0 else np.nan,
        long_leg=float(mr[Q - 1] - cs_mean),
        short_leg=float(cs_mean - mr[0]),
        monotonicity=mono,
        series=series,
    )
