"""Fama-MacBeth: does the signal survive next to what you already own?

Run a cross-sectional regression of forward returns on the signals each date,
then treat the sequence of slope coefficients as the sample. The t-stat of the
mean slope is the statistic of interest, and it is computed with HAC standard
errors because the slope series inherits the overlap of the forward return.

This is the right multivariate test for cross-sectional alpha, and it is
*different* from pooling all the observations into one big regression. The
pooled version treats 500 names on the same day as 500 independent draws, which
they are emphatically not -- they share that day's market move. Fama-MacBeth
gets the standard errors right by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qr.ic import newey_west_se


@dataclass
class FamaMacBeth:
    names: list[str]
    coef: np.ndarray            # (K,) time-series mean of the per-date slopes
    se: np.ndarray              # (K,) Newey-West
    t_stat: np.ndarray          # (K,)
    coef_series: np.ndarray     # (T, K) the per-date slopes themselves
    n_dates: int
    mean_r2: float
    lags: int

    def __str__(self) -> str:
        w = max(len(n) for n in self.names) if self.names else 8
        lines = [f"Fama-MacBeth  ({self.n_dates} dates, mean R2 {self.mean_r2:.4f}, "
                 f"HAC lags {self.lags})"]
        for i, n in enumerate(self.names):
            lines.append(f"  {n:<{w}}  coef {self.coef[i]:+.5f}  "
                         f"se {self.se[i]:.5f}  t {self.t_stat[i]:+.2f}")
        return "\n".join(lines)


def fama_macbeth(fwd, signals, mask=None, names=None, horizon: int | None = None,
                 add_intercept: bool = True, min_names: int = 20) -> FamaMacBeth:
    """Per-date cross-sectional OLS of ``fwd`` on ``signals``.

    ``signals`` is a (T, N, K) array or a list of K (T, N) arrays. Standardise
    them first (``rank_normal``) if you want the coefficients comparable.
    """
    y = np.asarray(fwd, dtype=float)
    if isinstance(signals, (list, tuple)):
        X = np.stack([np.asarray(s, dtype=float) for s in signals], axis=-1)
    else:
        X = np.asarray(signals, dtype=float)
        if X.ndim == 2:
            X = X[:, :, None]
    if X.shape[:2] != y.shape:
        raise ValueError(f"signals shape {X.shape[:2]} != fwd shape {y.shape}")
    T, N, K = X.shape
    if names is None:
        names = [f"x{i}" for i in range(K)]
    if len(names) != K:
        raise ValueError(f"got {len(names)} names for {K} signals")

    m = np.isfinite(y) & np.isfinite(X).all(axis=2)
    if mask is not None:
        m &= np.asarray(mask, dtype=bool)

    ncoef = K + (1 if add_intercept else 0)
    coefs = np.full((T, ncoef), np.nan)
    r2s = np.full(T, np.nan)
    for t in range(T):
        sel = m[t]
        n = int(sel.sum())
        if n < max(min_names, ncoef + 2):
            continue
        A = X[t, sel]
        if add_intercept:
            A = np.hstack([np.ones((n, 1)), A])
        b = y[t, sel]
        try:
            c, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        coefs[t] = c
        resid = b - A @ c
        ss_tot = float(((b - b.mean()) ** 2).sum())
        if ss_tot > 0:
            r2s[t] = 1.0 - float((resid ** 2).sum()) / ss_tot

    keep = slice(1, None) if add_intercept else slice(None)
    series = coefs[:, keep]
    ok = np.isfinite(series).all(axis=1)
    lags = None if horizon is None else max(int(horizon) - 1, 0)
    means = np.array([np.nanmean(series[ok, k]) if ok.any() else np.nan for k in range(K)])
    ses = np.array([newey_west_se(series[ok, k], lags) for k in range(K)])
    with np.errstate(divide="ignore", invalid="ignore"):
        ts = np.where(ses > 0, means / ses, np.nan)
    n_used = int(ok.sum())
    eff_lags = lags if lags is not None else (
        int(np.floor(4.0 * (max(n_used, 1) / 100.0) ** (2.0 / 9.0))))
    return FamaMacBeth(list(names), means, ses, ts, series, n_used,
                       float(np.nanmean(r2s)) if np.isfinite(r2s).any() else np.nan,
                       int(eff_lags))
