"""Cross-sectional conditioning, applied per date.

Every function here takes a (T, N) array and a boolean mask and operates
*within each row*, never across time. That is the whole discipline: a
cross-sectional signal must be comparable across names on a given day, and
must not be quietly re-levered by a time-series transformation.

Masked-out cells come back as NaN, not zero. Zero is a position; NaN is an
absence of opinion, and the difference matters downstream.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _prep(x, mask):
    """Coerce to (T, N) + mask. A 1-D input is treated as a single
    cross-section and the result is squeezed back, so callers can hand in one
    date's worth of data without reshaping -- silently returning all-NaN for a
    1-D array was a live footgun."""
    x = np.asarray(x, dtype=float)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"expected a 1-D cross-section or a 2-D (T, N) panel, got {x.shape}")
    if mask is None:
        m = np.isfinite(x)
    else:
        m = np.asarray(mask, dtype=bool)
        if m.ndim == 1:
            m = m[None, :]
        m = m & np.isfinite(x)
    return x, m, squeeze


def _out(v, squeeze):
    return v[0] if squeeze else v


def winsorise(x, mask=None, lo: float = 0.01, hi: float = 0.99):
    """Clip each row to its own [lo, hi] quantiles.

    Applied before z-scoring, because a single 20-sigma print otherwise sets
    the scale for the whole cross-section and everything else collapses to zero.
    """
    x, mask, sq = _prep(x, mask)
    out = np.full_like(x, np.nan)
    for t in range(x.shape[0]):
        m = mask[t]
        if m.sum() < 3:
            continue
        v = x[t, m]
        a, b = np.quantile(v, [lo, hi])
        out[t, m] = np.clip(v, a, b)
    return _out(out, sq)


def zscore(x, mask=None, ddof: int = 0):
    """Row-wise standardisation over the masked names."""
    x, mask, sq = _prep(x, mask)
    out = np.full_like(x, np.nan)
    for t in range(x.shape[0]):
        m = mask[t]
        if m.sum() < 2:
            continue
        v = x[t, m]
        s = v.std(ddof=ddof)
        if s <= 0 or not np.isfinite(s):
            continue
        out[t, m] = (v - v.mean()) / s
    return _out(out, sq)


def rank_normal(x, mask=None):
    """Rank within each row, then map to normal scores via the inverse CDF.

    This is the default conditioning for the whole package. It is invariant to
    any monotone transformation of the raw feature, which means:

      * outliers cannot set the scale,
      * the feature's units and skew stop mattering,
      * and a monotone-but-nonlinear relationship becomes linear, which is
        exactly the regime where a linear model beats a flexible one.

    The cost is real: if the relationship is genuinely non-monotone (humped in
    idiosyncratic volatility, leverage, size), ranking destroys it and you need
    a flexible model on the *unranked* feature. Check the quantile report --
    that is what it is for.
    """
    x, mask, sq = _prep(x, mask)
    out = np.full_like(x, np.nan)
    for t in range(x.shape[0]):
        m = mask[t]
        n = int(m.sum())
        if n < 3:
            continue
        r = stats.rankdata(x[t, m], method="average")
        out[t, m] = stats.norm.ppf(r / (n + 1.0))
    return _out(out, sq)


def neutralise(x, factors, mask=None, add_intercept: bool = True):
    """Residual of a per-date cross-sectional OLS of ``x`` on ``factors``.

    ``factors`` is a (T, N, K) array or a list of K (T, N) arrays. This is the
    single most important step in evaluating a new signal: run it against the
    factors you already own before you believe anything. A signal that dies
    here was never new, and a signal that survives here is worth the rest of
    the pipeline.
    """
    x, mask, sq = _prep(x, mask)
    if isinstance(factors, (list, tuple)):
        fs = [np.asarray(f, dtype=float) for f in factors]
        fs = [f[None, :] if f.ndim == 1 else f for f in fs]
        F = np.stack(fs, axis=-1)
    else:
        F = np.asarray(factors, dtype=float)
        if F.ndim == 1:
            F = F[None, :, None]
        elif F.ndim == 2:
            F = F[:, :, None] if not sq else F[None, :, :]
    if F.shape[:2] != x.shape:
        raise ValueError(f"factors shape {F.shape[:2]} != x shape {x.shape}")
    T, N, K = F.shape
    out = np.full_like(x, np.nan)
    for t in range(T):
        m = mask[t] & np.isfinite(F[t]).all(axis=1)
        n = int(m.sum())
        if n < K + 2:
            continue
        A = F[t, m]
        if add_intercept:
            A = np.hstack([np.ones((n, 1)), A])
        y = x[t, m]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        out[t, m] = y - A @ coef
    return _out(out, sq)


def demean_by_group(x, groups, mask=None):
    """Subtract each group's own cross-sectional mean, per date.

    ``groups`` is a (T, N) integer array of group labels (industry, sector,
    asset class); negative labels are treated as ungrouped and left alone.
    Cheaper and more robust than dummy-variable neutralisation when the group
    count is large relative to the cross-section.
    """
    x, mask, sq = _prep(x, mask)
    g = np.asarray(groups)
    if g.ndim == 1:
        g = g[None, :]
    if g.shape != x.shape:
        raise ValueError(f"groups shape {g.shape} != x shape {x.shape}")
    out = np.full_like(x, np.nan)
    for t in range(x.shape[0]):
        m = mask[t]
        if m.sum() < 2:
            continue
        row, gr = x[t], g[t]
        vals = np.full(x.shape[1], np.nan)
        for lab in np.unique(gr[m]):
            if lab < 0:
                sel = m & (gr == lab)
                vals[sel] = row[sel]
                continue
            sel = m & (gr == lab)
            if sel.sum() >= 2:
                vals[sel] = row[sel] - row[sel].mean()
            elif sel.sum() == 1:
                vals[sel] = 0.0
        out[t] = vals
    return _out(out, sq)
