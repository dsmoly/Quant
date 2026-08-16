"""Effective breadth from the correlation eigenstructure.

The fundamental law of active management says IR ~ IC sqrt(N), and it is wrong in
exactly the way that matters here: N is the number of *independent* bets, not the
number of names. US equities share a dominant market factor and strong sector
blocks, so twenty names is nowhere near twenty bets, and quoting IC*sqrt(20) will
overstate the achievable information ratio badly.

Three different numbers get conflated in practice, so this module keeps them
apart:

  raw N            the count of names. Never the right number.

  participation    ``(sum lambda)^2 / sum lambda^2`` over the eigenvalues of the
  ratio            correlation matrix. This is the inverse Herfindahl of the
                   eigenvalue spectrum -- the number of eigen-directions carrying
                   comparable variance. It equals N for a perfectly diagonal
                   correlation matrix and falls to 1 when one factor explains
                   everything. This is the default.

  Marchenko-Pastur the count of eigenvalues above the upper edge of the
  count            random-matrix null, ``(1 + sqrt(N/T))^2``. This answers a
                   different question -- how many factors are statistically
                   distinguishable from noise given T observations -- and is
                   usually much smaller. It is the right number for "how many
                   factors should I model", not for "how many bets do I have".

Which one to use depends on what the portfolio actually holds, and there is a
subtlety worth being explicit about: a **dollar-neutral cross-sectional portfolio
has already projected out most of the first principal component**. Applying a
breadth haircut computed on the *raw* correlation matrix would therefore
double-count the market factor -- penalising the strategy for a risk it has
already neutralised. The haircut belongs on the residual correlation, after the
dominant factor is removed, which is what ``residualise_covariance`` is for and
what ``SizingEngine`` calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_matrix(x) -> np.ndarray:
    m = x.to_numpy(dtype=float) if isinstance(x, pd.DataFrame) else np.asarray(x, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("expected a square matrix")
    return m


def cov_to_corr(cov) -> np.ndarray:
    cov = _to_matrix(cov)
    d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    corr = cov / np.outer(d, d)
    return np.clip(np.nan_to_num(corr, nan=0.0), -1.0, 1.0)


def eigen_spectrum(matrix) -> np.ndarray:
    """Eigenvalues of a symmetric matrix, descending, clipped at zero."""
    m = _to_matrix(matrix)
    m = 0.5 * (m + m.T)                              # enforce symmetry
    vals = np.linalg.eigvalsh(m)
    return np.clip(vals[::-1], 0.0, None)


def variance_explained(matrix) -> np.ndarray:
    vals = eigen_spectrum(matrix)
    total = vals.sum()
    return vals / total if total > 0 else vals


def effective_n_participation(matrix) -> float:
    """(sum lambda)^2 / sum lambda^2 -- the participation ratio of the spectrum.

    Scale-invariant, so it can be handed a covariance or a correlation matrix.
    Returns 1.0 for a degenerate (single-factor) matrix and N for a diagonal one.
    """
    vals = eigen_spectrum(matrix)
    denom = float((vals ** 2).sum())
    if denom <= 0:
        return 0.0
    return float(vals.sum() ** 2 / denom)


def effective_n_mp(matrix, n_obs: int) -> int:
    """Eigenvalues above the Marchenko-Pastur upper edge, i.e. above noise.

    With N series and T observations the null spectrum for pure noise tops out at
    ``(1 + sqrt(N/T))^2`` (on a correlation matrix, whose eigenvalues average 1).
    """
    corr = cov_to_corr(matrix)
    n = corr.shape[0]
    if n_obs <= 0:
        return 0
    edge = (1.0 + np.sqrt(n / n_obs)) ** 2
    return int((eigen_spectrum(corr) > edge).sum())


def residualise_covariance(cov, n_factors: int = 1) -> np.ndarray:
    """Remove the leading ``n_factors`` principal components from a covariance.

    This is the covariance of what is left after the dominant common moves are
    stripped out -- the risk a dollar-neutral book is actually exposed to. The
    result is positive semi-definite by construction (it is the same matrix with
    the top eigenvalues set to zero).
    """
    m = _to_matrix(cov)
    m = 0.5 * (m + m.T)
    if n_factors <= 0 or m.shape[0] <= n_factors:
        return m
    vals, vecs = np.linalg.eigh(m)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.clip(vals, 0.0, None)
    vals[:n_factors] = 0.0
    return (vecs * vals) @ vecs.T


def factor_share(cov, n_factors: int = 1) -> float:
    """Fraction of total variance carried by the leading factors."""
    ve = variance_explained(cov)
    return float(ve[:n_factors].sum()) if ve.size else 0.0


def breadth_report(returns: pd.DataFrame, *, n_factors: int = 1) -> dict:
    """Full correlation-structure summary for a block of returns.

    Reports both the raw and residual pictures, because the difference between
    them is the whole point: the raw number says how correlated the universe is,
    the residual number says how many bets a neutral book actually has.
    """
    r = returns.dropna(axis=1, how="all")
    r = r.loc[:, r.notna().sum() >= 3]
    if r.shape[1] < 2:
        return {"n_names": int(r.shape[1]), "n_obs": int(r.shape[0]),
                "n_eff_raw": float(r.shape[1]), "n_eff_residual": float(r.shape[1]),
                "n_eff_mp": 0, "pc1_share": float("nan"),
                "mean_pairwise_corr": float("nan"), "variance_explained": []}
    x = r.to_numpy(dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    cov = pd.DataFrame(x).cov().to_numpy()
    corr = cov_to_corr(cov)
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    resid = residualise_covariance(corr, n_factors=n_factors)
    return {
        "n_names": int(r.shape[1]),
        "n_obs": int(r.shape[0]),
        "mean_pairwise_corr": float(np.nanmean(off)),
        "pc1_share": factor_share(corr, 1),
        "n_eff_raw": effective_n_participation(corr),
        "n_eff_residual": effective_n_participation(resid),
        "n_eff_mp": effective_n_mp(corr, r.shape[0]),
        "variance_explained": variance_explained(corr)[:5].round(4).tolist(),
    }


def fundamental_law_ir(ic: float, n_eff: float, periods_per_year: int = 252) -> float:
    """IR = IC sqrt(N_eff x periods), with N_eff supplied rather than assumed.

    Kept as a named function precisely so that the naive version -- passing the
    raw name count -- has to be written out explicitly by whoever wants it.
    """
    if n_eff <= 0:
        return 0.0
    return float(ic * np.sqrt(max(n_eff, 0.0) * periods_per_year))
