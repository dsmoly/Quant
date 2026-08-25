"""Tail risk without pretending to a precision the data cannot support.

The brief is explicit that a naive Hill estimator will not do, and it is right.
Hill is consistent only in the limit, it requires choosing the number of order
statistics k, and its estimate moves substantially with that choice -- so a
single reported "tail index = 3.2" is a statement about the analyst's choice of k
at least as much as about the data. Worse, it is biased in finite samples in a
direction that depends on the second-order behaviour of the tail, which is not
identified from a few hundred observations.

What this module does instead, in increasing order of assumption:

  expected_shortfall     the mean loss beyond a quantile. Non-parametric,
                         assumption-free, directly interpretable, and the number
                         to lead with. Its weakness is that it is an average over
                         very few observations, so its own standard error is
                         large; that error is reported alongside it.

  realized_semivariance  variance computed from downside moves only. Cheap,
                         stable, and it captures the asymmetry that a symmetric
                         volatility number hides.

  hill_plot              the *full* curve of Hill estimates against k, not a
                         point. A tail index is reported as a range across the
                         plateau region, with the spread as the honest measure of
                         how badly determined it is.

  fit_gpd                a generalised Pareto fit to exceedances, with the
                         threshold chosen by a stated rule rather than by eye,
                         and goodness-of-fit reported so a bad fit is visible.

The order matters. If the expected shortfall and the GPD tail quantile disagree,
believe the expected shortfall and distrust the fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def _losses(x) -> np.ndarray:
    """Positive-oriented losses from a return series (a 5% loss becomes +0.05)."""
    a = np.asarray(pd.Series(x).dropna(), dtype=float)
    return -a[np.isfinite(a)]


# ------------------------------------------------------- non-parametric first


def value_at_risk(returns, q: float = 0.05) -> float:
    """Historical VaR at level q, reported as a positive loss."""
    losses = _losses(returns)
    if losses.size == 0:
        return float("nan")
    return float(np.quantile(losses, 1.0 - q))


def expected_shortfall(returns, q: float = 0.05) -> float:
    """Mean loss conditional on breaching the q-quantile. Positive = a loss."""
    losses = _losses(returns)
    if losses.size == 0:
        return float("nan")
    var = np.quantile(losses, 1.0 - q)
    tail = losses[losses >= var]
    return float(tail.mean()) if tail.size else float(var)


def expected_shortfall_se(returns, q: float = 0.05, n_boot: int = 500,
                          seed: int = 0) -> float:
    """Bootstrap standard error of the ES estimate.

    Reported because ES at 5% on 1000 observations averages ~50 points, and a
    number built from 50 observations of a fat-tailed variable deserves an error
    bar attached to it every time it is quoted.
    """
    losses = _losses(returns)
    n = losses.size
    if n < 20:
        return float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(losses, size=n, replace=True)
        var = np.quantile(s, 1.0 - q)
        tail = s[s >= var]
        draws[b] = tail.mean() if tail.size else var
    return float(draws.std(ddof=1))


def realized_semivariance(returns, threshold: float = 0.0,
                          annualise: int | None = 252) -> float:
    """Variance from downside moves only, optionally annualised as a volatility.

    Barndorff-Nielsen's realised semivariance: the sum of squared negative
    returns. Returned as a volatility when annualised, to be comparable with an
    ordinary vol number -- for a symmetric distribution it lands at 1/sqrt(2) of
    total vol, so a ratio above that is evidence of genuine downside asymmetry.
    """
    a = np.asarray(pd.Series(returns).dropna(), dtype=float)
    down = a[a < threshold]
    if a.size == 0:
        return float("nan")
    semivar = float((down ** 2).sum() / a.size)
    if annualise:
        return float(np.sqrt(semivar * annualise))
    return semivar


def downside_ratio(returns) -> float:
    """Downside vol over total vol; 1/sqrt(2) ~ 0.707 for a symmetric law."""
    a = np.asarray(pd.Series(returns).dropna(), dtype=float)
    sd = a.std(ddof=1)
    if sd <= 0 or a.size == 0:
        return float("nan")
    semi = np.sqrt((a[a < 0] ** 2).sum() / a.size)
    return float(semi / sd)


# ------------------------------------------------------------------ Hill plot


@dataclass
class HillPlot:
    k: np.ndarray
    alpha: np.ndarray
    se: np.ndarray
    plateau_lo: float
    plateau_hi: float
    plateau_k: tuple[int, int]

    def as_dict(self) -> dict:
        return {"k": self.k.tolist(), "alpha": self.alpha.tolist(),
                "se": self.se.tolist(), "plateau_lo": self.plateau_lo,
                "plateau_hi": self.plateau_hi, "plateau_k": list(self.plateau_k)}

    def summary(self) -> str:
        return (f"tail index in [{self.plateau_lo:.2f}, {self.plateau_hi:.2f}] "
                f"over k in {self.plateau_k}")


def hill_plot(returns, k_min: int = 10, k_max: int | None = None) -> HillPlot:
    """Hill estimates across all k, with a plateau range instead of a point.

    The Hill estimator of the tail index alpha, using the k largest losses, is

        1/alpha_hat = (1/k) sum_{i=1..k} log(X_(i) / X_(k+1))

    Its standard error is alpha/sqrt(k). Small k means low bias and high
    variance; large k drags in observations from the body and biases the estimate
    toward the wrong regime. There is no way to pick k from the data without
    assumptions, so this returns the whole curve and summarises the flattest
    stretch of it -- the region where the estimate is least sensitive to the
    choice -- as a *range*.

    Read the range, not its midpoint. If it is wide, the tail index is not
    determined by this sample, which is the usual state of affairs for a few
    thousand daily returns.
    """
    losses = _losses(returns)
    losses = np.sort(losses[losses > 0])[::-1]
    n = losses.size
    k_max = int(min(k_max or n // 4, n - 2))
    if n < 50 or k_max <= k_min:
        nan = np.array([np.nan])
        return HillPlot(nan, nan, nan, float("nan"), float("nan"), (0, 0))

    ks = np.arange(k_min, k_max + 1)
    logs = np.log(losses)
    alphas = np.empty(ks.size)
    for i, k in enumerate(ks):
        inv = float(np.mean(logs[:k] - logs[k]))
        alphas[i] = 1.0 / inv if inv > 0 else np.nan
    ses = alphas / np.sqrt(ks)

    # Plateau: the window of k over which the estimate varies least, measured by
    # the rolling standard deviation of the curve itself.
    w = max(5, ks.size // 5)
    s = pd.Series(alphas)
    roll = s.rolling(w, center=True).std()
    if roll.notna().any():
        centre = int(roll.idxmin())
        lo_i, hi_i = max(0, centre - w // 2), min(ks.size - 1, centre + w // 2)
    else:
        lo_i, hi_i = 0, ks.size - 1
    seg = alphas[lo_i:hi_i + 1]
    seg = seg[np.isfinite(seg)]
    return HillPlot(
        k=ks, alpha=alphas, se=ses,
        plateau_lo=float(np.nanmin(seg)) if seg.size else float("nan"),
        plateau_hi=float(np.nanmax(seg)) if seg.size else float("nan"),
        plateau_k=(int(ks[lo_i]), int(ks[hi_i])),
    )


# ------------------------------------------------------------- GPD, carefully


@dataclass
class GPDFit:
    threshold: float
    n_exceed: int
    shape: float          # xi; > 0 is heavy-tailed, and 1/xi is the tail index
    scale: float
    ad_stat: float        # Anderson-Darling on the fitted exceedances
    ks_p: float
    converged: bool

    @property
    def tail_index(self) -> float:
        return float(1.0 / self.shape) if self.shape > 0 else float("inf")

    def var(self, q: float, n_total: int) -> float:
        """Tail VaR at level q implied by the fit."""
        if not self.converged or self.n_exceed == 0:
            return float("nan")
        zeta = self.n_exceed / n_total
        if q >= zeta:
            return float("nan")
        xi, beta = self.shape, self.scale
        if abs(xi) < 1e-8:
            return float(self.threshold + beta * np.log(zeta / q))
        return float(self.threshold + beta / xi * ((q / zeta) ** (-xi) - 1.0))

    def expected_shortfall(self, q: float, n_total: int) -> float:
        v = self.var(q, n_total)
        if not np.isfinite(v) or self.shape >= 1:
            return float("nan")       # ES is infinite for xi >= 1
        return float((v + self.scale - self.shape * self.threshold) / (1.0 - self.shape))


def fit_gpd(returns, threshold: float) -> GPDFit:
    """Maximum-likelihood GPD fit to losses above a threshold."""
    losses = _losses(returns)
    exceed = losses[losses > threshold] - threshold
    n = exceed.size
    if n < 25:
        return GPDFit(threshold, n, float("nan"), float("nan"),
                      float("nan"), float("nan"), False)
    try:
        shape, loc, scale = stats.genpareto.fit(exceed, floc=0.0)
    except Exception:
        return GPDFit(threshold, n, float("nan"), float("nan"),
                      float("nan"), float("nan"), False)
    if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
        return GPDFit(threshold, n, shape, scale, float("nan"), float("nan"), False)
    frozen = stats.genpareto(shape, loc=0.0, scale=scale)
    # Anderson-Darling against the fitted CDF, computed directly rather than by
    # sampling a comparison set: the sampled version is itself random, which is a
    # poor property for a statistic used to accept or reject a threshold.
    u = np.clip(np.sort(frozen.cdf(exceed)), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    ad = float(-n - np.mean((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1]))))
    ks_p = float(stats.kstest(exceed, frozen.cdf).pvalue)
    return GPDFit(threshold, n, float(shape), float(scale), ad, ks_p, True)


def select_threshold(returns, *, quantiles=None, min_exceed: int = 40) -> pd.DataFrame:
    """Fit the GPD across candidate thresholds and report the whole table.

    The principled part is that the threshold is chosen by a *stated rule* over a
    stated grid, and the full table is returned so the choice can be inspected:

      * stability -- xi should be roughly constant above the true threshold, so
        the selected one is where the shape estimate stops drifting;
      * goodness of fit -- a Kolmogorov-Smirnov p-value that collapses says the
        GPD does not describe those exceedances, whatever the shape says;
      * enough data -- at least ``min_exceed`` exceedances, or the fit is noise.

    ``recommended`` marks the row this rule selects. It is still a choice, and
    the table exists so that a reader can disagree with it.
    """
    quantiles = quantiles if quantiles is not None else np.arange(0.90, 0.995, 0.005)
    losses = _losses(returns)
    n_total = losses.size
    rows = []
    for q in quantiles:
        u = float(np.quantile(losses, q))
        fit = fit_gpd(returns, u)
        rows.append({"quantile": float(q), "threshold": u, "n_exceed": fit.n_exceed,
                     "shape": fit.shape, "scale": fit.scale, "ks_p": fit.ks_p,
                     "tail_index": fit.tail_index, "converged": fit.converged})
    table = pd.DataFrame(rows)
    ok = table[(table.n_exceed >= min_exceed) & table.converged & (table.ks_p > 0.05)]
    table["recommended"] = False
    if not ok.empty:
        # Among admissible thresholds, take the one whose shape is closest to the
        # local median -- i.e. sitting in the stable region rather than at an end.
        target = ok["shape"].median()
        pick = (ok["shape"] - target).abs().idxmin()
        table.loc[pick, "recommended"] = True
    table.attrs["n_total"] = n_total
    return table


def tail_report(returns, *, q: float = 0.05) -> dict:
    """Everything above, in the order it should be trusted."""
    r = pd.Series(returns).dropna()
    hp = hill_plot(r)
    thresholds = select_threshold(r)
    rec = thresholds[thresholds.recommended]
    gpd_row = rec.iloc[0].to_dict() if not rec.empty else None
    out = {
        "n_obs": int(len(r)),
        "expected_shortfall": expected_shortfall(r, q),
        "expected_shortfall_se": expected_shortfall_se(r, q),
        "value_at_risk": value_at_risk(r, q),
        "downside_vol": realized_semivariance(r),
        "downside_ratio": downside_ratio(r),
        "skew": float(stats.skew(r)) if len(r) > 2 else float("nan"),
        "excess_kurtosis": float(stats.kurtosis(r)) if len(r) > 3 else float("nan"),
        "hill_range": [hp.plateau_lo, hp.plateau_hi],
        "hill_k_range": list(hp.plateau_k),
        "gpd": gpd_row,
    }
    if gpd_row is not None:
        fit = fit_gpd(r, gpd_row["threshold"])
        out["gpd_var_01"] = fit.var(0.01, len(r))
        out["gpd_es_01"] = fit.expected_shortfall(0.01, len(r))
    return out
