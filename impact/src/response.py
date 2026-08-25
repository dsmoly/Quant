"""Response functions, information coefficients, and signal persistence.

R(h) = E[ p(t+h) - p(t) | signed order flow at t ]

Three different quantities hide behind that one line, and conflating them is the
mistake this module was rewritten to fix.

  measure='impact'      The classical response function of the propagator
                        literature. The price change is measured from **just
                        before** the flow event to h after it, so it includes the
                        contemporaneous move the flow itself caused. This is the
                        quantity whose asymptote is permanent impact, and it is
                        the one that answers "is impact transient or permanent".

  measure='predictive'  Trailing flow against the *forward* return, from the end
                        of the flow window onward. This is what a strategy would
                        actually earn, and it is a genuinely different number: it
                        excludes the move the flow already caused and therefore
                        measures only what is left to capture.

The distinction is not pedantic. With a transient component that mean-reverts,
the predictive response is **negative** at horizons past the relaxation time --
elevated price falls back -- while the impact response is positive and flat. A
first version of this file computed only the predictive version and reported
strongly negative R(h) for a tape with a large planted *positive* permanent
component. That was arithmetically correct and completely the wrong measurement.

  aggregation_s         How much flow is summed into one observation. At one bar
                        this is the single-event response. Aggregating over a
                        longer window is the case the permanent-impact argument
                        is about: summing flow averages away the transient part,
                        which is roughly mean-zero over a long window, and leaves
                        the information component.

Standard errors
---------------
Overlapping windows are used for the point estimate (they use all the data) with
Newey-West standard errors, and every number is cross-checked against a strictly
non-overlapping subsample. The cross-check is not decoration: the equity work in
this repo established that Newey-West under-corrects for persistent signals, and
order flow is very persistent. Where the two disagree, the non-overlapping
estimate is the one to believe, because it makes no assumption at all.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .flow import Bars, normalise_flow

YEAR_SECONDS = 365.25 * 24 * 3600


# ------------------------------------------------------------------ regression


def ols_nw(y: np.ndarray, x: np.ndarray, lags: int) -> tuple[float, float, int]:
    """Univariate OLS through the origin-free model y = a + b x, with HAC errors.

    Returns (b, se_b, n). Newey-West with a Bartlett kernel; ``lags`` should be at
    least the overlap length, since that is the dependence the estimator is there
    to absorb.
    """
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = y.size
    if n < 30 or np.allclose(x, x[0]):
        return float("nan"), float("nan"), int(n)
    X = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    u = X * resid[:, None]
    S = u.T @ u
    L = max(0, min(int(lags), n - 2))
    for l in range(1, L + 1):
        G = u[l:].T @ u[:-l]
        S += (1.0 - l / (L + 1.0)) * (G + G.T)      # Bartlett
    cov = xtx_inv @ S @ xtx_inv
    return float(beta[1]), float(np.sqrt(max(cov[1, 1], 0.0))), int(n)


# --------------------------------------------------------------- persistence


def autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return np.full(max_lag + 1, np.nan)
    x = x - x.mean()
    v = float(x @ x / n)
    if v <= 0:
        return np.full(max_lag + 1, np.nan)
    out = np.ones(max_lag + 1)
    for k in range(1, min(max_lag, n - 2) + 1):
        out[k] = float(x[k:] @ x[:-k] / n / v)
    return out


@dataclass
class Persistence:
    ac1: float
    half_life_bars: float
    half_life_s: float
    integrated_time_bars: float
    integrated_time_s: float
    independent_obs_per_year: float

    def as_dict(self) -> dict:
        return asdict(self)


def persistence(x: pd.Series, bar_seconds: float, max_lag: int = 200) -> Persistence:
    """Autocorrelation, half-life and independent-observation count for a signal.

    Two persistence measures, deliberately both:

      half-life    from the lag-1 autocorrelation, assuming AR(1). Easy to quote,
                   and wrong whenever the decay is not exponential -- which for
                   order flow it is not, the autocorrelation being famously
                   power-law.
      integrated   tau = 1 + 2 sum rho_k, truncated at the first non-positive rho.
      time         This is the quantity that actually sets how many independent
                   observations a sample contains, and it is much larger than the
                   half-life for a power-law decay.

    ``independent_obs_per_year`` is what breadth should be computed from. Using
    the rebalance frequency instead is what turns a real 1.5 Sharpe into a
    claimed 5: sampling a signal ten times inside its own correlation time gives
    ten correlated observations, not ten bets.
    """
    a = autocorr(pd.Series(x).dropna().to_numpy(dtype=float), max_lag)
    ac1 = float(a[1]) if a.size > 1 else float("nan")
    hl = (float(-np.log(2) / np.log(ac1)) if np.isfinite(ac1) and 0 < ac1 < 1
          else float("inf") if np.isfinite(ac1) and ac1 >= 1 else float("nan"))
    # Integrated time: sum positive autocorrelations until the first crossing.
    tau = 1.0
    for k in range(1, a.size):
        if not np.isfinite(a[k]) or a[k] <= 0:
            break
        tau += 2.0 * a[k]
    tau = float(max(tau, 1.0))
    return Persistence(
        ac1=ac1, half_life_bars=hl,
        half_life_s=hl * bar_seconds if np.isfinite(hl) else float("nan"),
        integrated_time_bars=tau, integrated_time_s=tau * bar_seconds,
        independent_obs_per_year=float(YEAR_SECONDS / (tau * bar_seconds)))


# ------------------------------------------------------------------- response


@dataclass
class HorizonResult:
    horizon_s: float
    measure: str
    aggregation_s: float
    price_kind: str
    n_obs: int
    n_nonoverlap: int
    r_bps: float              # response, bps per 1 sd of signed flow
    r_se_bps: float           # Newey-West
    r_t: float
    r_bps_nonoverlap: float
    r_se_bps_nonoverlap: float
    r_t_nonoverlap: float
    flow_sd: float            # sd of the aggregated flow, in raw units
    lambda_per_unit: float    # bps per *unit* of signed flow, not per sd
    ic: float
    ic_t_nw: float
    ic_nonoverlap: float
    ic_t_nonoverlap: float

    def as_dict(self) -> dict:
        return asdict(self)


def _log_bps(p: pd.Series) -> pd.Series:
    return np.log(p.astype(float)) * 1e4


def _spearman_t(x, y, deflate: float = 1.0):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30:
        return float("nan"), float("nan")
    r = float(stats.spearmanr(x[ok], y[ok]).statistic)
    if not np.isfinite(r):
        return float("nan"), float("nan")
    t = r * np.sqrt(max(int(ok.sum()) - 2, 1) / max(1 - r ** 2, 1e-12))
    return r, float(t / max(np.sqrt(deflate), 1.0))


def response_at_horizon(bars: Bars, horizon_s: float, *, measure: str = "impact",
                        aggregation_s: float | None = None,
                        price_kind: str = "auto",
                        flow_col: str = "signed_volume") -> HorizonResult:
    """Measure the response at one horizon.

    ``measure='impact'`` regresses ``p[t+k] - p[t-a]`` on the flow summed over
    bars ``(t-a, t]`` -- the price is measured from just before the flow arrived,
    so the contemporaneous impact is included.

    ``measure='predictive'`` regresses ``p[t+k] - p[t]`` on the same flow: only
    what remains after the flow has already moved the price.
    """
    if measure not in ("impact", "predictive", "contemporaneous"):
        raise ValueError("measure must be 'impact', 'predictive' or 'contemporaneous'")
    if price_kind == "mid" and not bars.has_mid:
        raise ValueError("no mid price available; pass price_kind='trade'")
    use_mid = price_kind == "mid" or (price_kind == "auto" and bars.has_mid)
    price = bars.mid if use_mid else bars.trade_price

    k = max(1, int(round(horizon_s / bars.seconds)))
    a = max(1, int(round((aggregation_s if aggregation_s is not None
                          else bars.seconds) / bars.seconds)))

    lp = _log_bps(price)
    raw = bars.frame()[flow_col]

    if measure == "contemporaneous":
        # Price change over the window regressed on the flow arriving *in that
        # same window*. This is the estimator the permanent-impact argument
        # actually implies, and it is far better conditioned than fitting an
        # asymptote to a humped curve. Over a window of length W the transient
        # contributes I(t+W) - I(t), which reflects flow at the two edges and
        # becomes uncorrelated with the window's total flow as W grows; the
        # permanent component is proportional to that total by construction. So
        # the coefficient converges to permanent impact from above, and the rate
        # of convergence is itself the transient timescale.
        y = lp.shift(-k) - lp
        # Reversed rolling sum gives the total over [t, t+k-1] at index t; the
        # shift moves it to (t, t+k], matching the price window exactly.
        flow = raw[::-1].rolling(k, min_periods=max(1, k // 4)).sum()[::-1].shift(-1)
    else:
        y = (lp.shift(-k) - lp.shift(a)) if measure == "impact" else (lp.shift(-k) - lp)
        flow = raw.rolling(a, min_periods=max(1, a // 4)).sum() if a > 1 else raw
    z = normalise_flow(flow)

    flow_sd = float(flow.std(ddof=1))
    yv, xv = y.to_numpy(dtype=float), z.to_numpy(dtype=float)
    # Overlap in the dependent variable spans a + k bars for the impact measure.
    span = k + (a if measure == "impact" else 0)
    if measure == "contemporaneous":
        span = k
    b, se, n = ols_nw(yv, xv, lags=span)
    ic, ic_t = _spearman_t(xv, yv, deflate=span)

    # Strictly non-overlapping: step by the full span so no two observations
    # share a single second of price path or a single trade of flow.
    sub = slice(None, None, max(span, 1))
    ys, xs = yv[sub], xv[sub]
    b2, se2, _ = ols_nw(ys, xs, lags=0)
    ic2, ic2_t = _spearman_t(xs, ys)
    n2 = int((np.isfinite(ys) & np.isfinite(xs)).sum())

    div = lambda num, den: (num / den if den and np.isfinite(den) and den > 0
                            else float("nan"))
    return HorizonResult(
        horizon_s=float(horizon_s), measure=measure,
        aggregation_s=float(a * bars.seconds), price_kind="mid" if use_mid else "trade",
        n_obs=n, n_nonoverlap=n2,
        r_bps=b, r_se_bps=se, r_t=div(b, se),
        r_bps_nonoverlap=b2, r_se_bps_nonoverlap=se2, r_t_nonoverlap=div(b2, se2),
        flow_sd=flow_sd, lambda_per_unit=div(b, flow_sd),
        ic=ic, ic_t_nw=ic_t, ic_nonoverlap=ic2, ic_t_nonoverlap=ic2_t)


MIN_NONOVERLAP = 30


def response_curve(bars: Bars, horizons_s, *, measure: str = "impact",
                   aggregation_s: float | None = None, price_kind: str = "auto",
                   flow_col: str = "signed_volume",
                   min_nonoverlap: int = MIN_NONOVERLAP) -> pd.DataFrame:
    """Response across horizons, dropping any the sample cannot support.

    A horizon is dropped when it leaves fewer than ``min_nonoverlap`` independent
    windows. This is not tidying -- with eight non-overlapping windows the point
    estimate is meaningless and, worse, it anchors the asymptotic fit. Dropping
    it is how the fit avoids being driven by the noisiest point on the curve.
    """
    n_bars = len(bars.trade_price)
    rows = []
    for h in horizons_s:
        k = max(1, int(round(h / bars.seconds)))
        a = max(1, int(round((aggregation_s if aggregation_s is not None
                              else bars.seconds) / bars.seconds)))
        span = k + (a if measure == "impact" else 0)
        if n_bars // max(span, 1) < min_nonoverlap:
            continue
        rows.append(response_at_horizon(bars, h, measure=measure,
                                        aggregation_s=aggregation_s,
                                        price_kind=price_kind,
                                        flow_col=flow_col).as_dict())
    return pd.DataFrame(rows)


def persistence_curve(bars: Bars, horizons_s, flow_col: str = "signed_volume"
                      ) -> pd.DataFrame:
    """Signal persistence at each aggregation horizon."""
    f = bars.frame()[flow_col]
    rows = []
    for h in horizons_s:
        k = max(1, int(round(h / bars.seconds)))
        if k >= len(f) // 4:
            continue
        agg = f.rolling(k, min_periods=max(1, k // 4)).sum().iloc[::k]
        p = persistence(agg, bar_seconds=h)
        rows.append({"horizon_s": float(h), **p.as_dict()})
    return pd.DataFrame(rows)


def permanent_lambda_curve(bars: Bars, windows_s, *, price_kind: str = "auto",
                           flow_col: str = "signed_volume") -> pd.DataFrame:
    """Kyle lambda estimated over increasing windows -- the cleanest split.

    For each window W, regress the price change over W on the signed flow
    arriving in that same window, and report the coefficient **per unit of flow**
    rather than per standard deviation. The per-sd version rises with W simply
    because sd(Q_W) grows with W, which obscures the thing being measured.

    The two cases separate sharply:

      permanent impact present   lambda(W) converges to a positive constant, the
                                 permanent impact coefficient itself.
      purely transient           lambda(W) decays toward zero, because the
                                 transient contributes only through the two
                                 window edges and that contribution shrinks
                                 relative to the window's total flow.

    So the *shape* answers the question, and the converged level quantifies it.
    This is far better conditioned than fitting an asymptote to the humped impact
    response, where the asymptote is only weakly identified.
    """
    c = response_curve(bars, windows_s, measure="contemporaneous",
                       price_kind=price_kind, flow_col=flow_col)
    if c.empty:
        return c
    c = c.rename(columns={"horizon_s": "window_s"})
    lam = c["lambda_per_unit"]
    c["lambda_ratio_to_prev"] = lam / lam.shift(1)
    # Converging (ratio -> 1) means permanent; decaying (ratio < 1) means transient.
    c["converging"] = c["lambda_ratio_to_prev"].between(0.85, 1.25)
    return c
