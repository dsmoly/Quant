"""Signal decay: how fast the predictive power dies, and the beta it implies.

Why this module is fussier than it looks
----------------------------------------
The sizing engine discounts a signal by 1/(beta + rho), so beta is a real
parameter of the book, not a diagnostic. Getting it wrong mis-sizes everything.

There are two obvious ways to measure it and **both are biased**:

*Cumulative.* Regress log IC(signal_t, sum of returns t+1..t+h) on h. This
understates decay badly -- measured 2-6x too slow in simulation, because the
cumulative window keeps including the early periods when the signal was still
strong. A true beta of 0.20 comes out as 0.053.

*Marginal.* Use IC(signal_t, return at t+h alone). This is the right quantity
and is unbiased *while the marginal IC is distinguishable from zero*. Past that
point the log of a noise-dominated IC is garbage and the fitted slope explodes
-- true betas of 0.10 to 0.40 all came out near 0.5.

So the estimator here does the only honest thing: it fits the marginal curve,
weighted by each horizon's own standard error, and **restricts the fit to
horizons where the IC clears a significance floor**. If only one horizon
survives, it says so and returns NaN rather than a confident wrong number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qr.ic import ic_series, newey_west_se


@dataclass
class DecayFit:
    beta: float                      # exponential decay rate, per period
    half_life: float                 # log(2) / beta
    horizons: np.ndarray             # horizons actually used in the fit
    ic: np.ndarray                   # marginal IC at every requested horizon
    se: np.ndarray                   # standard error of each
    t: np.ndarray                    # t-stat of each
    used: np.ndarray                 # boolean: which horizons entered the fit
    ic0: float                       # fitted IC at horizon 0
    r2: float
    note: str = ""

    def __str__(self) -> str:
        if not np.isfinite(self.beta):
            return f"decay: UNIDENTIFIED ({self.note})"
        return (f"decay beta {self.beta:.4f}/period  half-life {self.half_life:.1f}  "
                f"IC(0) {self.ic0:.4f}  R2 {self.r2:.2f}  "
                f"fit on {int(self.used.sum())}/{len(self.horizons)} horizons")


def marginal_ic_curve(signal, returns, horizons=(1, 2, 3, 5, 10, 20, 40),
                      mask=None, method: str = "spearman"):
    """IC of the signal at t against the return at t+h *alone*, for each h.

    Note this is the single-period return at lag h, not the cumulative return
    to h. That is what makes the decay curve interpretable as a decay curve:
    each point is an independent read of "how much does the signal still know
    about a single day, h days later".
    """
    s = np.asarray(signal, dtype=float)
    r = np.asarray(returns, dtype=float)
    if s.shape != r.shape:
        raise ValueError(f"signal shape {s.shape} != returns shape {r.shape}")
    T = s.shape[0]
    hs = np.asarray(sorted(set(int(h) for h in horizons)), dtype=int)
    ics, ses, ts = [], [], []
    for h in hs:
        if h >= T - 2:
            ics.append(np.nan); ses.append(np.nan); ts.append(np.nan); continue
        sig = s[: T - h]
        fwd = r[h:]
        m = None if mask is None else np.asarray(mask, dtype=bool)[: T - h]
        series = ic_series(sig, fwd, mask=m, method=method)
        v = series[np.isfinite(series)]
        if v.size < 3:
            ics.append(np.nan); ses.append(np.nan); ts.append(np.nan); continue
        # marginal (non-overlapping) IC needs no HAC in principle, but the
        # signal itself is persistent, so keep a short lag.
        se = newey_west_se(v, lags=min(5, max(v.size - 2, 0)))
        ics.append(float(v.mean())); ses.append(float(se))
        ts.append(float(v.mean() / se) if se > 0 else np.nan)
    return hs, np.array(ics), np.array(ses), np.array(ts)


def fit_decay(horizons, ic, se=None, t=None, t_floor: float = 2.0,
              min_points: int = 3) -> DecayFit:
    """Weighted log-linear fit of IC(h) = IC(0) * exp(-beta h).

    Only horizons whose IC is same-signed as the shortest horizon and clears
    ``t_floor`` enter the fit -- see the module docstring for why that
    restriction is the whole point rather than a nicety.
    """
    hs = np.asarray(horizons, dtype=float)
    y = np.asarray(ic, dtype=float)
    se = np.full_like(y, np.nan) if se is None else np.asarray(se, dtype=float)
    t = (y / se) if t is None else np.asarray(t, dtype=float)

    finite = np.isfinite(y) & np.isfinite(hs)
    if not finite.any():
        return DecayFit(np.nan, np.nan, hs, y, se, t, np.zeros_like(hs, bool),
                        np.nan, np.nan, "no finite IC values")

    sign = np.sign(y[finite][0]) or 1.0
    used = finite & (np.sign(y) == sign) & (np.abs(y) > 0) & (np.abs(t) >= t_floor)

    if used.sum() < min_points:
        return DecayFit(np.nan, np.nan, hs, y, se, t, used, np.nan, np.nan,
                        f"only {int(used.sum())} horizon(s) clear |t|>={t_floor}; "
                        "signal is too weak or too short-lived to identify a decay rate")

    x = hs[used]
    ly = np.log(np.abs(y[used]))
    # delta-method SE of log|IC|, and weight by its inverse variance
    with np.errstate(divide="ignore", invalid="ignore"):
        lse = np.where(np.isfinite(se[used]) & (np.abs(y[used]) > 0),
                       se[used] / np.abs(y[used]), np.nan)
    w = np.where(np.isfinite(lse) & (lse > 0), 1.0 / lse ** 2, 1.0)
    w = w / w.sum()

    xm = np.sum(w * x); ym = np.sum(w * ly)
    sxx = np.sum(w * (x - xm) ** 2)
    if sxx <= 0:
        return DecayFit(np.nan, np.nan, hs, y, se, t, used, np.nan, np.nan,
                        "degenerate horizon spread")
    slope = np.sum(w * (x - xm) * (ly - ym)) / sxx
    intercept = ym - slope * xm
    pred = intercept + slope * x
    ss_res = np.sum(w * (ly - pred) ** 2)
    ss_tot = np.sum(w * (ly - ym) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    beta = float(-slope)
    if beta <= 0:
        return DecayFit(np.nan, np.nan, hs, y, se, t, used, float(np.exp(intercept)), r2,
                        "fitted decay is non-positive: IC is flat or rising with horizon, "
                        "which usually means the signal is a slow-moving risk exposure "
                        "rather than a decaying forecast")
    return DecayFit(beta, float(np.log(2.0) / beta), hs, y, se, t, used,
                    float(np.exp(intercept) * sign), r2, "")
