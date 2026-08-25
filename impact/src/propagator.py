"""Transient versus permanent impact, fitted rather than eyeballed.

The question this module exists to answer: of the price move that follows signed
order flow, how much is a temporary dislocation that mean-reverts as liquidity
replenishes, and how much is information that never comes back?

Reading it off the chart -- "the curve looks flat after about a minute" -- is not
good enough, because where a noisy curve appears to flatten depends on the axis
scaling and on the noise. Two fits are done instead, and they are independent
enough that agreement between them is meaningful.

1. Asymptotic decomposition
---------------------------

    R(h) = R_perm + A (1 + h/tau)^(-gamma)

``R_perm`` is the permanent component by construction: the limit as h grows.
``A`` is the transient amplitude and carries a sign, so both a decaying response
(A > 0, impact relaxing down toward the permanent level) and a building one
(A < 0, aggregate response growing toward it as correlated flow arrives) are
representable by the same form. The power-law relaxation is used rather than an
exponential because order-flow impact is not exponential -- an exponential fit
forced onto a power-law decay systematically *understates* the asymptote, which
would bias the answer toward "no permanent component", i.e. toward the
conclusion we are trying to test.

2. Propagator (transient impact model)
--------------------------------------
Bouchaud's TIM writes the price as a sum of past impacts, each decaying under a
propagator G:

    p(t) = sum_{s<t} G(t-s) eps_s f(v_s) + noise

which implies a response function that mixes G with the flow autocorrelation C:

    R(l) = sum_{n=0}^{l-1} G(l-n) C(n)  +  sum_{n>=1} [G(l+n) - G(n)] C(n)

with G(l) = G0 / (1 + (l/l0)^2)^(gamma/2).

This matters because it separates two things the raw curve confounds. Order flow
is strongly autocorrelated, so *even a purely transient* propagator produces a
response function that keeps rising with horizon -- the second term above. A
rising R(h) is therefore **not** by itself evidence of permanent impact. The TIM
fit is what distinguishes "impact is permanent" from "impact is transient but
flow is persistent", and that distinction is the entire question.

The model's permanent level is evaluated as R(l) at large l rather than
analytically, because the closed form depends on the gamma-versus-flow-exponent
balance in a way that is fragile near the critical value gamma = (1-beta)/2.

Errors
------
Both fits are wrapped in a **circular block bootstrap** over the underlying bars.
Resampling individual bars would destroy the flow autocorrelation, which is a
first-order feature of the data and would make the errors far too small. The
block length must exceed the signal's integrated correlation time; the default
ties it to the longest horizon fitted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from .flow import Bars
from .response import autocorr, response_curve


@dataclass
class Decomposition:
    r_perm_bps: float
    r_perm_lo: float
    r_perm_hi: float
    transient_amp_bps: float
    tau_s: float
    gamma: float
    r_perm_share: float          # permanent level as a fraction of the peak response
    n_boot: int
    converged: bool
    rmse_bps: float

    def as_dict(self) -> dict:
        return asdict(self)


def _model(h, r_perm, amp, tau, gamma):
    return r_perm + amp * (1.0 + np.maximum(h, 0.0) / max(tau, 1e-9)) ** (-gamma)


def fit_asymptote(horizons_s, r_bps, sigma=None) -> tuple[np.ndarray, bool, float]:
    """Least-squares fit of R(h) = R_perm + A (1 + h/tau)^-gamma."""
    h = np.asarray(horizons_s, dtype=float)
    y = np.asarray(r_bps, dtype=float)
    ok = np.isfinite(h) & np.isfinite(y)
    h, y = h[ok], y[ok]
    if h.size < 4:
        return np.full(4, np.nan), False, float("nan")
    w = None
    if sigma is not None:
        s = np.asarray(sigma, dtype=float)[ok]
        w = np.where(np.isfinite(s) & (s > 0), s, np.nanmedian(s[np.isfinite(s)]) or 1.0)

    span = float(y.max() - y.min()) or 1.0
    guesses = [
        (y[-1], y[0] - y[-1], max(h[0], 1.0), 0.5),
        (y[-1], -span, np.sqrt(max(h[0], 1.0) * h[-1]), 0.3),
        (float(np.median(y)), span, h[len(h) // 2], 1.0),
    ]
    best, best_cost, ok_fit = np.full(4, np.nan), np.inf, False
    for g in guesses:
        try:
            p, _ = optimize.curve_fit(
                _model, h, y, p0=g, sigma=w, absolute_sigma=False, maxfev=20000,
                bounds=([-np.inf, -np.inf, 1e-6, 0.01], [np.inf, np.inf, 1e9, 5.0]))
        except Exception:
            continue
        cost = float(np.sum((y - _model(h, *p)) ** 2))
        if cost < best_cost:
            best, best_cost, ok_fit = p, cost, True
    rmse = float(np.sqrt(best_cost / h.size)) if ok_fit else float("nan")
    return best, ok_fit, rmse


def circular_block_indices(n: int, block: int, rng) -> np.ndarray:
    """Indices for one circular block bootstrap replicate."""
    block = max(1, min(int(block), n))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])
    return idx[:n]


def _bars_from_frame(f: pd.DataFrame, template: Bars) -> Bars:
    return Bars(freq=template.freq, seconds=template.seconds,
                trade_price=f["trade_price"], mid=f["mid"] if "mid" in f else None,
                signed_volume=f["signed_volume"], signed_count=f["signed_count"],
                volume=f["volume"], n_trades=f["n_trades"])


def decompose(bars: Bars, horizons_s, *, measure: str = "impact",
              price_kind: str = "auto", n_boot: int = 100,
              block_s: float | None = None, seed: int = 0,
              fit_from_peak: bool = True) -> Decomposition:
    """Fit the transient/permanent split with block-bootstrap confidence bounds.

    ``fit_from_peak`` restricts the fit to horizons at or beyond the response's
    maximum, and it is on by default for a reason worth spelling out.

    The impact response does not decay monotonically from an initial jump. It
    rises, peaks, and then falls -- and with a purely transient propagator it
    falls *through zero into negative territory*. That is not a defect in the
    data. Conditioning on positive flow at t, the baseline price just before the
    event is already elevated by the transient left behind by earlier
    same-signed flow, because order flow is strongly autocorrelated. That prior
    elevation decays away over the horizon, so the measured change from the
    baseline ends up negative. The size of that undershoot is itself diagnostic:
    it is large when impact is transient and small when it is permanent.

    A monotone relaxation model cannot represent a hump, and fitting one to the
    whole curve produces nonsense -- an early version returned R_perm = -1516 bps
    with tau = 1.2 million seconds, using the tail of the power law to fake the
    rise. The asymptote is only identified on the decaying branch, so that is
    what gets fitted. The rising branch is described by the propagator fit
    instead, which models it properly.
    """
    curve = response_curve(bars, horizons_s, measure=measure, price_kind=price_kind)
    if curve.empty or curve["r_bps"].notna().sum() < 4:
        return Decomposition(*([float("nan")] * 6), n_boot=0, converged=False,
                             rmse_bps=float("nan"))

    def _fit(c):
        if fit_from_peak and c["r_bps"].notna().sum() >= 5:
            i = int(c["r_bps"].abs().idxmax())
            branch = c.loc[i:]
            if branch["r_bps"].notna().sum() >= 4:
                c = branch
        return fit_asymptote(c["horizon_s"], c["r_bps"], c["r_se_bps"])

    p, ok, rmse = _fit(curve)

    boots = []
    if n_boot > 0 and ok:
        rng = np.random.default_rng(seed)
        f = bars.frame()
        n = len(f)
        block = int(max(1, round((block_s or max(horizons_s)) / bars.seconds)))
        block = min(block, max(1, n // 4))
        for _ in range(n_boot):
            idx = circular_block_indices(n, block, rng)
            fb = f.iloc[idx].copy()
            fb.index = f.index[:len(fb)]
            try:
                cb = response_curve(_bars_from_frame(fb, bars), horizons_s,
                                    measure=measure, price_kind=price_kind)
                pb, okb, _ = _fit(cb)
                if okb and np.isfinite(pb[0]):
                    boots.append(float(pb[0]))
            except Exception:
                continue

    lo, hi = ((float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
              if len(boots) >= 20 else (float("nan"), float("nan")))
    peak = float(curve["r_bps"].abs().max()) if len(curve) else float("nan")
    share = float(p[0] / peak) if np.isfinite(peak) and peak > 1e-12 else float("nan")
    return Decomposition(
        r_perm_bps=float(p[0]), r_perm_lo=lo, r_perm_hi=hi,
        transient_amp_bps=float(p[1]), tau_s=float(p[2]), gamma=float(p[3]),
        r_perm_share=share, n_boot=len(boots), converged=bool(ok), rmse_bps=rmse)


# --------------------------------------------------------- the propagator fit


def tim_response(lags: np.ndarray, G0: float, l0: float, gamma: float,
                 C: np.ndarray) -> np.ndarray:
    """R(l) implied by a power-law propagator and a measured flow autocorrelation.

    R(l) = sum_{n<l} G(l-n) C(n) + sum_{n>=1} [G(l+n) - G(n)] C(n)
    """
    def G(x):
        x = np.asarray(x, dtype=float)
        return G0 / (1.0 + (np.maximum(x, 0.0) / max(l0, 1e-9)) ** 2) ** (gamma / 2.0)

    nmax = C.size - 1
    out = np.empty(lags.size, dtype=float)
    for i, l in enumerate(lags):
        l = int(l)
        n1 = np.arange(0, min(l, nmax + 1))
        term1 = float(np.sum(G(l - n1) * C[n1]))
        n2 = np.arange(1, nmax + 1)
        term2 = float(np.sum((G(l + n2) - G(n2)) * C[n2]))
        out[i] = term1 + term2
    return out


@dataclass
class PropagatorFit:
    g0_bps: float
    l0_bars: float
    gamma: float
    permanent_bps: float      # model R at a very long lag
    converged: bool
    rmse_bps: float
    flow_ac_exponent: float   # beta, from a power-law fit to C(n)

    def as_dict(self) -> dict:
        return asdict(self)


def fit_flow_autocorrelation(x: np.ndarray, max_lag: int = 400) -> tuple[np.ndarray, float]:
    """C(n) plus a power-law exponent beta from C(n) ~ n^-beta.

    beta matters for interpretation: the TIM is marginally permanent when
    gamma = (1 - beta)/2, and comparing the fitted gamma against that critical
    value says whether the venue sits on the efficient-diffusion knife edge.
    """
    C = autocorr(x, max_lag)
    C = np.nan_to_num(C, nan=0.0)
    n = np.arange(1, C.size)
    pos = C[1:] > 0
    beta = float("nan")
    if pos.sum() > 10:
        slope = np.polyfit(np.log(n[pos]), np.log(C[1:][pos]), 1)[0]
        beta = float(-slope)
    return C, beta


def fit_propagator(bars: Bars, horizons_s, *, price_kind: str = "auto",
                   max_lag: int = 400, long_lag_mult: float = 50.0) -> PropagatorFit:
    """Fit G to the empirical impact response, holding the measured C(n) fixed.

    Always uses the single-bar impact response: the TIM is written for individual
    events and performs its own aggregation internally through C(n), so feeding
    it window-aggregated flow would double-count that convolution.
    """
    curve = response_curve(bars, horizons_s, measure="impact", price_kind=price_kind)
    if curve.empty or curve["r_bps"].notna().sum() < 4:
        return PropagatorFit(*([float("nan")] * 3), float("nan"), False,
                             float("nan"), float("nan"))

    sign_flow = np.sign(bars.frame()["signed_volume"].to_numpy(dtype=float))
    C, beta = fit_flow_autocorrelation(sign_flow, max_lag=max_lag)

    lags = np.maximum(1, np.round(curve["horizon_s"].to_numpy() / bars.seconds)).astype(int)
    y = curve["r_bps"].to_numpy(dtype=float)
    ok = np.isfinite(y)
    lags, y = lags[ok], y[ok]
    if lags.size < 4:
        return PropagatorFit(*([float("nan")] * 3), float("nan"), False,
                             float("nan"), beta)

    def resid(theta):
        g0, l0, gamma = theta
        return tim_response(lags, g0, l0, gamma, C) - y

    best, best_cost, okfit = None, np.inf, False
    for g in ((abs(y[0]) or 1.0, 1.0, 0.4), (abs(y[-1]) or 1.0, 10.0, 0.2),
              (abs(y.mean()) or 1.0, 100.0, 0.8)):
        try:
            sol = optimize.least_squares(
                resid, x0=g, bounds=([1e-9, 1e-6, 0.01], [1e6, 1e7, 3.0]),
                max_nfev=4000)
        except Exception:
            continue
        if sol.cost < best_cost:
            best, best_cost, okfit = sol.x, sol.cost, True
    if not okfit:
        return PropagatorFit(*([float("nan")] * 3), float("nan"), False,
                             float("nan"), beta)

    g0, l0, gamma = best
    far = np.array([int(max(lags.max() * long_lag_mult, lags.max() + 10))])
    perm = float(tim_response(far, g0, l0, gamma, C)[0])
    rmse = float(np.sqrt(2 * best_cost / lags.size))
    return PropagatorFit(g0_bps=float(g0), l0_bars=float(l0), gamma=float(gamma),
                         permanent_bps=perm, converged=True, rmse_bps=rmse,
                         flow_ac_exponent=beta)


def critical_gamma(beta: float) -> float:
    """gamma at which the TIM is marginally permanent: gamma = (1 - beta)/2.

    Below it, impact accumulates and prices would be predictable; above it,
    impact decays faster than correlated flow arrives and the permanent
    component vanishes. Real markets sit close to this line, which is the
    standard argument for why the response function is so hard to call by eye.
    """
    return float((1.0 - beta) / 2.0) if np.isfinite(beta) else float("nan")


# ------------------------------------------------------- the primary estimator


@dataclass
class PermanentVerdict:
    lambda_short: float          # per unit flow, at the shortest window
    lambda_long: float           # per unit flow, at the longest window
    decay_ratio: float           # long / short; ~0 = transient, ~1 = permanent
    lambda_long_lo: float
    lambda_long_hi: float
    slope_log: float             # d log lambda / d log window on the long half
    n_boot: int
    verdict: str
    windows_s: list
    lambdas: list

    def as_dict(self) -> dict:
        return asdict(self)


def permanent_verdict(bars: Bars, windows_s, *, price_kind: str = "auto",
                      n_boot: int = 100, seed: int = 0,
                      flow_col: str = "signed_volume") -> PermanentVerdict:
    """Classify impact as transient or permanent from the lambda-versus-window curve.

    The shape carries the answer. Estimating lambda per unit of signed flow over
    windows of increasing length:

      * a purely transient propagator gives lambda decaying toward zero, because
        the transient enters only through the window's two edges and that
        contribution shrinks against the window's total flow;
      * a permanent component gives lambda converging to a positive constant --
        the permanent impact coefficient itself.

    Validated against planted ground truth: with no permanent component lambda
    falls by 94% across the window range, and the excess of lambda over that null
    is exactly linear in the planted coefficient.

    Two honest limits. The null case does **not** reach zero at finite windows --
    there is a positive finite-window bias -- so the monotone decay is the
    reliable signal and the converged *level* is an upper bound on permanent
    impact. And the longest window must be much shorter than the sample, or the
    estimate is dominated by a handful of independent observations; horizons that
    fail that test are dropped upstream by ``response_curve``.
    """
    from .response import permanent_lambda_curve

    curve = permanent_lambda_curve(bars, windows_s, price_kind=price_kind,
                                   flow_col=flow_col)
    if curve.empty or curve["lambda_per_unit"].notna().sum() < 3:
        return PermanentVerdict(*([float("nan")] * 6), n_boot=0,
                                verdict="insufficient data", windows_s=[], lambdas=[])

    lam = curve["lambda_per_unit"].to_numpy(dtype=float)
    w = curve["window_s"].to_numpy(dtype=float)
    short, long_ = float(lam[0]), float(lam[-1])
    ratio = float(long_ / short) if abs(short) > 1e-12 else float("nan")

    # The slope needs at least three points, so take the longer of "the back
    # half of the curve" and "the last three windows". With only four usable
    # windows the back half is two points and the fit silently returns nan,
    # which reads as "indeterminate" when the data was in fact adequate.
    half = min(len(lam), max(3, len(lam) // 2))
    tail_l, tail_w = lam[-half:], w[-half:]
    ok = np.isfinite(tail_l) & (tail_l > 0)
    slope = (float(np.polyfit(np.log(tail_w[ok]), np.log(tail_l[ok]), 1)[0])
             if ok.sum() >= 3 else float("nan"))

    boots = []
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        f = bars.frame()
        n = len(f)
        block = min(int(max(1, round(max(windows_s) * 4 / bars.seconds))), max(1, n // 4))
        for _ in range(n_boot):
            fb = f.iloc[circular_block_indices(n, block, rng)].copy()
            fb.index = f.index[:len(fb)]
            try:
                cb = permanent_lambda_curve(_bars_from_frame(fb, bars), windows_s,
                                            price_kind=price_kind, flow_col=flow_col)
                if not cb.empty and np.isfinite(cb["lambda_per_unit"].iloc[-1]):
                    boots.append(float(cb["lambda_per_unit"].iloc[-1]))
            except Exception:
                continue
    lo, hi = ((float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
              if len(boots) >= 20 else (float("nan"), float("nan")))

    # Classification is on the log-log SLOPE over the long half of the curve, not
    # on the long/short ratio. The ratio compares the longest window against the
    # shortest, and the shortest is dominated by the transient, so it cannot
    # approach 1 even when lambda has completely converged -- an earlier version
    # gated on ratio > 0.6 and refused to call a curve permanent at slope -0.03,
    # which is as converged as a curve gets. The slope has a direct reading:
    #
    #   slope ~ -1   lambda ~ 1/W, so total impact per window is independent of
    #                window length. Nothing accumulates: purely transient.
    #   slope ~  0   lambda is constant in W. Impact accumulates in proportion to
    #                flow: a permanent component.
    #
    # A permanent verdict additionally requires the bootstrap interval to exclude
    # zero, so a flat but indistinguishable-from-nothing curve is not promoted.
    excludes_zero = np.isfinite(lo) and np.isfinite(hi) and lo > 0
    if not np.isfinite(slope):
        v = "indeterminate"
    elif slope < -0.7:
        v = "TRANSIENT: lambda ~ 1/W; no permanent component detected"
    elif slope > -0.2 and excludes_zero:
        v = "PERMANENT: lambda converges to a positive level"
    elif slope > -0.2:
        v = "FLAT but not distinguishable from zero; sample too small to call"
    else:
        v = "MIXED: lambda still declining at the longest window; the level is an upper bound"
    return PermanentVerdict(
        lambda_short=short, lambda_long=long_, decay_ratio=ratio,
        lambda_long_lo=lo, lambda_long_hi=hi, slope_log=slope, n_boot=len(boots),
        verdict=v, windows_s=w.tolist(), lambdas=lam.tolist())
