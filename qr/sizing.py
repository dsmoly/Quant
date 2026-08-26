"""Signal -> positions. A re-derivation of the robust LQ control, plus the
portfolio steps, with the problems from the original version fixed.

The closed form (unchanged -- it checks out)
--------------------------------------------
Position q, trading rate nu, quadratic impact k nu^2, penalty (1/2) phi sigma^2 q^2,
signal decay beta. The ergodic HJB is solved by h(q, alpha) = (1/2) h2 q^2 + h1 alpha q with

    |h2| = sigma sqrt(2 k phi)      rho = |h2| / (2k) = sigma sqrt(phi / 2k)
    h1   = 1 / (beta + rho)         q*  = alpha / ((beta + rho) |h2|)
    b    = c / |h2|                 nu* = -rho (q - q*)

Substituting these back into the HJB leaves residuals of order 1e-18, so the
derivation is internally consistent.

What changed, and why
---------------------
**The band is now in the same units as the book.** ``b = c/|h2|`` is expressed
in the control's own position units, but the target it is compared against has
been rescaled by an arbitrary volatility-target factor. Measured on a 120-name
book at the old defaults, the band came out at 1.4x a typical target position:
only 42 of 120 names ever crossed it, the book ran at a fifth of its intended
size, and the whole thing flipped discontinuously with ``target_vol`` (nothing
traded at 5%, everything traded at 10%). The band is now multiplied by the same
scale as the target. That is a pragmatic restoration of dimensional
consistency rather than a fresh derivation -- the honest statement is that
volatility targeting sits outside the control problem, so something has to give,
and a band that is a fixed fraction of the book is the least surprising choice.

**The control's own trading rate is available.** ``nu* = -rho (q - q*)`` says to
close a fraction rho of the gap per period -- about 10% per day at the old
defaults, a 7-day half-life. The original jumped 100% to the band edge, which
is the pure-linear-cost impulse solution paired with the pure-quadratic-cost
value function: two different problems' answers stapled together. It is also
why ``k`` tested as inert across three orders of magnitude. ``adjustment="rate"``
implements the control as derived; ``"edge"`` keeps the old behaviour.

**Volatility is reported on the book actually held.** The old diagnostic was
computed before banding, clipping and neutralising, and read 5.9% on books
whose true ex-ante vol was 2.5-3.2% -- including 0.6% on a book with zero gross.

**Caps are enforced after neutralisation, not before.** Clipping and then
neutralising pushed names back over ``max_weight``; the cap is now iterated to a
fixed point so it actually binds.

**A NaN in the covariance no longer silently rescales the book.** It used to
return the unscaled raw-control weights, tripling gross while reporting
``scale=0``. Affected names are dropped from the active set instead.

**Directional mode.** ``neutral=False`` skips the dollar-neutrality passes, and
``residual_factors=0`` stops projecting out the dominant factor before counting
breadth -- for a directional book that factor *is* the bet, and removing it
overstates effective breadth by about half.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------- breadth

def eigen_spectrum(matrix) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("expected a square matrix")
    m = 0.5 * (m + m.T)
    return np.clip(np.linalg.eigvalsh(m)[::-1], 0.0, None)


def effective_n_participation(matrix) -> float:
    """(sum lambda)^2 / sum lambda^2.

    Beware: this is badly biased downward on a short sample. On a 1-factor
    population whose true residual value is 87.9, a 120-day sample window
    returns 50.3 and a 250-day window 64.6 -- and the bias is near-deterministic
    (sd 0.4), so averaging does not remove it. It is measuring T/N as much as
    breadth. Feed it a shrunk or factor-model covariance, not a sample one, and
    prefer a correlation matrix: on the same data the covariance gives 65.3 and
    the correlation 77.7, because the participation ratio of a covariance is
    dominated by the high-volatility names.
    """
    vals = eigen_spectrum(matrix)
    denom = float((vals ** 2).sum())
    return float(vals.sum() ** 2 / denom) if denom > 0 else 0.0


def to_correlation(cov) -> np.ndarray:
    c = np.asarray(cov, dtype=float)
    d = np.sqrt(np.clip(np.diag(c), 1e-300, None))
    return c / np.outer(d, d)


def residualise_covariance(cov, n_factors: int = 1) -> np.ndarray:
    """Zero the leading ``n_factors`` eigenvalues. PSD by construction.

    Correct for a market-neutral book, wrong for a directional one -- see the
    module docstring.
    """
    m = np.asarray(cov, dtype=float)
    m = 0.5 * (m + m.T)
    if n_factors <= 0 or m.shape[0] <= n_factors:
        return m
    vals, vecs = np.linalg.eigh(m)
    order = np.argsort(vals)[::-1]
    vals, vecs = np.clip(vals[order], 0.0, None), vecs[:, order]
    vals[:n_factors] = 0.0
    return (vecs * vals) @ vecs.T


def usable_names(cov, active) -> np.ndarray:
    """Largest subset of ``active`` whose covariance submatrix is fully finite.

    A single unusable name poisons an entire row *and* column, so testing
    ``isfinite(C).all(axis=1)`` drops the whole universe rather than the one
    offender. Instead: reject names with a bad variance outright, then greedily
    remove whichever remaining name contributes the most missing entries until
    the submatrix is clean. Greedy is not guaranteed optimal, but the realistic
    case is a handful of bad names and it removes exactly those.
    """
    C = np.asarray(cov, dtype=float)
    keep = np.asarray(active, dtype=bool).copy()
    d = np.diag(C)
    keep &= np.isfinite(d) & (d > 0)
    for _ in range(int(keep.sum()) + 1):
        idx = np.flatnonzero(keep)
        if idx.size <= 1:
            break
        sub = C[np.ix_(idx, idx)]
        bad = ~np.isfinite(sub)
        if not bad.any():
            break
        keep[idx[int(np.argmax(bad.sum(axis=1)))]] = False
    return keep


def ledoit_wolf_shrink(cov, intensity: float | None = None) -> np.ndarray:
    """Shrink a sample covariance toward a scaled identity.

    Not the full Ledoit-Wolf optimal intensity (that needs the raw returns);
    this is the constant-correlation-free version with either a supplied
    intensity or a simple condition-number-driven default. Enough to stop the
    breadth estimator reading T/N.
    """
    c = np.asarray(cov, dtype=float)
    c = 0.5 * (c + c.T)
    n = c.shape[0]
    mu = float(np.trace(c) / n)
    target = mu * np.eye(n)
    if intensity is None:
        vals = np.clip(np.linalg.eigvalsh(c), 0.0, None)
        hi, lo = float(vals.max()), float(vals[vals > 0].min()) if (vals > 0).any() else 1.0
        cond = hi / max(lo, 1e-12)
        intensity = float(np.clip(np.log10(max(cond, 1.0)) / 8.0, 0.0, 0.9))
    a = float(np.clip(intensity, 0.0, 1.0))
    return (1.0 - a) * c + a * target


# ------------------------------------------------------------------ params

@dataclass
class SizingParams:
    phi: float = 25.0
    k: float = 0.5
    beta: float = 0.2               # measure this per signal with qr.decay
    target_vol: float = 0.10
    max_gross: float = 2.0
    max_weight: float = 0.10
    trading_days: int = 252
    neutral: bool = True            # False for a directional book
    residual_factors: int = 1       # 0 for a directional book
    breadth_haircut: bool = True
    breadth_on_correlation: bool = True
    shrink_covariance: bool = True
    adjustment: str = "rate"        # "rate" (the control) | "edge" | "target"
    use_band: bool = True
    use_shrinkage: bool = True
    vol_scaling: bool = True
    confidence_scales_risk: bool = True

    def __post_init__(self) -> None:
        if self.phi <= 0:
            raise ValueError("phi must be positive: at zero the position is unbounded")
        if self.k <= 0 or self.beta <= 0:
            raise ValueError("k and beta must be positive")
        if self.target_vol <= 0 or self.max_gross <= 0:
            raise ValueError("target_vol and max_gross must be positive")
        if self.adjustment not in ("rate", "edge", "target"):
            raise ValueError("adjustment must be 'rate', 'edge' or 'target'")

    def abs_h2(self, sigma):
        return np.asarray(sigma, dtype=float) * np.sqrt(2.0 * self.k * self.phi)

    def rho(self, sigma):
        return np.asarray(sigma, dtype=float) * np.sqrt(self.phi / (2.0 * self.k))

    def target_position(self, alpha, sigma):
        alpha = np.asarray(alpha, dtype=float)
        denom = (self.beta + self.rho(sigma)) * self.abs_h2(sigma)
        return np.divide(alpha, denom, out=np.zeros_like(alpha),
                         where=np.isfinite(denom) & (denom > 0))

    def no_trade_band(self, cost, sigma):
        h2 = self.abs_h2(sigma)
        cost = np.asarray(cost, dtype=float)
        return np.divide(np.maximum(cost, 0.0), h2, out=np.zeros_like(h2),
                         where=np.isfinite(h2) & (h2 > 0))


# -------------------------------------------------------------- operations

def neutralise_weights(w, active):
    out = np.where(active, w, 0.0)
    if int(active.sum()) == 0:
        return out
    out[active] = out[active] - out[active].mean()
    return out


def apply_caps(w, active, max_weight, neutral: bool, iters: int = 12):
    """Clip and (optionally) re-neutralise until both hold simultaneously.

    Clipping alone breaks neutrality; neutralising alone breaks the cap. The
    original did one pass of each in the order that leaves the cap violated.
    """
    out = np.clip(np.where(active, w, 0.0), -max_weight, max_weight)
    if not neutral:
        return out
    for _ in range(iters):
        out = neutralise_weights(out, active)
        over = np.abs(out) > max_weight + 1e-12
        if not over.any():
            return out
        out = np.clip(out, -max_weight, max_weight)
    return np.clip(out, -max_weight, max_weight)


def apply_band(current, target, band, rho=None, adjustment: str = "rate"):
    """Hold inside the band; outside it, move by the chosen rule.

    "rate"   -- nu* = -rho (q - q*), the control's own solution
    "edge"   -- jump to the near edge of the no-trade region (linear-cost impulse)
    "target" -- jump all the way to target (the naive version)
    """
    gap = target - current
    outside = np.abs(gap) > band
    if adjustment == "target":
        return np.where(outside, target, current)
    if adjustment == "edge":
        return np.where(outside, current + gap - np.sign(gap) * band, current)
    r = np.clip(np.asarray(rho if rho is not None else 1.0, dtype=float), 0.0, 1.0)
    return np.where(outside, current + r * gap, current)


def predicted_vol(w, cov) -> float:
    w = np.nan_to_num(np.asarray(w, dtype=float))
    var = float(w @ np.asarray(cov, dtype=float) @ w)
    return float(np.sqrt(max(var, 0.0)))


@dataclass
class SizingDiagnostics:
    gross: float
    net: float
    n_active: int
    n_traded: int
    n_crossed_band: int          # names that crossed on their own, before neutralising
    turnover: float
    predicted_vol_annual: float  # of the book ACTUALLY returned
    target_vol_effective: float  # after confidence and breadth haircuts
    scale: float
    n_eff: float
    mean_shrinkage: float
    dropped_no_cov: int


class SizingEngine:
    def __init__(self, params: SizingParams | None = None):
        self.p = params or SizingParams()

    def size(self, *, alpha, sigma, cov, current, cost, alpha_se=None, active=None):
        p = self.p
        alpha = np.asarray(alpha, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        current = np.asarray(current, dtype=float)
        n = alpha.size

        base = np.isfinite(alpha) & np.isfinite(sigma) & (sigma > 0)
        active = base if active is None else (np.asarray(active, dtype=bool) & base)

        # names with an unusable covariance row are dropped, not silently
        # allowed to blow up the scale factor
        dropped = 0
        if cov is not None:
            keep = usable_names(cov, active)
            dropped = int((active & ~keep).sum())
            active = keep

        alpha = np.where(active, np.nan_to_num(alpha), 0.0)
        safe_sigma = np.where(active, sigma, 1.0)
        if not p.vol_scaling:
            mean_s = float(np.mean(safe_sigma[active])) if active.any() else 1.0
            safe_sigma = np.full_like(safe_sigma, mean_s)

        # 1. shrink by measurement quality
        if alpha_se is not None and p.use_shrinkage:
            se = np.asarray(alpha_se, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                t2 = np.where(np.isfinite(se) & (se > 0), (alpha / se) ** 2, 0.0)
            factor = np.where(np.isfinite(t2), t2 / (1.0 + t2), 0.0)
            shrunk = alpha * factor
            mean_shrink = float(np.mean(factor[active])) if active.any() else 0.0
        else:
            shrunk, mean_shrink = alpha, 1.0

        # 2. control target
        target = np.where(active, p.target_position(shrunk, safe_sigma), 0.0)
        if p.neutral:
            target = neutralise_weights(target, active)

        # 3. risk budget. Confidence multiplies the risk deployed, not the
        #    weights: volatility targeting is scale-invariant and would
        #    otherwise divide the shrinkage straight back out and re-lever a
        #    signal that shrinkage had just declared worthless.
        conf = float(np.clip(mean_shrink, 0.0, 1.0)) if np.isfinite(mean_shrink) else 0.0
        if not p.confidence_scales_risk:
            conf = 1.0
        target, scale, n_eff, tv_eff = self._risk_budget(target, cov, active, conf)

        # 4. band, in the SAME units as the (now scaled) book
        if p.use_band:
            band = np.where(active, p.no_trade_band(cost, safe_sigma) * max(scale, 0.0), 0.0)
        else:
            band = np.zeros(n)
        crossed = int((np.abs(target - current) > band)[active].sum()) if active.any() else 0
        rate = np.clip(p.rho(safe_sigma), 0.0, 1.0)
        new = apply_band(current, target, band, rho=rate, adjustment=p.adjustment)
        new = np.where(active, new, 0.0)     # names out of the universe are closed

        # 5. caps and neutrality together, then gross
        new = apply_caps(new, active, p.max_weight, p.neutral)
        gross = float(np.abs(new).sum())
        if gross > p.max_gross and gross > 0:
            new = new * (p.max_gross / gross)

        vol_ann = 0.0
        if cov is not None and active.any():
            idx = np.flatnonzero(active)
            sub = np.asarray(cov, dtype=float)[np.ix_(idx, idx)]
            if np.all(np.isfinite(sub)):
                vol_ann = predicted_vol(new[idx], sub) * np.sqrt(p.trading_days)

        traded = np.abs(new - current) > 1e-12
        diag = SizingDiagnostics(
            gross=float(np.abs(new).sum()), net=float(new.sum()),
            n_active=int(active.sum()), n_traded=int(traded.sum()),
            n_crossed_band=crossed,
            turnover=float(np.abs(new - current).sum()),
            predicted_vol_annual=float(vol_ann),
            target_vol_effective=float(tv_eff), scale=float(scale),
            n_eff=float(n_eff),
            mean_shrinkage=float(mean_shrink) if np.isfinite(mean_shrink) else 0.0,
            dropped_no_cov=dropped)
        return new, diag

    def _risk_budget(self, w, cov, active, confidence: float = 1.0):
        p = self.p
        idx = np.flatnonzero(active)
        if idx.size == 0 or cov is None:
            return np.zeros_like(w), 0.0, 0.0, 0.0
        sub = np.asarray(cov, dtype=float)[np.ix_(idx, idx)]
        if not np.all(np.isfinite(sub)):
            return np.zeros_like(w), 0.0, 0.0, 0.0

        breadth_src = ledoit_wolf_shrink(sub) if p.shrink_covariance else sub
        if p.breadth_on_correlation:
            breadth_src = to_correlation(breadth_src)
        n_eff = effective_n_participation(
            residualise_covariance(breadth_src, p.residual_factors))

        # vol from the FULL covariance: if the book really is neutral the factor
        # term is already near zero, and using the residual would understate
        # risk by exactly the exposure the book failed to neutralise.
        vol = predicted_vol(w[idx], sub)
        if vol <= 0:
            return np.zeros_like(w), 0.0, float(n_eff), 0.0

        target_vol = p.target_vol * float(np.clip(confidence, 0.0, 1.0))
        if p.breadth_haircut and idx.size > 0:
            target_vol *= float(np.sqrt(max(n_eff, 1.0) / idx.size))
        scale = (target_vol / np.sqrt(p.trading_days)) / vol
        out = w * scale
        gross = float(np.abs(out).sum())
        if gross > p.max_gross and gross > 0:
            adj = p.max_gross / gross
            out, scale = out * adj, scale * adj
        return out, float(scale), float(n_eff), float(target_vol)
