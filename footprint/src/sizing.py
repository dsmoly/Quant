"""The sizing engine: turning a cross-sectional signal into positions.

This is a re-derivation of the robust control worked out for the intraday
project, applied at daily frequency across names instead of over time in one
name. The microstructure code is not imported -- none of it applies at this
horizon -- but the closed form does, because the control problem has the same
shape: hold a position against a decaying signal, pay a quadratic cost to move
and a linear cost to cross, and distrust your own drift estimate.

The closed form
---------------
With position q, trading rate nu, quadratic impact k, signal decay beta, and an
entropic penalty phi on drift ambiguity, the ergodic HJB is solved by a quadratic
value function whose coefficients are

    h2 = -sqrt(2 k phi sigma^2)      (negative root: the concave one)
    rho = |h2| / (2k) = sigma sqrt(phi / 2k)
    h1 = 1 / (beta + rho)

and the four things this module needs all fall out of those:

  1. Target position.  q* = alpha h1 / |h2| = alpha / ((beta + rho) sigma sqrt(2 k phi)).
     Since |h2| is proportional to sigma, **q* is proportional to 1/sigma**:
     volatility targeting is not bolted on afterwards, it is what the optimal
     control does. Note it is 1/sigma and not the mean-variance 1/sigma^2,
     because the ambiguity penalty enters as phi sigma^2 q^2 and the impact cost
     as k nu^2; the geometric mean of the two exponents is what survives.

  2. Signal decay discount.  q* carries a factor 1/(beta + rho): a signal that
     dies faster than you can trade into it is worth less. At daily frequency
     beta is the decay of the *feature's* predictive power, measured in
     ``features.py`` rather than assumed.

  3. Confidence shrinkage.  alpha is replaced by alpha t^2/(1+t^2), where t is
     the signal's own t-statistic. A well-measured signal passes through almost
     untouched; a badly measured one collapses to nothing. This is the
     estimation-error counterpart of ambiguity aversion -- distrust proportional
     to what the data actually pins down, rather than a fixed preference knob.
     It is also the mechanism that should stop a zero-alpha signal from being
     traded at all, which is exactly what the dumb-signal experiment tests.

  4. No-trade band.  A unit of trade is worth |dh/dq| = |h2| (q - q*) and costs
     c, so nothing happens until the gap to target exceeds

         b = c / |h2|

     and when it does, the position moves to the *edge* of the band rather than
     to the target. That is the standard impulse-control solution under a linear
     cost, and it matters: moving to the target instead would pay the full cost
     to buy back the same trade tomorrow.

Portfolio construction on top
-----------------------------
The per-name control gives desired positions. Three portfolio-level steps follow,
in this order, and the order is load-bearing:

  neutralise -> band -> risk budget

Neutralising first means the band is applied to the trades actually intended.
Applying the band first and neutralising afterwards would shift every position by
a constant and silently push names back outside their own no-trade region. Even
in this order there is a genuine tension -- enforcing exact dollar neutrality
after banding perturbs banded positions -- so the residual net exposure is
measured and reported rather than assumed to be zero.

Risk budgeting uses the *residual* covariance, after the dominant factor is
projected out, because a dollar-neutral cross-sectional portfolio has already
neutralised most of its market exposure; sizing off the raw covariance would
double-count a risk that is not being taken. The effective breadth from the
eigenstructure (see ``breadth.py``) enters as an explicit haircut, not as the
naive IC*sqrt(N).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .breadth import effective_n_participation, residualise_covariance


@dataclass
class SizingParams:
    """Preferences and frictions for the daily cross-sectional control."""

    phi: float = 25.0            # ambiguity aversion / risk aversion; the sizing knob
    k: float = 0.5               # quadratic impact coefficient
    beta: float = 0.2            # signal decay rate, per day
    target_vol: float = 0.10     # annualised portfolio volatility target
    max_gross: float = 2.0       # cap on sum |w|, as a multiple of capital
    max_weight: float = 0.10     # per-name cap on |w|
    trading_days: int = 252
    breadth_haircut: bool = True  # scale exposure by sqrt(N_eff / N)
    move_to_band_edge: bool = True   # impulse-control solution; False = move to target
    use_shrinkage: bool = True    # ablation switch: trade the raw estimate instead
    vol_scaling: bool = True      # ablation switch: per-name sigma, or the mean
    use_band: bool = True         # ablation switch: rebalance to target every day

    def __post_init__(self) -> None:
        if self.phi <= 0:
            raise ValueError("phi must be positive: at zero the position is unbounded")
        if self.k <= 0 or self.beta <= 0:
            raise ValueError("k and beta must be positive")
        if self.target_vol <= 0 or self.max_gross <= 0:
            raise ValueError("target_vol and max_gross must be positive")

    # --------------------------------------------------- closed-form pieces

    def abs_h2(self, sigma):
        """|h2| = sigma sqrt(2 k phi). Proportional to sigma, which is the point."""
        return np.asarray(sigma, dtype=float) * np.sqrt(2.0 * self.k * self.phi)

    def rho(self, sigma):
        """Gap-closing rate; also the discount that a slow signal escapes."""
        return np.asarray(sigma, dtype=float) * np.sqrt(self.phi / (2.0 * self.k))

    def target_position(self, alpha, sigma):
        """q*(alpha, sigma) = alpha / ((beta + rho) |h2|), i.e. proportional to 1/sigma."""
        alpha = np.asarray(alpha, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        denom = (self.beta + self.rho(sigma)) * self.abs_h2(sigma)
        return np.divide(alpha, denom, out=np.zeros_like(alpha),
                         where=np.isfinite(denom) & (denom > 0))

    def no_trade_band(self, cost, sigma):
        """b = c / |h2|: how far from target you must be before trading pays."""
        cost = np.asarray(cost, dtype=float)
        h2 = self.abs_h2(sigma)
        return np.divide(np.maximum(cost, 0.0), h2, out=np.zeros_like(h2),
                         where=np.isfinite(h2) & (h2 > 0))


# ------------------------------------------------------------------ shrinkage


def shrink_by_tstat(alpha, se, min_obs_ok: bool = True):
    """Scale an estimate by t^2/(1+t^2), its own signal-to-noise ratio.

    This is the James-Stein-flavoured shrinkage carried over from the intraday
    work. At t=1 it halves the signal; at t=3 it keeps 90%; at t=0 it returns
    exactly zero, which is the property the dumb-signal control depends on.
    """
    alpha = np.asarray(alpha, dtype=float)
    se = np.asarray(se, dtype=float)
    if not min_obs_ok:
        return np.zeros_like(alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        t2 = np.where(se > 0, (alpha / se) ** 2, 0.0)
    factor = np.where(np.isfinite(t2), t2 / (1.0 + t2), 0.0)
    return alpha * factor


# ------------------------------------------------------- portfolio operations


def neutralise(w: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Make the active weights sum to zero, leaving inactive names untouched."""
    out = np.where(active, w, 0.0)
    n = int(active.sum())
    if n == 0:
        return out
    out[active] = out[active] - out[active].mean()
    return out


def apply_band(current: np.ndarray, target: np.ndarray, band: np.ndarray,
               to_edge: bool = True) -> np.ndarray:
    """Impulse control: hold inside the band, otherwise move to its edge.

    ``to_edge=False`` moves all the way to target, which is what a naive
    implementation does and what the experiments compare against.
    """
    gap = target - current
    outside = np.abs(gap) > band
    if not to_edge:
        return np.where(outside, target, current)
    # Move only as far as the near edge of the no-trade region.
    step = gap - np.sign(gap) * band
    return np.where(outside, current + step, current)


def predicted_vol(w: np.ndarray, cov: np.ndarray) -> float:
    """Ex-ante portfolio volatility, per period, from a covariance matrix."""
    w = np.nan_to_num(np.asarray(w, dtype=float))
    var = float(w @ np.asarray(cov, dtype=float) @ w)
    return float(np.sqrt(max(var, 0.0)))


@dataclass
class SizingDiagnostics:
    """What the engine did on one date, for attribution."""

    gross: float
    net: float
    n_active: int
    n_traded: int
    turnover: float
    predicted_vol_annual: float
    scale: float
    n_eff: float
    mean_shrinkage: float


class SizingEngine:
    """Applies the control above to a cross-section, one date at a time."""

    def __init__(self, params: SizingParams | None = None):
        self.p = params or SizingParams()

    def size(self, *, alpha, sigma, cov, current, cost, alpha_se=None,
             active=None) -> tuple[np.ndarray, SizingDiagnostics]:
        """Return next holdings and diagnostics.

        Parameters are all per-name arrays over one cross-section. ``cov`` is the
        per-period return covariance. ``current`` is yesterday's holding, which is
        what the no-trade band is measured against. ``cost`` is the one-way
        trading cost in return units (e.g. 20bps = 0.0020).
        """
        alpha = np.asarray(alpha, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        current = np.asarray(current, dtype=float)
        n = alpha.size

        if active is None:
            active = np.isfinite(alpha) & np.isfinite(sigma) & (sigma > 0)
        active = np.asarray(active, dtype=bool) & np.isfinite(alpha) & (sigma > 0)
        alpha = np.where(active, np.nan_to_num(alpha), 0.0)
        sigma = np.where(active & np.isfinite(sigma), sigma, np.nan)

        # 1. shrink the signal by how well it is measured
        if alpha_se is not None and self.p.use_shrinkage:
            shrunk = shrink_by_tstat(alpha, np.asarray(alpha_se, dtype=float))
            denom = np.where(np.abs(alpha) > 0, np.abs(alpha), np.nan)
            ratio = np.abs(shrunk) / denom
            mean_shrink = (float(np.nanmean(ratio)) if np.isfinite(ratio).any() else 0.0)
        else:
            shrunk, mean_shrink = alpha, 1.0

        # 2. the control's target position: 1/sigma scaling and decay discount.
        #
        #    A subtlety worth naming: alpha arrives in Grinold units, IC*sigma*z,
        #    which already carries sigma. Dividing by |h2| ~ sigma therefore
        #    cancels most of it, leaving weights proportional to z discounted by
        #    1/(beta + rho(sigma)). So the per-name volatility penalty is real but
        #    weaker than a raw 1/sigma reading of the formula suggests -- the
        #    strong version applies when alpha is expressed in score units. The
        #    portfolio-level volatility target below is what actually pins risk.
        safe_sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)
        if not self.p.vol_scaling:
            # Ablation: one common sigma for everyone, so no cross-sectional
            # volatility scaling survives anywhere in the control.
            safe_sigma = np.full_like(safe_sigma, float(np.mean(safe_sigma[active]))
                                      if active.any() else 1.0)
        target = self.p.target_position(shrunk, safe_sigma)
        target = np.where(active, target, 0.0)

        # 3. dollar-neutral, then scaled to the risk budget. Neutralising before
        #    the band keeps the band applied to the trades actually intended.
        #
        #    The confidence factor is passed into the risk budget rather than
        #    left to act on alpha alone. It has to be: volatility targeting is
        #    scale-invariant, so it divides any shrinkage straight back out and
        #    re-levers a signal that shrinkage had just declared worthless. That
        #    failure was found by the dumb-signal experiment, where a pure-noise
        #    signal was shrunk to 11% of its size and then scaled back up by a
        #    factor of 1e46 to hit the volatility target. Risk deployed is now
        #    proportional to confidence, which is what shrinkage was supposed to
        #    mean in the first place.
        conf = float(np.clip(mean_shrink, 0.0, 1.0)) if np.isfinite(mean_shrink) else 0.0
        target = neutralise(target, active)
        target, scale, n_eff, vol_ann = self._risk_budget(target, cov, active, conf)

        # 4. no-trade band against the *current* book
        band = self.p.no_trade_band(cost, safe_sigma) if self.p.use_band else np.zeros(n)
        band = np.where(active, band, 0.0)
        new = apply_band(current, target, band, to_edge=self.p.move_to_band_edge)
        new = np.where(active | (np.abs(current) > 0), new, 0.0)

        # Names that dropped out of the universe must be closed, band or no band.
        new = np.where(active, new, 0.0)

        # 5. caps, then a final neutrality pass. Enforcing neutrality after the
        #    band perturbs banded positions, so the residual is measured below
        #    rather than assumed away.
        new = np.clip(new, -self.p.max_weight, self.p.max_weight)
        new = neutralise(new, active)
        gross = float(np.abs(new).sum())
        if gross > self.p.max_gross and gross > 0:
            new = new * (self.p.max_gross / gross)

        traded = np.abs(new - current) > 1e-12
        diag = SizingDiagnostics(
            gross=float(np.abs(new).sum()),
            net=float(new.sum()),
            n_active=int(active.sum()),
            n_traded=int(traded.sum()),
            turnover=float(np.abs(new - current).sum()),
            predicted_vol_annual=vol_ann,
            scale=scale,
            n_eff=n_eff,
            mean_shrinkage=mean_shrink if np.isfinite(mean_shrink) else 0.0,
        )
        return new, diag

    # ------------------------------------------------------------- internals

    def _risk_budget(self, w, cov, active, confidence: float = 1.0):
        """Scale the book to its risk budget.

        Three separate quantities, kept separate on purpose:

        *Volatility* is predicted from the **full** covariance, not the residual
        one. An earlier version used the residual, on the reasoning that a
        dollar-neutral book has already shed its market exposure and sizing off
        the raw covariance would double-count it. That reasoning is wrong: if the
        book really is neutral then the factor term ``(beta'w)^2 sigma_m^2`` is
        already near zero and the full covariance gives the same answer by
        itself. Using the residual instead *understates* risk by exactly the
        factor exposure the book has failed to neutralise -- which is the one
        case where you most want to know about it.

        *Effective breadth* is computed on the residual correlation, because that
        is genuinely the right matrix for counting independent bets once the
        common factor is projected out.

        *Confidence* multiplies the target rather than the weights, so that a
        signal the shrinkage has declared unmeasurable is traded small -- see the
        note in ``size``.
        """
        idx = np.flatnonzero(active)
        if idx.size == 0 or cov is None:
            return w, 0.0, 0.0, 0.0
        sub = np.asarray(cov, dtype=float)[np.ix_(idx, idx)]
        if not np.all(np.isfinite(sub)):
            return w, 0.0, 0.0, 0.0

        n_eff = effective_n_participation(residualise_covariance(sub, n_factors=1))
        vol = predicted_vol(w[idx], sub)
        if vol <= 0:
            return np.zeros_like(w), 0.0, float(n_eff), 0.0

        # Risk actually deployed = target x confidence x breadth haircut.
        target_vol = self.p.target_vol * float(np.clip(confidence, 0.0, 1.0))
        if self.p.breadth_haircut and idx.size > 0:
            target_vol *= float(np.sqrt(max(n_eff, 1.0) / idx.size))
        daily_target = target_vol / np.sqrt(self.p.trading_days)

        scale = daily_target / vol
        out = w * scale
        gross = float(np.abs(out).sum())
        if gross > self.p.max_gross and gross > 0:
            out = out * (self.p.max_gross / gross)
            scale *= self.p.max_gross / gross
        vol_ann = predicted_vol(out[idx], sub) * np.sqrt(self.p.trading_days)
        return out, float(scale), float(n_eff), float(vol_ann)


def rolling_covariance(returns: pd.DataFrame, window: int = 120,
                       min_periods: int | None = None) -> dict:
    """Trailing sample covariance per date, shifted so date t uses data < t.

    Returned as a dict keyed by date to keep memory sane; only dates with enough
    history appear.
    """
    min_periods = min_periods or max(20, window // 3)
    out = {}
    vals = returns.to_numpy(dtype=float)
    dates = returns.index
    for i in range(len(dates)):
        lo = max(0, i - window)
        block = vals[lo:i]                       # strictly before date i
        if block.shape[0] < min_periods:
            continue
        ok = np.isfinite(block).sum(axis=0) >= min_periods
        cov = np.full((block.shape[1], block.shape[1]), np.nan)
        if ok.sum() >= 2:
            sub = np.nan_to_num(block[:, ok], nan=0.0)
            c = np.cov(sub, rowvar=False)
            cov[np.ix_(ok, ok)] = c
        out[dates[i]] = cov
    return out
