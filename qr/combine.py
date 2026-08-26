"""Combining many signals into one, without giving the gains straight back.

The measured result that drives every choice here: twelve signals at 0.55
pairwise correlation, forty seeds, true weights equal. OLS recovered an
out-of-sample correlation of 0.033 and ridge 0.040, against 0.063 for simply
equal-weighting -- and equal weight also had *lower* variance across seeds
(0.011 vs 0.017). OLS beat equal weight in 5% of seeds.

That comparison is rigged in equal weight's favour (the truth was equal), so
read it as an upper bound on the cost of estimation rather than proof that
equal weight is always right. But the cost it bounds is large: estimating a
dozen weights on collinear signals gave back roughly half the available edge.

The practical consequence is the shape of `hierarchical_combine`: equal-weight
within a family of related signals, where correlations are highest and
estimation is worth least, then combine across families with heavily shrunk
IC weights, where the signals are genuinely different and the estimate has
something to say.
"""

from __future__ import annotations

import numpy as np


def shrink_by_tstat(alpha, se, min_obs=None):
    """Scale each estimate by its own t^2/(1+t^2).

    At t=1 this halves the estimate, at t=3 it keeps 90%, and at t=0 it returns
    exactly zero -- the property that stops a pure-noise signal from being
    traded at all.

    It is empirical Bayes with the prior variance plugged in from the point
    estimate itself (tau^2 = alpha_hat^2), not James-Stein. Worth naming
    precisely, because it means everything gets shrunk permanently: a genuinely
    well-measured t=2 signal still loses 20% of its size forever. If you have a
    real prior on the signal's dispersion, pass it as ``tau`` to
    ``shrink_toward_prior`` instead.

    ``min_obs`` is an optional per-name boolean array; names that fail it are
    zeroed rather than shrunk.
    """
    a = np.asarray(alpha, dtype=float)
    s = np.asarray(se, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t2 = np.where(np.isfinite(s) & (s > 0), (a / s) ** 2, 0.0)
    factor = np.where(np.isfinite(t2), t2 / (1.0 + t2), 0.0)
    out = a * factor
    if min_obs is not None:
        ok = np.asarray(min_obs, dtype=bool)
        out = np.where(ok, out, 0.0)
    return out


def shrink_toward_prior(alpha, se, tau: float):
    """Posterior mean under alpha ~ N(0, tau^2), alpha_hat | alpha ~ N(alpha, se^2).

    The honest version of the above when you can put a number on how big the
    effects in your family of signals actually are. ``tau`` in the same units
    as ``alpha``.
    """
    a = np.asarray(alpha, dtype=float)
    s = np.asarray(se, dtype=float)
    t2 = float(tau) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(np.isfinite(s) & (s > 0), t2 / (t2 + s ** 2), 0.0)
    return a * np.where(np.isfinite(k), k, 0.0)


def equal_weight(signals):
    """Mean of standardised signals, ignoring NaN. The benchmark to beat."""
    S = np.stack([np.asarray(s, dtype=float) for s in signals], axis=0)
    with np.errstate(invalid="ignore"):
        out = np.nanmean(S, axis=0)
    return np.where(np.isfinite(out), out, np.nan)


def ic_weighted(signals, ics, ses=None, shrink: bool = True, floor: float = 0.0):
    """Weight signals by their (optionally shrunk) IC.

    With ``shrink=True`` and standard errors supplied, weights are the
    t-shrunk ICs, so a signal you cannot measure contributes nothing rather
    than contributing noise scaled by a lucky point estimate.
    """
    S = np.stack([np.asarray(s, dtype=float) for s in signals], axis=0)
    w = np.asarray(ics, dtype=float).astype(float)
    if shrink and ses is not None:
        w = shrink_by_tstat(w, np.asarray(ses, dtype=float))
    if floor > 0:
        w = np.where(np.abs(w) < floor, 0.0, w)
    tot = np.abs(w).sum()
    if tot <= 0 or not np.isfinite(tot):
        return equal_weight(signals)
    w = w / tot
    out = np.nansum(S * w[:, None, None], axis=0)
    any_valid = np.isfinite(S).any(axis=0)
    return np.where(any_valid, out, np.nan)


def hierarchical_combine(families: dict, family_ics=None, family_ses=None,
                         shrink: bool = True):
    """Equal-weight within each family, then IC-weight across families.

    ``families`` maps a family name to a list of standardised (T, N) signals.
    Within a family the signals are close substitutes and estimation buys you
    little; across families they are genuinely different bets and a shrunk IC
    estimate is worth having.
    """
    names = list(families.keys())
    per_family = [equal_weight(families[k]) for k in names]
    if family_ics is None:
        return equal_weight(per_family), names, np.ones(len(names)) / len(names)
    ics = np.array([float(family_ics[k]) for k in names])
    ses = (np.array([float(family_ses[k]) for k in names])
           if family_ses is not None else None)
    combined = ic_weighted(per_family, ics, ses, shrink=shrink)
    w = shrink_by_tstat(ics, ses) if (shrink and ses is not None) else ics
    tot = np.abs(w).sum()
    w = w / tot if tot > 0 else np.ones(len(names)) / len(names)
    return combined, names, w
