"""Differential-entropy tail estimation with a Pareto kernel.

Ported from the replication package for Matsushita, Nobre, Da Silva & Brandao,
"Beyond Variance: Using Differential Entropy to Detect Financial Market Regimes"
(2025), with three deliberate changes noted below.

The idea
--------
For a Gaussian, differential entropy is a fixed function of variance:

    H = 0.5 * log(2 * pi * e * sigma^2)

so plotting exp(H) against sigma^2 puts Gaussian data exactly on one curve.
Anything off that curve carries structure variance alone does not describe --
in practice, heavier tails. The distance from the curve is a tail diagnostic
that needs no threshold choice, which is its main attraction next to Hill (which
needs k) and the generalised Pareto fit (which needs a threshold).

The Pareto kernel
-----------------
    K(u) = alpha * beta^alpha / (2 * (beta + |u|)^(alpha + 1))

a symmetric two-sided power law, normalised to integrate to one. ``beta`` plays
the role of a bandwidth and ``alpha`` is the kernel's own tail index. A Gaussian
kernel has thin tails and systematically under-weights the tails of heavy-tailed
data, biasing the entropy estimate; a power-law kernel matches the data's own
tail behaviour.

Both parameters are fitted by minimising the leave-one-out entropy estimate, and
alpha-hat is read off as the data's tail index. **That objective is not
unconventional, contrary to how it first reads.** Since

    H_hat = -(1/n) sum_i log f_{-i}(x_i)

minimising H_hat is identical to *maximising the leave-one-out log-likelihood*,
which is maximum-likelihood cross-validation -- a standard bandwidth selector.
The one real caveat is that MLCV is known to behave poorly for some heavy-tailed
densities, which is exactly the regime of interest, so the estimator is validated
against known answers rather than trusted.

Changes from the original R
---------------------------
1. **Trailing windows, not centred.** The original uses ``w.begin = t - 30,
   w.end = t + 30``, so the regime label at time t is computed from thirty days
   of *future* data. That is fine for dating regimes in a paper and fatal as a
   trading input. ``rolling_tail_index`` here is strictly backward-looking.
   ``centred=True`` is available solely so the size of the look-ahead advantage
   can be measured rather than argued about.
2. **Bounded optimiser.** The original calls plain Nelder-Mead with no
   constraints, so alpha and beta can wander into regions where the kernel is not
   a valid density. Bounds are enforced, with multiple starts.
3. **Ties handled always.** The v3 tie handling (collapse duplicates to unique
   values plus frequency weights) is the only sensible default for market data,
   where discrete tick sizes produce repeated returns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import optimize, special

GAUSS_ALPHA = 2.0          # the alpha at which tails stop being power-law
LOG2PIE = float(np.log(2.0 * np.pi * np.e))


# ------------------------------------------------------------------- kernels


def pareto_kernel(u, alpha: float, beta: float):
    """K(u) = a b^a / (2 (b + |u|)^(a+1)). Integrates to 1 over the real line."""
    u = np.abs(np.asarray(u, dtype=float))
    a, b = float(alpha), float(beta)
    return a * b ** a / (2.0 * (b + u) ** (a + 1.0))


def gaussian_kernel(u, h: float):
    u = np.asarray(u, dtype=float) / h
    return np.exp(-0.5 * u * u) / (h * np.sqrt(2.0 * np.pi))


def _unique_weights(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse ties into unique values plus counts (the v3 behaviour)."""
    vals, counts = np.unique(np.asarray(x, dtype=float), return_counts=True)
    return vals, counts.astype(float)


def kernel_entropy(x, alpha: float, beta: float) -> float:
    """Leave-one-out differential entropy under the Pareto kernel.

    H = -(1/n) sum_i log f_{-i}(x_i), with ties carried as frequency weights so
    a repeated value contributes its multiplicity without being its own
    neighbour.
    """
    vals, w = _unique_weights(x)
    n = float(w.sum())
    if n < 3 or vals.size < 2:
        return float("nan")
    d = np.abs(vals[:, None] - vals[None, :])
    K = pareto_kernel(d, alpha, beta)
    # Density at each unique point, excluding one copy of the point itself.
    dens = (K @ w - pareto_kernel(0.0, alpha, beta)) / (n - 1.0)
    dens = np.clip(dens, 1e-300, None)
    return float(-(w @ np.log(dens)) / n)


# -------------------------------------------------------- non-kernel baselines


def kozachenko_leonenko(x) -> float:
    """Nearest-neighbour (Kozachenko-Leonenko) entropy: refuses to run on ties."""
    v = np.sort(np.asarray(pd.Series(x).dropna(), dtype=float))
    n = v.size
    if n < 3:
        return float("nan")
    gaps = np.diff(v)
    if np.any(gaps <= 0):
        return float("nan")                 # ties: the estimator is undefined
    rho = np.empty(n)
    rho[0], rho[-1] = gaps[0], gaps[-1]
    rho[1:-1] = np.minimum(gaps[:-1], gaps[1:])
    return float(np.mean(np.log(n * rho)) + np.log(2.0) - special.digamma(1.0))


def spacings_entropy(x, m: int | None = None) -> float:
    """m-spacings entropy: H = mean log( (n/m) (x_{i+m} - x_i) )."""
    v = np.sort(np.asarray(pd.Series(x).dropna(), dtype=float))
    n = v.size
    if n < 5:
        return float("nan")
    m = int(m or max(1, round(np.sqrt(n))))
    d = v[m:] - v[:-m]
    if np.any(d <= 0):
        return float("nan")
    return float(np.mean(np.log(n / m * d)))


def gaussian_entropy(sigma: float) -> float:
    """The reference curve: H for a Gaussian of this standard deviation."""
    return float(0.5 * (LOG2PIE + 2.0 * np.log(max(sigma, 1e-300))))


# --------------------------------------------------------------------- fitting


@dataclass
class EntropyFit:
    alpha: float          # fitted tail index; < 2 is the heavy-tailed regime
    beta: float           # fitted bandwidth
    entropy: float        # H at the optimum
    sigma: float            # sample sd; NOT meaningful when alpha < 2
    gaussian_entropy: float
    excess_entropy: float   # H - H_gaussian(sigma); < 0 means more concentrated
    regime: str
    n: int
    converged: bool

    def as_dict(self) -> dict:
        return asdict(self)


def fit_tail_index(x, *, alpha_bounds=(0.15, 6.0), beta_bounds=None,
                   n_starts: int = 4) -> EntropyFit:
    """Fit (alpha, beta) by minimising the leave-one-out entropy (i.e. MLCV).

    Bounded and multi-start, unlike the original's unconstrained Nelder-Mead:
    an unbounded search can leave the region where the kernel is a valid density,
    and the objective is not convex, so a single start can land anywhere.
    """
    v = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = v.size
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    if n < 8 or not np.isfinite(sd) or sd <= 0:
        return EntropyFit(float("nan"), float("nan"), float("nan"), sd,
                          float("nan"), float("nan"), "undetermined", n, False)

    # Bandwidth scale from a ROBUST spread, not the standard deviation.
    #
    # This is not a refinement, it is a correctness requirement. The whole point
    # of the estimator is data with alpha < 2, and an alpha-stable law with
    # alpha < 2 has *infinite variance* -- the sample standard deviation is then
    # not estimating anything, it simply grows with the largest observation
    # drawn. Scaling the bandwidth search off it put the entire admissible range
    # of beta in the wrong place: on alpha = 0.8 data the fit pinned alpha-hat at
    # its upper bound of 6.0, i.e. reported the heaviest-tailed sample as
    # Gaussian. The interquartile range is finite for every alpha and fixes it.
    iqr = float(np.subtract(*np.percentile(v, [75, 25])))
    mad = float(np.median(np.abs(v - np.median(v))))
    scale = max(iqr / 1.349, mad / 0.6745, 1e-12)
    h0 = 1.06 * scale * n ** (-0.2)
    bb = beta_bounds or (h0 / 100.0, h0 * 100.0)

    def obj(p):
        a, b = p
        if not (alpha_bounds[0] <= a <= alpha_bounds[1] and bb[0] <= b <= bb[1]):
            return 1e6
        val = kernel_entropy(v, a, b)
        return 1e6 if not np.isfinite(val) else val

    best, best_val, ok = None, np.inf, False
    for a0 in np.linspace(alpha_bounds[0] + 0.3, min(3.5, alpha_bounds[1]), n_starts):
        for b0 in (h0 * 0.5, h0 * 2.0):
            try:
                r = optimize.minimize(obj, x0=[a0, b0], method="L-BFGS-B",
                                      bounds=[alpha_bounds, bb])
            except Exception:
                continue
            if r.fun < best_val:
                best, best_val, ok = r.x, float(r.fun), True
    if not ok:
        return EntropyFit(float("nan"), float("nan"), float("nan"), sd,
                          float("nan"), float("nan"), "undetermined", n, False)

    a, b = float(best[0]), float(best[1])
    hg = gaussian_entropy(sd)
    # The paper's rule. Kept as reported, but see the bake-off: at short windows
    # the alpha estimate carries a large enough error bar that this label is much
    # less decisive than a bare threshold makes it look.
    regime = "alpha-stable" if a < GAUSS_ALPHA else "Gaussian"
    return EntropyFit(alpha=a, beta=b, entropy=best_val, sigma=sd,
                      gaussian_entropy=hg, excess_entropy=best_val - hg,
                      regime=regime, n=n, converged=True)


def rolling_tail_index(x, window: int = 61, step: int = 1,
                       centred: bool = False) -> pd.DataFrame:
    """Tail index through time.

    Trailing by default: the value stamped at index t uses observations
    ``(t-window, t]`` and nothing later. ``centred=True`` reproduces the paper's
    look-ahead construction and exists only so the advantage it confers can be
    measured.
    """
    s = pd.Series(x).dropna()
    v = s.to_numpy(dtype=float)
    n = v.size
    half = window // 2
    rows = []
    for t in range(0, n, max(1, step)):
        if centred:
            lo, hi = t - half, t + half + 1
        else:
            lo, hi = t - window + 1, t + 1
        if lo < 0 or hi > n:
            continue
        f = fit_tail_index(v[lo:hi])
        rows.append({"index": s.index[t], "alpha": f.alpha, "beta": f.beta,
                     "entropy": f.entropy, "sigma": f.sigma,
                     "excess_entropy": f.excess_entropy, "regime": f.regime})
    return pd.DataFrame(rows).set_index("index") if rows else pd.DataFrame()


# ------------------------------------------------------------ stable sampling


def rvs_symmetric_stable(alpha: float, size: int, scale: float = 1.0,
                         rng=None) -> np.ndarray:
    """Symmetric alpha-stable draws by Chambers-Mallows-Stuck.

    Used to generate data whose tail index is known exactly, which is what makes
    the bake-off a validation rather than a comparison of opinions.
    """
    rng = rng or np.random.default_rng()
    u = rng.uniform(-np.pi / 2, np.pi / 2, size)
    w = rng.exponential(1.0, size)
    if abs(alpha - 1.0) < 1e-9:
        x = np.tan(u)
    else:
        x = (np.sin(alpha * u) / np.cos(u) ** (1.0 / alpha)
             * (np.cos(u - alpha * u) / w) ** ((1.0 - alpha) / alpha))
    return scale * x
