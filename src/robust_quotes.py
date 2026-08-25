"""Robust market-making quotes under model uncertainty.

Implements Cartea, Donnelly and Jaimungal, "Algorithmic Trading with Model
Uncertainty" (SSRN 2310645).

The market maker maximises terminal wealth but does not trust her reference
model.  She plays a sup-inf over an entropy-penalised set of equivalent measures
(eq. 11), with *three separate* ambiguity budgets:

    phi_alpha  aversion to misspecification of the midprice drift
    phi_lam    aversion to misspecification of the market-order arrival rate
    phi_kap    aversion to misspecification of the limit-order fill probability

The value function admits the ansatz H(t, x, q, S) = x + q S + h_q(t), and
Proposition 3 gives the optimal depths in feedback form:

    delta_+*(q, t) = ( (1/phi_kap) log(1 + phi_kap/kappa_+) - h_{q-1} + h_q )_+
    delta_-*(q, t) = ( (1/phi_kap) log(1 + phi_kap/kappa_-) - h_{q+1} + h_q )_+
    eta*(q, t)     = alpha - phi_alpha sigma^2 q          (worst-case drift)

Scope of this implementation.  The paper gives a *closed form* for h only under
the symmetry conditions of Proposition 5: kappa_+ = kappa_- = kappa and
phi_lam = phi_kap = phi.  We solve exactly that case, so the two fill-side
budgets are tied to a single knob ``phi``; ``phi_alpha`` stays free.  This is
sufficient for the strategy, because the two knobs already push the quotes in
opposite directions (see below), which is the whole economic content we need.

Signs, which are what the strategy trades on:

  * phi_alpha up  -> the MM prices as if the midprice drifts against her
    inventory (eta* = alpha - phi_alpha sigma^2 q).  Total spread *widens* and
    inventory mean-reverts to zero faster.  Section 4.1 shows this is exactly
    equivalent to a running inventory penalty of phi = phi_alpha/2, so it is the
    defensive / inventory-control lever.
  * phi up (fill and arrival ambiguity together) -> the base depth
    (1/phi) log(1 + phi/kappa) is strictly below the ambiguity-neutral 1/kappa,
    so the MM quotes *tighter* and churns more.  It is the aggressive lever.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RobustParams:
    """Reference model plus ambiguity budgets."""

    kappa: float = 27.0        # fill-rate decay: fill intensity is lam * exp(-kappa * delta)
    lam_buy: float = 2.0       # lambda_+, buy MOs that lift our ask
    lam_sell: float = 2.0      # lambda_-, sell MOs that hit our bid
    sigma: float = 0.01        # reference midprice volatility
    alpha: float = 0.0         # reference midprice drift
    theta: float = 0.001       # liquidation penalty, l(q) = theta * q
    q_max: int = 8             # inventory bounds, q in [-q_max, q_max]
    phi_alpha: float = 0.0     # ambiguity aversion to drift
    phi: float = 0.0           # ambiguity aversion to arrival rate and fill prob (tied)

    def __post_init__(self) -> None:
        for name in ("kappa", "lam_buy", "lam_sell", "sigma"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.phi_alpha < 0 or self.phi < 0:
            raise ValueError("ambiguity parameters must be non-negative")

    @property
    def q_grid(self) -> np.ndarray:
        return np.arange(-self.q_max, self.q_max + 1)

    def base_depth(self) -> float:
        """The inventory-independent part of the optimal depth.

        (1/phi) log(1 + phi/kappa), which decreases in phi and tends to the
        ambiguity-neutral 1/kappa as phi -> 0.
        """
        if self.phi <= 1e-12:
            return 1.0 / self.kappa
        return np.log1p(self.phi / self.kappa) / self.phi

    def xi(self) -> tuple[float, float]:
        """xi_+/-, Proposition 5: (1 + phi/kappa)^(-(1 + kappa/phi)) * lambda_+/-.

        As phi -> 0 this tends to lambda / e, matching the ambiguity-neutral
        reduction of the HJB equation.
        """
        if self.phi <= 1e-12:
            factor = 1.0 / np.e
        else:
            factor = np.exp(-(1.0 + self.kappa / self.phi) * np.log1p(self.phi / self.kappa))
        return factor * self.lam_buy, factor * self.lam_sell


def _generator(p: RobustParams) -> tuple[np.ndarray, float, float]:
    """The matrix A of Proposition 5, acting on omega_q = exp(kappa * h_q).

    Substituting the optimal controls into (23) and setting omega = exp(kappa h)
    linearises the system to d(omega)/dt = -A omega with

        (A omega)_q = xi_+ omega_{q-1} + xi_- omega_{q+1}
                      + (alpha kappa q - 0.5 kappa phi_alpha sigma^2 q^2) omega_q

    with the off-diagonal term dropped at the inventory bound where that side is
    not quoted.
    """
    q = p.q_grid.astype(float)
    n = q.size
    xi_p, xi_m = p.xi()
    A = np.zeros((n, n))
    A[np.arange(n), np.arange(n)] = p.alpha * p.kappa * q - 0.5 * p.kappa * p.phi_alpha * p.sigma**2 * q**2
    # omega_{q-1} coupling: present for every q except the lower bound (no ask there).
    A[np.arange(1, n), np.arange(0, n - 1)] = xi_p
    # omega_{q+1} coupling: present for every q except the upper bound (no bid there).
    A[np.arange(0, n - 1), np.arange(1, n)] = xi_m
    return A, xi_p, xi_m


def solve_h_terminal(p: RobustParams, T: float, n_steps: int = 2000) -> np.ndarray:
    """Finite-horizon h_q(t) on a time grid, integrating omega backward from T.

    Returns an array of shape (n_steps + 1, n_q) with row 0 at t = 0 and row
    ``n_steps`` at t = T.  Terminal condition h_q(T) = -q l(q) = -theta q^2.
    """
    A, _, _ = _generator(p)
    q = p.q_grid.astype(float)
    dt = T / n_steps
    omega = np.exp(-p.kappa * p.theta * q**2)
    out = np.zeros((n_steps + 1, q.size))
    out[n_steps] = np.log(omega) / p.kappa
    for i in range(n_steps - 1, -1, -1):
        # d(omega)/dt = -A omega, stepped backward in time.
        omega = omega + dt * (A @ omega)
        omega = np.clip(omega, 1e-300, None)
        out[i] = np.log(omega) / p.kappa
    return out


def solve_h_stationary(p: RobustParams) -> np.ndarray:
    """Stationary inventory profile h_q - h_0, for a market maker with no fixed horizon.

    A live market maker runs indefinitely, so the useful object is not h itself
    (which grows without bound as the horizon recedes, at the ergodic rate) but
    the *differences* h_{q+/-1} - h_q, which are exactly what enters the optimal
    depths.  Since omega(t) = exp(A (T - t)) omega(T), letting the horizon recede
    drives the direction of omega to the dominant eigenvector of A.  A is
    tridiagonal with strictly positive off-diagonals, so A + cI is non-negative
    and irreducible for large enough c; Perron-Frobenius then gives a real
    dominant eigenvalue with a strictly positive eigenvector.  We take it
    directly rather than power-iterating.
    """
    A, _, _ = _generator(p)
    mid = p.q_grid.size // 2
    vals, vecs = np.linalg.eig(A)
    omega = np.real(vecs[:, int(np.argmax(np.real(vals)))])
    if omega[mid] < 0:
        omega = -omega
    if np.any(omega <= 0):
        raise RuntimeError("dominant eigenvector of the h-generator is not strictly positive")
    return np.log(omega / omega[mid]) / p.kappa


def optimal_depths(p: RobustParams, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal ask and bid depths from Proposition 3, given the profile h_q.

    Returns (delta_ask, delta_bid), each of length ``len(p.q_grid)``.  NaN marks
    the inventory bound on the side the MM is not allowed to quote: at q = q_max
    she posts no bid, at q = -q_max no ask.
    """
    h = np.asarray(h, dtype=float)
    base = p.base_depth()
    ask = np.full(h.shape, np.nan)
    bid = np.full(h.shape, np.nan)
    # Ask fill takes q -> q-1, so it references h_{q-1}; undefined at the lower bound.
    ask[1:] = np.maximum(base - h[:-1] + h[1:], 0.0)
    # Bid fill takes q -> q+1, so it references h_{q+1}; undefined at the upper bound.
    bid[:-1] = np.maximum(base - h[1:] + h[:-1], 0.0)
    return ask, bid


def worst_case_drift(p: RobustParams, q: float) -> float:
    """eta*(q) = alpha - phi_alpha sigma^2 q, eq. (24).

    The drift the ambiguity-averse MM prices against: adverse to whatever
    inventory she is holding, which is what accelerates mean reversion to flat.
    """
    return p.alpha - p.phi_alpha * p.sigma**2 * q


class RobustQuoter:
    """Stationary robust quoting policy: inventory -> (ask depth, bid depth).

    Caches the h profile and recomputes only when parameters actually change,
    so the live strategy can re-solve on every ambiguity update cheaply.
    """

    def __init__(self, params: RobustParams):
        self.params = params
        self._key: tuple | None = None
        self._ask = self._bid = self._h = None
        self.refresh()

    def _cache_key(self) -> tuple:
        p = self.params
        return (p.kappa, p.lam_buy, p.lam_sell, p.sigma, p.alpha, p.theta,
                p.q_max, p.phi_alpha, p.phi)

    def refresh(self) -> None:
        key = self._cache_key()
        if key == self._key:
            return
        self._h = solve_h_stationary(self.params)
        self._ask, self._bid = optimal_depths(self.params, self._h)
        self._key = key

    def update(self, **kwargs) -> None:
        """Update reference-model or ambiguity parameters and re-solve if needed."""
        for k, v in kwargs.items():
            if not hasattr(self.params, k):
                raise AttributeError(f"unknown parameter {k!r}")
            setattr(self.params, k, v)
        self.refresh()

    @property
    def h(self) -> np.ndarray:
        return self._h

    def depths(self, q: int) -> tuple[float, float]:
        """(ask depth, bid depth) at inventory ``q``; NaN where that side is barred."""
        i = int(q) + self.params.q_max
        if not 0 <= i < self.params.q_grid.size:
            raise ValueError(f"inventory {q} outside [-{self.params.q_max}, {self.params.q_max}]")
        return float(self._ask[i]), float(self._bid[i])

    def curves(self) -> tuple[np.ndarray, np.ndarray]:
        return self._ask.copy(), self._bid.copy()

    def total_depth(self) -> np.ndarray:
        """delta_+ + delta_- across inventories, the quantity in the paper's right-hand panels."""
        return self._ask + self._bid
