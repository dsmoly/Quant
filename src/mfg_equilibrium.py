"""Mean-field Nash equilibrium for a dealer market.

Implements the finite-state mean field game of Assayag, Barzykin, Cont and Xiong,
"Competition and Learning in Dealer Markets" (SSRN 4838181), sections 2 and 3.1.

A representative dealer quotes centred spreads (delta_a, delta_b) on top of a
reference price.  It wins the next RFQ with probability f(delta, mu), where mu is
the *average quote of the whole population* -- the only channel through which
competitors are felt.  Equilibrium is the fixed point where the population
strategy and the representative dealer's best response coincide.

The equilibrium is characterised by the coupled system (2.25): a backward
Hamilton-Jacobi equation for the value function V(t, q), a forward
Chapman-Kolmogorov equation for the inventory density m(t, q), and a fixed point
for the mean quotes mu_a(t), mu_b(t).  We solve it with the Picard iteration of
Algorithm 1, integrating until the flows stop moving so the result approximates
the stationary (T -> infinity) case the paper studies.

What the trading strategy uses this for: delta_star(q) is the *competitive*
spread.  A market quoting materially above it is in what the paper calls the
supra-competitive regime (tacit collusion).  The live strategy measures the
market's distance from this benchmark and reacts to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class IntensityParams:
    """Execution-rate function f(delta, mu) of eq. (3.1).

        f(delta, mu) = 1 / (C * exp(delta) + C_m * exp(k_m * delta - k * mu))

    Increasing in mu (wide competitors -> we win more), decreasing in delta.
    Bounded above by Lambda(delta) = exp(-delta) / C, the monopolistic rate.
    """

    C: float = 1.0
    C_m: float = 1.0
    k_m: float = 3.0
    k: float = 2.0

    def f(self, delta, mu):
        return 1.0 / (self.C * np.exp(delta) + self.C_m * np.exp(self.k_m * delta - self.k * mu))

    def monopolistic(self, delta):
        """Lambda(delta) of Assumption 2.2(3) -- the no-competition upper bound."""
        return np.exp(-delta) / self.C


@dataclass
class MFGConfig:
    Z: int = 10                    # inventory limit, q in {-Z, ..., Z}
    lam_a: float = 5.0             # ask-side RFQ arrival intensity
    lam_b: float = 5.0             # bid-side RFQ arrival intensity
    r: float = 0.01                # discount rate
    dt: float = 0.1
    max_steps: int = 60_000        # cap on backward/forward Euler steps
    tol: float = 1e-9              # stationarity tolerance on d/dt
    delta_floor: float = -1.0      # -delta_infinity, lower bound on quotes
    inventory_cost: float = 0.01   # psi(q) = inventory_cost * q^2
    terminal_cost: float = 0.01    # phi(q) = terminal_cost * q^2
    ask: IntensityParams = field(default_factory=IntensityParams)
    bid: IntensityParams = field(default_factory=IntensityParams)

    @property
    def q_grid(self) -> np.ndarray:
        return np.arange(-self.Z, self.Z + 1)

    def psi(self, q: np.ndarray) -> np.ndarray:
        return self.inventory_cost * q.astype(float) ** 2

    def phi(self, q: np.ndarray) -> np.ndarray:
        return self.terminal_cost * q.astype(float) ** 2


def best_response_quote(
    p: np.ndarray,
    mu: float,
    params: IntensityParams,
    delta_floor: float,
    warm: np.ndarray | None = None,
    n_iter: int = 12,
) -> np.ndarray:
    """Xi(p, mu) = argmax_{delta > -delta_inf} f(delta, mu) * (delta - p), eq. (2.23).

    ``p`` is the value-function difference -- the dealer's shadow price for a unit
    of inventory.  Writing f = 1/g with g(delta) = C e^delta + C_m e^(k_m delta - k mu),
    the first-order condition f'(delta)(delta - p) + f(delta) = 0 rearranges to the
    fixed point

        delta = p + g(delta) / g'(delta).

    Since g'/g is a convex combination of 1 and k_m, the ratio g/g' always lies in
    [1/k_m, 1], so the root is bracketed by [p + 1/k_m, p + 1] and the iteration
    contracts quickly.  Vectorised over ``p``, warm-startable across time steps.
    """
    p = np.asarray(p, dtype=float)
    delta = p + 0.5 * (1.0 + 1.0 / params.k_m) if warm is None else np.asarray(warm, dtype=float)
    for _ in range(n_iter):
        e1 = params.C * np.exp(delta)
        e2 = params.C_m * np.exp(params.k_m * delta - params.k * mu)
        g = e1 + e2
        gp = e1 + params.k_m * e2
        nxt = p + g / gp
        if np.max(np.abs(nxt - delta)) < 1e-13:
            delta = nxt
            break
        delta = nxt
    return np.maximum(delta, delta_floor)


def _joint_update(
    p: np.ndarray,
    m: np.ndarray,
    params: IntensityParams,
    cfg: MFGConfig,
    mu: float,
    warm: np.ndarray | None,
    n_iter: int = 4,
) -> tuple[float, np.ndarray]:
    """One damped pass at the coupled system mu = sum_q Xi(p_q, mu) m_q, eq. (2.28).

    Called every time step with ``mu`` and ``warm`` carried over, so the pair is
    effectively always converged even though each call is cheap.
    """
    quotes = warm
    for _ in range(n_iter):
        quotes = best_response_quote(p, mu, params, cfg.delta_floor, warm=quotes)
        mu_new = float(np.sum(quotes * m))
        if abs(mu_new - mu) < 1e-13:
            mu = mu_new
            break
        mu = 0.5 * mu + 0.5 * mu_new
    return mu, quotes


@dataclass
class MFGSolution:
    q_grid: np.ndarray
    value: np.ndarray          # stationary value function V(q)
    density: np.ndarray        # stationary inventory density m(q)
    ask_quote: np.ndarray      # delta_a_star(q)
    bid_quote: np.ndarray      # delta_b_star(q)
    mu_a: float                # population mean ask quote in equilibrium
    mu_b: float                # population mean bid quote in equilibrium
    residuals: list[float]

    @property
    def spread(self) -> np.ndarray:
        """Total quoted spread delta_a(q) + delta_b(q)."""
        return np.nan_to_num(self.ask_quote) + np.nan_to_num(self.bid_quote)

    def nash_half_spread(self) -> float:
        """Equilibrium half-spread at flat inventory: the competitive benchmark."""
        i = len(self.q_grid) // 2
        return 0.5 * float(self.ask_quote[i] + self.bid_quote[i])

    def mean_quote(self) -> float:
        """Population mean quote in equilibrium, averaged over the two sides."""
        return 0.5 * (self.mu_a + self.mu_b)


def solve_mfg(cfg: MFGConfig | None = None, n_picard: int = 8, verbose: bool = False) -> MFGSolution:
    """Picard fixed-point scheme (Algorithm 1) for the mean field Nash equilibrium."""
    cfg = cfg or MFGConfig()
    q = cfg.q_grid
    nq = q.size
    psi = cfg.psi(q)
    can_sell = q > -cfg.Z      # an ask fill lowers inventory; blocked at the floor
    can_buy = q < cfg.Z        # a bid fill raises inventory; blocked at the cap

    m = np.full(nq, 1.0 / nq)
    V = -cfg.phi(q)
    mu_a = mu_b = 0.0
    wa = wb = None
    residuals: list[float] = []

    for it in range(n_picard):
        V_prev, m_prev = V.copy(), m.copy()

        # --- backward value iteration, eq. (3.2), density held fixed ---
        V = -cfg.phi(q)
        for _ in range(cfg.max_steps):
            p_a = np.where(can_sell, V - np.roll(V, 1), 0.0)      # V(q) - V(q-1)
            p_b = np.where(can_buy, V - np.roll(V, -1), 0.0)      # V(q) - V(q+1)
            mu_a, wa = _joint_update(p_a, m, cfg.ask, cfg, mu_a, wa)
            mu_b, wb = _joint_update(p_b, m, cfg.bid, cfg, mu_b, wb)
            H_a = cfg.lam_a * cfg.ask.f(wa, mu_a) * (wa - p_a) * can_sell
            H_b = cfg.lam_b * cfg.bid.f(wb, mu_b) * (wb - p_b) * can_buy
            drift = cfg.r * V + psi - H_a - H_b
            V = V - cfg.dt * drift
            if np.max(np.abs(drift)) < cfg.tol:
                break

        # --- forward density iteration, eq. (3.3), value held fixed ---
        p_a = np.where(can_sell, V - np.roll(V, 1), 0.0)
        p_b = np.where(can_buy, V - np.roll(V, -1), 0.0)
        mu_a, wa = _joint_update(p_a, m, cfg.ask, cfg, mu_a, wa, n_iter=40)
        mu_b, wb = _joint_update(p_b, m, cfg.bid, cfg, mu_b, wb, n_iter=40)
        rate_up = cfg.lam_b * cfg.bid.f(wb, mu_b) * can_buy      # bid fill: q -> q+1
        rate_dn = cfg.lam_a * cfg.ask.f(wa, mu_a) * can_sell     # ask fill: q -> q-1

        m = np.full(nq, 1.0 / nq)
        for _ in range(cfg.max_steps):
            inflow_below = np.roll(rate_up * m, 1)
            inflow_below[0] = 0.0
            inflow_above = np.roll(rate_dn * m, -1)
            inflow_above[-1] = 0.0
            dm = inflow_below + inflow_above - (rate_up + rate_dn) * m
            m = np.clip(m + cfg.dt * dm, 0.0, None)
            m /= m.sum()
            if np.max(np.abs(dm)) < cfg.tol:
                break

        res = float(np.linalg.norm(V - V_prev) + np.linalg.norm(m - m_prev))
        residuals.append(res)
        if verbose:
            print(f"picard {it}: residual {res:.3e}  mu_a={mu_a:.4f} mu_b={mu_b:.4f}")
        if res < 1e-10:
            break

    p_a = np.where(can_sell, V - np.roll(V, 1), 0.0)
    p_b = np.where(can_buy, V - np.roll(V, -1), 0.0)
    mu_a, d_a = _joint_update(p_a, m, cfg.ask, cfg, mu_a, wa, n_iter=60)
    mu_b, d_b = _joint_update(p_b, m, cfg.bid, cfg, mu_b, wb, n_iter=60)
    d_a = np.where(can_sell, d_a, np.nan)
    d_b = np.where(can_buy, d_b, np.nan)

    return MFGSolution(q, V, m, d_a, d_b, mu_a, mu_b, residuals)


def monopolistic_quotes(cfg: MFGConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Benchmark quotes for a dealer facing no competition (intensity Lambda(delta)).

    Used as the paper's second benchmark (figures 4, 6, 7).  Learned strategies
    sitting between Nash and monopolistic are the supra-competitive regime.
    With Lambda(delta) = e^-delta / C the first-order condition is exact:
    delta* = 1 + p.
    """
    cfg = cfg or MFGConfig()
    q = cfg.q_grid
    psi = cfg.psi(q)
    can_sell = q > -cfg.Z
    can_buy = q < cfg.Z

    V = -cfg.phi(q)
    for _ in range(cfg.max_steps):
        p_a = np.where(can_sell, V - np.roll(V, 1), 0.0)
        p_b = np.where(can_buy, V - np.roll(V, -1), 0.0)
        d_a = np.maximum(1.0 + p_a, cfg.delta_floor)
        d_b = np.maximum(1.0 + p_b, cfg.delta_floor)
        H_a = cfg.lam_a * cfg.ask.monopolistic(d_a) * (d_a - p_a) * can_sell
        H_b = cfg.lam_b * cfg.bid.monopolistic(d_b) * (d_b - p_b) * can_buy
        drift = cfg.r * V + psi - H_a - H_b
        V = V - cfg.dt * drift
        if np.max(np.abs(drift)) < cfg.tol:
            break

    p_a = np.where(can_sell, V - np.roll(V, 1), 0.0)
    p_b = np.where(can_buy, V - np.roll(V, -1), 0.0)
    d_a = np.maximum(1.0 + p_a, cfg.delta_floor)
    d_b = np.maximum(1.0 + p_b, cfg.delta_floor)
    return np.where(can_sell, d_a, np.nan), np.where(can_buy, d_b, np.nan)
