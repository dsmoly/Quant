"""Robust directional trading on an order-flow signal.

The same framework, pointed the other way
-----------------------------------------
Cartea, Donnelly and Jaimungal note that their robust formulation "can also be
applied to algorithmic trading scenarios other than market making ... and other
strategies that aim to profit from price predictions".  This module does exactly
that: it keeps the ambiguity-averse machinery and replaces the market maker with
a *taker* who pays the spread to hold a directional position.

The signal is the one the market-making side was losing money to.  In the true
dynamics of the simulator (and of the paper's section 5) the midprice carries a
transient drift driven by market-order flow,

    dS     = alpha_t dt + sigma dW
    dalpha = -beta alpha dt + eps (dM+ - dM-)

so every market order tilts the price in its own direction for a while.  To the
market maker that tilt is adverse selection -- the cost that ate 5% of gross
spread under benign flow and 58% under toxic flow.  To a directional trader it is
revenue.  The two strategies are opposite sides of the same wedge, and which one
is viable is decided by whether the drift on offer outruns the spread you must
cross to get it.

The control problem
-------------------
Position q, trading rate nu = dq/dt, quadratic impact k, and the robust
formulation's entropic penalty on the drift.  Marking to market,

    d(X + qS) = q dS - k nu^2 dt

and the inner infimum over the drift is the same one the market-making paper
solves in eq. (24):

    inf_eta { q eta + (alpha - eta)^2 / (2 phi_a sigma^2) }
        = q alpha - (1/2) phi_a sigma^2 q^2      at  eta* = alpha - phi_a sigma^2 q.

So drift ambiguity is again exactly a running penalty on the position held --
here it is what stops the trader from taking an unbounded bet on the signal.  The
ergodic Hamilton-Jacobi-Bellman equation for the value h(q, alpha) is

    Gamma = sup_nu { nu h_q - k nu^2 } + q alpha - (1/2) phi_a sigma^2 q^2
            - beta alpha h_alpha + (1/2) sigma_alpha^2 h_alpha_alpha

which the quadratic ansatz h = (1/2) h2 q^2 + h1 q alpha + (1/2) h0 alpha^2
solves in closed form:

    h2 = -sqrt(2 k phi_a sigma^2)          (negative root; the concave one)
    h1 = 1 / (beta + rho),   rho = -h2/(2k) = sqrt(phi_a sigma^2 / (2k))
    h0 = h1^2 / (4 k beta)

Everything the strategy does follows from those three numbers:

    target position   q*(alpha) = alpha / ((beta + rho) * |h2|)
    trading rate      nu* = -rho (q - q*)
    marginal value    dh/dq = h2 (q - q*)

and, once a *linear* cost c per unit is charged as well -- which is what crossing
a dealer's spread actually is -- the first-order condition ``|dh/dq| > c`` turns
into a no-trade band around the target of half-width

    b = c / |h2|.

Read the levers:

  * ``phi_alpha`` is the position-sizing knob.  q* scales as 1/sqrt(phi_alpha):
    the more the trader distrusts her own drift estimate, the smaller the bet.
    At phi_alpha -> 0 the target diverges, which is the correct statement that a
    risk-neutral trader with a real edge and no penalty would take an unbounded
    position.
  * q* scales as 1/sigma, so volatility targeting falls out rather than being
    bolted on.
  * q* is discounted by (beta + rho): a signal that decays faster than you can
    trade into it is worth less.
  * The band is proportional to the half-spread, which is where the dealer-market
    paper enters.  When the dealer population quotes wide of the mean-field Nash
    level, crossing costs more, the band widens, and the trader does less.  That
    channel is mechanical -- it is the price actually paid -- rather than the
    regime switch that the market-making experiments refuted.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class DirectionalParams:
    """Reference model and preferences for the robust directional trader."""

    sigma: float = 0.01          # midprice volatility
    beta: float = 1.0            # decay rate of the alpha signal
    k: float = 0.02              # quadratic (temporary) impact coefficient
    phi_alpha: float = 40.0      # ambiguity aversion to the drift; the sizing knob
    sigma_alpha: float = 0.0     # volatility of the alpha process (affects only Gamma)
    q_max: int = 8               # position limit

    def __post_init__(self) -> None:
        if self.sigma <= 0 or self.k <= 0 or self.beta <= 0:
            raise ValueError("sigma, k and beta must be positive")
        if self.phi_alpha <= 0:
            raise ValueError("phi_alpha must be positive: at zero the optimal "
                             "position is unbounded")
        if self.q_max <= 0:
            raise ValueError("q_max must be positive")

    # ------------------------------------------------------- closed-form terms

    @property
    def h2(self) -> float:
        """Curvature of the value function in the position; strictly negative."""
        return -np.sqrt(2.0 * self.k * self.phi_alpha * self.sigma**2)

    @property
    def rho(self) -> float:
        """Rate at which the optimal policy closes the gap to its target."""
        return -self.h2 / (2.0 * self.k)

    @property
    def h1(self) -> float:
        """Sensitivity of the value function to the signal."""
        return 1.0 / (self.beta + self.rho)

    @property
    def h0(self) -> float:
        return self.h1**2 / (4.0 * self.k * self.beta)

    @property
    def gamma(self) -> float:
        """Ergodic value rate -- the long-run P&L per unit time under the model."""
        return 0.5 * self.sigma_alpha**2 * self.h0

    # --------------------------------------------------------------- the policy

    def target_position(self, alpha: float | np.ndarray) -> float | np.ndarray:
        """q*(alpha), the position the trader would hold with no trading cost."""
        q = np.asarray(alpha, dtype=float) * self.h1 / abs(self.h2)
        return np.clip(q, -self.q_max, self.q_max)

    def unclipped_target(self, alpha: float | np.ndarray):
        """q*(alpha) before the position limit, for diagnostics and tests."""
        return np.asarray(alpha, dtype=float) * self.h1 / abs(self.h2)

    def trade_rate(self, q: float, alpha: float) -> float:
        """nu*(q, alpha) = -rho (q - q*), the frictionless optimal trading rate."""
        return -self.rho * (q - float(self.target_position(alpha)))

    def marginal_value(self, q: float, alpha: float) -> float:
        """dh/dq: what one more unit of position is worth at the margin."""
        return self.h2 * (q - float(self.unclipped_target(alpha)))

    def no_trade_band(self, half_spread: float) -> float:
        """Half-width of the no-trade band when a linear cost is charged per unit.

        A unit trade is worth ``|dh/dq|`` and costs ``half_spread``, so the trader
        moves only once the gap to target is wider than ``half_spread / |h2|``.
        """
        return max(float(half_spread), 0.0) / abs(self.h2)

    def hjb_residual(self, q, alpha) -> np.ndarray:
        """Residual of the ergodic HJB equation at the closed-form solution.

        Zero everywhere is what makes the closed form a solution rather than an
        approximation; the tests assert it to floating point on a grid.
        """
        q = np.asarray(q, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        h_q = self.h2 * q + self.h1 * alpha
        h_a = self.h1 * q + self.h0 * alpha
        supremum = h_q**2 / (4.0 * self.k)
        robust_running = q * alpha - 0.5 * self.phi_alpha * self.sigma**2 * q**2
        return (supremum + robust_running - self.beta * alpha * h_a
                + 0.5 * self.sigma_alpha**2 * self.h0 - self.gamma)

    def worst_case_drift(self, q: float, alpha: float) -> float:
        """eta*(q) = alpha - phi_alpha sigma^2 q, the drift the trader prices against.

        Identical in form to the market-making case: the trader assumes the market
        is leaning against whatever position she already holds.
        """
        return alpha - self.phi_alpha * self.sigma**2 * q


# --------------------------------------------------------------- alpha signal


@dataclass
class AlphaEstimatorConfig:
    """Multi-timescale order-flow imbalance, fitted online to realised returns."""

    decays: tuple[float, ...] = (0.25, 1.0, 4.0)   # per-second decay rates
    rls_forgetting: float = 0.9995                 # recursive-least-squares forget factor
    rls_prior: float = 1e3                         # initial inverse-covariance scale
    horizon: float = 2.0                           # seconds ahead the fit targets
    max_alpha: float = 0.5                         # clamp on the fitted signal
    resid_halflife: float = 500.0                  # observations, for the residual variance


class OrderFlowAlpha:
    """Estimate the short-term drift from the tape.

    The trader does not know ``eps`` or ``beta``.  What she can see is the signed
    market-order stream, so she carries several exponentially weighted order-flow
    imbalances at different decay rates and fits their weights to realised forward
    returns by recursive least squares.  That handles an unknown decay without
    assuming one: whichever timescale actually predicts gets the weight.

    This is deliberately the *same* information set the market maker had -- order
    arrivals and midprices -- so the comparison between the two strategies is a
    comparison of what you do with the information, not of who sees more.
    """

    def __init__(self, cfg: AlphaEstimatorConfig | None = None):
        self.cfg = cfg or AlphaEstimatorConfig()
        n = len(self.cfg.decays)
        self.z = np.zeros(n)                       # the imbalance features
        self.coef = np.zeros(n)                    # fitted weights
        self.P = np.eye(n) * self.cfg.rls_prior    # RLS inverse covariance
        self._t = 0.0
        self._pending: deque[tuple[float, np.ndarray, float]] = deque()
        self.n_updates = 0
        self.resid_var = 0.0                       # EWMA of squared fit residuals

    def decay_to(self, t: float) -> None:
        """Advance the imbalance features to time ``t``."""
        dt = t - self._t
        if dt > 0:
            self.z = self.z * np.exp(-np.asarray(self.cfg.decays) * dt)
            self._t = t

    def observe_order(self, t: float, side: str) -> None:
        """Record a market order. ``side`` is 'buy' if it lifted the offer."""
        self.decay_to(t)
        self.z = self.z + (1.0 if side == "buy" else -1.0)

    def observe_mid(self, t: float, mid: float) -> None:
        """Feed a midprice observation and settle any fit targets that have matured.

        Each stored feature vector is scored against the return actually realised
        over the estimator's horizon, then folded in by recursive least squares.
        """
        self.decay_to(t)
        self._pending.append((t, self.z.copy(), mid))
        h = self.cfg.horizon
        while self._pending and t - self._pending[0][0] >= h:
            t0, z0, mid0 = self._pending.popleft()
            # Target: the average drift realised over the horizon.
            y = (mid - mid0) / h
            self._rls_update(z0, y)

    def _rls_update(self, x: np.ndarray, y: float) -> None:
        lam = self.cfg.rls_forgetting
        Px = self.P @ x
        denom = lam + float(x @ Px)
        if denom <= 0:
            return
        gain = Px / denom
        resid = y - float(x @ self.coef)
        self.coef = self.coef + gain * resid
        self.P = (self.P - np.outer(gain, Px)) / lam
        a = 1.0 - 0.5 ** (1.0 / max(self.cfg.resid_halflife, 1e-9))
        self.resid_var = (1 - a) * self.resid_var + a * resid**2
        self.n_updates += 1

    def _attenuation(self) -> np.ndarray:
        """Bias factor from fitting a decaying feature to an average forward return.

        The regression target is the mean drift realised over ``horizon`` h, but
        feature j decays at rate b_j across that window, so

            E[(S_{t+h} - S_t)/h | z] = c_j z_j (1 - e^{-b_j h}) / (b_j h)

        and the fitted coefficient is the instantaneous one shrunk by that factor
        -- about 0.63 at b = 1, h = 1. Dividing it back out is what makes ``alpha``
        an estimate of the drift *now* rather than of its average over the next
        few seconds, which is the quantity the control is written against.
        """
        b = np.asarray(self.cfg.decays, dtype=float)
        h = self.cfg.horizon
        return np.where(b * h > 1e-9, (1.0 - np.exp(-b * h)) / (b * h), 1.0)

    @property
    def alpha(self) -> float:
        """Current estimate of the instantaneous drift."""
        a = float(self.z @ (self.coef / self._attenuation()))
        return float(np.clip(a, -self.cfg.max_alpha, self.cfg.max_alpha))

    @property
    def alpha_se(self) -> float:
        """Standard error of the current drift estimate.

        The recursive least squares carries its own inverse-covariance ``P``, so
        the variance of the fitted value at the current features is
        ``resid_var * z' P z``, de-attenuated on the same scale as ``alpha``.
        """
        w = self.z / self._attenuation()
        var = self.resid_var * float(w @ (self.P @ w))
        return float(np.sqrt(max(var, 0.0)))

    @property
    def alpha_shrunk(self) -> float:
        """The drift estimate shrunk toward zero by how well it is measured.

        With ``t = alpha / se`` the estimate's own t-statistic, the shrinkage
        factor ``t^2 / (1 + t^2)`` leaves a well-measured signal essentially
        untouched and collapses a badly measured one to nothing.  This is the
        estimation-error counterpart of the paper's ambiguity aversion: the drift
        is distrusted precisely in proportion to how little the data pins it down,
        rather than by a fixed preference parameter.
        """
        a = self.alpha
        se = self.alpha_se
        if se <= 0 or self.n_updates < 2:
            return a
        t2 = (a / se) ** 2
        return a * t2 / (1.0 + t2)

    def effective_decay(self) -> float:
        """Weighted decay rate of the fitted signal, used as ``beta`` in the control.

        Weights can be negative once fitted, so the average is taken over their
        magnitudes; with no usable fit yet it falls back to the middle timescale.
        """
        w = np.abs(self.coef)
        if w.sum() <= 1e-12:
            return float(np.median(self.cfg.decays))
        return float(np.asarray(self.cfg.decays) @ w / w.sum())


# ------------------------------------------------------------------ the trader


@dataclass
class TraderConfig:
    """Wiring for the live directional trader."""

    phi_alpha: float = 40.0        # position-sizing knob
    k: float = 0.02                # quadratic impact coefficient
    sigma0: float = 0.01
    sigma_halflife: float = 30.0   # seconds
    warmup_updates: int = 200      # RLS fits required before any position is taken
    cost_multiple: float = 1.0     # scales the half-spread charged to the band
    q_max: int = 8
    adapt_sigma: bool = True
    signal_mode: str = "live"      # "live", "inverted", or "shuffled"
    seed: int = 0                  # only used by the "shuffled" control
    shrink_by_confidence: bool = True   # scale the signal by its own t-statistic


class RobustDirectionalTrader:
    """Trades an order-flow signal with the robust control of this module.

    At each market order it updates the signal, refreshes its volatility estimate,
    recomputes the target position and the no-trade band from the *currently
    quoted* half-spread, and moves only if the gap to target clears the band.
    """

    name = "robust-directional"

    def __init__(self, cfg: TraderConfig | None = None,
                 alpha_cfg: AlphaEstimatorConfig | None = None):
        self.cfg = cfg or TraderConfig()
        self.signal = OrderFlowAlpha(alpha_cfg)
        self.sigma = self.cfg.sigma0
        self._last_mid: float | None = None
        self.params = DirectionalParams(
            sigma=self.cfg.sigma0, beta=1.0, k=self.cfg.k,
            phi_alpha=self.cfg.phi_alpha, q_max=self.cfg.q_max,
        )
        self.alpha_log: list[float] = []
        self.target_log: list[float] = []
        self.band_log: list[float] = []
        self.n_trades = 0
        if self.cfg.signal_mode not in ("live", "inverted", "shuffled"):
            raise ValueError(f"unknown signal_mode {self.cfg.signal_mode!r}")
        self._rng = np.random.default_rng(self.cfg.seed)
        self._shuffle_pool: deque[float] = deque(maxlen=4000)

    # ------------------------------------------------------------- estimation

    def _update_sigma(self, mid: float, dt: float) -> None:
        if self._last_mid is not None and dt > 0:
            var = (mid - self._last_mid) ** 2 / dt
            a = 1.0 - 0.5 ** (dt / max(self.cfg.sigma_halflife, 1e-9))
            self.sigma = float(np.sqrt((1 - a) * self.sigma**2 + a * var))
        self._last_mid = mid

    # ---------------------------------------------------------------- policy

    def _apply_signal_mode(self, alpha: float) -> float:
        """Controls that destroy the signal while leaving everything else intact.

        ``inverted`` trades the negative of the signal: if the live arm earns a
        real edge, this one must give it back.  ``shuffled`` keeps the signal's
        distribution but breaks its timing, by serving a value drawn from the
        history rather than the current one -- so any P&L that survives is coming
        from exposure or from the cost structure, not from prediction.
        """
        mode = self.cfg.signal_mode
        if mode == "live":
            return alpha
        if mode == "inverted":
            return -alpha
        self._shuffle_pool.append(alpha)
        if len(self._shuffle_pool) < 50:
            return 0.0
        return float(self._shuffle_pool[self._rng.integers(len(self._shuffle_pool))])

    def decide(self, obs) -> int:
        """Return the position to hold after this event."""
        self.signal.observe_order(obs.time, obs.order_side)
        self.signal.observe_mid(obs.time, obs.mid)
        self._update_sigma(obs.mid, obs.dt)

        if self.signal.n_updates < self.cfg.warmup_updates:
            return obs.inventory          # not enough evidence to size anything yet

        if self.cfg.adapt_sigma:
            self.params.sigma = max(self.sigma, 1e-6)
        self.params.beta = max(self.signal.effective_decay(), 1e-3)

        raw = self.signal.alpha_shrunk if self.cfg.shrink_by_confidence else self.signal.alpha
        alpha = self._apply_signal_mode(raw)
        target = float(self.params.unclipped_target(alpha))
        band = self.params.no_trade_band(obs.half_spread * self.cfg.cost_multiple)
        self.alpha_log.append(alpha)
        self.target_log.append(target)
        self.band_log.append(band)

        # A working order is already going to move the position, so decide against
        # where the book *will* be, not where it is. Without this the trader
        # re-issues the same order on every event while the first is in flight.
        q = obs.inventory if obs.pending_target is None else obs.pending_target
        if obs.pending_target is not None:
            return obs.pending_target          # one order in flight at a time
        gap = target - q
        if abs(gap) <= band:
            return q                       # inside the band: the trade does not pay
        # Move to the near edge of the band, not to the target: the last unit of
        # the move is the one whose marginal value no longer covers the spread.
        desired = target - np.sign(gap) * band
        new_q = int(np.clip(round(desired), -self.cfg.q_max, self.cfg.q_max))
        if new_q != q:
            self.n_trades += 1
        return new_q


class BuyAndHoldTrader:
    """Control arm: takes one unit at the start and holds it.

    Isolates whether any P&L is the signal or just exposure to the simulated
    price path.
    """

    name = "buy-and-hold"

    def __init__(self, q_max: int = 8):
        self.q = 1

    def decide(self, obs) -> int:
        return self.q


class FlatTrader:
    """Control arm: never trades. P&L must be exactly zero."""

    name = "flat"

    def decide(self, obs) -> int:
        return 0
