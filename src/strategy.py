"""RAMM -- Robust Adaptive Market Maker with mean-field regime detection.

The trading idea, in one paragraph
----------------------------------
The robust market-making paper gives a market maker three ambiguity budgets that
move her quotes in *opposite* directions: aversion to drift misspecification
(phi_alpha) widens the total spread and pulls inventory to flat faster, while
aversion to fill-probability misspecification (phi_kappa) tightens quotes to
churn more.  It treats those budgets as fixed preferences.  The mean-field paper
supplies the missing state variable that says which one you should be running:
in a dealer market, fills depend on your quote *relative to the population mean
quote* mu_t, and a population of homogeneous learning market makers drifts to
supra-competitive quotes about 20% above the mean-field Nash level, with
heavy-tailed inventories as the tell.

So: estimate where the population is quoting, compare it to the Nash benchmark
computed from the mean-field game, and set the ambiguity vector from that gap.

  * Market quoting above Nash (supra-competitive, fat spreads to be won):
    raise phi.  Quote inside the crowd, harvest volume, be the heterogeneous
    agent that the mean-field paper shows pulls the market back to equilibrium.
  * Market at or below Nash (competitive, thin margins, every fill is a
    potential adverse selection): raise phi_alpha.  Widen, skew hard on
    inventory, and get flat.

Realised volatility and the *tail weight of our own inventory distribution* also
feed phi_alpha, because heavy inventory tails are precisely the state the
mean-field paper associates with a market maker being pushed around.

Everything the strategy consumes is observable to a real desk: its own quotes
and fills, RFQ arrival times, midprice moves, and cover prices on lost trades.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .robust_quotes import RobustParams, RobustQuoter


@dataclass
class EstimatorConfig:
    """Half-lives (in observations or seconds) for the online reference model."""

    kappa_lr: float = 0.02          # Robbins-Monro step for the fill-decay estimate
    kappa_bounds: tuple[float, float] = (5.0, 120.0)
    lam_halflife: float = 30.0      # seconds, for the RFQ arrival rate
    sigma_halflife: float = 30.0    # seconds, for realised volatility
    mu_halflife: float = 40.0       # observations, for the cover / population quote
    tail_halflife: float = 400.0    # observations, for the inventory tail weight


class OnlineReferenceModel:
    """Rolling estimates of the reference-model parameters and the mean field.

    The market maker recalibrates (lambda, kappa, sigma) slowly -- these are the
    long-run means the robust paper assumes she can estimate -- and tracks the
    population quote mu and her own inventory tail weight, which are the two
    mean-field state variables the ambiguity policy reads.
    """

    def __init__(self, cfg: EstimatorConfig, kappa0: float, lam0: float, sigma0: float,
                 mu0: float, q_max: int):
        self.cfg = cfg
        self.kappa = kappa0
        self.lam = lam0
        self.sigma = sigma0
        self.mu = mu0
        self.q_max = q_max
        self.tail = 0.0
        self._last_mid: float | None = None
        self.n_rfq = 0
        self.n_fills = 0

    @staticmethod
    def _alpha(halflife: float, step: float = 1.0) -> float:
        return 1.0 - 0.5 ** (step / max(halflife, 1e-9))

    def observe_mid(self, mid: float, dt: float) -> None:
        """Update realised volatility from the midprice increment."""
        if self._last_mid is not None and dt > 0:
            var = (mid - self._last_mid) ** 2 / dt
            a = self._alpha(self.cfg.sigma_halflife, dt)
            self.sigma = float(np.sqrt((1 - a) * self.sigma**2 + a * var))
        self._last_mid = mid

    def observe_rfq(self, depth: float, filled: bool, dt_since_last: float) -> None:
        """Update the fill-decay and arrival-rate estimates from one RFQ outcome.

        The reference model says an order posted at depth ``d`` fills with
        probability exp(-kappa d).  The log-likelihood of a Bernoulli outcome
        gives the score

            d/dkappa = -d              if filled
                     = d e^(-kd) / (1 - e^(-kd))   if not,

        which we follow with a small fixed step -- a standard recursive MLE, and
        the only thing the desk can actually do online.
        """
        self.n_rfq += 1
        if filled:
            self.n_fills += 1
        if np.isfinite(depth) and depth > 1e-9:
            z = float(np.exp(-self.kappa * depth))
            if filled:
                score = -depth
            else:
                score = depth * z / max(1.0 - z, 1e-6)
            # Scale by kappa so the step size is relative, not absolute.
            self.kappa = float(np.clip(self.kappa + self.cfg.kappa_lr * self.kappa * score,
                                       *self.cfg.kappa_bounds))
        if dt_since_last > 0:
            a = self._alpha(self.cfg.lam_halflife, dt_since_last)
            self.lam = float((1 - a) * self.lam + a * (1.0 / max(dt_since_last, 1e-6)) / 2.0)

    def observe_cover(self, cover: float) -> None:
        """Update the population mean quote from an observed cover price.

        Note the real-world caveat: cover is only seen on trades we lose, so on a
        live desk this estimator is selection-biased towards tighter competitor
        quotes.  A desk with top-of-book data should feed that instead.
        """
        if np.isfinite(cover) and cover > 0:
            a = self._alpha(self.cfg.mu_halflife)
            self.mu = float((1 - a) * self.mu + a * cover)

    def observe_inventory(self, q: int) -> None:
        """Track how often inventory sits in the tail of its own distribution."""
        a = self._alpha(self.cfg.tail_halflife)
        in_tail = 1.0 if abs(q) > 0.6 * self.q_max else 0.0
        self.tail = float((1 - a) * self.tail + a * in_tail)


@dataclass
class AmbiguityPolicy:
    """Map the mean-field state to the ambiguity vector (phi_alpha, phi)."""

    nash_depth: float = 0.0197      # mean-field Nash half-spread, price units
    gap_deadband: float = 0.03      # ignore gaps this small; they are noise
    gap_reference: float = 0.22     # the paper's ~20% supra-competitive level
    phi_max: float = 12.0           # aggressive lever cap
    phi_alpha_base: float = 1.5     # defensive lever at reference volatility
    phi_alpha_max: float = 25.0
    sigma_reference: float = 0.01
    sigma_sensitivity: float = 4.0
    tail_sensitivity: float = 20.0

    def gap(self, mu: float) -> float:
        """Relative distance of the population quote from the competitive benchmark.

        This is the live analogue of the mean-field paper's distance-to-equilibrium
        metric d(delta, delta*): positive means the market is quoting wide of Nash.
        """
        return (mu - self.nash_depth) / max(self.nash_depth, 1e-9)

    def __call__(self, mu: float, sigma: float, tail: float) -> tuple[float, float]:
        g = self.gap(mu)
        span = max(self.gap_reference - self.gap_deadband, 1e-9)
        aggression = float(np.clip((g - self.gap_deadband) / span, 0.0, 1.0))
        phi = self.phi_max * aggression

        vol_ratio = sigma / max(self.sigma_reference, 1e-12)
        phi_alpha = self.phi_alpha_base * (1.0 + self.sigma_sensitivity * max(vol_ratio - 1.0, 0.0))
        phi_alpha += self.tail_sensitivity * tail
        # A market at or inside Nash leaves no margin for error: lean defensive.
        phi_alpha *= 1.0 + max(-g, 0.0) * 3.0
        return float(np.clip(phi_alpha, 0.0, self.phi_alpha_max)), float(phi)


class MarketMaker:
    """Baseline: robust quoting with a *fixed* ambiguity vector.

    ``phi_alpha = phi = 0`` recovers the ambiguity-neutral reference-model market
    maker, which is the control arm in the backtest.
    """

    name = "fixed"

    def __init__(
        self,
        kappa0: float = 27.0,
        lam0: float = 2.0,
        sigma0: float = 0.01,
        theta: float = 0.001,
        q_max: int = 8,
        phi_alpha: float = 0.0,
        phi: float = 0.0,
        mu0: float = 0.0197,
        resolve_every: int = 100,
        estimators: EstimatorConfig | None = None,
        adapt_reference: bool = True,
    ):
        self.params = RobustParams(kappa=kappa0, lam_buy=lam0, lam_sell=lam0, sigma=sigma0,
                                   theta=theta, q_max=q_max, phi_alpha=phi_alpha, phi=phi)
        self.quoter = RobustQuoter(self.params)
        self.est = OnlineReferenceModel(estimators or EstimatorConfig(), kappa0, lam0, sigma0,
                                        mu0, q_max)
        self.resolve_every = resolve_every
        self.adapt_reference = adapt_reference
        self._since_resolve = 0

    # -------------------------------------------------------------- interface

    def quote(self, q: int) -> tuple[float, float]:
        """Ask and bid depths for the current inventory."""
        return self.quoter.depths(q)

    def _ambiguity(self) -> tuple[float, float]:
        return self.params.phi_alpha, self.params.phi

    def observe(self, ev) -> None:
        """Consume one market order and refresh the policy on schedule."""
        self.est.observe_mid(ev.mid, ev.dt)
        self.est.observe_inventory(ev.inventory)
        self.est.observe_rfq(ev.quoted_depth, ev.filled, ev.dt)
        if not ev.filled:
            self.est.observe_cover(ev.cover)
        self._since_resolve += 1
        if self._since_resolve >= self.resolve_every:
            self._since_resolve = 0
            self._resolve()

    def _resolve(self) -> None:
        phi_alpha, phi = self._ambiguity()
        updates = {"phi_alpha": phi_alpha, "phi": phi}
        if self.adapt_reference:
            updates.update(kappa=self.est.kappa, lam_buy=self.est.lam, lam_sell=self.est.lam,
                           sigma=self.est.sigma)
        self.quoter.update(**updates)


class AdaptiveRAMM(MarketMaker):
    """The strategy: ambiguity vector driven by the estimated mean-field state."""

    name = "ramm"

    def __init__(self, policy: AmbiguityPolicy | None = None, **kwargs):
        kwargs.setdefault("phi_alpha", 1.5)
        super().__init__(**kwargs)
        self.policy = policy or AmbiguityPolicy()
        self.phi_alpha_log: list[float] = []
        self.phi_log: list[float] = []

    def _ambiguity(self) -> tuple[float, float]:
        phi_alpha, phi = self.policy(self.est.mu, self.est.sigma, self.est.tail)
        self.phi_alpha_log.append(phi_alpha)
        self.phi_log.append(phi)
        return phi_alpha, phi


def neutral_market_maker(**kwargs) -> MarketMaker:
    """Ambiguity-neutral control arm: the reference-model optimal market maker."""
    kwargs.update(phi_alpha=0.0, phi=0.0)
    mm = MarketMaker(**kwargs)
    mm.name = "neutral"
    return mm


def fixed_robust_market_maker(phi_alpha: float = 2.0, phi: float = 8.0, **kwargs) -> MarketMaker:
    """Robust but non-adaptive arm: the robust paper's strategy as published."""
    mm = MarketMaker(phi_alpha=phi_alpha, phi=phi, **kwargs)
    mm.name = f"robust(pa={phi_alpha:g},p={phi:g})"
    return mm
