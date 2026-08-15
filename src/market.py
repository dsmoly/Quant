"""Simulated dealer market that is misspecified in both papers' senses at once.

Two independent sources of trouble for a market maker are combined here:

1.  *Model misspecification* (Cartea-Donnelly-Jaimungal, section 5).  The true
    dynamics are not the market maker's reference model.  The midprice carries a
    transient short-term alpha driven by market-order flow, market orders arrive
    as a mutually exciting bivariate Hawkes process, and the fill-rate decay
    kappa is itself a jumping mean-reverting process:

        dS      = alpha_t dt + sigma dW
        dalpha  = -beta_a alpha dt + eps (dM+ - dM-)
        dlam+/- = beta_l (theta_l - lam+/-) dt + eta_l dM+/- + nu_l dM-/+
        dkap+/- = beta_k (theta_k - kap+/-) dt + eta_k dM+/- + nu_k dM-/+

    The MM calibrates a *constant* (lambda, kappa, sigma, alpha=0) reference
    model to the long-run means of these processes -- with the paper's
    parameters, lambda = 2 and kappa = 27 -- and is wrong at every instant.

2.  *Competition* (Assayag-Barzykin-Cont-Xiong).  Winning a trade is not just a
    matter of the client's price tolerance; the MM has to beat the rest of the
    dealer population.  The fill probability is that paper's intensity function
    (3.1) evaluated at the MM's quote and the population mean quote mu_t:

        P(fill | MO, depth d) = 1 / (C e^(k d) + C_m e^(k_m k d - k_c k mu))

    where d is the MM's depth and mu the population mean depth, both scaled by
    the current kappa_t to move between the papers' units (paper A works in
    units where the monopolistic fill rate is e^-delta, paper B in price units
    where it is e^(-kappa delta), so delta_A = kappa * delta_price).

The population quote mu_t is not constant.  Section 4 of the mean-field paper
finds that a homogeneous population of *learning* dealers drifts to
supra-competitive quotes roughly 20% above the Nash level, coexisting with
heavy-tailed inventories.  We model that as a two-state Markov chain on the
target of an Ornstein-Uhlenbeck process: a "competitive" state anchored at the
mean-field Nash level and a "supra-competitive" state anchored above it.  This is
the regime the adaptive strategy in ``strategy.py`` is built to detect.

Adverse selection falls out of the construction rather than being imposed: an
ask fill happens exactly when a buy market order arrives, and that same order
pushes alpha up, so the MM is systematically short into a rising price.

Simulation is exact and event-driven.  All latent processes are advanced in
closed form between market orders, and arrival times come from Ogata thinning
against the decaying Hawkes intensity, so there is no time-discretisation error
and no wasted work on empty time slices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrueDynamics:
    """Parameters of the true (unobserved) market, Table 1 of the robust paper."""

    sigma: float = 0.01
    eps: float = 0.001          # market-order impact on the short-term alpha
    beta_alpha: float = 1.0     # alpha decay
    theta_lam: float = 0.2      # Hawkes baseline
    beta_lam: float = 70.0 / 9.0
    eta_lam: float = 5.0        # self-excitation
    nu_lam: float = 2.0         # cross-excitation
    theta_kap: float = 15.0
    beta_kap: float = 7.0 / 6.0
    eta_kap: float = 5.0
    nu_kap: float = 2.0
    S0: float = 100.0

    def stationary_lam(self) -> float:
        """Long-run mean arrival rate per side; 2.0 at the paper's parameters."""
        denom = self.beta_lam - self.eta_lam - self.nu_lam
        if denom <= 0:
            raise ValueError("Hawkes system is explosive: eta + nu must be below beta")
        return self.beta_lam * self.theta_lam / denom

    def stationary_kap(self) -> float:
        """Long-run mean fill decay; 27.0 at the paper's parameters."""
        return self.theta_kap + (self.eta_kap + self.nu_kap) * self.stationary_lam() / self.beta_kap


@dataclass
class CompetitionParams:
    """Mean-field competition, eq. (3.1) of the dealer-market paper, plus the
    regime process on the population mean quote."""

    C: float = 1.0
    C_m: float = 1.0
    k_m: float = 3.0
    k_c: float = 2.0
    nash_depth: float = 0.0197        # competitive population depth, price units
    supra_multiple: float = 1.25      # supra-competitive target, multiple of Nash
    mean_supra_seconds: float = 600.0   # mean dwell time in the supra-competitive state
    mean_comp_seconds: float = 600.0    # mean dwell time in the competitive state
    mu_reversion: float = 0.05        # OU speed of mu towards its regime target
    mu_vol: float = 0.0006            # OU volatility of mu
    cover_noise: float = 0.15         # relative noise on observed cover prices

    def win_probability(self, depth, mu: float, kappa: float):
        """Probability of winning an RFQ quoting ``depth`` against population ``mu``.

        Both depths are converted to the mean-field paper's dimensionless units by
        multiplying by kappa.  Bounded above by the monopolistic rate
        exp(-kappa * depth) / C, as Assumption 2.2(3) requires.
        """
        d = kappa * np.asarray(depth, dtype=float)
        m = kappa * mu
        return 1.0 / (self.C * np.exp(d) + self.C_m * np.exp(self.k_m * d - self.k_c * m))


@dataclass
class Fill:
    side: str          # "ask" (we sold) or "bid" (we bought)
    depth: float
    price: float
    time: float


@dataclass
class MarketEvent:
    """One market order, from the market maker's point of view."""

    time: float
    dt: float                        # time since the previous market order
    mid: float
    side: str                        # "ask" (buy MO lifts us) or "bid" (sell MO hits us)
    quoted_depth: float              # what we showed on that side; inf if not quoting
    filled: bool
    cover: float                     # observed competing depth when we lose; NaN otherwise
    inventory: int
    wealth: float


class DealerMarket:
    """Exact event-driven simulator for the combined model."""

    def __init__(
        self,
        dynamics: TrueDynamics | None = None,
        competition: CompetitionParams | None = None,
        q_max: int = 8,
        theta: float = 0.001,
        latency: float = 0.0,
        taker_fee: float = 0.0,
        taker_slippage: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        self.dyn = dynamics or TrueDynamics()
        self.comp = competition or CompetitionParams()
        self.q_max = q_max
        self.theta = theta            # liquidation penalty l(q) = theta * q
        # Round-trip delay between a decision and its effect. For a taker it is
        # order submission latency: the order executes at the price prevailing
        # ``latency`` seconds later, not the price that triggered it. For a maker
        # it is quote-update latency: a new quote only becomes live after the
        # delay, so the market can trade against a stale one in between.
        self.latency = float(latency)
        self.taker_fee = float(taker_fee)          # per unit, on top of the spread
        self.taker_slippage = float(taker_slippage)  # per unit, price concession
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self) -> None:
        d = self.dyn
        self.t = 0.0
        self.S = d.S0
        self.alpha = 0.0
        self.lam = np.array([d.stationary_lam(), d.stationary_lam()])  # [buy MOs, sell MOs]
        self.kap = np.array([d.stationary_kap(), d.stationary_kap()])
        self.q = 0
        self.cash = 0.0
        self.supra = False
        self.mu = self.comp.nash_depth
        self.fills: list[Fill] = []
        self.supra_time = 0.0
        self._pending_target: tuple[float, int] | None = None   # (execute_at, target)
        self._live_quote: tuple[float, float] = (np.inf, np.inf)
        self._quote_queue: list[tuple[float, float, float]] = []  # (live_at, ask, bid)
        self.taker_costs = 0.0        # fees and slippage paid, tracked separately
        self.stale_fills = 0          # fills against a quote that was already superseded

    @property
    def wealth(self) -> float:
        """Mark-to-market wealth: cash plus inventory at the midprice."""
        return self.cash + self.q * self.S

    @property
    def mu_target(self) -> float:
        c = self.comp
        return c.nash_depth * (c.supra_multiple if self.supra else 1.0)

    # ------------------------------------------------------- exact propagation

    def _advance(self, dt: float) -> None:
        """Advance every latent process in closed form over an interval with no jumps."""
        d = self.dyn
        # Midprice: the integral of an exponentially decaying drift, plus Brownian noise.
        decay_a = np.exp(-d.beta_alpha * dt)
        drift = self.alpha * (1.0 - decay_a) / d.beta_alpha if d.beta_alpha > 0 else self.alpha * dt
        self.S += drift + d.sigma * np.sqrt(dt) * self.rng.standard_normal()
        self.alpha *= decay_a

        self.lam = d.theta_lam + (self.lam - d.theta_lam) * np.exp(-d.beta_lam * dt)
        self.kap = d.theta_kap + (self.kap - d.theta_kap) * np.exp(-d.beta_kap * dt)
        self.kap = np.clip(self.kap, 1.0, None)

        # Competitor regime: exact two-state Markov switch over the interval.
        c = self.comp
        if self.supra:
            self.supra_time += dt
        mean_dwell = c.mean_supra_seconds if self.supra else c.mean_comp_seconds
        if self.rng.random() < 1.0 - np.exp(-dt / max(mean_dwell, 1e-9)):
            self.supra = not self.supra

        # Population mean quote: exact Ornstein-Uhlenbeck update towards the regime target.
        theta = self.mu_target
        decay_m = np.exp(-c.mu_reversion * dt)
        sd = c.mu_vol * np.sqrt((1.0 - decay_m**2) / (2.0 * c.mu_reversion)) if c.mu_reversion > 0 \
            else c.mu_vol * np.sqrt(dt)
        self.mu = theta + (self.mu - theta) * decay_m + sd * self.rng.standard_normal()
        self.mu = max(self.mu, 1e-4)

        self.t += dt

    def _apply_market_order(self, side_idx: int, excite_arrivals: bool = True) -> None:
        """Jumps triggered by a market order: alpha impact, Hawkes and depth excitation.

        ``excite_arrivals`` is False for the directional trader's own orders. The
        Hawkes process is calibrated to the *client* order flow the dealer faces,
        with a branching ratio of 0.9 at the paper's parameters; the taker is one
        participant already inside that calibration, so adding its orders to the
        self-excitation double-counts them and pushes the effective branching
        ratio above 1, at which point the arrival process is explosive rather than
        stationary. Price impact and book thinning still apply to every order,
        including the taker's, so a strategy always pays for its own footprint.
        """
        d = self.dyn
        self.alpha += (1.0 if side_idx == 0 else -1.0) * d.eps   # buy MOs push the price up
        other = 1 - side_idx
        if excite_arrivals:
            self.lam[side_idx] += d.eta_lam
            self.lam[other] += d.nu_lam
        self.kap[side_idx] += d.eta_kap
        self.kap[other] += d.nu_kap

    @property
    def taker_half_spread(self) -> float:
        """All-in cost of crossing one unit: spread, slippage and fee."""
        return self.mu + self.taker_slippage + self.taker_fee

    def _publish_quote(self, ask: float, bid: float) -> None:
        """Queue a quote update; it goes live one latency later."""
        if self.latency <= 0:
            self._live_quote = (ask, bid)
        else:
            self._quote_queue.append((self.t + self.latency, ask, bid))

    def _refresh_quote(self) -> None:
        """Promote any queued quote whose latency has elapsed."""
        while self._quote_queue and self._quote_queue[0][0] <= self.t:
            _, ask, bid = self._quote_queue.pop(0)
            self._live_quote = (ask, bid)

    def _intensity_bound(self) -> float:
        """Upper bound on the total arrival rate over the coming interval.

        Between jumps each lam decays monotonically towards theta_lam, so the
        current value bounds it from above when above the baseline, and the
        baseline bounds it when below.
        """
        return float(max(self.lam.sum(), 2.0 * self.dyn.theta_lam))

    # -------------------------------------------------------------------- run

    def run(self, strategy, horizon: float) -> "RunResult":
        """Simulate ``horizon`` seconds, asking ``strategy`` for quotes at each order.

        The market maker's quotes are a function of inventory only, and inventory
        changes only at fills, so evaluating the quote function at market-order
        times is exact -- no discretisation is involved.
        """
        t_prev_event = 0.0
        inv_path: list[int] = []
        pnl_path: list[float] = []
        mu_path: list[float] = []
        time_path: list[float] = []
        mid_path: list[float] = []
        n_events = 0

        while self.t < horizon:
            bound = self._intensity_bound()
            dt = self.rng.exponential(1.0 / bound)
            if self.t + dt > horizon:
                self._advance(horizon - self.t)
                break
            self._advance(dt)

            total = float(self.lam.sum())
            u = self.rng.random() * bound
            if u >= total:
                continue                       # thinning rejection: no order arrived
            side_idx = 0 if u < self.lam[0] else 1
            side = "ask" if side_idx == 0 else "bid"

            # Quote-update latency: the market trades against whatever quote is
            # currently live, which may predate the maker's latest decision.
            self._refresh_quote()
            fresh = strategy.quote(self.q)
            if self.latency <= 0:
                live_ask, live_bid = fresh
                self._live_quote = fresh
            else:
                live_ask, live_bid = self._live_quote
                if live_ask != fresh[0] or live_bid != fresh[1]:
                    self._publish_quote(*fresh)
                    self.stale_fills += 1
            depth = live_ask if side == "ask" else live_bid
            blocked = (side == "ask" and self.q <= -self.q_max) or \
                      (side == "bid" and self.q >= self.q_max)
            quoting = bool(np.isfinite(depth)) and not blocked

            filled = False
            cover = np.nan
            kappa_side = float(self.kap[side_idx])
            if quoting and self.rng.random() < self.comp.win_probability(depth, self.mu, kappa_side):
                price = self.S + depth if side == "ask" else self.S - depth
                if side == "ask":
                    self.cash += price
                    self.q -= 1
                else:
                    self.cash -= price
                    self.q += 1
                filled = True
                self.fills.append(Fill(side, float(depth), price, self.t))
            else:
                cover = max(self.mu * (1.0 + self.comp.cover_noise * self.rng.standard_normal()), 1e-5)

            self._apply_market_order(side_idx)

            ev = MarketEvent(
                time=self.t, dt=self.t - t_prev_event, mid=self.S, side=side,
                quoted_depth=float(depth) if quoting else np.inf,
                filled=filled, cover=cover, inventory=self.q, wealth=self.wealth,
            )
            t_prev_event = self.t
            strategy.observe(ev)

            inv_path.append(self.q)
            pnl_path.append(self.wealth)
            mu_path.append(self.mu)
            time_path.append(self.t)
            mid_path.append(self.S)
            n_events += 1
            if n_events >= self.MAX_EVENTS:
                break

        # Terminal payoff X_T + q_T (S_T - l(q_T)) with l(q) = theta * q: the cost of
        # unwinding the book relative to marking it at the midprice is theta * q^2.
        liquidation = self.theta * self.q**2
        return RunResult(
            pnl=self.wealth - liquidation,
            terminal_inventory=self.q,
            n_fills=len(self.fills),
            n_rfq=n_events,
            inventory_path=np.array(inv_path),
            pnl_path=np.array(pnl_path),
            mu_path=np.array(mu_path),
            time_path=np.array(time_path),
            mid_path=np.array(mid_path),
            fills=list(self.fills),
            final_mid=self.S,
            theta=self.theta,
            supra_fraction=self.supra_time / max(horizon, 1e-9),
        )


    # --------------------------------------------------------- taker interface

    MAX_EVENTS = 400_000        # guard against near-critical Hawkes bursts

    def run_taker(self, trader, horizon: float) -> "RunResult":
        """Simulate a directional trader who *crosses* the dealer spread.

        The taker sees exactly what the market maker saw -- signed order arrivals
        and midprices -- and may adjust its position at every market order.  Its
        own trades are market orders like any other: they pay the prevailing
        dealer half-spread ``mu_t``, tilt the short-term alpha by ``eps``, and
        excite the arrival intensities.  Nothing here lets the strategy trade
        without paying for its own footprint.
        """
        t_prev_event = 0.0
        inv_path: list[int] = []
        pnl_path: list[float] = []
        mu_path: list[float] = []
        time_path: list[float] = []
        mid_path: list[float] = []
        n_events = 0

        while self.t < horizon:
            bound = self._intensity_bound()
            dt = self.rng.exponential(1.0 / bound)
            t_next = self.t + dt

            # A queued order executes at its own time, at the price prevailing
            # then -- before the next market order, if it comes due first.
            if self._pending_target is not None and self._pending_target[0] <= min(t_next, horizon):
                exec_at, target = self._pending_target
                self._advance(max(exec_at - self.t, 0.0))
                self._pending_target = None
                self._execute_taker(target)

            if t_next > horizon:
                self._advance(max(horizon - self.t, 0.0))
                break
            self._advance(max(t_next - self.t, 0.0))

            total = float(self.lam.sum())
            u = self.rng.random() * bound
            if u >= total:
                continue                       # thinning rejection
            side_idx = 0 if u < self.lam[0] else 1
            side = "buy" if side_idx == 0 else "sell"

            # The order hits the market first; the taker reacts to it.
            self._apply_market_order(side_idx)

            obs = TakerObservation(
                time=self.t, dt=self.t - t_prev_event, mid=self.S, order_side=side,
                half_spread=self.taker_half_spread, inventory=self.q, wealth=self.wealth,
                pending_target=None if self._pending_target is None else self._pending_target[1],
            )
            t_prev_event = self.t
            target = trader.decide(obs)
            if self.latency <= 0:
                self._execute_taker(target)
            elif target != self.q:
                # Submission latency: the decision is made now, the fill happens
                # later, at whatever the price has become by then.
                self._pending_target = (self.t + self.latency, target)

            inv_path.append(self.q)
            pnl_path.append(self.wealth)
            mu_path.append(self.mu)
            time_path.append(self.t)
            mid_path.append(self.S)
            n_events += 1
            if n_events >= self.MAX_EVENTS:
                break

        liquidation = self.theta * self.q**2
        return RunResult(
            pnl=self.wealth - liquidation,
            terminal_inventory=self.q,
            n_fills=len(self.fills),
            n_rfq=n_events,
            inventory_path=np.array(inv_path),
            pnl_path=np.array(pnl_path),
            mu_path=np.array(mu_path),
            time_path=np.array(time_path),
            mid_path=np.array(mid_path),
            fills=list(self.fills),
            final_mid=self.S,
            theta=self.theta,
            supra_fraction=self.supra_time / max(horizon, 1e-9),
        )

    def _execute_taker(self, target: int) -> None:
        """Move the taker's position toward ``target``, one unit at a time.

        Each unit crosses the spread at ``S +/- mu`` and then impacts the market
        exactly as any other market order of that side would.
        """
        target = int(np.clip(target, -self.q_max, self.q_max))
        c = self.taker_half_spread
        while self.q != target:
            buying = target > self.q
            price = self.S + c if buying else self.S - c
            if buying:
                self.cash -= price
                self.q += 1
            else:
                self.cash += price
                self.q -= 1
            self.taker_costs += self.taker_slippage + self.taker_fee
            # Depth is recorded as negative: the taker pays what it crosses.
            self.fills.append(Fill("bid" if buying else "ask", -c, price, self.t))
            self._apply_market_order(0 if buying else 1, excite_arrivals=False)


@dataclass
class TakerObservation:
    """What a directional trader sees at each market order."""

    time: float
    dt: float                        # time since the previous order
    mid: float
    order_side: str                  # "buy" if the order lifted the offer
    half_spread: float               # all-in cost of crossing: spread + slippage + fee
    inventory: int
    wealth: float
    pending_target: int | None = None   # position an in-flight order will reach


@dataclass
class RunResult:
    pnl: float
    terminal_inventory: int
    n_fills: int
    n_rfq: int
    inventory_path: np.ndarray
    pnl_path: np.ndarray
    mu_path: np.ndarray
    time_path: np.ndarray = field(default_factory=lambda: np.array([]))
    mid_path: np.ndarray = field(default_factory=lambda: np.array([]))
    fills: list = field(default_factory=list)
    final_mid: float = float("nan")
    theta: float = 0.0
    supra_fraction: float = 0.0
    phi_alpha_path: np.ndarray = field(default_factory=lambda: np.array([]))
    phi_path: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def hit_rate(self) -> float:
        return self.n_fills / max(self.n_rfq, 1)
