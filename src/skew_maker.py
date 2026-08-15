"""Test 2: a market maker that skews its quotes on the order-flow signal.

The directional taker has to clear a high bar: its edge only pays when
``|alpha| > c (beta + rho)``, i.e. when the drift beats the spread it must cross.
A maker holding the same signal faces a different bar. It is already quoting on
both sides; leaning those quotes costs nothing at the margin. It does not need
alpha to beat the spread, only to beat zero.

The mechanism is a shift of the reservation price rather than a trade. With the
signal ``alpha`` and a skew coefficient ``gamma``,

    ask_depth = base_ask + gamma * alpha
    bid_depth = base_bid - gamma * alpha

so a positive alpha (price expected to rise) leans the bid in to buy more and
widens the offer to avoid selling into the rise. Depths are floored at zero, so
the maker never crosses the midprice and never takes liquidity: it is still
earning the spread, just asymmetrically.

Choosing gamma. The skew should be worth the price move the maker expects to
capture over the time it will hold the resulting inventory. A unit acquired now
is turned over at roughly the fill rate ``lambda_fill``, and the signal itself
decays at ``beta``, so the capturable fraction of alpha is

    gamma = 1 / (beta + lambda_fill)

which is the same discount that appears in the taker's target position, for the
same reason: a signal that dies before you can act on it is worth less. Both are
estimated online from the maker's own tape.

What this predicts. Skewing should show up in the P&L attribution as a fall in
the *adverse selection* term rather than a rise in gross spread capture -- the
maker is not quoting wider, it is choosing which side to be filled on. The
capture ratio is therefore the metric that should move.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .directional import AlphaEstimatorConfig, OrderFlowAlpha
from .strategy import AdaptiveRAMM, AmbiguityPolicy


@dataclass
class SkewConfig:
    """Controls for the alpha skew laid on top of the robust quotes."""

    skew_scale: float = 1.0        # multiplier on the derived gamma; 0 disables the skew
    max_skew: float = 0.05         # cap on the shift, in price units
    fill_rate_halflife: float = 60.0     # seconds, for the fill-rate estimate
    shrink_by_confidence: bool = True
    warmup_updates: int = 200
    invert_signal: bool = False    # control arm: skew the wrong way


class AlphaSkewMarketMaker(AdaptiveRAMM):
    """The robust market maker, with its quotes leaned on the order-flow signal."""

    name = "maker / alpha-skew"

    def __init__(self, skew: SkewConfig | None = None,
                 alpha_cfg: AlphaEstimatorConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        self.skew_cfg = skew or SkewConfig()
        self.signal = OrderFlowAlpha(alpha_cfg)
        self.fill_rate = 0.5
        self._last_event_t: float | None = None
        self.skew_log: list[float] = []
        self.alpha_log: list[float] = []

    # ----------------------------------------------------------- estimation

    def _update_fill_rate(self, filled: bool, dt: float) -> None:
        if dt <= 0:
            return
        a = 1.0 - 0.5 ** (dt / max(self.skew_cfg.fill_rate_halflife, 1e-9))
        self.fill_rate = float((1 - a) * self.fill_rate + a * (1.0 / dt if filled else 0.0))

    def observe(self, ev) -> None:
        # The maker sees the same tape the taker does: an "ask" event is our offer
        # being lifted, which is a buy market order.
        self.signal.observe_order(ev.time, "buy" if ev.side == "ask" else "sell")
        self.signal.observe_mid(ev.time, ev.mid)
        self._update_fill_rate(ev.filled, ev.dt)
        super().observe(ev)

    # --------------------------------------------------------------- policy

    @property
    def alpha(self) -> float:
        if self.signal.n_updates < self.skew_cfg.warmup_updates:
            return 0.0
        a = (self.signal.alpha_shrunk if self.skew_cfg.shrink_by_confidence
             else self.signal.alpha)
        return -a if self.skew_cfg.invert_signal else a

    @property
    def gamma(self) -> float:
        """Fraction of the signal the maker can expect to capture before it decays."""
        beta = max(self.signal.effective_decay(), 1e-3)
        return self.skew_cfg.skew_scale / (beta + max(self.fill_rate, 0.0))

    def quote(self, q: int) -> tuple[float, float]:
        ask, bid = super().quote(q)
        shift = float(np.clip(self.gamma * self.alpha,
                              -self.skew_cfg.max_skew, self.skew_cfg.max_skew))
        self.skew_log.append(shift)
        self.alpha_log.append(self.alpha)
        # Positive alpha: lean the bid in, push the offer out. Depths are floored
        # at zero so the quotes never cross the midprice -- this stays a maker.
        ask = ask + shift if np.isfinite(ask) else ask
        bid = bid - shift if np.isfinite(bid) else bid
        if np.isfinite(ask):
            ask = max(ask, 0.0)
        if np.isfinite(bid):
            bid = max(bid, 0.0)
        return ask, bid


def plain_maker(**kwargs) -> AdaptiveRAMM:
    """The current best maker arm, for comparison."""
    mm = AdaptiveRAMM(policy=kwargs.pop("policy", None) or AmbiguityPolicy(), **kwargs)
    mm.name = "maker / RAMM"
    return mm


def skew_maker(scale: float = 1.0, invert: bool = False, **kwargs) -> AlphaSkewMarketMaker:
    mm = AlphaSkewMarketMaker(skew=SkewConfig(skew_scale=scale, invert_signal=invert),
                              **kwargs)
    mm.name = f"maker / skew x{scale:g}" + (" (inverted)" if invert else "")
    return mm
