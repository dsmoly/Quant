"""Test 3: is there a window where the signal is still alive but a passive order has filled?

Two curves, measured on the same paths.

  signal decay   the information coefficient of the fitted alpha against the
                 realised midprice change over a horizon h -- how much of the
                 prediction is still there h seconds later.

  passive fill   the probability that a limit order posted at the maker's own
                 quoted depth has been filled within h seconds, including the
                 latency before it goes live.

A passive strategy on this signal only works where both are non-trivial at the
same h. If the fill curve reaches useful probability only after the signal
correlation has collapsed, then the signal cannot be harvested passively at all
and the only ways to use it are to cross the spread (test 1) or to skew a quote
you are already showing (test 2).

Nothing here trades. The market runs with a flat agent so the probe cannot
perturb the paths it is measuring; passive fills are evaluated against the same
win-probability function the maker faces, using the mu and kappa in force at each
arrival.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.backtest import nash_benchmark_depth
from src.directional import AlphaEstimatorConfig, FlatTrader, OrderFlowAlpha
from src.market import CompetitionParams, DealerMarket, TrueDynamics

HORIZONS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0)


class Recorder:
    """Rides along a run, recording the tape and the fitted signal. Never trades."""

    name = "recorder"

    def __init__(self, alpha_cfg: AlphaEstimatorConfig | None = None, warmup: int = 200):
        self.signal = OrderFlowAlpha(alpha_cfg)
        self.warmup = warmup
        self.t: list[float] = []
        self.mid: list[float] = []
        self.alpha: list[float] = []
        self.mu: list[float] = []
        self.kap_ask: list[float] = []
        self.kap_bid: list[float] = []
        self.side: list[int] = []          # +1 buy order, -1 sell order
        self._market = None

    def bind(self, market) -> None:
        self._market = market

    def decide(self, obs) -> int:
        self.signal.observe_order(obs.time, obs.order_side)
        self.signal.observe_mid(obs.time, obs.mid)
        m = self._market
        self.t.append(obs.time)
        self.mid.append(obs.mid)
        self.alpha.append(self.signal.alpha_shrunk
                          if self.signal.n_updates >= self.warmup else 0.0)
        self.mu.append(m.mu)
        self.kap_ask.append(float(m.kap[0]))
        self.kap_bid.append(float(m.kap[1]))
        self.side.append(1 if obs.order_side == "buy" else -1)
        return 0                                   # never trade


def information_coefficient(t, mid, alpha, horizons):
    """Correlation of the signal with the forward midprice change at each horizon."""
    t = np.asarray(t)
    mid = np.asarray(mid)
    a = np.asarray(alpha)
    live = np.abs(a) > 0
    out = {}
    for h in horizons:
        j = np.searchsorted(t, t + h, side="left")
        ok = live & (j < len(t))
        j = np.clip(j, 0, len(t) - 1)
        fwd = mid[j] - mid
        m = ok & np.isfinite(fwd)
        if m.sum() < 50 or a[m].std() == 0 or fwd[m].std() == 0:
            out[h] = float("nan")
            continue
        out[h] = float(np.corrcoef(a[m], fwd[m])[0, 1])
    return out


def passive_fill_curve(rec, comp, depth, horizons, latency, rng, n_posts=4000):
    """Empirical distribution of the time for a passive order to fill.

    A post at index i goes live at ``t_i + latency``. Every later market order on
    the opposite side is a chance to be filled, with exactly the probability the
    maker faces at that arrival.
    """
    t = np.asarray(rec.t)
    mu = np.asarray(rec.mu)
    kap = np.stack([np.asarray(rec.kap_ask), np.asarray(rec.kap_bid)])
    side = np.asarray(rec.side)
    n = len(t)
    if n < 100:
        return {h: float("nan") for h in horizons}, float("nan")

    max_h = max(horizons)
    starts = rng.integers(0, n - 1, size=min(n_posts, n - 1))
    fill_times = []
    for i in starts:
        live_at = t[i] + latency
        want = 1 if rng.random() < 0.5 else -1     # post on a random side
        filled_at = np.inf
        for j in range(i + 1, n):
            if t[j] < live_at:
                continue
            if t[j] - t[i] > max_h:
                break
            if side[j] != want:                     # only the opposite side fills us
                continue
            k = kap[0 if want == 1 else 1, j]
            if rng.random() < comp.win_probability(depth, mu[j], k):
                filled_at = t[j] - t[i]
                break
        fill_times.append(filled_at)
    fill_times = np.asarray(fill_times)
    curve = {h: float(np.mean(fill_times <= h)) for h in horizons}
    finite = fill_times[np.isfinite(fill_times)]
    median = float(np.median(finite)) if finite.size else float("nan")
    return curve, median


def static_fill_curve(rec, comp, depth, horizons, latency, rng, n_posts=3000):
    """The same probe for a *static* resting order rather than a pegged quote.

    The distinction matters. ``passive_fill_curve`` re-prices the order at the
    prevailing midprice on every arrival, which is what a market maker's quote
    does and what the skewing maker in test 2 actually is. A resting order sits at
    a fixed price, so its distance from the mid moves with the mid: it becomes
    *easier* to fill exactly when the price is coming toward it, and harder when
    the price is running away. That asymmetry is adverse selection, and it is the
    reason a resting order can fill often and still lose.

    Returns the fill curve and the mean signed midprice move between posting and
    filling, in the direction of the position taken. Negative means the fills
    arrive after the price has already moved against the order.
    """
    t = np.asarray(rec.t)
    mid = np.asarray(rec.mid)
    mu = np.asarray(rec.mu)
    kap = np.stack([np.asarray(rec.kap_ask), np.asarray(rec.kap_bid)])
    side = np.asarray(rec.side)
    n = len(t)
    if n < 100:
        return {h: float("nan") for h in horizons}, float("nan")

    max_h = max(horizons)
    starts = rng.integers(0, n - 1, size=min(n_posts, n - 1))
    fill_times, drifts = [], []
    for i in starts:
        live_at = t[i] + latency
        want = 1 if rng.random() < 0.5 else -1     # +1: resting buy, -1: resting sell
        price = mid[i] - want * depth              # fixed limit price
        filled_at = np.inf
        for j in range(i + 1, n):
            if t[j] < live_at:
                continue
            if t[j] - t[i] > max_h:
                break
            if side[j] != -want:                   # only the opposite side fills us
                continue
            # Distance from the *current* mid to our fixed price.
            eff = want * (mid[j] - price)
            if eff <= 0:                           # the mid has crossed our price
                filled_at = t[j] - t[i]
                drifts.append(want * (mid[j] - mid[i]))
                break
            k = kap[0 if want == -1 else 1, j]
            if rng.random() < comp.win_probability(eff, mu[j], k):
                filled_at = t[j] - t[i]
                drifts.append(want * (mid[j] - mid[i]))
                break
        fill_times.append(filled_at)
    fill_times = np.asarray(fill_times)
    curve = {h: float(np.mean(fill_times <= h)) for h in horizons}
    return curve, (float(np.mean(drifts)) if drifts else float("nan"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=int, default=12)
    ap.add_argument("--horizon", type=float, default=1800.0)
    ap.add_argument("--eps", type=float, default=0.015)
    ap.add_argument("--latency", type=float, default=0.05)
    ap.add_argument("--q-max", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20240614)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    nash = nash_benchmark_depth(q_max=args.q_max)
    comp = CompetitionParams(nash_depth=nash)
    rng = np.random.default_rng(args.seed)
    seeds = list(rng.integers(0, 2**31 - 1, size=args.paths))
    beta_true = TrueDynamics().beta_alpha

    ics, curves_at, curves_wide, medians = [], [], [], []
    curves_static, static_drifts = [], []
    for seed in seeds:
        rec = Recorder()
        market = DealerMarket(TrueDynamics(eps=args.eps), comp, q_max=args.q_max,
                              latency=args.latency, rng=np.random.default_rng(seed))
        rec.bind(market)
        market.run_taker(rec, args.horizon)
        ics.append(information_coefficient(rec.t, rec.mid, rec.alpha, HORIZONS))
        c1, m1 = passive_fill_curve(rec, comp, nash, HORIZONS, args.latency,
                                    np.random.default_rng(seed + 1))
        c2, _ = passive_fill_curve(rec, comp, nash * 2.0, HORIZONS, args.latency,
                                   np.random.default_rng(seed + 2))
        c3, d3 = static_fill_curve(rec, comp, nash, HORIZONS, args.latency,
                                   np.random.default_rng(seed + 3))
        curves_at.append(c1)
        curves_wide.append(c2)
        curves_static.append(c3)
        static_drifts.append(d3)
        medians.append(m1)

    def avg(dicts, h):
        v = [d[h] for d in dicts if np.isfinite(d[h])]
        return float(np.mean(v)) if v else float("nan")

    print(f"eps={args.eps:g}  latency={args.latency*1000:.0f}ms  "
          f"{args.paths} paths x {args.horizon:g}s   half-spread {nash:.4f}   "
          f"true alpha half-life {np.log(2)/beta_true:.2f}s")
    hdr = (f"{'horizon':>9}{'signal IC':>11}{'IC retained':>13}"
           f"{'fill pegged':>13}{'fill resting':>14}{'joint':>9}")
    print(hdr)
    print("-" * len(hdr))

    ic0 = max((avg(ics, h) for h in HORIZONS if np.isfinite(avg(ics, h))), default=np.nan)
    rows = []
    for h in HORIZONS:
        ic = avg(ics, h)
        f1 = avg(curves_at, h)
        f2 = avg(curves_static, h)
        joint = ic * f1 if np.isfinite(ic) and np.isfinite(f1) else float("nan")
        rows.append(dict(horizon=h, ic=ic, ic_retained=ic / ic0 if ic0 else np.nan,
                         fill_pegged=f1, fill_resting=f2, joint=joint))
        print(f"{h:>9.2f}{ic:>11.3f}{100*ic/ic0:>12.0f}%{100*f1:>12.0f}%"
              f"{100*f2:>13.0f}%{joint:>9.3f}")

    med = float(np.nanmean(medians))
    peak = max(rows, key=lambda r: (r["joint"] if np.isfinite(r["joint"]) else -1))
    print(f"\n  peak IC {ic0:.3f}; median time to a passive fill at the quote: {med:.2f}s")
    print(f"  signal retained at the median fill time: "
          f"{100*np.exp(-beta_true*med):.0f}% of the impulse (true decay)")
    print(f"  best joint (IC x fill prob) at horizon {peak['horizon']:g}s: {peak['joint']:.3f}")

    # Is the window empty? Require both legs to be meaningful at the same horizon.
    IC_FLOOR, FILL_FLOOR = 0.5, 0.5     # half the peak signal, half the orders filled
    window = [r["horizon"] for r in rows
              if np.isfinite(r["ic_retained"]) and r["ic_retained"] >= IC_FLOOR
              and np.isfinite(r["fill_pegged"]) and r["fill_pegged"] >= FILL_FLOOR]
    print(f"\n  horizons where >={100*IC_FLOOR:.0f}% of peak IC survives AND "
          f">={100*FILL_FLOOR:.0f}% of passive orders have filled: "
          + (", ".join(f"{h:g}s" for h in window) if window else "NONE — the window is empty"))

    sd = float(np.nanmean(static_drifts))
    print(f"\n  resting order: mean midprice move between posting and filling, in the "
          f"direction of the position taken: {sd:+.5f}")
    print(f"  (half-spread {nash:.4f}; negative means the fill arrives only after the "
          f"price has already moved against the order -- classic adverse selection)")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"eps": args.eps, "latency": args.latency, "paths": args.paths,
                       "horizon": args.horizon, "nash_depth": nash,
                       "median_fill_seconds": med, "peak_ic": ic0,
                       "static_drift": sd, "rows": rows, "window": window}, fh)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
