"""Test 1: cost sensitivity and selectivity for the directional arm.

Raising the crossing cost ``c`` in this control does two separate things: it
charges the strategy more per trade, and -- because the no-trade band is
``b = c / |h2|`` -- it also makes the strategy more selective. Sweeping "c" alone
therefore cannot answer whether selectivity rescues a marginal edge, because the
selectivity is confounded with the cost that motivated it. So this runs two
sweeps:

  cost        the fee/slippage actually charged rises, and the band responds to
              it as the control says it should. This is the real question "what
              happens as trading gets more expensive".

  selectivity the charged cost is held fixed at a realistic level and only the
              band is scaled, by ``cost_multiple``. This is the real question
              "does trading less, but better, help".

There is a theoretical prior for the first: if the policy re-optimises for each
cost, the envelope theorem gives dP/dc = -E[turnover] < 0, so total P&L must fall
monotonically. Any interior maximum in the cost sweep would mean the policy is
*not* optimising, which is itself worth knowing. The second sweep has no such
prior and is where an interior optimum can genuinely live.

Both run with submission latency on.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.backtest import nash_benchmark_depth
from src.directional import RobustDirectionalTrader, TraderConfig
from src.market import CompetitionParams, DealerMarket, TrueDynamics

BPS = 0.01          # one basis point on an asset priced at 100


def run_config(seeds, *, eps, latency, fee, selectivity, horizon, q_max, theta,
               nash_depth, phi_alpha):
    pnls, fills, alphas, targets = [], [], [], []
    for seed in seeds:
        trader = RobustDirectionalTrader(TraderConfig(
            phi_alpha=phi_alpha, q_max=q_max, cost_multiple=selectivity))
        market = DealerMarket(
            dynamics=TrueDynamics(eps=eps),
            competition=CompetitionParams(nash_depth=nash_depth),
            q_max=q_max, theta=theta, latency=latency,
            taker_fee=fee, rng=np.random.default_rng(seed))
        res = market.run_taker(trader, horizon)
        pnls.append(res.pnl)
        fills.append(res.n_fills)
        if trader.alpha_log:
            # Signed, not folded: folding destroys exactly the tail statistic the
            # Hawkes-clustering question is about.
            alphas.append(np.asarray(trader.alpha_log))
            targets.append(np.asarray(trader.target_log))
    pnls = np.array(pnls, dtype=float)
    fills = np.array(fills, dtype=float)
    sd = pnls.std(ddof=1)
    allalpha = np.concatenate(alphas) if alphas else np.array([0.0])
    return {
        "mean_pnl": float(pnls.mean()),
        "sem": float(sd / np.sqrt(len(pnls))),
        "sharpe": float(pnls.mean() / sd) if sd > 0 else float("nan"),
        "fills": float(fills.mean()),
        "pnl_per_fill": float(pnls.mean() / max(fills.mean(), 1e-9)),
        "win_rate": float((pnls > 0).mean()),
        "alpha_abs_mean": float(np.abs(allalpha).mean()),
        "alpha_kurtosis": float(((allalpha - allalpha.mean()) ** 4).mean()
                                / max(allalpha.var() ** 2, 1e-30) - 3.0),
        "alpha_sd": float(allalpha.std()),
    }


def sweep(name, seeds, levels, label_fmt, **base):
    print(f"\n=== {name} ===")
    hdr = (f"{'level':>12}{'meanPnL':>10}{'sem':>7}{'Sharpe':>8}{'fills':>8}"
           f"{'P&L/fill':>10}{'win%':>7}")
    print(hdr)
    print("-" * len(hdr))
    out = []
    for lvl, kw in levels:
        r = run_config(seeds, **{**base, **kw})
        r["level"] = lvl
        out.append(r)
        print(f"{label_fmt(lvl):>12}{r['mean_pnl']:>10.3f}{r['sem']:>7.3f}"
              f"{r['sharpe']:>8.2f}{r['fills']:>8.0f}{r['pnl_per_fill']:>10.4f}"
              f"{r['win_rate']*100:>7.0f}")
    return out


def verdict(rows, key, what):
    vals = [r[key] for r in rows]
    i = int(np.argmax(vals))
    interior = 0 < i < len(vals) - 1
    # Only call it an interior maximum if it clears its neighbours by more than
    # Monte Carlo noise; otherwise it is a flat top, not an optimum.
    noise = max(rows[i]["sem"], 1e-12)
    clears = interior and key == "mean_pnl" and (
        vals[i] - max(vals[i - 1], vals[i + 1])) > 2 * noise
    tag = ("INTERIOR MAXIMUM" if (interior and (clears or key != "mean_pnl"))
           else "monotone / edge optimum")
    print(f"  {what} argmax at level {rows[i]['level']} -> {tag}"
          + (f"  (beats neighbours by {vals[i]-max(vals[i-1],vals[i+1]):+.3f}, "
             f"noise ~{noise:.3f})" if interior and key == "mean_pnl" else ""))
    return dict(argmax=rows[i]["level"], interior=bool(interior and (clears or key != "mean_pnl")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=int, default=60)
    ap.add_argument("--horizon", type=float, default=1200.0)
    ap.add_argument("--eps", type=float, default=0.015)
    ap.add_argument("--latency", type=float, default=0.05)
    ap.add_argument("--fixed-fee-bps", type=float, default=1.0,
                    help="fee held fixed during the selectivity sweep")
    ap.add_argument("--phi-alpha", type=float, default=40.0)
    ap.add_argument("--q-max", type=int, default=8)
    ap.add_argument("--theta", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=20240614)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    nash = nash_benchmark_depth(q_max=args.q_max)
    seeds = list(np.random.default_rng(args.seed).integers(0, 2**31 - 1, size=args.paths))
    base = dict(eps=args.eps, latency=args.latency, horizon=args.horizon,
                q_max=args.q_max, theta=args.theta, nash_depth=nash,
                phi_alpha=args.phi_alpha)

    print(f"eps={args.eps:g}  latency={args.latency*1000:.0f}ms  {args.paths} paths x "
          f"{args.horizon:g}s   dealer half-spread ≈ {nash:.4f} "
          f"({nash/BPS:.2f} bps)")

    fee_levels = [(b, dict(fee=b * BPS, selectivity=1.0))
                  for b in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)]
    cost_rows = sweep("COST SWEEP — fee charged and band responds", seeds, fee_levels,
                      lambda b: f"{b:g} bps", **base)

    sel_levels = [(m, dict(fee=args.fixed_fee_bps * BPS, selectivity=m))
                  for m in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)]
    sel_rows = sweep(f"SELECTIVITY SWEEP — fee fixed at {args.fixed_fee_bps:g} bps, "
                     f"band scaled", seeds, sel_levels, lambda m: f"x{m:g}", **base)

    print("\n--- verdicts ---")
    v = {
        "cost_pnl": verdict(cost_rows, "mean_pnl", "cost sweep, mean P&L:"),
        "cost_sharpe": verdict(cost_rows, "sharpe", "cost sweep, Sharpe:    "),
        "sel_pnl": verdict(sel_rows, "mean_pnl", "selectivity, mean P&L: "),
        "sel_sharpe": verdict(sel_rows, "sharpe", "selectivity, Sharpe:   "),
    }

    k = cost_rows[0]["alpha_kurtosis"]
    print(f"\n  excess kurtosis of signed alpha at the decision points: {k:.2f}"
        f"  (Gaussian = 0)")
    print("  (Hawkes branching ratio 0.9 -> clustered flow -> fat-tailed alpha;"
          " the question is whether that is enough for selectivity to pay)")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"eps": args.eps, "latency": args.latency, "paths": args.paths,
                       "horizon": args.horizon, "nash_depth": nash,
                       "fixed_fee_bps": args.fixed_fee_bps,
                       "cost": cost_rows, "selectivity": sel_rows,
                       "verdicts": v}, fh)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
