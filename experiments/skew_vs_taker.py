"""Test 2: does skewing quotes beat crossing the spread?

Runs, on identical paths and with latency on:

  maker / RAMM        the current best market maker, no signal
  maker / skew        the same maker with its quotes leaned on the order-flow alpha
  maker / skew inv    the same skew applied backwards, as a control
  taker / directional the strategy that crosses the spread on the same signal

The hypothesis under test is that skewing converts adverse selection from a cost
toward zero, and beats the taker outright, because leaning a quote you are
already showing only requires alpha to beat zero rather than to beat the spread.

The metric that should move is the adverse-selection term -- not gross spread
capture, since the maker is choosing which side to be filled on rather than
quoting wider. Note that the capture ratio, useful while a maker is paying
markout, stops being interpretable the moment skewing drives adverse selection
negative: it is then no longer a share of anything, and is reported as n/a.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.analytics import decompose_pnl, equity_grid, summarise
from src.backtest import nash_benchmark_depth
from src.directional import RobustDirectionalTrader, TraderConfig
from src.market import CompetitionParams, DealerMarket, TrueDynamics
from src.skew_maker import plain_maker, skew_maker
from src.strategy import AmbiguityPolicy

BPS = 0.01


def build_market(seed, nash, q_max, theta, eps, latency, fee):
    return DealerMarket(
        dynamics=TrueDynamics(eps=eps),
        competition=CompetitionParams(nash_depth=nash),
        q_max=q_max, theta=theta, latency=latency, taker_fee=fee,
        rng=np.random.default_rng(seed))


def run_arm(name, factory, kind, seeds, *, eps, latency, fee, horizon, q_max, theta, nash):
    results, skews = [], []
    for seed in seeds:
        agent = factory()
        market = build_market(seed, nash, q_max, theta, eps, latency, fee)
        res = market.run_taker(agent, horizon) if kind == "taker" else market.run(agent, horizon)
        results.append(res)
        if hasattr(agent, "skew_log") and agent.skew_log:
            skews.append(float(np.mean(np.abs(agent.skew_log))))
    m = summarise(name, results, horizon)
    attrs = [decompose_pnl(r) for r in results]
    grid = equity_grid(results, horizon, n_points=150)
    return {
        "name": name, "kind": kind, "metrics": m.as_dict(),
        "spread": float(np.mean([a.spread_capture for a in attrs])),
        "adverse": float(np.mean([a.adverse_selection for a in attrs])),
        # The capture ratio is only interpretable while the maker is *paying*
        # markout on a positive gross spread. Once skewing drives adverse
        # selection negative the ratio exceeds 100% and stops being a share of
        # anything; for the taker, gross spread capture is negative and the ratio
        # is undefined outright. Report it only where it means something.
        "capture_ratio": (float(np.mean([a.capture_ratio for a in attrs
                                         if np.isfinite(a.capture_ratio)]))
                          if all(a.spread_capture > 0 and a.adverse_selection > 0
                                 for a in attrs) else float("nan")),
        "mean_abs_skew": float(np.mean(skews)) if skews else float("nan"),
        "sem": float(m.std_pnl / np.sqrt(len(results))),
        "equity_time": np.linspace(0, horizon, grid.shape[1]).round(2).tolist(),
        "equity_mean": grid.mean(axis=0).round(5).tolist(),
        "pnl_distribution": sorted(float(r.pnl) for r in results),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=int, default=60)
    ap.add_argument("--horizon", type=float, default=1200.0)
    ap.add_argument("--eps", type=float, default=0.015)
    ap.add_argument("--latency", type=float, default=0.05)
    ap.add_argument("--fee-bps", type=float, default=1.0)
    ap.add_argument("--selectivity", type=float, default=1.5,
                    help="band multiplier for the taker; test 1's optimum")
    ap.add_argument("--q-max", type=int, default=8)
    ap.add_argument("--theta", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=20240614)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    nash = nash_benchmark_depth(q_max=args.q_max)
    seeds = list(np.random.default_rng(args.seed).integers(0, 2**31 - 1, size=args.paths))
    common = dict(kappa0=27.0, lam0=2.0, sigma0=0.01, theta=args.theta,
                  q_max=args.q_max, mu0=nash)
    pol = AmbiguityPolicy(nash_depth=nash)
    base = dict(eps=args.eps, latency=args.latency, fee=args.fee_bps * BPS,
                horizon=args.horizon, q_max=args.q_max, theta=args.theta, nash=nash)

    arms = [
        ("maker / RAMM", "maker", lambda: plain_maker(policy=pol, **common)),
        ("maker / skew", "maker", lambda: skew_maker(1.0, policy=pol, **common)),
        ("maker / skew inverted", "maker",
         lambda: skew_maker(1.0, invert=True, policy=pol, **common)),
        ("taker / directional", "taker",
         lambda: RobustDirectionalTrader(TraderConfig(
             phi_alpha=40.0, q_max=args.q_max, cost_multiple=args.selectivity))),
    ]

    print(f"eps={args.eps:g}  latency={args.latency*1000:.0f}ms  fee={args.fee_bps:g}bps  "
          f"taker selectivity x{args.selectivity:g}  {args.paths} paths x {args.horizon:g}s")
    hdr = (f"{'arm':<24}{'meanPnL':>10}{'sem':>7}{'Sharpe':>8}{'fills':>7}"
           f"{'spread':>9}{'adverse':>9}{'capture':>9}{'invRMS':>8}")
    print(hdr)
    print("-" * len(hdr))

    out = []
    for name, kind, factory in arms:
        r = run_arm(name, factory, kind, seeds, **base)
        out.append(r)
        m = r["metrics"]
        cap = (f"{r['capture_ratio']*100:>8.1f}%" if np.isfinite(r["capture_ratio"])
               else f"{'n/a':>9}")
        print(f"{name:<24}{m['mean_pnl']:>10.3f}{r['sem']:>7.3f}{m['sharpe_episode']:>8.2f}"
              f"{m['mean_fills']:>7.0f}{r['spread']:>9.2f}{r['adverse']:>9.2f}"
              f"{cap}{m['inv_rms']:>8.2f}")

    by = {r["name"]: r for r in out}
    ramm, skew, tak = by["maker / RAMM"], by["maker / skew"], by["taker / directional"]
    print("\n--- verdicts ---")
    d_adv = skew["adverse"] - ramm["adverse"]
    print(f"  adverse selection: {ramm['adverse']:.3f} -> {skew['adverse']:.3f} "
          f"({d_adv:+.3f}, {100*d_adv/abs(ramm['adverse']):+.1f}%)")
    if np.isfinite(skew["capture_ratio"]):
        print(f"  capture ratio:     {ramm['capture_ratio']*100:.1f}% -> "
              f"{skew['capture_ratio']*100:.1f}%")
    else:
        print(f"  capture ratio:     {ramm['capture_ratio']*100:.1f}% -> n/a "
              f"(adverse selection turned negative; the ratio is no longer a share "
              f"of gross spread)")
    dp = skew["metrics"]["mean_pnl"] - ramm["metrics"]["mean_pnl"]
    se = np.hypot(skew["sem"], ramm["sem"])
    print(f"  skew vs RAMM:      {dp:+.3f} P&L ({dp/se:.1f} sigma), "
          f"Sharpe {ramm['metrics']['sharpe_episode']:.2f} -> "
          f"{skew['metrics']['sharpe_episode']:.2f}")
    dt_ = skew["metrics"]["mean_pnl"] - tak["metrics"]["mean_pnl"]
    st = np.hypot(skew["sem"], tak["sem"])
    print(f"  skew vs taker:     {dt_:+.3f} P&L ({dt_/st:.1f} sigma), "
          f"Sharpe {tak['metrics']['sharpe_episode']:.2f} -> "
          f"{skew['metrics']['sharpe_episode']:.2f}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"eps": args.eps, "latency": args.latency, "fee_bps": args.fee_bps,
                       "paths": args.paths, "horizon": args.horizon, "nash_depth": nash,
                       "selectivity": args.selectivity, "arms": out}, fh)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
