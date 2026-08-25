"""Where does the directional strategy overtake the market-making one?

The two strategies are opposite sides of one wedge. A market maker earns the
spread and pays the post-fill drift; a directional taker pays the spread and
earns the drift. Which is viable is decided by the size of the drift relative to
the spread -- in this simulator, by the market-order impact ``eps``.

This sweeps ``eps`` and runs both strategies on identical paths, alongside three
controls that exist to catch the strategy fooling itself:

  flat        never trades; P&L must be exactly zero
  buy-hold    holds one unit throughout; isolates exposure from prediction
  inverted    the directional trader on the negative of its own signal
  shuffled    the same signal distribution with its timing destroyed

If the live arm's P&L is real prediction, ``inverted`` should lose roughly what
``live`` makes and ``shuffled`` should sit near zero minus costs. If all four
look alike, the P&L was never a signal.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.analytics import decompose_pnl, equity_grid, summarise
from src.backtest import nash_benchmark_depth
from src.directional import (
    BuyAndHoldTrader,
    FlatTrader,
    RobustDirectionalTrader,
    TraderConfig,
)
from src.market import CompetitionParams, DealerMarket, TrueDynamics
from src.strategy import AdaptiveRAMM, AmbiguityPolicy


def build_market(seed, nash_depth, q_max, theta, eps):
    return DealerMarket(
        dynamics=TrueDynamics(eps=eps),
        competition=CompetitionParams(nash_depth=nash_depth),
        q_max=q_max, theta=theta, rng=np.random.default_rng(seed),
    )


def run_taker_arm(factory, seeds, horizon, nash_depth, q_max, theta, eps):
    results, trades = [], []
    for seed in seeds:
        trader = factory()
        market = build_market(seed, nash_depth, q_max, theta, eps)
        results.append(market.run_taker(trader, horizon))
        trades.append(getattr(trader, "n_trades", 0))
    return results, float(np.mean(trades))


def run_maker_arm(factory, seeds, horizon, nash_depth, q_max, theta, eps):
    results = []
    for seed in seeds:
        market = build_market(seed, nash_depth, q_max, theta, eps)
        results.append(market.run(factory(), horizon))
    return results, 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=int, default=80)
    ap.add_argument("--horizon", type=float, default=1200.0)
    ap.add_argument("--q-max", type=int, default=8)
    ap.add_argument("--theta", type=float, default=0.001)
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--phi-alpha", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=20240614)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    nash_depth = nash_benchmark_depth(q_max=args.q_max)
    seeds = list(np.random.default_rng(args.seed).integers(0, 2**31 - 1, size=args.paths))
    tc = dict(phi_alpha=args.phi_alpha, q_max=args.q_max)

    arms = [
        ("flat", "taker", lambda: FlatTrader()),
        ("buy-hold", "taker", lambda: BuyAndHoldTrader()),
        ("directional / live", "taker",
         lambda: RobustDirectionalTrader(TraderConfig(signal_mode="live", **tc))),
        ("directional / inverted", "taker",
         lambda: RobustDirectionalTrader(TraderConfig(signal_mode="inverted", **tc))),
        ("directional / shuffled", "taker",
         lambda: RobustDirectionalTrader(TraderConfig(signal_mode="shuffled", **tc))),
        ("market maker (RAMM)", "maker",
         lambda: AdaptiveRAMM(policy=AmbiguityPolicy(nash_depth=nash_depth),
                              kappa0=27.0, lam0=2.0, sigma0=0.01, theta=args.theta,
                              q_max=args.q_max, mu0=nash_depth)),
    ]

    print(f"eps = {args.eps:g}   {args.paths} paths x {args.horizon:g}s   "
          f"phi_alpha = {args.phi_alpha:g}")
    header = (f"{'arm':<24}{'meanPnL':>9}{'sem':>7}{'Sharpe':>8}{'trades':>8}"
              f"{'win%':>7}{'spread':>9}{'markout':>9}")
    print(header)
    print("-" * len(header))

    payload = {"eps": args.eps, "paths": args.paths, "horizon": args.horizon,
               "phi_alpha": args.phi_alpha, "nash_depth": nash_depth, "arms": []}

    for name, kind, factory in arms:
        runner = run_taker_arm if kind == "taker" else run_maker_arm
        results, mean_trades = runner(factory, seeds, args.horizon, nash_depth,
                                      args.q_max, args.theta, args.eps)
        m = summarise(name, results, args.horizon)
        attrs = [decompose_pnl(r) for r in results]
        spread = float(np.mean([a.spread_capture for a in attrs]))
        markout = float(np.mean([-a.adverse_selection for a in attrs]))
        sem = m.std_pnl / np.sqrt(len(results))
        n_trades = mean_trades if kind == "taker" else m.mean_fills
        print(f"{name:<24}{m.mean_pnl:>9.3f}{sem:>7.3f}{m.sharpe_episode:>8.3f}"
              f"{n_trades:>8.0f}{m.win_rate*100:>7.0f}{spread:>9.3f}{markout:>9.3f}")
        grid = equity_grid(results, args.horizon, n_points=150)
        payload["arms"].append({
            "name": name, "kind": kind, "metrics": m.as_dict(),
            "mean_trades": n_trades, "sem": sem,
            "spread": spread, "markout": markout,
            "pnl_distribution": sorted(float(r.pnl) for r in results),
            "equity_time": np.linspace(0, args.horizon, grid.shape[1]).round(2).tolist(),
            "equity_mean": grid.mean(axis=0).round(5).tolist(),
            "equity_p05": np.percentile(grid, 5, axis=0).round(5).tolist(),
            "equity_p25": np.percentile(grid, 25, axis=0).round(5).tolist(),
            "equity_p75": np.percentile(grid, 75, axis=0).round(5).tolist(),
            "equity_p95": np.percentile(grid, 95, axis=0).round(5).tolist(),
        })

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
