"""Generate equity curves and the full metric set for every arm.

Writes a JSON blob consumed by ``experiments/build_report.py`` to render the
report page.  Run once per flow-toxicity level.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.analytics import equity_grid, summarise
from src.backtest import _build_market, nash_benchmark_depth
from src.strategy import (
    AdaptiveRAMM,
    AmbiguityPolicy,
    fixed_robust_market_maker,
    neutral_market_maker,
)


def build_arms(nash_depth: float, q_max: int, theta: float):
    common = dict(kappa0=27.0, lam0=2.0, sigma0=0.01, theta=theta, q_max=q_max,
                  mu0=nash_depth)
    frozen = dict(common, adapt_reference=False)
    policy = AmbiguityPolicy(nash_depth=nash_depth)
    return [
        ("neutral / frozen", lambda: neutral_market_maker(**frozen)),
        ("neutral / adaptive-ref", lambda: neutral_market_maker(**common)),
        ("robust / frozen", lambda: fixed_robust_market_maker(phi_alpha=6.0, phi=4.0, **frozen)),
        ("robust / adaptive-ref", lambda: fixed_robust_market_maker(phi_alpha=6.0, phi=4.0, **common)),
        ("ramm / adaptive-phi", lambda: AdaptiveRAMM(policy=policy, **common)),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=int, default=120)
    ap.add_argument("--horizon", type=float, default=1200.0)
    ap.add_argument("--q-max", type=int, default=8)
    ap.add_argument("--theta", type=float, default=0.001)
    ap.add_argument("--eps", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=20240614)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args(argv)

    nash_depth = nash_benchmark_depth(q_max=args.q_max)
    seeds = list(np.random.default_rng(args.seed).integers(0, 2**31 - 1, size=args.paths))

    payload = {
        "eps": args.eps, "paths": args.paths, "horizon": args.horizon,
        "q_max": args.q_max, "nash_depth": nash_depth, "arms": [],
    }

    for name, factory in build_arms(nash_depth, args.q_max, args.theta):
        results = []
        phi_alpha_log, phi_log = [], []
        for seed in seeds:
            mm = factory()
            market = _build_market(seed, nash_depth, args.q_max, args.theta, args.eps)
            results.append(market.run(mm, args.horizon))
            if isinstance(mm, AdaptiveRAMM) and mm.phi_log:
                phi_alpha_log.append(float(np.mean(mm.phi_alpha_log)))
                phi_log.append(float(np.mean(mm.phi_log)))

        metrics = summarise(name, results, args.horizon)
        grid = equity_grid(results, args.horizon, n_points=180)
        payload["arms"].append({
            "name": name,
            "metrics": metrics.as_dict(),
            "mean_pnl": float(metrics.mean_pnl),
            "mean_phi_alpha": float(np.mean(phi_alpha_log)) if phi_alpha_log else None,
            "mean_phi": float(np.mean(phi_log)) if phi_log else None,
            # Equity curve summary: mean and a fan of percentiles across paths.
            "equity_time": np.linspace(0, args.horizon, grid.shape[1]).round(2).tolist(),
            "equity_mean": grid.mean(axis=0).round(5).tolist(),
            "equity_p05": np.percentile(grid, 5, axis=0).round(5).tolist(),
            "equity_p25": np.percentile(grid, 25, axis=0).round(5).tolist(),
            "equity_p75": np.percentile(grid, 75, axis=0).round(5).tolist(),
            "equity_p95": np.percentile(grid, 95, axis=0).round(5).tolist(),
            # A handful of individual paths, for texture.
            "equity_samples": grid[:8].round(5).tolist(),
            "pnl_distribution": sorted(float(r.pnl) for r in results),
        })
        m = metrics
        print(f"{name:<24} meanPnL {m.mean_pnl:>8.3f}  Sharpe {m.sharpe_episode:>7.3f}  "
              f"maxDD {m.max_drawdown:>6.3f}  fills {m.mean_fills:>5.0f}  "
              f"capture {m.capture_ratio:>6.3f}")
        del results

    with open(args.out, "w") as fh:
        json.dump(payload, fh)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
