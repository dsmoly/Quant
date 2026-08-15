"""Does the optimal ambiguity vector actually depend on the competitive regime?

This is the test that decides whether the adaptive idea has any content.  We pin
the market in one regime at a time -- permanently competitive (population quoting
at the mean-field Nash level) and permanently supra-competitive (population
quoting 25% wide of it, the level the mean-field paper's learning dealers drift
to) -- and sweep a *fixed* ambiguity vector over each.

The decision-relevant quantity is not whether the two argmaxes happen to land on
different grid cells.  It is the *value of adaptation*:

    best achievable by switching per regime  -  best achievable by one fixed vector

If that gap is near zero, a single static robust strategy is all anyone needs and
regime detection is wasted machinery, however good the story sounds.

``--eps`` scales the market-order impact on the short-term alpha, i.e. how toxic
the flow is.  The papers' own calibration (eps = 0.001 against spreads of order
0.03) makes adverse selection a small share of the spread captured; this sweep is
the place to find out whether that is why regime switching does or does not pay.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from src.backtest import nash_benchmark_depth
from src.market import CompetitionParams, DealerMarket, TrueDynamics
from src.strategy import MarketMaker

PHI_ALPHAS = (0.0, 2.0, 6.0, 15.0)
PHIS = (0.0, 8.0, 16.0, 32.0)


def pinned_competition(nash_depth: float, supra: bool, supra_multiple: float = 1.25):
    """Competition parameters with the regime frozen (no switching)."""
    return CompetitionParams(
        nash_depth=nash_depth * (supra_multiple if supra else 1.0),
        supra_multiple=1.0,
        mean_supra_seconds=1e12,
        mean_comp_seconds=1e12,
    )


def run_grid(nash_depth: float, supra: bool, seeds, horizon: float, q_max: int,
             eps: float, phi_alphas=PHI_ALPHAS, phis=PHIS) -> dict:
    out = {}
    comp = pinned_competition(nash_depth, supra)
    dyn = TrueDynamics(eps=eps)
    for pa, ph in itertools.product(phi_alphas, phis):
        pnls, fills, tails = [], [], []
        for seed in seeds:
            mm = MarketMaker(kappa0=27.0, lam0=2.0, sigma0=0.01, theta=0.001, q_max=q_max,
                             phi_alpha=pa, phi=ph, mu0=nash_depth)
            market = DealerMarket(dyn, comp, q_max=q_max, theta=0.001,
                                  rng=np.random.default_rng(seed))
            res = market.run(mm, horizon)
            pnls.append(res.pnl)
            fills.append(res.n_fills)
            if res.inventory_path.size:
                tails.append(float(np.mean(np.abs(res.inventory_path) > 0.6 * q_max)))
        arr = np.array(pnls)
        mean, std = float(arr.mean()), float(arr.std(ddof=1))
        out[(pa, ph)] = dict(
            mean=mean, std=std, sem=std / np.sqrt(len(arr)),
            sharpe=mean / std if std else float("nan"),
            fills=float(np.mean(fills)),
            tail=float(np.mean(tails)) if tails else float("nan"),
        )
    return out


def report(title: str, grid: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"{'phi_alpha':>9} {'phi':>6} {'meanPnL':>9} {'+/-sem':>7} {'sdPnL':>8} "
          f"{'Sharpe':>8} {'fills':>7} {'invTail':>8}")
    for (pa, ph), s in grid.items():
        print(f"{pa:>9.1f} {ph:>6.1f} {s['mean']:>9.3f} {s['sem']:>7.3f} {s['std']:>8.3f} "
              f"{s['sharpe']:>8.3f} {s['fills']:>7.0f} {s['tail']:>8.3f}")


def value_of_adaptation(comp_grid: dict, supra_grid: dict, key: str) -> dict:
    """Gap between the per-regime best and the best single fixed vector.

    Both regimes are weighted equally, matching the symmetric dwell times in the
    live simulator.
    """
    best_c = max(comp_grid, key=lambda k: comp_grid[k][key])
    best_s = max(supra_grid, key=lambda k: supra_grid[k][key])
    switching = 0.5 * (comp_grid[best_c][key] + supra_grid[best_s][key])
    fixed_key = max(comp_grid, key=lambda k: 0.5 * (comp_grid[k][key] + supra_grid[k][key]))
    static = 0.5 * (comp_grid[fixed_key][key] + supra_grid[fixed_key][key])
    # Monte Carlo noise floor on the switching estimate, for mean P&L only.
    noise = float("nan")
    if key == "mean":
        noise = 0.5 * np.hypot(comp_grid[best_c]["sem"], supra_grid[best_s]["sem"])
    return dict(metric=key, best_competitive=best_c, best_supra=best_s,
                best_static=fixed_key, switching=switching, static=static,
                gain=switching - static, noise=noise)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", type=int, default=60)
    ap.add_argument("--horizon", type=float, default=900.0)
    ap.add_argument("--q-max", type=int, default=8)
    ap.add_argument("--eps", type=float, default=0.001,
                    help="market-order impact on the short-term alpha (flow toxicity)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quiet", action="store_true", help="print only the verdict")
    args = ap.parse_args(argv)

    nash_depth = nash_benchmark_depth(q_max=args.q_max)
    seeds = list(np.random.default_rng(args.seed).integers(0, 2**31 - 1, size=args.paths))

    print(f"mean-field Nash half-spread: {nash_depth:.5f}   flow toxicity eps={args.eps:g}")
    print(f"{args.paths} common-random-number paths x {args.horizon:g}s per grid point")

    comp_grid = run_grid(nash_depth, False, seeds, args.horizon, args.q_max, args.eps)
    supra_grid = run_grid(nash_depth, True, seeds, args.horizon, args.q_max, args.eps)

    if not args.quiet:
        report("COMPETITIVE market (population at Nash)", comp_grid)
        report("SUPRA-COMPETITIVE market (population 25% wide of Nash)", supra_grid)

    print("\n--- value of regime adaptation ---")
    for key in ("mean", "sharpe"):
        v = value_of_adaptation(comp_grid, supra_grid, key)
        label = "mean P&L" if key == "mean" else "Sharpe"
        print(f"\n{label}:")
        print(f"  best fixed vector overall        : phi_alpha={v['best_static'][0]:g}, "
              f"phi={v['best_static'][1]:g}   -> {v['static']:.4f}")
        print(f"  best in competitive regime       : phi_alpha={v['best_competitive'][0]:g}, "
              f"phi={v['best_competitive'][1]:g}")
        print(f"  best in supra-competitive regime : phi_alpha={v['best_supra'][0]:g}, "
              f"phi={v['best_supra'][1]:g}")
        print(f"  oracle switching                 : {v['switching']:.4f}")
        gain_pct = 100.0 * v["gain"] / abs(v["static"]) if v["static"] else float("nan")
        print(f"  VALUE OF ADAPTATION              : {v['gain']:+.4f} ({gain_pct:+.2f}%)"
              + (f"   [MC noise ~{v['noise']:.4f}]" if np.isfinite(v["noise"]) else ""))
        if v["best_competitive"] == v["best_supra"]:
            print("  -> same argmax in both regimes: nothing to switch on for this metric.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
