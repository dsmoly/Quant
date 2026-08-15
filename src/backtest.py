"""Monte Carlo comparison of the market-making arms.

Three strategies are run on identical simulated paths (common random numbers, so
the differences are the strategies and not the draws):

  neutral       ambiguity-neutral reference-model market maker (phi = 0)
  robust        the robust paper's strategy with a fixed ambiguity vector
  ramm          the adaptive strategy: ambiguity vector driven by the estimated
                distance between the population quote and the mean-field Nash
                benchmark

Reported per arm: mean and standard deviation of terminal P&L, the Sharpe-like
ratio the robust paper uses (mean over standard deviation of P&L), fill counts,
P&L per fill, and the tail weight of the inventory distribution -- the last
because the mean-field paper identifies heavy inventory tails as the signature of
a market maker being pushed into supra-competitive quoting.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np

from .market import CompetitionParams, DealerMarket, TrueDynamics
from .mfg_equilibrium import MFGConfig, solve_mfg
from .strategy import (
    AdaptiveRAMM,
    AmbiguityPolicy,
    MarketMaker,
    fixed_robust_market_maker,
    neutral_market_maker,
)

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", ".mfg_cache.json")


def nash_benchmark_depth(kappa_ref: float = 27.0, lam: float = 2.0, q_max: int = 8,
                         use_cache: bool = True) -> float:
    """Mean-field Nash half-spread in price units, the strategy's competitive benchmark.

    The mean-field game is solved in dimensionless units where the monopolistic
    fill rate is exp(-delta); the robust model works in price units where it is
    exp(-kappa * delta).  The two agree under delta_price = delta_mfg / kappa.
    The solve takes several seconds, so the result is cached on disk.
    """
    key = f"{kappa_ref}:{lam}:{q_max}"
    if use_cache and os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH) as fh:
                cache = json.load(fh)
            if key in cache:
                return float(cache[key])
        except (OSError, ValueError):
            cache = {}
    else:
        cache = {}

    sol = solve_mfg(MFGConfig(Z=q_max, lam_a=lam, lam_b=lam))
    depth = sol.mean_quote() / kappa_ref
    cache[key] = depth
    try:
        with open(CACHE_PATH, "w") as fh:
            json.dump(cache, fh, indent=2)
    except OSError:
        pass
    return depth


@dataclass
class ArmStats:
    name: str
    mean_pnl: float
    std_pnl: float
    sharpe: float
    mean_fills: float
    pnl_per_fill: float
    hit_rate: float
    inv_rms: float
    inv_tail: float          # fraction of time |q| > 0.6 q_max
    mean_phi: float = float("nan")
    mean_phi_alpha: float = float("nan")

    def row(self) -> str:
        return (f"{self.name:<22} {self.mean_pnl:>9.3f} {self.std_pnl:>9.3f} {self.sharpe:>8.3f} "
                f"{self.mean_fills:>8.0f} {self.pnl_per_fill:>10.4f} {self.hit_rate:>7.3f} "
                f"{self.inv_rms:>7.2f} {self.inv_tail:>8.3f} "
                f"{self.mean_phi_alpha:>7.2f} {self.mean_phi:>6.2f}")


def _named(mm, name: str):
    """Label an arm for the results table."""
    mm.name = name
    return mm


def _build_market(seed: int, nash_depth: float, q_max: int, theta: float,
                  eps: float = 0.001) -> DealerMarket:
    return DealerMarket(
        dynamics=TrueDynamics(eps=eps),
        competition=CompetitionParams(nash_depth=nash_depth),
        q_max=q_max,
        theta=theta,
        rng=np.random.default_rng(seed),
    )


def run_arm(factory, seeds, horizon: float, nash_depth: float, q_max: int,
            theta: float, eps: float = 0.001) -> ArmStats:
    pnls, fills, rfqs, inv_rms, inv_tail = [], [], [], [], []
    phis, phi_alphas = [], []
    name = None
    for seed in seeds:
        mm = factory()
        name = name or mm.name
        market = _build_market(seed, nash_depth, q_max, theta, eps)
        res = market.run(mm, horizon)
        pnls.append(res.pnl)
        fills.append(res.n_fills)
        rfqs.append(res.n_rfq)
        if res.inventory_path.size:
            inv_rms.append(float(np.sqrt(np.mean(res.inventory_path.astype(float) ** 2))))
            inv_tail.append(float(np.mean(np.abs(res.inventory_path) > 0.6 * q_max)))
        if isinstance(mm, AdaptiveRAMM) and mm.phi_log:
            phis.append(float(np.mean(mm.phi_log)))
            phi_alphas.append(float(np.mean(mm.phi_alpha_log)))

    pnls = np.array(pnls, dtype=float)
    mean, std = float(pnls.mean()), float(pnls.std(ddof=1))
    total_fills = float(np.mean(fills))
    return ArmStats(
        name=name or "?",
        mean_pnl=mean,
        std_pnl=std,
        sharpe=mean / std if std > 0 else float("nan"),
        mean_fills=total_fills,
        pnl_per_fill=mean / max(total_fills, 1e-9),
        hit_rate=float(np.mean(fills)) / max(float(np.mean(rfqs)), 1e-9),
        inv_rms=float(np.mean(inv_rms)) if inv_rms else float("nan"),
        inv_tail=float(np.mean(inv_tail)) if inv_tail else float("nan"),
        mean_phi=float(np.mean(phis)) if phis else float("nan"),
        mean_phi_alpha=float(np.mean(phi_alphas)) if phi_alphas else float("nan"),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", type=int, default=200, help="Monte Carlo paths per arm")
    ap.add_argument("--horizon", type=float, default=1800.0, help="episode length in seconds")
    ap.add_argument("--q-max", type=int, default=8, help="inventory limit")
    ap.add_argument("--theta", type=float, default=0.001, help="liquidation penalty")
    ap.add_argument("--eps", type=float, default=0.001,
                    help="market-order impact on the short-term alpha (flow toxicity)")
    ap.add_argument("--seed", type=int, default=20240614)
    ap.add_argument("--json", type=str, default=None, help="write results to this JSON file")
    args = ap.parse_args(argv)

    nash_depth = nash_benchmark_depth(q_max=args.q_max)
    seeds = list(np.random.default_rng(args.seed).integers(0, 2**31 - 1, size=args.paths))

    common = dict(kappa0=27.0, lam0=2.0, sigma0=0.01, theta=args.theta, q_max=args.q_max,
                  mu0=nash_depth)
    policy = AmbiguityPolicy(nash_depth=nash_depth)

    # The arms decompose the strategy into its two independent claims.
    #
    #   frozen vs adaptive reference model
    #       Isolates the mean-field channel.  A frozen market maker prices off the
    #       long-run (lambda, kappa, sigma) forever.  An adaptive one recalibrates
    #       kappa online, which implicitly tracks the population quote: when the
    #       crowd widens, the same depth wins more often, the estimated kappa
    #       falls, and the quote widens with the crowd.  That is the *right*
    #       direction under the mean-field model, where quotes are strategic
    #       complements -- not the undercutting the adaptive-phi story assumes.
    #
    #   phi = 0 vs fixed phi vs adaptive phi
    #       Isolates the robust channel and then the regime switch on top of it.
    def frozen(**kw):
        return dict(common, adapt_reference=False, **kw)

    arms = [
        lambda: _named(neutral_market_maker(**frozen()), "neutral/frozen"),
        lambda: _named(neutral_market_maker(**common), "neutral/adaptive-ref"),
        lambda: _named(fixed_robust_market_maker(phi_alpha=6.0, phi=4.0, **frozen()),
                       "robust/frozen"),
        lambda: _named(fixed_robust_market_maker(phi_alpha=6.0, phi=4.0, **common),
                       "robust/adaptive-ref"),
        lambda: _named(AdaptiveRAMM(policy=policy, **common), "ramm/adaptive-phi"),
    ]

    print(f"mean-field Nash half-spread benchmark: {nash_depth:.5f} "
          f"(= {nash_depth * 27:.4f} in mean-field units)")
    print(f"{args.paths} paths x {args.horizon:g}s, inventory limit +/-{args.q_max}, "
          f"flow toxicity eps={args.eps:g}\n")
    header = (f"{'arm':<22} {'meanPnL':>9} {'sdPnL':>9} {'Sharpe':>8} {'fills':>8} "
              f"{'PnL/fill':>10} {'hit':>7} {'invRMS':>7} {'invTail':>8} {'phi_a':>7} {'phi':>6}")
    print(header)
    print("-" * len(header))

    results = []
    for factory in arms:
        stats = run_arm(factory, seeds, args.horizon, nash_depth, args.q_max,
                        args.theta, args.eps)
        results.append(stats)
        print(stats.row())

    if args.json:
        with open(args.json, "w") as fh:
            json.dump([vars(r) for r in results], fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
