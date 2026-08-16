"""What does the sizing engine contribute, independent of alpha?

This runs *before* any footprint feature is built, deliberately, so the
attribution is clean. Two questions, and the first one matters more:

  1. Does the engine manufacture P&L from a signal with no predictive power?
     A framework with volatility targeting, shrinkage, a no-trade band and risk
     budgeting has a lot of moving parts, and several of them could quietly
     introduce a real exposure -- to short volatility, to mean reversion, to the
     market -- that shows up as profit and gets mistaken for the signal working.
     The dumb-signal arm is pure noise, redrawn independently of returns. Its net
     Sharpe must be indistinguishable from zero. **If it is not, that is a bug,
     and no result from the real features can be trusted until it is found.**

  2. On a signal that *does* predict, by construction and with a known IC, how
     much does the engine add over the naive quintile portfolio, and which
     component adds it? Ablations turn each piece off in turn:

         no vol target   equal weight instead of q ~ 1/sigma
         no shrinkage    trade the raw IC estimate, however badly measured
         no band         rebalance all the way to target every day
         no haircut      size as if N names were N independent bets

Read the ablations as costs of removal, not as standalone contributions -- the
components interact, and the band in particular only pays because the target it
is applied to is stable enough to sit inside it.

The panel is synthetic, and that is the point: the true IC is set by
construction, so "how much of the available edge did the engine capture" has an
exact answer. No number here describes US equities.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np
import pandas as pd

from src.backtest import BacktestConfig, run_backtest, run_quintile_backtest
from src.breadth import breadth_report, fundamental_law_ir
from src.metrics import summarise_performance
from src.sizing import SizingEngine, SizingParams
from src.synthetic import SyntheticSpec, simulate
from src.universe import liquid_universe


def _one_run(seed, *, planted_ic, cfg, sizing, arm, n_names, n_days, horizon):
    spec = SyntheticSpec(seed=seed, n_names=n_names, n_days=n_days)
    sim = simulate(spec, planted_ic=planted_ic, horizon=horizon)
    close = sim.panel.close
    univ = liquid_universe(sim.panel, n=min(20, n_names), window=60)

    if arm == "quintile":
        res = run_quintile_backtest(signal=sim.signal, close=close, config=cfg,
                                    universe=univ)
    else:
        res = run_backtest(signal=sim.signal, close=close, config=replace(cfg, sizing=sizing),
                           universe=univ, engine=SizingEngine(sizing))

    perf = summarise_performance(res.net_returns, turnover=res.turnover,
                                 gross_returns=res.gross_returns)
    rep = breadth_report(close.pct_change().iloc[-500:])
    realised_vol = float(res.net_returns.std(ddof=1) * np.sqrt(252))
    return {
        "seed": seed, "sharpe": perf.sharpe, "ann_return": perf.ann_return,
        "ann_vol": perf.ann_vol, "max_dd": perf.max_drawdown,
        "turnover": perf.mean_turnover, "breakeven_bps": perf.breakeven_cost_bps,
        "gross_sharpe": summarise_performance(res.gross_returns).sharpe,
        "realised_vol": realised_vol,
        "n_eff_raw": rep["n_eff_raw"], "n_eff_residual": rep["n_eff_residual"],
        "pc1_share": rep["pc1_share"], "planted_ic_realised": sim.realised_ic,
        "mean_shrinkage": (float(res.diagnostics["mean_shrinkage"].mean())
                           if not res.diagnostics.empty else float("nan")),
        "mean_gross_exposure": (float(res.diagnostics["gross"].mean())
                                if not res.diagnostics.empty else float("nan")),
    }


def run_arm(name, seeds, **kw):
    rows = [_one_run(s, **kw) for s in seeds]
    df = pd.DataFrame(rows)
    n = len(df)
    out = {"arm": name, "n_seeds": n}
    for c in ("sharpe", "gross_sharpe", "ann_return", "ann_vol", "realised_vol",
              "max_dd", "turnover", "breakeven_bps", "mean_shrinkage",
              "mean_gross_exposure", "n_eff_raw", "n_eff_residual", "pc1_share"):
        vals = df[c].replace([np.inf, -np.inf], np.nan).dropna()
        out[c] = float(vals.mean()) if len(vals) else float("nan")
        if c == "sharpe":
            out["sharpe_sem"] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")
    return out, df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--names", type=int, default=25)
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--planted-ic", type=float, default=0.03)
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--rebalance", type=str, default="D")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    seeds = list(range(args.seeds))
    cfg = BacktestConfig(cost_bps=args.cost_bps, rebalance=args.rebalance,
                         horizon=args.horizon, lag=1)
    full = SizingParams()
    base = dict(cfg=cfg, n_names=args.names, n_days=args.days, horizon=args.horizon)

    print(__doc__.split("\n\n")[0])
    print(f"\n{args.seeds} seeds x {args.days} sessions x {args.names} names   "
          f"cost={args.cost_bps:g}bps/side   rebalance={args.rebalance}   "
          f"planted IC={args.planted_ic:g}")

    arms = [
        # ---- question 1: does the framework invent P&L from noise?
        ("DUMB noise / quintile", dict(planted_ic=0.0, sizing=full, arm="quintile")),
        ("DUMB noise / full engine", dict(planted_ic=0.0, sizing=full, arm="engine")),
        # ---- question 2: on known alpha, what does each piece contribute?
        ("planted / quintile", dict(planted_ic=args.planted_ic, sizing=full, arm="quintile")),
        ("planted / full engine", dict(planted_ic=args.planted_ic, sizing=full, arm="engine")),
        ("planted / no vol target", dict(planted_ic=args.planted_ic, arm="engine",
                                         sizing=replace(full, vol_scaling=False))),
        ("planted / no shrinkage", dict(planted_ic=args.planted_ic, arm="engine",
                                        sizing=replace(full, use_shrinkage=False))),
        ("planted / no band", dict(planted_ic=args.planted_ic, arm="engine",
                                   sizing=replace(full, use_band=False))),
        ("planted / no haircut", dict(planted_ic=args.planted_ic, arm="engine",
                                      sizing=replace(full, breadth_haircut=False))),
    ]

    hdr = (f"{'arm':<26}{'netSh':>7}{'sem':>6}{'grossSh':>8}{'annVol':>8}"
           f"{'maxDD':>8}{'turn':>7}{'be(bps)':>9}{'shrink':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))

    results, frames = [], {}
    for name, kw in arms:
        r, df = run_arm(name, seeds, **base, **kw)
        results.append(r)
        frames[name] = df
        print(f"{name:<26}{r['sharpe']:>7.2f}{r['sharpe_sem']:>6.2f}"
              f"{r['gross_sharpe']:>8.2f}{r['ann_vol']:>8.3f}{r['max_dd']:>8.2f}"
              f"{r['turnover']:>7.3f}{r['breakeven_bps']:>9.1f}"
              f"{r['mean_shrinkage']:>8.2f}")

    by = {r["arm"]: r for r in results}
    print("\n--- question 1: does the engine invent P&L from noise? ---")
    print("  The test is on GROSS Sharpe. Net Sharpe on a zero-alpha signal is")
    print("  *supposed* to be negative -- that is the cost of trading a signal")
    print("  that does not predict, and it is the correct answer, not a bug.")
    print("  What would be a bug is gross P&L appearing out of noise.")
    for arm in ("DUMB noise / quintile", "DUMB noise / full engine"):
        r = by[arm]
        gs = r["gross_sharpe"]
        verdict = ("PASS (no gross edge from noise)" if abs(gs) < 0.5
                   else "FAIL -- gross edge from a zero-alpha signal; find it")
        print(f"  {arm:<26} gross {gs:+.2f}  net {r['sharpe']:+.2f}  "
              f"turnover {r['turnover']:.3f}  exposure {r['mean_gross_exposure']:.3f}"
              f"  {verdict}")
    dumb = by["DUMB noise / full engine"]
    print(f"\n  The engine's own defence against a worthless signal is to size it "
          f"small:\n  mean shrinkage factor {dumb['mean_shrinkage']:.3f}, "
          f"mean gross exposure {dumb['mean_gross_exposure']:.3f} "
          f"(vs {by['planted / full engine']['mean_gross_exposure']:.3f} on real alpha).")

    print("\n--- question 2: what does the engine add on known alpha? ---")
    q, f = by["planted / quintile"], by["planted / full engine"]
    d = f["sharpe"] - q["sharpe"]
    se = np.hypot(f["sharpe_sem"], q["sharpe_sem"])
    print(f"  full engine vs naive quintile: {d:+.2f} Sharpe ({d/se:+.1f} sigma)")
    print(f"  turnover {q['turnover']:.3f} -> {f['turnover']:.3f}, "
          f"breakeven {q['breakeven_bps']:.1f} -> {f['breakeven_bps']:.1f} bps")
    print("\n  ablations (cost of removing each piece, vs full engine):")
    for arm in ("planted / no vol target", "planted / no band", "planted / no haircut"):
        r = by[arm]
        print(f"    {arm.split('/')[1].strip():<16} {r['sharpe']-f['sharpe']:+.2f} Sharpe"
              f"   turnover {r['turnover']:.3f}   vol {r['ann_vol']:.3f}")

    print("\n--- volatility targeting: did it hit the target? ---")
    nominal = SizingParams().target_vol
    haircut = float(np.sqrt(max(f["n_eff_residual"], 1.0) / args.names))
    effective = nominal * f["mean_shrinkage"] * haircut
    print(f"  nominal target        {nominal:.3f}")
    print(f"  x confidence {f['mean_shrinkage']:.2f}  x breadth haircut {haircut:.2f}"
          f"  = effective target {effective:.3f}")
    print(f"  realised              {f['realised_vol']:.3f}  "
          f"(ratio to effective {f['realised_vol']/max(effective,1e-9):.2f})")
    print("  The gap between nominal and effective is deliberate: risk deployed is")
    print("  proportional to measured confidence. Comparing realised vol to the")
    print("  nominal target would make a working engine look broken.")
    print(f"\n  planted IC realised in the panel: "
          f"{np.mean([_one_run(s, planted_ic=args.planted_ic, sizing=full, arm='quintile', **base)['planted_ic_realised'] for s in seeds[:2]]):.4f} "
          f"(target {args.planted_ic:g})")

    print("\n--- breadth ---")
    print(f"  {args.names} names, PC1 explains {f['pc1_share']*100:.1f}% of variance")
    print(f"  effective N: raw {f['n_eff_raw']:.1f}, residual after removing PC1 "
          f"{f['n_eff_residual']:.1f}")
    ic = args.planted_ic
    print(f"  fundamental law at IC={ic:g}: naive IR = {fundamental_law_ir(ic, args.names):.2f}"
          f"  vs residual-breadth IR = {fundamental_law_ir(ic, f['n_eff_residual']):.2f}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"config": vars(args), "arms": results}, fh, indent=1, default=float)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
