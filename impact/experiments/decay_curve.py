"""Does order-flow impact decay in seconds, or is part of it permanent?

    python3 -m experiments.decay_curve --data-dir data --symbols BTCUSDT ETHUSDT

The question. A signal measured only at 1-10 second horizons appears to die
almost immediately. But what decays in seconds is *transient* impact -- the
dislocation that mean-reverts as liquidity replenishes -- and that is a different
quantity from *permanent* impact, the information component that by construction
does not decay. Extrapolating the first to minutes, hours and days is not
supported, and if a permanent component exists at those horizons then the whole
footprint formulation is the right one.

What is reported, in the order it should be read:

  1. Bounce contamination. How much the trade-price version of the response
     differs from the mid version, measured not modelled.
  2. lambda(W): impact per unit of signed flow, over windows from 1s to 5 days.
     The SHAPE is the answer -- decaying to zero means transient, converging to a
     positive level means permanent. This is the primary result.
  3. R(h): the full response curve, impact and predictive versions, with
     Newey-West errors and a strictly non-overlapping cross-check.
  4. Null floors from shuffled timing, at every horizon. A bare t-statistic is
     not interpretable; this is the bar each estimate must clear.
  5. Persistence: autocorrelation, half-life and integrated time, giving
     independent observations per year -- the number that sets breadth.
  6. Economics: predictable move in bps against round-trip costs, and where the
     crossover sits.

Without data it runs on fixture tapes with a *known* answer and says so loudly.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.binance import BinanceError, read_local
from src.econ import HORIZONS_S, CostModel, crossover, economics, fmt_horizon
from src.flow import (align_mid_to_trades, bounce_diagnostics, build_bars,
                      effective_half_spread_bps)
from src.nulls import null_curve
from src.propagator import fit_propagator, permanent_verdict, critical_gamma
from src.response import permanent_lambda_curve, persistence_curve, response_curve

BANNER = "!" * 78


def load(data_dir, symbol, market, max_days):
    if data_dir:
        try:
            d = read_local(data_dir, symbol, market=market, max_days=max_days)
            span = d.trades["ts"].max() - d.trades["ts"].min()
            print(f"REAL DATA {symbol}: {len(d.trades):,} trades over {len(d.days)} "
                  f"day(s), span {span}, mid={'yes' if d.has_mid else 'NO'}")
            return d, True
        except BinanceError as exc:
            print(f"{BANNER}\n!! {symbol}: {exc}\n"
                  f"!! FALLING BACK TO A FIXTURE TAPE WITH A KNOWN ANSWER.\n"
                  f"!! Nothing below describes a real venue.\n{BANNER}")
    from src.synth_tick import TickSpec, simulate_ticks
    kp = 0.0 if symbol.upper().startswith("TRANS") else 0.8
    print(f"{BANNER}\n!! SYNTHETIC TAPE for {symbol}: planted kappa_perm={kp}.\n"
          f"!! This validates the estimator. It is not a measurement.\n{BANNER}")
    t = simulate_ticks(TickSpec(n_trades=150_000, kappa_perm=kp, sigma_bps=0.2, seed=1))
    class _D:
        trades, book, days, symbol_ = t.trades, t.book, ["synthetic"], symbol
        has_mid = True
    return _D(), False


def analyse(symbol, d, args) -> dict:
    bars = build_bars(d.trades, args.bar, d.book if args.use_mid else None)
    n = len(bars.trade_price)
    print(f"\n{'='*78}\n{symbol}   {n:,} bars of {args.bar}\n{'='*78}")

    # ---- 1. bounce
    print("\n-- 1. bid-ask bounce contamination --")
    diag = bounce_diagnostics(bars, d.trades)
    if bars.has_mid:
        joined = align_mid_to_trades(d.trades, d.book)
        eff = effective_half_spread_bps(joined)
        print(f"   effective half-spread (measured against mid): {eff:.3f} bps")
    else:
        eff = float("nan")
        print("   NO BOOK: trade prints only.")
    rs = diag.get("tick_roll_spread_bps", float("nan"))
    print(f"   Roll estimate from trade prints: "
          + (f"{rs:.3f} bps" if np.isfinite(rs) else
             "UNUSABLE (serial covariance positive -- order flow is "
             "autocorrelated, which breaks Roll's assumption)"))
    print(f"   bar-level return autocorrelation: trade {diag['trade_ac1']:+.4f}"
          + (f"   mid {diag['mid_ac1']:+.4f}" if bars.has_mid else ""))

    if bars.has_mid:
        a = response_curve(bars, [1, 10, 60], measure="impact", price_kind="mid")
        b = response_curve(bars, [1, 10, 60], measure="impact", price_kind="trade")
        m = a.merge(b, on="horizon_s", suffixes=("_mid", "_trade"))
        print("   R(h) on mid vs on trade prints:")
        for _, r in m.iterrows():
            gap = r["r_bps_trade"] - r["r_bps_mid"]
            print(f"     {fmt_horizon(r['horizon_s']):>6}  mid {r['r_bps_mid']:+8.3f}"
                  f"   trade {r['r_bps_trade']:+8.3f}   gap {gap:+.3f} bps "
                  f"({100*gap/abs(r['r_bps_mid']) if r['r_bps_mid'] else float('nan'):+.1f}%)")

    price_kind = "mid" if bars.has_mid else "trade"

    # ---- 2. the primary result
    print("\n-- 2. lambda(W): impact per UNIT of signed flow (the primary result) --")
    lam = permanent_lambda_curve(bars, args.horizons, price_kind=price_kind)
    if lam.empty:
        print("   sample too short for any window")
        return {"symbol": symbol, "error": "no windows"}
    print(f"   {'window':>8}{'lambda':>12}{'per-sd bps':>12}{'n_indep':>9}")
    for _, r in lam.iterrows():
        print(f"   {fmt_horizon(r['window_s']):>8}{r['lambda_per_unit']:>12.4f}"
              f"{r['r_bps']:>12.3f}{int(r['n_nonoverlap']):>9}")
    v = permanent_verdict(bars, args.horizons, price_kind=price_kind,
                          n_boot=args.boot)
    print(f"\n   log-log slope over the long half: {v.slope_log:+.3f}"
          f"   (-1 = purely transient, 0 = fully permanent)")
    print(f"   lambda at the longest window: {v.lambda_long:.4f} "
          f"[{v.lambda_long_lo:.4f}, {v.lambda_long_hi:.4f}] (95% block bootstrap)")
    print(f"   VERDICT: {v.verdict}")

    # ---- 3. response curves
    print("\n-- 3. response function R(h) --")
    hdr = (f"   {'horizon':>8}{'R impact':>11}{'NW t':>8}{'non-ov t':>10}"
           f"{'R predict':>11}{'NW t':>8}{'n_indep':>9}")
    print(hdr)
    imp = response_curve(bars, args.horizons, measure="impact", price_kind=price_kind)
    pre = response_curve(bars, args.horizons, measure="predictive",
                         aggregation_s=None, price_kind=price_kind)
    merged = imp.merge(pre, on="horizon_s", suffixes=("_i", "_p"), how="outer")
    for _, r in merged.iterrows():
        print(f"   {fmt_horizon(r['horizon_s']):>8}{r['r_bps_i']:>11.3f}"
              f"{r['r_t_i']:>8.2f}{r['r_t_nonoverlap_i']:>10.2f}"
              f"{r['r_bps_p']:>11.3f}{r['r_t_p']:>8.2f}"
              f"{int(r['n_nonoverlap_i']) if np.isfinite(r['n_nonoverlap_i']) else 0:>9}")

    # ---- 4. nulls
    print("\n-- 4. null floors from shuffled timing --")
    nulls = null_curve(bars, args.horizons, n_draws=args.null_draws,
                       measure="impact", price_kind=price_kind)
    if not nulls.empty:
        j = imp.merge(nulls, on="horizon_s", how="left")
        print(f"   {'horizon':>8}{'|R|':>10}{'null 95%':>11}{'ratio':>8}{'clears':>8}"
              f"{'null |t| 95%':>14}")
        for _, r in j.iterrows():
            ratio = abs(r["r_bps"]) / r["r_abs_95_bps"] if r["r_abs_95_bps"] else np.nan
            print(f"   {fmt_horizon(r['horizon_s']):>8}{abs(r['r_bps']):>10.3f}"
                  f"{r['r_abs_95_bps']:>11.3f}{ratio:>8.2f}"
                  f"{'yes' if ratio > 1 else 'NO':>8}{r['t_abs_95']:>14.2f}")
        print("   (a nominal |t| of 1.96 is not the bar; the null column is)")

    # ---- 5. persistence
    print("\n-- 5. signal persistence and independent observations --")
    per = persistence_curve(bars, args.horizons)
    print(f"   {'horizon':>8}{'ac1':>9}{'half-life':>12}{'integ. time':>13}"
          f"{'indep/yr':>12}")
    for _, r in per.iterrows():
        print(f"   {fmt_horizon(r['horizon_s']):>8}{r['ac1']:>9.3f}"
              f"{fmt_horizon(r['half_life_s']) if np.isfinite(r['half_life_s']) else 'n/a':>12}"
              f"{fmt_horizon(r['integrated_time_s']):>13}"
              f"{r['independent_obs_per_year']:>12,.0f}")

    # ---- 6. economics
    print("\n-- 6. economics --")
    costs = CostModel(taker_bps=args.taker_bps, spread_bps=args.spread_bps,
                      slippage_bps=args.slippage_bps)
    econ = economics(pre.rename(columns={}), per, costs, capture=args.capture)
    print(f"   round trip: taker {costs.round_trip_taker:.1f} bps, "
          f"maker {costs.round_trip_maker:.1f} bps; capture {args.capture:.0%}")
    print(f"   {'horizon':>8}{'gross bps':>11}{'net taker':>11}{'pays?':>7}"
          f"{'Sharpe':>9}{'naive Sh':>10}{'inflation':>11}")
    for _, r in econ.iterrows():
        print(f"   {fmt_horizon(r['horizon_s']):>8}{r['gross_bps']:>11.3f}"
              f"{r['net_taker_bps']:>11.3f}{'yes' if r['pays_taker'] else 'no':>7}"
              f"{r['sharpe_taker']:>9.2f}{r['sharpe_naive_rebalance']:>10.2f}"
              f"{r['sharpe_inflation_factor']:>11.1f}x")
    cx = crossover(econ)
    print(f"\n   crossover: "
          + ("never pays at any measured horizon" if cx["never_pays"]
             else f"first pays at {fmt_horizon(cx['crossover_horizon_s'])}, "
                  f"best net {cx['best_bps']:.2f} bps at "
                  f"{fmt_horizon(cx['best_bps_horizon_s'])}"))
    print("   'naive Sh' counts one bet per rebalance; 'Sharpe' counts one per")
    print("   correlation time. The inflation column is the size of that error.")

    # ---- propagator, for the record
    pf = fit_propagator(bars, args.horizons[:6], price_kind=price_kind)
    if pf.converged:
        print(f"\n-- propagator (TIM) --\n   G0={pf.g0_bps:.3f}bps l0={pf.l0_bars:.1f}bars "
              f"gamma={pf.gamma:.3f}  flow-autocorr exponent beta={pf.flow_ac_exponent:.3f}")
        cg = critical_gamma(pf.flow_ac_exponent)
        print(f"   critical gamma = (1-beta)/2 = {cg:.3f}; fitted gamma is "
              f"{'above' if pf.gamma > cg else 'below'} it "
              f"({'impact decays faster than correlated flow arrives' if pf.gamma > cg else 'impact accumulates'})")

    return {"symbol": symbol, "verdict": v.as_dict(),
            "effective_half_spread_bps": eff,
            "lambda": lam.to_dict("records"), "impact": imp.to_dict("records"),
            "predictive": pre.to_dict("records"),
            "nulls": nulls.to_dict("records"), "persistence": per.to_dict("records"),
            "economics": econ.to_dict("records"), "crossover": cx,
            "propagator": pf.as_dict(), "bounce": diag}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--symbols", nargs="+", default=["PERMUSDT", "TRANSUSDT"])
    ap.add_argument("--market", default="um")
    ap.add_argument("--bar", default="1s")
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--horizons", type=float, nargs="+", default=HORIZONS_S)
    ap.add_argument("--use-mid", action="store_true", default=True)
    ap.add_argument("--no-mid", dest="use_mid", action="store_false")
    ap.add_argument("--boot", type=int, default=60)
    ap.add_argument("--null-draws", type=int, default=80)
    ap.add_argument("--taker-bps", type=float, default=5.0)
    ap.add_argument("--spread-bps", type=float, default=0.5)
    ap.add_argument("--slippage-bps", type=float, default=0.5)
    ap.add_argument("--capture", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    out, real_any = [], False
    for sym in args.symbols:
        d, real = load(args.data_dir, sym, args.market, args.max_days)
        real_any = real_any or real
        try:
            out.append(analyse(sym, d, args))
        except Exception as exc:
            print(f"   {sym}: analysis failed: {exc}")

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    for r in out:
        if "verdict" in r:
            print(f"  {r['symbol']:<12} slope {r['verdict']['slope_log']:+.3f}   "
                  f"{r['verdict']['verdict']}")
    if not real_any:
        print(f"\n{BANNER}\n!! FIXTURE TAPES ONLY. The verdicts above are the planted\n"
              f"!! answers being recovered, which validates the estimator and\n"
              f"!! says nothing about any real venue.\n{BANNER}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"real_data": real_any, "config": vars(args), "results": out},
                      fh, indent=1, default=str)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
