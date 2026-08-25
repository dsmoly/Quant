"""Feature evaluation under purged walk-forward, with costs and controls.

Run on real data as soon as data/ has files::

    python3 -m experiments.walk_forward --data-dir data

With no data it falls back to a synthetic panel and says so, loudly and in every
section of the output. That fallback is a **pipeline test, not a result**: the
generator contains no institutional order-working, so the footprint features have
nothing to detect there and their measured IC should be indistinguishable from
zero. That makes the synthetic run a useful null -- a footprint feature scoring
well on a panel with no footprints in it would mean the feature is picking up a
mechanical artifact of its own construction, which is worth catching before real
data arrives -- but it says nothing whatever about whether these features predict
US equity returns.

What is reported per feature:

  IC by horizon      mean rank IC at 2, 5 and 10 days, in-sample and out-of-sample,
                     with a Newey-West t-statistic because overlapping forward
                     windows make the naive one far too generous.
  breakeven cost     the one-way cost that exactly consumes the gross edge.
  net Sharpe         at 5, 20 and 60 bps per side, daily and weekly rebalanced.
  controls           flat, buy-and-hold, inverted, shuffled.

Out-of-sample means: the IC used to scale positions inside a test fold is
estimated on that fold's *training* window only, and training samples whose
forward windows reach into the test block are purged.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import features as F
from src.backtest import (BacktestConfig, run_backtest, run_quintile_backtest,
                          tradable_forward_returns)
from src.breadth import breadth_report, fundamental_law_ir
from src.controls import (buy_and_hold_returns, flat_signal, invert_signal,
                          shuffle_signal_timing)
from src.loader import DataError, load_panel
from src.metrics import (breakeven_cost_bps, cross_sectional_ic, ic_newey_west_t,
                         summarise_ic, summarise_performance)
from src.sizing import SizingEngine, SizingParams
from src.synthetic import SyntheticSpec, simulate
from src.tailrisk import tail_report
from src.universe import SURVIVORSHIP_NOTE, liquid_universe
from src.validation import assert_no_leakage, fold_report, purged_walk_forward

BANNER = "!" * 78
MIN_NAMES = 5


def load_or_simulate(data_dir, n_names, n_days, seed, min_coverage=0.6):
    """Real files if present, synthetic otherwise -- and say which, unmistakably."""
    if data_dir:
        try:
            panel = load_panel(data_dir, min_coverage=min_coverage)
            print(f"REAL DATA: {len(panel.tickers)} tickers, "
                  f"{panel.dates.min().date()} to {panel.dates.max().date()}, "
                  f"{len(panel.dates)} sessions")
            if getattr(panel, "skipped", None):
                print(f"  skipped {len(panel.skipped)} file(s): {panel.skipped[:3]}")
            return panel, True
        except DataError as exc:
            print(f"{BANNER}\n!! no usable data in {data_dir}: {exc}\n"
                  f"!! FALLING BACK TO SYNTHETIC -- nothing below is a result about\n"
                  f"!! real equities. It is a test that the pipeline runs.\n{BANNER}")
    else:
        print(f"{BANNER}\n!! SYNTHETIC PANEL. The generator contains no institutional\n"
              f"!! order-working, so the footprint features have nothing to detect.\n"
              f"!! Near-zero IC below is the CORRECT answer, and a large IC would\n"
              f"!! indicate a construction artifact. Not a result about equities.\n{BANNER}")
    sim = simulate(SyntheticSpec(seed=seed, n_names=n_names, n_days=n_days))
    return sim.panel, False


def feature_frames(panel) -> dict:
    feats = F.compute_features(panel)
    feats["footprint_score"] = F.footprint_score(panel, features=feats)
    return feats


def ic_table(signal, close, horizons, lag) -> dict:
    out = {}
    for h in horizons:
        fwd = tradable_forward_returns(close, h, lag)
        ic = cross_sectional_ic(signal, fwd, min_names=MIN_NAMES)
        s = summarise_ic(ic)
        out[h] = {"mean_ic": s.mean_ic, "t_naive": s.t_stat,
                  "t_nw": ic_newey_west_t(ic, lags=h + lag),
                  "hit_rate": s.hit_rate, "n_days": s.n_days}
    return out


def oos_ic(signal, close, folds, horizon, lag) -> tuple[float, float, int]:
    """Mean IC over test blocks only, pooled across folds."""
    fwd = tradable_forward_returns(close, horizon, lag)
    ic = cross_sectional_ic(signal, fwd, min_names=MIN_NAMES)
    pieces = []
    for f in folds:
        pieces.append(ic.reindex(f.test_dates).dropna())
    if not pieces:
        return float("nan"), float("nan"), 0
    pooled = pd.concat(pieces)
    return (float(pooled.mean()), ic_newey_west_t(pooled, lags=horizon + lag),
            int(len(pooled)))


def walk_forward_backtest(signal, close, universe, folds, cfg, engine_params,
                          ic_from=None):
    """Run the strategy inside each test block using train-only IC estimates.

    ``ic_from`` supplies the series the IC is estimated from, when that differs
    from the signal being traded. This is not a nicety -- it is what makes the
    inverted control meaningful. A strategy that estimates the sign of its own
    edge is **invariant to a sign flip of its signal**: inverting gives
    ``alpha = (-IC) * sigma * (-z) = IC * sigma * z``, exactly the original. The
    first run of this experiment duly printed identical rows for the live signal
    and its inversion, which is correct behaviour and a useless control. To
    invert meaningfully, the IC must be held at the original signal's value so
    the strategy is forced to trade the flipped signal with the unflipped
    conviction.
    """
    fwd = tradable_forward_returns(close, cfg.horizon, cfg.lag)
    full_ic = cross_sectional_ic(signal if ic_from is None else ic_from, fwd)
    nets, grosses, turns = [], [], []
    for f in folds:
        train_ic = full_ic.reindex(f.train_dates).dropna()
        if len(train_ic) < cfg.min_ic_obs:
            continue
        # Constant IC over the test block, estimated only on training dates.
        res = run_backtest(signal=signal, close=close, config=cfg, universe=universe,
                           engine=SizingEngine(engine_params),
                           ic_series=train_ic)
        nets.append(res.net_returns.reindex(f.test_dates))
        grosses.append(res.gross_returns.reindex(f.test_dates))
        turns.append(res.turnover.reindex(f.test_dates))
    if not nets:
        return None
    return (pd.concat(nets).sort_index(), pd.concat(grosses).sort_index(),
            pd.concat(turns).sort_index())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--names", type=int, default=25)
    ap.add_argument("--days", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--universe-size", type=int, default=20)
    ap.add_argument("--min-coverage", type=float, default=0.6,
                    help="drop tickers present for less than this share of sessions")
    ap.add_argument("--min-names", type=int, default=5,
                    help="minimum cross-section width to score a date")
    ap.add_argument("--horizons", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--primary-horizon", type=int, default=5)
    ap.add_argument("--costs-bps", type=float, nargs="+", default=[5.0, 20.0, 60.0])
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    global MIN_NAMES
    MIN_NAMES = args.min_names
    panel, is_real = load_or_simulate(args.data_dir, args.names, args.days, args.seed,
                                     min_coverage=args.min_coverage)
    close = panel.close
    lag = 1

    print(f"\n{'-'*78}\nSURVIVORSHIP\n{'-'*78}")
    print(SURVIVORSHIP_NOTE.strip())

    universe = liquid_universe(panel, n=args.universe_size, window=60)
    feats = feature_frames(panel)

    folds = purged_walk_forward(close.index, n_splits=args.splits,
                                horizon=args.primary_horizon, lag=lag,
                                min_train=args.min_train)
    assert_no_leakage(folds, horizon=args.primary_horizon, lag=lag)
    print(f"\n{'-'*78}\nWALK-FORWARD FOLDS (purged, embargoed)\n{'-'*78}")
    print(fold_report(folds).to_string(index=False))

    # ------------------------------------------------------------ IC by horizon
    print(f"\n{'-'*78}\nINFORMATION COEFFICIENT BY HORIZON\n{'-'*78}")
    print("t_nw is Newey-West; the naive t is shown alongside to make the size of")
    print("the overlap correction visible. Trust t_nw.")
    hdr = f"{'feature':<24}{'h':>4}{'IS mean IC':>12}{'t_naive':>9}{'t_nw':>8}{'OOS IC':>9}{'OOS t_nw':>10}"
    print(hdr); print("-" * len(hdr))
    ic_results = {}
    for name, frame in feats.items():
        ic_results[name] = {}
        table = ic_table(frame, close, args.horizons, lag)
        for h in args.horizons:
            o_ic, o_t, o_n = oos_ic(frame, close, folds, h, lag)
            row = {**table[h], "oos_ic": o_ic, "oos_t_nw": o_t, "oos_n": o_n}
            ic_results[name][h] = row
            print(f"{name if h == args.horizons[0] else '':<24}{h:>4}"
                  f"{row['mean_ic']:>12.4f}{row['t_naive']:>9.2f}{row['t_nw']:>8.2f}"
                  f"{o_ic:>9.4f}{o_t:>10.2f}")

    # ---------------------------------------------------------------- backtests
    print(f"\n{'-'*78}\nWALK-FORWARD BACKTEST (out-of-sample blocks only)\n{'-'*78}")
    sizing = SizingParams()
    results = {}
    hdr = (f"{'arm':<30}{'cost':>6}{'reb':>5}{'netSh':>8}{'grossSh':>9}"
           f"{'turn':>7}{'be(bps)':>9}{'maxDD':>8}")
    print(hdr); print("-" * len(hdr))

    arms = {name: frame for name, frame in feats.items()}
    arms["CONTROL flat"] = flat_signal(feats["footprint_score"])
    arms["CONTROL inverted"] = invert_signal(feats["footprint_score"])
    arms["CONTROL shuffled"] = shuffle_signal_timing(feats["footprint_score"], seed=5)

    for name, frame in arms.items():
        # The inverted control must keep the *original* signal's IC, or the
        # strategy simply re-learns the flipped sign and the control is a no-op.
        ic_from = feats["footprint_score"] if name == "CONTROL inverted" else None
        for reb in ("D", "W"):
            for cost in args.costs_bps:
                cfg = BacktestConfig(cost_bps=cost, rebalance=reb, lag=lag,
                                     horizon=args.primary_horizon, sizing=sizing,
                                     min_ic_obs=60)
                out = walk_forward_backtest(frame, close, universe, folds, cfg, sizing,
                                            ic_from=ic_from)
                if out is None:
                    continue
                net, gross, turn = out
                p = summarise_performance(net, turnover=turn, gross_returns=gross)
                g = summarise_performance(gross)
                key = f"{name}|{reb}|{cost}"
                results[key] = {"net_sharpe": p.sharpe, "gross_sharpe": g.sharpe,
                                "turnover": p.mean_turnover,
                                "breakeven_bps": breakeven_cost_bps(gross, turn),
                                "max_dd": p.max_drawdown, "ann_return": p.ann_return}
                print(f"{name:<30}{cost:>6.0f}{reb:>5}{p.sharpe:>8.2f}"
                      f"{g.sharpe:>9.2f}{p.mean_turnover:>7.3f}"
                      f"{results[key]['breakeven_bps']:>9.1f}{p.max_drawdown:>8.2f}")

    # naive quintile construction, for contrast with the engine
    print(f"\n{'-'*78}\nNAIVE QUINTILE CONSTRUCTION (same signal, no sizing engine)\n{'-'*78}")
    for cost in args.costs_bps:
        cfg = BacktestConfig(cost_bps=cost, rebalance="D", lag=lag,
                             horizon=args.primary_horizon)
        res = run_quintile_backtest(signal=feats["footprint_score"], close=close,
                                    config=cfg, universe=universe)
        test_dates = pd.DatetimeIndex(np.concatenate([f.test_dates for f in folds]))
        p = summarise_performance(res.net_returns.reindex(test_dates),
                                  turnover=res.turnover.reindex(test_dates),
                                  gross_returns=res.gross_returns.reindex(test_dates))
        print(f"  cost {cost:>5.0f}bps   net Sharpe {p.sharpe:>6.2f}   "
              f"turnover {p.mean_turnover:.3f}   breakeven {p.breakeven_cost_bps:.1f}bps")

    # buy and hold
    print(f"\n{'-'*78}\nBUY AND HOLD (survivorship-inflated benchmark)\n{'-'*78}")
    for cost in args.costs_bps[:1]:
        bh = buy_and_hold_returns(universe, close, lag=lag, cost_bps=cost)
        test_dates = pd.DatetimeIndex(np.concatenate([f.test_dates for f in folds]))
        p = summarise_performance(bh["net_returns"].reindex(test_dates),
                                  turnover=bh["turnover"].reindex(test_dates))
        print(f"  cost {cost:>5.0f}bps   Sharpe {p.sharpe:>6.2f}   "
              f"ann return {p.ann_return:>7.3f}   maxDD {p.max_drawdown:>6.2f}")

    # ------------------------------------------------------------------ breadth
    print(f"\n{'-'*78}\nBREADTH AND TAIL RISK\n{'-'*78}")
    rep = breadth_report(close.pct_change().dropna(how="all"))
    print(f"  names {rep['n_names']}, mean pairwise correlation {rep['mean_pairwise_corr']:.3f}")
    print(f"  PC1 explains {rep['pc1_share']*100:.1f}% of variance")
    print(f"  effective N: raw {rep['n_eff_raw']:.1f}   residual (PC1 removed) "
          f"{rep['n_eff_residual']:.1f}   Marchenko-Pastur factor count {rep['n_eff_mp']}")
    ref_ic = ic_results["footprint_score"][args.primary_horizon]["oos_ic"]
    if np.isfinite(ref_ic):
        print(f"  fundamental law at OOS IC={ref_ic:.4f}:")
        print(f"    naive  IR = {fundamental_law_ir(ref_ic, rep['n_names']):.2f} "
              f"(using raw N -- overstates)")
        print(f"    raw    IR = {fundamental_law_ir(ref_ic, rep['n_eff_raw']):.2f} "
              f"(raw effective N -- right for a net-exposed book)")
        print(f"    resid  IR = {fundamental_law_ir(ref_ic, rep['n_eff_residual']):.2f} "
              f"(residual effective N -- right for a dollar-neutral book)")

    best = max(results.items(),
               key=lambda kv: kv[1]["net_sharpe"] if np.isfinite(kv[1]["net_sharpe"]) else -9)
    print(f"\n  tail risk of the best arm ({best[0]}):")
    cfg = BacktestConfig(cost_bps=args.costs_bps[1] if len(args.costs_bps) > 1 else 20.0,
                         horizon=args.primary_horizon)
    out = walk_forward_backtest(feats["footprint_score"], close, universe, folds,
                                cfg, sizing)
    if out is not None:
        tr = tail_report(out[0].dropna())
        print(f"    expected shortfall (5%) {tr['expected_shortfall']:.4f} "
              f"+/- {tr['expected_shortfall_se']:.4f}")
        print(f"    downside vol {tr['downside_vol']:.4f}, downside ratio "
              f"{tr['downside_ratio']:.3f} (0.707 = symmetric)")
        print(f"    excess kurtosis {tr['excess_kurtosis']:.2f}")
        print(f"    Hill tail index in [{tr['hill_range'][0]:.2f}, "
              f"{tr['hill_range'][1]:.2f}] over k in {tr['hill_k_range']} "
              f"-- a range, not a point")

    if not is_real:
        print(f"\n{BANNER}\n!! REMINDER: synthetic panel. Every IC and Sharpe above is a\n"
              f"!! property of the generator, not of US equities.\n{BANNER}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"real_data": is_real, "config": vars(args),
                       "ic": {k: {str(h): v for h, v in d.items()}
                              for k, d in ic_results.items()},
                       "backtests": results, "breadth": rep}, fh, indent=1, default=float)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
