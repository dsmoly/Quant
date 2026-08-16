"""How large an IC can one panel produce from nothing?

This is the calibration experiment, and on reflection it is the most useful thing
in the project, because it sets the bar every other number has to clear.

The motivation was a mistake made while building this. On one synthetic panel --
generated with *no* planted signal at all -- the dollar-volume-anomaly feature
produced an out-of-sample IC of +0.0195 with a Newey-West t of 2.05, and +0.0279
with t=2.75 at a ten-day horizon. Read on its own that looks like a discovery. It
was not: across six independent generator seeds the same feature ranged from
-0.0151 to +0.0007, mostly negative, and the shuffled-timing control on the same
panel reached t=1.73. The single-panel t-statistic was badly overstated even
*after* the Newey-West correction.

So this experiment measures the null distribution directly: many independent
panels, each with zero true predictive power, each scored exactly as a real
feature would be. What comes out is the empirical critical value -- the IC a
feature must beat before it means anything at this universe size and sample
length.

Why the Newey-West correction is not enough on its own. It handles the
autocorrelation induced by overlapping forward windows, which is a within-panel
problem. It does nothing about cross-sectional dependence: on any given day the
25 names share a market factor, so the effective number of independent
observations in a cross-section is closer to the effective breadth than to the
name count. Both corrections are needed and only one is standard.

The practical conclusion is worth stating in advance of the numbers: an expected
out-of-sample IC of 0.02-0.04 -- the range this project is aiming at -- sits
close to the noise floor of a single six-year panel of twenty-odd names. That is
not an argument against the strategy. It is an argument that the sample has to be
bigger than one currently-liquid US large-cap list before any IC in that range
can be told apart from luck.
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from src import features as F
from src.backtest import tradable_forward_returns
from src.breadth import breadth_report
from src.metrics import cross_sectional_ic, ic_newey_west_t
from src.synthetic import SyntheticSpec, simulate


def one_panel(seed, n_names, n_days, horizon, lag, feature):
    """Measure one feature on one zero-signal panel."""
    sim = simulate(SyntheticSpec(seed=seed, n_names=n_names, n_days=n_days))
    close = sim.panel.close
    fwd = tradable_forward_returns(close, horizon, lag)
    if feature == "random":
        rng = np.random.default_rng(10_000 + seed)
        sig = pd.DataFrame(rng.normal(size=close.shape),
                           index=close.index, columns=close.columns)
    else:
        sig = F.FEATURES[feature](sim.panel) if feature in F.FEATURES \
            else F.footprint_score(sim.panel)
    ic = cross_sectional_ic(sig, fwd)
    if ic.empty:
        return None
    rep = breadth_report(close.pct_change().dropna(how="all"))
    return {"seed": seed, "mean_ic": float(ic.mean()),
            "t_naive": float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))),
            "t_nw": float(ic_newey_west_t(ic, lags=horizon + lag)),
            "n_days": int(len(ic)), "n_eff_raw": rep["n_eff_raw"]}


def summarise(rows, label, alpha=0.05):
    df = pd.DataFrame(rows)
    ic, t = df["mean_ic"], df["t_nw"]
    q = [alpha / 2, 0.5, 1 - alpha / 2]
    ic_lo, ic_med, ic_hi = np.quantile(ic, q)
    crit = float(np.quantile(np.abs(ic), 0.95))
    t_crit = float(np.quantile(np.abs(t), 0.95))
    false_pos = float((np.abs(t) > 1.96).mean())
    print(f"\n{label}  ({len(df)} panels)")
    print(f"  mean IC       {ic.mean():+.4f}   sd across panels {ic.std(ddof=1):.4f}")
    print(f"  95% interval  [{ic_lo:+.4f}, {ic_hi:+.4f}]   median {ic_med:+.4f}")
    print(f"  |IC| 95th pct {crit:.4f}   <- an IC below this means nothing")
    print(f"  |t_nw| 95th pct {t_crit:.2f}   (nominal critical value 1.96)")
    print(f"  false-positive rate at |t_nw|>1.96: {false_pos*100:.0f}% "
          f"(nominal 5%)")
    return {"label": label, "n_panels": int(len(df)), "mean_ic": float(ic.mean()),
            "sd_ic": float(ic.std(ddof=1)), "ic_95_lo": float(ic_lo),
            "ic_95_hi": float(ic_hi), "ic_abs_95": crit, "t_abs_95": t_crit,
            "false_positive_rate": false_pos}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panels", type=int, default=60)
    ap.add_argument("--names", type=int, default=25)
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--features", nargs="+",
                    default=["random", "dollar_volume_anomaly", "abnormal_lambda",
                             "persistent_imbalance", "footprint_score"])
    ap.add_argument("--size-scan", action="store_true",
                    help="also scan universe size and history length")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    print(__doc__.split("\n\n")[0])
    print(f"\nAll panels have ZERO planted signal. Every IC below is noise by "
          f"construction.\n{args.panels} panels x {args.days} sessions x "
          f"{args.names} names, horizon {args.horizon}d")

    out = {"config": vars(args), "features": {}, "scan": []}
    for feat in args.features:
        rows = [r for r in (one_panel(s, args.names, args.days, args.horizon, 1, feat)
                            for s in range(args.panels)) if r]
        if rows:
            out["features"][feat] = summarise(rows, f"feature: {feat}")

    if args.size_scan:
        print(f"\n{'-'*70}\nNOISE FLOOR vs UNIVERSE SIZE AND HISTORY\n{'-'*70}")
        print("The IC a zero-signal feature reaches 5% of the time. Bigger is worse:")
        print("it is the bar a real feature must clear to be believed.\n")
        print(f"{'names':>7}{'sessions':>10}{'|IC| 95th':>12}{'|t_nw| 95th':>13}")
        for n_names in (10, 25, 50):
            for n_days in (750, 1500, 3000):
                rows = [r for r in (one_panel(s, n_names, n_days, args.horizon, 1,
                                              "random")
                                    for s in range(max(20, args.panels // 3))) if r]
                if not rows:
                    continue
                df = pd.DataFrame(rows)
                crit = float(np.quantile(df["mean_ic"].abs(), 0.95))
                tcrit = float(np.quantile(df["t_nw"].abs(), 0.95))
                print(f"{n_names:>7}{n_days:>10}{crit:>12.4f}{tcrit:>13.2f}")
                out["scan"].append({"n_names": n_names, "n_days": n_days,
                                    "ic_abs_95": crit, "t_abs_95": tcrit})

    print(f"\n{'-'*70}\nWHAT THIS MEANS FOR THE TARGET\n{'-'*70}")
    rnd = out["features"].get("random")
    real = {k: v for k, v in out["features"].items() if k != "random"}
    if rnd and real:
        # The bar that matters is the one for a *persistent* feature, not for
        # white noise. Real features are autocorrelated, and the Newey-West lag
        # is chosen for the forward-window overlap rather than for the feature's
        # own persistence, so it under-corrects exactly where it is needed.
        worst = max(real.values(), key=lambda v: v["ic_abs_95"])
        floor = worst["ic_abs_95"]
        print(f"  white-noise signal:  |IC| floor {rnd['ic_abs_95']:.4f}, "
              f"|t_nw| floor {rnd['t_abs_95']:.2f}, "
              f"false positives {rnd['false_positive_rate']*100:.0f}%")
        print(f"  persistent feature:  |IC| floor {floor:.4f}, "
              f"|t_nw| floor {worst['t_abs_95']:.2f}, "
              f"false positives {worst['false_positive_rate']*100:.0f}% "
              f"({worst['label'].split(': ')[-1]})")
        print("\n  Newey-West is roughly calibrated for a white-noise signal and")
        print("  under-corrects for a persistent one: the features' own")
        print("  autocorrelation is not in the lag choice. Use the persistent-")
        print("  feature floor as the bar.\n")
        for target in (0.02, 0.03, 0.04):
            if target < floor:
                verdict = "BELOW the floor -- indistinguishable from luck"
            elif target < 1.5 * floor:
                verdict = "MARGINAL -- within 1.5x of the floor, treat with suspicion"
            else:
                verdict = "clears the floor on this sample size"
            print(f"  an OOS IC of {target:.2f}: {verdict}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
