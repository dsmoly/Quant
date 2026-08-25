"""Do entropy-based tail estimates beat Hill and the GPD fit?

Three estimators of the tail index, on data whose answer is known exactly
(symmetric alpha-stable, where the stability parameter *is* the tail index):

  entropy   Pareto-kernel differential entropy, alpha fitted by leave-one-out
            maximum-likelihood cross-validation. No threshold choice.
  hill      classical Hill, reported as the midpoint of the plateau region --
            already known from earlier work to be biased even on 20,000 exact
            Pareto draws.
  gpd       generalised Pareto fitted above a selected threshold; tail index 1/xi.

Two sample sizes, and the small one is the point: n=61 is the rolling window the
paper actually uses, so an estimator that only works at n=1000 cannot support the
regime labels it is used for.

Also run, because they are the controls that matter:

  gaussian null    how often is Gaussian data labelled heavy-tailed?
  AR(1) Gaussian   does mere autocorrelation trigger false heavy-tail
                   detection? Thin-tailed but serially dependent data is the
                   obvious false-positive mode for any windowed estimator, and
                   financial returns are serially dependent in volatility.
  look-ahead       the paper's centred window against a strictly trailing one.
                   The gap is the size of the advantage the published figures
                   get from seeing the future.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

from src.entropy import (fit_tail_index, rolling_tail_index,
                         rvs_symmetric_stable)
from src.tailrisk import fit_gpd, hill_plot, select_threshold


def est_entropy(x):
    return fit_tail_index(x).alpha


def est_hill(x):
    hp = hill_plot(pd.Series(-np.abs(x)), k_min=max(5, x.size // 20))
    if not np.isfinite(hp.plateau_lo):
        return float("nan")
    return 0.5 * (hp.plateau_lo + hp.plateau_hi)


def est_gpd(x):
    s = pd.Series(-np.abs(x))
    t = select_threshold(s, min_exceed=max(15, x.size // 8))
    rec = t[t.recommended]
    if rec.empty:
        return float("nan")
    f = fit_gpd(s, float(rec.iloc[0]["threshold"]))
    return f.tail_index if f.converged else float("nan")


ESTIMATORS = {"entropy": est_entropy, "hill": est_hill, "gpd": est_gpd}


def recovery(alphas, n, reps, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for a in alphas:
        acc = {k: [] for k in ESTIMATORS}
        for _ in range(reps):
            x = rvs_symmetric_stable(a, n, rng=rng)
            for k, f in ESTIMATORS.items():
                try:
                    acc[k].append(float(f(x)))
                except Exception:
                    acc[k].append(float("nan"))
        row = {"true_alpha": a, "n": n}
        for k, v in acc.items():
            v = np.array(v, dtype=float)
            ok = np.isfinite(v)
            row[f"{k}_med"] = float(np.median(v[ok])) if ok.any() else float("nan")
            row[f"{k}_bias"] = row[f"{k}_med"] - a
            row[f"{k}_iqr"] = (float(np.subtract(*np.percentile(v[ok], [75, 25])))
                               if ok.sum() > 3 else float("nan"))
            row[f"{k}_fail"] = float(1.0 - ok.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def discrimination(n, reps, seed=1):
    """Can the estimator tell alpha=1.5 from alpha=2.0 at this sample size?

    This is the question regime detection actually rests on, and it is stricter
    than bias: an estimator can be badly biased and still separate two regimes,
    or nearly unbiased and still useless because its spread swamps the gap.
    Reported as the AUC of a one-sided comparison -- 0.5 is a coin flip.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for k, f in ESTIMATORS.items():
        heavy, light = [], []
        for _ in range(reps):
            for lst, a in ((heavy, 1.5), (light, 2.0)):
                try:
                    lst.append(float(f(rvs_symmetric_stable(a, n, rng=rng))))
                except Exception:
                    lst.append(float("nan"))
        h = np.array(heavy)[np.isfinite(heavy)]
        l = np.array(light)[np.isfinite(light)]
        if h.size < 5 or l.size < 5:
            out[k] = float("nan")
            continue
        # P(heavy sample scores lower than light sample)
        wins = (h[:, None] < l[None, :]).mean() + 0.5 * (h[:, None] == l[None, :]).mean()
        out[k] = float(wins)
    return out


def false_positives(n, reps, seed=2, ar=0.0):
    """Rate at which thin-tailed data is labelled heavy (alpha_hat < 2)."""
    rng = np.random.default_rng(seed)
    flags = {k: [] for k in ESTIMATORS}
    for _ in range(reps):
        e = rng.normal(size=n)
        if ar:
            x = np.empty(n)
            x[0] = e[0]
            for i in range(1, n):
                x[i] = ar * x[i - 1] + np.sqrt(1 - ar ** 2) * e[i]
        else:
            x = e
        for k, f in ESTIMATORS.items():
            try:
                v = float(f(x))
            except Exception:
                v = float("nan")
            flags[k].append(bool(np.isfinite(v) and v < 2.0))
    return {k: float(np.mean(v)) for k, v in flags.items()}


def lookahead(n=1200, window=61, seed=3):
    """How much does the paper's centred window gain from seeing the future?

    A regime series is only useful if it is available before the fact. The test
    is whether the trailing label anticipates the next window's realised
    volatility as well as the centred one does.
    """
    rng = np.random.default_rng(seed)
    # A tape with genuine regime structure: quiet, then heavy-tailed, then quiet.
    seg = n // 3
    x = np.concatenate([rng.normal(0, 1, seg),
                        rvs_symmetric_stable(1.3, seg, scale=0.7, rng=rng),
                        rng.normal(0, 1, n - 2 * seg)])
    s = pd.Series(x)
    out = {}
    for mode in (False, True):
        r = rolling_tail_index(s, window=window, step=5, centred=mode)
        if r.empty:
            continue
        fwd = s.rolling(window).std().shift(-window).reindex(r.index)
        d = pd.concat([r["alpha"], fwd], axis=1).dropna()
        d.columns = ["alpha", "fwd_vol"]
        out["centred" if mode else "trailing"] = {
            "n": int(len(d)),
            "corr_with_forward_vol": float(d["alpha"].corr(d["fwd_vol"], method="spearman")),
            "mean_alpha_in_heavy_segment": float(
                r.loc[(r.index >= seg) & (r.index < 2 * seg), "alpha"].mean()),
            "mean_alpha_in_quiet_segments": float(
                r.loc[(r.index < seg) | (r.index >= 2 * seg), "alpha"].mean()),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--sizes", type=int, nargs="+", default=[61, 500])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    alphas = [0.8, 1.2, 1.5, 1.8, 2.0]
    res = {"recovery": {}, "discrimination": {}, "false_positives": {}}

    for n in args.sizes:
        t0 = time.time()
        df = recovery(alphas, n, args.reps)
        res["recovery"][n] = df.to_dict("records")
        print(f"\n{'='*74}\nRECOVERY at n={n}  ({args.reps} reps, {time.time()-t0:.0f}s)\n{'='*74}")
        print(f"{'true':>6} | {'entropy':>18} | {'hill':>18} | {'gpd':>18}")
        print(f"{'':>6} | {'med':>7}{'bias':>6}{'fail':>5} | "
              f"{'med':>7}{'bias':>6}{'fail':>5} | {'med':>7}{'bias':>6}{'fail':>5}")
        print("-" * 74)
        for _, r in df.iterrows():
            line = f"{r['true_alpha']:>6.1f} |"
            for k in ("entropy", "hill", "gpd"):
                line += (f" {r[f'{k}_med']:>7.2f}{r[f'{k}_bias']:>+6.2f}"
                         f"{r[f'{k}_fail']*100:>4.0f}% |")
            print(line)

        d = discrimination(n, args.reps)
        res["discrimination"][n] = d
        print(f"\n  discriminating alpha=1.5 from alpha=2.0 (AUC, 0.5 = coin flip):")
        for k, v in d.items():
            verdict = ("useless" if not np.isfinite(v) or v < 0.65 else
                       "weak" if v < 0.8 else "usable")
            print(f"    {k:<9} {v:.3f}   {verdict}")

        fp = false_positives(n, args.reps)
        fpar = false_positives(n, args.reps, ar=0.7)
        res["false_positives"][n] = {"iid": fp, "ar07": fpar}
        print(f"\n  false 'heavy-tailed' rate on THIN-tailed data:")
        print(f"    {'':<9}{'iid Gaussian':>14}{'AR(1) rho=0.7':>16}")
        for k in ESTIMATORS:
            print(f"    {k:<9}{fp[k]*100:>13.0f}%{fpar[k]*100:>15.0f}%")

    print(f"\n{'='*74}\nLOOK-AHEAD: centred window vs trailing\n{'='*74}")
    la = lookahead()
    res["lookahead"] = la
    for mode, v in la.items():
        print(f"  {mode:<9} corr(alpha, forward vol) = {v['corr_with_forward_vol']:+.3f}"
              f"   alpha heavy-seg {v['mean_alpha_in_heavy_segment']:.2f}"
              f" vs quiet {v['mean_alpha_in_quiet_segments']:.2f}")
    if len(la) == 2:
        gap = (la["centred"]["mean_alpha_in_quiet_segments"]
               - la["centred"]["mean_alpha_in_heavy_segment"])
        tgap = (la["trailing"]["mean_alpha_in_quiet_segments"]
                - la["trailing"]["mean_alpha_in_heavy_segment"])
        print(f"\n  regime separation (quiet alpha - heavy alpha):")
        print(f"    centred  {gap:+.3f}   <- uses 30 days of future data")
        print(f"    trailing {tgap:+.3f}   <- what a live system could have")
        if gap != 0:
            print(f"    trailing retains {100*tgap/gap:.0f}% of the separation")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
