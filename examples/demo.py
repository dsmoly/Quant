"""End-to-end demo on synthetic data with a known answer.

Runs the full protocol on three signals whose truth we control:

  * a genuinely predictive one,
  * pure noise,
  * a "new" signal that is really a known factor in disguise.

The point is not the numbers -- they are synthetic -- but that the harness
reaches the right verdict on all three, including the third, which is the case
that fools people in practice.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qr
from qr.sizing import SizingEngine, SizingParams


def build(T=1500, N=200, seed=0):
    rng = np.random.default_rng(seed)
    # one common factor plus idiosyncratic noise
    beta = rng.uniform(0.6, 1.4, N)
    mkt = rng.normal(0, 0.01, T)
    idio_vol = rng.uniform(0.008, 0.030, N)

    # a persistent, decaying true signal
    phi = np.exp(-0.10)
    truth = np.zeros((T, N))
    truth[0] = rng.normal(0, 1, N)
    for t in range(1, T):
        truth[t] = phi * truth[t - 1] + np.sqrt(1 - phi ** 2) * rng.normal(0, 1, N)

    value = rng.normal(0, 1, (T, N))       # a "known factor" we already own
    r = np.zeros((T, N))
    # calibrated so the planted signals sit at realistic strength: the true
    # signal lands near IC 0.05 and the factor near 0.035. A demo with IC 0.3
    # would make every downstream statistic look absurd and teach nothing.
    r[1:] = (np.outer(mkt[1:], beta)
             + 0.056 * truth[:-1] * idio_vol
             + 0.040 * value[:-1] * idio_vol
             + rng.normal(0, 1, (T - 1, N)) * idio_vol)
    r[0] = rng.normal(0, 1, N) * idio_vol
    return r, truth, value, mkt, beta, idio_vol


def main():
    r, truth, value, mkt, beta, idio_vol = build()
    T, N = r.shape
    market_factor = np.tile(mkt[:, None], (1, N)) * beta[None, :]
    factors = [market_factor, value]

    log_path = os.path.join(tempfile.mkdtemp(), "research_log.jsonl")
    log = qr.TrialLog(log_path)
    # simulate an honest research history: 40 variants were tried before these
    for i in range(40):
        log.record("exploratory sweep", {"variant": i}, {"sharpe": 0.02}, family="demo")

    rng = np.random.default_rng(99)
    candidates = {
        "real_signal": truth,
        "pure_noise": rng.normal(0, 1, (T, N)),
        "value_in_disguise": 2.0 * value + rng.normal(0, 0.4, (T, N)),
    }

    for name, sig in candidates.items():
        rep = qr.evaluate_signal(
            sig, r, name=name, horizon=1, factors=factors,
            factor_names=["market", "value"], trial_log=log, family="demo",
            hypothesis=f"{name} predicts next-day cross-sectional returns",
            cost_per_unit=0.0005)
        print(rep)
        print()

    print(log.summary())
    print()

    # --- sizing the surviving signal -------------------------------------
    print("=== sizing, market-neutral vs directional ===")
    cov = np.cov(r[:500], rowvar=False)
    sigma = np.sqrt(np.diag(cov))
    cond = qr.rank_normal(truth[600])   # 1-D cross-section is fine
    alpha = 0.05 * sigma * cond
    se = np.abs(alpha) / 2.5

    for label, params in (
        ("market-neutral", SizingParams(neutral=True, residual_factors=1, beta=0.10)),
        ("directional   ", SizingParams(neutral=False, residual_factors=0, beta=0.10)),
    ):
        w, d = SizingEngine(params).size(
            alpha=alpha, sigma=sigma, cov=cov, current=np.zeros(N),
            cost=np.full(N, 0.0003), alpha_se=se)
        print(f"  {label}: gross {d.gross:.3f}  net {d.net:+.4f}  "
              f"vol {d.predicted_vol_annual:.2%} (target after haircuts "
              f"{d.target_vol_effective:.2%})  n_eff {d.n_eff:.1f}  "
              f"crossed band {d.n_crossed_band}/{d.n_active}")

    print("\n=== the rate control converges; it does not jump ===")
    print("   nu* = -rho (q - q*) closes a fraction rho of the gap per day, so a")
    print("   cold start takes ~1/rho days to reach the risk target. That is the")
    print("   control as derived -- jumping straight to target is a different")
    print("   problem's answer and pays the quadratic impact cost it was avoiding.")
    for label, adj in (("rate  ", "rate"), ("target", "target")):
        eng = SizingEngine(SizingParams(neutral=True, residual_factors=1,
                                        beta=0.10, adjustment=adj))
        w = np.zeros(N)
        cum_turnover = 0.0
        row = []
        for day in range(12):
            w, d = eng.size(alpha=alpha, sigma=sigma, cov=cov, current=w,
                            cost=np.full(N, 0.0003), alpha_se=se)
            cum_turnover += d.turnover
            row.append(d.predicted_vol_annual)
        print(f"   {label}: vol by day " +
              " ".join(f"{v:.1%}" for v in row[:8]) +
              f"  | cumulative turnover {cum_turnover:.2f}")

    print("\n=== decay measured from the panel, feeding SizingParams.beta ===")
    hs, ic, se_, t_ = qr.marginal_ic_curve(qr.rank_normal(truth), r,
                                           horizons=(1, 2, 3, 5, 8, 12, 20))
    fit = qr.fit_decay(hs, ic, se_, t_)
    print(f"  {fit}")
    print(f"  (data-generating process used beta=0.10)")


if __name__ == "__main__":
    main()
