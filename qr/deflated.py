"""How much of this Sharpe is the search itself?

Bailey & Lopez de Prado (2014). The logic in one line: if you try N strategies
on noise, the best one has an expected Sharpe well above zero, and that
expectation is computable. The Deflated Sharpe Ratio asks whether your observed
Sharpe beats *that* benchmark, not zero.

The simulation that motivates this module: 200 pure-noise signals tested against
pure-noise returns produce a best |t| above 3 about thirty percent of the time,
and at 1000 trials it is a certainty. Your effective trial count includes every
variant you tried and quietly discarded, which is why `qr.trials` exists -- the
number you pass in here is only honest if something has been counting.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(n_trials: int, sr_std: float = 1.0) -> float:
    """Expected maximum per-period Sharpe across ``n_trials`` independent trials.

    ``sr_std`` is the cross-sectional standard deviation of the trial Sharpes.
    If you have the actual distribution of what you tried, use its std; the
    default of 1.0 expresses the benchmark in units of that spread.
    """
    n = int(n_trials)
    if n < 2:
        return 0.0
    g = EULER_MASCHERONI
    a = stats.norm.ppf(1.0 - 1.0 / n)
    b = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(sr_std * ((1.0 - g) * a + g * b))


def _moments(returns):
    v = np.asarray(returns, dtype=float)
    v = v[np.isfinite(v)]
    s = v.std(ddof=1)
    sr = float(v.mean() / s) if s > 0 else 0.0
    skew = float(stats.skew(v, bias=False)) if v.size > 2 else 0.0
    kurt = float(stats.kurtosis(v, fisher=False, bias=False)) if v.size > 3 else 3.0
    return v, sr, skew, kurt


def deflated_sharpe(returns=None, n_trials: int = 1, sr_std: float = 1.0,
                    sharpe: float | None = None, n_obs: int | None = None,
                    skew: float = 0.0, kurtosis: float = 3.0,
                    benchmark_sr: float | None = None) -> dict:
    """Probability the true Sharpe exceeds the best-of-N-trials benchmark.

    Pass either a ``returns`` series (moments computed for you) or the triple
    ``sharpe`` / ``n_obs`` / ``skew`` / ``kurtosis`` directly. All Sharpes here
    are **per period**, not annualised -- annualising first and deflating after
    silently rescales the variance term and inflates the result.

    ``kurtosis`` is non-excess (3.0 for a normal).
    """
    if returns is not None:
        v, sr, sk, ku = _moments(returns)
        T = v.size
    else:
        if sharpe is None or n_obs is None:
            raise ValueError("pass either returns, or both sharpe and n_obs")
        sr, T, sk, ku = float(sharpe), int(n_obs), float(skew), float(kurtosis)

    sr0 = (expected_max_sharpe(n_trials, sr_std) if benchmark_sr is None
           else float(benchmark_sr))

    var_term = 1.0 - sk * sr + ((ku - 1.0) / 4.0) * sr ** 2
    if T < 3 or var_term <= 0:
        return {"dsr": float("nan"), "sharpe": sr, "benchmark_sr": sr0,
                "n_trials": int(n_trials), "n_obs": T, "skew": sk,
                "kurtosis": ku, "z": float("nan")}
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(var_term)
    return {"dsr": float(stats.norm.cdf(z)), "sharpe": sr, "benchmark_sr": sr0,
            "n_trials": int(n_trials), "n_obs": T, "skew": sk, "kurtosis": ku,
            "z": float(z)}


def min_track_record_length(sharpe: float, benchmark_sr: float = 0.0,
                            skew: float = 0.0, kurtosis: float = 3.0,
                            confidence: float = 0.95) -> float:
    """Observations needed for a Sharpe of ``sharpe`` to beat the benchmark.

    The honest answer to "how long do I have to run this before I know?".
    Returns inf when the observed Sharpe does not exceed the benchmark at all.
    """
    sr, sr0 = float(sharpe), float(benchmark_sr)
    if sr <= sr0:
        return float("inf")
    var_term = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr ** 2
    if var_term <= 0:
        return float("nan")
    z = stats.norm.ppf(confidence)
    return float(1.0 + var_term * (z / (sr - sr0)) ** 2)
