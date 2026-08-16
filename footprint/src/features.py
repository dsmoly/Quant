"""Footprint features: what an institutional order leaves behind in daily data.

The premise is that a fund working a large order over days cannot hide it in the
daily record, even though it can hide each individual child order in the book.
Three observable consequences, and one control:

  Amihud illiquidity      |return| per dollar traded, averaged over a trailing
                          window. This is a daily-frequency proxy for Kyle's
                          lambda -- the price impact per unit of order flow. It
                          is a *level*: how hard this name is to move, which is
                          mostly a fact about the name rather than about today.

  abnormal lambda         today's impact per unit volume against its own trailing
                          median. This is the actual footprint signal, because it
                          differences out the level. Reading the sign takes care:
                          impact *below* normal means unusual volume was absorbed
                          without moving the price, which is what a patient
                          liquidity *supplier* looks like; impact *above* normal
                          means the price moved more than the volume justifies,
                          which is a liquidity *demander* paying up to get done.

  persistent imbalance    consecutive same-sign sessions, weighted by dollar
                          volume. A fund working an order leaves a one-sided
                          print across several days; random flow does not string
                          days together. This is the feature that most directly
                          targets multi-day execution.

  dollar volume anomaly   volume against its trailing median. A control as much
                          as a signal: it says something is happening without
                          saying what, and including it keeps the other features
                          from taking credit for plain volume effects.

Every feature is strictly trailing. A feature dated t uses information available
at the close of t and no later, and the backtest applies a further execution lag
on top. The two are separate protections and both are tested.

A caution about Amihud that applies to everything downstream: |return|/dollar
volume is a ratio of two quantities that are both driven by news. A stock that
gaps on an earnings release has both a large |return| and large volume, and which
one moves more is close to arbitrary. That makes the measure noisy in exactly the
periods where the signal is supposed to be strongest, and it is why the abnormal
version -- differenced against the name's own recent history -- is the one this
project treats as the signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import Panel

# Amihud values are ~1e-9 in raw units; the conventional scaling makes them
# readable without changing any ranking.
AMIHUD_SCALE = 1e6


def _safe_ratio(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    d = den.where(den > 0)
    return num / d


def daily_impact(panel: Panel) -> pd.DataFrame:
    """|return| per dollar of volume: the single-day Amihud ratio."""
    ret = panel.returns().abs()
    return _safe_ratio(ret, panel.dollar_volume) * AMIHUD_SCALE


def amihud(panel: Panel, window: int = 21, min_periods: int | None = None) -> pd.DataFrame:
    """Mean single-day impact over a trailing window -- the Kyle lambda proxy.

    The mean of a ratio, not a ratio of means, which is Amihud's own definition
    and the more robust of the two here: a single huge-volume day would otherwise
    dominate the denominator and drag the whole window's illiquidity down.
    """
    mp = min_periods or max(5, window // 2)
    return daily_impact(panel).rolling(window, min_periods=mp).mean()


def abnormal_lambda(panel: Panel, window: int = 60, min_periods: int | None = None,
                    clip: float = 5.0) -> pd.DataFrame:
    """Today's impact against its own trailing median, in logs.

    Positive  = more impact than this name's volume usually causes -> someone is
                *demanding* liquidity, paying up to get filled.
    Negative  = unusual volume absorbed with less impact than normal -> someone is
                *supplying* liquidity.

    The trailing median excludes today, so the comparison is against history
    rather than against a window containing the observation being scored. Logs
    make the ratio symmetric -- half the usual impact and twice the usual impact
    become -0.69 and +0.69 rather than -0.5 and +1.0 -- which matters because the
    raw ratio's asymmetry would otherwise show up as a spurious cross-sectional
    tilt toward whichever names happen to be having a quiet week.
    """
    mp = min_periods or max(10, window // 3)
    impact = daily_impact(panel)
    base = impact.rolling(window, min_periods=mp).median().shift(1)
    ratio = _safe_ratio(impact, base)
    out = np.log(ratio.where(ratio > 0))
    return out.clip(-clip, clip)


def persistent_imbalance(panel: Panel, max_run: int = 5,
                         volume_window: int = 60) -> pd.DataFrame:
    """Signed run length, weighted by how heavy the volume was on those days.

    For each name, the current run of same-signed daily returns is accumulated,
    each day contributing its relative dollar volume (against a trailing median).
    A three-day rally on double-normal volume scores far higher than a three-day
    rally on thin volume, which is the distinction the feature exists to draw:
    the first looks like someone working an order, the second looks like nothing.

    Runs are capped at ``max_run`` days. Uncapped, a long quiet drift would
    dominate the cross-section on length alone, which is a momentum signal rather
    than a footprint one.
    """
    ret = panel.returns()
    dv = panel.dollar_volume
    mp = max(10, volume_window // 3)
    rel_vol = _safe_ratio(dv, dv.rolling(volume_window, min_periods=mp).median().shift(1))
    rel_vol = rel_vol.clip(upper=10.0).fillna(0.0)

    sign = np.sign(ret.to_numpy())
    weight = rel_vol.to_numpy()
    out = np.zeros_like(sign, dtype=float)
    T, N = sign.shape
    run_len = np.zeros(N)
    run_sign = np.zeros(N)
    acc = np.zeros(N)
    for t in range(T):
        s = sign[t]
        same = (s == run_sign) & (s != 0)
        run_len = np.where(same, run_len + 1, np.where(s != 0, 1.0, 0.0))
        acc = np.where(same, acc + weight[t], np.where(s != 0, weight[t], 0.0))
        run_sign = np.where(s != 0, s, run_sign)
        capped = np.minimum(run_len, max_run)
        # Score is signed accumulated volume over the (capped) run.
        out[t] = run_sign * acc * (capped / np.maximum(run_len, 1.0))
    frame = pd.DataFrame(out, index=ret.index, columns=ret.columns)
    return frame.where(ret.notna())


def dollar_volume_anomaly(panel: Panel, window: int = 60,
                          min_periods: int | None = None,
                          clip: float = 4.0) -> pd.DataFrame:
    """Log dollar volume against its own trailing median, excluding today."""
    mp = min_periods or max(10, window // 3)
    dv = panel.dollar_volume
    base = dv.rolling(window, min_periods=mp).median().shift(1)
    ratio = _safe_ratio(dv, base)
    return np.log(ratio.where(ratio > 0)).clip(-clip, clip)


def turnover_zscore(panel: Panel, window: int = 60) -> pd.DataFrame:
    """Volume anomaly expressed as a z-score rather than a log ratio."""
    dv = panel.dollar_volume
    mp = max(10, window // 3)
    mu = dv.rolling(window, min_periods=mp).mean().shift(1)
    sd = dv.rolling(window, min_periods=mp).std().shift(1)
    return ((dv - mu) / sd.where(sd > 0)).clip(-6, 6)


# ------------------------------------------------------------------ the score


FEATURES = {
    "amihud": lambda p: amihud(p),
    "abnormal_lambda": lambda p: abnormal_lambda(p),
    "persistent_imbalance": lambda p: persistent_imbalance(p),
    "dollar_volume_anomaly": lambda p: dollar_volume_anomaly(p),
}

# Signs applied when combining into the footprint score. These encode the
# *hypothesis*, and the experiments test it rather than assume it:
#
#   persistent_imbalance  +1  a name being accumulated on heavy volume keeps
#                             going, because the order is not finished.
#   abnormal_lambda       -1  low impact per unit volume means someone is
#                             supplying liquidity into demand that has not
#                             finished arriving; high impact means the move has
#                             already paid for itself and is more likely to
#                             revert.
#   dollar_volume_anomaly  0  entered as a control, not a directional bet.
#   amihud                 0  a level, not an event; carried for diagnostics.
DEFAULT_SIGNS = {
    "persistent_imbalance": 1.0,
    "abnormal_lambda": -1.0,
    "dollar_volume_anomaly": 0.0,
    "amihud": 0.0,
}


def compute_features(panel: Panel, which: list[str] | None = None) -> dict:
    names = which or list(FEATURES)
    return {n: FEATURES[n](panel) for n in names}


def cross_sectional_z(frame: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Standardise within each date, so features combine on a common scale."""
    mu = frame.mean(axis=1)
    sd = frame.std(axis=1)
    z = frame.sub(mu, axis=0).div(sd.where(sd > 0), axis=0)
    return z.clip(-clip, clip)


def footprint_score(panel: Panel, signs: dict | None = None,
                    features: dict | None = None) -> pd.DataFrame:
    """Combine the components into one signal, equal-weighted after z-scoring.

    Equal weights on purpose. Fitting the combination weights on the same data
    that then evaluates the result is the most reliable way to manufacture an
    out-of-sample disappointment, and with four features there is not enough
    structure to justify the extra parameters. If a weighting is ever fitted, it
    belongs inside the walk-forward's training fold.
    """
    signs = signs or DEFAULT_SIGNS
    feats = features if features is not None else compute_features(panel)
    used = {k: v for k, v in signs.items() if v != 0.0 and k in feats}
    if not used:
        raise ValueError("no features carry a non-zero sign")
    parts = [cross_sectional_z(feats[k]) * s for k, s in used.items()]
    stacked = pd.concat(parts).groupby(level=0).mean()
    return stacked.reindex(index=feats[next(iter(used))].index)
