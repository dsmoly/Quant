"""Null floors for the response function, by shuffled timing.

Reusing the lesson from the equity work rather than the code: a bare
t-statistic is not interpretable. There, a Newey-West t of 2.05 on a persistent
feature turned out to be inside the noise, and the empirical critical value was
|IC| ~ 0.016 rather than the nominal 0.010. The same correction is needed here,
and it is needed *more*, because order flow is far more persistent than any of
the daily equity features and the horizons run out to five days on a few months
of data.

The control is **shuffled timing**, not sign inversion. Inversion is useless for
a self-calibrating estimator -- established earlier in this repo -- and here it
would be worse than useless, because flipping every sign flips the response
function exactly and tells you nothing.

The shuffle has to preserve the flow's own autocorrelation while destroying its
alignment with the price path. A plain permutation would break the long memory
and produce a null that is far too tight, making everything look significant. A
**circular block shuffle** with a block much longer than the flow's integrated
correlation time keeps the within-block memory intact and moves whole stretches
of flow to unrelated stretches of price.

Two nulls are computed and they answer different questions:

  shuffled flow    flow blocks are moved against a fixed price path. This is the
                   null for "does *this* flow predict *this* price path", and is
                   the primary control.
  shuffled price   the reverse. Reported as a consistency check: the two should
                   agree, and disagreement means the block length is wrong for
                   one of the series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .flow import Bars
from .propagator import circular_block_indices
from .response import response_at_horizon


@dataclass
class NullFloor:
    horizon_s: float
    n_draws: int
    r_abs_95_bps: float       # |R| exceeded 5% of the time under the null
    ic_abs_95: float
    t_abs_95: float
    r_mean: float
    r_sd: float

    def as_dict(self) -> dict:
        return {"horizon_s": self.horizon_s, "n_draws": self.n_draws,
                "r_abs_95_bps": self.r_abs_95_bps, "ic_abs_95": self.ic_abs_95,
                "t_abs_95": self.t_abs_95, "r_mean": self.r_mean, "r_sd": self.r_sd}


def _shuffled_bars(bars: Bars, block: int, rng, what: str = "flow") -> Bars:
    f = bars.frame()
    n = len(f)
    idx = circular_block_indices(n, block, rng)
    flow_cols = ["signed_volume", "signed_count", "volume", "n_trades"]
    price_cols = [c for c in ("trade_price", "mid") if c in f.columns]
    out = f.copy()
    if what == "flow":
        out[flow_cols] = f[flow_cols].to_numpy()[idx]
    else:
        out[price_cols] = f[price_cols].to_numpy()[idx]
    return Bars(freq=bars.freq, seconds=bars.seconds,
                trade_price=out["trade_price"],
                mid=out["mid"] if "mid" in out.columns else None,
                signed_volume=out["signed_volume"], signed_count=out["signed_count"],
                volume=out["volume"], n_trades=out["n_trades"])


def null_floor(bars: Bars, horizon_s: float, *, n_draws: int = 200,
               measure: str = "impact", price_kind: str = "auto",
               block_s: float | None = None, shuffle: str = "flow",
               seed: int = 0) -> NullFloor:
    """Empirical null distribution of R(h), IC and t at one horizon.

    ``block_s`` defaults to ten times the horizon, which is long enough that the
    block carries the flow memory relevant at that horizon. Too short a block
    and the null is too tight; too long and there are too few distinct blocks to
    shuffle.
    """
    rng = np.random.default_rng(seed)
    block = int(max(1, round((block_s or 10.0 * horizon_s) / bars.seconds)))
    block = min(block, max(1, len(bars.trade_price) // 8))
    rs, ics, ts = [], [], []
    for _ in range(n_draws):
        sb = _shuffled_bars(bars, block, rng, what=shuffle)
        try:
            r = response_at_horizon(sb, horizon_s, measure=measure, price_kind=price_kind)
        except Exception:
            continue
        if np.isfinite(r.r_bps):
            rs.append(r.r_bps)
        if np.isfinite(r.ic):
            ics.append(r.ic)
        if np.isfinite(r.r_t):
            ts.append(r.r_t)
    q = lambda a: float(np.quantile(np.abs(a), 0.95)) if len(a) >= 20 else float("nan")
    return NullFloor(
        horizon_s=float(horizon_s), n_draws=len(rs), r_abs_95_bps=q(rs),
        ic_abs_95=q(ics), t_abs_95=q(ts),
        r_mean=float(np.mean(rs)) if rs else float("nan"),
        r_sd=float(np.std(rs, ddof=1)) if len(rs) > 1 else float("nan"))


def null_curve(bars: Bars, horizons_s, *, n_draws: int = 200, measure: str = "impact",
               price_kind: str = "auto", shuffle: str = "flow",
               seed: int = 0) -> pd.DataFrame:
    rows = []
    for h in horizons_s:
        k = max(1, int(round(h / bars.seconds)))
        if k >= len(bars.trade_price) // 4:
            continue
        rows.append(null_floor(bars, h, n_draws=n_draws, measure=measure,
                               price_kind=price_kind, shuffle=shuffle,
                               seed=seed).as_dict())
    return pd.DataFrame(rows)


def compare_to_null(curve: pd.DataFrame, nulls: pd.DataFrame) -> pd.DataFrame:
    """Join measured response to its null floor and mark what clears it."""
    m = curve.merge(nulls, on="horizon_s", how="left", suffixes=("", "_null"))
    m["clears_null"] = m["r_bps"].abs() > m["r_abs_95_bps"]
    m["ic_clears_null"] = m["ic"].abs() > m["ic_abs_95"]
    m["r_over_null"] = m["r_bps"].abs() / m["r_abs_95_bps"]
    return m
