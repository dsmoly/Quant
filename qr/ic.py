"""Information coefficient: does the signal rank next period's returns?

Two decisions are baked in here and both are deliberate.

**Spearman by default.** Return cross-sections are fat-tailed enough that a
Pearson IC is often a report on the three largest movers. Rank IC answers the
question the sizing engine actually cares about -- ordering -- and is stable.

**Newey-West standard errors, always.** The IC series is autocorrelated
whenever the horizon exceeds the sampling interval, which for any horizon > 1
day sampled daily is always. A naive t-stat on overlapping IC overstates
significance by roughly sqrt(horizon), which is the single most common way a
dead signal looks alive. There is no flag to turn this off; if you want the
naive number, read `ic_summary(...).t_naive` and feel bad about it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def _rank_rows(A, mask):
    """Average-rank each masked row; NaN elsewhere. Vectorised where it matters."""
    out = np.full(A.shape, np.nan)
    for t in range(A.shape[0]):
        m = mask[t]
        if m.sum() >= 3:
            out[t, m] = stats.rankdata(A[t, m], method="average")
    return out


def ic_series(signal, fwd, mask=None, method: str = "spearman", min_names: int = 10):
    """Per-date cross-sectional IC between ``signal`` and ``fwd``.

    Returns a (T,) array with NaN on dates that had too few valid names.
    """
    s = np.asarray(signal, dtype=float)
    f = np.asarray(fwd, dtype=float)
    if s.shape != f.shape:
        raise ValueError(f"signal shape {s.shape} != fwd shape {f.shape}")
    m = np.isfinite(s) & np.isfinite(f)
    if mask is not None:
        m &= np.asarray(mask, dtype=bool)
    if method == "spearman":
        A, B = _rank_rows(s, m), _rank_rows(f, m)
    elif method == "pearson":
        A, B = np.where(m, s, np.nan), np.where(m, f, np.nan)
    else:
        raise ValueError("method must be 'spearman' or 'pearson'")

    T = s.shape[0]
    out = np.full(T, np.nan)
    for t in range(T):
        sel = m[t]
        if sel.sum() < min_names:
            continue
        a, b = A[t, sel], B[t, sel]
        a = a - a.mean()
        b = b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        if d > 0:
            out[t] = float((a * b).sum() / d)
    return out


def newey_west_se(x, lags: int | None = None) -> float:
    """Standard error of the mean of ``x``, Bartlett-kernel HAC.

    ``lags=None`` uses the standard automatic rule floor(4*(T/100)^(2/9)); pass
    the overlap horizon explicitly when you know it -- for an h-period
    overlapping IC series the right answer is at least h-1.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = x.size
    if T < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    lags = int(max(0, min(lags, T - 2)))
    e = x - x.mean()
    gamma0 = float(e @ e / T)
    var = gamma0
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        g = float(e[l:] @ e[:-l] / T)
        var += 2.0 * w * g
    var = max(var, 1e-300)
    return float(np.sqrt(var / T))


@dataclass
class ICSummary:
    mean: float
    std: float
    ir: float               # mean / std, per period -- the "IC information ratio"
    t_stat: float           # Newey-West
    t_naive: float          # what you would have reported without HAC
    se_nw: float
    n_obs: int
    lags: int
    hit_rate: float         # fraction of dates with IC > 0

    def __str__(self) -> str:
        return (f"IC {self.mean:+.4f}  IR {self.ir:+.3f}  "
                f"t(NW,{self.lags}) {self.t_stat:+.2f}  (naive {self.t_naive:+.2f})  "
                f"hit {self.hit_rate:.1%}  n={self.n_obs}")


def ic_summary(ic, lags: int | None = None, horizon: int | None = None) -> ICSummary:
    """Summarise an IC series with autocorrelation-aware inference.

    Pass ``horizon`` when the IC came from overlapping h-period returns and the
    lag length will default to h-1, which is the minimum defensible choice.
    """
    x = np.asarray(ic, dtype=float)
    v = x[np.isfinite(x)]
    n = v.size
    if n < 3:
        return ICSummary(*( [float("nan")] * 6 + [n, 0, float("nan")] ))
    if lags is None and horizon is not None:
        lags = max(int(horizon) - 1, 0)
    se = newey_west_se(v, lags)
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    sd = float(v.std(ddof=1))
    naive_se = sd / np.sqrt(n)
    return ICSummary(
        mean=float(v.mean()),
        std=sd,
        ir=float(v.mean() / sd) if sd > 0 else float("nan"),
        t_stat=float(v.mean() / se) if se > 0 else float("nan"),
        t_naive=float(v.mean() / naive_se) if naive_se > 0 else float("nan"),
        se_nw=float(se),
        n_obs=n,
        lags=int(lags),
        hit_rate=float((v > 0).mean()),
    )
