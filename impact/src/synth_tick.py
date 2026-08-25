"""Synthetic tick data with a *known* transient/permanent split.

This exists to validate the estimator, not to substitute for data. The point of
the whole exercise is to measure how much of impact is permanent; an estimator
that cannot recover a planted answer is worthless, and one that reports a
permanent component when none was planted is worse than worthless. Both
directions are tested.

The generator
-------------
Trade signs come from an **order-splitting** process rather than an AR(1),
because the long memory of order flow is the single feature that makes this
measurement hard. A parent order of random Pareto-distributed size emits that
many same-signed child trades; the resulting sign autocorrelation decays as a
power law, as it does on every real venue. An AR(1) would decay exponentially,
the flow would decorrelate in seconds, and the estimator would face none of the
difficulty it is built for.

The price has three parts, all controllable:

    efficient mid   m(t) = m(t-1) + kappa_perm * eps_t * sqrt(q_t) + sigma * noise
    transient       I(t) = sum_{s<=t} kappa_trans * eps_s * sqrt(q_s) * exp(-(t-s)/tau)
    observed mid    mid(t) = m(t) + I(t)
    trade print     p(t) = mid(t) * (1 +/- half_spread)   sign = aggressor side

So ``kappa_perm`` is the permanent impact per unit of signed root-volume and is
known exactly; ``kappa_trans`` and ``tau`` set a transient component that
mean-reverts and contributes nothing asymptotically; and the trade print carries
a bid-ask bounce of known size that the mid does not.

Setting ``kappa_perm = 0`` gives the null the whole project turns on: flow with
strong long memory, real short-horizon impact, and **no permanent component at
all**. Any estimator that reports permanent impact there is broken.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TickSpec:
    n_trades: int = 200_000
    start: str = "2024-01-01"
    mean_interval_s: float = 0.25       # average seconds between trades
    price0: float = 50_000.0
    kappa_perm: float = 0.6             # permanent impact, bps per unit signed root-qty
    kappa_trans: float = 2.5            # transient amplitude, same units
    tau_trades: float = 200.0           # transient decay, in trade time
    sigma_bps: float = 1.2              # per-trade efficient-price noise
    half_spread_bps: float = 0.5        # bid-ask bounce in the trade print
    parent_alpha: float = 1.5           # Pareto exponent for parent order size
    parent_min: int = 1
    qty_lognorm_sigma: float = 0.9
    qty_median: float = 0.05
    seed: int = 0


@dataclass
class SyntheticTicks:
    trades: pd.DataFrame        # ts, price, qty, is_buyer_maker
    book: pd.DataFrame          # ts, bid, ask, mid   (the true mid, uncontaminated)
    spec: TickSpec
    true_permanent_bps: float   # per 1 unit of signed root-qty
    signs: np.ndarray

    @property
    def has_planted_permanent(self) -> bool:
        return abs(self.spec.kappa_perm) > 1e-12


def _splitting_signs(n: int, alpha: float, kmin: int, rng) -> np.ndarray:
    """Signs from order splitting: Pareto parent sizes, same sign within a parent."""
    signs = np.empty(n, dtype=float)
    i = 0
    while i < n:
        size = int(kmin * (1.0 - rng.random()) ** (-1.0 / alpha))
        size = max(1, min(size, n - i, 5000))
        signs[i:i + size] = 1.0 if rng.random() < 0.5 else -1.0
        i += size
    return signs


def simulate_ticks(spec: TickSpec | None = None) -> SyntheticTicks:
    spec = spec or TickSpec()
    rng = np.random.default_rng(spec.seed)
    n = spec.n_trades

    signs = _splitting_signs(n, spec.parent_alpha, spec.parent_min, rng)
    qty = spec.qty_median * np.exp(rng.normal(0.0, spec.qty_lognorm_sigma, n))
    impulse = signs * np.sqrt(qty)

    # Efficient price: a random walk with a permanent flow component.
    perm_incr = spec.kappa_perm * impulse + spec.sigma_bps * rng.normal(size=n)
    log_mid_bps = np.cumsum(perm_incr)

    # Transient impact: exponential relaxation in trade time, computed by
    # recursion so it is O(n) rather than O(n^2).
    decay = float(np.exp(-1.0 / max(spec.tau_trades, 1e-9)))
    transient = np.empty(n)
    acc = 0.0
    for t in range(n):
        acc = acc * decay + spec.kappa_trans * impulse[t]
        transient[t] = acc

    mid = spec.price0 * np.exp((log_mid_bps + transient) / 1e4)
    half = spec.half_spread_bps / 1e4
    trade_price = mid * (1.0 + signs * half)

    dt = rng.exponential(spec.mean_interval_s, n)
    ts = pd.Timestamp(spec.start, tz="UTC") + pd.to_timedelta(np.cumsum(dt), unit="s")

    trades = pd.DataFrame({
        "ts": ts, "price": trade_price, "qty": qty,
        # is_buyer_maker True  <=>  aggressor was a SELLER  <=>  sign = -1
        "is_buyer_maker": signs < 0,
    })
    book = pd.DataFrame({"ts": ts, "bid": mid * (1 - half), "ask": mid * (1 + half),
                         "mid": mid})
    return SyntheticTicks(trades=trades, book=book, spec=spec,
                          true_permanent_bps=float(spec.kappa_perm), signs=signs)


# ------------------------------------------------------- Binance-format output


def _zip_bytes(df: pd.DataFrame, inner_name: str, header: bool) -> bytes:
    csv = df.to_csv(index=False, header=header).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(inner_name, csv)
    return buf.getvalue()


def write_binance_zips(ticks: SyntheticTicks, out_dir, symbol: str = "SYNUSDT",
                       market: str = "um", day: str = "2024-01-01",
                       header: bool = False, micros: bool = False) -> list[Path]:
    """Write the synthetic tape in real Binance daily-zip format.

    Fixtures go through the *actual* loader rather than bypassing it, so the
    column ordering, the header sniffing, the millisecond/microsecond ambiguity
    and the boolean parsing are all exercised by the tests. ``header`` and
    ``micros`` flip the two format variants Binance actually ships.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    unit = 1_000_000 if micros else 1000
    t = ticks.trades
    agg = pd.DataFrame({
        "agg_id": np.arange(len(t)),
        "price": t["price"].round(2),
        "qty": t["qty"].round(6),
        "first_id": np.arange(len(t)),
        "last_id": np.arange(len(t)),
        "ts": (t["ts"].astype("int64") // (1_000_000_000 // unit)),
        "is_buyer_maker": t["is_buyer_maker"].map({True: "true", False: "false"}),
        "is_best_match": "true",
    })
    b = ticks.book
    book = pd.DataFrame({
        "update_id": np.arange(len(b)),
        "bid_price": b["bid"].round(2), "bid_qty": 1.0,
        "ask_price": b["ask"].round(2), "ask_qty": 1.0,
        "transaction_time": (b["ts"].astype("int64") // (1_000_000_000 // unit)),
        "event_time": (b["ts"].astype("int64") // (1_000_000_000 // unit)),
    })
    paths = []
    for kind, frame in (("aggTrades", agg), ("bookTicker", book)):
        name = f"{market}-{symbol}-{kind}-{day}.zip"
        inner = f"{symbol}-{kind}-{day}.csv"
        p = out / name
        p.write_bytes(_zip_bytes(frame, inner, header))
        paths.append(p)
    return paths
