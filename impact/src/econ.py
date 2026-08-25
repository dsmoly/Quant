"""From basis points to whether it pays.

A response function is a statement about prediction. Turning it into a statement
about money needs three more things, and the third is the one usually skipped:

  1. the size of the predictable move at horizon h, in bps  (from response.py)
  2. the round-trip cost of capturing it, in bps
  3. how many *independent* attempts a year contains  (from persistence)

Point 3 is where Sharpe claims go wrong. Independent bets per year is set by the
signal's own correlation time, not by how often you choose to rebalance. Sampling
a signal ten times inside its correlation time gives ten correlated observations,
not ten bets, and treating them as independent inflates the annualised Sharpe by
roughly sqrt(10). That is the difference between a real 1.5 and a claimed 5.

Fee schedules are published and are given as defaults, but they are arguments:
the crossover horizon is what the whole exercise is for, and it should not
silently depend on a hard-coded assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

YEAR_SECONDS = 365.25 * 24 * 3600


@dataclass
class CostModel:
    """Round-trip cost in bps. Defaults are Binance USD-M futures, VIP 0.

    taker_bps      0.05% per side is the published VIP-0 USD-M taker fee at the
                   time of writing; spot VIP 0 is 0.10%, or 0.075% paying in BNB.
    spread_bps     half-spread paid on each side when crossing. For BTCUSDT
                   perpetual this is typically well under a basis point; it is an
                   argument because it is the number most likely to be wrong.
    slippage_bps   everything else: queue position, latency, partial fills.
    """

    taker_bps: float = 5.0
    spread_bps: float = 0.5
    slippage_bps: float = 0.5
    maker_bps: float = 2.0

    @property
    def round_trip_taker(self) -> float:
        return 2.0 * (self.taker_bps + self.spread_bps + self.slippage_bps)

    @property
    def round_trip_maker(self) -> float:
        """Both sides passive: no spread paid, but fills are not guaranteed."""
        return 2.0 * (self.maker_bps + self.slippage_bps)


def economics(curve: pd.DataFrame, persistence: pd.DataFrame, costs: CostModel,
              *, capture: float = 0.5, sd_multiple: float = 1.0) -> pd.DataFrame:
    """Attach costs and an annualised Sharpe estimate to a response curve.

    ``capture`` is the fraction of the predicted move actually realised. Assuming
    100% is the standard way to make a marginal edge look tradable: it ignores
    that entry and exit both happen at prices the signal has already moved. Half
    is a conventional, still generous, haircut.

    ``sd_multiple`` sets how large a flow event is being traded -- the response is
    measured per one standard deviation of signed flow, and a selective strategy
    would only act on larger ones.
    """
    need = ["integrated_time_s", "independent_obs_per_year"]
    df = curve.copy()
    missing = [c for c in need if c not in df.columns]
    if missing:
        df = df.merge(persistence[["horizon_s"] + missing], on="horizon_s", how="left")
    gross = df["r_bps"].abs() * capture * sd_multiple
    df["gross_bps"] = gross
    df["round_trip_taker_bps"] = costs.round_trip_taker
    df["round_trip_maker_bps"] = costs.round_trip_maker
    df["net_taker_bps"] = gross - costs.round_trip_taker
    df["net_maker_bps"] = gross - costs.round_trip_maker
    df["pays_taker"] = df["net_taker_bps"] > 0
    df["pays_maker"] = df["net_maker_bps"] > 0

    # Annualised Sharpe from independent attempts, not from rebalance frequency.
    n = df["independent_obs_per_year"].clip(lower=0)
    # Per-attempt edge relative to the noise of the same horizon's return.
    with np.errstate(divide="ignore", invalid="ignore"):
        per_attempt_ir = df["net_taker_bps"] / df["r_se_bps"].replace(0, np.nan) / \
            np.sqrt(np.maximum(df["n_obs"], 1))
    df["sharpe_taker"] = per_attempt_ir * np.sqrt(n)
    df["sharpe_naive_rebalance"] = per_attempt_ir * np.sqrt(
        YEAR_SECONDS / df["horizon_s"].clip(lower=1e-9))
    df["sharpe_inflation_factor"] = (df["sharpe_naive_rebalance"] /
                                     df["sharpe_taker"].replace(0, np.nan))
    return df


def crossover(df: pd.DataFrame, col: str = "net_taker_bps") -> dict:
    """Where the predictable move first covers costs.

    Reports the shortest horizon at which the net is positive and the horizon of
    the best net, since they need not be the same: the move keeps growing with
    horizon while costs stay fixed, so net usually rises monotonically once it
    turns -- but the *number of attempts* falls, which is why the Sharpe column
    matters more than the bps column.
    """
    ok = df[df[col] > 0]
    best = df.loc[df[col].idxmax()] if df[col].notna().any() else None
    best_sh = (df.loc[df["sharpe_taker"].idxmax()]
               if "sharpe_taker" in df and df["sharpe_taker"].notna().any() else None)
    return {
        "crossover_horizon_s": float(ok["horizon_s"].min()) if not ok.empty else None,
        "best_bps_horizon_s": float(best["horizon_s"]) if best is not None else None,
        "best_bps": float(best[col]) if best is not None else None,
        "best_sharpe_horizon_s": float(best_sh["horizon_s"]) if best_sh is not None else None,
        "best_sharpe": float(best_sh["sharpe_taker"]) if best_sh is not None else None,
        "never_pays": ok.empty,
    }


def fmt_horizon(seconds: float) -> str:
    s = float(seconds)
    if s < 60:
        return f"{s:g}s"
    if s < 3600:
        return f"{s/60:g}min"
    if s < 86400:
        return f"{s/3600:g}hr"
    return f"{s/86400:g}d"


HORIZONS_S = [1, 10, 60, 300, 900, 3600, 14400, 86400, 432000]
