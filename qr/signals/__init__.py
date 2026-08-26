"""Candidate signals, chosen for having a reason not to be arbitraged away.

None of these are validated. They are hypotheses with a stated mechanism, a
stated reason the mechanism might survive contact with a crowded market, and
stated kill criteria. Run them through `qr.evaluate.evaluate_signal` with a
trial log before believing any of them -- and note that reading this file and
picking the two that sound best is itself a form of selection that the trial
count will not capture.

Everything here returns a (T, N) array aligned to the panel, observable at the
close of date t, with NaN where it is not computable. Nothing here looks ahead.
"""

from qr.signals.price import (
    overnight_intraday_split,
    intraday_reversal,
    turn_of_month_flow,
    dispersion_conditioned_trend,
    idiosyncratic_volatility,
    amihud_illiquidity_change,
    trend_ensemble,
    time_series_momentum,
)

__all__ = [
    "overnight_intraday_split", "intraday_reversal", "turn_of_month_flow",
    "dispersion_conditioned_trend", "idiosyncratic_volatility",
    "amihud_illiquidity_change", "trend_ensemble", "time_series_momentum",
]
