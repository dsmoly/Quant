"""qr -- a research harness for cross-sectional and directional alpha work.

The order of the modules is the order of the workflow:

    panel       aligned (T, N) data with an explicit universe mask
    transforms  cross-sectional conditioning: winsorise, rank, neutralise
    ic          does the signal predict? rank IC with autocorrelation-aware t-stats
    decay       how fast does it die? -> beta for the sizing engine
    quantiles   is it monotone, or is one tail carrying it?
    famamacbeth multivariate: does it survive next to what you already own?
    bootstrap   confidence intervals that respect serial dependence
    deflated    how much of this Sharpe is the search itself?
    trials      the persistent trial log that `deflated` needs to be honest
    cv          purged + embargoed splits for anything fitted
    combine     many signals into one, with the shrinkage that survives OOS
    sizing      signal -> positions
    backtest    the loop, sharing sizing with live

Design rules the whole package follows:

1. Nothing takes a DataFrame of mixed semantics. Features are (T, N) float
   arrays with an explicit `mask` of tradable names. Alignment bugs are the
   most common silent killer in this kind of work, so alignment is the one
   thing the types make explicit.
2. Forward returns are constructed in exactly one place (`panel.forward_return`)
   and every evaluator consumes that. The off-by-one that produces a beautiful
   backtest is not reachable from the public API.
3. Every statistic that can be inflated by serial dependence reports an
   autocorrelation-aware version by default, not as an option.
"""

from qr.panel import Panel, forward_return
from qr.transforms import winsorise, zscore, rank_normal, neutralise, demean_by_group
from qr.ic import ic_series, ic_summary, ICSummary, newey_west_se
from qr.decay import marginal_ic_curve, fit_decay, DecayFit
from qr.quantiles import quantile_report, QuantileReport
from qr.famamacbeth import fama_macbeth, FamaMacBeth
from qr.bootstrap import stationary_bootstrap, sharpe_ci, lo_annualisation_factor
from qr.deflated import deflated_sharpe, expected_max_sharpe, min_track_record_length
from qr.trials import TrialLog
from qr.cv import PurgedKFold, walk_forward_splits
from qr.combine import shrink_by_tstat, equal_weight, ic_weighted, hierarchical_combine
from qr.evaluate import evaluate_signal, SignalReport
from qr.sizing import (SizingEngine, SizingParams, SizingDiagnostics,
                       effective_n_participation, residualise_covariance,
                       ledoit_wolf_shrink, to_correlation, predicted_vol,
                       usable_names, apply_caps, apply_band)

__all__ = [
    "Panel", "forward_return",
    "winsorise", "zscore", "rank_normal", "neutralise", "demean_by_group",
    "ic_series", "ic_summary", "ICSummary", "newey_west_se",
    "marginal_ic_curve", "fit_decay", "DecayFit",
    "quantile_report", "QuantileReport",
    "fama_macbeth", "FamaMacBeth",
    "stationary_bootstrap", "sharpe_ci", "lo_annualisation_factor",
    "deflated_sharpe", "expected_max_sharpe", "min_track_record_length",
    "TrialLog",
    "PurgedKFold", "walk_forward_splits",
    "shrink_by_tstat", "equal_weight", "ic_weighted", "hierarchical_combine",
    "evaluate_signal", "SignalReport",
    "SizingEngine", "SizingParams", "SizingDiagnostics",
    "effective_n_participation", "residualise_covariance", "ledoit_wolf_shrink",
    "to_correlation", "predicted_vol", "usable_names", "apply_caps", "apply_band",
]
