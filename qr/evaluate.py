"""The one call that runs the whole protocol on a candidate signal.

`evaluate_signal` is deliberately opinionated about order, because the order is
where most of the value is:

  1. condition the feature cross-sectionally (rank -> normal scores)
  2. **neutralise against what you already own, before measuring anything**
  3. IC with HAC inference, on both the raw and the neutralised version
  4. quantile sorts, to see whether it is monotone and which leg pays
  5. decay curve -> the beta the sizing engine needs
  6. long-short portfolio returns -> Sharpe with a block-bootstrap CI
  7. deflate that Sharpe by the trial count, which the log supplies

Step 2 is the one people skip and it is the one that kills most candidates. A
signal that survives neutralisation against market, size, value and momentum is
worth the rest of the pipeline; one that does not was a factor exposure wearing
a new name, and every downstream statistic will look great and mean nothing.

The report prints a verdict, but read `notes` -- it collects the specific
reasons a signal is fragile even when the headline numbers pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qr.bootstrap import sharpe_ci
from qr.decay import fit_decay, marginal_ic_curve, DecayFit
from qr.deflated import deflated_sharpe, min_track_record_length
from qr.ic import ICSummary, ic_series, ic_summary
from qr.panel import forward_return
from qr.quantiles import QuantileReport, quantile_report
from qr.transforms import neutralise, rank_normal, winsorise


@dataclass
class SignalReport:
    name: str
    horizon: int
    ic_raw: ICSummary
    ic_neutral: ICSummary | None
    quantiles: QuantileReport
    decay: DecayFit
    sharpe: dict
    deflated: dict
    min_trl: float
    turnover: float
    net_ic_estimate: float
    n_trials: int
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.ic_neutral.t_stat if self.ic_neutral else self.ic_raw.t_stat):
            return "UNMEASURABLE"
        t = (self.ic_neutral or self.ic_raw).t_stat
        d = self.deflated.get("dsr", float("nan"))
        if abs(t) < 2.0:
            return "DEAD"
        if np.isfinite(d) and d < 0.95:
            return "NOT DEFLATION-PROOF"
        if self.quantiles.monotonicity is not None and \
           np.isfinite(self.quantiles.monotonicity) and abs(self.quantiles.monotonicity) < 0.5:
            return "NON-MONOTONE"
        return "SURVIVES"

    def __str__(self) -> str:
        L = [f"=== {self.name}  (horizon {self.horizon}) ===",
             f"  raw       {self.ic_raw}"]
        if self.ic_neutral is not None:
            L.append(f"  neutral   {self.ic_neutral}")
        L += [f"  {self.quantiles}",
              f"  {self.decay}",
              f"  Sharpe {self.sharpe['annualised']:+.2f} "
              f"[{self.sharpe['ci_low']:+.2f}, {self.sharpe['ci_high']:+.2f}]  "
              f"(naive {self.sharpe['naive_annualised']:+.2f}, "
              f"Lo factor {self.sharpe['lo_factor']:.2f})",
              f"  deflated SR p={self.deflated.get('dsr', float('nan')):.3f} "
              f"vs best-of-{self.n_trials} benchmark "
              f"{self.deflated.get('benchmark_sr', float('nan')):+.4f}",
              f"  min track record: {self.min_trl:,.0f} periods",
              f"  turnover {self.turnover:.3f}/period, "
              f"net IC estimate {self.net_ic_estimate:+.4f}",
              f"  VERDICT: {self.verdict}"]
        for n in self.notes:
            L.append(f"    ! {n}")
        return "\n".join(L)


def _long_short_returns(signal, returns, mask, n_quantiles=5):
    """Equal-weight top-minus-bottom quantile portfolio, rebalanced every period."""
    from scipy import stats as _st
    T, N = signal.shape
    out = np.full(T, np.nan)
    w_prev = np.zeros(N)
    turnover = []
    for t in range(T - 1):
        m = mask[t] & np.isfinite(signal[t])
        n = int(m.sum())
        if n < n_quantiles * 4:
            continue
        v = signal[t, m]
        r = _st.rankdata(v, method="average")
        q = np.clip(((r - 0.5) / n * n_quantiles).astype(int), 0, n_quantiles - 1)
        w = np.zeros(N)
        idx = np.flatnonzero(m)
        top, bot = idx[q == n_quantiles - 1], idx[q == 0]
        if top.size == 0 or bot.size == 0:
            continue
        w[top] = 0.5 / top.size
        w[bot] = -0.5 / bot.size
        nxt = returns[t + 1]
        ok = np.isfinite(nxt)
        out[t] = float(np.nansum(w[ok] * nxt[ok]))
        turnover.append(float(np.abs(w - w_prev).sum()))
        w_prev = w
    return out, (float(np.mean(turnover)) if turnover else float("nan"))


def evaluate_signal(signal, returns, mask=None, name: str = "signal",
                    horizon: int = 1, factors=None, factor_names=None,
                    trial_log=None, family: str = "default",
                    n_quantiles: int = 5, cost_per_unit: float = 0.0005,
                    periods_per_year: int = 252,
                    decay_horizons=(1, 2, 3, 5, 10, 20, 40),
                    hypothesis: str | None = None,
                    config: dict | None = None) -> SignalReport:
    """Run the full protocol. See module docstring for the order and why.

    ``factors`` is what you already own -- a list of (T, N) arrays. Omitting it
    is allowed and the report will say so, loudly, because an unneutralised IC
    is not evidence of anything new.

    ``trial_log`` is a ``qr.trials.TrialLog``. Without one the deflated Sharpe
    is computed against a single trial, which is almost certainly a lie; the
    report records a note saying so.
    """
    s = np.asarray(signal, dtype=float)
    r = np.asarray(returns, dtype=float)
    if s.shape != r.shape:
        raise ValueError(f"signal shape {s.shape} != returns shape {r.shape}")
    m = np.isfinite(s) if mask is None else (np.asarray(mask, dtype=bool) & np.isfinite(s))
    notes: list[str] = []

    cond = rank_normal(winsorise(s, m), m)
    fwd = forward_return(r, horizon)

    ic_raw = ic_summary(ic_series(cond, fwd, m), horizon=horizon)

    ic_neu, cond_used = None, cond
    if factors is not None:
        neu = neutralise(cond, factors, m)
        cond_used = rank_normal(neu, m & np.isfinite(neu))
        ic_neu = ic_summary(ic_series(cond_used, fwd, m), horizon=horizon)
        if np.isfinite(ic_raw.mean) and np.isfinite(ic_neu.mean) and abs(ic_raw.mean) > 0:
            kept = abs(ic_neu.mean) / abs(ic_raw.mean)
            if kept < 0.5:
                notes.append(
                    f"neutralisation removed {1 - kept:.0%} of the IC -- this is mostly "
                    f"an exposure to {', '.join(factor_names) if factor_names else 'known factors'}")
    else:
        notes.append("NO FACTOR NEUTRALISATION: the IC below may be a known factor "
                     "in disguise. Pass `factors=` before believing this.")

    qr_ = quantile_report(cond_used, fwd, m, n_quantiles=n_quantiles, horizon=horizon)
    if np.isfinite(qr_.monotonicity) and abs(qr_.monotonicity) < 0.5:
        notes.append("quantile means are not monotone in the signal -- ranking is "
                     "destroying information; try a flexible model on the raw feature")
    if np.isfinite(qr_.long_leg) and np.isfinite(qr_.short_leg):
        tot = abs(qr_.long_leg) + abs(qr_.short_leg)
        if tot > 0 and abs(qr_.short_leg) / tot > 0.75:
            notes.append("over 75% of the spread comes from the short leg -- check "
                         "borrow availability and fees before believing the Sharpe")

    hs, ics, ses, ts = marginal_ic_curve(cond_used, r, decay_horizons, m)
    dec = fit_decay(hs, ics, ses, ts)
    if not np.isfinite(dec.beta):
        notes.append(f"decay unidentified: {dec.note}")

    ls, turn = _long_short_returns(cond_used, r, m, n_quantiles)
    sh = sharpe_ci(ls, periods_per_year=periods_per_year)

    n_trials, sr_std = 1, 1.0
    if trial_log is not None:
        n_trials = max(trial_log.count(family=family), 1)
        sr_std = trial_log.sr_std(family=family)
    else:
        notes.append("NO TRIAL LOG: deflated Sharpe assumes a single trial, which "
                     "is almost never true. Pass `trial_log=` to deflate honestly.")

    dsr = deflated_sharpe(returns=ls, n_trials=n_trials, sr_std=sr_std)
    trl = min_track_record_length(dsr["sharpe"], dsr["benchmark_sr"],
                                  dsr["skew"], dsr["kurtosis"])

    net_ic = float("nan")
    if np.isfinite(turn) and np.isfinite(sh["sharpe_per_period"]):
        gross = float(np.nanmean(ls))
        net = gross - turn * cost_per_unit
        sd = float(np.nanstd(ls, ddof=1))
        net_ic = net / sd if sd > 0 else float("nan")
        if np.isfinite(net) and gross > 0 and net <= 0:
            notes.append(f"costs eat the whole edge: gross {gross:+.5f}/period vs "
                         f"turnover {turn:.2f} x {cost_per_unit:.4f} = "
                         f"{turn * cost_per_unit:.5f}")

    rep = SignalReport(name, horizon, ic_raw, ic_neu, qr_, dec, sh, dsr, trl,
                       turn, net_ic, n_trials, notes)

    if trial_log is not None:
        cfg = dict(config or {})
        cfg.update({"horizon": horizon, "n_quantiles": n_quantiles,
                    "neutralised": factors is not None})
        trial_log.record(
            hypothesis or name, cfg,
            {"sharpe": dsr["sharpe"], "ic": (ic_neu or ic_raw).mean,
             "ic_t": (ic_neu or ic_raw).t_stat, "verdict": rep.verdict},
            family=family)
    return rep
