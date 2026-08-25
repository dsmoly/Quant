"""Synthetic panels for *controlled experiments*, not as a stand-in for data.

The distinction matters and is worth stating at the top of the file, because the
same code would be dishonest if used for the other purpose.

Legitimate uses, and the only ones made of this module:

  * Testing the machinery. Does the backtest respect its execution lag? Does the
    walk-forward purge what it claims to purge? Does the effective-N estimator
    return 1 for a one-factor matrix and N for a diagonal one? These are
    questions about code, and a panel with known properties answers them exactly.

  * Controlled attribution. The sizing engine is measured against a signal whose
    true predictive power is *set by construction* -- zero for the dumb-signal
    control, a known IC for the efficiency measurement. That is an experiment
    about the framework, and it requires knowing the right answer in advance,
    which real data never provides.

Illegitimate use, deliberately not made: reporting any IC, Sharpe or breakeven
cost from this module as if it described US equities. The footprint features are
*not* planted in this generator -- the volume process here has no institutional
order-working in it -- so a feature that scores well on this panel has told you
nothing about whether it scores well on real prices.

The generator does try to reproduce the statistical features that make equity
backtesting hard, since machinery that only works on clean Gaussian data is
machinery that will break on arrival:

  * a dominant market factor, so that effective breadth really is far below N
  * fat-tailed innovations (Student-t), so tail-risk estimators face real tails
  * volatility clustering, so trailing vol estimates are always somewhat stale
  * lognormal, autocorrelated volume with occasional spikes
  * cross-sectionally varying volatility and price levels
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .loader import Panel


@dataclass
class SyntheticSpec:
    n_names: int = 25
    n_days: int = 1500
    start: str = "2015-01-02"
    market_vol: float = 0.011          # daily market factor vol
    n_sectors: int = 4                 # sector blocks beneath the market factor
    # Chosen, not fitted: at 0.015 the panel shows PC1 ~32% of variance and a
    # residual effective N of ~15 out of 25, which is in the range usually
    # reported for a diversified US large-cap set. It is a modelling choice and
    # the breadth result moves with it -- see the sensitivity table in FINDINGS.
    sector_vol: float = 0.015          # per-sector factor vol
    idio_vol_lo: float = 0.010
    idio_vol_hi: float = 0.028
    beta_lo: float = 0.6
    beta_hi: float = 1.5
    tail_df: float = 4.0               # Student-t degrees of freedom; <5 is fat
    vol_persistence: float = 0.94      # GARCH-like clustering
    price_lo: float = 20.0
    price_hi: float = 400.0
    adv_lo: float = 2e7                # daily dollar volume, low end
    adv_hi: float = 5e9
    volume_ar: float = 0.6
    volume_vol: float = 0.45
    seed: int = 0


@dataclass
class SyntheticPanel:
    panel: Panel
    signal: pd.DataFrame            # the planted signal, if any
    true_alpha: pd.DataFrame        # the true expected forward return it encodes
    spec: SyntheticSpec
    realised_ic: float              # what was actually planted, measured back

    @property
    def close(self) -> pd.DataFrame:
        return self.panel.close


def _student_t(rng, df, size):
    """Unit-variance Student-t draws."""
    x = rng.standard_t(df, size=size)
    return x / np.sqrt(df / (df - 2.0)) if df > 2 else x


def simulate(spec: SyntheticSpec | None = None, *, planted_ic: float = 0.0,
             signal_decay: float = 0.15, horizon: int = 5) -> SyntheticPanel:
    """Generate a panel, optionally with a signal of known predictive power.

    ``planted_ic`` is the target Pearson correlation between the standardised
    signal and the *tradable* forward return over ``horizon`` days. Zero gives a
    pure-noise signal, which is the dumb-signal control.
    """
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    n, T = spec.n_names, spec.n_days
    dates = pd.bdate_range(spec.start, periods=T)
    names = [f"SYN{i:02d}" for i in range(n)]

    betas = rng.uniform(spec.beta_lo, spec.beta_hi, n)
    idio_scale = rng.uniform(spec.idio_vol_lo, spec.idio_vol_hi, n)

    # Volatility clustering: a shared AR(1) in log-variance plus a per-name one.
    shocks = rng.normal(size=(T, n))
    logvar = np.zeros((T, n))
    common = np.zeros(T)
    for t in range(1, T):
        common[t] = spec.vol_persistence * common[t - 1] + 0.12 * rng.normal()
        logvar[t] = (spec.vol_persistence * logvar[t - 1]
                     + 0.10 * rng.normal(size=n) + 0.5 * (common[t] - common[t - 1]))
    vol_mult = np.exp(0.5 * (logvar + common[:, None]))

    market = _student_t(rng, spec.tail_df, T) * spec.market_vol * np.exp(0.5 * common)
    idio = _student_t(rng, spec.tail_df, (T, n)) * idio_scale * vol_mult

    # Sector blocks beneath the market factor. Without these the panel has
    # exactly one common factor, and projecting out PC1 would leave independent
    # residuals -- which makes the effective-breadth machinery look better than
    # it is, because the hard case (correlation that survives neutralisation) has
    # been assumed away. Sector factors are what make residual N_eff < N.
    sector_of = rng.integers(0, max(spec.n_sectors, 1), n)
    sector_returns = np.zeros((T, n))
    if spec.n_sectors > 0 and spec.sector_vol > 0:
        f = _student_t(rng, spec.tail_df, (T, spec.n_sectors)) * spec.sector_vol
        loadings = rng.uniform(0.7, 1.3, n)
        sector_returns = f[:, sector_of] * loadings

    # The planted signal: a slowly decaying AR(1) state, standardised each day.
    state = np.zeros((T, n))
    innov = rng.normal(size=(T, n))
    phi = float(np.exp(-signal_decay))
    for t in range(1, T):
        state[t] = phi * state[t - 1] + np.sqrt(1 - phi ** 2) * innov[t]
    z = (state - state.mean(axis=1, keepdims=True)) / (
        state.std(axis=1, keepdims=True) + 1e-12)

    returns = market[:, None] * betas + sector_returns + idio
    if planted_ic != 0.0:
        # Spread the predictable move over the horizon so it is capturable rather
        # than arriving in one jump: a signal at t nudges each of the next
        # `horizon` sessions. The per-day nudge is calibrated so the cumulative
        # correlation with the horizon return lands near `planted_ic`.
        per_day = planted_ic * float(np.mean(idio_scale)) / np.sqrt(max(horizon, 1))
        nudge = np.zeros_like(returns)
        for h in range(1, horizon + 1):
            nudge[h:] += per_day * z[:-h] / horizon * horizon ** 0.5
        returns = returns + nudge

    prices = np.zeros((T, n))
    prices[0] = rng.uniform(spec.price_lo, spec.price_hi, n)
    for t in range(1, T):
        prices[t] = prices[t - 1] * (1.0 + returns[t])
    prices = np.clip(prices, 0.5, None)

    # Volume: lognormal AR(1) around a per-name level, with fat upside spikes.
    level = np.log(rng.uniform(spec.adv_lo, spec.adv_hi, n))
    lv = np.zeros((T, n))
    lv[0] = level
    for t in range(1, T):
        lv[t] = (level + spec.volume_ar * (lv[t - 1] - level)
                 + spec.volume_vol * rng.normal(size=n))
    # Volume rises with the size of the move, as it does in reality.
    lv += 0.8 * np.abs(returns) / (idio_scale + 1e-9) * 0.1
    dollar_volume = np.exp(lv)
    share_volume = dollar_volume / prices

    close = pd.DataFrame(prices, index=dates, columns=names)
    intraday = np.abs(rng.normal(scale=0.004, size=(T, n)))
    panel = Panel(
        close=close,
        open=close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(scale=0.002, size=(T, n))),
        high=close * (1 + intraday),
        low=close * (1 - intraday),
        volume=pd.DataFrame(share_volume, index=dates, columns=names),
    )

    signal = pd.DataFrame(z, index=dates, columns=names)
    if planted_ic == 0.0:
        # A pure-noise signal, redrawn independently of returns: the dumb signal.
        signal = pd.DataFrame(rng.normal(size=(T, n)), index=dates, columns=names)

    fwd = (close.shift(-(1 + horizon)) / close.shift(-1) - 1.0)
    aligned = signal.where(fwd.notna())
    realised = float(pd.concat(
        [aligned.stack(), fwd.stack()], axis=1).dropna().corr().iloc[0, 1]) \
        if fwd.notna().to_numpy().any() else float("nan")

    true_alpha = signal * float(np.mean(idio_scale)) * planted_ic
    out = SyntheticPanel(panel=panel, signal=signal, true_alpha=true_alpha,
                         spec=spec, realised_ic=realised)
    out.sectors = pd.Series(sector_of, index=names, name="sector")
    return out


def write_stooq_csvs(panel: Panel, out_dir, *, suffix: str = ".us.txt") -> list[str]:
    """Write a panel back out as Stooq-format files, for fixture generation."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for t in panel.tickers:
        df = pd.DataFrame({
            "Date": panel.close.index.strftime("%Y-%m-%d"),
            "Open": panel.open[t].to_numpy().round(4),
            "High": panel.high[t].to_numpy().round(4),
            "Low": panel.low[t].to_numpy().round(4),
            "Close": panel.close[t].to_numpy().round(4),
            "Volume": panel.volume[t].to_numpy().round(0).astype("int64"),
        })
        path = out / f"{t.lower()}{suffix}"
        df.to_csv(path, index=False)
        written.append(str(path))
    return written
