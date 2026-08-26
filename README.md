# Quant

A research harness for cross-sectional and directional alpha work, plus the
sizing engine that turns a signal into positions.

The harness exists because at IC ≈ 0.03 — where real cross-sectional equity
signals live — **you cannot tell a real signal from a lucky one by looking
harder at the data.** Only protocol saves you, and protocol has to be built
before the research, not bolted on after. Building it afterward means you can
never reconstruct how many things you tried.

## Install

```bash
pip install numpy scipy scikit-learn pytest
python -m pytest tests/ -q
python examples/demo.py
```

## The workflow

```python
import qr

report = qr.evaluate_signal(
    signal, returns,
    name="my_idea", horizon=5,
    factors=[market, size, value, momentum],   # what you already own
    factor_names=["market", "size", "value", "momentum"],
    trial_log=qr.TrialLog("research_log.jsonl"),
    family="flow_signals",
)
print(report)
```

The order inside `evaluate_signal` is the point:

1. condition the feature cross-sectionally (rank → normal scores)
2. **neutralise against what you already own, before measuring anything**
3. IC with HAC inference, raw *and* neutralised
4. quantile sorts — monotone, or is one leg carrying it?
5. decay curve → the `beta` the sizing engine needs
6. long-short returns → Sharpe with a block-bootstrap CI and Lo's correction
7. deflate that Sharpe by the trial count the log supplies

Step 2 kills most candidates. From the demo, a signal that is really a known
factor in disguise:

```
=== value_in_disguise  (horizon 1) ===
  raw       IC +0.0349  IR +0.511  t(NW,0) +19.79   hit 69.4%
  neutral   IC -0.0026  IR -0.037  t(NW,0)  -1.42   hit 48.3%
  VERDICT: DEAD
    ! neutralisation removed 93% of the IC -- this is mostly an exposure to market, value
```

A t-stat of 19.8 that evaporates on neutralisation. Without step 2 every
downstream number looks superb and means nothing.

## Modules

| module | what it does |
|---|---|
| `panel` | aligned (T, N) data; **the only definition of a forward return** |
| `transforms` | winsorise, z-score, rank→normal, neutralise, group-demean |
| `ic` | rank IC with Newey–West t-stats (never the naive one by default) |
| `decay` | marginal-IC decay curve → `beta`, with the fit restricted honestly |
| `quantiles` | monotonicity and long/short leg decomposition |
| `famamacbeth` | per-date cross-sectional regression, HAC t-stats |
| `bootstrap` | stationary bootstrap, Lo (2002) Sharpe annualisation |
| `deflated` | deflated Sharpe, expected max Sharpe, min track record length |
| `trials` | append-only trial log — the thing that makes deflation honest |
| `cv` | purged + embargoed k-fold and walk-forward splits |
| `combine` | shrinkage and hierarchical combination |
| `sizing` | robust LQ control → positions, neutral or directional |
| `signals` | candidate alphas with stated mechanisms |

## Three findings baked into the design

**Autocorrelation-aware inference is not optional.** Overlapping forward
returns make IC series autocorrelated, and a naive t-stat overstates
significance by roughly √horizon. There is no flag to turn HAC off. Lo's
correction to Sharpe annualisation is applied by default too — at AR(1)
ρ=0.2 the textbook √252 overstates the annualised Sharpe by 20%.

**Multiple testing dominates model choice.** Testing pure-noise signals against
pure-noise returns: 50 trials gives a median best |t| of 2.42; at 200 trials
there is a 30% chance of |t| > 3; at 1000 it is a certainty. Your effective
trial count includes every variant you abandoned without writing down, which is
why `TrialLog` is append-only and hashed on config.

**Estimating combination weights on collinear signals is expensive.** Twelve
signals at 0.55 pairwise correlation, 40 seeds: OLS reached 0.033 out-of-sample
correlation and ridge 0.040, against 0.063 for equal weight — which also had
lower variance. `hierarchical_combine` therefore equal-weights within a family
and only estimates across families.

## The sizing engine

The closed form is a robust LQ control (quadratic impact `k`, entropic
ambiguity penalty `phi`, signal decay `beta`); substituting the solved
coefficients back into the HJB leaves residuals of order 1e-18. What changed
from the earlier version:

- **The band is in the same units as the book.** It was computed in raw control
  units and compared against a volatility-rescaled target — measured at 1.4× a
  typical position, so only 42 of 120 names ever traded and the book ran at a
  fifth of its intended size.
- **The control's own trading rate is available.** `nu* = -rho (q - q*)` closes
  a fraction of the gap per period. The old code jumped to the band edge, which
  is the linear-cost impulse solution paired with the quadratic-cost value
  function — and is why `k` tested as inert across three orders of magnitude.
- **Volatility is reported on the book actually held**, not on a pre-band
  intermediate that read 5.9% while the real book was at 2.5%.
- **Caps are enforced after neutralisation**, iterated to a fixed point, so
  `max_weight` actually binds.
- **A NaN covariance row drops that name**, rather than silently returning
  unscaled raw-control weights.
- **Directional mode**: `neutral=False`, `residual_factors=0`. For a directional
  book the dominant factor *is* the bet, and projecting it out overstates
  effective breadth by about half.

Be careful with `effective_n_participation`: on a 1-factor population whose true
residual breadth is 87.9, a 120-day sample window returns 50.3 and a 250-day
window 64.6, with near-deterministic bias. It measures T/N as much as breadth.
Feed it a shrunk covariance (`shrink_covariance=True`, the default) and a
correlation rather than a covariance matrix.

## Signals

See [`docs/signal_catalogue.md`](docs/signal_catalogue.md). Selection principle:
prefer **forced flows** and **limits to arbitrage** over risk premia. Risk premia
are compensation rather than error, so they persist — but they are crowded and
give you maybe 0.3–0.5 Sharpe. Forced flows have a price-insensitive
counterparty, known timing, and capacity small enough that large funds cannot
be bothered.

Implemented today from OHLCV + calendar: conditioned turn-of-month rebalancing
flow, intraday-leg reversal, dispersion-conditioned trend, Amihud illiquidity
change, and idiosyncratic volatility (included as a deliberately non-monotone
control). Higher-conviction ideas needing index-membership, borrow, or options
data are documented with mechanism, data requirements and kill criteria.

## Status

Not validated on real data. The harness is tested (34 invariant tests, all
passing) and the demo reaches correct verdicts on a planted signal, pure noise,
and a factor in disguise. No execution layer yet.
