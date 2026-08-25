# Findings

Daily cross-sectional US equity strategy on institutional execution footprints.
Built against a data-loader interface. **No footprint feature has been tested on
real data.** One real-price dataset has since been converted (13 leveraged ETFs,
IEX-only volume) — it is sound enough to measure the correlation structure and
useless for the features themselves; see *Data status*.

Everything here is therefore one of two things, and the difference is marked in
every section:

* **Measured** — a property of the code, established on data with known
  construction. These conclusions hold.
* **Unmeasurable without real data** — anything about whether footprint features
  predict US equity returns. Not attempted, not guessed at.

---

## Data status

`api.exchange.coinbase.com`, `data.lobsterdata.com`, `data.binance.vision` and
`stooq.com` are all rejected by this environment's egress policy (403 to
CONNECT). The allowlist reaches package registries and GitHub and nothing else —
`www.google.com` is refused too. No prices were fabricated to fill the gap.

GitHub *is* reachable, so `piekstra/market-data` was cloned and converted by
`experiments/convert_parquet.py` into 13 daily CSVs now sitting in `data/`.
Read `data/README.md` before using them: the volume column comes from Alpaca's
`feed=iex`, a single venue holding ~2% of US consolidated volume, and every
footprint feature is a function of volume. The universe is also entirely
leveraged ETFs, which have no multi-day institutional accumulation to detect.
The prices are real, which is why the breadth measurement in section 2 stands
while nothing about the features does.

The project reads Stooq-format CSVs from `data/` and runs end to end the moment
files land there:

```
data/aapl.us.txt      Date,Open,High,Low,Close,Volume
data/msft.us.txt      2015-01-02,111.39,111.44,107.35,109.33,53204626
```

```bash
python3 -m experiments.walk_forward --data-dir data
```

With no data it falls back to a synthetic panel behind a banner repeated in every
section of the output.

---

## 1. The sizing engine, measured against a dumb signal

Run first, before any feature work, so the attribution is clean.
`experiments/sizing_value_add.py`, 12 seeds × 1500 sessions × 25 names, 20 bps
per side.

| arm | net Sharpe | gross Sharpe | turnover | breakeven | shrink |
|---|---|---|---|---|---|
| DUMB noise / quintile | **−6.63** | +0.08 | 1.578 | 0.2 bp | — |
| DUMB noise / full engine | **−1.52** | −0.19 | 0.029 | — | 0.12 |
| planted / quintile | −1.09 | +1.50 | 0.627 | 11.7 bp | — |
| planted / full engine | **+0.15** | +1.57 | 0.154 | 22.1 bp | 0.71 |
| planted / no vol target | +0.15 | +1.37 | 0.129 | 22.2 bp | 0.71 |
| planted / no shrinkage | −0.04 | +1.64 | 0.245 | 19.7 bp | 1.00 |
| planted / no band | −1.40 | +1.51 | 0.347 | 10.4 bp | 0.71 |
| planted / no haircut | −0.01 | +1.58 | 0.217 | 19.8 bp | 0.71 |

**No P&L is manufactured from noise.** Gross Sharpe on a zero-alpha signal is
+0.08 (quintile) and −0.19 (engine) — the test is on *gross*, because net Sharpe
on a worthless signal is supposed to be negative. That is the cost of trading
something that does not predict, and it is the correct answer, not a bug.

**The framework earns its place, and the band is why.** Full engine beats naive
quintile construction by **+1.24 Sharpe (6.3σ)**, turnover 0.627 → 0.154,
breakeven 11.7 → 22.1 bps. The ablations say almost all of that is one component:

* **no band −1.56 Sharpe** — dominant. Gross Sharpe barely moves (1.51 vs 1.57),
  so the band adds nothing to prediction; it just stops paying for the same
  position twice.
* no shrinkage −0.19 net, but **+0.07 gross**. Shrinkage *costs* gross return and
  more than repays it in costs.
* no haircut −0.16, no vol target −0.01 (inside the ±0.15 standard error — not
  measurable here).

**Against the dumb signal the engine cuts turnover 54× (1.578 → 0.029) and gross
exposure to 0.073 versus 0.588 on real alpha.** It notices there is nothing there.

### A real bug this experiment found

The first run had the dumb-signal arm trading hard and losing 5.7 Sharpe. The
cause was not noise: **volatility targeting is scale-invariant, so it divided the
confidence shrinkage straight back out.** The shrinkage correctly collapsed the
noise signal to 11% of its size, and the risk budget then multiplied it by
**9.5 × 10⁴⁶** to hit the 10% vol target. Confidence never reached the book.

Fixed by making deployed risk proportional to confidence —
`target_vol × confidence × breadth haircut` — rather than applying shrinkage to
alpha alone. Pinned by `test_an_unmeasured_signal_is_sized_at_essentially_zero`.

A second thing was wrong and is worth recording because the original reasoning
*sounded* right: risk was being predicted from the **residual** covariance, on
the argument that a dollar-neutral book has already shed its market exposure so
using the raw covariance would double-count it. That is backwards. If the book is
genuinely neutral the factor term is already ≈0 and the full covariance gives the
same answer by itself; using the residual instead **understates** risk by exactly
the factor exposure the book failed to neutralise — the one case where you most
need to know. Volatility is now predicted from the full covariance; the residual
matrix is used only for counting bets.

### Volatility targeting does not hit its nominal target, by design

Nominal 0.100 → × confidence 0.67 → × breadth haircut 0.92 → effective 0.062;
realised 0.051 (ratio to effective 0.83). Comparing realised vol to the *nominal*
target would make a working engine look broken. The residual 0.83 is trailing
covariance being stale, which is a real cost of daily rebalancing and not
removable.

---

## 2. Effective breadth — a result that partly contradicts the brief

**Measured.** The estimator is exact on known-answer matrices: identity(20) →
20.00, rank-one → 1.00, equicorrelation ρ=0.5 on 25 names → 3.57 against an
analytic 3.5714.

The brief expected naive `IC·√N` to overstate Sharpe badly. It does — but **not
for a dollar-neutral book**, and the distinction turns out to be the whole point:

| matrix | effective N (of 25) |
|---|---|
| raw correlation | **8.7** |
| residual, after projecting out PC1 | **21.3** |
| Marchenko–Pastur factor count | 4 |

Projecting out the dominant factor *flattens* the spectrum, so residual breadth
is **higher** than raw breadth. A dollar-neutral cross-sectional portfolio has
already neutralised most of PC1 by construction, so the haircut it deserves is
the residual one (≈0.92), not the raw one (≈0.59). Applying the raw haircut to a
neutral book penalises it for a risk it is not taking.

So the correct statement is narrower than "20 names is nowhere near 20 bets": for
a **net-exposed** book that is right and the haircut is severe; for a
**dollar-neutral** book breadth is close to N and the naive calculation is not far
wrong. Which one applies depends on the portfolio, not on the universe.

### Measured on real data, and it corrects the claim above

The convergence claim was made on synthetic panels. A partial check is now
possible on real prices (13 leveraged ETFs from `piekstra/market-data`; volume
there is unusable but *returns* are sound, since IEX prices track the NBBO for
liquid names). On the four names with enough overlapping history, 1393 sessions:

| | value |
|---|---|
| mean pairwise correlation | 0.752 |
| PC1 share of variance | **82.0%** |
| effective N (raw) | **1.45** of 4 |
| effective N (residual, PC1 removed) | **1.74** |
| Marchenko-Pastur factor count | 1 |

At IC = 0.03 the naive `IC*sqrt(N)` gives IR 0.95; the honest figure is **0.57**,
a 40% overstatement.

This cuts against the synthetic finding. There, residual breadth came out close
to N and the naive calculation was not far wrong for a neutral book. Here the
haircut is severe *even after* projecting out PC1 -- with one factor at 82% there
is almost nothing left underneath to diversify across. The synthetic generator's
sector structure was too weak, exactly as flagged when the value was chosen.

Neither number is the truth. Four 3x leveraged ETFs is the most correlated
universe available and is the opposite extreme from the synthetic panel; real
single-name equities sit somewhere between. The defensible conclusion is
narrower than either: **the size of the breadth haircut is an empirical property
of the universe and has to be measured per universe, not assumed** -- which is
why `breadth_report` is called on the actual returns in every experiment rather
than configured.

**Caveat, and it is not small.** Residual breadth is highly sensitive to sector
structure, which is a *modelling choice* in the synthetic generator rather than a
measurement:

| sector factor vol | PC1 share | N_eff raw | N_eff residual |
|---|---|---|---|
| 0.000 | 30.9% | 8.5 | 22.3 |
| 0.015 *(default)* | 31.7% | 7.5 | 14.6 |
| 0.030 | 38.5% | 5.0 | 7.1 |
| 0.060 | 46.8% | 3.4 | 4.0 |

The default was chosen to put PC1 near 32%, in the range usually reported for
diversified US large caps. **Real sector structure is an empirical question this
container cannot answer.** With strong sectors, residual breadth collapses to ~7
and the haircut becomes severe even for a neutral book.

---

## 3. The noise floor — the most useful number here

**Measured**, and it started as a mistake I made.

On one synthetic panel with **no planted signal**, `dollar_volume_anomaly`
produced an out-of-sample IC of +0.0195 (Newey-West t = 2.05) at five days and
+0.0279 (t = 2.75) at ten. Read alone that looks like a discovery. It was not:
across six independent generator seeds the same feature ran −0.0151 to +0.0007,
mostly negative, and the shuffled-timing control on the same panel reached
t = 1.73.

So `experiments/noise_floor.py` measures the null distribution directly — 60
panels, zero true signal, scored exactly as a real feature would be:

| signal | \|IC\| 95th pct | \|t_nw\| 95th pct | false positives at t>1.96 |
|---|---|---|---|
| white noise | 0.0095 | 1.77 | 3% |
| persistent_imbalance | 0.0114 | 1.59 | 2% |
| footprint_score | 0.0126 | 1.80 | 3% |
| abnormal_lambda | 0.0146 | 2.26 | **8%** |
| dollar_volume_anomaly | 0.0158 | 1.99 | **7%** |

**Newey-West is calibrated for a white-noise signal and under-corrects for a
persistent one.** The lag is chosen for the forward-window overlap; the feature's
*own* autocorrelation is not in it. For the persistent features the nominal 5%
test actually rejects 7–8% of the time, and the honest critical value is
|IC| ≈ 0.016, not 0.010.

Noise floor against sample size (|IC| a zero-signal feature reaches 5% of the
time — the bar a real feature must clear):

| names | sessions | \|IC\| 95th |
|---|---|---|
| 25 | 750 | 0.0157 |
| 25 | 1500 | 0.0095 |
| 25 | 3000 | 0.0080 |
| 50 | 1500 | 0.0069 |
| 50 | 3000 | 0.0032 |

**Bearing on the target.** Against the persistent-feature floor of ≈0.016, the
brief's expected OOS IC of 0.02–0.04 is *marginal at the bottom and comfortable
at the top*. An IC of 0.02 on 25 names and six years is within 1.5× of the floor
and should be treated as unproven; 0.04 clears it. I had predicted before running
this that the whole range would sit at or below the floor — that was too
pessimistic, and only the bottom of it is in doubt. Doubling names and history
cuts the floor by roughly 3×, so **universe size buys statistical power faster
than history does** (50×1500 beats 25×3000).

---

## 4. The inverted control does not work, and why

**Measured, and a genuine methodological trap.**

The first walk-forward run printed *byte-identical* rows for the live signal and
its inversion. Not a bug — a property:

```
alpha = IC · σ · z      flip the signal:  IC → −IC  and  z → −z
                        so  alpha → (−IC)·σ·(−z) = alpha
```

**Any strategy that estimates the sign of its own edge is invariant to a sign
flip of its signal.** The standard inverted-signal control is therefore vacuous
for self-calibrating strategies — and it fails *silently*, printing a plausible
number rather than an error.

A meaningful inversion has to hold the IC at the original signal's value, forcing
the strategy to trade the flipped signal with unflipped conviction. Fixed, it
mirrors exactly: gross Sharpe +0.67 → −0.67, +0.87 → −0.87. Pinned by
`test_a_self_calibrating_strategy_is_invariant_to_a_signal_sign_flip`.

The shuffled-timing control is unaffected and remains the more reliable of the
two.

---

## 5. Tail risk

**Measured** (estimator behaviour, not equity tails):

* GPD recovers a planted shape ξ=0.25 to within 0.08 on 4000 exceedances.
* Hill on 20 000 *exact* Pareto draws with α=3 gives a plateau of **[3.09, 3.11]**
  — close, but **not bracketing the truth**. That residual bias on ideal data is
  the argument for reporting a range: on a few thousand real daily returns, from
  a tail that is not exactly Pareto, a point estimate is not defensible.
* ES is reported with a bootstrap standard error every time, because ES at 5% on
  1000 observations averages ~50 points.
* ξ ≥ 1 returns NaN for expected shortfall rather than a number — the mean does
  not exist.
* The Anderson–Darling statistic is computed against the fitted CDF analytically;
  the sampled version is itself random, which is a poor property for a statistic
  used to accept or reject a threshold.

Threshold selection is by stated rule (≥40 exceedances, KS p > 0.05, shape
closest to the local median) with **the full candidate table returned**, so the
choice can be disagreed with.

---

## 6. What is *not* measured

Bluntly: **whether any of this predicts US equity returns.**

The synthetic generator contains no institutional order-working. The footprint
features have nothing to detect there, and near-zero IC on synthetic data is the
*correct* result — it is a null test for construction artifacts, not evidence
about equities. The walk-forward run confirms every feature sits inside its own
noise floor on that panel, which is what it should do.

Also unresolved without data:

* **Survivorship.** Documented at length in `universe.SURVIVORSHIP_NOTE` and
  printed at the top of every experiment. A current liquid-name list excludes
  everything delisted; historically worth 1–4%/yr on long-only US samples, which
  is comparable to or larger than the entire expected edge. It bites this project
  specifically hard: screening on *present* liquidity conditions on a cousin of
  the outcome variable, and the names whose λ dynamics are most informative are
  exactly the ones most likely to be missing.
* Real sector structure, and therefore the real breadth haircut (§2).
* Whether Amihud's known weakness bites — |return|/dollar volume is a ratio of
  two quantities both driven by news, so it is noisiest exactly when the signal
  should be strongest. This is why `abnormal_lambda`, differenced against the
  name's own history, is treated as the signal rather than the level.

**No Sharpe above 3 was produced anywhere**, so the brief's bug-hunt trigger was
never pulled. The one implausible number that did appear — a dumb signal earning
gross P&L — was investigated and turned out to be the vol-targeting flaw in §1.

---

## Reproducing

```bash
cd footprint
python3 -m pytest tests/ -q                      # 89 tests
python3 -m experiments.sizing_value_add          # §1
python3 -m experiments.noise_floor --size-scan   # §3
python3 -m experiments.walk_forward              # §4, §5, §6
python3 -m experiments.walk_forward --data-dir data   # with real files
```

---

## 7. Entropy tail estimation: implemented, measured, and it loses

Ported the Pareto-kernel differential-entropy estimator from Matsushita et al.
(2025) into `src/entropy.py` and ran it head-to-head against the two estimators
already here, on symmetric alpha-stable data where the stability parameter *is*
the tail index. `experiments/tail_bakeoff.py`, 25 reps.

**Recovery of a known alpha** (median estimate, bias):

| true α | entropy | bias | Hill | bias | GPD |
|---|---|---|---|---|---|
| **n=61** | | | | | |
| 0.8 | 0.72 | −0.08 | 0.78 | **−0.02** | fails 100% |
| 1.2 | 1.35 | +0.15 | 1.31 | **+0.11** | fails 100% |
| 1.5 | 1.91 | +0.41 | 1.77 | **+0.27** | fails 100% |
| 1.8 | 5.32 | +3.52 | 2.47 | **+0.67** | fails 100% |
| **n=500** | | | | | |
| 0.8 | 0.67 | −0.13 | 0.79 | **−0.01** | fails 100% |
| 1.2 | 1.06 | −0.14 | 1.21 | **+0.01** | fails 100% |
| 1.5 | 1.68 | **+0.18** | 1.83 | +0.33 | fails 100% |

**Discriminating α=1.5 from α=2.0** (AUC; 0.5 is a coin flip):

| | n=61 | n=500 |
|---|---|---|
| Hill | **0.962** | **1.000** |
| entropy | 0.876 | 0.978 |
| GPD | — | — |

**Verdict: Hill wins.** It is less biased at almost every alpha and discriminates
better at both sample sizes. The entropy estimator's selling point was avoiding
Hill's choice of k -- and that is real, it has no threshold to choose -- but the
plateau rule already in `tailrisk.py` handles k well enough that the tradeoff is
not worth taking. **There is no case for routing it into the sizing engine**, and
it is not wired into anything; it sits beside Hill and GPD as a third opinion.

The GPD column is its own finding: the threshold selector never finds an
admissible threshold on alpha-stable data at these sizes, so it fails outright.
That is honest behaviour rather than a bug -- it declines instead of inventing a
number -- but it means GPD is unusable for windowed tail work.

### A real bug the validation caught

The bandwidth search was originally scaled by the sample standard deviation.
**An alpha-stable law with alpha < 2 has infinite variance**, so that statistic
does not estimate anything -- it just tracks the largest draw -- and the search
range landed in the wrong place. On alpha = 0.8 data the fit pinned alpha-hat at
its 6.0 upper bound, i.e. reported *the heaviest-tailed sample in the suite as
Gaussian*. Fixed by scaling from the interquartile range, which is finite for
every alpha. Pinned by `test_heavy_tails_are_not_reported_as_gaussian`.

### Two predictions of mine that were wrong

Both were arguments I made against the method before measuring, and both were
too pessimistic:

1. **"The centred window is fatal."** The paper computes its label at t from
   `t-30 .. t+30`, i.e. thirty days of future data, and I expected removing that
   to destroy the signal. On a synthetic tape with a genuine embedded regime, the
   trailing window **retains 92%** of the regime separation (3.12 vs 3.38 in
   alpha units). The look-ahead is real and must be removed -- it is, and a test
   pins it -- but it was not doing the work.
2. **"A rolling tail index at n=61 is mostly noise."** At n=61 both estimators
   discriminate α=1.5 from α=2.0 with AUC 0.88-0.96. That is usable. The window
   is short but not hopeless.

### The controls pass

False "heavy-tailed" rate on thin-tailed data: **0% for all three estimators**,
on both iid Gaussian and AR(1) with rho=0.7. The autocorrelation control is the
one worth having -- serially dependent but thin-tailed data is the obvious
false-positive mode for any windowed estimator, and financial returns are
serially dependent in volatility. Nothing spurious fires.

### What this does *not* answer

Whether any of it raises IC. It cannot: a tail estimate feeds position *sizing*,
which moves Sharpe, while IC is the correlation between signal and forward
return. The testable version is conditional IC -- is the signal's IC higher in
light-tailed regimes? -- and that needs a measured base IC first, which does not
exist. Splitting a sample into two regimes also raises the noise floor in each by
roughly sqrt(2), from |IC| ~ 0.016 to ~0.023, so the base edge has to be
comfortably real before slicing it is worth doing.

---

## 8. Run on real data: an underpowered test, not a negative result

The features were finally run on real prices — the 13 converted ETF series,
`experiments/walk_forward.py --data-dir data --min-coverage 0.15`, purged
walk-forward, three folds, 1394 sessions from 2020-07 to 2026-02.

Out-of-sample IC against a **shuffled-timing null measured on this exact panel**
(200 shuffles per feature per horizon):

| feature | h | OOS IC | null 95% | null t95 | clears? |
|---|---|---|---|---|---|
| amihud | 2 | −0.0333 | 0.0670 | 2.13 | no |
| amihud | 10 | −0.0059 | 0.0955 | 2.41 | no |
| abnormal_lambda | 10 | +0.0127 | 0.0620 | 2.08 | no |
| persistent_imbalance | 10 | +0.0312 | 0.0560 | 1.99 | no |
| dollar_volume_anomaly | 10 | +0.0377 | 0.0548 | 1.89 | no |
| footprint_score | 5 | +0.0158 | 0.0615 | 2.10 | no |

**Nothing clears. And the result means almost nothing, because the test has no
power.**

The null floor here is **0.05–0.10**. On the 25-name synthetic panel it was
0.016. The difference is width, not history: the cross-section is only about
**five names wide on a median day**, and the rank correlation of a five-element
cross-section is nearly information-free. The effect being hunted is 0.02–0.04,
so **the floor is two to three times larger than the signal**. This design cannot
distinguish "no edge" from "an edge of exactly the expected size".

Reporting this as "footprint refuted on real data" would be inheriting a
conclusion the data cannot support. It is a failed measurement.

Two things are worth keeping:

* **No feature fired spuriously.** On real prices, with all their gaps and
  discreteness, nothing produced an anomalously large IC. That is a weak but
  genuine construction check that the features are not picking up an artifact of
  their own definition.
* The null t95 came in at **1.77–2.53 against a nominal 1.96**, reproducing the
  Newey-West under-correction on real data rather than synthetic.

One detail to check first when a wider panel exists, worth nothing on its own:
`persistent_imbalance` and `dollar_volume_anomaly` are both *positive* at ten days
(+0.031, +0.038) — the continuation direction assumed in `DEFAULT_SIGNS`, not the
reversal direction the short-term-reversal literature would predict. Both sit
well inside the noise.

**What the test needed and did not have is width.** 1394 sessions is ample
history. Twenty-plus names per day would put the floor near 0.016 and make the
target range detectable. Until then the footprint hypothesis remains untested
rather than refuted.
