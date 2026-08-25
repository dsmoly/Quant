# impact

Does order-flow impact decay in seconds, or is part of it permanent?

Built to answer one prior question before any more feature work: a signal
measured only at 1-10 second horizons appears to die almost immediately, but what
decays in seconds is **transient** impact -- the dislocation that mean-reverts as
liquidity replenishes. That is a different quantity from **permanent** impact,
the information component that by construction does not decay. Extrapolating the
first to minutes, hours and days is unsupported.

**No real data has been used.** `data.binance.vision`, `api.exchange.coinbase.com`,
`data.lobsterdata.com` and `stooq.com` are all still refused by this
environment's egress policy (403 to CONNECT). The harness is complete, validated
against tapes with known ground truth, and runs on real zips the moment a session
has access.

## Running

```bash
python3 -m pytest tests/ -q                      # 40 tests

# with network:
python3 -m experiments.fetch --symbols BTCUSDT ETHUSDT SOLUSDT --start 2024-01-01 --end 2024-12-31
python3 -m experiments.decay_curve --data-dir data --symbols BTCUSDT ETHUSDT SOLUSDT

# without: fixture tapes with a planted answer, behind a banner
python3 -m experiments.decay_curve --data-dir tests/fixtures/tapes \
    --symbols PERMUSDT TRANSUSDT --horizons 1 10 60 300 900
```

## The primary estimator

Fitting an asymptote to the response curve does not work, and the failure is
instructive. The impact response **rises, peaks, and then falls through zero**
when impact is purely transient: conditioning on positive flow at *t*, the
baseline price just before the event is already elevated by the transient left by
earlier same-signed flow, and that elevation decays away. A monotone relaxation
model cannot represent a hump; fitting one to the whole curve returned
`R_perm = -1516 bps` with `tau = 1.2 million seconds`.

What works is the estimator the permanent-impact argument itself implies: regress
the price change over a window on the signed flow arriving **in that same
window**, and report the coefficient per *unit* of flow.

| | shape of λ(W) |
|---|---|
| purely transient | decays as 1/W toward zero — nothing accumulates |
| permanent component | converges to a positive constant |

Classification is on the log-log slope (−1 transient, 0 permanent), not on a
long/short ratio — the ratio can never approach 1 because the shortest window is
always dominated by the transient.

**Validated in both directions** on tapes with planted ground truth:

| planted κ_perm | 1s | 10s | 60s | 300s | 600s |
|---|---|---|---|---|---|
| 0.0 | 6.06 | 5.64 | 3.16 | 0.80 | **0.36** |
| 0.3 | 6.87 | 6.57 | 4.13 | 1.78 | **1.33** |
| 0.6 | 7.69 | 7.50 | 5.09 | 2.75 | **2.30** |
| 1.2 | 9.31 | 9.36 | 7.01 | 4.69 | **4.25** |

The excess over the null is exactly linear in the planted coefficient
(0.973/0.3 = 1.946/0.6 = 3.893/1.2 = 3.24). With nothing planted the verdict is
TRANSIENT and the bootstrap interval straddles zero; with enough tape the
planted case reaches PERMANENT with an interval that excludes it.

## Bid-ask bounce

Two corrections to earlier claims, both found by building this.

**bookTicker is not published for spot.** Only `futures/um` and `futures/cm`
carry it. Reconstructing a true mid therefore requires futures; on spot the
analysis is stuck with trade prints. `day_url` refuses the combination rather
than 404-ing later.

**Roll's estimator frequently fails on a real crypto tape**, and not by accident.
Roll assumes serially independent order flow; order flow is long-memory because
large orders are split into many child trades. When impact autocorrelation
exceeds the bounce, the serial covariance turns *positive* and the estimator has
no real solution. On the fixture tapes it recovers a planted spread exactly when
impact is switched off and reports UNUSABLE when it is not. Where a book exists
the bounce is measured directly instead — signed distance from mid to trade
price — which recovers the planted 0.5 bps to three decimals and needs no model.

Bounce must also be measured **at trade level**: one-second bars hold several
trades, the alternation cancels within the bar, and the bar-level covariance goes
positive.

## Layout

```
src/binance.py     bulk download + offline zip reading; ms/us and header variants
src/flow.py        trade signing, bars, as-of mid join, bounce diagnostics
src/response.py    impact / predictive / contemporaneous responses, persistence
src/propagator.py  TIM propagator fit, permanent verdict, block bootstrap
src/nulls.py       shuffled-timing null floors
src/econ.py        bps against costs, crossover, independent bets per year
src/synth_tick.py  tapes with known transient/permanent split -- validation only
```

## Three measurements, kept apart

- **impact** — price measured from just *before* the flow event, so the
  contemporaneous move is included. Its asymptote is permanent impact.
- **predictive** — trailing flow against the *forward* return. What a strategy
  earns, and genuinely different: with a transient component it goes **negative**
  past the relaxation time.
- **contemporaneous** — price change and flow over the same window. Feeds λ(W).

An earlier version computed only the predictive form and reported strongly
negative R(h) on a tape with a large planted *positive* permanent component. That
was arithmetically correct and the wrong measurement.

## Breadth

`independent_obs_per_year` comes from the signal's integrated correlation time,
not the rebalance frequency. The experiment prints both and the inflation factor
between them — on the fixture tapes, counting bets by rebalance frequency
overstates Sharpe by 1.3x to 3.6x depending on horizon. For power-law flow the
AR(1) half-life badly understates persistence, so both are reported.

## Nulls

Shuffled timing, never inversion — inversion is invariant for a self-calibrating
estimator and here would simply flip the response exactly. The shuffle is a
**circular block** shuffle with blocks longer than the flow's correlation time,
so the long memory survives and only the alignment to price is destroyed. A plain
permutation would break the memory and give a null far too tight.

On the fixture tapes the empirical 95th-percentile |t| under the null runs 1.54
to 2.41 against a nominal 1.96 — the same lesson as the equity work, where the
floor sat near |IC| 0.016 rather than the nominal 0.010.

## What this cannot tell you yet

Whether BTCUSDT has a permanent impact component at minute-to-day horizons. The
verdicts printed on fixtures are planted answers being recovered.

Two things worth deciding before fetching: a year rather than three months (73
independent 5-day windows versus 18 — 18 resolves nothing at that end of the
curve), and futures rather than spot, so the mid is real.
