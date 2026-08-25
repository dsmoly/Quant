# footprint

Daily cross-sectional US equity strategy on institutional execution footprints.

Separate project from the microstructure work in `../src`. Nothing is shared with
it: no simulator, no order book, no robust quoting. The one thing carried across
is a *derivation* — the closed-form control in `src/sizing.py` — re-derived here
for a daily cross-section rather than imported.

## The idea

An institutional order takes days to work. Each child order can hide in the book,
but the parent cannot hide in the daily record: it shows up as unusual volume
absorbed at unusual impact, repeated across sessions. That footprint lives at
daily frequency, which keeps the strategy out of the latency contest entirely.

## Layout

```
src/loader.py       Stooq CSV -> aligned Panel. The only thing that knows about files.
src/universe.py     point-in-time liquid selection + SURVIVORSHIP_NOTE
src/features.py     Amihud, abnormal lambda, persistent imbalance, volume anomaly
src/sizing.py       the control: 1/sigma targeting, shrinkage, no-trade band, risk budget
src/breadth.py      effective N from the correlation eigenstructure
src/backtest.py     execution lag, cost accounting, engine and quintile arms
src/validation.py   purged walk-forward with embargo
src/tailrisk.py     ES, semivariance, Hill plot, GPD with threshold selection
src/metrics.py      IC, Newey-West t, Sharpe, breakeven cost, deflated Sharpe
src/controls.py     flat, buy-and-hold, inverted, shuffled
src/synthetic.py    controlled panels for experiments -- NOT a data substitute
experiments/        sizing_value_add, noise_floor, walk_forward
tests/              89 tests against recorded fixtures
```

## Data

Drop one Stooq-format file per ticker in `data/`:

```
Date,Open,High,Low,Close,Volume
2015-01-02,111.39,111.44,107.35,109.33,53204626
```

Named `aapl.us.txt`, `AAPL.csv`, or similar — the exchange suffix is stripped.
The loader tolerates what real exports actually contain: an `OpenInt` column,
newest-first ordering, duplicate dates, ragged rows, zero-volume halts, bad
prints. It does **not** forward-fill gaps: a synthetic flat return would read as
"no impact today" and feed straight into the abnormal-lambda signal as a false
liquidity-supply print.

**There is no data in `data/` and none was invented.** This container's egress
policy blocks every market-data host tried. See `FINDINGS.md`.

## Running

```bash
python3 -m pytest tests/ -q

python3 -m experiments.sizing_value_add            # what the sizing engine adds
python3 -m experiments.noise_floor --size-scan     # the bar every IC must clear
python3 -m experiments.walk_forward                # features, controls, costs
python3 -m experiments.walk_forward --data-dir data
```

Every experiment prints a banner when it is running on synthetic data, and the
survivorship note in full.

## Reading the results

`FINDINGS.md`. The short version:

* The sizing engine adds **+1.24 Sharpe (6.3σ)** over naive quintile
  construction on a known-alpha signal, and almost all of it is the no-trade
  band. It refuses to trade a worthless signal (turnover 1.578 → 0.029).
* Finding a real bug: **volatility targeting divides confidence shrinkage back
  out**, because it is scale-invariant. Deployed risk is now proportional to
  confidence.
* For a **dollar-neutral** book, effective breadth is close to N — the severe
  haircut applies to net-exposed books. This narrows the usual claim.
* The **noise floor** for a persistent feature on 25 names × 6 years is
  |IC| ≈ 0.016, and Newey-West under-corrects for feature persistence
  (7–8% false positives at a nominal 5%).
* The **inverted-signal control is vacuous** for any strategy that estimates the
  sign of its own edge, and fails silently.

Nothing here says whether footprint features predict US equity returns. That
needs data.
