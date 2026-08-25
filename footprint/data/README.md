# data/

Drop one Stooq-format daily CSV per ticker here:

    Date,Open,High,Low,Close,Volume
    2015-01-02,111.39,111.44,107.35,109.33,53204626

Filenames may be `aapl.us.txt`, `AAPL.csv`, etc. The exchange suffix is stripped
to form the ticker (`.us`, `.uk`, ...).

## What is currently in here, and why you should not trust it

13 files converted from https://github.com/piekstra/market-data via
`experiments/convert_parquet.py` (5-minute candles aggregated over the regular
session, 2020-07 to 2026-02).

**These are real prices. The volume column is not usable for this project.**

* The upstream provider is hardcoded to Alpaca's `feed=iex`
  (`crates/market-data-providers/src/alpaca.rs`, `("feed", "iex")`). IEX is a
  single venue carrying roughly 2% of US consolidated volume. TQQQ shows ~1.4M
  shares/day against a real 50-100M; FAS shows ~3.3k against millions. The
  shortfall is not a constant multiple -- venue share varies by name and over
  time.

  Every footprint feature is a function of volume: Amihud is |return| / dollar
  volume, abnormal lambda differences that ratio, persistent imbalance weights by
  volume, and the volume anomaly *is* volume. Run on this data they measure IEX
  market-share fluctuation, not institutional footprint.

* **All 18 upstream tickers are leveraged ETFs** (3x sector and index funds, plus
  single-stock leveraged products). The footprint hypothesis is about a fund
  working a parent order in a *company* over days. A 3x ETF's price is
  mechanically pinned to 3x the underlying index's daily return and
  creation/redemption arbitrage absorbs the flow, so there is no institutional
  accumulation to detect. The hypothesis has no referent in this universe.

* Prices are split-adjusted but **not** dividend-adjusted (`adjustment=split`).

What the data *is* good for: exercising the whole pipeline on genuine prices with
real gaps and quirks, and measuring the correlation structure -- returns are
sound even though volume is not, because IEX prices track the NBBO for liquid
names. See the breadth section of ../FINDINGS.md.

To test the strategy properly this directory needs daily OHLCV with
**consolidated** volume for 20+ **individual companies** over several years.
