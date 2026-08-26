# Signal catalogue

Candidate alphas, chosen for having a **reason not to be arbitraged away**. None
are validated. Each entry states the mechanism, why the mechanism might survive
a crowded market, the data required, and what would kill it.

The organising idea: predictable returns exist for four reasons, and they are
not equally durable.

| Source | Durability | Crowding | Notes |
|---|---|---|---|
| Risk premia | Very high | Very high | Compensation, not error. Your baseline, not your edge. |
| Forced flows | High | **Low** | Counterparty must trade regardless of price. Capacity-limited. |
| Slow diffusion | Medium | Medium | Decays as it is arbitraged, but keeps reappearing. |
| Limits to arbitrage | High | Low | Persists precisely because it is hard to trade. |

Almost everything below is in the middle two rows. That is deliberate: the top
row is where the textbooks are and the bottom of the barrel is where the
alternative-data arms race is, and neither is where a solo researcher wins.

---

## Tier 1 — implemented in `qr.signals.price`

These need only OHLCV and a calendar, so you can test the harness on them
immediately.

### 1. Conditioned turn-of-month rebalancing flow
**Mechanism.** Balanced funds and pensions rebalance to fixed weights at month
end. If equities outperformed bonds during the month, they must *sell* equities
and *buy* bonds, and the size of that trade is proportional to the divergence.

**Why it may not be arbed.** Nearly every published version trades the calendar
unconditionally, which averages months where the flow pointed one way with
months where it pointed the other. The conditioning *is* the edge. Capacity is
small enough that a multi-billion fund cannot express it meaningfully.

**Data.** Returns + a calendar. Available today.

**Kills it.** No relationship between divergence magnitude and subsequent
reversal; or the entire effect sits in a handful of months (check the
distribution, not the mean).

### 2. Intraday-leg reversal
**Mechanism.** Overnight and intraday returns are different animals. Overnight
carries news and most of the equity risk premium; intraday carries liquidity
provision and dealer inventory. Intraday moves are disproportionately
liquidity-driven and the compensation for absorbing them accrues over following
days.

**Why it may not be arbed.** Capturing either leg cleanly requires trading at
both the open and the close — doubling costs and forcing you into the two most
competitive moments of the day. The effect is well documented and *still* hard
to monetise, which is the good kind of persistence: it survives on frictions
rather than on secrecy, and frictions do not get competed away.

**Data.** Open and close prices. Available today.

**Kills it.** Dies at 5bps costs; or the effect is entirely in the smallest
dollar-volume quintile.

### 3. Dispersion-conditioned trend
**Mechanism.** A cross-sectional signal needs cross-sectional variation. When
correlations spike and dispersion collapses there is nothing for a relative
signal to pick up, but the forecast is unchanged and you take the same risk
against a much worse opportunity set.

**Why it may not be arbed.** It is a *conditioning* variable, not a signal. It
changes magnitude rather than direction, so it does not appear in an
unconditional factor regression and screens built on unconditional IC will not
surface it.

**Data.** Returns. Available today.

**Kills it.** Adds nothing once you control for realised volatility — dispersion
and volatility are correlated and only one can be the mechanism.

### 4. Change in Amihud illiquidity
**Mechanism.** The *level* of illiquidity is a well-known priced risk premium.
The *change* is a flow signal: a name becoming harder to trade is a name where
someone is working a large order whose price impact is not yet complete.

**Why it may not be arbed.** The level is in every commercial factor model; the
change is not, because it is noisy per-name and only works in aggregate. It also
needs clean volume data, which is more annoying than it sounds once splits,
halts and consolidated-tape quirks are handled.

**Kills it.** Subsumed by short-term reversal; or the entire effect is in the
bottom liquidity decile where you cannot trade it.

### 5. Idiosyncratic volatility (deliberately included as a control)
Its relationship to forward returns is **known to be non-monotone** — humped,
with both extremes underperforming the middle for different reasons. It is the
natural test case for the one regime where a flexible model beats a linear one,
and a check that your quantile report works. If it comes back monotone,
something in your pipeline is broken.

---

## Tier 2 — higher conviction, needs more data

### 6. Index migration, not index addition
**Mechanism.** Everyone trades S&P 500 additions. Far fewer trade *migrations*:
a stock crossing the Russell 1000/2000 boundary is simultaneously bought by
small-cap funds and sold by large-cap ones, and because it is a small stock
entering a small-cap index, the flow is enormous relative to its float.

**Why it may not be arbed.** The headline event is saturated; the migration is
not. Assembling index membership history is genuinely tedious, and tedium is a
moat.

**Data.** Index membership history, float shares, reconstitution calendars.

**Kills it.** The effect has fully migrated into the pre-announcement window
(check the timing profile, not just the event return).

### 7. Borrow fee *changes*
**Mechanism.** The level of borrow cost is well known (expensive-to-short names
underperform). A *change* is new information arriving at sophisticated short
sellers before it reaches the tape.

**Why it may not be arbed.** Historical borrow data is expensive and
inconsistently formatted across vendors. The signal is also self-limiting: acting
on it means shorting things that are becoming expensive to short.

**Data.** Historical borrow fee / utilisation. Paid.

**Kills it.** Net of the borrow fee itself, nothing is left.

### 8. PEAD conditioned on the reaction pattern
**Mechanism.** Raw post-earnings drift is crowded. But *how* the market reacted
carries information about disagreement: a stock that gaps up then fades intraday
is one where the initial reaction met resistance; gap-and-continue is
conviction. Drift after the two should differ.

**Why it may not be arbed.** Requires joining intraday price paths to earnings
timestamps, and getting the timestamp right (pre-market vs post-close) is
fiddly enough that most implementations skip it.

**Data.** Earnings dates with time-of-day, intraday prices.

**Kills it.** The conditioning adds nothing beyond the raw surprise magnitude.

### 9. ETF NAV dislocation in less-liquid wrappers
**Mechanism.** Authorised participants arbitrage price-vs-NAV gaps, but only
when the gap exceeds their own costs. In fixed income, EM and small commodity
ETFs that band is wide, and dislocations persist and then revert.

**Why it may not be arbed.** Capacity is genuinely small and requires
intraday NAV estimation from the underlying basket. Exactly the size of
opportunity a large fund cannot be bothered with.

**Data.** Intraday NAV estimates or holdings baskets.

**Kills it.** The dislocation is a *correct* price signal about the underlying
basket's staleness rather than a mispricing — check whether the ETF leads or
lags the basket.

### 10. Commodity ETF roll-schedule pressure
**Mechanism.** Roll dates are published in advance. Predictable front-month
selling and next-month buying creates predictable pressure.

**Why it may not be arbed.** Well known in futures; less exploited via the ETF
wrapper, where the roll is mechanical and the holder base is unsophisticated.

**Data.** ETF roll schedules (published), futures curve.

**Kills it.** Fully priced in by the time the roll starts.

---

## Protocol reminder

Reading this file and picking the two that sound most appealing is itself a
selection step, and the trial count will not capture it. Before testing
anything:

1. Write the one-sentence mechanism down first.
2. Pre-specify the test and the kill criteria.
3. Log every variant with `qr.TrialLog`, including the ones you abandon.
4. Deflate against that count.
5. Keep one holdout period you touch exactly once.

At IC ≈ 0.03 you cannot distinguish a real signal from a lucky one by looking
harder at the data. Only the protocol saves you.
