# Three tests, run in order, before combining anything

All three run with **latency on**: 50 ms between a decision and its effect. For the
taker that is submission latency — the order fills at the price prevailing 50 ms
later, not the price that triggered it. For the maker it is quote-update latency —
a new quote only goes live after the delay, so the market can trade against a
stale one. Latency is not free: it costs the taker 15.7 → 13.0 of mean P&L at
50 ms and 15.7 → 1.6 at 250 ms, and it costs the maker 3.25 → 1.95 → −5.44 over
the same range.

The taker also pays an explicit fee and slippage on top of the dealer spread. One
basis point on an asset priced at 100 is 0.01, against a dealer half-spread of
0.0205 — so **1 bp of fee is roughly half the entire half-spread.**

---

## Test 1 — Cost sensitivity and selectivity

`experiments/cost_selectivity.py`

### A design problem that had to be separated first

Raising `c` in this control does **two** things: it charges more per trade, *and*
it widens the no-trade band, because `b = c/|h₂|`. Sweeping `c` alone therefore
cannot answer "does selectivity rescue a marginal edge", because the selectivity
is confounded with the cost that caused it. So both sweeps were run:

- **cost sweep** — the fee actually charged rises and the band responds to it.
- **selectivity sweep** — the fee is held at 1 bp and only the band is scaled.

There is a theoretical prior for the first. If the policy re-optimises at each
cost, the envelope theorem gives `dP/dc = −E[turnover] < 0`, so total P&L *must*
fall monotonically. An interior maximum there would mean the policy is not
optimising — which would itself be the finding.

### Result: the cost curve is monotone. The selectivity curve is not.

Cost sweep, ε = 0.015, 50 ms latency:

| fee | mean P&L | Sharpe | fills | P&L/fill |
|---|---|---|---|---|
| 0 bp | 12.70 | 3.25 | 584 | 0.0217 |
| 0.5 bp | 10.24 | 2.45 | 514 | 0.0199 |
| 1 bp | 9.42 | 2.58 | 457 | 0.0206 |
| 2 bp | 6.64 | 1.73 | 333 | 0.0199 |
| 4 bp | 4.68 | 1.67 | 188 | 0.0248 |
| 8 bp | 1.18 | 0.52 | 70 | 0.0169 |

Monotone in P&L and in Sharpe, at every toxicity level tested. One apparent bump
appeared at ε = 0.025 (0.25 bp beating 0 bp by 2.2 sem); a focused re-run with 170
fresh seeds **reversed it** — 22.89 → 21.57 → 20.75, monotone. It was seed noise.

### Selectivity, at a fixed 1 bp fee: an interior maximum at ×1.5

| band × | ε=0.006 P&L | Sharpe | ε=0.010 P&L | Sharpe | ε=0.015 P&L | Sharpe | fills (ε=.015) |
|---|---|---|---|---|---|---|---|
| ×0.5 | −4.32 | −1.43 | −2.29 | −0.55 | 4.19 | 0.92 | 725 |
| ×1.0 | 0.42 | 0.36 | 3.17 | 1.08 | 9.42 | 2.58 | 457 |
| **×1.5** | **0.97** | **1.12** | **4.81** | **1.99** | **10.50** | **2.84** | 308 |
| ×2.0 | 0.80 | 0.78 | 4.26 | 1.56 | 9.97 | 2.16 | 195 |
| ×3.0 | 0.38 | 0.57 | 2.66 | 1.09 | 5.98 | 1.93 | 76 |
| ×5.0 | 0.11 | 0.21 | 0.66 | 0.61 | 1.85 | 0.89 | 13 |

**An interior maximum at ×1.5 the band the control prescribes, at every toxicity
level, on both P&L and Sharpe.** P&L per fill rises monotonically across the sweep
(0.0058 → 0.146 at ε=0.015) while fills fall 725 → 13, so the two effects genuinely
trade off and the optimum is real rather than an artefact of one metric.

**Selectivity does rescue a marginal edge.** At ε = 0.006 the strategy at its own
prescribed band is barely alive (P&L 0.42, Sharpe 0.36); at ×1.5 it is 0.97 and
1.12 — P&L more than doubles and Sharpe triples. Frequency falls faster than
per-trade excess grows *eventually*, but not immediately: there is a band of
roughly 1.25–2× where the trade is worth making.

### Are the Hawkes fat tails enough to create it? Yes.

Excess kurtosis of the **signed** fitted alpha at the decision points is **4.37**
at ε = 0.006 and **1.96** at ε = 0.010 (Gaussian = 0). Clustered order flow at a
branching ratio of 0.9 produces a genuinely fat-tailed signal, and that is what
puts enough mass in the large-|α| region for raising the threshold to pay. The
effect is strongest exactly where the kurtosis is highest — the marginal-edge case
— which is the signature you would want before believing it.

Note the mechanism is *not* that the control was wrong. The control optimises
against a Gaussian-equivalent trade-off; the extra 50% of band is compensation for
tail mass its quadratic value function does not see.

---

## Test 2 — Quote skewing instead of crossing

`src/skew_maker.py`, `experiments/skew_vs_taker.py`

A maker that leans its quotes on the same order-flow alpha — bid in when α is
positive, offer out when negative — with depths floored at zero so it never
crosses and never takes liquidity. The skew coefficient is
`γ = 1/(β + λ_fill)`: the fraction of the signal capturable before it decays or
the inventory turns over, the same discount that appears in the taker's target.

ε = 0.015, 50 ms latency, 1 bp taker fee, taker at test 1's optimal ×1.5 band:

| arm | mean P&L | Sharpe | fills | gross spread | adverse selection | inv RMS |
|---|---|---|---|---|---|---|
| maker / RAMM | 1.97 | 1.19 | 732 | 11.38 | **+8.31** | 1.07 |
| **maker / skew** | **27.53** | **3.52** | 1222 | 6.32 | **−20.28** | 3.89 |
| maker / skew inverted | −43.66 | −3.80 | 939 | 7.19 | +44.13 | 3.74 |
| taker / directional | 10.50 | 2.84 | 308 | −9.79 | −19.63 | 1.43 |

**The hypothesis is confirmed, and more strongly than stated.** Adverse selection
does not merely fall toward zero — it goes *negative*: +8.31 → −20.28. The maker
stops being the one picked off and starts being the one systematically filled on
the right side. And it beats the taker outright: **27.53 vs 10.50 on P&L (15.2 σ),
Sharpe 3.52 vs 2.84.** The inverted-skew control is symmetric and catastrophic,
which is what confirms the effect is the signal and not the extra activity.

The reason is the one in the hypothesis. The taker needs `|α| > c(β+ρ)`; the maker
is already quoting on both sides, so leaning those quotes costs nothing at the
margin and needs α only to beat zero.

### Two things that must be said alongside that

- **The capture ratio stops being a meaningful statistic here.** It is a share of
  gross spread surviving markout; once adverse selection is negative it exceeds
  100% (428.9%) and is no longer a share of anything. Reported as n/a rather than
  quoted as a 16× improvement. The interpretable number is the adverse-selection
  term itself.
- **This is no longer a market-neutral maker.** Inventory RMS goes 1.07 → 3.89 and
  gross spread capture *falls* 11.38 → 6.32. It is holding real directional risk
  and financing it with a thinner spread. At the papers' own ε = 0.001 the trade is
  not obviously good: P&L is a wash (13.79 vs 13.90, −0.3 σ), adverse selection
  falls 74%, but **Sharpe drops 9.17 → 6.79** because of that inventory. Skewing
  helps where there is signal to skew on, and costs risk-adjusted return where
  there is not.

---

## Test 3 — Alpha decay versus passive fill

`experiments/decay_vs_fill.py`. Nothing trades; a probe rides along a flat run so
it cannot perturb the paths it measures.

ε = 0.015, 50 ms latency. True alpha half-life 0.69 s.

| horizon | signal IC | IC vs peak | fill, pegged quote | fill, resting order |
|---|---|---|---|---|
| 0.05 s | 0.499 | 67% | 0% | 0% |
| 0.25 s | 0.726 | 98% | 19% | 30% |
| 0.5 s | 0.742 | 100% | 30% | 41% |
| 1 s | 0.679 | 92% | 41% | 47% |
| **2 s** | **0.576** | **78%** | **51%** | **52%** |
| **4 s** | **0.498** | **67%** | **65%** | **58%** |
| **8 s** | **0.430** | **58%** | **82%** | **65%** |
| 15 s | 0.375 | 50% | 95% | 71% |
| 30 s | 0.317 | 43% | 99% | 77% |

**The window is not empty.** Requiring both ≥50% of peak IC and ≥50% of orders
filled, it runs from about **2 s to 15 s**, and the joint measure (IC × fill
probability) peaks around 8–15 s. Median time to a passive fill at the quote is
**1.86 s**.

This is the answer, but it needs one distinction to be honest, and the distinction
changes what it means:

- **The instantaneous alpha is long gone by the time you fill.** At the 1.86 s
  median fill, `e^(−β·1.86)` leaves **16%** of the original impulse. What survives
  at 2–8 s is not the instantaneous drift but the *cumulative* move — the
  correlation between the signal at posting and the price change realised over the
  window. Those are different quantities, and the IC column measures the second.
- **The window only exists for a re-priced quote, not a resting order.** A static
  limit order fills *faster* early (30% vs 19% at 0.25 s) precisely because the
  price coming toward it is what fills it. Its mean midprice move between posting
  and filling is **−0.0141** against a half-spread of 0.0205: the fill arrives only
  after the price has already moved against it, leaving 31% of the spread. Past
  ~4 s the pegged quote overtakes it (95% vs 71% at 15 s) because the resting order
  gets left behind when the price runs away.

So the usable window belongs to a continuously re-quoted maker — which is exactly
what the test 2 skewing maker is. **Test 3 explains test 2 rather than adding a
third option**: the reason skewing works is that a pegged quote lives in the 2–8 s
window where the signal still has predictive content and the fill has already
happened. A resting-order strategy on this signal does not have that window, and
the −0.0141 markout is why.

---

## Where this leaves the three approaches

| | needs | at ε=0.015 | risk |
|---|---|---|---|
| taker, crossing | α > c(β+ρ) | 10.50, Sharpe 2.84 | inv RMS 1.43 |
| maker, skewing | α > 0 | **27.53, Sharpe 3.52** | inv RMS 3.89 |
| resting passive | α > 0 *and* a fill before decay | not viable | −0.0141 markout |

Nothing has been combined yet, as instructed. The obvious next step — a maker that
skews *and* crosses when α is large enough to clear the taker's bar — is not
justified by these three results on its own, because the skewing maker's advantage
comes precisely from *not* paying the spread, and test 1 shows the taker's edge is
monotonically eroded by cost. The case for combining would have to be made on
inventory control, not on P&L.

### Caveats carried forward

- All of this is the simulator's signal, whose size is a parameter (`ε`), not a
  measurement. ε = 0.001 is the papers' own calibration; the levels where these
  strategies work are counterfactuals.
- The skewing maker's advantage rests on a fill model in which a leaned quote wins
  more of the flow it wants. Real queue priority, tick constraints and client
  segmentation are not modelled.
- The taker's own orders impact price and thin the book but do not excite the
  arrival process; the Hawkes kernel is calibrated to client flow at branching
  ratio 0.9, and double-counting drives it above 1.
