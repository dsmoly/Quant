# Results

All figures from the simulator in `src/market.py`, which combines the robust
paper's misspecified dynamics (short-term alpha, bivariate Hawkes arrivals,
jumping fill decay) with the mean-field paper's competition. Common random
numbers across arms, so differences are the strategies and not the draws.

Reproduce with:

```bash
python -m src.backtest --paths 150 --horizon 1200 --eps 0.001
python -m src.backtest --paths 150 --horizon 1200 --eps 0.015
python -m experiments.regime_sweep --paths 60 --horizon 500 --eps 0.001
```

`eps` is the market-order impact on the short-term alpha — how toxic the flow is.
`eps = 0.001` is the robust paper's own calibration (Table 1).

---

## 1. The headline hypothesis is refuted

The strategy's original premise was that a market maker should switch her
ambiguity vector on the competitive regime: aggressive when the population quotes
wide of the mean-field Nash level, defensive when it quotes at or inside it.

`experiments/regime_sweep.py` tests this by pinning the market in each regime and
sweeping a fixed `(φ_α, φ)` over a 4×4 grid in both. The decision-relevant number
is the **value of adaptation** — what an oracle switching per regime earns minus
what the single best fixed vector earns. It is reported twice: in-sample (pick
and score on the same paths, biased upward by grid search) and out-of-sample
(pick on the first half of paths, score on the held-out half).

| flow toxicity | metric | in-sample gain | **out-of-sample gain** | MC noise |
|---|---|---|---|---|
| ε = 0.001 | mean P&L | +0.00% | **+0.00%** | ±0.135 |
| ε = 0.001 | Sharpe | +4.22% | **−5.20%** | |
| ε = 0.006 | mean P&L | +0.24% | **−4.57%** | ±0.107 |
| ε = 0.006 | Sharpe | +4.18% | **−0.06%** | |
| ε = 0.015 | mean P&L | +1.81% | **+1.19%** | ±0.084 |
| ε = 0.015 | Sharpe | +3.39% | **+0.00%** | |

**Every in-sample gain collapses out of sample.** The consistent ~3–4% in-sample
Sharpe improvements were grid-search noise over 16 cells; none survives the
split, and two go negative. The one out-of-sample mean-P&L gain that is positive
(+0.022 at ε = 0.015) sits well inside its own ±0.084 Monte Carlo noise.

The mechanism is straightforward in hindsight, and the model says so directly. A
supra-competitive market is uniformly *better* for a market maker — a level shift
in profitability, not a change in the shape of the depth/fill trade-off — so
there is nothing to switch on. Worse, the premise had the sign wrong: in the
mean-field intensity function a wider crowd *lowers* your fill elasticity, so the
model implies you should widen with the crowd, not undercut it. Quotes there are
strategic complements (`tests/test_mfg_equilibrium.py::test_quotes_are_strategic_complements`),
which is precisely why a supra-competitive equilibrium sustains itself.

The regime switch is retained in `AmbiguityPolicy` but **disabled by default**
(`gap_sensitivity = 0`), so the refuted hypothesis stays reproducible.

## 2. What the sweep did find: toxicity, not competition

The same grids show the optimal defensive budget `φ_α` moving sharply with flow
toxicity and not at all with the competitive regime:

| flow toxicity | argmax φ_α (mean P&L) |
|---|---|
| ε = 0.001 | 0 |
| ε = 0.006 | 6–15 |
| ε = 0.015 | 15 (grid ceiling) |

The aggressive budget `φ` wants to be high at every toxicity level, pinned at the
top of the swept range.

This is what `AmbiguityPolicy` now implements: `φ_α` is driven by **measured
post-fill markout** as a share of the half-spread being quoted, with realised
volatility and inventory tail weight as secondary terms. `φ` runs at a high fixed
base.

---

## 3. Backtest, benign flow (ε = 0.001, the papers' calibration)

150 paths × 1200s, inventory limit ±8.

| arm | mean P&L | sd P&L | **Sharpe** | fills | P&L/fill | hit | inv RMS | inv tail |
|---|---|---|---|---|---|---|---|---|
| neutral / frozen | 7.932 | 1.145 | 6.929 | 225 | 0.0353 | 0.047 | 2.32 | 0.044 |
| neutral / adaptive-ref | 12.653 | 1.712 | 7.389 | 668 | 0.0189 | 0.138 | 2.66 | 0.089 |
| robust / frozen | 8.372 | 0.921 | 9.090 | 253 | 0.0331 | 0.053 | 1.27 | 0.001 |
| **robust / adaptive-ref** | 12.555 | 1.264 | **9.937** | 660 | 0.0190 | 0.140 | 1.47 | 0.008 |
| ramm / adaptive-φ | **13.993** | 1.634 | 8.564 | 841 | 0.0166 | 0.173 | 2.04 | 0.035 |

Two clean, roughly orthogonal effects:

- **The robust layer** lifts Sharpe by ~31% with a frozen reference model
  (6.93 → 9.09) and ~34% with an adaptive one (7.39 → 9.94), while cutting
  inventory RMS roughly in half. This reproduces the robust paper's headline
  claim on dynamics it was never fitted to.
- **The mean-field layer, expressed as online recalibration**, lifts mean P&L by
  ~50–60% (7.93 → 12.65, 8.37 → 12.56) and roughly triples fills. A market maker
  pricing off a frozen reference model quotes about 2.5× too wide for this
  competitive market and simply does not trade.

At this calibration **`robust / adaptive-ref` is the best risk-adjusted
configuration.** RAMM's extra aggression buys P&L but costs Sharpe, and its
toxicity channel stays dormant (mean `φ_α` = 1.41) — correctly, because at
ε = 0.001 the flow is not toxic.

## 4. Backtest, toxic flow (ε = 0.015)

Same paths, 15× the market-order impact.

| arm | mean P&L | sd P&L | **Sharpe** | fills | P&L/fill | inv RMS | mean φ_α |
|---|---|---|---|---|---|---|---|
| neutral / frozen | 2.893 | 3.656 | 0.791 | 225 | 0.0129 | 2.32 | |
| neutral / adaptive-ref | **−2.590** | 4.612 | **−0.562** | 668 | −0.0039 | 2.66 | |
| robust / frozen | 2.975 | 2.271 | 1.310 | 253 | 0.0118 | 1.27 | |
| robust / adaptive-ref | 3.363 | 1.562 | 2.153 | 596 | 0.0056 | 0.99 | |
| **ramm / adaptive-φ** | **3.843** | **1.281** | **2.999** | 636 | 0.0060 | **0.89** | 13.34 |

This is where the pieces separate:

- **Online recalibration without robustness is actively dangerous.** The
  `neutral / adaptive-ref` arm loses money (−2.59, Sharpe −0.56). It correctly
  infers that it can win more flow by quoting tighter, and walks straight into
  toxic flow. Adapting the fill model while ignoring model uncertainty is exactly
  the failure the robust paper is about.
- **RAMM's markout channel fires as designed**, taking mean `φ_α` from 1.41 to
  13.34, and it is best on *both* metrics: +14% mean P&L and **+39% Sharpe**
  (2.999 vs 2.153) over the next best arm, with the tightest inventory control of
  any arm tested.

## 5. Recommended configuration

| flow | configuration |
|---|---|
| benign, measured markout below ~25% of half-spread | `robust / adaptive-ref`, φ_α ≈ 6, φ ≈ 4 |
| toxic, or unknown | `AdaptiveRAMM` with the markout-driven policy, `gap_sensitivity = 0` |

Since the markout-driven policy collapses to a low `φ_α` when flow is benign, the
adaptive arm is the safer default of the two if toxicity is not known in advance —
at the cost of ~14% of Sharpe in the benign case.

---

## 6. Validation against the papers

The engines are checked against published results rather than only against
themselves (`tests/`, 69 tests):

| check | result |
|---|---|
| Robust paper, Figure 1 sell depths | match to 3e-3 |
| Robust paper, Proposition 6 (symmetry) | exact |
| Robust paper, Proposition 7 (sign pattern in φ_α) | holds at every inventory |
| Robust paper, Figures 5 / 7 (opposite total-depth responses) | both directions reproduced |
| Robust paper, ξ → λ/e as φ → 0 | exact |
| Stationary eigenvector solve vs long-horizon integration | agree to 1e-6 |
| Mean-field paper, quote skew and monopolistic > Nash | reproduced |
| Mean-field paper, inventory density peaked and symmetric at flat | reproduced |
| Simulator arrival rate vs Hawkes stationary rate | 4.004 vs 4.000 over 12 paths |
| Simulator long-run λ, κ vs Table 1 | 2.0 and 27.0 exactly |

---

## 7. What these numbers are not

They come from a simulator that implements the papers' own dynamics. It can show
that a strategy is internally inconsistent, mis-calibrated, or dominated — and it
did exactly that to the headline hypothesis — but it cannot confirm that the
surviving effects are present in a real market, because the simulator and the
strategy share their assumptions. In particular:

- Cover-price feedback is unbiased here; on a live desk it is selection-biased
  toward tighter competitor quotes, which would degrade the `μ` estimate.
- Competitors are a single scalar `μ`. Real RFQ flow is client-segmented and not
  exchangeable.
- No latency, quote throttling, or minimum tick.
- The `φ` optimum sits at the edge of the swept grid, so the fixed-vector optima
  should be read as "at least this aggressive", not as interior optima.
