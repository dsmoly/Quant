# RAMM — a dealer-market making strategy from two papers

An automated market-making strategy built by joining two papers that turn out to
fit together unusually well:

- **Cartea, Donnelly, Jaimungal — *Algorithmic Trading with Model Uncertainty***
  ([SSRN 2310645](https://ssrn.com/abstract=2310645)). Closed-form robust quotes
  for a market maker who knows her model is wrong.
- **Assayag, Barzykin, Cont, Xiong — *Competition and Learning in Dealer
  Markets*** ([SSRN 4838181](https://ssrn.com/abstract=4838181)). A mean field
  game of competing dealers, and what happens when they all learn.

---

## The idea

The robust paper gives the market maker three **ambiguity budgets**, and its most
useful result is that they move quotes in *opposite* directions:

| budget | what it distrusts | what it does to quotes |
|---|---|---|
| `φ_α` | the midprice drift | **widens** total spread, pulls inventory to flat faster |
| `φ_λ` | the market-order arrival rate | **widens** total spread |
| `φ_κ` | the fill probability of limit orders | **tightens** quotes, increases churn |

The paper treats these as fixed preferences of the market maker — you pick them
once and they express how much you trust each part of your model.

The mean-field paper supplies the state variable that says which one you should
be running. In a dealer market you do not fill against an abstract client; you
fill only when you beat everyone else, so the execution intensity is
`f(δ, μ)` — a function of your quote *and the population mean quote* `μ`. And its
section 4 finding is that a population of homogeneous *learning* dealers does not
converge to the competitive Nash equilibrium. It drifts about **20% wide of it**
("tacit collusion"), with heavy-tailed inventories as the tell. Introducing a
heterogeneous agent pulls the market back toward equilibrium.

That was the starting hypothesis:

> **Solve the mean field game to get the competitive Nash spread. Estimate where
> the population is actually quoting. Set the ambiguity vector from the gap** —
> raise `φ_κ` and undercut when the crowd is wide, raise `φ_α` and get flat when
> it is tight.

---

## What testing did to it

The hypothesis is only worth building if the optimal ambiguity vector genuinely
*depends* on the competitive regime, so that was tested head-on
(`experiments/regime_sweep.py`): pin the market in each regime, sweep a fixed
`(φ_α, φ)` over both, and measure the **value of adaptation** — what an oracle
switching per regime earns minus what the single best fixed vector earns, scored
**out of sample** so grid-search noise cannot masquerade as signal.

**It is refuted.** Every in-sample gain (~3–4% Sharpe) collapses to zero or
negative on held-out paths, at three different levels of flow toxicity. And the
premise had the sign backwards: in the mean-field intensity function a wider
crowd *lowers* your fill elasticity, so the model implies you should widen *with*
the crowd, not undercut it. Quotes there are strategic complements — which is
exactly why a supra-competitive market sustains itself without anyone colluding.

Two things did survive, and they are what the strategy now implements:

1. **The mean-field channel is about the reference model, not a switch.** Fills
   depend on your quote relative to the crowd's, so a market maker pricing off a
   frozen reference model quotes ~2.5× too wide for a competitive market and
   barely trades. Recalibrating `κ` online tracks the crowd implicitly and in the
   right direction. Worth **+50–60% mean P&L**.
2. **The defensive lever is driven by flow toxicity, not competition.** The sweep
   finds the optimal `φ_α` moving sharply with market-order impact (0 at the
   papers' `ε`, at the grid ceiling at 15×) and not at all with the competitive
   regime. So `φ_α` is driven by measured **post-fill markout**.

The regime switch is kept but **disabled by default** (`gap_sensitivity = 0`), so
the refuted hypothesis stays reproducible rather than quietly deleted.

### Headline numbers

Benign flow, at the papers' own calibration (150 paths × 1200s):

| arm | mean P&L | Sharpe | fills |
|---|---|---|---|
| neutral / frozen | 7.93 | 6.93 | 225 |
| robust / frozen | 8.37 | 9.09 | 253 |
| **robust / adaptive-ref** | 12.56 | **9.94** | 660 |
| ramm / adaptive-φ | **13.99** | 8.56 | 841 |

Toxic flow (15× market-order impact) — where the pieces separate:

| arm | mean P&L | Sharpe | mean φ_α |
|---|---|---|---|
| neutral / adaptive-ref | **−2.59** | **−0.56** | |
| robust / adaptive-ref | 3.36 | 2.15 | |
| **ramm / adaptive-φ** | **3.84** | **3.00** | 13.3 |

Adapting the fill model *without* robustness is actively dangerous: it correctly
infers it can win more flow by tightening, and walks into toxic flow. RAMM's
markout channel fires and delivers **+39% Sharpe** over the next best arm.

Full grids, methodology and caveats in [`RESULTS.md`](RESULTS.md).

---

## Layout

```
src/mfg_equilibrium.py   mean field Nash equilibrium (dealer-market paper)
src/robust_quotes.py     robust optimal depths (model-uncertainty paper)
src/market.py            simulator combining both sources of misspecification
src/strategy.py          online estimation + the ambiguity policy (RAMM)
src/backtest.py          Monte Carlo comparison of the arms
experiments/regime_sweep.py   the test of whether adaptation has any value
src/analytics.py         P&L attribution and backtest metrics
experiments/performance_report.py  equity curves + metrics data
experiments/build_report.py        renders the report page
tests/                   85 tests, incl. checks against published figures
```

### `mfg_equilibrium.py`

Solves the coupled system (2.25) — backward Hamilton–Jacobi for `V(t,q)`, forward
Chapman–Kolmogorov for the inventory density `m(t,q)`, fixed point for the mean
quotes — by the paper's Picard scheme, integrated to stationarity.

Two implementation notes. Best responses `Ξ(p, μ) = argmax f(δ,μ)(δ−p)` are found
by the fixed point `δ = p + g(δ)/g'(δ)` from the first-order condition; because
`g/g'` is a convex combination of `1` and `k_m`, the root is bracketed in
`[p + 1/k_m, p + 1]` and this converges in a handful of iterations instead of a
grid search. And the time loop stops on stationarity rather than running a fixed
`T = 3000`.

Reproduces the paper's structure: inventory-skewed quotes, monopolistic quotes
strictly above Nash, inventory density peaked and symmetric at flat.

### `robust_quotes.py`

Optimal depths from Proposition 3:

```
δ⁺*(q) = ( (1/φ_κ)·log(1 + φ_κ/κ) − h_{q−1} + h_q )₊
δ⁻*(q) = ( (1/φ_κ)·log(1 + φ_κ/κ) − h_{q+1} + h_q )₊
η*(q)  = α − φ_α·σ²·q                      (the worst-case drift)
```

**Scope.** The paper gives a closed form for `h` only under Proposition 5's
symmetry conditions, `κ₊ = κ₋` and `φ_λ = φ_κ`. That is exactly what is
implemented, so the two fill-side budgets are tied to a single knob `φ`, and
`φ_α` stays free. This is enough for the strategy, since the two knobs still push
in opposite directions. The general `φ_λ ≠ φ_κ` case would need the penalty
functional `K^{φ_λ,φ_κ}` evaluated numerically and is not implemented.

For a live market maker the relevant object is the *stationary* profile, since
there is no terminal date. Rather than integrating from a distant horizon, note
that `ω(t) = e^{A(T−t)}ω(T)` with `ω = e^{κh}`, so letting the horizon recede
drives `ω` to the dominant eigenvector of `A`. `A` is tridiagonal with positive
off-diagonals, so Perron–Frobenius gives a real dominant eigenvalue with a
positive eigenvector — taken directly. Verified equal to long-horizon integration
to 1e-6, and about 300× faster, which is what makes re-solving inside the live
loop free.

Validated against the paper: Figure 1's depths to within 3e-3, Proposition 6's
symmetry, Proposition 7's sign pattern, and the opposite total-depth responses of
Figures 5 and 7.

### `market.py`

The true dynamics are the robust paper's section 5 — short-term alpha driven by
order flow, bivariate Hawkes arrivals with self- and cross-excitation, and a
jumping mean-reverting fill decay `κ_t` — calibrated so the long-run means are
`λ = 2` and `κ = 27`, the values the market maker's constant reference model uses
and is wrong about at every instant.

Layered on top is mean-field competition: the fill probability is the dealer
paper's intensity function (3.1), with the population quote `μ_t` following an
OU process whose target switches between the Nash level and a supra-competitive
one.

Adverse selection is not imposed, it falls out: an ask fill happens exactly when
a buy market order arrives, and that same order pushes alpha up.

Simulation is exact — all latent processes advance in closed form between orders,
and arrival times come from Ogata thinning against the decaying Hawkes intensity,
so there is no discretisation error.

### Unit bridge

The two papers work in different units. The mean-field paper's monopolistic fill
rate is `e^{−δ}`; the robust paper's is `e^{−κδ}` in price units. They agree
under `δ_price = δ_mfg / κ`, which is how the Nash benchmark is converted for the
strategy.

---

## Running it

```bash
pip install -r requirements.txt

# compare the arms, benign flow then toxic flow
python -m src.backtest --paths 150 --horizon 1200 --eps 0.001
python -m src.backtest --paths 150 --horizon 1200 --eps 0.015

# the test that refuted the regime-switching premise
python -m experiments.regime_sweep --paths 60 --horizon 500 --eps 0.001

python -m pytest tests/ -q                             # 85 tests
```

The mean-field solve takes about 15 seconds and is cached to `.mfg_cache.json`.
Each backtest arm is a few minutes; the sweep is ~5 minutes per toxicity level.

---

## Caveats before anyone trades this

- **It is validated on a simulator, not on data.** Both papers' dynamics are
  implemented faithfully, and the simulator was strong enough to refute the
  headline hypothesis — but it shares assumptions with the strategy, so it cannot
  confirm that the surviving effects exist in a real market. Real fill data is
  the next step.
- **Cover-price feedback is selection-biased in reality.** `μ` is estimated from
  cover on lost trades, which on a live desk is biased toward tighter competitor
  quotes — the trades you lose are the ones where someone was aggressive. The
  simulator does not reproduce that bias. A desk with top-of-book data should
  feed that instead.
- **No latency, quote throttling, tiering, or client segmentation.** Real RFQ
  flow is not exchangeable across clients, and the mean-field abstraction of
  competitors as a single scalar `μ` is a strong one.
- **`φ_λ` and `φ_κ` are tied**, per the scope note above.
- The `φ` grid optimum sits at the edge of the swept range in some cells, so the
  fixed-vector optima quoted should be read as "at least this aggressive", not as
  interior optima.
