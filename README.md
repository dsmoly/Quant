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

So the strategy writes itself:

> **Solve the mean field game to get the competitive Nash spread. Estimate where
> the population is actually quoting. Set the ambiguity vector from the gap.**

- Market quoting **wide of Nash** — fat spreads on offer, low marginal risk in
  winning one: raise `φ_κ`. Quote inside the crowd and harvest volume. Be the
  heterogeneous agent.
- Market quoting **at or inside Nash** — thin margins, and every fill is a
  possible adverse selection: raise `φ_α`. Widen, skew hard on inventory, get flat.

Realised volatility and the tail weight of the strategy's own inventory
distribution also feed `φ_α`, because heavy inventory tails are exactly the state
the mean-field paper associates with a dealer being pushed around.

Everything consumed is available to a real desk: own quotes and fills, RFQ
arrival times, midprice moves, and cover prices on lost trades.

---

## Does it work? A partly negative result

**Read this before believing the story above.** The idea is only worth building
if the optimal ambiguity vector genuinely *depends* on the competitive regime. So
that was tested directly, by pinning the market in each regime and sweeping a
fixed `(φ_α, φ)` over both (`experiments/regime_sweep.py`). The decision-relevant
number is the **value of adaptation**: what an oracle that switches per regime
earns, minus what the single best fixed vector earns.

See [`RESULTS.md`](RESULTS.md) for the full grids. The summary:

- **The robust layer works, and reproduces the paper.** Ambiguity aversion lifts
  the Sharpe ratio of the P&L substantially over an ambiguity-neutral market
  maker, which is the robust paper's headline claim.
- **The adaptive layer, at the papers' own calibration, does not earn its
  keep.** With flow toxicity at the published `ε = 0.001`, the two regimes want
  essentially the same ambiguity vector. A supra-competitive market is simply
  *better* for a market maker across the board — it is a level shift, not a
  change in the shape of the trade-off — so there is nothing to switch on.
- The regime signal itself is real and detectable; it is the *response* to it
  that turns out to be near-flat.

The honest reading is that the mean-field paper's contribution to this strategy
is **the benchmark, not the switch**: knowing where Nash sits tells you how much
of the spread you are leaving on the table, and the robust machinery is what you
should be running regardless. The adaptive arm is retained as an implemented,
tested, and *measured* hypothesis, with the measurement reported rather than
buried.

---

## Layout

```
src/mfg_equilibrium.py   mean field Nash equilibrium (dealer-market paper)
src/robust_quotes.py     robust optimal depths (model-uncertainty paper)
src/market.py            simulator combining both sources of misspecification
src/strategy.py          online estimation + the ambiguity policy (RAMM)
src/backtest.py          Monte Carlo comparison of the arms
experiments/regime_sweep.py   the test of whether adaptation has any value
tests/                   64 tests, incl. checks against published figures
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

python -m src.backtest --paths 200 --horizon 1800      # compare the arms
python -m experiments.regime_sweep --paths 60          # test the adaptive premise
python -m pytest tests/ -q                             # 64 tests
```

The mean-field solve takes about 15 seconds and is cached to `.mfg_cache.json`.

---

## Caveats before anyone trades this

- **It is validated on a simulator, not on data.** Both papers' dynamics are
  implemented faithfully, but a simulator that shares the strategy's assumptions
  cannot falsify them. Real fill data is the next step, and the adaptive layer in
  particular should be re-tested there rather than assumed.
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
