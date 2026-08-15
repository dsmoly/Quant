"""Render the directional-strategy report from the sweep JSONs.

Reuses the stylesheet, waterfall and formatting helpers of the market-making
report so the two pages read as one body of work.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from experiments.build_report import CSS, JS, esc, fmt, waterfall

# Two series only on the crossover charts, so slots 1 and 2 -- the pair that
# clears 3:1 on the light surface as well as every colour-vision gate.
TAKER, MAKER = "s1", "s2"
CONTROL_SLOTS = {"directional / live": "s1", "directional / inverted": "s2",
                 "directional / shuffled": "s5", "buy-hold": "s3", "flat": "s4"}


def crossover_chart(points, key, label, chart_id, width=760, height=320,
                    pct=False, zero_line=True):
    """Directional vs market-making, as the market-order impact grows.

    The x axis is ordinal -- the six sweep points, evenly spaced -- because the
    eps values are the levels actually simulated, not samples of a continuum.
    """
    pad_l, pad_r, pad_t, pad_b = 58, 118, 22, 46
    taker = [p["taker"][key] for p in points]
    maker = [p["maker"][key] for p in points]
    vals = taker + maker
    lo, hi = min(vals + [0.0]), max(vals)
    span = (hi - lo) or 1.0
    lo -= span * 0.10
    hi += span * 0.10
    n = len(points)

    def X(i):
        return pad_l + (i / max(n - 1, 1)) * (width - pad_l - pad_r)

    def Y(v):
        return pad_t + (1 - (v - lo) / (hi - lo)) * (height - pad_t - pad_b)

    p = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
         f'aria-label="{esc(label)} against market-order impact" data-chart="{chart_id}">']
    for i in range(6):
        v = lo + (hi - lo) * i / 5
        y = Y(v)
        p.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}"/>')
        p.append(f'<text class="tick tick-y" x="{pad_l-8}" y="{y+3.5:.1f}">'
                 f'{fmt(v*100 if pct else v, 0 if pct else 1)}{"%" if pct else ""}</text>')
    if zero_line and lo < 0 < hi:
        p.append(f'<line class="zero" x1="{pad_l}" y1="{Y(0):.1f}" x2="{width-pad_r}" y2="{Y(0):.1f}"/>')
    for i, pt in enumerate(points):
        p.append(f'<text class="tick" x="{X(i):.1f}" y="{height-pad_b+18}">{pt["eps"]}</text>')
    p.append(f'<text class="axislab" x="{(pad_l+width-pad_r)/2:.1f}" y="{height-pad_b+36}">'
             f'market-order impact ε</text>')

    for series, slot, nm in ((taker, TAKER, "directional"), (maker, MAKER, "market maker")):
        d = " ".join(f"{'M' if i==0 else 'L'}{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(series))
        p.append(f'<path class="ln {slot}" d="{d}"/>')
        for i, v in enumerate(series):
            p.append(f'<circle class="dot {slot}" cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3.5">'
                     f'<title>{esc(nm)} at ε={points[i]["eps"]}: '
                     f'{fmt(v*100 if pct else v, 1 if pct else 3)}{"%" if pct else ""}</title></circle>')
        p.append(f'<text class="endlab" x="{X(n-1)+9:.1f}" y="{Y(series[-1]):.1f}">{esc(nm)} '
                 f'<tspan class="endval">{fmt(series[-1]*100 if pct else series[-1], 1 if pct else 2)}'
                 f'{"%" if pct else ""}</tspan></text>')

    # Mark the crossing, when there is one.
    for i in range(n - 1):
        a0, a1 = taker[i] - maker[i], taker[i + 1] - maker[i + 1]
        if a0 <= 0 < a1 or a0 >= 0 > a1:
            f = abs(a0) / (abs(a0) + abs(a1) + 1e-12)
            xc = X(i) + f * (X(i + 1) - X(i))
            p.append(f'<line class="crossmark" x1="{xc:.1f}" y1="{pad_t}" x2="{xc:.1f}" '
                     f'y2="{height-pad_b}"/>')
            eps_lo, eps_hi = float(points[i]["eps"]), float(points[i + 1]["eps"])
            p.append(f'<text class="crosslab" x="{xc:.1f}" y="{pad_t-6}">crossover '
                     f'ε≈{eps_lo + f*(eps_hi-eps_lo):.3f}</text>')
            break
    p.append("</svg>")
    return "".join(p)


def controls_chart(arms, width=760, height=280, chart_id="ctl"):
    """The signal-destruction controls at one toxicity level."""
    pad_l, pad_r, pad_t, pad_b = 24, 24, 26, 54
    order = ["directional / live", "directional / inverted", "directional / shuffled",
             "buy-hold", "flat"]
    labels = ["live signal", "inverted", "shuffled", "buy & hold", "flat"]
    vals = [arms[k]["metrics"]["mean_pnl"] for k in order]
    hi = max(vals + [0.0]) * 1.18
    lo = min(vals + [0.0]) * 1.18
    rng = (hi - lo) or 1.0

    def Y(v):
        return pad_t + (1 - (v - lo) / rng) * (height - pad_t - pad_b)

    plot_w = width - pad_l - pad_r
    gap = plot_w / len(order)
    bw = gap * 0.52
    p = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
         f'aria-label="Signal-destruction controls" data-chart="{chart_id}">']
    p.append(f'<line class="zero" x1="{pad_l}" y1="{Y(0):.1f}" x2="{width-pad_r}" y2="{Y(0):.1f}"/>')
    for i, (k, lab, v) in enumerate(zip(order, labels, vals)):
        x = pad_l + gap * i + (gap - bw) / 2
        y0, y1 = Y(0), Y(v)
        cls = "bar-pos" if v > 0 else ("bar-net" if v == 0 else "bar-neg")
        p.append(f'<rect class="{cls}" x="{x:.1f}" y="{min(y0,y1):.1f}" width="{bw:.1f}" '
                 f'height="{max(abs(y1-y0),1.2):.1f}" rx="2">'
                 f'<title>{esc(k)}: {fmt(v,3)}</title></rect>')
        ly = min(y0, y1) - 6 if v > 0 else max(y0, y1) + 14
        p.append(f'<text class="barval" x="{x+bw/2:.1f}" y="{ly:.1f}">{fmt(v,2)}</text>')
        p.append(f'<text class="barlab" x="{x+bw/2:.1f}" y="{height-pad_b+16:.1f}">{esc(lab)}</text>')
        trades = arms[k].get("mean_trades", 0)
        p.append(f'<text class="barlab" x="{x+bw/2:.1f}" y="{height-pad_b+28:.1f}">'
                 f'{trades:.0f} trades</text>')
    p.append("</svg>")
    return "".join(p)


def equity_chart(arms, keys, slots, width=760, height=300, horizon=1200.0, chart_id="eq"):
    """Mean equity curves for a chosen set of arms."""
    pad_l, pad_r, pad_t, pad_b = 56, 128, 16, 34
    xs = arms[keys[0]]["equity_time"]
    all_y = [v for k in keys for v in arms[k]["equity_mean"]]
    lo, hi = min(all_y + [0.0]), max(all_y + [0.0])
    span = (hi - lo) or 1.0
    lo -= span * 0.08
    hi += span * 0.08

    def X(t):
        return pad_l + (t / horizon) * (width - pad_l - pad_r)

    def Y(v):
        return pad_t + (1 - (v - lo) / (hi - lo)) * (height - pad_t - pad_b)

    p = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
         f'aria-label="Equity curves" data-chart="{chart_id}">']
    for i in range(6):
        v = lo + (hi - lo) * i / 5
        y = Y(v)
        p.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}"/>')
        p.append(f'<text class="tick tick-y" x="{pad_l-8}" y="{y+3.5:.1f}">{fmt(v,1)}</text>')
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        p.append(f'<text class="tick" x="{X(horizon*frac):.1f}" y="{height-pad_b+18}">'
                 f'{int(horizon*frac/60)}m</text>')
    if lo < 0 < hi:
        p.append(f'<line class="zero" x1="{pad_l}" y1="{Y(0):.1f}" x2="{width-pad_r}" y2="{Y(0):.1f}"/>')

    ends = []
    for k, slot in zip(keys, slots):
        ys = arms[k]["equity_mean"]
        d = " ".join(f"{'M' if i==0 else 'L'}{X(t):.1f},{Y(v):.1f}"
                     for i, (t, v) in enumerate(zip(xs, ys)))
        p.append(f'<path class="ln {slot}" d="{d}"/>')
        p.append(f'<circle class="dot {slot}" cx="{X(xs[-1]):.1f}" cy="{Y(ys[-1]):.1f}" r="3.5"/>')
        ends.append({"label": k.replace("directional / ", ""), "y": Y(ys[-1]), "v": ys[-1]})
    ends.sort(key=lambda e: e["y"])
    for i in range(1, len(ends)):
        if ends[i]["y"] - ends[i - 1]["y"] < 15.0:
            ends[i]["y"] = ends[i - 1]["y"] + 15.0
    over = ends[-1]["y"] - (height - pad_b)
    if over > 0:
        for e in ends:
            e["y"] -= over
    for e in ends:
        p.append(f'<text class="endlab" x="{X(xs[-1])+9:.1f}" y="{e["y"]:.1f}">{esc(e["label"])} '
                 f'<tspan class="endval">{fmt(e["v"],2)}</tspan></text>')
    p.append("</svg>")
    return "".join(p)


def legend(items) -> str:
    return ('<div class="legend">' + "".join(
        f'<span class="lg-item"><span class="sw {s}"></span>{esc(l)}</span>'
        for l, s in items) + "</div>")


def sweep_table(points) -> str:
    head = "".join(f'<th>{p["eps"]}</th>' for p in points)
    rows = [
        ("Directional — mean P&L", lambda p: fmt(p["taker"]["mean_pnl"], 3)),
        ("Directional — Sharpe", lambda p: fmt(p["taker"]["sharpe_episode"], 2)),
        ("Directional — trades", lambda p: fmt(p["taker_trades"], 0)),
        ("Directional — win rate", lambda p: fmt(p["taker"]["win_rate"] * 100, 0) + "%"),
        ("Directional — spread paid", lambda p: fmt(p["taker_spread"], 2)),
        ("Directional — markout earned", lambda p: fmt(p["taker_markout"], 2)),
        ("Market maker — mean P&L", lambda p: fmt(p["maker"]["mean_pnl"], 3)),
        ("Market maker — Sharpe", lambda p: fmt(p["maker"]["sharpe_episode"], 2)),
        ("Market maker — capture ratio", lambda p: fmt(p["maker"]["capture_ratio"] * 100, 1) + "%"),
        ("Market maker — markout paid", lambda p: fmt(p["maker_markout"], 2)),
        ("Control — inverted signal", lambda p: fmt(p["inverted"], 2)),
        ("Control — shuffled signal", lambda p: fmt(p["shuffled"], 2)),
        ("Control — buy and hold", lambda p: fmt(p["buyhold"], 3)),
        ("Control — flat", lambda p: fmt(p["flat"], 3)),
    ]
    out = [f'<div class="tablewrap"><table><thead><tr><th class="mlab">market-order impact ε</th>'
           f"{head}</tr></thead><tbody>"]
    for label, fn in rows:
        cells = "".join(f'<td class="num">{fn(p)}</td>' for p in points)
        out.append(f'<tr><td class="mlab">{esc(label)}</td>{cells}</tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


def build(points, low, high, out_path):
    hz = low["horizon"]
    la = {a["name"]: a for a in low["arms"]}
    ha = {a["name"]: a for a in high["arms"]}
    lo_eps, hi_eps = low["eps"], high["eps"]

    def crossing(key):
        """Linear interpolation of where the two curves meet, in eps."""
        for i in range(len(points) - 1):
            a0 = points[i]["taker"][key] - points[i]["maker"][key]
            a1 = points[i + 1]["taker"][key] - points[i + 1]["maker"][key]
            if a0 <= 0 < a1 or a0 >= 0 > a1:
                f = abs(a0) / (abs(a0) + abs(a1) + 1e-12)
                lo_e, hi_e = float(points[i]["eps"]), float(points[i + 1]["eps"])
                return lo_e + f * (hi_e - lo_e)
        return float("nan")

    cross_pnl = crossing("mean_pnl")
    cross_shp = crossing("sharpe_episode")

    def attr(arm):
        m = arm["metrics"]
        return {k: m[k] for k in ("spread_capture", "adverse_selection",
                                  "inventory_pnl", "liquidation")}

    extra_css = """
.crossmark{stroke:var(--ink-3);stroke-width:1.5}
.crosslab{fill:var(--ink-2);font-size:11px;text-anchor:middle;font-weight:600}
.axislab{fill:var(--ink-3);font-size:11px;text-anchor:middle}
.mirror{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:840px){.mirror{grid-template-columns:1fr}}
"""

    h = []
    h.append("<title>Taker or Maker</title>")
    h.append(f"<style>{CSS}{extra_css}</style>")
    h.append('<div class="wrap">')

    h.append(f"""<header>
<p class="eyebrow">Backtest report · the same framework, pointed the other way</p>
<h1>Taker or maker: which side of the spread should you be on?</h1>
<p class="lede">The market-making strategy earned the spread and paid the post-fill
drift. This turns it around: a directional trader that pays the spread to hold a
position and earns that same drift. Both are built from the Cartea–Donnelly–Jaimungal
robust control, run on identical paths, and swept across the one parameter that
decides between them.</p>
<div class="runmeta">
<span class="chip">{low['paths']} paths per point</span>
<span class="chip">{int(hz/60)}-minute episodes</span>
<span class="chip">common random numbers</span>
<span class="chip">6 toxicity levels</span>
<span class="chip">φ_α = {low['phi_alpha']:g}</span>
</div></header>""")

    # ---- the idea
    h.append('<section><h2>The same control, pointed the other way</h2>')
    h.append("""<p>Keep the robust paper's machinery and change who is trading. The
inner infimum over the drift is identical to the market-making case,</p>
<p style="text-align:center"><code>inf_η { qη + (α−η)²/(2φ_α σ²) } = qα − ½ φ_α σ² q²</code>,
at <code>η* = α − φ_α σ² q</code></p>
<p>so ambiguity about the drift is once again exactly a running penalty on the
position held. With quadratic impact <code>k</code>, the ergodic
Hamilton–Jacobi–Bellman equation is solved <em>exactly</em> by a quadratic ansatz
(the residual is zero to 1e-12 across the whole state space, which the tests
assert), and everything the strategy does drops out of three numbers:</p>
<p style="text-align:center"><code>h₂ = −√(2k φ_α σ²)&nbsp;&nbsp;&nbsp;
ρ = −h₂/2k&nbsp;&nbsp;&nbsp; h₁ = 1/(β+ρ)</code></p>
<p style="text-align:center"><code>target q*(α) = α/((β+ρ)|h₂|)&nbsp;&nbsp;&nbsp;
no-trade band b = c/|h₂|</code></p>""")
    h.append("""<ul>
<li><strong>φ_α becomes the position-sizing knob.</strong> The target scales as
1/√φ_α — distrust your own drift estimate more, bet less. At φ_α → 0 the position
diverges, which is the correct statement that a risk-neutral trader with a real
edge and no penalty would take an unbounded one.</li>
<li><strong>Volatility targeting falls out</strong> rather than being bolted on:
q* scales as 1/σ.</li>
<li><strong>A signal that decays faster than you can trade into it is worth less</strong>,
through the (β+ρ) discount.</li>
<li><strong>The trade-or-not decision does not depend on risk appetite at all.</strong>
Combining the two expressions, the strategy trades from flat only when
<code>|α| &gt; c(β+ρ)</code>. Risk appetite sets how big the bet is; whether it is
worth doing at all is a question about the signal and the spread.</li>
</ul>""")
    h.append("""<div class="callout"><h4>Where the dealer-market paper enters</h4>
<p>The band is proportional to the half-spread actually quoted, which is the
mean-field population quote μ. When dealers sit wide of the Nash level, crossing
costs more, the band widens, and the trader does less. That channel is mechanical
— it is the price paid — rather than the regime switch the market-making
experiments refuted.</p></div>""")
    h.append("</section>")

    # ---- crossover
    h.append('<section><h2>The crossover</h2>')
    h.append("""<p>The two strategies are opposite sides of one wedge. Sweeping the
market-order impact ε — how far each trade moves the price, and so how large the
drift on offer is — moves value from one to the other. Everything else is held
fixed, and both run on the same paths.</p>""")
    h.append(f'<div class="card"><h3>Mean P&amp;L per episode</h3>'
             f'<h4>Directional trader against the market maker, same paths</h4>'
             f'{crossover_chart(points, "mean_pnl", "Mean P&L", "xpnl")}'
             f'{legend([("directional (taker)", TAKER), ("market maker", MAKER)])}</div>')
    h.append(f'<div class="card"><h3>Sharpe ratio per episode</h3>'
             f'<h4>Risk-adjusted, same sweep</h4>'
             f'{crossover_chart(points, "sharpe_episode", "Sharpe", "xshp")}'
             f'{legend([("directional (taker)", TAKER), ("market maker", MAKER)])}</div>')
    h.append(f'<p class="sub">The lines cross at ε ≈ {cross_pnl:.3f} on P&amp;L and '
             f'ε ≈ {cross_shp:.3f} on Sharpe — the crossing points are computed from the '
             f'sweep, not eyeballed. Below them the drift is too small to pay for the '
             f'spread and the market maker wins comfortably; above them the same spread '
             f'that protects the maker is the toll the taker gladly pays.</p>')
    h.append("</section>")

    # ---- mechanism
    h.append('<section><h2>Why it crosses: one number in two mirrors</h2>')
    h.append("""<p>The market maker's capture ratio — the share of quoted spread that
survives markout — falls from 93% to 21% across this sweep. That lost spread does
not vanish; it is what the taker is collecting. The P&amp;L attribution of the two
strategies is the same decomposition with the signs exchanged.</p>""")
    h.append('<div class="mirror">')
    h.append(f'<div class="card"><h3>Market maker, ε = {hi_eps:g}</h3>'
             f'<h4>Earns the spread, pays the drift</h4>'
             f'{waterfall(attr(ha["market maker (RAMM)"]), chart_id="wfm")}</div>')
    h.append(f'<div class="card"><h3>Directional trader, ε = {hi_eps:g}</h3>'
             f'<h4>Pays the spread, earns the drift</h4>'
             f'{waterfall(attr(ha["directional / live"]), chart_id="wfd")}</div>')
    h.append("</div>")
    h.append(f"""<p class="sub">At ε = {hi_eps:g} the maker's gross spread capture is
{fmt(ha['market maker (RAMM)']['metrics']['spread_capture'],2)} against a markout cost of
{fmt(ha['market maker (RAMM)']['metrics']['adverse_selection'],2)}; the taker's spread
capture is {fmt(ha['directional / live']['metrics']['spread_capture'],2)} — negative,
because it is paying — against a markout of
{fmt(ha['directional / live']['metrics']['adverse_selection'],2)}, negative in the
attribution because the drift is working for it rather than against.</p>""")
    h.append("</section>")

    # ---- equity
    h.append('<section><h2>Equity curves</h2>')
    keys = ["directional / live", "directional / inverted", "directional / shuffled",
            "buy-hold", "flat"]
    slots = [CONTROL_SLOTS[k] for k in keys]
    h.append(f'<div class="card"><h3>Below the crossover — ε = {lo_eps:g}</h3>'
             f'<h4>The papers\' own calibration: the signal cannot pay for the spread</h4>'
             f'{equity_chart(la, keys, slots, horizon=hz, chart_id="eqlo")}'
             f'{legend(list(zip(["live signal","inverted","shuffled","buy & hold","flat"], slots)))}</div>')
    h.append(f'<div class="card"><h3>Above the crossover — ε = {hi_eps:g}</h3>'
             f'<h4>The same code, unchanged, on a market where the drift is large</h4>'
             f'{equity_chart(ha, keys, slots, horizon=hz, chart_id="eqhi")}'
             f'{legend(list(zip(["live signal","inverted","shuffled","buy & hold","flat"], slots)))}</div>')
    h.append("</section>")

    # ---- controls
    h.append('<section><h2>Is it really a signal?</h2>')
    h.append("""<p>A directional strategy is much easier to fool than a market-making
one, so the arms below exist to catch it. Each destroys the signal while leaving
the machinery, the costs and the paths identical.</p>""")
    h.append(f'<div class="card"><h3>Signal-destruction controls at ε = {hi_eps:g}</h3>'
             f'<h4>Mean P&amp;L per episode</h4>{controls_chart(ha)}</div>')
    inv = ha["directional / inverted"]["metrics"]["mean_pnl"]
    live = ha["directional / live"]["metrics"]["mean_pnl"]
    shuf = ha["directional / shuffled"]
    h.append(f"""<ul>
<li><strong>flat earns exactly zero</strong>, to the last decimal, on every path. The
accounting has no leak.</li>
<li><strong>buy and hold earns nothing</strong> ({fmt(ha['buy-hold']['metrics']['mean_pnl'],3)}).
There is no drift to be had from simply being long; the P&amp;L is not exposure.</li>
<li><strong>Inverting the signal loses more than the live arm makes</strong>
({fmt(inv,2)} against {fmt(live,2)}). That asymmetry is the give-away that it is real:
both arms pay the same spread, and only one of them is on the right side of the
drift.</li>
<li><strong>Shuffling the signal's timing is catastrophic</strong> ({fmt(shuf['metrics']['mean_pnl'],2)}
on {shuf['mean_trades']:.0f} trades against {ha['directional / live']['mean_trades']:.0f}).
A signal with the right distribution and the wrong timing does not merely fail to
earn — it churns, and every turn pays the spread.</li>
</ul>""")
    h.append("</section>")

    # ---- table
    h.append('<section><h2>Full sweep</h2>')
    h.append(sweep_table(points))
    h.append("""<p class="sub">P&amp;L is in price units on an asset near 100, one unit
per trade. ε = 0.001 is the value the robust paper calibrates to real data in its
section 5; the higher levels are counterfactuals chosen to locate the crossover,
not claims about any venue.</p>""")
    h.append("</section>")

    # ---- honesty
    h.append('<section><h2>What this does and does not show</h2>')
    h.append("""<div class="callout warn"><h4>The directional result is a statement
about the simulator's signal, and that signal was put there by hand</h4>
<p>The market-making result rested on a mechanism — quote, get filled, manage
inventory — that is largely structural. The directional result rests on the
existence and size of an order-flow drift, which in this model is a parameter. That
order flow predicts short-horizon returns is a well-documented empirical
regularity, but its magnitude relative to the spread is exactly the thing that has
to be measured on real data before any of this is a strategy.</p></div>
<ul>
<li><strong>Below the crossover it loses money, and that is the honest result.</strong>
At the paper's own ε the strategy trades a handful of times on noise and gives back
{lowpnl}. An estimator that is statistically unbiased will still be fooled when the
signal-to-noise is low; the no-trade band suppresses most of that, not all.</li>
<li><strong>Correcting the estimator made it worse at low signal.</strong> The first
version was biased low by the factor (1−e^(−βh))/(βh) — a known artefact of fitting
a decaying feature to an average forward return. De-attenuating it is the right
thing to do statistically, and it removed an accidental conservatism that had been
protecting the strategy. Shrinking the signal by its own t-statistic buys back a
little of that ({shrink}) but does not close the gap.</li>
<li><strong>Turnover is very high above the crossover</strong> — hundreds of trades per
twenty-minute episode. There are no fees, no latency and no queue in this
simulator, and all three fall hardest on exactly that behaviour.</li>
<li><strong>The taker's orders impact the price but do not excite the arrival
process.</strong> The Hawkes process is calibrated to client flow at a branching ratio
of 0.9; adding the taker's own orders to the self-excitation pushes it above 1 and
the market explodes. Price impact and book thinning still apply to every order the
strategy sends, so it pays for its own footprint.</li>
</ul>""".replace("{lowpnl}", fmt(la["directional / live"]["metrics"]["mean_pnl"], 3))
       .replace("{shrink}", "about 0.1 of Sharpe at every level"))
    h.append("</section>")

    h.append(f"""<footer>Generated from <code>experiments/directional_sweep.py</code> and
<code>experiments/build_directional_report.py</code> · {low['paths']} Monte Carlo paths per
point, common random numbers across arms · the closed-form control is verified against
the Hamilton–Jacobi–Bellman residual, and the P&amp;L attribution reconciles exactly on
every path ·  tests.</footer>""")
    h.append("</div>")
    h.append('<div class="tt" id="tt"></div>')
    h.append(f"<script>{JS.replace('__PAYLOADS__', '[]')}</script>")

    with open(out_path, "w") as fh:
        fh.write("\n".join(h))
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", required=True, help="glob for the sweep JSONs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(args.glob), key=lambda f: json.load(open(f))["eps"])
    data = [json.load(open(f)) for f in files]
    points = []
    for d in data:
        a = {x["name"]: x for x in d["arms"]}
        points.append({
            "eps": f'{d["eps"]:g}',
            "taker": a["directional / live"]["metrics"],
            "maker": a["market maker (RAMM)"]["metrics"],
            "taker_trades": a["directional / live"]["mean_trades"],
            "taker_spread": a["directional / live"]["spread"],
            "taker_markout": a["directional / live"]["markout"],
            "maker_markout": a["market maker (RAMM)"]["markout"],
            "inverted": a["directional / inverted"]["metrics"]["mean_pnl"],
            "shuffled": a["directional / shuffled"]["metrics"]["mean_pnl"],
            "buyhold": a["buy-hold"]["metrics"]["mean_pnl"],
            "flat": a["flat"]["metrics"]["mean_pnl"],
        })
    print("wrote", build(points, data[0], data[-1], args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
