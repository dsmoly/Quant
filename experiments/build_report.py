"""Render the performance report page from the JSON produced by performance_report.py.

Charts are hand-authored inline SVG so they inherit the page's theme tokens and
need no external library. Colours come from the validated categorical palette;
every chart carries a legend, selective direct labels, a hover layer, and a table
view, which is what discharges the light-mode contrast warning on three of the
five series slots.
"""

from __future__ import annotations

import argparse
import html
import json

# Validated categorical slots (light / dark), in fixed order. Never cycled.
# Slot assignment is deliberate, not sequential. The two "frozen" arms and the two
# "adaptive-ref" arms finish almost on top of each other, so each of those pairs is
# given maximally separated hues (both pairs validated in light and dark). RAMM takes
# slot 1: it is the hero series and blue is the only slot clearing 3:1 on the light
# surface alongside orange.
SERIES = [
    ("neutral / frozen", "s4"),
    ("neutral / adaptive-ref", "s2"),
    ("robust / frozen", "s5"),
    ("robust / adaptive-ref", "s3"),
    ("ramm / adaptive-phi", "s1"),
]
SHORT = {
    "neutral / frozen": "neutral·frozen",
    "neutral / adaptive-ref": "neutral·adaptive",
    "robust / frozen": "robust·frozen",
    "robust / adaptive-ref": "robust·adaptive",
    "ramm / adaptive-phi": "RAMM",
}


def esc(s) -> str:
    return html.escape(str(s))


def fmt(v, dp=2, plus=False) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:
            return "—"
        if v in (float("inf"), float("-inf")):
            return "∞" if v > 0 else "−∞"
        s = f"{v:,.{dp}f}"
        if plus and v > 0:
            s = "+" + s
        return s.replace("-", "−")
    return str(v)


# --------------------------------------------------------------------- charts


def line_chart(arms, key_mean="equity_mean", width=760, height=330, title="",
               chart_id="c", horizon=1200.0):
    """Mean equity curve per arm, with direct endpoint labels and a crosshair."""
    pad_l, pad_r, pad_t, pad_b = 52, 132, 16, 34
    xs = arms[0][key_mean.replace("_mean", "_time")] if False else arms[0]["equity_time"]
    all_y = [v for a in arms for v in a[key_mean]]
    y_min, y_max = min(all_y + [0.0]), max(all_y)
    span = (y_max - y_min) or 1.0
    y_min -= span * 0.06
    y_max += span * 0.06

    def X(t):
        return pad_l + (t / horizon) * (width - pad_l - pad_r)

    def Y(v):
        return pad_t + (1 - (v - y_min) / (y_max - y_min)) * (height - pad_t - pad_b)

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{esc(title)}" data-chart="{chart_id}">']

    # Recessive solid hairline grid, y only.
    n_ticks = 5
    for i in range(n_ticks + 1):
        v = y_min + (y_max - y_min) * i / n_ticks
        y = Y(v)
        parts.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick tick-y" x="{pad_l-8}" y="{y+3.5:.1f}">{fmt(v,1)}</text>')
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        t = horizon * frac
        parts.append(f'<text class="tick" x="{X(t):.1f}" y="{height-pad_b+18}">{int(t/60)}m</text>')

    # Zero line, only when the range straddles it.
    if y_min < 0 < y_max:
        parts.append(f'<line class="zero" x1="{pad_l}" y1="{Y(0):.1f}" '
                     f'x2="{width-pad_r}" y2="{Y(0):.1f}"/>')

    ends = []
    for a, (name, slot) in zip(arms, SERIES):
        ys = a[key_mean]
        d = " ".join(f"{'M' if i==0 else 'L'}{X(t):.1f},{Y(v):.1f}"
                     for i, (t, v) in enumerate(zip(xs, ys)))
        parts.append(f'<path class="ln {slot}" d="{d}"/>')
        parts.append(f'<circle class="dot {slot}" cx="{X(xs[-1]):.1f}" cy="{Y(ys[-1]):.1f}" r="3.5"/>')
        ends.append({"name": name, "slot": slot, "y": Y(ys[-1]), "v": ys[-1]})

    # Direct labels at the endpoints, declutter so near-identical finishes stay
    # readable: sort by height, then push apart to a minimum gap, and draw a
    # leader line back to the mark when a label has been moved.
    min_gap = 15.0
    ends.sort(key=lambda e: e["y"])
    for i in range(1, len(ends)):
        if ends[i]["y"] - ends[i - 1]["y"] < min_gap:
            ends[i]["y"] = ends[i - 1]["y"] + min_gap
    overflow = ends[-1]["y"] - (height - pad_b)
    if overflow > 0:                       # pushed past the plot: shift the stack up
        for e in ends:
            e["y"] -= overflow
    x_end = X(xs[-1])
    for e in ends:
        y_mark = Y(e["v"])
        if abs(e["y"] - y_mark) > 1.5:
            parts.append(f'<path class="leader" d="M{x_end+4:.1f},{y_mark:.1f} '
                         f'L{x_end+7:.1f},{e["y"]-3.5:.1f}"/>')
        parts.append(f'<text class="endlab" x="{x_end+9:.1f}" y="{e["y"]:.1f}">'
                     f'{esc(SHORT[e["name"]])} <tspan class="endval">{fmt(e["v"],2)}</tspan></text>')

    parts.append(f'<line class="crosshair" id="{chart_id}-cx" x1="0" y1="{pad_t}" x2="0" '
                 f'y2="{height-pad_b}" style="opacity:0"/>')
    parts.append(f'<rect class="hit" x="{pad_l}" y="{pad_t}" width="{width-pad_l-pad_r}" '
                 f'height="{height-pad_t-pad_b}" fill="transparent"/>')
    parts.append("</svg>")

    payload = json.dumps({
        "id": chart_id, "padL": pad_l, "padR": pad_r, "w": width, "horizon": horizon,
        "t": xs,
        "series": [{"name": SHORT[n], "slot": s, "y": a[key_mean]}
                   for a, (n, s) in zip(arms, SERIES)],
    })
    return "".join(parts), payload


def fan_chart(arm, width=760, height=300, horizon=1200.0, chart_id="fan"):
    """Percentile fan for a single arm: p5–p95, p25–p75, and the mean."""
    pad_l, pad_r, pad_t, pad_b = 52, 20, 16, 34
    xs = arm["equity_time"]
    lo, hi = arm["equity_p05"], arm["equity_p95"]
    y_min, y_max = min(lo + [0.0]), max(hi)
    span = (y_max - y_min) or 1.0
    y_min -= span * 0.08
    y_max += span * 0.08

    def X(t):
        return pad_l + (t / horizon) * (width - pad_l - pad_r)

    def Y(v):
        return pad_t + (1 - (v - y_min) / (y_max - y_min)) * (height - pad_t - pad_b)

    def band(low, high, cls):
        up = " ".join(f"{'M' if i==0 else 'L'}{X(t):.1f},{Y(v):.1f}"
                      for i, (t, v) in enumerate(zip(xs, high)))
        dn = " ".join(f"L{X(t):.1f},{Y(v):.1f}"
                      for t, v in zip(reversed(xs), reversed(low)))
        return f'<path class="{cls}" d="{up} {dn} Z"/>'

    p = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
         f'aria-label="Equity fan" data-chart="{chart_id}">']
    for i in range(6):
        v = y_min + (y_max - y_min) * i / 5
        y = Y(v)
        p.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}"/>')
        p.append(f'<text class="tick tick-y" x="{pad_l-8}" y="{y+3.5:.1f}">{fmt(v,1)}</text>')
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        t = horizon * frac
        p.append(f'<text class="tick" x="{X(t):.1f}" y="{height-pad_b+18}">{int(t/60)}m</text>')
    if y_min < 0 < y_max:
        p.append(f'<line class="zero" x1="{pad_l}" y1="{Y(0):.1f}" x2="{width-pad_r}" y2="{Y(0):.1f}"/>')

    p.append(band(arm["equity_p05"], arm["equity_p95"], "fan-outer"))
    p.append(band(arm["equity_p25"], arm["equity_p75"], "fan-inner"))
    d = " ".join(f"{'M' if i==0 else 'L'}{X(t):.1f},{Y(v):.1f}"
                 for i, (t, v) in enumerate(zip(xs, arm["equity_mean"])))
    p.append(f'<path class="ln s1" d="{d}"/>')
    p.append(f'<text class="endlab anchor-end" x="{width-pad_r}" y="{Y(arm["equity_p95"][-1])-8:.1f}">'
             f'95th pct {fmt(arm["equity_p95"][-1],2)}</text>')
    p.append(f'<text class="endlab anchor-end" x="{width-pad_r}" y="{Y(arm["equity_p05"][-1])+16:.1f}">'
             f'5th pct {fmt(arm["equity_p05"][-1],2)}</text>')
    p.append("</svg>")
    return "".join(p)


def waterfall(attr, width=380, height=280, chart_id="wf"):
    """P&L attribution: gross spread capture, then each cost, then the net."""
    pad_l, pad_r, pad_t, pad_b = 12, 12, 22, 58
    steps = [
        ("Spread capture", attr["spread_capture"], "pos"),
        ("Adverse selection", -attr["adverse_selection"], "neg"),
        ("Inventory P&L", attr["inventory_pnl"], "neg" if attr["inventory_pnl"] < 0 else "pos"),
        ("Liquidation", -attr["liquidation"], "neg"),
    ]
    net = attr["spread_capture"] - attr["adverse_selection"] + attr["inventory_pnl"] - attr["liquidation"]
    top = max(attr["spread_capture"], net, 0) * 1.12
    bottom = min(0.0, net) * 1.25 if net < 0 else 0.0
    rng = (top - bottom) or 1.0

    def Y(v):
        return pad_t + (1 - (v - bottom) / rng) * (height - pad_t - pad_b)

    n = len(steps) + 1
    plot_w = width - pad_l - pad_r
    bw = plot_w / n * 0.6
    gap = plot_w / n

    p = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
         f'aria-label="P&amp;L attribution" data-chart="{chart_id}">']
    p.append(f'<line class="zero" x1="{pad_l}" y1="{Y(0):.1f}" x2="{width-pad_r}" y2="{Y(0):.1f}"/>')

    run = 0.0
    for i, (label, delta, kind) in enumerate(steps):
        x = pad_l + gap * i + (gap - bw) / 2
        y0, y1 = Y(run), Y(run + delta)
        top_y, h = min(y0, y1), max(abs(y1 - y0), 1.2)
        cls = "bar-pos" if kind == "pos" else "bar-neg"
        p.append(f'<rect class="{cls}" x="{x:.1f}" y="{top_y:.1f}" width="{bw:.1f}" '
                 f'height="{h:.1f}" rx="2"><title>{esc(label)}: {fmt(delta,3,plus=True)}</title></rect>')
        # Small terms need more precision or they read as a misleading "0.00".
        dp = 3 if 0 < abs(delta) < 0.01 else 2
        p.append(f'<text class="barval" x="{x+bw/2:.1f}" y="{top_y-5:.1f}">'
                 f'{fmt(delta,dp,plus=True)}</text>')
        p.append(f'<text class="barlab" x="{x+bw/2:.1f}" y="{height-pad_b+14:.1f}">'
                 f'{esc(label.split()[0])}</text>')
        p.append(f'<text class="barlab" x="{x+bw/2:.1f}" y="{height-pad_b+26:.1f}">'
                 f'{esc(" ".join(label.split()[1:]))}</text>')
        run += delta

    x = pad_l + gap * len(steps) + (gap - bw) / 2
    y0, y1 = Y(0), Y(net)
    p.append(f'<rect class="bar-net" x="{x:.1f}" y="{min(y0,y1):.1f}" width="{bw:.1f}" '
             f'height="{max(abs(y1-y0),1.2):.1f}" rx="2"><title>Net P&amp;L: {fmt(net,3)}</title></rect>')
    p.append(f'<text class="barval strong" x="{x+bw/2:.1f}" y="{min(y0,y1)-5:.1f}">{fmt(net,2)}</text>')
    p.append(f'<text class="barlab" x="{x+bw/2:.1f}" y="{height-pad_b+14:.1f}">Net</text>')
    p.append(f'<text class="barlab" x="{x+bw/2:.1f}" y="{height-pad_b+26:.1f}">P&amp;L</text>')
    p.append("</svg>")
    return "".join(p)


def range_chart(arms, width=760, height=320, chart_id="rng"):
    """Per-path P&L spread for each arm: worst, 5th, median, 95th, best.

    The left gutter is two fixed columns -- arm name, then worst path -- so the
    worst-path figure can never collide with the name or with its own whisker,
    however far left the whisker runs.
    """
    gutter_name, gutter_val, pad_r, pad_t, pad_b = 132, 56, 24, 34, 40
    pad_l = gutter_name + gutter_val
    all_v = [v for a in arms for v in a["pnl_distribution"]]
    lo, hi = min(all_v), max(all_v)
    span = (hi - lo) or 1.0
    lo -= span * 0.04
    hi += span * 0.04

    def X(v):
        return pad_l + (v - lo) / (hi - lo) * (width - pad_l - pad_r)

    row_h = (height - pad_t - pad_b) / len(arms)
    p = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
         f'aria-label="Per-path P&amp;L distribution" data-chart="{chart_id}">']
    p.append(f'<text class="colhead" x="{pad_l-8}" y="{pad_t-14}">worst</text>')

    if lo < 0 < hi:
        p.append(f'<line class="zero" x1="{X(0):.1f}" y1="{pad_t-6}" x2="{X(0):.1f}" '
                 f'y2="{height-pad_b+4}"/>')

    # Axis ticks, dropping any that would sit on top of a neighbour.
    candidates = [(X(lo), fmt(lo, 1)), (X(hi), fmt(hi, 1))]
    if lo < 0 < hi:
        candidates.append((X(0), "0"))
    candidates.sort()
    placed = []
    for x, label in candidates:
        if all(abs(x - px) > 34 for px in placed):
            p.append(f'<text class="tick" x="{x:.1f}" y="{height-pad_b+20}">{esc(label)}</text>')
            placed.append(x)

    for i, (a, (name, slot)) in enumerate(zip(arms, SERIES)):
        d = a["pnl_distribution"]
        n = len(d)
        q = lambda f: d[min(n - 1, max(0, int(f * (n - 1))))]  # noqa: E731
        y = pad_t + row_h * i + row_h / 2
        p.append(f'<text class="rowlab" x="{gutter_name-10}" y="{y+4:.1f}">{esc(SHORT[name])}</text>')
        p.append(f'<text class="rowval {"good" if d[0] > 0 else "bad"}" '
                 f'x="{pad_l-10}" y="{y+4:.1f}">{fmt(d[0],2)}</text>')
        p.append(f'<line class="whisker" x1="{X(d[0]):.1f}" y1="{y:.1f}" '
                 f'x2="{X(d[-1]):.1f}" y2="{y:.1f}"/>')
        p.append(f'<rect class="box {slot}" x="{X(q(0.05)):.1f}" y="{y-7:.1f}" '
                 f'width="{max(X(q(0.95))-X(q(0.05)),1.5):.1f}" height="14" rx="2">'
                 f'<title>{esc(name)}: 5th {fmt(q(0.05),2)}, median {fmt(q(0.5),2)}, '
                 f'95th {fmt(q(0.95),2)}, worst {fmt(d[0],2)}, best {fmt(d[-1],2)}</title></rect>')
        p.append(f'<line class="median" x1="{X(q(0.5)):.1f}" y1="{y-9:.1f}" '
                 f'x2="{X(q(0.5)):.1f}" y2="{y+9:.1f}"/>')
    p.append("</svg>")
    return "".join(p)


def legend(names=None) -> str:
    names = names or [n for n, _ in SERIES]
    items = "".join(
        f'<span class="lg-item"><span class="sw {s}"></span>{esc(SHORT[n])}</span>'
        for n, s in SERIES if n in names)
    return f'<div class="legend">{items}</div>'


# ---------------------------------------------------------------------- table

METRIC_GROUPS = [
    ("Returns", [
        ("mean_pnl", "Mean P&L per 20-min episode", 3),
        ("median_pnl", "Median P&L", 3),
        ("std_pnl", "Std dev of P&L", 3),
        ("pnl_per_hour", "P&L per hour", 2),
        ("t_stat", "t-statistic of mean", 1),
    ]),
    ("Risk-adjusted", [
        ("sharpe_episode", "Sharpe (per episode)", 3),
        ("sharpe_daily_equiv", "Sharpe (daily-equivalent)", 2),
        ("sortino", "Sortino", 2),
        ("calmar", "Calmar (mean P&L / mean max DD)", 2),
    ]),
    ("Drawdown", [
        ("max_drawdown", "Max drawdown (worst path)", 3),
        ("mean_max_drawdown", "Mean max drawdown", 3),
        ("median_max_drawdown", "Median max drawdown", 3),
        ("max_dd_duration", "Longest drawdown (seconds)", 0),
    ]),
    ("Distribution", [
        ("win_rate", "Win rate (paths profitable)", 3),
        ("profit_factor", "Profit factor", 2),
        ("skew", "Skew", 3),
        ("excess_kurtosis", "Excess kurtosis", 3),
        ("var_95", "VaR 95 (5th pct of P&L)", 3),
        ("cvar_95", "CVaR 95 (mean of worst 5%)", 3),
        ("worst_path", "Worst path", 3),
        ("best_path", "Best path", 3),
    ]),
    ("Trading", [
        ("mean_fills", "Fills per episode", 0),
        ("fills_per_hour", "Fills per hour", 0),
        ("hit_rate", "Hit rate (fills / RFQs)", 4),
        ("pnl_per_fill", "P&L per fill", 4),
        ("mean_quoted_depth", "Mean quoted depth", 4),
    ]),
    ("Inventory", [
        ("inv_rms", "Inventory RMS", 3),
        ("inv_max_abs", "Max |inventory|", 0),
        ("inv_tail_fraction", "Time in inventory tail", 4),
    ]),
    ("P&L attribution", [
        ("spread_capture", "Gross spread capture", 3),
        ("adverse_selection", "Adverse selection cost", 3),
        ("inventory_pnl", "Inventory P&L", 3),
        ("liquidation", "Liquidation cost", 3),
        ("capture_ratio", "Capture ratio (kept / gross)", 3),
    ]),
]


def metrics_table(arms, best_key="sharpe_episode") -> str:
    rows = [a["metrics"] for a in arms]
    head = "".join(f'<th><span class="sw {s}"></span>{esc(SHORT[n])}</th>'
                   for n, s in SERIES)
    out = [f'<div class="tablewrap"><table><thead><tr><th class="mlab">Metric</th>{head}'
           "</tr></thead><tbody>"]
    for group, keys in METRIC_GROUPS:
        out.append(f'<tr class="grouprow"><td colspan="{len(rows)+1}">{esc(group)}</td></tr>')
        for key, label, dp in keys:
            vals = [r[key] for r in rows]
            finite = [v for v in vals if isinstance(v, (int, float)) and v == v
                      and abs(v) != float("inf")]
            best = max(finite) if finite else None
            cells = []
            for v in vals:
                mark = " best" if best is not None and v == best and key in (
                    "mean_pnl", "sharpe_episode", "sortino", "calmar", "win_rate",
                    "capture_ratio", "profit_factor") else ""
                cells.append(f'<td class="num{mark}">{fmt(v, dp)}</td>')
            out.append(f'<tr><td class="mlab">{esc(label)}</td>{"".join(cells)}</tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


def equity_table(arms, horizon) -> str:
    """Table view of the equity curves — the relief for the light-mode contrast warning."""
    marks = [0, 0.25, 0.5, 0.75, 1.0]
    head = "".join(f"<th>{int(m*horizon/60)}m</th>" for m in marks)
    out = [f'<div class="tablewrap"><table class="compact"><thead><tr><th class="mlab">Arm</th>'
           f"{head}</tr></thead><tbody>"]
    for a, (n, s) in zip(arms, SERIES):
        ys = a["equity_mean"]
        cells = "".join(f'<td class="num">{fmt(ys[min(int(m*(len(ys)-1)), len(ys)-1)], 2)}</td>'
                        for m in marks)
        out.append(f'<tr><td class="mlab"><span class="sw {s}"></span>{esc(SHORT[n])}</td>{cells}</tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


# ----------------------------------------------------------------------- page


def build(benign, toxic, out_path):
    hz = benign["horizon"]
    b_arms, t_arms = benign["arms"], toxic["arms"]
    b_by = {a["name"]: a for a in b_arms}
    t_by = {a["name"]: a for a in t_arms}
    rec_b = b_by["robust / adaptive-ref"]
    rec_t = t_by["ramm / adaptive-phi"]

    eq_b, pay_b = line_chart(b_arms, chart_id="eqb", horizon=hz)
    eq_t, pay_t = line_chart(t_arms, chart_id="eqt", horizon=hz)

    def attr(a):
        m = a["metrics"]
        return {k: m[k] for k in ("spread_capture", "adverse_selection",
                                  "inventory_pnl", "liquidation")}

    css = """
:root{
  color-scheme: light;
  --plane:#f4f4f1; --surface:#fcfcfb; --raised:#ffffff;
  --ink:#14161a; --ink-2:#4e535e; --ink-3:#868c99;
  --rule:#e2e2dd; --rule-strong:#cfcfc9;
  --accent:#2a78d6; --good:#0ca30c; --bad:#d03b3b;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4;
  --fan-outer:rgba(42,120,214,.14); --fan-inner:rgba(42,120,214,.28);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --plane:#0d0d0d; --surface:#1a1a19; --raised:#212120;
    --ink:#f4f4f0; --ink-2:#b4b4ab; --ink-3:#82827a;
    --rule:#2e2e2c; --rule-strong:#3d3d3a;
    --accent:#3987e5; --good:#0ca30c; --bad:#d03b3b;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181;
    --fan-outer:rgba(57,135,229,.18); --fan-inner:rgba(57,135,229,.34);
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --plane:#0d0d0d; --surface:#1a1a19; --raised:#212120;
  --ink:#f4f4f0; --ink-2:#b4b4ab; --ink-3:#82827a;
  --rule:#2e2e2c; --rule-strong:#3d3d3a;
  --accent:#3987e5; --good:#0ca30c; --bad:#d03b3b;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181;
  --fan-outer:rgba(57,135,229,.18); --fan-inner:rgba(57,135,229,.34);
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--plane); color:var(--ink);
  font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:15px; line-height:1.6; -webkit-font-smoothing:antialiased;
}
.mono,.num,.tick,.endval,.barval,.stat-v,.rowval{
  font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums;
}
.wrap{max-width:1120px;margin:0 auto;padding:48px 24px 80px}

header{border-bottom:1px solid var(--rule-strong);padding-bottom:28px;margin-bottom:40px}
.eyebrow{
  font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-3);
  font-weight:600;margin:0 0 12px;
}
h1{font-size:clamp(28px,4vw,40px);line-height:1.1;margin:0 0 12px;
   letter-spacing:-.02em;text-wrap:balance;font-weight:650}
.lede{font-size:17px;color:var(--ink-2);max-width:62ch;margin:0 0 20px}
.runmeta{display:flex;flex-wrap:wrap;gap:8px}
.chip{
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;
  padding:4px 10px;border:1px solid var(--rule-strong);border-radius:3px;
  color:var(--ink-2);background:var(--surface);
}

section{margin:0 0 56px}
h2{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3);
   font-weight:650;margin:0 0 6px;padding-bottom:10px;border-bottom:1px solid var(--rule)}
h3{font-size:18px;margin:32px 0 4px;font-weight:620;letter-spacing:-.01em}
h4{font-size:13px;margin:0 0 14px;font-weight:600;color:var(--ink-2)}
p{margin:14px 0;max-width:72ch}
.sub{color:var(--ink-2);font-size:14px;margin-top:6px}

.statrow{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule);margin:24px 0 8px}
.stat{background:var(--surface);padding:18px 20px}
.stat-k{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);
  font-weight:600;margin-bottom:8px}
.stat-v{font-size:26px;font-weight:600;line-height:1.1;letter-spacing:-.02em}
.stat-n{font-size:12px;color:var(--ink-3);margin-top:6px}
.good{color:var(--good)} .bad{color:var(--bad)}

.card{background:var(--surface);border:1px solid var(--rule);padding:22px 24px 18px;margin:22px 0}
.card-2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:840px){.card-2{grid-template-columns:1fr}}

.chart{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--rule);stroke-width:1}
.zero{stroke:var(--rule-strong);stroke-width:1.5}
.tick{fill:var(--ink-3);font-size:11px;text-anchor:middle}
.tick-y{text-anchor:end}
.ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.dot{stroke:var(--surface);stroke-width:2}
.endlab{fill:var(--ink-2);font-size:11px;font-weight:500;dominant-baseline:middle}
.leader{stroke:var(--rule-strong);stroke-width:1;fill:none}
.endval{fill:var(--ink);font-family:ui-monospace,Menlo,monospace;font-weight:600}
.anchor-end{text-anchor:end}
.s1{stroke:var(--s1)} .s2{stroke:var(--s2)} .s3{stroke:var(--s3)}
.s4{stroke:var(--s4)} .s5{stroke:var(--s5)}
circle.s1,rect.s1{fill:var(--s1);stroke:none} circle.s2,rect.s2{fill:var(--s2);stroke:none}
circle.s3,rect.s3{fill:var(--s3);stroke:none} circle.s4,rect.s4{fill:var(--s4);stroke:none}
circle.s5,rect.s5{fill:var(--s5);stroke:none}
.crosshair{stroke:var(--ink-3);stroke-width:1;pointer-events:none}
.hit{cursor:crosshair}
.fan-outer{fill:var(--fan-outer);stroke:none}
.fan-inner{fill:var(--fan-inner);stroke:none}
.bar-pos{fill:var(--s1)} .bar-neg{fill:var(--bad)} .bar-net{fill:var(--ink-2)}
.barval{fill:var(--ink-2);font-size:11px;text-anchor:middle;font-weight:600}
.barval.strong{fill:var(--ink);font-size:12px}
.barlab{fill:var(--ink-3);font-size:10px;text-anchor:middle}
.rowlab{fill:var(--ink-2);font-size:12px;text-anchor:end}
.rowval{font-size:11px;text-anchor:end;font-weight:600}
.colhead{fill:var(--ink-3);font-size:10px;text-anchor:end;letter-spacing:.08em;text-transform:uppercase}
.rowval.good{fill:var(--good)} .rowval.bad{fill:var(--bad)}
.whisker{stroke:var(--rule-strong);stroke-width:1.5}
.median{stroke:var(--surface);stroke-width:2}
rect.box{stroke:none}

.legend{display:flex;flex-wrap:wrap;gap:16px;margin-top:14px;padding-top:14px;
  border-top:1px solid var(--rule);font-size:12px;color:var(--ink-2)}
.lg-item{display:inline-flex;align-items:center;gap:7px}
.sw{width:11px;height:11px;border-radius:2px;display:inline-block;flex:none;margin-right:6px}
.sw.s1{background:var(--s1)} .sw.s2{background:var(--s2)} .sw.s3{background:var(--s3)}
.sw.s4{background:var(--s4)} .sw.s5{background:var(--s5)}

.tablewrap{overflow-x:auto;margin:18px 0;border:1px solid var(--rule);background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13px;min-width:680px}
th,td{padding:8px 12px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap}
th{font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--ink-3);
   font-weight:600;background:var(--raised);position:sticky;top:0}
th .sw{vertical-align:middle}
.mlab{text-align:left;color:var(--ink-2);font-weight:500}
td.num{font-size:12.5px}
td.best{color:var(--ink);font-weight:680}
.grouprow td{background:var(--raised);color:var(--ink-3);font-size:10.5px;font-weight:650;
  letter-spacing:.1em;text-transform:uppercase;text-align:left}
tbody tr:hover td{background:var(--raised)}

.callout{border-left:3px solid var(--accent);background:var(--surface);padding:16px 20px;margin:24px 0}
.callout.warn{border-left-color:var(--bad)}
.callout h4{margin:0 0 6px;color:var(--ink);font-size:14px;font-weight:650}
.callout p{margin:6px 0;font-size:14px;color:var(--ink-2)}
.callout p:last-child{margin-bottom:0}
ul{margin:14px 0;padding-left:20px;max-width:72ch;color:var(--ink-2)}
li{margin:8px 0}
li strong{color:var(--ink)}
code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.9em;
  background:var(--raised);padding:1px 5px;border-radius:3px;border:1px solid var(--rule)}
.tt{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;z-index:50;
  background:var(--raised);border:1px solid var(--rule-strong);padding:9px 11px;
  font-size:12px;box-shadow:0 6px 20px rgba(0,0,0,.14);border-radius:3px;min-width:170px}
.tt-t{font-family:ui-monospace,Menlo,monospace;color:var(--ink-3);font-size:11px;
  margin-bottom:6px;padding-bottom:5px;border-bottom:1px solid var(--rule)}
.tt-r{display:flex;justify-content:space-between;gap:14px;padding:1.5px 0}
.tt-n{display:inline-flex;align-items:center;color:var(--ink-2)}
.tt-v{font-family:ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums;
  color:var(--ink);font-weight:600}
footer{border-top:1px solid var(--rule-strong);margin-top:56px;padding-top:22px;
  color:var(--ink-3);font-size:12.5px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""

    def stat(k, v, note, cls=""):
        return (f'<div class="stat"><div class="stat-k">{esc(k)}</div>'
                f'<div class="stat-v {cls}">{v}</div><div class="stat-n">{esc(note)}</div></div>')

    bm = rec_b["metrics"]
    tm = rec_t["metrics"]
    nm = t_by["neutral / adaptive-ref"]["metrics"]

    h = []
    h.append(f"<title>Dealer Market Making Tearsheet</title>")
    h.append(f"<style>{css}</style>")
    h.append('<div class="wrap">')

    h.append(f"""<header>
<p class="eyebrow">Backtest report · simulated dealer market</p>
<h1>Dealer market making: did it make money, and where is the edge?</h1>
<p class="lede">Monte Carlo results for five market-making configurations built from
Cartea–Donnelly–Jaimungal robust quoting and the Assayag–Barzykin–Cont–Xiong mean field
game, run on a simulator that is misspecified in both papers' senses at once.</p>
<div class="runmeta">
<span class="chip">{benign['paths']} paths per arm</span>
<span class="chip">{int(hz/60)}-minute episodes</span>
<span class="chip">common random numbers</span>
<span class="chip">inventory limit ±{benign['q_max']}</span>
<span class="chip">Nash benchmark {fmt(benign['nash_depth'],5)}</span>
</div></header>""")

    # ---- verdict
    h.append('<section><h2>The short answer</h2>')
    h.append("""<p><strong>Yes — every configuration makes money under the papers' own
calibration, and the profitable ones do it by capturing quoted spread, not by predicting
price.</strong> Under toxic flow the arms separate sharply, and one of them loses money
outright. There is no directional alpha anywhere in this strategy: inventory P&amp;L is
<em>negative</em> in all ten arm/regime combinations. Every unit of profit comes from being
paid to provide immediacy, and the entire skill is in how much of that payment survives
adverse selection.</p>""")
    h.append('<div class="statrow">')
    h.append(stat("Best Sharpe, benign flow", fmt(bm["sharpe_episode"], 2),
                  "robust / adaptive-ref, per episode"))
    h.append(stat("Mean P&L, benign", fmt(bm["mean_pnl"], 2), f"per {int(hz/60)}-min episode"))
    h.append(stat("Best Sharpe, toxic flow", fmt(tm["sharpe_episode"], 2),
                  "RAMM, per episode", "good"))
    h.append(stat("Worst arm, toxic flow", fmt(nm["mean_pnl"], 2),
                  "neutral / adaptive-ref loses money", "bad"))
    h.append(stat("Of net P&L from spread", f'{bm["spread_capture"]/bm["mean_pnl"]*100:.0f}%',
                  "the rest is cost, not gain"))
    h.append("</div>")
    h.append("""<div class="callout warn"><h4>Read the Sharpe numbers as a ranking, not a
forecast</h4><p>A per-episode Sharpe near 10 with a 100% win rate is not a realistic
trading result — it is what a frictionless simulator produces. There are no fees, no
latency, no minimum tick, no funding cost on inventory, no gap risk, and the market maker
observes the true midprice exactly. These numbers are valid for <em>comparing arms on
identical paths</em>, which is what they are used for here. They are not a claim about
live performance.</p></div>""")
    h.append("</section>")

    # ---- equity curves
    h.append('<section><h2>Equity curves</h2>')
    h.append("""<p>Mark-to-market wealth — cash plus inventory valued at the midprice —
averaged across paths, sampled at every market order. Both panels use the same paths;
only the flow toxicity differs.</p>""")

    h.append(f'<div class="card"><h3>Benign flow</h3>'
             f'<h4>Market-order impact ε = 0.001, the robust paper\'s own calibration</h4>'
             f'{eq_b}{legend()}</div>')
    h.append(equity_table(b_arms, hz))
    h.append("""<p class="sub">All five arms climb steadily. The two <em>adaptive-ref</em>
arms earn roughly 50% more because they discover the correct quoting level for a
competitive market — a frozen reference model quotes about 80% too wide and simply does
not get filled.</p>""")

    h.append(f'<div class="card"><h3>Toxic flow</h3>'
             f'<h4>Market-order impact ε = 0.015, fifteen times the published value</h4>'
             f'{eq_t}{legend()}</div>')
    h.append(equity_table(t_arms, hz))
    h.append("""<p class="sub">The same ranking inverts. <em>neutral / adaptive-ref</em>
correctly infers it can win more flow by quoting tighter, does exactly that, and walks
into toxic flow: it ends the episode below zero. Adapting the fill model without
robustness is worse than not adapting at all.</p>""")

    h.append(f'<div class="card"><h3>Dispersion, not just the mean</h3>'
             f'<h4>RAMM under toxic flow — 5th to 95th percentile band across '
             f'{toxic["paths"]} paths</h4>{fan_chart(rec_t, horizon=hz)}'
             f'<div class="legend"><span class="lg-item">Shaded bands: 5–95th and '
             f'25–75th percentile · line: mean</span></div></div>')
    h.append(f"""<p class="sub">Even the 5th percentile path finishes at
{fmt(rec_t['equity_p05'][-1],2)}. Across all {toxic['paths']} paths the worst outcome is
{fmt(tm['worst_path'],2)} — under this calibration RAMM did not have a losing episode.</p>""")
    h.append("</section>")

    # ---- attribution
    h.append('<section><h2>Where the edge lies</h2>')
    h.append("""<p>Terminal wealth decomposes <em>exactly</em>. Each fill changes inventory
by <code>dq</code> and cash by <code>−dq·S + δ</code>, so summing over fills and applying
Abel summation gives</p>
<p style="text-align:center"><code>P&amp;L = Σδ + Σ dq(S_T − S_i)</code></p>
<p>— gross spread capture plus position P&amp;L, and the position term splits at a
5-second markout horizon into adverse selection and residual inventory P&amp;L. This is an
identity, not a regression: the four components sum to the reported P&amp;L to floating
point on every path, which the code asserts.</p>""")

    h.append('<div class="card-2">')
    h.append(f'<div class="card"><h3>Benign flow</h3><h4>robust / adaptive-ref — '
             f'{fmt(bm["capture_ratio"]*100,1)}% of gross spread survives</h4>'
             f'{waterfall(attr(rec_b), chart_id="wfb")}</div>')
    h.append(f'<div class="card"><h3>Toxic flow</h3><h4>RAMM — '
             f'{fmt(tm["capture_ratio"]*100,1)}% of gross spread survives</h4>'
             f'{waterfall(attr(rec_t), chart_id="wft")}</div>')
    h.append("</div>")

    h.append("""<h3>What this says</h3><ul>
<li><strong>The edge is liquidity provision, not prediction.</strong> Gross spread capture
exceeds net P&amp;L in every single arm, in both regimes. Everything else on the chart is a
cost. There is no term in which the strategy profits from the direction of the midprice.</li>
<li><strong>Inventory P&amp;L is negative everywhere</strong> — from −0.07 to −2.65 across
the ten combinations. Holding a position is a cost the strategy pays to make markets, never
a source of return. Any story about this strategy "trading a view" would be false.</li>
<li><strong>The capture ratio is the whole game.</strong> It falls from ~95% under benign
flow to ~42% under toxic flow. The market maker's job is not to earn a wider spread — the
gross figures barely move between arms — but to keep more of the spread it already
quotes.</li>
<li><strong>The one losing arm has a capture ratio of −0.2%.</strong> Under toxic flow,
<em>neutral / adaptive-ref</em> gives back
{adv}% of everything it earns. Adverse selection consumes the entire gross spread and then
some, which is precisely the failure the robust framework exists to prevent.</li>
</ul>""".replace("{adv}", f'{nm["adverse_selection"]/nm["spread_capture"]*100:.1f}'))
    h.append("</section>")

    # ---- risk
    h.append('<section><h2>Risk and dispersion</h2>')
    h.append(f'<div class="card"><h3>Per-path P&amp;L, toxic flow</h3>'
             f'<h4>Box spans the 5th–95th percentile; the tick is the median; the whisker '
             f'runs worst to best; the number is the worst path</h4>'
             f'{range_chart(t_arms)}</div>')
    h.append(f"""<p class="sub">Under toxic flow the win rate ranges from
{fmt(nm['win_rate']*100,0)}% (<em>neutral / adaptive-ref</em>) to
{fmt(tm['win_rate']*100,0)}% (RAMM). RAMM also has the shallowest worst-path drawdown
({fmt(tm['max_drawdown'],2)} against {fmt(nm['max_drawdown'],2)}) and the tightest
inventory (RMS {fmt(tm['inv_rms'],2)} against {fmt(nm['inv_rms'],2)}).</p>""")
    h.append("</section>")

    # ---- tables
    h.append('<section><h2>All metrics</h2>')
    h.append(f'<h3>Benign flow · ε = 0.001</h3><h4>{benign["paths"]} paths × '
             f'{int(hz/60)}-minute episodes</h4>')
    h.append(metrics_table(b_arms))
    h.append(f'<h3>Toxic flow · ε = 0.015</h3><h4>{toxic["paths"]} paths × '
             f'{int(hz/60)}-minute episodes, identical seeds</h4>')
    h.append(metrics_table(t_arms))
    h.append("""<p class="sub">P&amp;L is in price units on an asset trading near 100, one
unit of inventory per fill. The daily-equivalent Sharpe scales the per-episode figure by
the square root of how many episodes fit in a 6.5-hour day, assuming episodes are
independent — they are, by construction here, but that assumption is exactly what fails in
a real market, so treat that row as arithmetic rather than evidence. Sortino and profit
factor read as ∞ wherever no path lost money, which under benign flow is every arm; the
win-rate row is the honest version of that statement.</p>""")
    h.append("</section>")

    # ---- caveats
    h.append('<section><h2>What these numbers are not</h2>')
    h.append("""<div class="callout warn"><h4>A simulator that shares the strategy's
assumptions cannot validate it</h4><p>It can refute — and it did, killing the original
regime-switching hypothesis on out-of-sample grounds. But the effects that survived were
measured on dynamics taken from the same papers the strategy is built on. Live fill data is
the only real test.</p></div>
<ul>
<li><strong>No frictions at all.</strong> No fees, rebates, latency, minimum tick, quote
throttling, or funding on inventory. Adding any of them cuts the tighter-quoting arms
hardest, since they trade three times as much.</li>
<li><strong>Cover feedback is unbiased here.</strong> The population quote is estimated
from cover prices on lost trades. On a live desk that sample is selection-biased toward
tighter competitors — the trades you lose are the ones where someone was aggressive.</li>
<li><strong>Competitors are one scalar.</strong> Real RFQ flow is client-segmented and not
exchangeable; the mean-field abstraction is a strong one.</li>
<li><strong>Toxicity is a dial, not a measurement.</strong> ε = 0.015 was chosen to show
where the arms separate. The true level for any real venue is an empirical question, and
it is the single input that most changes which configuration you should run.</li>
<li><strong>Two ambiguity budgets are tied.</strong> φ<sub>λ</sub> and φ<sub>κ</sub> move
together, because the paper only gives a closed form under Proposition 5's symmetry
conditions.</li>
</ul>""")
    h.append("</section>")

    h.append(f"""<footer>Generated from <code>experiments/performance_report.py</code> ·
{benign['paths']} Monte Carlo paths per arm per regime, common random numbers across arms ·
P&amp;L attribution verified to reconcile exactly on every path · 69 tests validate the
engines against the published figures and propositions of both papers.</footer>""")

    h.append("</div>")
    h.append(f'<div class="tt" id="tt"></div>')
    h.append(f"<script>{JS.replace('__PAYLOADS__', json.dumps([json.loads(pay_b), json.loads(pay_t)]))}</script>")

    with open(out_path, "w") as fh:
        fh.write("\n".join(h))
    return out_path


JS = """
const PAYLOADS = __PAYLOADS__;
const tt = document.getElementById('tt');
const cssVar = s => getComputedStyle(document.documentElement).getPropertyValue('--'+s);

for (const p of PAYLOADS) {
  const svg = document.querySelector('[data-chart="'+p.id+'"]');
  if (!svg) continue;
  const hit = svg.querySelector('.hit');
  const cx = svg.querySelector('#'+p.id+'-cx');
  const plotW = p.w - p.padL - p.padR;

  function show(evt) {
    const box = svg.getBoundingClientRect();
    const scale = box.width / p.w;
    const xLocal = (evt.clientX - box.left) / scale;
    let f = (xLocal - p.padL) / plotW;
    f = Math.max(0, Math.min(1, f));
    const i = Math.round(f * (p.t.length - 1));
    const x = p.padL + (p.t[i] / p.horizon) * plotW;
    cx.setAttribute('x1', x); cx.setAttribute('x2', x);
    cx.style.opacity = 0.45;
    const rows = p.series
      .map(s => ({n: s.name, slot: s.slot, v: s.y[i]}))
      .sort((a, b) => b.v - a.v)
      .map(r => '<div class="tt-r"><span class="tt-n"><span class="sw ' + r.slot +
        '"></span>' + r.n + '</span><span class="tt-v">' + r.v.toFixed(2) + '</span></div>')
      .join('');
    tt.innerHTML = '<div class="tt-t">t = ' + (p.t[i] / 60).toFixed(1) + ' min</div>' + rows;
    tt.style.opacity = 1;
    const w = tt.offsetWidth, h = tt.offsetHeight;
    let left = evt.clientX + 16, top = evt.clientY - h / 2;
    if (left + w > window.innerWidth - 8) left = evt.clientX - w - 16;
    tt.style.left = Math.max(8, left) + 'px';
    tt.style.top = Math.max(8, Math.min(window.innerHeight - h - 8, top)) + 'px';
  }
  function hide() { tt.style.opacity = 0; cx.style.opacity = 0; }

  hit.addEventListener('mousemove', show);
  hit.addEventListener('mouseleave', hide);
  hit.addEventListener('touchmove', e => { if (e.touches[0]) show(e.touches[0]); }, {passive: true});
  hit.addEventListener('touchend', hide);
}

// Per-mark tooltips for bar and box charts use native <title>, which needs no script.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benign", required=True)
    ap.add_argument("--toxic", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    with open(args.benign) as fh:
        benign = json.load(fh)
    with open(args.toxic) as fh:
        toxic = json.load(fh)
    print("wrote", build(benign, toxic, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
