"""Performance analytics and P&L attribution for the market-making arms.

The attribution is the part that answers "where does the edge come from", and it
is an exact identity rather than a regression.

Write the market maker's terminal wealth as cash plus inventory marked at the
final midprice.  Each fill ``i`` changes inventory by ``dq_i`` (-1 on an ask, +1
on a bid) and changes cash by ``-dq_i * S_i + delta_i``, where ``S_i`` is the
midprice at the fill and ``delta_i`` the depth quoted.  Summing and applying Abel
summation to ``q_T S_T - sum_i dq_i S_i``:

    wealth_T  =  sum_i delta_i  +  sum_i dq_i (S_T - S_i)
                 ^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^
                 spread capture    position P&L

The position term splits at a markout horizon ``h`` into the part that lands
immediately after each fill and the part that accrues afterwards:

    sum_i dq_i (S_{i+h} - S_i)   =  -adverse selection
    sum_i dq_i (S_T - S_{i+h})   =   residual inventory P&L

so

    P&L  =  spread capture  -  adverse selection  +  inventory P&L  -  liquidation

Every term is computed from the realised fill log and midprice path, and the four
must sum to the reported P&L exactly; :func:`decompose_pnl` asserts that.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Attribution:
    """Exact decomposition of one episode's P&L."""

    spread_capture: float       # sum of quoted depths over all fills
    adverse_selection: float    # midprice move against us within the markout horizon
    inventory_pnl: float        # mark-to-market on positions held beyond that horizon
    liquidation: float          # cost of unwinding terminal inventory
    total: float                # must equal the episode's reported P&L
    n_fills: int
    markout_horizon: float

    @property
    def gross_edge_per_fill(self) -> float:
        return self.spread_capture / max(self.n_fills, 1)

    @property
    def adverse_per_fill(self) -> float:
        return self.adverse_selection / max(self.n_fills, 1)

    @property
    def capture_ratio(self) -> float:
        """Share of gross spread capture that survives adverse selection."""
        if self.spread_capture <= 0:
            return float("nan")
        return (self.spread_capture - self.adverse_selection) / self.spread_capture


def decompose_pnl(result, markout_horizon: float = 5.0, tol: float = 1e-8) -> Attribution:
    """Attribute one episode's P&L to spread capture, adverse selection and inventory.

    ``result`` is a :class:`~src.market.RunResult`.  Midprices at ``t_i + h`` are
    read off the event path by taking the last observation at or before that time,
    which is what a desk would do with a trade-and-quote tape.
    """
    fills = result.fills
    S_T = float(result.final_mid)
    times = np.asarray(result.time_path, dtype=float)
    mids = np.asarray(result.mid_path, dtype=float)

    if not fills:
        liq = result.theta * result.terminal_inventory**2
        return Attribution(0.0, 0.0, 0.0, liq, -liq, 0, markout_horizon)

    depth = np.array([f.depth for f in fills], dtype=float)
    t_fill = np.array([f.time for f in fills], dtype=float)
    dq = np.array([-1.0 if f.side == "ask" else 1.0 for f in fills])
    # Midprice at the fill, recovered from the traded price and the depth quoted.
    S_fill = np.array([f.price - f.depth if f.side == "ask" else f.price + f.depth
                       for f in fills], dtype=float)

    # Midprice one markout horizon after each fill: last observation at or before it.
    idx = np.searchsorted(times, t_fill + markout_horizon, side="right") - 1
    S_mark = np.where(idx >= 0, mids[np.clip(idx, 0, len(mids) - 1)], S_fill)
    # Past the end of the episode, mark to the final midprice.
    S_mark = np.where(t_fill + markout_horizon > times[-1], S_T, S_mark)

    spread_capture = float(depth.sum())
    adverse_selection = float(-(dq * (S_mark - S_fill)).sum())
    inventory_pnl = float((dq * (S_T - S_mark)).sum())
    liquidation = float(result.theta * result.terminal_inventory**2)
    total = spread_capture - adverse_selection + inventory_pnl - liquidation

    if abs(total - result.pnl) > tol * max(1.0, abs(result.pnl)):
        raise AssertionError(
            f"attribution does not reconcile: {total!r} vs reported {result.pnl!r}")

    return Attribution(spread_capture, adverse_selection, inventory_pnl, liquidation,
                       total, len(fills), markout_horizon)


@dataclass
class Metrics:
    """Backtest metrics for one arm over a set of Monte Carlo paths."""

    name: str
    n_paths: int
    horizon: float

    # Returns
    mean_pnl: float
    median_pnl: float
    std_pnl: float
    total_pnl: float
    pnl_per_hour: float
    t_stat: float
    p_value_normal: float

    # Risk-adjusted
    sharpe_episode: float          # mean / sd across episodes (the papers' measure)
    sharpe_daily_equiv: float      # scaled to a 6.5h day, assuming iid episodes
    sortino: float
    calmar: float

    # Drawdown, measured within episodes on the mark-to-market equity curve
    max_drawdown: float            # worst across paths
    mean_max_drawdown: float
    median_max_drawdown: float
    max_dd_duration: float         # seconds, worst across paths

    # Distribution
    win_rate: float                # share of paths ending profitable
    profit_factor: float
    skew: float
    excess_kurtosis: float
    var_95: float                  # 5th percentile of path P&L
    cvar_95: float                 # mean of the worst 5%
    worst_path: float
    best_path: float

    # Trading
    mean_fills: float
    fills_per_hour: float
    hit_rate: float                # fills / RFQs seen
    pnl_per_fill: float
    mean_quoted_depth: float

    # Inventory
    inv_rms: float
    inv_max_abs: float
    inv_tail_fraction: float

    # Attribution (per episode)
    spread_capture: float
    adverse_selection: float
    inventory_pnl: float
    liquidation: float
    capture_ratio: float

    def as_dict(self) -> dict:
        return asdict(self)


def _max_drawdown(equity: np.ndarray, times: np.ndarray) -> tuple[float, float]:
    """Maximum peak-to-trough decline of an equity curve, and its duration in seconds."""
    if equity.size == 0:
        return 0.0, 0.0
    running_peak = np.maximum.accumulate(equity)
    drawdown = running_peak - equity
    i = int(np.argmax(drawdown))
    depth = float(drawdown[i])
    if depth <= 0:
        return 0.0, 0.0
    # Walk back to the peak that started this drawdown.
    peak_idx = int(np.argmax(equity[: i + 1]))
    duration = float(times[i] - times[peak_idx]) if times.size == equity.size else 0.0
    return depth, duration


def _normal_two_sided_p(t: float) -> float:
    """Two-sided p-value of a t statistic under a normal approximation.

    Uses the complementary error function so no SciPy dependency is needed.
    """
    from math import erfc, sqrt
    return float(erfc(abs(t) / sqrt(2.0)))


def summarise(name: str, results: list, horizon: float,
              markout_horizon: float = 5.0, trading_day_seconds: float = 6.5 * 3600) -> Metrics:
    """Compute the full metric set for one arm from its per-path results."""
    pnls = np.array([r.pnl for r in results], dtype=float)
    n = pnls.size
    fills = np.array([r.n_fills for r in results], dtype=float)
    rfqs = np.array([r.n_rfq for r in results], dtype=float)

    mean, std = float(pnls.mean()), float(pnls.std(ddof=1))
    sharpe = mean / std if std > 0 else float("nan")
    # Episodes are independent by construction, so Sharpe scales with the square
    # root of how many fit in a trading day.
    episodes_per_day = trading_day_seconds / horizon
    sharpe_daily = sharpe * np.sqrt(episodes_per_day) if np.isfinite(sharpe) else float("nan")

    downside = pnls[pnls < 0]
    downside_dev = float(np.sqrt(np.mean(downside**2))) if downside.size else 0.0
    sortino = mean / downside_dev if downside_dev > 0 else float("inf")

    dds, durs = [], []
    for r in results:
        d, u = _max_drawdown(np.asarray(r.pnl_path, dtype=float),
                             np.asarray(r.time_path, dtype=float))
        dds.append(d)
        durs.append(u)
    dds = np.array(dds)
    mean_dd = float(dds.mean()) if dds.size else 0.0
    calmar = mean / mean_dd if mean_dd > 0 else float("inf")

    gains = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    profit_factor = float(gains / losses) if losses > 0 else float("inf")

    centred = pnls - mean
    skew = float((centred**3).mean() / std**3) if std > 0 else float("nan")
    kurt = float((centred**4).mean() / std**4 - 3.0) if std > 0 else float("nan")

    var95 = float(np.percentile(pnls, 5))
    tail = pnls[pnls <= var95]
    cvar95 = float(tail.mean()) if tail.size else var95

    inv_rms, inv_max, inv_tail, depths = [], [], [], []
    for r in results:
        inv = np.asarray(r.inventory_path, dtype=float)
        if inv.size:
            inv_rms.append(float(np.sqrt(np.mean(inv**2))))
            inv_max.append(float(np.max(np.abs(inv))))
            q_max = max(1.0, float(np.max(np.abs(inv))))
            inv_tail.append(float(np.mean(np.abs(inv) > 0.6 * 8)))
        depths.extend(f.depth for f in r.fills)

    attrs = [decompose_pnl(r, markout_horizon) for r in results]
    spread = float(np.mean([a.spread_capture for a in attrs]))
    adverse = float(np.mean([a.adverse_selection for a in attrs]))
    inv_pnl = float(np.mean([a.inventory_pnl for a in attrs]))
    liq = float(np.mean([a.liquidation for a in attrs]))

    t_stat = mean / (std / np.sqrt(n)) if std > 0 else float("nan")

    return Metrics(
        name=name,
        n_paths=n,
        horizon=horizon,
        mean_pnl=mean,
        median_pnl=float(np.median(pnls)),
        std_pnl=std,
        total_pnl=float(pnls.sum()),
        pnl_per_hour=mean * 3600.0 / horizon,
        t_stat=float(t_stat),
        p_value_normal=_normal_two_sided_p(t_stat) if np.isfinite(t_stat) else float("nan"),
        sharpe_episode=sharpe,
        sharpe_daily_equiv=float(sharpe_daily),
        sortino=float(sortino),
        calmar=float(calmar),
        max_drawdown=float(dds.max()) if dds.size else 0.0,
        mean_max_drawdown=mean_dd,
        median_max_drawdown=float(np.median(dds)) if dds.size else 0.0,
        max_dd_duration=float(np.max(durs)) if durs else 0.0,
        win_rate=float((pnls > 0).mean()),
        profit_factor=profit_factor,
        skew=skew,
        excess_kurtosis=kurt,
        var_95=var95,
        cvar_95=cvar95,
        worst_path=float(pnls.min()),
        best_path=float(pnls.max()),
        mean_fills=float(fills.mean()),
        fills_per_hour=float(fills.mean() * 3600.0 / horizon),
        hit_rate=float(fills.sum() / max(rfqs.sum(), 1.0)),
        pnl_per_fill=mean / max(float(fills.mean()), 1e-9),
        mean_quoted_depth=float(np.mean(depths)) if depths else float("nan"),
        inv_rms=float(np.mean(inv_rms)) if inv_rms else float("nan"),
        inv_max_abs=float(np.max(inv_max)) if inv_max else float("nan"),
        inv_tail_fraction=float(np.mean(inv_tail)) if inv_tail else float("nan"),
        spread_capture=spread,
        adverse_selection=adverse,
        inventory_pnl=inv_pnl,
        liquidation=liq,
        capture_ratio=(spread - adverse) / spread if spread > 0 else float("nan"),
    )


def equity_grid(results: list, horizon: float, n_points: int = 240) -> np.ndarray:
    """Resample every path's equity curve onto a common time grid.

    Returns an array of shape ``(n_paths, n_points)``.  Steps are held between
    events, which is the correct interpolation for a mark-to-market curve
    observed only when something happens.
    """
    grid = np.linspace(0.0, horizon, n_points)
    out = np.zeros((len(results), n_points))
    for i, r in enumerate(results):
        t = np.asarray(r.time_path, dtype=float)
        w = np.asarray(r.pnl_path, dtype=float)
        if t.size == 0:
            continue
        idx = np.searchsorted(t, grid, side="right") - 1
        out[i] = np.where(idx >= 0, w[np.clip(idx, 0, w.size - 1)], 0.0)
    return out
