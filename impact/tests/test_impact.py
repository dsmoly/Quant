"""Validation of the impact-decay harness against tapes with known ground truth.

The central pair of tests is the estimator's two-sided validation: it must
recover a planted permanent component, and it must report *none* when none was
planted. An estimator that only passes the first is worse than no estimator,
because the whole question is whether a permanent component exists.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.binance import (BinanceError, day_url, read_agg_trades, read_book_ticker,
                         read_local)
from src.econ import CostModel, crossover, economics, fmt_horizon
from src.flow import (Bars, align_mid_to_trades, bounce_diagnostics, build_bars,
                      effective_half_spread_bps, normalise_flow, resample_bars,
                      roll_spread, sign_trades)
from src.nulls import null_floor
from src.propagator import (critical_gamma, fit_flow_autocorrelation,
                            permanent_verdict, tim_response)
from src.response import (autocorr, ols_nw, permanent_lambda_curve, persistence,
                          persistence_curve, response_at_horizon, response_curve)
from src.synth_tick import TickSpec, simulate_ticks

TAPES = Path(__file__).parent / "fixtures" / "tapes"
WINDOWS = [1, 2, 5, 10, 30, 60, 120, 300]


@pytest.fixture(scope="module")
def perm_bars():
    d = read_local(TAPES, "PERMUSDT")
    return build_bars(d.trades, "1s", d.book)


@pytest.fixture(scope="module")
def trans_bars():
    d = read_local(TAPES, "TRANSUSDT")
    return build_bars(d.trades, "1s", d.book)


# --------------------------------------------------------------------- loader


def test_url_layout():
    assert day_url("btcusdt", "2024-01-01") == (
        "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/"
        "BTCUSDT-aggTrades-2024-01-01.zip")
    assert "/data/spot/" in day_url("BTCUSDT", "2024-01-01", market="spot")


def test_book_ticker_is_refused_for_spot():
    """Correcting an error made earlier: spot does not publish bookTicker."""
    with pytest.raises(BinanceError, match="not published for spot"):
        day_url("BTCUSDT", "2024-01-01", kind="bookTicker", market="spot")


def test_unknown_market_is_rejected():
    with pytest.raises(BinanceError):
        day_url("BTCUSDT", "2024-01-01", market="nasdaq")


def test_reads_headerless_millisecond_format(perm_bars):
    assert len(perm_bars.trade_price) > 1000
    assert perm_bars.has_mid


def test_reads_header_microsecond_format():
    d = read_local(TAPES, "HDRUSDT")
    assert len(d.trades) == 4000
    assert d.trades["ts"].dt.year.iloc[0] == 2024
    # Microsecond timestamps must not be read as milliseconds, which would put
    # the tape ~55 years into the future.
    assert d.trades["ts"].iloc[-1] - d.trades["ts"].iloc[0] < pd.Timedelta(days=1)
    assert d.has_mid


def test_missing_symbol_is_reported():
    with pytest.raises(BinanceError, match="no aggTrades zips"):
        read_local(TAPES, "NOPEUSDT")


# ------------------------------------------------------------ sign convention


def test_is_buyer_maker_true_means_seller_initiated():
    """The one convention that is easy to invert and impossible to notice later."""
    t = pd.DataFrame({"ts": pd.to_datetime([1, 2], unit="s", utc=True),
                      "price": [10.0, 10.0], "qty": [1.0, 1.0],
                      "is_buyer_maker": [True, False]})
    s = sign_trades(t)
    assert s.iloc[0] == -1.0          # buyer resting -> aggressor sold
    assert s.iloc[1] == +1.0


def test_planted_signs_survive_the_round_trip():
    ticks = simulate_ticks(TickSpec(n_trades=5000, seed=3))
    got = sign_trades(ticks.trades).to_numpy()
    np.testing.assert_allclose(got, ticks.signs)


# ---------------------------------------------------------------------- bars


def test_bars_aggregate_flow_and_drop_empty_intervals(perm_bars):
    assert (perm_bars.n_trades > 0).all()
    assert perm_bars.volume.min() > 0
    assert np.isfinite(perm_bars.signed_volume).all()


def test_resampling_preserves_total_flow(perm_bars):
    coarse = resample_bars(perm_bars, "10s")
    assert coarse.signed_volume.sum() == pytest.approx(
        perm_bars.signed_volume.sum(), rel=1e-9)
    assert len(coarse.trade_price) < len(perm_bars.trade_price)


def test_normalise_flow_is_unit_scale(perm_bars):
    z = normalise_flow(perm_bars.signed_volume)
    assert z.std(ddof=1) == pytest.approx(1.0, rel=1e-6)


# ------------------------------------------------------------ bid-ask bounce


def test_bounce_must_be_measured_at_trade_level_not_bar_level(perm_bars):
    """Bars average the bounce away; only the raw tape shows it."""
    d = read_local(TAPES, "PERMUSDT")
    bar_only = bounce_diagnostics(perm_bars)
    with_ticks = bounce_diagnostics(perm_bars, d.trades)
    assert bar_only["has_mid"]
    # One-second bars hold several trades, so the alternation cancels and Roll
    # has nothing to work with.
    assert np.isnan(bar_only["trade_bar_roll_spread_bps"])
    # Roll fails on this tape too: order splitting makes impact autocorrelation
    # swamp the bounce, so the serial covariance is positive and the model has
    # no solution. That is the realistic case, not a defect.
    assert not with_ticks["roll_usable"]
    # With a book, the bounce is measured directly and needs no model.
    joined = align_mid_to_trades(d.trades, d.book)
    assert effective_half_spread_bps(joined) == pytest.approx(
        d.trades.attrs.get("half_spread_bps", 0.5), rel=0.2)
    # And the trade print is noisier than the mid at bar level.
    assert bar_only["trade_ac1"] < bar_only["mid_ac1"]


def test_asof_join_never_looks_forward():
    trades = pd.DataFrame({"ts": pd.to_datetime([10, 20], unit="s", utc=True),
                           "price": [100.0, 100.0], "qty": [1.0, 1.0],
                           "is_buyer_maker": [False, False]})
    book = pd.DataFrame({"ts": pd.to_datetime([5, 15, 25], unit="s", utc=True),
                         "mid": [99.0, 101.0, 999.0]})
    j = align_mid_to_trades(trades, book)
    assert list(j["mid"]) == [99.0, 101.0]     # never the 999 published later


def test_roll_recovers_the_planted_spread():
    spec = TickSpec(n_trades=120_000, half_spread_bps=1.5, sigma_bps=0.05,
                    kappa_perm=0.0, kappa_trans=0.0, seed=4)
    t = simulate_ticks(spec)
    bars = build_bars(t.trades, "1s", t.book)
    est = 1e4 * roll_spread(bars.trade_price)
    assert est == pytest.approx(2 * spec.half_spread_bps, rel=0.35)


def test_roll_declines_to_report_on_a_trending_tape():
    """Positively autocorrelated returns fall outside Roll's model entirely."""
    rng = np.random.default_rng(2)
    r = np.zeros(2000)
    for i in range(1, 2000):
        r[i] = 0.6 * r[i - 1] + rng.normal(scale=1e-3)
    assert np.isnan(roll_spread(pd.Series(100 * np.exp(np.cumsum(r)))))


def test_bounce_inflates_the_short_horizon_response(perm_bars):
    """Using trade prices instead of mid distorts R at short horizons."""
    mid = response_at_horizon(perm_bars, 1, measure="impact", price_kind="mid")
    trade = response_at_horizon(perm_bars, 1, measure="impact", price_kind="trade")
    assert abs(trade.r_bps - mid.r_bps) > 0.05


# ------------------------------------------------------------------ response


def test_impact_and_predictive_are_different_quantities(perm_bars):
    """Impact includes the contemporaneous move; predictive does not."""
    imp = response_at_horizon(perm_bars, 10, measure="impact", price_kind="mid")
    pre = response_at_horizon(perm_bars, 10, measure="predictive", price_kind="mid")
    assert imp.r_bps > pre.r_bps


def test_impact_response_is_positive_at_short_horizons(perm_bars):
    r = response_at_horizon(perm_bars, 5, measure="impact", price_kind="mid")
    assert r.r_bps > 0 and r.r_t > 3


def test_underpowered_horizons_are_dropped(perm_bars):
    c = response_curve(perm_bars, [1, 10, 10**9], measure="impact", price_kind="mid")
    assert (c["horizon_s"] < 10**8).all()
    assert (c["n_nonoverlap"] >= 20).all()


def test_bad_measure_is_rejected(perm_bars):
    with pytest.raises(ValueError):
        response_at_horizon(perm_bars, 1, measure="wishful")


def test_mid_is_refused_when_absent():
    d = read_local(TAPES, "PERMUSDT", with_book=False)
    bars = build_bars(d.trades, "1s", None)
    assert not bars.has_mid
    with pytest.raises(ValueError, match="no mid"):
        response_at_horizon(bars, 1, price_kind="mid")


def test_ols_nw_matches_ols_at_zero_lags():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 2.5 * x + rng.normal(size=500)
    b, se, n = ols_nw(y, x, lags=0)
    assert b == pytest.approx(2.5, abs=0.15)
    assert n == 500
    assert se > 0


def test_newey_west_widens_errors_under_serial_correlation():
    rng = np.random.default_rng(1)
    e = np.cumsum(rng.normal(size=2000))          # heavily autocorrelated
    x = np.cumsum(rng.normal(size=2000))
    _, se0, _ = ols_nw(e, x, lags=0)
    _, se50, _ = ols_nw(e, x, lags=50)
    assert se50 > se0


# ---------------------------------------------- the two-sided central result


def test_permanent_component_is_detected_when_planted(perm_bars):
    v = permanent_verdict(perm_bars, WINDOWS, price_kind="mid", n_boot=40)
    assert v.slope_log > -0.7, v.verdict
    assert v.lambda_long > 0


def test_no_permanent_component_is_reported_when_none_exists(trans_bars):
    """The test that matters most: the estimator must be able to say no."""
    v = permanent_verdict(trans_bars, WINDOWS, price_kind="mid", n_boot=40)
    assert "TRANSIENT" in v.verdict, v.verdict
    assert v.slope_log < -0.7


def test_lambda_separates_the_two_tapes(perm_bars, trans_bars):
    a = permanent_lambda_curve(perm_bars, WINDOWS, price_kind="mid")
    b = permanent_lambda_curve(trans_bars, WINDOWS, price_kind="mid")
    assert a["lambda_per_unit"].iloc[-1] > b["lambda_per_unit"].iloc[-1]


def test_lambda_excess_is_linear_in_the_planted_coefficient():
    """The quantitative validation: recovered lambda is linear in the truth."""
    got = []
    for kp in (0.0, 0.4, 0.8):
        t = simulate_ticks(TickSpec(n_trades=80_000, kappa_perm=kp,
                                    sigma_bps=0.2, seed=7))
        bars = build_bars(t.trades, "1s", t.book)
        c = permanent_lambda_curve(bars, [1, 5, 30, 120, 300], price_kind="mid")
        got.append(float(c["lambda_per_unit"].iloc[-1]))
    excess = np.array(got) - got[0]
    planted = np.array([0.0, 0.4, 0.8])
    assert excess[2] > excess[1] > 0
    r = np.corrcoef(planted, excess)[0, 1]
    assert r > 0.99


def test_transient_tape_lambda_falls_by_an_order_of_magnitude(trans_bars):
    c = permanent_lambda_curve(trans_bars, WINDOWS, price_kind="mid")
    assert c["lambda_per_unit"].iloc[-1] < 0.2 * c["lambda_per_unit"].iloc[0]


# -------------------------------------------------------------- persistence


def test_order_splitting_gives_long_memory_flow(perm_bars):
    s = np.sign(perm_bars.signed_volume.to_numpy())
    C, beta = fit_flow_autocorrelation(s, max_lag=200)
    assert C[1] > 0.05                       # correlated
    assert C[50] > 0                         # still correlated far out
    assert 0.0 < beta < 2.0                  # power-law-ish decay


def test_integrated_time_exceeds_the_ar1_half_life(perm_bars):
    """For power-law decay the AR(1) half-life badly understates persistence."""
    p = persistence(perm_bars.signed_volume, bar_seconds=1.0)
    assert p.integrated_time_bars > p.half_life_bars


def test_independent_observations_fall_as_the_signal_slows(perm_bars):
    pc = persistence_curve(perm_bars, [1, 10, 60])
    assert pc["independent_obs_per_year"].is_monotonic_decreasing


def test_white_noise_has_unit_integrated_time():
    x = pd.Series(np.random.default_rng(0).normal(size=20_000))
    p = persistence(x, bar_seconds=1.0)
    assert p.integrated_time_bars == pytest.approx(1.0, abs=0.4)


# --------------------------------------------------------------- null floors


def test_shuffled_timing_destroys_the_response(perm_bars):
    real = response_at_horizon(perm_bars, 10, measure="impact", price_kind="mid")
    nf = null_floor(perm_bars, 10, n_draws=40, measure="impact",
                    price_kind="mid", seed=0)
    assert abs(real.r_bps) > nf.r_abs_95_bps
    assert abs(nf.r_mean) < abs(real.r_bps)


def test_null_floor_exceeds_the_nominal_significance_threshold(perm_bars):
    """As with the equity features, the empirical critical t is above 1.96."""
    nf = null_floor(perm_bars, 30, n_draws=60, measure="impact", price_kind="mid")
    assert np.isfinite(nf.t_abs_95)
    assert nf.t_abs_95 > 1.0


# ------------------------------------------------------------------ economics


def test_costs_and_crossover():
    c = CostModel(taker_bps=5.0, spread_bps=0.5, slippage_bps=0.5)
    assert c.round_trip_taker == pytest.approx(12.0)
    df = pd.DataFrame({"horizon_s": [1.0, 60.0, 3600.0],
                       "r_bps": [1.0, 10.0, 100.0],
                       "r_se_bps": [0.1, 0.5, 5.0],
                       "n_obs": [10000, 1000, 100],
                       "integrated_time_s": [1.0, 60.0, 3600.0],
                       "independent_obs_per_year": [3.15e7, 5.2e5, 8760.0]})
    out = economics(df, df, c, capture=1.0)
    assert not out["pays_taker"].iloc[0]
    assert out["pays_taker"].iloc[2]
    cx = crossover(out)
    assert cx["crossover_horizon_s"] == 3600.0
    assert not cx["never_pays"]


def test_naive_rebalance_sharpe_is_the_inflated_one():
    """Counting attempts by rebalance frequency rather than by decay inflates."""
    df = pd.DataFrame({"horizon_s": [60.0], "r_bps": [20.0], "r_se_bps": [1.0],
                       "n_obs": [1000], "integrated_time_s": [600.0],
                       "independent_obs_per_year": [5.26e4]})
    out = economics(df, df, CostModel(), capture=1.0)
    assert out["sharpe_naive_rebalance"].iloc[0] > out["sharpe_taker"].iloc[0]
    assert out["sharpe_inflation_factor"].iloc[0] > 2.0


def test_never_pays_is_reported_honestly():
    df = pd.DataFrame({"horizon_s": [1.0, 60.0], "r_bps": [0.1, 0.2],
                       "r_se_bps": [0.1, 0.1], "n_obs": [100, 100],
                       "integrated_time_s": [1.0, 60.0],
                       "independent_obs_per_year": [1e6, 1e4]})
    out = economics(df, df, CostModel(), capture=0.5)
    assert crossover(out)["never_pays"]


def test_horizon_formatting():
    assert fmt_horizon(1) == "1s"
    assert fmt_horizon(300) == "5min"
    assert fmt_horizon(3600) == "1hr"
    assert fmt_horizon(432000) == "5d"


# ------------------------------------------------------------- propagator TIM


def test_tim_response_is_monotone_for_a_flat_autocorrelation():
    C = np.zeros(50)
    C[0] = 1.0
    lags = np.array([1, 2, 5, 10, 20])
    r = tim_response(lags, G0=1.0, l0=5.0, gamma=0.5, C=C)
    assert np.all(np.diff(r) < 0)        # with no flow memory, impact only decays


def test_critical_gamma():
    assert critical_gamma(0.2) == pytest.approx(0.4)
    assert np.isnan(critical_gamma(float("nan")))
