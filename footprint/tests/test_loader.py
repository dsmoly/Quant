"""Loader tests, against the recorded messy fixtures."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.loader import DataError, Panel, load_panel, read_stooq_csv, ticker_from_path

MESSY = Path(__file__).parent / "fixtures" / "messy"
PANEL = Path(__file__).parent / "fixtures" / "panel"


def test_ticker_strips_the_exchange_suffix():
    assert ticker_from_path("data/aapl.us.txt") == "AAPL"
    assert ticker_from_path("data/MSFT.csv") == "MSFT"
    assert ticker_from_path("/x/y/brk-b.us.txt") == "BRK-B"


def test_extra_columns_are_ignored():
    df = read_stooq_csv(MESSY / "extra_col.us.txt")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 3


def test_reversed_input_is_sorted_and_duplicates_keep_the_last_print():
    df = read_stooq_csv(MESSY / "reversed_dup.us.txt")
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique
    # The second 2020-01-03 row carries the 99.9 close and must win.
    assert df.loc[pd.Timestamp("2020-01-03"), "Close"] == pytest.approx(99.9)


def test_a_ragged_row_is_dropped_without_losing_the_file():
    df = read_stooq_csv(MESSY / "ragged.us.txt")
    assert len(df) == 2
    assert pd.Timestamp("2020-01-03") not in df.index
    assert pd.Timestamp("2020-01-06") in df.index


def test_lowercase_headers_are_normalised():
    df = read_stooq_csv(MESSY / "lowercase.csv")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_zero_close_is_dropped_but_zero_volume_is_kept():
    """A zero price is a bad print; a zero-volume session is a real halt."""
    df = read_stooq_csv(MESSY / "zeros.us.txt")
    assert pd.Timestamp("2020-01-06") not in df.index      # zero close removed
    assert pd.Timestamp("2020-01-03") in df.index          # zero volume retained
    assert df.loc[pd.Timestamp("2020-01-03"), "Volume"] == 0.0


def test_garbage_raises_or_is_skipped():
    with pytest.raises(DataError):
        read_stooq_csv(MESSY / "garbage.us.txt")
    # ... and in a directory load it is skipped rather than fatal.
    panel = load_panel(MESSY, strict=False)
    assert "GARBAGE" not in panel.tickers
    assert any("garbage" in s for s in panel.skipped)


def test_strict_mode_propagates_the_error():
    with pytest.raises(DataError):
        load_panel(MESSY, strict=True)


def test_empty_directory_gives_an_actionable_message(tmp_path):
    with pytest.raises(DataError, match="Stooq-format"):
        load_panel(tmp_path)


def test_missing_directory_is_reported(tmp_path):
    with pytest.raises(DataError, match="no such data directory"):
        load_panel(tmp_path / "nope")


def test_panel_alignment_and_fields():
    p = load_panel(PANEL)
    assert len(p.tickers) == 8
    assert p.close.index.is_monotonic_increasing
    for f in (p.open, p.high, p.low, p.volume):
        assert f.index.equals(p.close.index)
        assert list(f.columns) == p.tickers
    assert (p.dollar_volume.dropna() >= 0).all().all()


def test_panel_rejects_misaligned_fields():
    p = load_panel(PANEL)
    bad = p.close.iloc[:-1]
    with pytest.raises(DataError, match="not aligned"):
        Panel(close=p.close, open=bad, high=p.high, low=p.low, volume=p.volume)


def test_gaps_are_left_missing_rather_than_filled():
    """A forward-filled gap would read as a zero-return session downstream."""
    p = load_panel(MESSY, strict=False)
    # zeros.us.txt has 2020-01-07; extra_col.us.txt does not.
    assert np.isnan(p.close.loc[pd.Timestamp("2020-01-07"), "EXTRA_COL"])


def test_returns_and_coverage():
    p = load_panel(PANEL)
    r = p.returns()
    assert r.iloc[0].isna().all()
    assert (p.coverage() > 0.9).all()


def test_select_and_slice():
    p = load_panel(PANEL)
    two = p.select(p.tickers[:2])
    assert two.tickers == p.tickers[:2]
    mid = p.close.index[len(p.close) // 2]
    sliced = p.slice_dates(start=mid)
    assert sliced.close.index.min() >= mid
    with pytest.raises(DataError):
        p.select(["NOPE"])
