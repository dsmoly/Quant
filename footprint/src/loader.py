"""Reading daily OHLCV from a directory of Stooq-format CSVs.

The whole project is written against this one interface, so that the moment real
files land in ``data/`` everything downstream runs unchanged. Nothing else in the
codebase knows where prices come from.

Stooq daily exports look like::

    Date,Open,High,Low,Close,Volume
    2015-01-02,111.39,111.44,107.35,109.33,53204626

with one file per ticker, named after it (``aapl.us.txt``, ``AAPL.csv``, ...).
Real exports are messier than the specification: some carry an ``OpenInt``
column, some quote a ``Volume`` of 0 on holidays, some repeat a date, and some
arrive newest-first. The loader normalises all of that rather than assuming a
clean file, because a silent misparse here would propagate into every number the
project reports.

What the loader deliberately does *not* do
------------------------------------------
It does not forward-fill prices across gaps. A missing session is left missing,
so that features computed over a trailing window see a shorter window rather than
a synthetic flat return -- a fabricated zero return would read as "no impact
today" and feed straight into the abnormal-lambda signal as a false liquidity
supply print. Gaps are reported instead, and coverage screening drops names that
have too many.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# Stooq names files like "aapl.us.txt"; the exchange suffix is not part of the
# ticker. Anything else falls back to the bare filename stem.
_STOOQ_SUFFIX = re.compile(r"\.(us|uk|de|jp|hk|f|pl)$", re.IGNORECASE)


class DataError(ValueError):
    """Raised when a file cannot be parsed into a usable price series."""


def ticker_from_path(path: str | Path) -> str:
    stem = Path(path).stem
    return _STOOQ_SUFFIX.sub("", stem).upper()


def read_stooq_csv(path: str | Path) -> pd.DataFrame:
    """Parse one Stooq-format daily file into a clean OHLCV frame.

    Returns a frame indexed by ``DatetimeIndex`` (ascending, unique) with float
    columns Open/High/Low/Close/Volume. Raises :class:`DataError` rather than
    returning something subtly wrong.
    """
    path = Path(path)
    try:
        raw = pd.read_csv(path, skipinitialspace=True)
    except pd.errors.ParserError:
        # Ragged rows happen in bulk exports (a stray delimiter, a truncated
        # download). Dropping the broken lines and keeping the file is the right
        # trade for a loader that has to survive a whole directory: one bad line
        # should not cost the entire history of a name.
        try:
            raw = pd.read_csv(path, skipinitialspace=True, on_bad_lines="skip",
                              engine="python")
        except Exception as exc:
            raise DataError(f"{path.name}: unparseable ({exc})") from exc
    except Exception as exc:                       # unreadable / empty / binary
        raise DataError(f"{path.name}: unreadable ({exc})") from exc

    # Column names vary in case and in the presence of extras (OpenInt, Adj Close).
    raw.columns = [str(c).strip().title().replace(" ", "") for c in raw.columns]
    if "Date" not in raw.columns:
        raise DataError(f"{path.name}: no Date column (found {list(raw.columns)})")
    missing = [c for c in OHLCV if c not in raw.columns]
    if missing:
        raise DataError(f"{path.name}: missing columns {missing}")

    out = raw[["Date"] + OHLCV].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for c in OHLCV:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["Date"]).set_index("Date").sort_index()
    # Duplicate dates: keep the last print for that session.
    out = out[~out.index.duplicated(keep="last")]

    # A non-positive close is not a price. Volume of zero is a real thing (a
    # halted or untraded session) but it makes Amihud undefined, so it is carried
    # as NaN and handled by the features rather than silently becoming infinity.
    out.loc[out["Close"] <= 0, "Close"] = np.nan
    out.loc[out["Volume"] < 0, "Volume"] = np.nan
    out = out.dropna(subset=["Close"])
    if out.empty:
        raise DataError(f"{path.name}: no usable rows after cleaning")
    return out.astype(float)


@dataclass
class Panel:
    """Aligned daily OHLCV across a set of tickers.

    Every field is a ``DataFrame`` indexed by date with one column per ticker, on
    a common date index (the union of the individual files' dates). Missing
    sessions stay NaN.
    """

    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame

    def __post_init__(self) -> None:
        ref = self.close
        for name in ("open", "high", "low", "volume"):
            other = getattr(self, name)
            if not other.index.equals(ref.index) or list(other.columns) != list(ref.columns):
                raise DataError(f"panel field '{name}' is not aligned with close")

    # ------------------------------------------------------------- accessors

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def dollar_volume(self) -> pd.DataFrame:
        """Close x share volume. Stooq gives no notional, so this is the proxy."""
        return self.close * self.volume

    def returns(self) -> pd.DataFrame:
        """Simple close-to-close returns; the first row is NaN by construction."""
        return self.close.pct_change()

    def log_returns(self) -> pd.DataFrame:
        return np.log(self.close).diff()

    def coverage(self) -> pd.Series:
        """Fraction of the panel's sessions each ticker actually has a close for."""
        return self.close.notna().mean()

    def select(self, tickers: list[str]) -> "Panel":
        keep = [t for t in tickers if t in self.close.columns]
        if not keep:
            raise DataError("selection left no tickers")
        return Panel(**{f: getattr(self, f)[keep] for f in
                        ("close", "open", "high", "low", "volume")})

    def slice_dates(self, start=None, end=None) -> "Panel":
        idx = self.close.index
        mask = np.ones(len(idx), dtype=bool)
        if start is not None:
            mask &= idx >= pd.Timestamp(start)
        if end is not None:
            mask &= idx <= pd.Timestamp(end)
        return Panel(**{f: getattr(self, f).loc[mask] for f in
                        ("close", "open", "high", "low", "volume")})


def load_panel(data_dir: str | Path, *, tickers: list[str] | None = None,
               min_coverage: float = 0.0, min_rows: int = 2,
               strict: bool = False) -> Panel:
    """Load every Stooq CSV in ``data_dir`` into an aligned :class:`Panel`.

    Parameters
    ----------
    tickers
        Restrict to these (case-insensitive). Missing ones are ignored unless
        ``strict``.
    min_coverage
        Drop tickers present for less than this fraction of the panel's sessions.
        Screening on coverage is a survivorship decision in itself -- see
        ``universe.SURVIVORSHIP_NOTE``.
    strict
        Raise on any unparseable file instead of skipping it.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise DataError(f"no such data directory: {data_dir}")

    paths = sorted(p for p in data_dir.iterdir()
                   if p.suffix.lower() in (".csv", ".txt") and p.is_file())
    if not paths:
        raise DataError(
            f"{data_dir} contains no .csv or .txt files. Drop one Stooq-format "
            f"file per ticker in here (Date,Open,High,Low,Close,Volume).")

    wanted = {t.upper() for t in tickers} if tickers else None
    frames: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for p in paths:
        tick = ticker_from_path(p)
        if wanted is not None and tick not in wanted:
            continue
        try:
            df = read_stooq_csv(p)
        except DataError as exc:
            if strict:
                raise
            skipped.append(str(exc))
            continue
        if len(df) < min_rows:
            skipped.append(f"{p.name}: only {len(df)} rows")
            continue
        frames[tick] = df

    if not frames:
        raise DataError(f"no usable files in {data_dir}"
                        + (f"; skipped: {skipped[:5]}" if skipped else ""))

    if wanted is not None and strict:
        absent = sorted(wanted - set(frames))
        if absent:
            raise DataError(f"requested tickers not found: {absent}")

    names = sorted(frames)
    index = pd.DatetimeIndex(sorted(set().union(*(frames[t].index for t in names))))
    fields = {}
    for field in ("Open", "High", "Low", "Close", "Volume"):
        fields[field.lower()] = pd.DataFrame(
            {t: frames[t][field].reindex(index) for t in names}, index=index)

    panel = Panel(**fields)
    if min_coverage > 0:
        keep = panel.coverage()[lambda s: s >= min_coverage].index.tolist()
        if not keep:
            raise DataError(f"no ticker meets min_coverage={min_coverage}")
        panel = panel.select(keep)
    panel.skipped = skipped                       # kept for reporting, not logic
    return panel
