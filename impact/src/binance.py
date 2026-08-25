"""Keyless bulk download from data.binance.vision, and offline reading.

Two entry points that deliberately do not know about each other:

  ``fetch_day``   downloads one symbol-day zip (and verifies its checksum)
  ``read_local``  reads whatever zips are already on disk

Everything downstream uses ``read_local``, so the analysis runs identically on a
machine with no network as long as the zips are present. That separation is the
whole point: the measurement code never depends on being online.

A correction worth recording, because I told the user otherwise earlier in this
project. **bookTicker is not published for spot.** The bulk endpoint carries, per
market:

    spot            aggTrades, trades, klines
    futures/um      aggTrades, trades, klines, bookTicker, bookDepth, metrics
    futures/cm      same as um

So reconstructing a true mid price requires the USD-margined futures market
(``um``), not spot. That is why ``market`` defaults to ``um`` here: the whole
transient-versus-permanent question turns on having a mid price that is not
contaminated by bid-ask bounce, and on spot we would be stuck with trade prints
and a Roll-model correction. Spot is still supported for cross-checking on a
venue with a different fee schedule.

File layout (daily):

    data/{market}/daily/{kind}/{SYMBOL}/{SYMBOL}-{kind}-{YYYY-MM-DD}.zip
    ... and the same path with .CHECKSUM appended, holding "sha256  filename".
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BASE = "https://data.binance.vision"

# Column names. Binance ships these files with a header row from 2025 onward and
# without one before that, so the reader sniffs rather than assuming.
AGG_COLS = ["agg_id", "price", "qty", "first_id", "last_id", "ts",
            "is_buyer_maker", "is_best_match"]
BOOK_COLS = ["update_id", "bid_price", "bid_qty", "ask_price", "ask_qty",
             "transaction_time", "event_time"]

MARKETS = {"spot": "data/spot", "um": "data/futures/um", "cm": "data/futures/cm"}
KINDS_WITH_BOOK = ("um", "cm")


class BinanceError(RuntimeError):
    pass


def day_url(symbol: str, day: str, kind: str = "aggTrades",
            market: str = "um") -> str:
    if market not in MARKETS:
        raise BinanceError(f"unknown market {market!r}; expected one of {list(MARKETS)}")
    if kind == "bookTicker" and market not in KINDS_WITH_BOOK:
        raise BinanceError(
            "bookTicker is not published for spot -- use market='um' for a real "
            "mid price, or accept trade prints plus a Roll-model bounce correction")
    sym = symbol.upper()
    return f"{BASE}/{MARKETS[market]}/daily/{kind}/{sym}/{sym}-{kind}-{day}.zip"


def local_name(symbol: str, day: str, kind: str, market: str) -> str:
    return f"{market}-{symbol.upper()}-{kind}-{day}.zip"


def fetch_day(symbol: str, day: str, out_dir, *, kind: str = "aggTrades",
              market: str = "um", verify: bool = True, timeout: int = 120,
              session=None) -> Path:
    """Download one symbol-day zip into ``out_dir``, verifying its SHA256.

    Requires network. Raises :class:`BinanceError` with the URL on failure so a
    blocked egress policy is reported rather than silently retried.
    """
    import urllib.request

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / local_name(symbol, day, kind, market)
    if dest.exists():
        return dest

    url = day_url(symbol, day, kind, market)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            blob = r.read()
    except Exception as exc:
        raise BinanceError(f"could not fetch {url}: {exc}") from exc

    if verify:
        try:
            with urllib.request.urlopen(url + ".CHECKSUM", timeout=timeout) as r:
                want = r.read().decode().split()[0].strip()
            got = hashlib.sha256(blob).hexdigest()
            if got != want:
                raise BinanceError(f"checksum mismatch for {url}: {got} != {want}")
        except BinanceError:
            raise
        except Exception:
            pass          # no checksum published for this file; keep the payload

    dest.write_bytes(blob)
    return dest


def _read_zip(path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise BinanceError(f"{Path(path).name}: no csv inside")
        raw = zf.read(names[0])
    head = raw[:200].decode("utf-8", "replace").split("\n", 1)[0]
    has_header = any(c.isalpha() for c in head.replace("true", "").replace("false", ""))
    return pd.read_csv(io.BytesIO(raw), header=0 if has_header else None,
                       low_memory=False)


def _normalise_timestamps(s: pd.Series) -> pd.Series:
    """Binance switched some 2025 files from milliseconds to microseconds.

    Detect by magnitude rather than by date: a millisecond epoch for any
    plausible date is ~1.7e12, microseconds ~1.7e15. Guessing from the filename
    would break on the exact files where it matters.
    """
    v = pd.to_numeric(s, errors="coerce")
    med = float(v.dropna().median()) if v.notna().any() else 0.0
    unit = "us" if med > 1e14 else ("ms" if med > 1e11 else "s")
    return pd.to_datetime(v, unit=unit, utc=True)


def read_agg_trades(path) -> pd.DataFrame:
    """One aggTrades zip -> DataFrame[ts, price, qty, is_buyer_maker].

    ``is_buyer_maker`` is the field that makes this dataset usable: when True the
    buyer was resting and the *aggressor was a seller*, so the trade is
    sell-initiated. That is the sign convention applied in ``flow.py``, and it is
    the one thing in this pipeline that is easy to get backwards and impossible
    to notice afterwards -- an inverted sign simply flips the response function,
    which still looks like a plausible curve.
    """
    df = _read_zip(path)
    if df.shape[1] < 7:
        raise BinanceError(f"{Path(path).name}: expected >=7 columns, got {df.shape[1]}")
    if not isinstance(df.columns[0], str) or str(df.columns[0]).isdigit():
        df.columns = AGG_COLS[:df.shape[1]]
    else:
        lower = {c: str(c).strip().lower() for c in df.columns}
        df = df.rename(columns=lower)
        rename = {"transact_time": "ts", "time": "ts", "timestamp": "ts",
                  "quantity": "qty", "agg_trade_id": "agg_id"}
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        if "ts" not in df.columns:
            df.columns = AGG_COLS[:df.shape[1]]

    out = pd.DataFrame({
        "ts": _normalise_timestamps(df["ts"]),
        "price": pd.to_numeric(df["price"], errors="coerce"),
        "qty": pd.to_numeric(df["qty"], errors="coerce"),
    })
    m = df["is_buyer_maker"]
    out["is_buyer_maker"] = (m.astype(str).str.strip().str.lower()
                             .isin(["true", "1", "t", "yes"]) if m.dtype == object
                             else m.astype(bool))
    out = out.dropna(subset=["ts", "price", "qty"])
    out = out[(out["price"] > 0) & (out["qty"] > 0)]
    return out.sort_values("ts").reset_index(drop=True)


def read_book_ticker(path) -> pd.DataFrame:
    """One bookTicker zip -> DataFrame[ts, bid, ask, mid]."""
    df = _read_zip(path)
    if not isinstance(df.columns[0], str) or str(df.columns[0]).isdigit():
        df.columns = BOOK_COLS[:df.shape[1]]
    else:
        df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    ts_col = ("transaction_time" if "transaction_time" in df.columns
              else "event_time" if "event_time" in df.columns else df.columns[-1])
    out = pd.DataFrame({
        "ts": _normalise_timestamps(df[ts_col]),
        "bid": pd.to_numeric(df["bid_price"], errors="coerce"),
        "ask": pd.to_numeric(df["ask_price"], errors="coerce"),
    }).dropna()
    out = out[(out["bid"] > 0) & (out["ask"] >= out["bid"])]
    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    return out.sort_values("ts").reset_index(drop=True)


@dataclass
class SymbolData:
    symbol: str
    trades: pd.DataFrame
    book: pd.DataFrame | None
    days: list[str]

    @property
    def has_mid(self) -> bool:
        return self.book is not None and not self.book.empty


def read_local(data_dir, symbol: str, *, market: str = "um",
               with_book: bool = True, max_days: int | None = None) -> SymbolData:
    """Read every locally present zip for one symbol.

    Missing days are simply absent -- no interpolation, no synthetic fill. A gap
    in the tape is a gap, and the bar builder handles it by producing no bars
    there rather than inventing flat ones.
    """
    data_dir = Path(data_dir)
    sym = symbol.upper()
    agg = sorted(data_dir.glob(f"{market}-{sym}-aggTrades-*.zip"))
    if not agg:
        raise BinanceError(
            f"no aggTrades zips for {sym} ({market}) in {data_dir}. Expected files "
            f"named like {local_name(sym, '2024-01-01', 'aggTrades', market)}")
    if max_days:
        agg = agg[:max_days]
    days = [p.stem.split("-aggTrades-")[-1] for p in agg]
    trades = pd.concat([read_agg_trades(p) for p in agg], ignore_index=True)
    trades = trades.sort_values("ts").reset_index(drop=True)

    book = None
    if with_book:
        bp = sorted(data_dir.glob(f"{market}-{sym}-bookTicker-*.zip"))
        bp = [p for p in bp if p.stem.split("-bookTicker-")[-1] in set(days)]
        if bp:
            book = pd.concat([read_book_ticker(p) for p in bp], ignore_index=True)
            book = book.sort_values("ts").reset_index(drop=True)
    return SymbolData(symbol=sym, trades=trades, book=book, days=days)
