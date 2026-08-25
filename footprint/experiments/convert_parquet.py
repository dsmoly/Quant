"""Convert a piekstra/market-data parquet store into Stooq-format daily CSVs.

The store holds 5-minute candles as one parquet per symbol per day, with prices
as exact decimal strings and UTC-microsecond timestamps. This aggregates the
regular session (09:30-16:00 ET) into daily OHLCV and writes the format the
loader already reads.

Read the health warnings in the output before using anything downstream. Two are
serious enough to be printed rather than buried:

  * The upstream provider is hardcoded to Alpaca's ``feed=iex``. IEX is a single
    venue carrying roughly 2% of US consolidated volume, so the volume column is
    NOT total traded volume -- it is one venue's slice, and that slice varies by
    name and over time. Every footprint feature is a function of volume, so this
    is a first-order problem, not a rounding detail.
  * Prices are split-adjusted but not dividend-adjusted (``adjustment=split``).
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

SESSION_START, SESSION_END = "09:30", "16:00"


def daily_bars(symbol_dir: str, regular_hours_only: bool = True) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(symbol_dir, "*", "*", "*.parquet")))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        d = pd.read_parquet(f)
        if d.empty:
            continue
        t = d["timestamp"].dt.tz_convert("America/New_York")
        if regular_hours_only:
            keep = (t.dt.time >= pd.Timestamp(SESSION_START).time()) & \
                   (t.dt.time < pd.Timestamp(SESSION_END).time())
            d, t = d[keep], t[keep]
            if d.empty:
                continue
        px = {c: pd.to_numeric(d[c], errors="coerce") for c in
              ("open", "high", "low", "close")}
        frames.append({
            "Date": t.iloc[0].date(),
            "Open": px["open"].iloc[0], "High": px["high"].max(),
            "Low": px["low"].min(), "Close": px["close"].iloc[-1],
            "Volume": int(d["volume"].sum()), "Bars": len(d),
        })
    out = pd.DataFrame(frames)
    return out.dropna(subset=["Close"]).sort_values("Date") if not out.empty else out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True, help="path to the repo's data/ dir")
    ap.add_argument("--out", default="data")
    ap.add_argument("--min-bars", type=int, default=20,
                    help="drop days with fewer intraday bars than this")
    ap.add_argument("--min-days", type=int, default=250)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    symbols = sorted(d for d in os.listdir(args.store)
                     if os.path.isdir(os.path.join(args.store, d)))
    print(f"{len(symbols)} symbols in {args.store}\n")
    print(f"{'sym':<6}{'raw days':>9}{'kept':>7}{'median bars':>13}{'first':>12}{'last':>12}")
    written = 0
    for s in symbols:
        d = daily_bars(os.path.join(args.store, s))
        if d.empty:
            print(f"{s:<6}{'0':>9}   (no regular-session data)")
            continue
        raw = len(d)
        kept = d[d["Bars"] >= args.min_bars]
        med = int(d["Bars"].median())
        mark = "" if len(kept) >= args.min_days else "   DROPPED (too few usable days)"
        print(f"{s:<6}{raw:>9}{len(kept):>7}{med:>13}"
              f"{str(d['Date'].iloc[0]):>12}{str(d['Date'].iloc[-1]):>12}{mark}")
        if len(kept) >= args.min_days:
            kept[["Date", "Open", "High", "Low", "Close", "Volume"]].to_csv(
                os.path.join(args.out, f"{s.lower()}.us.txt"), index=False)
            written += 1

    print(f"\nwrote {written} files to {args.out}/")
    print("\n" + "!" * 74)
    print("!! VOLUME IS IEX-ONLY (~2% of consolidated). Every footprint feature is")
    print("!! a function of volume, so treat feature results as UNRELIABLE.")
    print("!! Prices are split-adjusted, not dividend-adjusted.")
    print("!" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
