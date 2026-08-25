"""Download the tapes. The only script that needs network.

    python3 -m experiments.fetch --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT \
        --start 2024-01-01 --end 2024-12-31 --out data

Defaults to USD-margined futures (``um``) because that is the only market where
Binance publishes bookTicker, and a true mid price is what keeps bid-ask bounce
out of the short-horizon response. Spot is available with --market spot for a
fee-schedule cross-check, but there the analysis is stuck with trade prints.

Sizing the request. The horizons run to five days, and the number of *independent*
five-day windows is what determines whether that end of the curve says anything:
one year gives about 73, three months about 18. Eighteen is not enough to resolve
anything, so a year is the default. Roughly 100-200MB per symbol-year of
aggTrades and rather more of bookTicker, which is why bookTicker can be fetched
at a lower cadence with --book-every.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from src.binance import BinanceError, fetch_day


def daterange(start: str, end: str):
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    if d1 < d0:
        raise SystemExit("end is before start")
    d = d0
    while d <= d1:
        yield d.isoformat()
        d += timedelta(days=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="+",
                    default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--market", default="um", choices=["um", "cm", "spot"])
    ap.add_argument("--out", default="data")
    ap.add_argument("--book-every", type=int, default=1,
                    help="fetch bookTicker every Nth day (0 disables)")
    ap.add_argument("--stop-on-error", action="store_true")
    args = ap.parse_args(argv)

    days = list(daterange(args.start, args.end))
    print(f"{len(args.symbols)} symbols x {len(days)} days -> {args.out} "
          f"(market={args.market})")

    got = failed = 0
    for sym in args.symbols:
        for i, day in enumerate(days):
            kinds = ["aggTrades"]
            if args.book_every and args.market != "spot" and i % args.book_every == 0:
                kinds.append("bookTicker")
            for kind in kinds:
                try:
                    fetch_day(sym, day, args.out, kind=kind, market=args.market)
                    got += 1
                except BinanceError as exc:
                    failed += 1
                    print(f"  MISS {sym} {day} {kind}: {exc}", file=sys.stderr)
                    if args.stop_on_error:
                        return 2
        print(f"  {sym}: done")
    print(f"\n{got} files fetched, {failed} missing")
    if got == 0:
        print("\nNothing downloaded. If every request failed with a 403 the "
              "egress policy is blocking data.binance.vision; that is a "
              "environment setting, not something to retry.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
