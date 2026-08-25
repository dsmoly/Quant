"""Regenerate tick fixtures. Run once; the zips are committed.

    python3 -m tests.fixtures.make_fixtures

Three tapes, all written in genuine Binance daily-zip format so the tests go
through the real loader rather than around it:

  PERMUSDT   a planted permanent component
  TRANSUSDT  the same flow and the same transient, but no permanent component
  HDRUSDT    a small tape in the *other* format variant Binance ships --
             a header row and microsecond timestamps
"""
from __future__ import annotations

from pathlib import Path

from src.synth_tick import TickSpec, simulate_ticks, write_binance_zips

HERE = Path(__file__).parent
TAPES = HERE / "tapes"


def main() -> None:
    TAPES.mkdir(parents=True, exist_ok=True)
    for f in TAPES.glob("*.zip"):
        f.unlink()
    common = dict(n_trades=60_000, sigma_bps=0.2, seed=5)
    write_binance_zips(simulate_ticks(TickSpec(kappa_perm=1.2, **common)),
                       TAPES, symbol="PERMUSDT", day="2024-01-01")
    write_binance_zips(simulate_ticks(TickSpec(kappa_perm=0.0, **common)),
                       TAPES, symbol="TRANSUSDT", day="2024-01-01")
    write_binance_zips(simulate_ticks(TickSpec(n_trades=4_000, seed=6)),
                       TAPES, symbol="HDRUSDT", day="2024-01-02",
                       header=True, micros=True)
    print(f"wrote {len(list(TAPES.glob('*.zip')))} zips to {TAPES}")


if __name__ == "__main__":
    main()
