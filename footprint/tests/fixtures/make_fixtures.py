"""Regenerate the recorded fixtures. Run once; the CSVs are committed.

Fixtures are committed rather than generated at test time so that the tests are
pinned to bytes on disk. A test that regenerates its own input tests the
generator as much as the code, and drifts silently when the generator changes.

    python3 -m tests.fixtures.make_fixtures

Two kinds are written:

  panel/    a small synthetic multi-name panel in Stooq format, used for the
            pipeline tests (features, backtest, walk-forward).

  messy/    deliberately awkward single files that mirror what real Stooq
            exports actually contain: an OpenInt column, a duplicated date,
            newest-first ordering, a zero-volume session, a blank line, and a
            zero close. The loader has to survive all of them.
"""

from __future__ import annotations

from pathlib import Path

from src.synthetic import SyntheticSpec, simulate, write_stooq_csvs

HERE = Path(__file__).parent


def make_panel() -> None:
    out = HERE / "panel"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.txt"):
        f.unlink()
    sim = simulate(SyntheticSpec(seed=7, n_names=8, n_days=420), planted_ic=0.05,
                   horizon=5)
    write_stooq_csvs(sim.panel, out)


MESSY = {
    # Extra OpenInt column, as Stooq actually ships.
    "extra_col.us.txt": """Date,Open,High,Low,Close,Volume,OpenInt
2020-01-02,10.0,10.5,9.8,10.2,1000000,0
2020-01-03,10.2,10.6,10.0,10.4,1100000,0
2020-01-06,10.4,10.9,10.3,10.8,1200000,0
""",
    # Newest first, plus a duplicated date whose last print should win.
    "reversed_dup.us.txt": """Date,Open,High,Low,Close,Volume
2020-01-06,10.4,10.9,10.3,10.8,1200000
2020-01-03,10.2,10.6,10.0,10.4,1100000
2020-01-03,10.2,10.6,10.0,99.9,1
2020-01-02,10.0,10.5,9.8,10.2,1000000
""",
    # A ragged row: one line has a stray extra field and must be dropped
    # without costing the rest of the file.
    "ragged.us.txt": """Date,Open,High,Low,Close,Volume
2020-01-02,10.0,10.5,9.8,10.2,1000000
2020-01-03,10.2,10.6,10.0,10.4,1100000,999,7
2020-01-06,10.4,10.9,10.3,10.8,1200000
""",
    # Zero volume (a halt), a zero close (bad print), and a blank line.
    "zeros.us.txt": """Date,Open,High,Low,Close,Volume
2020-01-02,10.0,10.5,9.8,10.2,1000000

2020-01-03,10.2,10.6,10.0,10.4,0
2020-01-06,10.4,10.9,10.3,0.0,1200000
2020-01-07,10.5,11.0,10.4,10.9,1300000
""",
    # Lowercase headers and whitespace.
    "lowercase.csv": """date, open, high, low, close, volume
2020-01-02,10.0,10.5,9.8,10.2,1000000
2020-01-03,10.2,10.6,10.0,10.4,1100000
""",
    # Not a price file at all.
    "garbage.us.txt": "this is not a csv\nneither is this\n",
}


def make_messy() -> None:
    out = HERE / "messy"
    out.mkdir(parents=True, exist_ok=True)
    for name, body in MESSY.items():
        (out / name).write_text(body)


if __name__ == "__main__":
    make_panel()
    make_messy()
    print(f"wrote fixtures under {HERE}")
