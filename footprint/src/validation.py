"""Purged walk-forward cross-validation with an embargo.

Why not k-fold
--------------
Random k-fold on a panel of daily returns leaks in two directions at once, and
both are fatal here:

  1. *Horizon overlap.* A signal on date t is scored against a return running to
     t+h. If date t is in the training fold and t+2 is in the test fold, the two
     samples share h-2 days of the same realised return. The model is then fitted
     on data that literally contains the answer to part of its test.

  2. *Cross-sectional co-movement.* Names co-move, so the same calendar date
     appearing in both folds leaks even without horizon overlap: the market
     return on that day is common to the training names and the test names. This
     is why the fold boundary must be a *date* boundary, never a row boundary.

The fix, from Lopez de Prado: split on time, purge from the training set any
sample whose forward window overlaps the test window, and add an embargo after
the test block so that serial correlation just past the boundary cannot be used
either.

Walk-forward on top of that means training data always *precedes* test data.
This costs statistical power -- the earliest test fold has the least training
history -- but it is the only arrangement that answers the question actually
being asked, which is whether the signal would have worked going forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One train/test split, as positional slices into the date index."""

    train: np.ndarray
    test: np.ndarray
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    purged: int
    embargoed: int

    def __repr__(self) -> str:                        # pragma: no cover - display
        return (f"Fold(train={self.train_dates[0].date()}..{self.train_dates[-1].date()} "
                f"n={len(self.train)}, test={self.test_dates[0].date()}.."
                f"{self.test_dates[-1].date()} n={len(self.test)}, "
                f"purged={self.purged}, embargo={self.embargoed})")


def purged_walk_forward(dates: pd.DatetimeIndex, *, n_splits: int = 5,
                        horizon: int = 5, lag: int = 1, embargo: int | None = None,
                        min_train: int = 252, expanding: bool = True) -> list[Fold]:
    """Build walk-forward folds with horizon purging and an embargo.

    Parameters
    ----------
    horizon, lag
        Together these fix how far a sample's forward window reaches: a signal on
        date t is not resolved until ``t + lag + horizon``. Training samples whose
        window reaches into the test block are purged.
    embargo
        Sessions dropped from training *after* the test block. Defaults to
        ``lag + horizon``, which is the minimum that blocks the symmetric leak;
        a longer embargo is the right call when features use long trailing
        windows, since those reach backwards across the boundary too.
    expanding
        True gives an expanding training window (all history before the test
        block); False gives a rolling window of ``min_train`` sessions.
    """
    dates = pd.DatetimeIndex(dates)
    n = len(dates)
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    reach = int(lag) + int(horizon)
    emb = int(reach if embargo is None else embargo)
    if min_train + n_splits <= 0 or n <= min_train + n_splits:
        raise ValueError(f"not enough dates ({n}) for {n_splits} splits with "
                         f"min_train={min_train}")

    usable = n - min_train
    block = usable // n_splits
    if block < 1:
        raise ValueError("test blocks would be empty; reduce n_splits or min_train")

    folds: list[Fold] = []
    for s in range(n_splits):
        t0 = min_train + s * block
        t1 = n if s == n_splits - 1 else min_train + (s + 1) * block
        test_idx = np.arange(t0, t1)
        if test_idx.size == 0:
            continue

        train_hi = t0
        candidate = np.arange(0, train_hi)
        if not expanding:
            candidate = candidate[-min_train:]

        # Purge: a training sample at u resolves at u + reach. Drop it if that
        # reaches the test block.
        resolved_before_test = candidate + reach < t0
        purged = int((~resolved_before_test).sum())
        train_idx = candidate[resolved_before_test]

        # Embargo: sessions immediately after the test block are excluded from
        # any later training set. With strict walk-forward they are already
        # excluded here, so this matters only for the expanding case where a
        # later fold would otherwise pick them up -- it is applied there.
        embargoed = 0
        if s > 0 and emb > 0:
            prev_end = min_train + s * block
            banned = np.arange(prev_end, min(prev_end + emb, n))
            before = train_idx.size
            train_idx = train_idx[~np.isin(train_idx, banned)]
            embargoed = before - train_idx.size

        if train_idx.size == 0:
            continue
        folds.append(Fold(train=train_idx, test=test_idx,
                          train_dates=dates[train_idx], test_dates=dates[test_idx],
                          purged=purged, embargoed=embargoed))
    return folds


def assert_no_leakage(folds: list[Fold], *, horizon: int, lag: int) -> None:
    """Hard check that no training sample's forward window touches its test block.

    Called by the tests, and cheap enough to call in experiments too. It is the
    kind of invariant that is easy to break with an off-by-one and impossible to
    notice from the P&L, which will simply look better than it should.
    """
    reach = int(lag) + int(horizon)
    for f in folds:
        if f.train.size == 0 or f.test.size == 0:
            continue
        if f.train.max() >= f.test.min():
            raise AssertionError("training index reaches into or past the test block")
        if f.train.max() + reach >= f.test.min():
            raise AssertionError(
                f"unpurged overlap: last train index {f.train.max()} + reach {reach} "
                f">= first test index {f.test.min()}")


def fold_report(folds: list[Fold]) -> pd.DataFrame:
    return pd.DataFrame([{
        "fold": i,
        "train_start": f.train_dates[0].date(), "train_end": f.train_dates[-1].date(),
        "n_train": len(f.train),
        "test_start": f.test_dates[0].date(), "test_end": f.test_dates[-1].date(),
        "n_test": len(f.test), "purged": f.purged, "embargoed": f.embargoed,
    } for i, f in enumerate(folds)])
