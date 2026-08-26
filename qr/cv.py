"""Purged and embargoed splits, for anything that gets fitted.

Standard k-fold leaks in a financial panel for two separate reasons, and you
need both fixes:

**Purging.** A training sample at date t carries a label built from returns over
t+1..t+h. If a test fold starts at t+2, that training label overlaps the test
period. Remove every training sample whose label window intersects the test
window.

**Embargo.** Even non-overlapping samples adjacent to the test fold are
contaminated by serial correlation in features and returns. Drop a further
buffer after the test fold.

Without both, a model can score well out-of-sample while having effectively
seen the answer. With both, expect your cross-validated numbers to fall -- that
drop is the leak you were previously counting as skill.
"""

from __future__ import annotations

import numpy as np


class PurgedKFold:
    """K-fold over time with purging and an embargo.

    Yields (train_idx, test_idx) over *date indices*, not observations: the
    whole cross-section on a date moves together, which is the only split that
    makes sense for a panel.
    """

    def __init__(self, n_splits: int = 5, horizon: int = 1, embargo: int = 0):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = int(n_splits)
        self.horizon = int(max(horizon, 1))
        self.embargo = int(max(embargo, 0))

    def split(self, T: int):
        T = int(T)
        if T < self.n_splits * (self.horizon + self.embargo + 2):
            raise ValueError(
                f"T={T} too short for {self.n_splits} splits at horizon "
                f"{self.horizon} with embargo {self.embargo}")
        idx = np.arange(T)
        bounds = np.linspace(0, T, self.n_splits + 1).astype(int)
        for k in range(self.n_splits):
            lo, hi = bounds[k], bounds[k + 1]
            test = idx[lo:hi]
            # a training date t is purged if its label window [t+1, t+horizon]
            # touches the test window, or if it falls inside the embargo after it
            purge_lo = lo - self.horizon
            purge_hi = hi + self.embargo
            train = idx[(idx < purge_lo) | (idx >= purge_hi)]
            if train.size and test.size:
                yield train, test

    def get_n_splits(self, *_args) -> int:
        return self.n_splits


def walk_forward_splits(T: int, n_splits: int = 5, horizon: int = 1,
                        embargo: int = 0, min_train: int | None = None,
                        expanding: bool = True):
    """Anchored (expanding) or rolling walk-forward splits.

    Closer to how the strategy will actually be run than k-fold, because it
    never trains on the future. Prefer this for the final validation and keep
    k-fold for faster iteration.
    """
    T = int(T)
    horizon = int(max(horizon, 1))
    embargo = int(max(embargo, 0))
    if min_train is None:
        min_train = T // (n_splits + 1)
    idx = np.arange(T)
    bounds = np.linspace(min_train, T, n_splits + 1).astype(int)
    for k in range(n_splits):
        lo, hi = bounds[k], bounds[k + 1]
        if hi - lo < 1:
            continue
        train_hi = lo - horizon - embargo
        if train_hi <= 0:
            continue
        train = idx[:train_hi] if expanding else idx[max(0, train_hi - min_train):train_hi]
        test = idx[lo:hi]
        if train.size and test.size:
            yield train, test
