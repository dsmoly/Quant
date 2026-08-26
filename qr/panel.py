"""Aligned panel data, and the single definition of a forward return.

Every silent look-ahead bug this package could have has the same shape: a
feature observed at date t evaluated against a return that was already
partially known at t. So forward returns are built in exactly one function,
and that function is explicit about the convention:

    forward_return(returns, horizon=h)[t] = sum of returns[t+1 : t+1+h]

A feature observed at the *close* of date t is therefore matched against the
return earned from the close of t to the close of t+h. There is no overlap
between the information set at t and the return being predicted. If your
feature is only knowable at the open of t+1, shift it yourself before
handing it over -- the harness cannot know that and will not guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _as2d(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D (T, N), got shape {a.shape}")
    return a


@dataclass
class Panel:
    """A (T, N) panel: returns, an optional universe mask, and named features.

    ``mask[t, i]`` is True when name *i* is tradable on date *t*. Everything in
    the package respects it; a name outside the mask contributes to no
    statistic and receives no position. Defaults to "wherever returns and the
    feature are both finite", which is right for research and wrong for live
    (where you also need borrow, halts and listing status) -- so pass it
    explicitly once you have that data.
    """

    returns: np.ndarray
    mask: np.ndarray | None = None
    features: dict[str, np.ndarray] = field(default_factory=dict)
    dates: np.ndarray | None = None
    ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.returns = _as2d(self.returns, "returns")
        T, N = self.returns.shape
        if self.mask is None:
            self.mask = np.isfinite(self.returns)
        else:
            self.mask = np.asarray(self.mask, dtype=bool)
            if self.mask.shape != (T, N):
                raise ValueError(f"mask shape {self.mask.shape} != returns shape {(T, N)}")
        for k, v in list(self.features.items()):
            v = _as2d(v, f"features[{k!r}]")
            if v.shape != (T, N):
                raise ValueError(f"feature {k!r} shape {v.shape} != returns shape {(T, N)}")
            self.features[k] = v
        if self.dates is not None:
            self.dates = np.asarray(self.dates)
            if len(self.dates) != T:
                raise ValueError("dates length must match returns rows")
        if self.ids is not None:
            self.ids = np.asarray(self.ids)
            if len(self.ids) != N:
                raise ValueError("ids length must match returns columns")

    @property
    def shape(self) -> tuple[int, int]:
        return self.returns.shape

    @property
    def T(self) -> int:
        return self.returns.shape[0]

    @property
    def N(self) -> int:
        return self.returns.shape[1]

    def add(self, name: str, values) -> "Panel":
        v = _as2d(values, name)
        if v.shape != self.shape:
            raise ValueError(f"feature {name!r} shape {v.shape} != panel shape {self.shape}")
        self.features[name] = v
        return self

    def forward(self, horizon: int = 1) -> np.ndarray:
        """Forward return over ``horizon`` periods; see module docstring."""
        return forward_return(self.returns, horizon)

    def valid(self, *feature_names: str) -> np.ndarray:
        """Mask AND finite-ness of the named features. The evaluation universe."""
        m = self.mask.copy()
        for n in feature_names:
            if n not in self.features:
                raise KeyError(f"no feature named {n!r}")
            m &= np.isfinite(self.features[n])
        return m


def forward_return(returns, horizon: int = 1) -> np.ndarray:
    """Sum of the next ``horizon`` returns, NaN where the window runs off the end.

    Simple (not log) returns are summed rather than compounded. Over the
    horizons this package is aimed at (1-60 days) the difference is third-order
    and summing keeps the estimator linear, which matters for the decay fit.
    """
    r = _as2d(returns, "returns")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    T = r.shape[0]
    out = np.full_like(r, np.nan)
    if horizon >= T:
        return out
    # cumulative with a leading zero row so slices are unambiguous
    c = np.vstack([np.zeros((1, r.shape[1])), np.nancumsum(r, axis=0)])
    # rows t = 0 .. T-1-horizon get sum of r[t+1 .. t+horizon]
    hi = np.arange(1 + horizon, T + 1)
    lo = np.arange(1, T - horizon + 1)
    out[: T - horizon] = c[hi] - c[lo]
    # NaN-propagate: a window containing a missing return is not a return
    nan_c = np.vstack([np.zeros((1, r.shape[1])), np.cumsum(~np.isfinite(r), axis=0)])
    bad = (nan_c[hi] - nan_c[lo]) > 0
    out[: T - horizon][bad] = np.nan
    return out
