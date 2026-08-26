"""The trial log: the part of the protocol that makes the statistics honest.

A deflated Sharpe is only as good as the trial count you feed it, and the trial
count people remember is always far below the truth -- every abandoned lookback,
every "let me just try it winsorised at 2% instead", every re-run with a
different universe filter is a trial. None of them feel like trials at the time.

So this writes them down. Append-only JSONL, keyed by a content hash of the
config so re-running the identical experiment does not inflate the count but
changing any parameter does.

Usage is deliberately blunt: you cannot get a deflated Sharpe out of
``qr.evaluate`` without a log, because the number would be a lie.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


@dataclass
class Trial:
    trial_id: str
    hypothesis: str
    config: dict
    result: dict
    timestamp: float
    family: str = "default"


class TrialLog:
    """Append-only record of every experiment run against a dataset.

    ``family`` groups variants of the same underlying idea. ``count(family=...)``
    gives the trial count to deflate against: use the family count when the
    question is "is this signal real", and the total when the question is "is
    anything in my research programme real".
    """

    def __init__(self, path: str = "research_log.jsonl"):
        self.path = path
        self._seen: set[str] = set()
        if os.path.exists(path):
            for row in self._read():
                self._seen.add(row["trial_id"])

    def _read(self):
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out

    @staticmethod
    def trial_id(hypothesis: str, config: dict) -> str:
        return hashlib.sha256(
            (_canonical(hypothesis) + "|" + _canonical(config)).encode()
        ).hexdigest()[:16]

    def record(self, hypothesis: str, config: dict, result: dict,
               family: str = "default") -> Trial:
        """Log one experiment. Idempotent on (hypothesis, config)."""
        tid = self.trial_id(hypothesis, config)
        t = Trial(tid, hypothesis, dict(config), dict(result), time.time(), family)
        if tid in self._seen:
            return t
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(_canonical(asdict(t)) + "\n")
        self._seen.add(tid)
        return t

    def count(self, family: str | None = None) -> int:
        rows = self._read()
        if family is None:
            return len(rows)
        return sum(1 for r in rows if r.get("family") == family)

    def sharpes(self, family: str | None = None) -> list[float]:
        """Every logged Sharpe, for estimating the trial-Sharpe dispersion."""
        out = []
        for r in self._read():
            if family is not None and r.get("family") != family:
                continue
            v = r.get("result", {}).get("sharpe")
            if isinstance(v, (int, float)) and v == v:
                out.append(float(v))
        return out

    def sr_std(self, family: str | None = None, default: float = 1.0) -> float:
        """Cross-sectional std of logged Sharpes; the scale for `expected_max_sharpe`."""
        import numpy as np
        s = self.sharpes(family)
        if len(s) < 3:
            return default
        v = float(np.std(s, ddof=1))
        return v if v > 0 else default

    def summary(self) -> str:
        rows = self._read()
        fams: dict[str, int] = {}
        for r in rows:
            fams[r.get("family", "default")] = fams.get(r.get("family", "default"), 0) + 1
        lines = [f"{len(rows)} trials logged at {self.path}"]
        for k, v in sorted(fams.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {k:<30} {v}")
        return "\n".join(lines)
