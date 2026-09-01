"""Append-only run log (plan 5.13). JSONL, elapsed seconds on every row.

Internal math is positive-means-faster; human-facing lines are signed
milliseconds where negative means faster. fmt_signed_ms is the one formatter
that converts, so the two conventions can never drift apart.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path


def wall_now() -> str:
    """Local wall-clock stamp for log rows; `t` carries the precision."""
    return datetime.now().astimezone().isoformat(timespec="seconds")

SIGN_CONVENTION = "human-facing signed ms: negative means faster than baseline"


def fmt_signed_ms(delta_faster_ms: float) -> str:
    """Render an internal positive-means-faster delta for humans."""
    return f"{-delta_faster_ms + 0.0:+.3f}ms"  # +0.0 keeps zero from printing as -0.000


class RunLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._t0 = time.perf_counter()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, **row: object) -> None:
        record = {"t": round(time.perf_counter() - self._t0, 3), "wall": wall_now(),
                  "kind": kind, **row}
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]
