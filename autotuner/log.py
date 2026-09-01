"""Append-only run log: one JSON line per event, elapsed seconds and wall
time on every row, plus a plain-text log for people."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path


def wall_now() -> str:
    """Local wall-clock stamp for log rows; `t` carries the precision."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def json_safe(obj):
    """Infinity and NaN are not JSON; write them as strings so every row parses."""
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return str(obj)
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj

class RunLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._t0 = time.perf_counter()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, **row: object) -> None:
        record = {"t": round(time.perf_counter() - self._t0, 3), "wall": wall_now(),
                  "kind": kind, **row}
        with self.path.open("a") as f:
            f.write(json.dumps(json_safe(record), default=str, allow_nan=False) + "\n")

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]


class TextLog:
    """Append-only plain-text log for people: one line per call."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, line: str) -> None:
        with self.path.open("a") as f:
            f.write(line.rstrip("\n") + "\n")
