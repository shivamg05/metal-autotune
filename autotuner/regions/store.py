"""Boundary store: saved region-edge tensors on disk, keyed by
region fingerprint, workload, and input-set index. Safetensors keys are
"a<array_id>" against the priced trace's ids; workers stream sets from disk so
nothing large stays resident between evaluations.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote
import json
import struct

import mlx.core as mx


class BoundaryMismatch(RuntimeError):
    """Saved evaluation data does not match the recorded region boundary."""


class BoundaryStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, fingerprint: str, workload: str, set_idx: int, kind: str) -> Path:
        # Labels are metadata, not paths. Keep distinct labels distinct even
        # when they contain slashes, percent signs or traversal components.
        label = quote(workload, safe="")
        if label in {".", ".."}:
            label = "%2E" * len(label)
        return self.root / fingerprint / label / f"set{set_idx}.{kind}.safetensors"

    def save(self, fingerprint: str, workload: str, set_idx: int, kind: str,
             arrays: dict[int, mx.array]) -> Path:
        """kind is 'inputs' or 'outputs' (the library's reference values)."""
        path = self._path(fingerprint, workload, set_idx, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        mx.eval(list(arrays.values()))
        mx.save_safetensors(str(path), {f"a{aid}": arr for aid, arr in arrays.items()})
        return path

    def load(self, fingerprint: str, workload: str, set_idx: int, kind: str) -> dict[int, mx.array]:
        return load_set(self._path(fingerprint, workload, set_idx, kind))

    def set_count(self, fingerprint: str, workload: str) -> int:
        d = self._path(fingerprint, workload, 0, "inputs").parent
        if not d.exists():
            return 0
        return len(list(d.glob("set*.inputs.safetensors")))

    def validate(self, fingerprint: str, workload: str, set_idx: int, kind: str,
                 expected_ids: tuple[int, ...]) -> None:
        """Check the safetensors header without loading weights or using the GPU."""
        path = self._path(fingerprint, workload, set_idx, kind)
        try:
            size = path.stat().st_size
            with path.open("rb") as f:
                length, = struct.unpack("<Q", f.read(8))
                if length > size - 8:
                    raise ValueError("truncated tensor header")
                header = json.loads(f.read(length))
            expected = {f"a{a}" for a in expected_ids}
            actual = set(header) - {"__metadata__"}
            if actual != expected:
                raise ValueError(f"missing arrays {sorted(expected - actual)}; "
                                 f"unexpected arrays {sorted(actual - expected)}")
            for key in actual:
                lo, hi = header[key]["data_offsets"]
                if not 0 <= lo <= hi <= size - 8 - length:
                    raise ValueError(f"truncated or invalid tensor data for {key}")
        except (OSError, ValueError, TypeError, KeyError, struct.error) as error:
            raise BoundaryMismatch(f"Saved boundary data mismatch at {path}: {error}") from error


def load_set(path: str | Path) -> dict[int, mx.array]:
    loaded = mx.load(str(path))
    return {int(k[1:]): v for k, v in loaded.items()}
