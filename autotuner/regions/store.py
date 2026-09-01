"""Boundary store (plan 5.7): saved region-edge tensors on disk, keyed by
region fingerprint, workload, and input-set index. Safetensors keys are
"a<array_id>" against the priced trace's ids; workers stream sets from disk so
nothing large stays resident between evaluations.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx


class BoundaryStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, fingerprint: str, workload: str, set_idx: int, kind: str) -> Path:
        return self.root / fingerprint / workload / f"set{set_idx}.{kind}.safetensors"

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

    def set_count(self, fingerprint: str, workload: str, kind: str = "inputs") -> int:
        d = self.root / fingerprint / workload
        if not d.exists():
            return 0
        return len(list(d.glob(f"set*.{kind}.safetensors")))


def load_set(path: str | Path) -> dict[int, mx.array]:
    loaded = mx.load(str(path))
    return {int(k[1:]): v for k, v in loaded.items()}
