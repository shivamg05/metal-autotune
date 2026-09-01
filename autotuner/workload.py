"""Workload tensors: bind named dims and materialize seeded inputs.

The harness fixes the synthesis seeds (derived from the job seed, recorded);
the manifest never supplies them. The same manifest and seed must materialize
byte-identical tensors in any process, which a pinned test proves.
"""

from __future__ import annotations

import hashlib
from typing import Mapping

import mlx.core as mx

from .manifest import FLOAT_DTYPES, INT_DTYPES, InputSpec, Manifest, Workload, resolve_dtype


def bind_shape(shape: tuple[int | str, ...], dims: Mapping[str, int]) -> tuple[int, ...]:
    """Replace named dims with concrete sizes; integer dims pass through."""
    bound: list[int] = []
    for d in shape:
        if isinstance(d, str):
            if d not in dims:
                raise ValueError(f"named dim {d!r} has no binding in {dict(dims)}")
            bound.append(dims[d])
        else:
            bound.append(d)
    return tuple(bound)


def workload_seeds(job_seed: int, workload_name: str, k: int) -> list[int]:
    """k stable per-workload seeds. Hash-derived so workload order never matters."""
    seeds = []
    for i in range(k):
        digest = hashlib.sha256(f"{job_seed}:{workload_name}:{i}".encode()).digest()
        seeds.append(int.from_bytes(digest[:8], "big"))
    return seeds


def _materialize_input(spec: InputSpec, shape: tuple[int, ...], key: mx.array) -> mx.array:
    dtype = resolve_dtype(spec.dtype)
    if spec.dtype in FLOAT_DTYPES:
        return mx.random.normal(shape, dtype=dtype, key=key)
    if spec.dtype in INT_DTYPES:
        return mx.random.randint(spec.low, spec.high, shape, dtype=dtype, key=key)
    if spec.dtype == "bool":
        return mx.random.randint(0, 2, shape, key=key).astype(mx.bool_)
    raise ValueError(f"cannot synthesize dtype {spec.dtype!r}")


def materialize(workload: Workload, dims: Mapping[str, int], seed: int) -> list[mx.array]:
    """One tensor per positional argument of the model, deterministically from seed."""
    keys = mx.random.split(mx.random.key(seed), len(workload.inputs))
    tensors = []
    for i, spec in enumerate(workload.inputs):
        shape = bind_shape(spec.shape, dims)
        tensors.append(_materialize_input(spec, shape, keys[i]))
    mx.eval(tensors)
    return tensors


def primary_tensors(manifest: Manifest, workload: Workload, k: int) -> list[list[mx.array]]:
    """The k input sets for one workload at its primary dim binding."""
    seeds = workload_seeds(manifest.seed, workload.name, k)
    return [materialize(workload, manifest.primary, s) for s in seeds]
