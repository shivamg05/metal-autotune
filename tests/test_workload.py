"""Seeded workload materialization. Determinism is the load-bearing property."""

import hashlib
import subprocess
import sys
import textwrap

import mlx.core as mx
import pytest

from autotuner.manifest import BOUNDARY_INPUT_SETS, InputSpec, Workload
from autotuner.workload import bind_shape, materialize, workload_seeds


def spec(shape, dtype="float32", **kw):
    return InputSpec(shape=tuple(shape), dtype=dtype, **kw)


def test_bind_shape():
    assert bind_shape((8, "L"), {"L": 512}) == (8, 512)
    assert bind_shape((4, 4), {}) == (4, 4)
    with pytest.raises(ValueError, match="no binding"):
        bind_shape((8, "L"), {"M": 2})


def test_workload_seeds_stable_and_distinct():
    a = workload_seeds(1, "decode", 3)
    assert a == workload_seeds(1, "decode", 3)
    assert len(set(a)) == 3
    assert a != workload_seeds(1, "midbatch", 3)
    assert a != workload_seeds(2, "decode", 3)


def test_materialize_shapes_and_dtypes():
    w = Workload(
        inputs=(
            spec([2, "L"], "float16"),
            spec([3], "bfloat16"),
            spec(["L", 4], "int32", low=5, high=9),
            spec([2, 2], "bool"),
        ),
        name="w",
    )
    t = materialize(w, {"L": 7}, seed=42)
    assert [x.shape for x in t] == [(2, 7), (3,), (7, 4), (2, 2)]
    assert [x.dtype for x in t] == [mx.float16, mx.bfloat16, mx.int32, mx.bool_]
    assert mx.all(t[2] >= 5).item() and mx.all(t[2] < 9).item()


def test_materialize_deterministic_in_process():
    w = Workload(inputs=(spec([16, 16]), spec([8], "int32")), name="w")
    a = materialize(w, {}, seed=7)
    b = materialize(w, {}, seed=7)
    c = materialize(w, {}, seed=8)
    assert all(mx.array_equal(x, y).item() for x, y in zip(a, b))
    assert not mx.array_equal(a[0], c[0]).item()


def _digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(bytes(memoryview(t)))
    return h.hexdigest()


def test_materialize_byte_identical_across_processes():
    """The same manifest and seed materialize byte-identical tensors in two
    processes."""
    w = Workload(inputs=(spec([32, 64]), spec([4, "L"], "int32")), name="w")
    here = _digest(materialize(w, {"L": 13}, seed=1234))
    child_src = textwrap.dedent("""
        import hashlib
        import mlx.core as mx
        from autotuner.manifest import InputSpec, Workload
        from autotuner.workload import materialize
        w = Workload(inputs=(
            InputSpec(shape=(32, 64), dtype="float32"),
            InputSpec(shape=(4, "L"), dtype="int32"),
        ), name="w")
        h = hashlib.sha256()
        for t in materialize(w, {"L": 13}, seed=1234):
            h.update(bytes(memoryview(t)))
        print(h.hexdigest())
    """)
    proc = subprocess.run([sys.executable, "-c", child_src], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == here


def test_k_seed_sets_differ():
    w = Workload(inputs=(spec([16, 16]),), name="w")
    seeds = workload_seeds(0, "w", BOUNDARY_INPUT_SETS)
    sets = [materialize(w, {}, s) for s in seeds]
    digests = {_digest(t) for t in sets}
    assert len(digests) == BOUNDARY_INPUT_SETS
