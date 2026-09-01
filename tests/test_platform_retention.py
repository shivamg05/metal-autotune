"""Pins from spike_03_weakref_retention (PLATFORM.md): the liveness facts plan 5.1
builds retention detection on. If an mlx upgrade breaks any of these, freeze-time
liveness classification (consumed vs python_retained) silently misclassifies arrays,
so these tests must fail loudly instead.
"""

import gc
import weakref

import mlx.core as mx
import mlx.nn as nn
import pytest


def collect():
    for _ in range(3):
        gc.collect()


def walk_arrays(root):
    """The freeze-time snapshot walk in miniature: every mx.array reachable from
    an object's attributes/pytree, id -> path."""
    found = {}
    seen = set()
    stack = [(root, "model")]
    while stack:
        obj, path = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, mx.array):
            found.setdefault(id(obj), path)
            continue
        children = []
        if isinstance(obj, dict):  # nn.Module subclasses dict
            children += [(f"{path}.{k}", v) for k, v in obj.items()]
        if hasattr(obj, "__dict__"):
            children += [(f"{path}.{k}", v) for k, v in vars(obj).items()]
        if isinstance(obj, (list, tuple)):
            children += [(f"{path}[{i}]", v) for i, v in enumerate(obj)]
        for p, v in children:
            if isinstance(v, (mx.array, dict, list, tuple)) or hasattr(v, "__dict__"):
                stack.append((v, p))
    return found


def test_weakref_accepted_and_lazy_array_dies():
    """Protects plan 5.1's GC probe: retention is detected by dropping the
    recorder's references and probing weakrefs, so mx.array must accept
    weakref.ref and a lazy (never evaluated) array must die after del+gc."""
    a = mx.random.normal((8, 8))
    b = mx.random.normal((8, 8))
    lazy = a + b
    wr = weakref.ref(lazy)
    assert wr() is not None
    del lazy
    collect()
    assert wr() is None


def test_weakref_evaluated_array_dies():
    """Protects the same GC probe for materialized arrays: mid-record evaluation
    (a model calling .item() or mx.eval) must not make an array undetectable as
    released, or the recorder's memory warning and liveness both go wrong."""
    ev = mx.random.normal((8, 8)) * 2.0
    mx.eval(ev)
    wr = weakref.ref(ev)
    assert wr() is not None
    del ev
    collect()
    assert wr() is None


def test_retained_and_consumed_array_found_by_attribute_walk():
    """Protects the freeze-time snapshot walk: an array that is BOTH consumed by
    later ops and retained in python state (the KV-append pattern) can only be
    classified python_retained because the walk finds it on the model. The step
    output, not stored on the model, must not be found (a consumed-only array
    must not be misclassified as retained)."""

    class CacheModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)
            self.cache = []

        def __call__(self, x):
            k = self.proj(x)
            self.cache.append(k)   # retained by python state
            return (k * 2.0) + x   # and consumed by later ops

    model = CacheModel()
    out = model(mx.random.normal((1, 4)))
    mx.eval(out)  # out really depends on k, so k was consumed
    k = model.cache[0]
    wr_k = weakref.ref(k)

    found = walk_arrays(model)
    assert found.get(id(k)) == "model.cache[0]"
    assert id(out) not in found

    # The GC probe agrees with the walk: dropping the python state releases k.
    del out
    model.cache.clear()
    del k
    collect()
    assert wr_k() is None


def test_module_reserves_state_attribute():
    """Protects fixture and wrapper naming: nn.Module reserves 'state' as a
    property, so assigning self.state raises AttributeError. Generated wrappers
    and the fixture zoo avoid that name; this pin flags an mlx change to the
    reservation."""

    class StateName(nn.Module):
        def __init__(self):
            super().__init__()
            self.state = {}

    with pytest.raises(AttributeError, match="state"):
        StateName()
