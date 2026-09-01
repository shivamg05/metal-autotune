"""Spike 03: weakref/GC probe on mx.array plus the positive retention snapshot walk.

Proves the plan 5.1 liveness bullets by experiment on this machine:
- mx.array accepts weakref.ref; the ref dies after all strong refs drop plus
  gc.collect(), for a lazy array and an evaluated one.
- A toy stateful model can retain AND consume an array (KV-append pattern), and a
  snapshot walk over the model object's attributes/pytree is what finds it.
- The walk finds arrays nested in lists, dicts, and tuples on module attributes.

Output: one line per fact, `FACT <slug>: PASS|FAIL|INFO - <detail>`. Exit 0 if the
script ran to completion, even with FAILs.
"""

import gc
import sys
import traceback

import mlx.core as mx
import mlx.nn as nn


def fact(slug, status, detail):
    print(f"FACT {slug}: {status} - {detail}")


def collect():
    for _ in range(3):
        gc.collect()


def snapshot_walk(root):
    """Every mx.array reachable from the object's attributes/pytree: id -> first path."""
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
        if isinstance(obj, dict):  # mlx nn.Module subclasses dict
            children += [(f"{path}.{k}", v) for k, v in obj.items()]
        if hasattr(obj, "__dict__"):
            children += [(f"{path}.{k}", v) for k, v in vars(obj).items()]
        if isinstance(obj, (list, tuple)):
            children += [(f"{path}[{i}]", v) for i, v in enumerate(obj)]
        for p, v in children:
            if isinstance(v, (mx.array, dict, list, tuple)) or hasattr(v, "__dict__"):
                stack.append((v, p))
    return found


class CacheModel(nn.Module):
    """Toy stateful model: a KV-style python list cache retained on the module."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.cache = []             # list attribute
        self.store = {"hist": []}   # dict attribute with a nested list
        self.snap = None            # becomes a tuple holding an array

    def __call__(self, x):
        k = self.proj(x)
        self.cache.append(k)                 # k retained by python state
        self.store["hist"].append(k * 1.0)   # a second array retained inside a dict
        self.snap = (mx.sum(k),)             # a third retained inside a tuple
        return (k * 2.0) + x                 # k consumed by later ops


def run():
    import weakref

    mx.random.seed(0)

    # Fact: weakref.ref is accepted by mx.array, lazy and evaluated.
    a = mx.random.normal((8, 8))
    b = mx.random.normal((8, 8))
    lazy = a + b            # never evaluated
    ev = a * 2.0
    mx.eval(ev)             # materialized
    try:
        wr_lazy = weakref.ref(lazy)
        wr_ev = weakref.ref(ev)
        fact("weakref-accepted", "PASS",
             f"weakref.ref succeeded on lazy (a+b) and evaluated (mx.eval'd) arrays, type={type(lazy).__name__}")
    except TypeError as e:
        fact("weakref-accepted", "FAIL", f"weakref.ref raised TypeError: {e}")
        fact("weakref-lazy-dies", "FAIL", "skipped: weakref not accepted")
        fact("weakref-evaluated-dies", "FAIL", "skipped: weakref not accepted")
        wr_lazy = wr_ev = None

    if wr_lazy is not None:
        alive_before = wr_lazy() is not None
        del lazy
        collect()
        dead_after = wr_lazy() is None
        fact("weakref-lazy-dies", "PASS" if (alive_before and dead_after) else "FAIL",
             f"lazy array: alive while referenced={alive_before}, dead after del+gc.collect()={dead_after}")

        alive_before = wr_ev() is not None
        del ev
        collect()
        dead_after = wr_ev() is None
        fact("weakref-evaluated-dies", "PASS" if (alive_before and dead_after) else "FAIL",
             f"evaluated array: alive while referenced={alive_before}, dead after del+gc.collect()={dead_after}")

    # Found while building this spike: nn.Module reserves `state` as a property,
    # so a model attribute named `state` cannot be assigned. Pin the observation.
    class StateName(nn.Module):
        def __init__(self):
            super().__init__()
            self.state = {}

    try:
        StateName()
        fact("module-state-attr-reserved", "INFO", "assigning self.state on a Module subclass worked")
    except AttributeError as e:
        fact("module-state-attr-reserved", "INFO",
             f"assigning self.state on a Module subclass raises AttributeError ({e}); "
             "toy fixtures must avoid that attribute name")

    # Positive retention check on the toy stateful model.
    model = CacheModel()
    x = mx.random.normal((1, 4))
    out = model(x)
    k = model.cache[0]              # same python object the forward appended
    wr_k = weakref.ref(k)
    hist_arr = model.store["hist"][0]
    snap_arr = model.snap[0]

    found = snapshot_walk(model)
    fact("walk-array-count", "INFO",
         f"snapshot walk reached {len(found)} distinct arrays on the model object")

    k_path = found.get(id(k))
    mx.eval(out)                    # k really is consumed: the output depends on it
    consumed_ok = out.shape == (1, 4)
    if k_path is not None and wr_k() is not None and consumed_ok:
        fact("retained-and-consumed", "PASS",
             f"k appended to cache AND used in later op; walk found it at {k_path}; "
             f"weakref alive after forward=True; out=f(k) evaluated to shape {out.shape}")
    else:
        fact("retained-and-consumed", "FAIL",
             f"walk path={k_path}, weakref alive={wr_k() is not None}, consumed_ok={consumed_ok}")

    # Nested containers: list, dict-nested list, tuple, and a direct parameter.
    hist_path = found.get(id(hist_arr))
    snap_path = found.get(id(snap_arr))
    weight_path = found.get(id(model.proj.weight))
    nested_ok = all(p is not None for p in (k_path, hist_path, snap_path, weight_path))
    fact("walk-finds-nested-containers", "PASS" if nested_ok else "FAIL",
         f"list={k_path}, dict-nested={hist_path}, tuple={snap_path}, parameter={weight_path}")

    # Negative control: the step output is not on the model, so the walk must not find it.
    fact("walk-negative-control", "PASS" if id(out) not in found else "FAIL",
         f"step output (not stored on model) found by walk={id(out) in found}")

    # Dropping the python state releases k: the GC probe agrees with the walk.
    del out
    model.cache.clear()
    model.store["hist"].clear()
    model.snap = None
    del k, hist_arr, snap_arr, found
    collect()
    fact("retention-clears-on-drop", "PASS" if wr_k() is None else "FAIL",
         f"after clearing cache/store/snap and gc.collect(), k weakref dead={wr_k() is None}")


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
