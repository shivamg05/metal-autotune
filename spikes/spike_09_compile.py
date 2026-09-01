"""Spike 09: mx.compile behavior on mlx (plan section 2 baseline choice, law 10, M0 "mx.compile").

Proves by experiment, on the installed mlx version:
- an already-compiled callable keeps returning the pre-swap graph after a child module swap
- whether a fresh mx.compile of the same callable reflects the swap
- what a compiled function does when module state changes between calls
- the python body reruns once per distinct input shape, not per call
- plain vs compiled step time on a small MLP, paired and interleaved
- what a compiled fn does when it internally calls another compiled fn
"""

import gc
import statistics
import sys
import time
import traceback

import mlx.core as mx
import mlx.nn as nn

mx.random.seed(0)


def fact(slug, status, detail):
    print(f"FACT {slug}: {status} - {detail}")


class Affine(nn.Module):
    """y = x * weight, one scalar parameter."""

    def __init__(self, scale):
        super().__init__()
        self.weight = mx.array(float(scale))

    def __call__(self, x):
        return x * self.weight


class Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = Affine(2.0)

    def __call__(self, x):
        return self.child(x)


def check_identity_caching():
    model = Parent()
    x = mx.array([1.0, 2.0, 3.0, 4.0])
    want_pre = x * 2.0
    want_post = x * 3.0

    def step(inp):
        return model(inp)

    compiled1 = mx.compile(step)
    y_pre = compiled1(x)
    mx.eval(y_pre, want_pre, want_post)

    model.child = Affine(3.0)
    y_eager = step(x)
    y_stale = compiled1(x)
    mx.eval(y_eager, y_stale)
    eager_sees_swap = bool(mx.array_equal(y_eager, want_post).item())
    stale = bool(mx.array_equal(y_stale, want_pre).item())
    fact(
        "compiled-callable-stale-after-swap",
        "PASS" if stale and eager_sees_swap else "FAIL",
        f"after child swap 2x->3x: eager step gives {y_eager.tolist()}, the already-compiled "
        f"callable gives {y_stale.tolist()} (pre-swap graph={stale})",
    )

    # recompile the very same function object while the old compiled object is alive
    compiled2 = mx.compile(step)
    y_fresh = compiled2(x)
    mx.eval(y_fresh)
    reflects = bool(mx.array_equal(y_fresh, want_post).item())
    fact(
        "fresh-compile-same-callable-reflects-swap",
        "PASS" if reflects else "FAIL",
        f"mx.compile(step) again (old compiled object still alive) gives {y_fresh.tolist()}, "
        f"want post-swap {want_post.tolist()}",
    )

    del compiled1, compiled2
    gc.collect()
    compiled3 = mx.compile(step)
    y_dropped = compiled3(x)
    mx.eval(y_dropped)
    sees_swap = bool(mx.array_equal(y_dropped, want_post).item())
    fact(
        "fresh-compile-after-dropping-old",
        "INFO",
        f"after deleting both old compiled objects, mx.compile(step) gives {y_dropped.tolist()} "
        f"(sees the swap={sees_swap})",
    )

    def step_b(inp):
        return model(inp)

    compiled4 = mx.compile(step_b)
    y_new = compiled4(x)
    mx.eval(y_new)
    sees_swap = bool(mx.array_equal(y_new, want_post).item())
    fact(
        "fresh-compile-new-closure",
        "INFO",
        f"compiling a newly defined closure over the same model gives {y_new.tolist()} "
        f"(sees the swap={sees_swap})",
    )


class Stateful(nn.Module):
    """Reads and writes non-parameter state on every call."""

    def __init__(self):
        super().__init__()
        self.count = 0
        self.gain = mx.array(2.0)

    def __call__(self, x):
        self.count += 1
        return x * self.gain + float(self.count)


def check_state_freezing():
    model = Stateful()
    x = mx.array([1.0, 2.0])

    def step(inp):
        return model(inp)

    compiled = mx.compile(step)
    y1 = compiled(x)
    mx.eval(y1)
    y2 = compiled(x)
    mx.eval(y2)
    identical = bool(mx.array_equal(y1, y2).item())
    fact(
        "attr-counter-frozen",
        "INFO",
        f"__call__ increments a python counter: compiled call1={y1.tolist()} call2={y2.tolist()} "
        f"(identical={identical}), count attr={model.count} after 2 calls (body ran only at trace)",
    )

    model.gain = mx.array(5.0)
    y3 = compiled(x)
    mx.eval(y3)
    frozen = bool(mx.array_equal(y3, y1).item())
    fact(
        "array-attr-reassign-frozen",
        "INFO",
        f"gain reassigned 2.0->5.0 between calls: compiled returns {y3.tolist()} "
        f"(old array baked into the trace={frozen})",
    )


def check_per_shape_retrace():
    traced_shapes = []

    def step(inp):
        traced_shapes.append(tuple(inp.shape))
        return inp * 2.0

    compiled = mx.compile(step)
    a = mx.zeros((2, 3))
    b = mx.zeros((4, 3))
    for inp in (a, a, b, b, a):
        mx.eval(compiled(inp))
    ok = traced_shapes == [(2, 3), (4, 3)]
    fact(
        "per-shape-retrace",
        "PASS" if ok else "FAIL",
        f"5 calls over shapes [(2,3),(2,3),(4,3),(4,3),(2,3)] ran the python body for "
        f"{traced_shapes} (want once per distinct shape)",
    )


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = [nn.Linear(dim, dim) for _ in range(4)]
        self.norm = nn.LayerNorm(dim)

    def __call__(self, x):
        for layer in self.layers:
            x = nn.gelu(layer(x))
        return self.norm(x)


def sample_ms(fn, xs, offset, inner=20):
    # one sample: inner calls with rotating inputs, all outputs evaluated
    outs = []
    mx.synchronize()
    t0 = time.perf_counter()
    for j in range(inner):
        outs.append(fn(xs[(offset + j) % len(xs)]))
    mx.eval(outs)
    mx.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / inner


def warm_until_stable(fn, xs, cap=30):
    prev = sample_ms(fn, xs, 0)
    for i in range(1, cap):
        cur = sample_ms(fn, xs, i)
        if abs(cur - prev) / prev < 0.01:
            return i + 1
        prev = cur
    return cap


def check_plain_vs_compiled():
    model = MLP(512)
    mx.eval(model.parameters())
    xs = [mx.random.normal((32, 512)) for _ in range(4)]
    mx.eval(xs)

    def plain(inp):
        return model(inp)

    compiled = mx.compile(lambda inp: model(inp))
    warm_p = warm_until_stable(plain, xs)
    warm_c = warm_until_stable(compiled, xs)

    pairs = 40
    plain_ms, comp_ms, deltas = [], [], []
    for i in range(pairs):
        # alternate order so drift is common-mode
        if i % 2 == 0:
            tp = sample_ms(plain, xs, i)
            tc = sample_ms(compiled, xs, i)
        else:
            tc = sample_ms(compiled, xs, i)
            tp = sample_ms(plain, xs, i)
        plain_ms.append(tp)
        comp_ms.append(tc)
        deltas.append((tp - tc) / tp * 100.0)

    med_p = statistics.median(plain_ms)
    med_c = statistics.median(comp_ms)
    med_d = statistics.median(deltas)
    fact(
        "plain-vs-compiled-step",
        "INFO",
        f"4-layer MLP dim 512 batch 32: plain median {med_p:.3f} ms/step, compiled median "
        f"{med_c:.3f} ms/step, median paired delta {med_d:+.1f}% (positive means compiled "
        f"faster), warmed in {warm_p}/{warm_c} samples over {pairs} interleaved pairs",
    )


def check_nested_compile():
    inner_runs, outer_runs = [], []

    def inner(inp):
        inner_runs.append(1)
        return inp * 2.0 + 1.0

    inner_c = mx.compile(inner)
    x = mx.array([1.0, 2.0, 3.0])
    mx.eval(inner_c(x))
    runs_before = len(inner_runs)

    def outer(inp):
        outer_runs.append(1)
        return inner_c(inp) - 3.0

    outer_c = mx.compile(outer)
    try:
        y = outer_c(x)
        mx.eval(y)
        ref = x * 2.0 + 1.0 - 3.0
        correct = bool(mx.array_equal(y, ref).item())
        inner_reruns = len(inner_runs) - runs_before
        mx.eval(outer_c(x))
        fact(
            "nested-compile",
            "INFO",
            f"mx.compile(outer) calling a compiled inner: correct output={correct}, inner python "
            f"body ran {inner_reruns} more time(s) during the outer trace (shape already in the "
            f"inner cache), outer body ran {len(outer_runs)} time(s) over 2 calls",
        )
    except Exception as e:
        fact("nested-compile", "INFO", f"raises {type(e).__name__}: {e}")


def main():
    fact("mlx-version", "INFO", f"mlx {mx.__version__}, device {mx.default_device()}")
    check_identity_caching()
    check_state_freezing()
    check_per_shape_retrace()
    check_plain_vs_compiled()
    check_nested_compile()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
