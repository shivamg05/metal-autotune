"""Pinned platform facts from spike_07 (validation env vars) and spike_09
(mx.compile caching). Each test names the design argument it protects; if an
mlx or macOS upgrade changes the behavior, the design must be revisited, so
the test must fail loudly. Behavior only, never timing.
"""

import os
import subprocess
import sys

import mlx.core as mx
import mlx.nn as nn

# Child for the launch-time-only env var facts. Reads 48KB past the end of a
# 16KB buffer. With --set-env-then-retry it sets both validation vars
# mid-process and reruns under a fresh kernel name, so a fresh pipeline
# compiles after the set.
CHILD_OOB = """
import os, sys
import mlx.core as mx

SIZE = 4096
OFF = 4 * SIZE

def oob_read(tag):
    inp = mx.full((SIZE,), 1.0, dtype=mx.float32)
    mx.eval(inp)
    k = mx.fast.metal_kernel(
        name=f"pin_oob_{tag}", input_names=["inp"], output_names=["out"],
        source=f"uint i = thread_position_in_grid.x; out[i] = inp[i + {OFF}u];")
    (out,) = k(inputs=[inp], output_shapes=[(SIZE,)], output_dtypes=[mx.float32],
               grid=(SIZE, 1, 1), threadgroup=(256, 1, 1))
    mx.eval(out)
    mx.synchronize()
    print(tag, "zeros", int(mx.sum(out == 0).item()), flush=True)

oob_read("first")
if "--set-env-then-retry" in sys.argv:
    os.environ["MTL_SHADER_VALIDATION"] = "1"
    os.environ["MTL_SHADER_VALIDATION_REPORT_TO_STDERR"] = "1"
    oob_read("second")
"""

VALIDATION_VARS = ("MTL_SHADER_VALIDATION", "MTL_SHADER_VALIDATION_REPORT_TO_STDERR")


def run_oob_child(args=(), extra_env=None):
    env = {k: v for k, v in os.environ.items() if k not in VALIDATION_VARS}
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", CHILD_OOB, *args],
        capture_output=True, text=True, env=env, timeout=60)


def test_oob_read_without_validation_is_silent_zeros():
    """Protects: the ladder's poison gate and validate mode exist because score
    mode is blind to OOB. An unvalidated OOB read zero-fills silently: exit 0,
    all zeros, no stderr diagnostic. Correctness must rest on value comparison.
    """
    p = run_oob_child()
    assert p.returncode == 0
    assert "first zeros 4096" in p.stdout
    assert "Invalid device load" not in p.stderr


def test_validation_with_stderr_report_at_launch_flags_oob_read():
    """Protects: validate mode is BOTH env vars at subprocess launch plus
    parsing child stderr for 'Invalid device load' (spike_07).
    Validation zerofills and never faults, so exit codes carry no signal.
    """
    p = run_oob_child(extra_env={
        "MTL_SHADER_VALIDATION": "1",
        "MTL_SHADER_VALIDATION_REPORT_TO_STDERR": "1"})
    assert p.returncode == 0
    assert "first zeros 4096" in p.stdout
    assert "Invalid device load" in p.stderr


def test_validation_env_set_mid_process_changes_nothing():
    """Protects: the sandbox law that every kernel evaluation runs out of
    process. Metal reads the validation vars at process launch; setting them
    mid-process, even with a fresh pipeline compiled after, reports nothing.
    """
    p = run_oob_child(args=("--set-env-then-retry",))
    assert p.returncode == 0
    assert "first zeros 4096" in p.stdout
    assert "second zeros 4096" in p.stdout
    assert "Invalid device load" not in p.stderr


class _Affine(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.weight = mx.array(float(scale))

    def __call__(self, x):
        return x * self.weight


class _Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = _Affine(2.0)

    def __call__(self, x):
        return self.child(x)


PRE_SWAP = [2.0, 4.0, 6.0, 8.0]
POST_SWAP = [3.0, 6.0, 9.0, 12.0]


def swapped_model_and_input():
    # compile a step, run it once, then swap the child module 2x -> 3x
    model = _Parent()
    x = mx.array([1.0, 2.0, 3.0, 4.0])

    def step(inp):
        return model(inp)

    compiled = mx.compile(step)
    mx.eval(compiled(x))
    model.child = _Affine(3.0)
    return model, x, step, compiled


def test_compiled_callable_is_stale_after_module_swap():
    """Protects: bind must never run a swap through a callable compiled before
    it. The already-compiled step keeps replaying the pre-swap graph while
    eager calls see the swap, so post-swap verification retraces from scratch.
    """
    model, x, step, compiled = swapped_model_and_input()
    y_eager = step(x)
    y_stale = compiled(x)
    mx.eval(y_eager, y_stale)
    assert y_eager.tolist() == POST_SWAP
    assert y_stale.tolist() == PRE_SWAP


def test_recompiling_same_function_object_stays_stale_while_old_alive():
    """Protects: the fresh-callable rule after a swap must be stronger than
    'call mx.compile again'.
    The compile cache is keyed on function object identity and kept while any
    old compiled object is alive, so recompiling step returns the stale graph.
    """
    model, x, step, compiled = swapped_model_and_input()
    recompiled = mx.compile(step)
    y = recompiled(x)
    mx.eval(y)
    assert y.tolist() == PRE_SWAP


def test_newly_defined_closure_sees_the_swap():
    """Protects: the harness's verified post-swap remedy. Compiling a newly
    defined function object over the same model retraces and sees the swap,
    even while the stale compiled object is still alive.
    """
    model, x, step, compiled = swapped_model_and_input()

    def fresh_step(inp):
        return model(inp)

    y = mx.compile(fresh_step)(x)
    mx.eval(y)
    assert y.tolist() == POST_SWAP


def test_compiled_per_shape_retrace():
    """Protects: the shape sweep design. A compiled step reruns its python
    body exactly once per distinct input shape, not per call, so per-shape
    recompiles are automatic and the tracer must expect the body to
    re-execute inside compiled sections when a new shape arrives.
    """
    traced = []

    def step(inp):
        traced.append(tuple(inp.shape))
        return inp * 2.0

    compiled = mx.compile(step)
    a = mx.zeros((2, 3))
    b = mx.zeros((4, 3))
    for inp in (a, a, b, b, a):
        mx.eval(compiled(inp))
    assert traced == [(2, 3), (4, 3)]
