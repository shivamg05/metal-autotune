"""Pinned platform facts from spike_07 (validation env vars) and spike_09
(mx.compile caching). Each test names the design argument it protects; if an
mlx or macOS upgrade changes the behavior, the design must be revisited, so
the test must fail loudly. Behavior only, never timing.
"""

import os
import subprocess
import sys

import mlx.core as mx
import pytest
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


def test_compiled_graph_accepts_a_custom_kernel_bitwise():
    """Protects: the compiled baseline. The patched model is timed under
    mx.compile, so a graph holding a custom metal kernel must compile and
    match the plain call bit for bit (spike_11)."""
    kernel = mx.fast.metal_kernel(
        name="pin_compile_custom", input_names=["inp"], output_names=["out"],
        source="uint i = thread_position_in_grid.x; out[i] = inp[i] * 2.0f + 1.0f;")

    class Patched(nn.Module):
        def __call__(self, x):
            y = kernel(inputs=[x], grid=(x.size, 1, 1), threadgroup=(256, 1, 1),
                       output_shapes=[x.shape], output_dtypes=[x.dtype])[0]
            return y - 0.5

    model = Patched()
    x = mx.random.normal((64, 128), key=mx.random.key(3))
    plain = model(x)
    compiled = mx.compile(lambda a: model(a))(x)
    mx.eval(plain, compiled)
    assert mx.array_equal(plain, compiled).item()


def test_compiled_elementwise_chain_matches_plain_in_fp16():
    """Protects: the compiled library arm. Pricing and the ship clock replay a
    region's ops as one compiled graph under the compiled baseline, so the
    fused graph must give the plain ops' bits, including in half precision
    (spike_11)."""
    a = mx.random.normal((64, 1024), key=mx.random.key(1)).astype(mx.float16)
    b = mx.random.normal((1024,), key=mx.random.key(2)).astype(mx.float16)

    def chain(x, w):
        y = mx.maximum(x * 2.0 + w, 0.0) * x
        return (mx.minimum(y + w, 8.0) - 1.0) * 0.5

    plain = chain(a, b)
    compiled = mx.compile(chain)(a, b)
    mx.eval(plain, compiled)
    assert mx.array_equal(plain, compiled).item()


def test_compile_cannot_swap_state_held_as_an_attribute():
    """Protects: the plain baseline for a step that keeps state, such as a KV
    cache. A buffer a model writes in place through a plain attribute is not
    declarable compile state: declaring it raises, and an undeclared compiled
    call leaves the buffer holding a tracer, so the next plain call raises
    (spike_11)."""
    import types

    class Stateful(nn.Module):
        def __init__(self):
            super().__init__()
            self.cache = types.SimpleNamespace(buf=mx.zeros((4,)))

        def __call__(self, t):
            self.cache.buf[1:2] = t[:1]
            return self.cache.buf * 2

    x = mx.array([1.0, 2.0, 3.0])
    m = Stateful()
    mx.eval(m(x))
    with pytest.raises(ValueError, match="uncaptured"):
        mx.eval(mx.compile(lambda t: m(t), inputs=[m.cache.buf], outputs=[m.cache.buf])(x))
    m = Stateful()
    mx.eval(m(x))
    mx.eval(mx.compile(lambda t: m(t))(x))  # the compiled call itself succeeds
    with pytest.raises(RuntimeError, match="without a primitive"):
        mx.eval(m(x))


def test_compile_swaps_state_held_in_a_dict():
    """Protects: how a model file can make a stateful step compilable if it
    wants the compiled baseline: keep the state in a dict or list and declare
    that container as inputs and outputs. An in-place write then flows
    through the compiled call, and the plain call still works after."""
    state = {"buf": mx.zeros((4,))}

    def step(t):
        state["buf"][1:2] = t[:1]
        return state["buf"] * 2

    x = mx.array([1.0, 2.0, 3.0])
    y = mx.compile(step, inputs=state, outputs=state)(x)
    mx.eval(y)
    assert y.tolist() == [0.0, 2.0, 0.0, 0.0]
    assert state["buf"].tolist() == [0.0, 1.0, 0.0, 0.0]
    z = step(x)
    mx.eval(z)
    assert z.tolist() == [0.0, 2.0, 0.0, 0.0]


def test_compile_accepts_a_state_free_submodule_of_a_stateful_step():
    """Protects: fusing inside a step whose cache blocks compiling the whole
    thing. The write poisons the step (see the attribute test above), but a
    submodule sitting between two writes compiles, returns the same bits, and
    keeps working across repeated calls rather than breaking on the second
    like the whole-step compile does (spike_15)."""
    import types

    class Mlp(nn.Module):
        """The real shape of the run being compiled: quantized matmuls behind
        nested modules, not a bare elementwise chain."""

        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(64, 128, bias=False)
            self.up = nn.Linear(64, 128, bias=False)
            self.down = nn.Linear(128, 64, bias=False)

        def __call__(self, x):
            return self.down(nn.silu(self.gate(x)) * self.up(x))

    class Compiled(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            object.__setattr__(self, "_fn", mx.compile(lambda a: inner(a)))

        def __call__(self, a):
            return self._fn(a)

    class Step(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Mlp()
            self.cache = types.SimpleNamespace(buf=mx.zeros((4, 64)))

        def __call__(self, x):
            h = self.mlp(x)
            self.cache.buf[0:1] = h[0:1]  # the write that makes the step uncompilable
            return h + self.cache.buf.sum(axis=0)

    mx.random.seed(23)
    m = Step()
    nn.quantize(m.mlp, group_size=64, bits=4)
    x = mx.random.normal((4, 64))
    mx.eval(m(x))  # settle the buffer: every call writes the same slot the same way
    want = m(x)
    mx.eval(want)

    plain_mlp = m.mlp
    m.mlp = Compiled(plain_mlp)
    got_first, got_second = m(x), m(x)
    mx.eval(got_first, got_second)
    assert mx.array_equal(got_first, want).item()
    assert mx.array_equal(got_second, want).item()

    # The write stayed outside the compiled region, so the cache never held a
    # tracer: swapping the plain module back still works, which is what fails
    # after a whole-step compile.
    m.mlp = plain_mlp
    after = m(x)
    mx.eval(after)
    assert mx.array_equal(after, want).item()


# -- graph insertion beside compiled helpers and in-place writes (2026-09-15) --


def test_compiled_in_place_write_to_an_input_stays_inside():
    """Protects bind.graph.graph_scope_reason's mutation rule: under mx.compile
    an in-place write to an array the caller still holds never reaches the
    caller, while the eager module changes it, so such a scope cannot be
    compiled without changing behavior. A write to the scope's own
    intermediate is ordinary dataflow either way."""
    def writes_input(x):
        x[0] = 9
        return x * 2

    x = mx.ones(3)
    eager = writes_input(x)
    mx.eval(x, eager)
    assert x.tolist() == [9.0, 1.0, 1.0]
    x = mx.ones(3)
    compiled = mx.compile(writes_input)(x)
    mx.eval(x, compiled)
    assert x.tolist() == [1.0, 1.0, 1.0]  # the caller never sees the write
    assert compiled.tolist() == eager.tolist()

    def writes_intermediate(x):
        t = x * 3
        t += 1
        t[1] = 0
        return t

    x = mx.ones(3)
    assert mx.compile(writes_intermediate)(x).tolist() == writes_intermediate(x).tolist() == [4.0, 0.0, 4.0]


def test_compiled_scope_keeps_ops_beside_a_compiled_helper_visible():
    """Protects graph delivery on scopes that call a compiled helper (nn.silu,
    mlx_lm's swiglu): compiling the scope again still exposes the ops around
    the helper to the graph rewriter, the rewrite runs once per shape, and
    the result is bitwise the original's."""
    from autotuner_runtime import graph_native
    from autotuner_runtime.exact import bitwise_equal

    w = mx.random.normal((64, 64)).astype(mx.float16)
    mx.eval(w)

    def block(x):
        return nn.silu(x @ w) * 2  # nn.silu is compiled shapeless by mlx

    px, pw = mx.zeros((4, 64), mx.float16), mx.zeros((64, 64), mx.float16)
    pattern = [px @ pw]
    rewrites = []

    def transformed(x):
        roots, hits = graph_native.rewrite([block(x)], pattern, [px, pw],
                                           lambda matched: [mx.matmul(*matched)])
        rewrites.append(hits)
        return roots[0]

    compiled = mx.compile(transformed)
    x = mx.random.normal((4, 64)).astype(mx.float16)
    for i in range(3):
        assert bitwise_equal(compiled(x + i), block(x + i))
    assert rewrites == [1]


def test_raise_inside_a_trace_strands_tracers_in_captured_state():
    """Protects autotuner_runtime.graph's rule that the traced function never
    raises: mx.compile puts its tracers into the captured `inputs=` state
    before tracing and swaps the real arrays back only when the trace
    returns, so an exception leaves the module holding tracers that cannot be
    evaluated."""
    state = {"w": mx.ones(3)}

    def bad(x):
        raise ValueError("inside the trace")

    with pytest.raises(ValueError):
        mx.compile(bad, inputs=state)(mx.ones(3))
    with pytest.raises(RuntimeError, match="without a primitive"):
        mx.eval(state["w"])

    state = {"w": mx.ones(3)}
    good = mx.compile(lambda x: x + state["w"], inputs=state)
    mx.eval(good(mx.ones(3)))
    assert state["w"].tolist() == [1.0, 1.0, 1.0]  # a returning trace restores it


def test_a_compiled_step_reuses_the_python_ints_it_read():
    """Protects the graph delivery rule that a cache position belongs in the
    call signature: a compiled trace bakes in every Python int it read (a rope
    offset, a slice bound) and never runs the Python again, so reusing it at a
    new position returns the first position's answer and the position itself
    stops advancing."""
    class Cache:
        def __init__(self):
            self.keys, self.offset = None, 0

        def update(self, k):
            prev = self.offset
            if self.keys is None:
                self.keys = mx.zeros((1, 1, 8, 4), k.dtype)
            self.offset += k.shape[2]
            self.keys[..., prev:self.offset, :] = k
            return self.keys[..., :self.offset, :]

    rope = nn.RoPE(4)

    def step(x, cache):
        return cache.update(rope(x, offset=cache.offset)).sum()

    x = mx.random.normal((1, 1, 1, 4))
    plain, compiled = Cache(), Cache()
    step_compiled = mx.compile(lambda x: step(x, compiled))
    want = [step(x, plain).item() for _ in range(3)]
    got = [step_compiled(x).item() for _ in range(3)]
    assert want[0] == got[0] and len(set(want)) == 3
    assert got == [want[0]] * 3
    assert plain.offset == 3 and compiled.offset == 1
