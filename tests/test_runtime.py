"""The runtime package: launch grammar and the one kernel call site."""

import mlx.core as mx
import pytest

from autotuner_runtime.grammar import Expr, GrammarError
from autotuner_runtime.kernels import KernelSpec, LoadedKernel, call


def test_grammar_evaluates():
    assert Expr("in0.shape[0] * in0.shape[1]").evaluate([(4, 8)]) == 32
    assert Expr("ceil_div(in0.shape[0], 32) * 32").evaluate([(50,)]) == 64
    assert Expr("in0.ndim == 2 and in0.shape[1] % 4 == 0").evaluate([(3, 8)]) is True
    assert Expr("in1.shape[0] - in0.shape[0]").evaluate([(3,), (10,)]) == 7
    assert Expr("-in0.shape[0] + 5").evaluate([(2,)]) == 3


@pytest.mark.parametrize("bad", [
    "__import__('os')", "in0.shape[0] ** 2", "lambda: 1", "x", "min(1,2,3)",
    "1.5", "in0.dtype", "f(1,2)", "in0.shape", "[1,2]", "'s'",
])
def test_grammar_rejects_everything_outside(bad):
    with pytest.raises(GrammarError):
        Expr(bad)


def test_grammar_runtime_bounds():
    with pytest.raises(GrammarError, match="out of range"):
        Expr("in2.shape[0]").evaluate([(4,)])
    with pytest.raises(GrammarError, match="axis"):
        Expr("in0.shape[5]").evaluate([(4,)])


def _double_spec():
    return KernelSpec(
        kernel_id="t_double", name="rt_test_double",
        input_names=("inp",), output_names=("out",),
        source="uint i = thread_position_in_grid.x;\nout[i] = inp[i] * T(2);",
        grid=("in0.shape[0] * in0.shape[1]", "1", "1"),
        threadgroup=("min(in0.shape[0] * in0.shape[1], 256)", "1", "1"),
        output_shapes=(("in0.shape[0]", "in0.shape[1]"),),
        output_dtypes=("float32",),
        template=(("T", "in0"),),
    )


def test_kernel_call_site_and_shape_generality():
    """One spec launches correctly at every size: the launch grammar is what
    makes a kernel valid across the sweep."""
    spec = _double_spec()
    for shape in [(4, 8), (7, 13), (64, 32)]:
        x = mx.random.normal(shape, key=mx.random.key(0))
        out = call(spec, [x])[0]
        mx.eval(out)
        assert mx.array_equal(out, x * 2).item()


def test_kernel_spec_round_trips():
    spec = _double_spec()
    again = KernelSpec.from_json(spec.to_json())
    assert again == spec


def test_fallback_predicate_and_poison():
    spec = KernelSpec(
        kernel_id="t_half", name="rt_test_half_write",
        input_names=("inp",), output_names=("out",),
        source="uint i = thread_position_in_grid.x;\nif (i < 16) out[i] = inp[i];",
        grid=("32", "1", "1"), threadgroup=("32", "1", "1"),
        output_shapes=(("32",),), output_dtypes=("float32",),
        fallback_predicate="in0.shape[0] != 32",
    )
    lk = LoadedKernel(spec)
    assert lk.fallback_fires([mx.zeros((7,))])
    assert not lk.fallback_fires([mx.zeros((32,))])
    y = call(spec, [mx.ones((32,))], init_value=float("nan"))[0]
    mx.eval(y)
    assert mx.isnan(y[16:]).all().item()
    assert mx.array_equal(y[:16], mx.ones((16,))).item()


def test_the_fallback_decides_before_the_launch_is_evaluated():
    """A shape-specialized kernel's launch expressions need only hold on the
    shapes it covers: at a shape its fallback predicate excludes, the grid
    arithmetic may not even be defined, and the wrapper must take the
    original ops instead of raising."""
    from autotuner_runtime.kernels import LAUNCH_CACHE_MAX, try_call

    spec = KernelSpec(kernel_id="guarded", name="rt_guarded_copy", input_names=("in0",),
                      output_names=("out0",),
                      source="uint i = thread_position_in_grid.x; out0[i] = in0[i];",
                      grid=("in0.shape[0]", "1", "1"),
                      threadgroup=("in0.shape[0] // (in0.shape[0] // 8)", "1", "1"),  # divides by zero under 8
                      output_shapes=(("in0.shape[0]",),), output_dtypes=("float32",),
                      fallback_predicate="in0.shape[0] < 8")
    lk = LoadedKernel(spec)
    assert lk.fallback_fires([mx.zeros((3,))])          # no ZeroDivisionError
    assert try_call(spec, [mx.zeros((3,))]) is None
    x = mx.arange(16, dtype=mx.float32)
    assert try_call(spec, [x])[0].tolist() == x.tolist()
    with pytest.raises(ValueError, match="fallback"):
        lk([mx.zeros((3,))])
    for n in range(LAUNCH_CACHE_MAX + 5):
        lk.fallback_fires([mx.zeros((8 + n,))])
    assert len(lk._launches) <= LAUNCH_CACHE_MAX      # a long shape sweep does not grow it forever


def test_launch_is_evaluated_once_per_call_signature(monkeypatch):
    """Evaluating the launch grammar on every call cost a small kernel more
    than its GPU time. The launch is a function of the inputs' shapes and
    dtypes, so a repeat call must not touch the evaluator."""
    from autotuner_runtime import grammar

    calls = []
    real = grammar.Expr.evaluate

    def counting(self, shapes):
        calls.append(self.text)
        return real(self, shapes)

    monkeypatch.setattr(grammar.Expr, "evaluate", counting)
    spec = KernelSpec(kernel_id="once", name="once_copy", input_names=("in0",), output_names=("out0",),
                      source="uint i = thread_position_in_grid.x; out0[i] = in0[i];",
                      grid=("in0.shape[0]", "1", "1"), threadgroup=("min(in0.shape[0], 256)", "1", "1"),
                      output_shapes=(("in0.shape[0]",),), output_dtypes=("float32",),
                      fallback_predicate="in0.shape[0] > 64")
    lk = LoadedKernel(spec)
    x = mx.arange(8, dtype=mx.float32)
    assert lk([x])[0].tolist() == x.tolist()
    first = len(calls)
    assert first > 0
    lk([x + 1])
    assert not lk.fallback_fires([x])
    assert len(calls) == first                       # same signature: nothing re-evaluated
    assert lk.fallback_fires([mx.zeros((128,))])     # a new shape is evaluated afresh
    assert len(calls) > first
