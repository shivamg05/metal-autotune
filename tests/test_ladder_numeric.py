"""M6 numeric core: gate 1 static checks, the gate 5/8 comparison rules and
value regimes, and the fp32 golden evaluator for assoc-changing compares.

The golden tests trace tiny inline models with the repo Tracer; the patch
surface is process-global, so one module-scoped Tracer serves them and the
final test uninstalls and checks exact restoration.
"""

import math

import mlx.core as mx
import pytest

from autotuner.ladder.golden import (
    err,
    golden_outputs,
    passes,
    promote_bindings,
    substitution_table,
)
from autotuner.ladder.numeric import (
    OUTLIER_COUNT,
    OUTLIER_VALUE,
    REGIMES,
    compare,
    max_abs_diff,
    value_regimes,
)
from autotuner.ladder.static_checks import RegionContract, check
from autotuner.trace import Tracer
from tests.conftest import current_tracer, tracer_for_module
from autotuner.trace.walk import arrays_by_path
from autotuner_runtime.kernels import KernelSpec

_module_tracer = tracer_for_module()


def tracer() -> Tracer:
    return current_tracer()


# -- gate 1: static checks ---------------------------------------------------


def make_spec(**over) -> KernelSpec:
    base = dict(
        kernel_id="k0",
        name="fused_k",
        input_names=("x", "w"),
        output_names=("y",),
        source="// body",
        grid=("ceil_div(in0.shape[0], 32) * 32", "1", "1"),
        threadgroup=("32", "1", "1"),
        output_shapes=(("in0.shape[0]", "in1.shape[1]"),),
        output_dtypes=("float16",),
        template=(("T", "in0"),),
    )
    base.update(over)
    return KernelSpec(**base)


def make_contract(**over) -> RegionContract:
    base = dict(
        input_names=("x", "w"),
        input_ranks=(2, 2),
        input_dtypes=("float16", "float16"),
        output_names=("y",),
        output_ranks=(2,),
        output_dtypes=("float16",),
        live_outputs=("y",),
    )
    base.update(over)
    return RegionContract(**base)


def failed_checks(spec, contract) -> list[str]:
    return [f.check for f in check(spec, contract)]


def test_static_valid_spec_passes():
    assert check(make_spec(), make_contract()) == []


def test_static_valid_spec_with_fallback_passes():
    spec = make_spec(fallback_predicate="in0.shape[0] > 64")
    assert check(spec, make_contract(requires_fallback=True)) == []


def test_static_bad_kernel_name():
    # spike_04: a bad name breaks the generated signature only at probe eval
    for name in ("fused-k", "1fused", "fused k", ""):
        assert "kernel_name" in failed_checks(make_spec(name=name), make_contract())


def test_static_input_names_mismatch():
    assert "input_names" in failed_checks(
        make_spec(input_names=("x", "v")), make_contract())
    assert "input_names" in failed_checks(
        make_spec(input_names=("x",)), make_contract())
    # order is the call-site binding order, so it is part of the contract
    assert "input_names" in failed_checks(
        make_spec(input_names=("w", "x")), make_contract())


def test_static_live_output_dropped():
    contract = make_contract(
        output_names=("y", "z"), output_ranks=(2, 2),
        output_dtypes=("float16", "float16"), live_outputs=("y", "z"))
    fails = failed_checks(make_spec(), contract)
    assert "output_names" in fails
    assert "live_output_dropped" in fails
    detail = next(f.detail for f in check(make_spec(), contract)
                  if f.check == "live_output_dropped")
    assert "z" in detail


def test_static_output_dtype_change():
    fails = failed_checks(make_spec(output_dtypes=("float32",)), make_contract())
    assert "output_dtype" in fails


def test_static_output_rank_mismatch():
    spec = make_spec(output_shapes=(("in0.shape[0]",),))
    assert "output_rank" in failed_checks(spec, make_contract())


def test_static_output_arity():
    spec = make_spec(output_shapes=())
    assert "output_arity" in failed_checks(spec, make_contract())


def test_static_launch_arity():
    spec = make_spec(grid=("1", "1"))
    assert "launch_arity" in failed_checks(spec, make_contract())


def test_static_grammar_parse_failures():
    assert "launch_grammar" in failed_checks(
        make_spec(grid=("in0.shape[0] ** 2", "1", "1")), make_contract())
    assert "launch_grammar" in failed_checks(
        make_spec(fallback_predicate="__import__('os')"), make_contract())
    assert "launch_grammar" in failed_checks(
        make_spec(output_shapes=(("in0.shape[0]", "len(w)"),)), make_contract())


def test_static_grammar_rank_probe():
    # no input rank declaration exists on the kernel side; the probe evaluation
    # against the contract's ranks is what catches a bad axis or input index
    assert "launch_grammar" in failed_checks(
        make_spec(grid=("in0.shape[5]", "1", "1")), make_contract())
    assert "launch_grammar" in failed_checks(
        make_spec(threadgroup=("in7.ndim", "1", "1")), make_contract())


def test_static_value_dependent_division_is_not_static():
    # a zero divisor at the dummy size is value-dependent, not a static failure
    spec = make_spec(grid=("64 // (in0.shape[0] - 4)", "1", "1"))
    assert check(spec, make_contract()) == []


def test_static_fallback_missing():
    fails = failed_checks(make_spec(), make_contract(requires_fallback=True))
    assert "fallback_missing" in fails


def test_static_template_checks():
    assert "template" in failed_checks(
        make_spec(template=(("T", "float64"),)), make_contract())
    assert "template" in failed_checks(
        make_spec(template=(("T", "in9"),)), make_contract())
    spec = make_spec(template=(("T", "in0"), ("U", "uint32"), ("V", "int32")))
    assert check(spec, make_contract()) == []


# -- gates 5/8: the comparison rules -----------------------------------------


def test_compare_exact_passes():
    a = mx.random.normal((3, 5), key=mx.random.key(0))
    res = compare(a, a, rtol=0.0, atol=0.0)
    assert res.passed and res.reason == ""
    assert res.max_excess <= 0.0


def test_compare_within_tolerance():
    ref = mx.ones((4,))
    cand = ref + 0.05
    assert compare(cand, ref, rtol=0.0, atol=0.1).passed


def test_compare_over_tolerance_reports_worst_offender():
    ref = mx.ones((2, 3))
    bump = mx.zeros((2, 3))
    bump[1, 2] = 0.5
    cand = ref + bump
    res = compare(cand, ref, rtol=0.0, atol=0.1)
    assert not res.passed and res.reason == "tolerance"
    assert res.index == (1, 2)
    assert res.max_excess == pytest.approx(0.4, rel=1e-5)


def test_compare_wobble_floor_rescues():
    ref = mx.ones((2, 3))
    bump = mx.zeros((2, 3))
    bump[1, 2] = 0.5
    cand = ref + bump
    res = compare(cand, ref, rtol=0.0, atol=0.1, wobble_floor=0.6)
    assert res.passed
    assert res.max_excess <= 0.0


def test_compare_candidate_nonfinite_where_reference_finite():
    ref = mx.ones((2, 3))
    cand = mx.ones((2, 3))
    cand[0, 1] = float("nan")
    cand[1, 2] = float("inf")
    res = compare(cand, ref, rtol=1.0, atol=1e6)  # huge tolerance cannot save it
    assert not res.passed and res.reason == "nonfinite_pattern"
    assert res.index == (0, 1)  # first violation
    assert math.isinf(res.max_excess)


def test_compare_reference_nonfinite_pattern():
    ref = mx.array([1.0, float("nan"), float("inf"), float("-inf")])
    same = mx.array([1.0, float("nan"), float("inf"), float("-inf")])
    assert compare(same, ref, rtol=0.0, atol=0.0).passed
    # NaN where the reference is NaN, inf sign reproduced exactly
    wrong_sign = mx.array([1.0, float("nan"), float("-inf"), float("-inf")])
    res = compare(wrong_sign, ref, rtol=0.0, atol=0.0)
    assert not res.passed and res.reason == "nonfinite_pattern"
    assert res.index == (2,)
    finite_for_nan = mx.array([1.0, 2.0, float("inf"), float("-inf")])
    res = compare(finite_for_nan, ref, rtol=0.0, atol=0.0)
    assert not res.passed and res.reason == "nonfinite_pattern"
    assert res.index == (1,)


def test_compare_shape_and_dtype_mismatch():
    a = mx.ones((2, 3))
    assert compare(mx.ones((3, 2)), a, rtol=0.0, atol=0.0).reason == "shape"
    assert compare(a.astype(mx.float16), a, rtol=0.0, atol=0.0).reason == "dtype"


def test_compare_fp16():
    ref = mx.random.normal((8,), key=mx.random.key(1)).astype(mx.float16)
    assert compare(ref, ref, rtol=1e-2, atol=2e-2).passed
    cand = (ref.astype(mx.float32) + 1.0).astype(mx.float16)
    assert not compare(cand, ref, rtol=1e-2, atol=2e-2).passed


def test_max_abs_diff():
    a = mx.array([1.0, 2.0])
    assert max_abs_diff(a, a) == 0.0
    assert max_abs_diff(mx.array([1.0, 2.5]), a) == pytest.approx(0.5)
    both_nan = mx.array([float("nan"), 2.0])
    assert max_abs_diff(both_nan, both_nan) == 0.0
    assert math.isinf(max_abs_diff(mx.array([float("nan"), 2.0]), a))
    assert math.isinf(max_abs_diff(mx.ones((3,)), mx.ones((2,))))


# -- gate 5: value regimes ---------------------------------------------------


def regime_inputs():
    a = mx.random.normal((4, 8), key=mx.random.key(2)).astype(mx.float16)
    b = mx.random.normal((6,), key=mx.random.key(3))
    idx = mx.arange(5, dtype=mx.int32)
    return [a, b, idx]


def test_regimes_scaled():
    a, b, _ = regime_inputs()
    r = value_regimes(regime_inputs(), seed=7)
    for i, orig in enumerate((a, b)):
        up, down = r["scaled_up"][i], r["scaled_down"][i]
        assert up.dtype == orig.dtype and down.dtype == orig.dtype
        assert mx.allclose(up.astype(mx.float32), orig.astype(mx.float32) * 1e3,
                           rtol=1e-2, atol=1e-2).item()
        assert mx.allclose(down.astype(mx.float32), orig.astype(mx.float32) * 1e-4,
                           rtol=1e-2, atol=1e-6).item()


def test_regimes_outliers():
    a = regime_inputs()[0]
    r = value_regimes(regime_inputs(), seed=7)["outliers"][0]
    assert r.dtype == a.dtype and r.shape == a.shape
    planted = r == OUTLIER_VALUE
    assert int(mx.sum(planted).item()) == OUTLIER_COUNT
    # everything off the planted positions is untouched
    assert mx.array_equal(mx.where(planted, a, r), a).item()


def test_regimes_zeros():
    inputs = regime_inputs()
    r = value_regimes(inputs, seed=7)["zeros"]
    for orig, z in zip(inputs[:2], r[:2]):
        assert z.dtype == orig.dtype and z.shape == orig.shape
        assert int(mx.sum(mx.abs(z)).item()) == 0


def test_regimes_nonfinite_lanes():
    a = regime_inputs()[0]  # (4, 8)
    r = value_regimes(regime_inputs(), seed=7)["nonfinite"][0]
    assert r.dtype == a.dtype
    assert int(mx.sum(mx.isinf(r)).item()) == a.shape[1]  # one full inf lane
    assert int(mx.sum(mx.isnan(r)).item()) == a.shape[1]  # one full nan lane
    for i in range(a.shape[0]):
        row = r[i]
        if mx.all(mx.isinf(row)).item() or mx.all(mx.isnan(row)).item():
            continue
        assert mx.array_equal(row, a[i]).item()


def test_regimes_nonfloat_passthrough():
    idx = regime_inputs()[2]
    r = value_regimes(regime_inputs(), seed=7)
    for name in REGIMES:
        got = r[name][2]
        assert got.dtype == idx.dtype
        assert mx.array_equal(got, idx).item()


def test_regimes_scalar_input():
    r = value_regimes([mx.array(1.5, dtype=mx.float16)], seed=0)
    assert mx.isnan(r["nonfinite"][0]).item()
    assert r["outliers"][0].item() == OUTLIER_VALUE
    assert r["zeros"][0].item() == 0.0


def test_regimes_deterministic():
    r1 = value_regimes(regime_inputs(), seed=11)
    r2 = value_regimes(regime_inputs(), seed=11)
    assert list(r1) == list(REGIMES)
    for name in REGIMES:
        for u, v in zip(r1[name], r2[name], strict=True):
            assert mx.array_equal(u, v, equal_nan=True).item()


# -- gate 8: the fp32 golden -------------------------------------------------


def build_fp16_chain():
    w = (0.25 * mx.random.normal((16, 8), key=mx.random.key(4))).astype(mx.float16)
    b = mx.random.normal((8,), key=mx.random.key(5)).astype(mx.float16)

    def model(x):
        return mx.matmul(x, w) + b

    return model


def build_quantized(explicit: bool):
    wf = mx.random.normal((32, 64), key=mx.random.key(6)).astype(mx.float16)
    if explicit:
        wq, sc, bi = mx.quantize(wf, group_size=32, bits=4)

        def model(x):
            return mx.quantized_matmul(x, wq, sc, bi, transpose=True,
                                       group_size=32, bits=4)
    else:
        wq, sc, bi = mx.quantize(wf)  # library defaults

        def model(x):
            return mx.quantized_matmul(x, wq, sc, bi)

    return model, wf, (wq, sc, bi)


def trace_and_bind(model, x):
    trace, _ = tracer().trace(model, [x])
    by_path = arrays_by_path(model)
    bindings = {aid: x for aid in trace.inputs}
    for aid in trace.weights:
        path = trace.weight_paths.get(aid)
        if path in by_path:
            bindings[aid] = by_path[path]
    return trace, bindings


def test_golden_fp16_chain_gate():
    model = build_fp16_chain()
    x = mx.random.normal((4, 16), key=mx.random.key(7)).astype(mx.float16)
    trace, bindings = trace_and_bind(model, x)
    assert [n.op for n in trace.nodes] == ["mx.matmul", "array.__add__"]

    g = golden_outputs(trace.nodes, bindings, trace.step_outputs)[trace.step_outputs[0]]
    assert g.dtype == mx.float32
    # promotion is exact for fp16, so the golden is the hand fp32 chain, bitwise
    want = mx.matmul(x.astype(mx.float32),
                     bindings_weight(model, "w")) + bindings_weight(model, "b")
    assert mx.array_equal(g, want).item()

    lib = model(x)
    err_lib = err([lib], [g], denom_clamp=1.0)
    assert 0.0 < err_lib < 0.05  # fp16 rounding, nothing more
    # the library's own result passes the gate against itself for any kappa >= 1
    assert passes(err_lib, err_lib)
    # a deliberately sloppier fp16 result fails
    sloppy = (lib.astype(mx.float32) * 1.5).astype(mx.float16)
    err_sloppy = err([sloppy], [g], denom_clamp=1.0)
    assert err_sloppy > err_lib
    assert not passes(err_sloppy, err_lib)


def bindings_weight(model, name):
    cells = {v: c.cell_contents for v, c in
             zip(model.__code__.co_freevars, model.__closure__)}
    return cells[name].astype(mx.float32)


def test_golden_quantized_substitution():
    model, wf, (wq, sc, bi) = build_quantized(explicit=True)
    x = mx.random.normal((4, 64), key=mx.random.key(8)).astype(mx.float16)
    trace, bindings = trace_and_bind(model, x)
    assert [n.op for n in trace.nodes] == ["mx.quantized_matmul"]
    assert trace.nodes[0].scalar_args["kwargs"]["group_size"] == 32

    g = golden_outputs(trace.nodes, bindings, trace.step_outputs)[trace.step_outputs[0]]
    assert g.dtype == mx.float32
    # the substitution pins group_size/bits from the recorded args: the golden
    # is exactly dequantize(32, 4) + fp32 matmul
    dq = mx.dequantize(wq, sc.astype(mx.float32), bi.astype(mx.float32),
                       group_size=32, bits=4)
    want = mx.matmul(x.astype(mx.float32), mx.swapaxes(dq, -1, -2))
    assert mx.array_equal(g, want).item()

    # the golden sits nearer the unquantized ideal than a perturbed candidate
    ideal = mx.matmul(x.astype(mx.float32),
                      mx.swapaxes(wf.astype(mx.float32), -1, -2))
    err_golden = err([g], [ideal], denom_clamp=1.0)
    err_perturbed = err([g + 5.0], [ideal], denom_clamp=1.0)
    assert err_golden < err_perturbed


def test_golden_quantized_default_args_pin():
    # recorded defaults (no group_size/bits in the call) resolve to the same
    # library defaults inside the substitution
    model, _, (wq, sc, bi) = build_quantized(explicit=False)
    x = mx.random.normal((4, 64), key=mx.random.key(9)).astype(mx.float16)
    trace, bindings = trace_and_bind(model, x)
    g = golden_outputs(trace.nodes, bindings, trace.step_outputs)[trace.step_outputs[0]]
    dq = mx.dequantize(wq, sc.astype(mx.float32), bi.astype(mx.float32),
                       group_size=64, bits=4)
    want = mx.matmul(x.astype(mx.float32), mx.swapaxes(dq, -1, -2))
    assert mx.array_equal(g, want).item()


def test_substitution_table_extensible():
    marker = lambda *a, **k: None
    table = substitution_table({"mx.exp": marker})
    assert table["mx.exp"] is marker
    assert "mx.quantized_matmul" in table
    assert "mx.exp" not in substitution_table()  # the default table is unmodified


def test_promote_bindings():
    f16 = mx.ones((2,), dtype=mx.float16)
    f32 = mx.ones((2,), dtype=mx.float32)
    u32 = mx.ones((2,), dtype=mx.uint32)
    out = promote_bindings({0: f16, 1: f32, 2: u32})
    assert out[0].dtype == mx.float32
    assert out[1] is f32   # already fp32: untouched
    assert out[2] is u32   # packed quantized weights: untouched


def test_err_nonfinite_golden_positions():
    g = mx.array([1.0, float("inf"), float("nan")])
    matching = mx.array([1.0, float("inf"), float("nan")])
    assert err([matching], [g], denom_clamp=1.0) == 0.0
    broken = mx.array([1.0, 2.0, float("nan")])  # finite where the golden is inf
    assert math.isinf(err([broken], [g], denom_clamp=1.0))


def test_err_denominator_clamp():
    g = mx.array([0.0, 2.0])
    c = mx.array([1e-3, 2.0])
    # |c - g| / max(|g|, clamp): the zero-golden element is scored on the clamp
    assert err([c], [g], denom_clamp=1.0) == pytest.approx(1e-3, rel=1e-5)
    assert err([c], [g], denom_clamp=1e-4) == pytest.approx(10.0, rel=1e-4)
