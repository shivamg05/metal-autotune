"""Naive lowering. Fixture regions are lowered to one Metal kernel each,
run through autotuner_runtime.kernels.call, and compared against the library
replay at several traced sizes, including one the lowering never saw.

Comparison policy: regions whose lowering preserves the library's operation
order bit for bit (elementwise chains via metal::precise::, comparisons, max
reductions) must match bitwise. Regions containing matmul, rms_norm, or
sum/mean reductions reassociate the accumulation (serial k loop, threadgroup
tree), so those compare at fp32 rtol 1e-5 atol 1e-6, the legal internal
precision bound."""

import importlib.util
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from autotuner.regions.build import _boundary
from autotuner.regions.price import capture_boundaries
from autotuner.regions.sweep import locate_span
from autotuner.regions.types import Stretch
from autotuner.scaffold import NoScaffold, lower_naive, stretch_input_shapes
from autotuner.trace import Trace, Tracer, TraceNode
from autotuner.trace.recorder import ArrayRef
from autotuner.trace.replay import replay
from autotuner_runtime.kernels import KernelSpec, call
from tests.conftest import current_tracer, tracer_for_module

FIXTURES = Path(__file__).parent / "fixtures"

_module_tracer = tracer_for_module()


def tracer() -> Tracer:
    return current_tracer()


def load_fixture(name: str):
    tracer()
    spec = importlib.util.spec_from_file_location(f"fixture_s_{name}", FIXTURES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def traced(model, shape, key, dtype=None):
    x = mx.random.normal(shape, key=mx.random.key(key))
    if dtype is not None:
        x = x.astype(dtype)
        mx.eval(x)
    trace, _ = tracer().trace(model, [x])
    return x, trace


def cut(trace, start, end) -> Stretch:
    return _boundary(trace, "w", start, end)


def check(spec: KernelSpec, model, xs, trace, stretch, bitwise: bool,
          rtol=1e-5, atol=1e-6) -> None:
    """Run the kernel on the region's captured boundary inputs and compare
    every region output against the library replay."""
    ids = set(stretch.input_ids) | set(stretch.output_ids)
    arrays = capture_boundaries(tracer(), model, list(xs), trace, ids)
    nodes = trace.nodes[stretch.start_seq:stretch.end_seq + 1]
    want = replay(nodes, {a: arrays[a] for a in stretch.input_ids}, stretch.output_ids)
    outs = call(spec, [arrays[a] for a in stretch.input_ids])
    mx.eval(outs)
    mx.eval(list(want.values()))
    for i, aid in enumerate(stretch.output_ids):
        got, ref = outs[i], want[aid]
        assert tuple(got.shape) == tuple(ref.shape)
        assert got.dtype == ref.dtype
        if bitwise:
            assert mx.array_equal(got, ref).item(), f"output {i} is not bitwise equal"
        else:
            assert mx.allclose(got, ref, rtol=rtol, atol=atol).item(), f"output {i} out of tolerance"


def lower_swept(model, shape_of, span, keys=(0, 1, 2), dtype=None, batches=(64, 7)):
    """Trace at batch 4 and batches[0] (the two lowering instances) plus the
    remaining batches the lowering never saw, lower once, and return the spec
    with every size's (x, trace, stretch)."""
    sizes = []
    x4, t4 = traced(model, shape_of(4), keys[0], dtype)
    s4 = cut(t4, *span)
    sizes.append((x4, t4, s4))
    for i, b in enumerate(batches):
        x, t = traced(model, shape_of(b), keys[1] + i, dtype)
        sizes.append((x, t, locate_span(t4, s4, t, "w")))
    _, t64, s64 = sizes[1]
    spec = lower_naive(t4, s4, [stretch_input_shapes(t64, s64)])
    return spec, sizes


# -- (a) rms_norm + three matmuls, three outputs ------------------------------


def test_norm_three_proj_chain():
    """The full norm_three_proj chain: rms_norm feeding three projections.
    Tolerance comparison: the serial matmul k loop and the rms threadgroup
    tree reassociate the library's accumulation order."""
    model = load_fixture("norm_three_proj")
    spec, sizes = lower_swept(model, lambda b: (b, 32), (0, 3))
    # batch is swept, so launch and output shapes are grammar, never baked
    assert "in0.shape[0]" in spec.grid[1]
    assert spec.output_shapes[0] == ("in0.shape[0]", "32")
    # region outputs come first, scratch after
    assert spec.output_names[:3] == ("out0", "out1", "out2")
    assert all(n.startswith("tmp") for n in spec.output_names[3:])
    for x, t, s in sizes:
        assert len(s.output_ids) == 3
        check(spec, model, [x], t, s, bitwise=False)


# -- (b) pure elementwise chain, two outputs, one a step output ---------------


def test_step_output_elementwise_bitwise():
    """relu + scale from step_output_region, cut below the matmul: pure
    elementwise in fp32 preserves the library's per-op rounding, so the
    comparison is bitwise."""
    model = load_fixture("step_output_region")
    spec, sizes = lower_swept(model, lambda b: (b, 16), (1, 2), keys=(3, 4, 5))
    assert spec.grid[1] == "in0.shape[0]"
    for x, t, s in sizes:
        assert len(s.output_ids) == 2  # h is a step output and consumed inside
        check(spec, model, [x], t, s, bitwise=True)


# -- (c) matmul + view run + add ----------------------------------------------


def test_views_only_chain():
    """The views_only chain: matmul, reshape/transpose/reshape/squeeze, add.
    The model's reshape targets bake batch 4, so batch is a model constant,
    there is only one traceable size, and literal dims are correct. Tolerance
    comparison because of the matmul; the view indexing itself is exact and
    any folding mistake would blow far past the tolerance."""
    model = load_fixture("views_only")
    x, t = traced(model, (4, 8), 6)
    s = cut(t, 0, len(t.nodes) - 1)
    spec = lower_naive(t, s)
    assert spec.output_shapes[0] == ("4", "8")
    check(spec, model, [x], t, s, bitwise=False)


# -- (d) row reductions after an elementwise op -------------------------------


class _AbsScaleSum:
    def __call__(self, x):
        h = mx.abs(x) * 2.0
        return mx.sum(h, axis=-1), mx.mean(h, axis=-1)


def test_inline_reduction_sum_and_mean():
    """sum and mean over the last axis after an elementwise op, one shared
    row. Tolerance comparison: the threadgroup tree reassociates the
    library's row order. Row length 33 exercises the tree with idle lanes."""
    model = _AbsScaleSum()
    spec, sizes = lower_swept(model, lambda b: (b, 33), (0, 3), keys=(7, 8, 9))
    assert spec.grid[1] == "in0.shape[0]"  # one threadgroup per row
    for x, t, s in sizes:
        assert len(s.output_ids) == 2
        check(spec, model, [x], t, s, bitwise=False)


class _ScaleMax:
    def __call__(self, x):
        return mx.max(x * 0.5, axis=-1, keepdims=True)


def test_inline_reduction_max_bitwise():
    """max over the last axis, keepdims: order-insensitive and roundoff-free,
    so the tree reduction must reproduce the library bits exactly."""
    model = _ScaleMax()
    spec, sizes = lower_swept(model, lambda b: (b, 33), (0, 1), keys=(10, 11, 12))
    assert spec.output_shapes[0] == ("in0.shape[0]", "1")
    for x, t, s in sizes:
        check(spec, model, [x], t, s, bitwise=True)


class _NormRow:
    def __call__(self, x):
        h = mx.abs(x) + 1.0
        return h / mx.sum(h, axis=-1, keepdims=True)


def test_broadcast_of_intermediate_falls_back_single():
    """h / rowsum(h): the divide broadcasts a stage-written (B, 1) buffer
    across the row, which is not a row-local read, so the lowering must fall
    back to the single-threadgroup launch and still be correct. Tolerance
    because of the sum tree."""
    model = _NormRow()
    spec, sizes = lower_swept(model, lambda b: (b, 33), (0, 3), keys=(29, 30, 31))
    assert spec.grid[1] == "1"  # single threadgroup: correct, serial, ladder's problem
    for x, t, s in sizes:
        check(spec, model, [x], t, s, bitwise=False)


class _BigEps(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = mx.full((32,), 1.5)

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.g, eps=0.25)


def test_rms_norm_eps_placement():
    """rms_norm with a large eps: a misplaced eps (outside the sqrt, or
    unscaled by the mean) shifts results by percent, far past tolerance,
    which a tiny fixture eps could never expose at rtol 1e-5."""
    model = _BigEps()
    spec, sizes = lower_swept(model, lambda b: (b, 32), (0, 0), keys=(32, 33, 34))
    for x, t, s in sizes:
        check(spec, model, [x], t, s, bitwise=False)


# -- elementwise coverage -----------------------------------------------------


class _Soup:
    def __call__(self, x):
        a = 2.0 - x            # __rsub__
        b = mx.sin(a)
        c = 0.5 * b            # __rmul__
        d = mx.cos(c)
        e = mx.tanh(d)
        f = mx.abs(e)
        g = mx.sqrt(f)
        h = mx.minimum(g, 0.75)
        i = -h                 # __neg__
        j = i / 3.0            # __truediv__
        k = mx.maximum(j, d)   # binary over two arrays
        return k - b           # __sub__


def test_elementwise_soup_bitwise():
    """A chain over the dunder and mx elementwise surface, scalar-broadcast
    and reflected forms included. metal::precise:: transcendentals match
    library bits (PLATFORM.md spike_04) and every stage rounds per op, so the
    whole chain is bitwise."""
    model = _Soup()
    spec, sizes = lower_swept(model, lambda b: (b, 16), (0, 11), keys=(13, 14, 15))
    for x, t, s in sizes:
        check(spec, model, [x], t, s, bitwise=True)


class _Mask:
    def __call__(self, x):
        return x > 0.5


def test_comparison_bool_output():
    """A comparison region produces a bool array, compared bitwise."""
    model = _Mask()
    spec, sizes = lower_swept(model, lambda b: (b, 16), (0, 0), keys=(16, 17, 18))
    assert spec.output_dtypes[0] == "bool"
    for x, t, s in sizes:
        check(spec, model, [x], t, s, bitwise=True)


class _Sig:
    def __call__(self, x):
        return mx.sigmoid(x)


def test_sigmoid_tolerance():
    """Tolerance comparison: the library's sigmoid kernel is not the
    1/(1+exp(-x)) composition bit for bit (measured within 1 ulp on this
    machine), so bitwise is not achievable for this op."""
    model = _Sig()
    spec, sizes = lower_swept(model, lambda b: (b, 64), (0, 0), keys=(19, 20, 21))
    for x, t, s in sizes:
        check(spec, model, [x], t, s, bitwise=False)


# -- matmul through a transposed weight view ----------------------------------


class _LinT(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(22)
        self.w = mx.random.normal((24, 16))

    def __call__(self, x):
        return x @ self.w.T  # the bias-free nn.Linear pattern


def test_matmul_transposed_weight():
    """x @ w.T: the array.T view folds into the matmul's B indexing.
    Tolerance because of the matmul accumulation order. Batch 9 straddles a
    partial register block (RB rows per threadgroup)."""
    model = _LinT()
    spec, sizes = lower_swept(model, lambda b: (b, 16), (0, 1), keys=(23, 24),
                              batches=(64, 7, 9))
    # the T node reads w first, so w is in0 and x is in1
    assert "in1.shape[0]" in spec.grid[1]
    for x, t, s in sizes:
        check(spec, model, [x], t, s, bitwise=False)


# -- frozen dtypes ------------------------------------------------------------


class _HalfLin(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(35)
        self.w = mx.random.normal((16, 24)).astype(mx.float16)

    def __call__(self, x):
        return mx.maximum(x @ self.w, 0.0)


def test_float16_matmul_accumulates_in_float():
    """fp16 region: the kernel accumulates in float (legal internal
    precision) and stores fp16, like the library, so results agree to a
    couple of fp16 ulps. The template binds T to the input's dtype."""
    model = _HalfLin()
    spec, sizes = lower_swept(model, lambda b: (b, 16), (0, 1),
                              keys=(36, 37, 38), dtype=mx.float16)
    assert spec.template == (("T", "in0"),)
    assert spec.output_dtypes[0] == "float16"
    for x, t, s in sizes:
        check(spec, model, [x], t, s, bitwise=False, rtol=2e-3, atol=1e-3)


class _Mixed:
    def __call__(self, a, b):
        return a * 2.0, mx.exp(b)


def test_mixed_dtypes_use_concrete_casts():
    """Two independent chains in float32 and bfloat16: no shared float dtype
    means no template, and the bfloat16_t store cast must compile. Loose
    tolerance: bfloat16 carries an 8-bit mantissa."""
    model = _Mixed()
    a = mx.random.normal((4, 16), key=mx.random.key(39))
    b = mx.random.normal((4, 16), key=mx.random.key(40)).astype(mx.bfloat16)
    mx.eval(a, b)
    trace, _ = tracer().trace(model, [a, b])
    s = cut(trace, 0, 1)
    spec = lower_naive(trace, s)
    assert spec.template == ()
    assert "bfloat16_t" in spec.source
    check(spec, model, [a, b], trace, s, bitwise=False, rtol=1e-2, atol=1e-2)


# -- determinism --------------------------------------------------------------


def test_kernel_determinism():
    """Two launches on the same inputs agree bitwise: fixed-order loops and a
    fixed reduction tree, no atomics, so the determinism gate has nothing to
    catch."""
    model = load_fixture("norm_three_proj")
    x, t = traced(model, (64, 32), 41)
    s = cut(t, 0, 3)
    spec = lower_naive(t, s)
    arrays = capture_boundaries(tracer(), model, [x], t,
                                set(s.input_ids) | set(s.output_ids))
    inputs = [arrays[a] for a in s.input_ids]
    first = call(spec, inputs)
    second = call(spec, inputs)
    mx.eval(first, second)
    for p, q in zip(first, second):
        assert mx.array_equal(p, q).item()


# -- serialization ------------------------------------------------------------


def test_spec_roundtrips_json():
    """A lowered spec survives KernelSpec JSON serialization and still runs."""
    model = _Soup()
    x, t = traced(model, (4, 16), 26)
    s = cut(t, 0, 11)
    spec = lower_naive(t, s)
    restored = KernelSpec.from_json(spec.to_json())
    arrays = capture_boundaries(tracer(), model, [x], t, set(s.input_ids) | set(s.output_ids))
    inputs = [arrays[a] for a in s.input_ids]
    a_outs = call(spec, inputs)
    b_outs = call(restored, inputs)
    mx.eval(a_outs, b_outs)
    for ga, gb in zip(a_outs, b_outs):
        assert mx.array_equal(ga, gb).item()


# -- quantized matmul ---------------------------------------------------------


# The quantized comparisons use the harness's fp16 preserving tolerance
# (manifest.DEFAULT_TOLERANCES): the kernel and the library both accumulate
# fp32 but in different legal orders, and that is the gate a scaffold must
# actually clear. A region composing TWO quantized levels compounds the
# reassociation past that tolerance, so each test region holds one level,
# which is also what per-projection regions look like on real models.


def test_quantized_4bit_with_glue():
    """4-bit affine quantized_matmul plus elementwise glue as one kernel."""
    model = load_fixture("quantized_linear")
    x, trace = traced(model, (3, 128), 31, dtype=mx.float16)
    s = cut(trace, 0, 1)
    spec = lower_naive(trace, s)
    assert "acc_ += sc_ * dq_ + bi_ * dx_;" in spec.source
    check(spec, model, [x], trace, s, bitwise=False, rtol=1e-2, atol=2e-2)


def test_quantized_8bit_alone():
    """The 8-bit projection as its own region; its boundary input is the
    library's own value, so nothing compounds."""
    model = load_fixture("quantized_linear")
    x, trace = traced(model, (5, 128), 33, dtype=mx.float16)
    s = cut(trace, 2, 2)
    spec = lower_naive(trace, s)
    check(spec, model, [x], trace, s, bitwise=False, rtol=1e-2, atol=2e-2)


def test_quantized_decode_shape():
    """Rank-3 single-row input, the decode GEMV shape."""
    model = load_fixture("quantized_linear")
    x, trace = traced(model, (1, 1, 128), 32, dtype=mx.float16)
    s = cut(trace, 0, 1)
    spec = lower_naive(trace, s)
    check(spec, model, [x], trace, s, bitwise=False, rtol=1e-2, atol=2e-2)


def test_build_scaffold_prefers_stitch_for_lone_qmm():
    """The entry point: a single quantized_matmul region gets the wheel's
    own kernel (bitwise at decode shapes); a region with glue falls back to
    naive lowering."""
    from autotuner.scaffold import build_scaffold

    model = load_fixture("quantized_linear")
    x, trace = traced(model, (1, 1, 128), 34, dtype=mx.float16)
    lone = cut(trace, 0, 0)
    spec = build_scaffold(trace, lone)
    assert spec.kernel_id.startswith("stitch_affine_qmv")
    check(spec, model, [x], trace, lone, bitwise=True)
    glued = cut(trace, 0, 1)
    spec = build_scaffold(trace, glued)
    assert spec.kernel_id.startswith("scaffold_")
    check(spec, model, [x], trace, glued, bitwise=False, rtol=1e-2, atol=2e-2)


# -- rope ---------------------------------------------------------------------


class _Rope:
    def __init__(self, dims, traditional, offset):
        self.dims, self.traditional, self.offset = dims, traditional, offset

    def __call__(self, x):
        y = mx.fast.rope(x, self.dims, traditional=self.traditional,
                         base=500000.0, scale=1.0, offset=self.offset)
        return y * 2.0


def test_rope_with_glue_matches_library():
    """Non-traditional rope at a decode-like offset, plus glue, one kernel.
    fp16 tolerance: the library computes its angles differently at the last
    bit, verified at rounding scale before this lowering was written."""
    x, t = traced(_Rope(32, False, 512), (2, 4, 3, 32), 41, dtype=mx.float16)
    spec = lower_naive(t, cut(t, 0, 1))
    check(spec, _Rope(32, False, 512), [x], t, cut(t, 0, 1),
          bitwise=False, rtol=1e-2, atol=2e-2)


def test_rope_traditional_and_partial_dims():
    """Adjacent-pair rotation over only the leading half of the last axis;
    the tail must copy through untouched."""
    x, t = traced(_Rope(16, True, 7), (2, 3, 5, 32), 42, dtype=mx.float16)
    spec = lower_naive(t, cut(t, 0, 0))
    check(spec, _Rope(16, True, 7), [x], t, cut(t, 0, 0),
          bitwise=False, rtol=1e-2, atol=2e-2)


def test_rope_fp32_small_offset_is_tight():
    """fp32 agreement is tight only while angles are small: the angle grows
    with position, so a last-bit frequency difference amplifies linearly
    with offset. Large offsets are covered by the fp16 gate tests above."""
    x, t = traced(_Rope(64, False, 3), (1, 8, 1, 64), 43, dtype=mx.float32)
    spec = lower_naive(t, cut(t, 0, 0))
    check(spec, _Rope(64, False, 3), [x], t, cut(t, 0, 0),
          bitwise=False, rtol=1e-4, atol=1e-5)


def test_rope_fp16_decode_offset():
    """Single-position row at the real decode offset, model dtype."""
    x, t = traced(_Rope(64, False, 512), (1, 8, 1, 64), 44, dtype=mx.float16)
    spec = lower_naive(t, cut(t, 0, 0))
    check(spec, _Rope(64, False, 512), [x], t, cut(t, 0, 0),
          bitwise=False, rtol=1e-2, atol=2e-2)


# -- layer_norm ---------------------------------------------------------------


def test_layer_norm_affine():
    """layer_norm with weight and bias over the last axis, swept batch. Multi
    mode: one threadgroup per row, two row reductions (mean then centered
    variance) sharing sh_. Tolerance because the threadgroup tree reassociates
    the library's row order; the source assertions pin the two-pass formula
    and the barrier that guards the sh_ reuse."""
    model = load_fixture("layer_norm_affine")
    spec, sizes = lower_swept(model, lambda b: (b, 32), (0, 0), keys=(45, 46, 47))
    assert "shape[0]" in spec.grid[1]  # one threadgroup per row, batch swept
    assert spec.output_shapes[0][1] == "32"
    assert "sv_ += c_ * c_;" in spec.source  # centered (two-pass) variance
    m = spec.source.index("const float mean_ = sh_[0]")
    reuse = spec.source.index("sh_[lid_] = sv_;", m)  # sh_ overwritten for the 2nd reduction
    assert "threadgroup_barrier" in spec.source[m:reuse]  # a barrier guards that reuse
    for x, t, s in sizes:
        check(spec, model, [x], t, s, bitwise=False)


class _LNWeightOnly(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(49)
        self.w = mx.random.normal((32,))

    def __call__(self, x):
        return mx.fast.layer_norm(x, self.w, None, 1e-5)


class _LNNoAffine:
    def __call__(self, x):
        return mx.fast.layer_norm(x, None, None, 1e-5)


def test_layer_norm_optional_affine():
    """weight or bias may be None (affine off on that side). The kernel skips
    the missing term, so a None-affine region reads only x and still matches
    the library."""
    wonly = _LNWeightOnly()
    x, t = traced(wonly, (5, 32), 50)
    s = cut(t, 0, 0)
    spec = lower_naive(t, s)
    assert len(spec.input_names) == 2  # x and weight, no bias
    check(spec, wonly, [x], t, s, bitwise=False)

    plain = _LNNoAffine()
    x, t = traced(plain, (5, 32), 51)
    s = cut(t, 0, 0)
    spec = lower_naive(t, s)
    assert spec.input_names == ("in0",)  # x only, no affine buffers
    check(spec, plain, [x], t, s, bitwise=False)


class _LNBigEps(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(52)
        self.w = mx.random.normal((32,))
        self.b = mx.random.normal((32,))

    def __call__(self, x):
        return mx.fast.layer_norm(x, self.w, self.b, 0.25)


def test_layer_norm_eps_placement():
    """A large eps: misplacing it (outside the sqrt, or unscaled by the row
    length) shifts results by percent, far past tolerance, which a tiny
    fixture eps could never expose."""
    model = _LNBigEps()
    spec, sizes = lower_swept(model, lambda b: (b, 32), (0, 0), keys=(53, 54, 55))
    for x, t, s in sizes:
        check(spec, model, [x], t, s, bitwise=False)


class _LNThenNormalize(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(56)
        self.w = mx.random.normal((32,))
        self.b = mx.random.normal((32,))

    def __call__(self, x):
        h = mx.fast.layer_norm(x, self.w, self.b, 1e-5)
        return h / mx.sum(mx.abs(h), axis=-1, keepdims=True)


def test_layer_norm_single_threadgroup():
    """layer_norm feeding a broadcast divide by its own row sum: the divide
    reads a stage-written (B, 1) buffer across the row, not row-local, so the
    lowering falls back to the single-threadgroup launch. layer_norm must be
    correct in that serial mode too."""
    model = _LNThenNormalize()
    x, t = traced(model, (8, 32), 57)
    s = cut(t, 0, len(t.nodes) - 1)
    spec = lower_naive(t, s)
    assert spec.grid[1] == "1"  # single threadgroup: serial across rows
    check(spec, model, [x], t, s, bitwise=False)


# -- refusal paths ------------------------------------------------------------


def _fake_trace(op: str, kwargs: dict | None = None) -> tuple[Trace, Stretch]:
    node = TraceNode(
        seq=0, op=op, in_arrays=(0,), out_arrays=(1,),
        in_specs=(((4, 4), "float32"),), out_specs=(((4, 4), "float32"),),
        scalar_args={"args": (ArrayRef(0),), "kwargs": kwargs or {}},
        module_address="@0", position_in_module=0, module_stack=("@0",),
    )
    trace = Trace(nodes=(node,), edges={}, step_outputs=(1,), weights=frozenset(),
                  inputs=frozenset({0}), liveness={})
    return trace, Stretch("w", 0, 0, (0,), (1,), ("@0",))


def test_unlowerable_op_raises_named_reason():
    trace, stretch = _fake_trace("mx.frobnicate")
    with pytest.raises(NoScaffold) as e:
        lower_naive(trace, stretch)
    assert e.value.reason == "op-not-lowered"
    assert "frobnicate" in str(e.value)


def test_reduction_over_first_axis_raises():
    class FirstAxis:
        def __call__(self, x):
            return mx.sum(x, axis=0)

    x, t = traced(FirstAxis(), (4, 33), 27)
    with pytest.raises(NoScaffold) as e:
        lower_naive(t, cut(t, 0, 0))
    assert e.value.reason == "reduction-axis-not-last"


def test_view_op_not_lowered_raises():
    class Swap:
        def __call__(self, x):
            return mx.abs(mx.swapaxes(x, 0, 1))  # a view op the scaffold does not fold

    x, t = traced(Swap(), (4, 8), 28)
    with pytest.raises(NoScaffold) as e:
        lower_naive(t, cut(t, 0, len(t.nodes) - 1))
    assert e.value.reason in ("op-not-lowered", "view-not-lowered")


# -- getitem (basic slicing) --------------------------------------------------


class _GetLast:
    def __call__(self, x):
        return x[:, 2:5]  # last-axis slice, start offset zero, batch swept


def test_getitem_last_axis_slice():
    """A last-axis slice of a swept-batch input. The batch axis passes through
    as an accessor and reaches the multi-threadgroup mode (a slice of an input
    is row-local), the sliced axis bakes its literal length. Order preserved,
    so bitwise."""
    spec, sizes = lower_swept(_GetLast(), lambda b: (b, 6), (0, 0), keys=(50, 51, 52))
    assert spec.output_shapes[0] == ("in0.shape[0]", "3")
    assert "in0.shape[0]" in spec.grid[1]  # multi mode: one threadgroup per row
    for x, t, s in sizes:
        check(spec, _GetLast(), [x], t, s, bitwise=True)


class _GetFirst:
    def __call__(self, x):
        return x[1:3, :] + 1.0  # first-axis slice: dense strides, nonzero offset


def test_getitem_first_axis_offset_add():
    """A first-axis slice has dense strides yet a nonzero start offset, so the
    add must read it through the offset path, not the plain flat index. Bitwise
    catches an offset the fast path would drop: it would read the wrong rows."""
    spec, sizes = lower_swept(_GetFirst(), lambda b: (b, 6), (0, 1), keys=(53, 54, 55))
    assert spec.output_shapes[0] == ("2", "6")  # sliced batch is a constant 2
    for x, t, s in sizes:
        check(spec, _GetFirst(), [x], t, s, bitwise=True)


class _GetIntScalar:
    def __call__(self, x):
        return x[1]  # int index on a 1-D input drops the axis: a scalar view


def test_getitem_int_index_scalar():
    """x[1] on a 1-D array has shape () and offset 1: the only nonzero term is
    the offset, so _offset must emit it and not fall back to "0". The output is
    a scalar the copy stage still fills."""
    spec, sizes = lower_swept(_GetIntScalar(), lambda b: (b,), (0, 0), keys=(56, 57, 58))
    assert spec.output_shapes[0] == ()
    for x, t, s in sizes:
        check(spec, _GetIntScalar(), [x], t, s, bitwise=True)


def test_getitem_multi_axis():
    """A multi-axis slice from the fixture: axis 0 whole, axes 1 and 2 sliced,
    so the view carries a compound offset with dense strides."""
    model = load_fixture("slice_read")
    spec, sizes = lower_swept(model, lambda b: (b, 4, 8), (0, 0), keys=(59, 60, 61))
    assert spec.output_shapes[0] == ("in0.shape[0]", "2", "4")
    for x, t, s in sizes:
        check(spec, model, [x], t, s, bitwise=True)


class _MaxOfSlice:
    def __call__(self, x):
        return mx.max(x[1:3, :], axis=-1)  # a reduction reads the offset view


def test_reduce_of_slice_bitwise():
    """A row reduction over a first-axis slice. The reduce's row-offset fast
    path is valid only at zero offset; the slice must route through the offset
    path. max is order-insensitive, so any offset mistake shows up bitwise."""
    spec, sizes = lower_swept(_MaxOfSlice(), lambda b: (b, 6), (0, 1), keys=(62, 63, 64))
    for x, t, s in sizes:
        check(spec, _MaxOfSlice(), [x], t, s, bitwise=True)


def test_getitem_gather_and_step_refused():
    """The cases getitem cannot lower cleanly: an array key is a gather, and a
    step other than 1 is a strided read. Both raise, so a region holding one is
    skipped rather than lowered wrong."""
    class _Gather(nn.Module):
        def __init__(self):
            super().__init__()
            self.idx = mx.array([0, 2, 1])

        def __call__(self, x):
            return mx.abs(x[self.idx])

    x, t = traced(_Gather(), (4, 8), 65)
    with pytest.raises(NoScaffold) as e:
        lower_naive(t, cut(t, 0, len(t.nodes) - 1))
    assert e.value.reason == "op-args-not-lowered" and "gather" in str(e.value)

    class _Step:
        def __call__(self, x):
            return x[:, ::2]

    x, t = traced(_Step(), (4, 8), 66)
    with pytest.raises(NoScaffold) as e:
        lower_naive(t, cut(t, 0, 0))
    assert e.value.reason == "op-args-not-lowered" and "step" in str(e.value)


# -- split (multi-output) -----------------------------------------------------


def test_split_equal_sections():
    """Equal-sections split from the fixture: two views of one buffer at
    different offsets, both region outputs. mlx requires the axis to divide
    evenly, which the recorded call guarantees."""
    model = load_fixture("split_heads")
    spec, sizes = lower_swept(model, lambda b: (b, 6), (0, 1), keys=(67, 68, 69))
    assert spec.output_names[:2] == ("out0", "out1")
    for x, t, s in sizes:
        assert len(s.output_ids) == 2
        check(spec, model, [x], t, s, bitwise=True)


class _SplitIdx:
    def __call__(self, x):
        return mx.split(x, [2, 5], axis=-1)  # cut points -> parts of 2, 3, 1


def test_split_index_list():
    """Split at explicit cut points, three uneven parts as three outputs."""
    spec, sizes = lower_swept(_SplitIdx(), lambda b: (b, 6), (0, 0), keys=(70, 71, 72))
    assert spec.output_names[:3] == ("out0", "out1", "out2")
    for x, t, s in sizes:
        assert len(s.output_ids) == 3
        check(spec, _SplitIdx(), [x], t, s, bitwise=True)


# -- concatenate --------------------------------------------------------------


class _ConcatSliced:
    def __call__(self, x):
        return mx.concatenate([x[:, 0:2], x[:, 4:6]], axis=-1)  # sliced sources


def test_concat_last_axis_sliced_sources():
    """Concatenate two slices along the last axis. The concat reads each source
    through its own view, so the second source's nonzero offset is honored. It
    runs single-threadgroup; the ladder, not this module, judges its speed."""
    spec, sizes = lower_swept(_ConcatSliced(), lambda b: (b, 6), (0, 2), keys=(73, 74, 75))
    assert spec.output_shapes[0] == ("in0.shape[0]", "4")
    assert spec.grid[1] == "1"  # concat regions launch single-threadgroup
    for x, t, s in sizes:
        check(spec, _ConcatSliced(), [x], t, s, bitwise=True)


class _ConcatMid:
    def __call__(self, x):
        return mx.concatenate([mx.abs(x), x], axis=1)  # join along a middle axis


def test_concat_middle_axis():
    """Concatenate along a middle axis, so the flat write index decomposes and
    the if-ladder selects the source by the middle-axis index."""
    spec, sizes = lower_swept(_ConcatMid(), lambda b: (b, 3, 8), (0, 1), keys=(76, 77, 78))
    assert spec.output_shapes[0] == ("in0.shape[0]", "6", "8")
    for x, t, s in sizes:
        check(spec, _ConcatMid(), [x], t, s, bitwise=True)


def test_composed_split_ew_concat():
    """The composed pipeline from the fixture: split into two offset views, run
    a different elementwise op on each, concatenate back. It proves offset views
    feed stages and that concat reads their outputs, all in one kernel."""
    model = load_fixture("concat_join")
    spec, sizes = lower_swept(model, lambda b: (b, 6), (0, 3), keys=(79, 80, 81))
    assert spec.output_shapes[0] == ("in0.shape[0]", "6")
    for x, t, s in sizes:
        check(spec, model, [x], t, s, bitwise=True)
