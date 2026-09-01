"""Stitched quantized_matmul scaffold vs the library kernel, on the real GPU.

mx.quantized_matmul's own dispatch is the reference. On decode shapes (x rows
M == 1) both sides run the wheel's qmv algorithm, so outputs must be bitwise
equal, aligned or not. For M > 1 the library may dispatch a different kernel
with another accumulation order, so that case compares within a few fp16
ulps instead."""

import statistics
import time

import mlx.core as mx
import pytest

from autotuner.scaffold import build_scaffold
from autotuner.scaffold.stitch import (
    _QMV_ROOTS,
    flatten_header,
    stitch_qmm_chain,
    stitch_quantized_matmul,
)
from autotuner.scaffold.symshape import NoScaffold
from autotuner.trace.recorder import dtype_name
from autotuner_runtime.kernels import KernelSpec, call
from tests.conftest import require_healthy_gpu, require_quiet_load
from tests.test_scaffold import check, cut, tracer

DECODE_KN = [(4096, 4096), (4096, 1024), (4096, 14336), (14336, 4096)]


def make_case(lead, K, N, group_size, bits, dtype):
    w = mx.random.normal((N, K), key=mx.random.key(1)).astype(dtype)
    wq, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    x = (mx.random.normal((*lead, K), key=mx.random.key(2)) * 0.5).astype(dtype)
    mx.eval(x, wq, scales, biases)
    return x, wq, scales, biases


def spec_of(a):
    return (tuple(a.shape), dtype_name(a.dtype))


def stitched_and_ref(lead, K, N, group_size, bits, dtype):
    inputs = make_case(lead, K, N, group_size, bits, dtype)
    spec = stitch_quantized_matmul(*(spec_of(a) for a in inputs),
                                   group_size=group_size, bits=bits)
    out = call(spec, list(inputs))[0]
    ref = mx.quantized_matmul(*inputs, transpose=True, group_size=group_size, bits=bits)
    mx.eval(out, ref)
    return spec, out, ref


def assert_bitwise(out, ref):
    assert tuple(out.shape) == tuple(ref.shape)
    assert out.dtype == ref.dtype
    diff = mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item()
    assert mx.array_equal(out, ref).item(), f"not bitwise equal, max abs diff {diff}"


@pytest.mark.parametrize("K,N", DECODE_KN)
def test_decode_shapes_bitwise_4bit(K, N):
    spec, out, ref = stitched_and_ref((1, 1), K, N, 64, 4, mx.float16)
    assert "fast" in spec.name
    assert_bitwise(out, ref)


def test_decode_bitwise_8bit():
    spec, out, ref = stitched_and_ref((1, 1), 4096, 4096, 64, 8, mx.float16)
    assert "fast" in spec.name
    assert_bitwise(out, ref)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_decode_bitwise_other_dtypes(dtype):
    _, out, ref = stitched_and_ref((1, 1), 512, 512, 64, 4, dtype)
    assert_bitwise(out, ref)


@pytest.mark.parametrize("group_size", [32, 128])
def test_decode_bitwise_other_group_sizes(group_size):
    _, out, ref = stitched_and_ref((1, 1), 512, 512, group_size, 4, mx.float16)
    assert_bitwise(out, ref)


@pytest.mark.parametrize("K,N,bits", [(4096, 1001, 4), (4160, 1000, 4), (4160, 1000, 8)])
def test_general_variant_unaligned_bitwise(K, N, bits):
    """Off the fast-path alignment (N not a multiple of 8, or K not a multiple
    of a full block) the library and the stitch both fall back to the guarded
    qmv, still bitwise on decode shapes."""
    spec, out, ref = stitched_and_ref((1, 1), K, N, 64, bits, mx.float16)
    assert "gen" in spec.name
    assert_bitwise(out, ref)


def test_small_batch_rows_within_fp16():
    """M=6: the stitched qmv covers every x row via the grid; the library may
    use another kernel here, so agree within accumulation-order rounding."""
    _, out, ref = stitched_and_ref((2, 3), 4096, 1024, 64, 4, mx.float16)
    assert tuple(out.shape) == tuple(ref.shape)
    ref32 = ref.astype(mx.float32)
    diff = mx.max(mx.abs(out.astype(mx.float32) - ref32)).item()
    bound = 8 * 2**-10 * max(1.0, mx.max(mx.abs(ref32)).item())  # 8 fp16 ulps
    assert diff <= bound, f"max abs diff {diff} exceeds {bound}"


def test_spec_survives_json_roundtrip():
    """The artifact stores specs as JSON; a reloaded spec must launch the
    same kernel and reproduce the library bitwise."""
    inputs = make_case((1, 1), 4096, 1024, 64, 4, mx.float16)
    spec = stitch_quantized_matmul(*(spec_of(a) for a in inputs))
    reloaded = KernelSpec.from_json(spec.to_json())
    assert reloaded == spec
    out = call(reloaded, list(inputs))[0]
    ref = mx.quantized_matmul(*inputs, transpose=True, group_size=64, bits=4)
    mx.eval(out, ref)
    assert_bitwise(out, ref)


def test_flattened_header_is_self_contained():
    header = flatten_header(_QMV_ROOTS)
    assert header.endswith("\n")  # spike_08: no trailing newline swallows the signature
    assert not any(l.lstrip().startswith('#include "') for l in header.splitlines())
    for symbol in ("qmv_fast_impl", "qmv_impl", "qdot", "load_vector"):
        assert symbol in header
    # the prelude set stays out: re-inlining utils.h redefines its symbols
    assert "bfloat16_to_uint16" not in header


X, W, S = ((1, 1, 4096), "float16"), ((4096, 512), "uint32"), ((4096, 64), "float16")


@pytest.mark.parametrize("args,reason", [
    (((X, W, S, S), {"bits": 3}), "stitch-bits"),
    (((((1, 1, 4096), "int32"), W, S, S), {}), "stitch-dtype"),
    (((X, ((4096, 512), "uint16"), S, S), {}), "stitch-dtype"),
    (((X, W, ((4096, 64), "float32"), S), {}), "stitch-dtype"),
    (((X, ((4096, 500), "uint32"), S, S), {}), "stitch-shape"),          # w does not pack K
    (((X, W, ((4096, 63), "float16"), S), {}), "stitch-shape"),          # wrong group count
    (((X, W, S, S), {"group_size": 60}), "stitch-shape"),                # K not divisible
    (((((4096,), "float16"), ((10, 1024), "uint32"),
       ((10, 2048), "float16"), ((10, 2048), "float16")), {"group_size": 2, "bits": 8}),
     "stitch-group-size"),                                               # gen path needs gs % pack
])
def test_rejects_unsupported(args, reason):
    posargs, kwargs = args
    with pytest.raises(NoScaffold) as e:
        stitch_quantized_matmul(*posargs, **kwargs)
    assert e.value.reason == reason


# -- stitched qmm + row-local elementwise chain --------------------------------
#
# Real traces (the recorder, region cutting, build_scaffold) at the Llama 3 8B
# decode shapes: x is (1, 1, K) fp16, weights 4-bit affine group 64 transposed.
# The matmul bits are the library's own; the fused chain keeps float
# intermediates and library sigmoid is 1 ulp off the precise::exp composition
# (PLATFORM.md), so comparisons use the harness's fp16 tolerance.

FP16_TOL = dict(rtol=1e-2, atol=2e-2)


class _QmmChain:
    """quantized_matmul then a trailing elementwise tail, decode-shaped."""

    def __init__(self, K, N, tail, key=1, dtype=mx.float16):
        w = mx.random.normal((N, K), key=mx.random.key(key)).astype(dtype)
        self.wq, self.scales, self.biases = mx.quantize(w, group_size=64, bits=4)
        self.tail = tail
        mx.eval(self.wq, self.scales, self.biases)

    def __call__(self, x, *rest):
        y = mx.quantized_matmul(x, self.wq, self.scales, self.biases,
                                transpose=True, group_size=64, bits=4)
        return self.tail(y, *rest)


def chain_case(K, N, tail, extras=(), key=20, lead=(1, 1), dtype=mx.float16):
    """Trace the tail on a decode x (*lead, K) and cut the whole trace."""
    model = _QmmChain(K, N, tail, key=key, dtype=dtype)
    x = (mx.random.normal((*lead, K), key=mx.random.key(key + 1)) * 0.5).astype(dtype)
    inputs = [x, *extras]
    mx.eval(inputs)
    trace, _ = tracer().trace(model, inputs)
    return model, inputs, trace, cut(trace, 0, len(trace.nodes) - 1)


def test_chain_sigmoid_decode():
    """quantized_matmul then mx.sigmoid at a decode projection shape."""
    model, xs, t, s = chain_case(4096, 1024, lambda y: mx.sigmoid(y))
    spec = build_scaffold(t, s)
    assert "fast" in spec.name and "_chain1_" in spec.name
    check(spec, model, xs, t, s, bitwise=False, **FP16_TOL)


def test_chain_silu_decode():
    """sigmoid then multiply by the matmul's own output (silu), at the
    gate-proj shape 4096 -> 14336."""
    model, xs, t, s = chain_case(4096, 14336, lambda y: mx.sigmoid(y) * y, key=22)
    spec = build_scaffold(t, s)
    assert "_chain2_" in spec.name
    check(spec, model, xs, t, s, bitwise=False, **FP16_TOL)


def test_chain_silu_times_region_input():
    """sigmoid, mul by own output, mul by a same-shape region input: the
    fused gate * up pattern. The extra input binds after the qmm's four."""
    up = (mx.random.normal((1, 1, 14336), key=mx.random.key(8)) * 0.5).astype(mx.float16)
    model, xs, t, s = chain_case(4096, 14336, lambda y, u: mx.sigmoid(y) * y * u,
                                 extras=(up,), key=24)
    spec = build_scaffold(t, s)
    assert "_chain3_" in spec.name
    assert spec.input_names == ("in0", "in1", "in2", "in3", "in4")
    check(spec, model, xs, t, s, bitwise=False, **FP16_TOL)


def test_chain_scalar_and_row_aligned_add():
    """A reflected python-scalar multiply (0.5 * y records array.__rmul__ and
    exercises the operand swap), then adding a rank-1 (N,) region input that
    broadcasts across the output rows."""
    u = (mx.random.normal((1024,), key=mx.random.key(9)) * 0.5).astype(mx.float16)
    model, xs, t, s = chain_case(4096, 1024, lambda y, u: 0.5 * y + u,
                                 extras=(u,), key=26)
    spec = build_scaffold(t, s)
    assert "_chain2_" in spec.name
    check(spec, model, xs, t, s, bitwise=False, **FP16_TOL)


def test_chain_scalar_shaped_array_input():
    """A (1, 1, 1) region input broadcasts to every element via a fixed load."""
    u = (mx.ones((1, 1, 1)) * 0.75).astype(mx.float16)
    mx.eval(u)
    model, xs, t, s = chain_case(512, 512, lambda y, u: mx.sigmoid(y) * u,
                                 extras=(u,), key=32)
    spec = build_scaffold(t, s)
    assert "in4[0]" in spec.source
    check(spec, model, xs, t, s, bitwise=False, **FP16_TOL)


def test_chain_bfloat16():
    """The template dtype carries the chain: bf16 silu at a small shape."""
    model, xs, t, s = chain_case(512, 512, lambda y: mx.sigmoid(y) * y,
                                 key=34, dtype=mx.bfloat16)
    spec = build_scaffold(t, s)
    assert "_chain2_" in spec.name
    check(spec, model, xs, t, s, bitwise=False, rtol=5e-2, atol=5e-2)


def test_chain_small_batch_rows():
    """M=6: the epilogue's flat row index covers every x row of the grid. The
    library may dispatch another matmul kernel at M > 1, so compare with the
    fp16-ulps-of-scale bound of test_small_batch_rows_within_fp16."""
    model, xs, t, s = chain_case(512, 512, lambda y: mx.sigmoid(y) * y,
                                 key=36, lead=(2, 3))
    spec = build_scaffold(t, s)
    assert "_chain2_" in spec.name
    out = call(spec, [xs[0], model.wq, model.scales, model.biases])[0]
    ref = model(xs[0])
    mx.eval(out, ref)
    assert tuple(out.shape) == (2, 3, 512) and out.dtype == ref.dtype
    ref32 = ref.astype(mx.float32)
    diff = mx.max(mx.abs(out.astype(mx.float32) - ref32)).item()
    bound = 8 * 2**-10 * max(1.0, mx.max(mx.abs(ref32)).item())  # 8 fp16 ulps
    assert diff <= bound, f"max abs diff {diff} exceeds {bound}"


class _TwoQmm:
    """Two chained quantized_matmuls: the second is not stitchable."""

    def __init__(self, K, N):
        for name, shape, key in (("a", (N, K), 11), ("b", (K, N), 12)):
            w = mx.random.normal(shape, key=mx.random.key(key)).astype(mx.float16)
            setattr(self, name, mx.quantize(w, group_size=64, bits=4))

    def __call__(self, x):
        y = mx.quantized_matmul(x, *self.a, transpose=True, group_size=64, bits=4)
        return mx.quantized_matmul(mx.sigmoid(y), *self.b,
                                   transpose=True, group_size=64, bits=4)


def test_chain_second_qmm_falls_back_to_naive():
    """A second quantized_matmul in the tail raises NoScaffold, and
    build_scaffold falls back to naive lowering, which stays correct. The
    naive serial k loop reassociates both accumulations, so the comparison
    is the fp16-ulps-of-scale bound of test_small_batch_rows_within_fp16,
    not elementwise allclose (cancellation makes small elements miss atol)."""
    model = _TwoQmm(512, 512)
    x = (mx.random.normal((1, 1, 512), key=mx.random.key(13)) * 0.5).astype(mx.float16)
    mx.eval(x)
    t, _ = tracer().trace(model, [x])
    s = cut(t, 0, len(t.nodes) - 1)
    with pytest.raises(NoScaffold) as e:
        stitch_qmm_chain(t.nodes, s.input_ids, s.output_ids)
    assert e.value.reason == "stitch-chain-op"
    spec = build_scaffold(t, s)
    assert spec.name.startswith("scaffold_")
    out = call(spec, [x, *model.a, *model.b])[0]
    ref = model(x)
    mx.eval(out, ref)
    ref32 = ref.astype(mx.float32)
    diff = mx.max(mx.abs(out.astype(mx.float32) - ref32)).item()
    bound = 8 * 2**-10 * max(1.0, mx.max(mx.abs(ref32)).item())  # 8 fp16 ulps
    assert diff <= bound, f"max abs diff {diff} exceeds {bound}"


def test_chain_off_fast_alignment_falls_back_to_naive():
    """N % 8 != 0 selects the guarded qmv variant, whose edge threadgroup
    redoes rows of the previous block; the chain fuses only the fast variant,
    so it refuses and naive takes the region."""
    model, xs, t, s = chain_case(512, 1001, lambda y: mx.sigmoid(y), key=28)
    with pytest.raises(NoScaffold) as e:
        stitch_qmm_chain(t.nodes, s.input_ids, s.output_ids)
    assert e.value.reason == "stitch-chain-variant"
    spec = build_scaffold(t, s)
    assert spec.name.startswith("scaffold_")
    check(spec, model, xs, t, s, bitwise=False, **FP16_TOL)


def test_chain_prefix_ships_matmul_and_sigmoid():
    """The real ranked [quantized_matmul, sigmoid] region: cut below the mul,
    the matmul output and the sigmoid are both live outside, so the kernel
    ships both. The matmul output passes through the qmv untouched, so it
    stays bitwise; the sigmoid is tolerance-level."""
    model, xs, t, _ = chain_case(4096, 1024, lambda y: mx.sigmoid(y) * y, key=29)
    s = cut(t, 0, 1)
    assert len(s.output_ids) == 2
    assert s.output_ids[0] == t.nodes[0].out_arrays[0]  # y ships first
    spec = build_scaffold(t, s)
    assert "_chain1_" in spec.name
    assert spec.output_names == ("out0", "out1")
    outs = call(spec, [xs[0], model.wq, model.scales, model.biases])
    y_ref = mx.quantized_matmul(xs[0], model.wq, model.scales, model.biases,
                                transpose=True, group_size=64, bits=4)
    s_ref = mx.sigmoid(y_ref)
    mx.eval(outs, y_ref, s_ref)
    assert_bitwise(outs[0], y_ref)
    assert mx.allclose(outs[1], s_ref, **FP16_TOL).item()


def test_chain_prefix_ships_matmul_and_mul():
    """The real ranked [quantized_matmul, sigmoid, mul] region at the 8B MLP
    shape: the mul takes the up projection from outside, and the raw matmul
    output is consumed again after the region, so both values ship."""
    up = (mx.random.normal((1, 1, 14336), key=mx.random.key(31)) * 0.5).astype(mx.float16)
    model, xs, t, _ = chain_case(4096, 14336, lambda y, u: mx.sigmoid(y) * u * y,
                                 extras=(up,), key=33)
    s = cut(t, 0, 2)
    assert len(s.output_ids) == 2
    spec = build_scaffold(t, s)
    assert "_chain2_" in spec.name
    outs = call(spec, [xs[0], model.wq, model.scales, model.biases, up])
    y_ref = mx.quantized_matmul(xs[0], model.wq, model.scales, model.biases,
                                transpose=True, group_size=64, bits=4)
    m_ref = mx.sigmoid(y_ref) * up
    mx.eval(outs, y_ref, m_ref)
    assert_bitwise(outs[0], y_ref)
    assert mx.allclose(outs[1], m_ref, **FP16_TOL).item()


class _UpSilu:
    """The real 8B MLP wiring around the up projection: the chain applies
    sigmoid and mul to the gate value g, a region INPUT, and the qmm's own
    output joins only at the end (or not at all in shorter cuts)."""

    def __init__(self, K, N, key=41):
        w = mx.random.normal((N, K), key=mx.random.key(key)).astype(mx.float16)
        self.wq, self.scales, self.biases = mx.quantize(w, group_size=64, bits=4)
        mx.eval(self.wq, self.scales, self.biases)

    def __call__(self, x, g):
        y = mx.quantized_matmul(x, self.wq, self.scales, self.biases,
                                transpose=True, group_size=64, bits=4)
        s = mx.sigmoid(g)
        return g * s * y


def up_silu_case(K, N, end, key=42):
    """Trace the real wiring and cut [qmm, sigmoid, ...] up to node `end`."""
    model = _UpSilu(K, N, key=key)
    x = (mx.random.normal((1, 1, K), key=mx.random.key(key + 1)) * 0.5).astype(mx.float16)
    g = (mx.random.normal((1, 1, N), key=mx.random.key(key + 2)) * 0.5).astype(mx.float16)
    mx.eval(x, g)
    t, _ = tracer().trace(model, [x, g])
    assert [n.op for n in t.nodes] == \
        ["mx.quantized_matmul", "mx.sigmoid", "array.__mul__", "array.__mul__"]
    s = cut(t, 0, end)
    outs = call(build_scaffold(t, s), [x, model.wq, model.scales, model.biases, g])
    y_ref = mx.quantized_matmul(x, model.wq, model.scales, model.biases,
                                transpose=True, group_size=64, bits=4)
    mx.eval(outs, y_ref)
    return s, outs, y_ref, g


@pytest.mark.parametrize("end,n_out", [(1, 2), (2, 2), (3, 1)])
def test_chain_on_region_input_real_wiring(end, n_out):
    """The three ranked 8B decode cuts of the up-projection region, at the
    real MLP shape: [qmm, sigmoid], [qmm, sigmoid, mul], and the full
    [qmm, sigmoid, mul, mul]. The chain reads g from outside the region; the
    untouched qmm output ships bitwise wherever it is live."""
    s, outs, y_ref, g = up_silu_case(4096, 14336, end)
    assert len(s.output_ids) == n_out
    refs = {1: [y_ref, mx.sigmoid(g)],
            2: [y_ref, g * mx.sigmoid(g)],
            3: [g * mx.sigmoid(g) * y_ref]}[end]
    mx.eval(refs)
    for got, ref in zip(outs, refs):
        assert tuple(got.shape) == tuple(ref.shape) and got.dtype == ref.dtype
        assert mx.allclose(got, ref, **FP16_TOL).item()
    if n_out == 2:
        assert_bitwise(outs[0], y_ref)


def test_chain_intermediate_fp16_overflow_matches_library():
    """The library materializes an fp16 tensor after every op, so a product
    past fp16 max becomes inf there and inf * 0 becomes nan. The fused chain
    must round-trip each intermediate through the boundary dtype so the same
    elements overflow: the non-finite pattern must match the library exactly.
    A float32-register chain would keep the product finite and return 0."""
    model, xs, t, s = chain_case(512, 512, lambda y: y * 4096.0 * 0.0, key=38)
    x = (xs[0].astype(mx.float32) * 16).astype(mx.float16)  # push |y| past 16
    mx.eval(x)
    t, _ = tracer().trace(model, [x])
    s = cut(t, 0, len(t.nodes) - 1)
    spec = build_scaffold(t, s)
    assert "_chain2_" in spec.name
    out = call(spec, [x, model.wq, model.scales, model.biases])[0]
    ref = model(x)  # y fp16 -> inf where |y * 4096| > 65504 -> nan after * 0
    mx.eval(out, ref)
    ref_nan = mx.isnan(ref)
    assert 0 < mx.sum(ref_nan).item() < ref.size  # the pattern is nontrivial
    assert mx.array_equal(mx.isnan(out), ref_nan).item(), "nan pattern differs"
    assert mx.array_equal(mx.isinf(out), mx.isinf(ref)).item()
    assert mx.array_equal(mx.where(ref_nan, mx.zeros_like(out), out),
                          mx.where(ref_nan, mx.zeros_like(ref), ref)).item()


def test_chain_output_ids_must_be_chain_values():
    """No outputs, or an output that is not the matmul output or a chain
    value, has nothing the epilogue could ship."""
    _, _, t, s = chain_case(512, 512, lambda y: mx.sigmoid(y), key=35)
    for bad in ((), (s.input_ids[0],)):
        with pytest.raises(NoScaffold) as e:
            stitch_qmm_chain(t.nodes, s.input_ids, bad)
        assert e.value.reason == "stitch-chain-outputs"


def test_chain_timing_within_3x_of_library():
    """Fused silu chain vs the separate library calls at the biggest decode
    shape. Near 1x is expected; 3x is the failure line (naive was ~9x)."""
    require_healthy_gpu()
    require_quiet_load()
    model, xs, t, s = chain_case(4096, 14336, lambda y: mx.sigmoid(y) * y, key=30)
    spec = build_scaffold(t, s)
    assert "_chain2_" in spec.name
    x = xs[0]
    kin = [x, model.wq, model.scales, model.biases]  # slots 0..3 by construction

    def stitched():
        return call(spec, kin)[0]

    def library():
        y = mx.quantized_matmul(x, model.wq, model.scales, model.biases,
                                transpose=True, group_size=64, bits=4)
        return mx.sigmoid(y) * y

    mx.eval(stitched(), library())  # warm and compile both
    ts, tl = [], []
    for _ in range(12):  # paired and interleaved (plan 6); short for the fanless chip
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(stitched())
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        mx.eval(library())
        mx.synchronize()
        tl.append(time.perf_counter() - t0)
    ratio = statistics.median(ts) / statistics.median(tl)
    assert ratio < 3.0, f"stitched/library ratio {ratio:.2f}"


def test_stitch_uninstall_last():
    """This file re-installs the shared tracer after test_scaffold's own
    uninstall-last has run, so it must restore the patch surface itself or
    the next module inherits a patched mx."""
    from tests import test_scaffold

    tr = tracer()
    tr.uninstall()
    assert tr.verify_restored() == []
    test_scaffold._tracer = None
