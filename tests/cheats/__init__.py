"""The cheat zoo: bad kernels, each built to die at ONE
ladder gate. The harness's job is rejecting bad kernels, so its tests are
these. Builders return (KernelSpec, description); tests/test_ladder_gates.py
runs each up the full ladder and asserts the intended gate catches it.

Two shared toy regions, both plain-function models the repo Tracer records:

- the elementwise toy: y = 1 / (x*x + 1e-6) on fp32 x of shape (L, 8), L
  sweepable. Inputs are drawn from U(0.5, 1.5) so dropping the eps only shows
  at the tiny-scale regime, and the primary L=64 gives 512 elements, small
  enough to bake into source literals for the cached-by-shape cheat.
- the reduction toy: y = sum(x, axis=-1). The fp16 variant (8 x 4096, values
  0.003 * randn) hosts the sloppy-accumulation cheat: sequential half
  accumulation stays inside fp16 tolerances at recorded values and drifted far
  outside them at the 1e3 scaled-up regime, which was removed 2026-09-17; no
  gate catches it now (the ladder test is a strict xfail). The
  fp32 variant (4 x 16384, positive values) hosts the atomic-racy cheat:
  reordering error ~1e-6 relative passes the changing gate while three runs
  are bitwise distinct (PLATFORM spike_06 float-atomics nondeterminism).
"""

from __future__ import annotations

import mlx.core as mx

from autotuner_runtime.kernels import KernelSpec

TOY_L, TOY_D = 64, 8
TOY_EPS = 1e-6
RED16_L, RED16_D = 8, 4096
RED32_L, RED32_D = 4, 16384


def toy_model(x):
    return 1.0 / (x * x + TOY_EPS)


def toy_input(key: int, L: int = TOY_L) -> mx.array:
    return mx.random.uniform(low=0.5, high=1.5, shape=(L, TOY_D), key=mx.random.key(key))


def reduction_model(x):
    return mx.sum(x, axis=-1)


def reduction_input_f16(key: int) -> mx.array:
    x = mx.random.normal((RED16_L, RED16_D), key=mx.random.key(key)) * 0.003
    # center each row so its sum sits near zero: the tolerance's rtol term
    # stays negligible and any off-library accumulation order shows at scale
    return (x - mx.mean(x, axis=-1, keepdims=True)).astype(mx.float16)


def reduction_input_f32(key: int) -> mx.array:
    x = mx.abs(mx.random.normal((RED32_L, RED32_D), key=mx.random.key(key)))
    return x * 0.01 + 0.05


_TOY_CORRECT = """uint i = thread_position_in_grid.x;
float v = x[i];
y[i] = 1.0f / (v * v + 1e-6f);
"""


def _toy_spec(name: str, source: str, **over) -> KernelSpec:
    base = dict(
        kernel_id=name,
        name=name,
        input_names=("x",),
        output_names=("y",),
        source=source,
        grid=("in0.shape[0] * in0.shape[1]", "1", "1"),
        threadgroup=("32", "1", "1"),
        output_shapes=(("in0.shape[0]", "in0.shape[1]"),),
        output_dtypes=("float32",),
    )
    base.update(over)
    return KernelSpec(**base)


def toy_correct() -> tuple[KernelSpec, str]:
    return _toy_spec("cheat_toy_correct", _TOY_CORRECT), "the honest fused toy kernel"


def partial_write() -> tuple[KernelSpec, str]:
    src = """uint i = thread_position_in_grid.x;
uint total = (uint)(x_shape[0] * x_shape[1]);
if (i < total / 2u) {
  float v = x[i];
  y[i] = 1.0f / (v * v + 1e-6f);
}
"""
    return _toy_spec("cheat_partial_write", src), (
        "writes only the first half of its output; init_value=nan makes the "
        "rest NaN deterministically, so the poison gate catches it"
    )


def shape_hardcoded() -> tuple[KernelSpec, str]:
    src = """uint i = thread_position_in_grid.x;
float v = x[i % 512u];
y[i] = 1.0f / (v * v + 1e-6f);
"""
    return _toy_spec("cheat_shape_hardcoded", src), (
        "bakes the primary size (64 x 8 = 512 elements) into its indexing: "
        "exact at the recorded shape, wrong values at the large sweep size"
    )


def stride_lying() -> tuple[KernelSpec, str]:
    return _toy_spec("cheat_stride_lying", _TOY_CORRECT, ensure_row_contiguous=False), (
        "correct math with flat indexing under ensure_row_contiguous=False: "
        "the transposed-input variant hands it the raw transposed buffer"
    )


def cached_by_shape(reference: mx.array) -> tuple[KernelSpec, str]:
    """reference is the stored set-0 library output; its values become source
    literals, so the kernel reproduces set 0 exactly and dies on k-set
    variation."""
    flat = reference.reshape(-1).astype(mx.float32)
    lits = ", ".join(f"{float(v):.9g}f" for v in flat.tolist())
    header = f"constant float mao_baked[{flat.size}] = {{{lits}}};\n"
    src = """uint i = thread_position_in_grid.x;
(void)x;
y[i] = mao_baked[i];
"""
    return _toy_spec("cheat_cached_by_shape", src, header=header), (
        "ignores input values and emits outputs precomputed for the primary "
        "set-0 inputs, smuggled in as source literals"
    )


def eps_dropping() -> tuple[KernelSpec, str]:
    src = """uint i = thread_position_in_grid.x;
float v = x[i];
y[i] = 1.0f / (v * v);
"""
    return _toy_spec("cheat_eps_dropping", src), (
        "skips the + 1e-6: indistinguishable at recorded values bounded away "
        "from zero, wildly wrong under the tiny-scale regime"
    )


def fallback_declared_but_dead() -> tuple[KernelSpec, str]:
    spec, _ = shape_hardcoded()
    spec = _toy_spec("cheat_fallback_dead", spec.source,
                     fallback_predicate="in0.shape[0] > 100000")
    return spec, (
        "shape-hardcoded wrongness plus a fallback predicate that never "
        "fires at any sweep size: declared coverage, dead in practice"
    )


def live_output_dropping() -> tuple[KernelSpec, str]:
    return _toy_spec("cheat_live_output_dropped", _TOY_CORRECT), (
        "produces only y against a contract whose live values include z: "
        "gate 1 static checks, no GPU work"
    )


def fallback_missing() -> tuple[KernelSpec, str]:
    return _toy_spec("cheat_fallback_missing", _TOY_CORRECT), (
        "shape-specialized (contract requires a fallback) but declares no "
        "fallback predicate: gate 1 static checks"
    )


def _reduction_spec(name: str, source: str, dtype: str, **over) -> KernelSpec:
    base = dict(
        kernel_id=name,
        name=name,
        input_names=("x",),
        output_names=("y",),
        source=source,
        grid=("in0.shape[0]", "1", "1"),
        threadgroup=("8", "1", "1"),
        output_shapes=(("in0.shape[0]",),),
        output_dtypes=(dtype,),
    )
    base.update(over)
    return KernelSpec(**base)


_SLOPPY = """uint r = thread_position_in_grid.x;
uint D = (uint)x_shape[1];
half acc = half(0.0);
for (uint j = 0; j < D; ++j) { acc += x[r * D + j]; }
y[r] = acc;
"""

_FP32_ACC = """uint r = thread_position_in_grid.x;
uint D = (uint)x_shape[1];
float acc = 0.0f;
for (uint j = 0; j < D; ++j) { acc += (float)x[r * D + j]; }
y[r] = (half)acc;
"""


def fp16_sloppy_accumulation() -> tuple[KernelSpec, str]:
    return _reduction_spec("cheat_fp16_sloppy", _SLOPPY, "float16"), (
        "sequential half accumulator over a 4096-long sum: inside fp16 "
        "tolerances at recorded values, far outside them once inputs are "
        "scaled up 1e3 (no regime does that since 2026-09-17)"
    )


def fp16_fp32_accumulation() -> tuple[KernelSpec, str]:
    """Not a cheat: the deliberate reordering control. Accumulates in fp32,
    so it is MORE accurate than the library's fp16 tree sum; the preserving
    gate still kills it (it does not reproduce the library's own rounding),
    and the changing gate passes it within the configured tolerance."""
    return _reduction_spec("ctrl_fp16_fp32acc", _FP32_ACC, "float16"), (
        "fp32 accumulator over the fp16 sum: assoc-changing by intent"
    )


def atomic_racy() -> tuple[KernelSpec, str]:
    src = """uint i = thread_position_in_grid.x;
uint D = (uint)x_shape[1];
uint r = i / D;
metal::atomic_fetch_add_explicit(&y[r], x[i], metal::memory_order_relaxed);
"""
    spec = _reduction_spec(
        "cheat_atomic_racy", src, "float32",
        grid=("in0.shape[0] * in0.shape[1]", "1", "1"),
        threadgroup=("256", "1", "1"),
        atomic_outputs=True,
    )
    return spec, (
        "atomic float sum: values pass the changing golden gate, but three "
        "runs are never bitwise identical, so determinism kills it"
    )


# name -> (builder, the gate that must catch it). cached_by_shape takes the
# stored set-0 reference and is registered by the test that owns the store.
CHEATS = {
    "partial_write": (partial_write, "poison"),
    "shape_hardcoded": (shape_hardcoded, "sweep"),
    "stride_lying": (stride_lying, "sweep"),
    "eps_dropping": (eps_dropping, "smoke"),
    "fallback_declared_but_dead": (fallback_declared_but_dead, "sweep"),
    "live_output_dropping": (live_output_dropping, "static"),
    "fallback_missing": (fallback_missing, "static"),
    "fp16_sloppy_accumulation": (fp16_sloppy_accumulation, "smoke"),
    "atomic_racy": (atomic_racy, "determinism"),
}
