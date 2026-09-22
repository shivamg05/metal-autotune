"""The complete ladder. Every cheat in the zoo climbs the full ladder and
must die at its intended gate; the positive controls prove correct kernels get
honest verdicts (correct_slower with numbers, tentative_ship for a planted
win). Eval sets are built with the repo's real Tracer, capture_boundaries,
and BoundaryStore, so what the child sees is exactly what the loop will feed.
"""

import statistics
from dataclasses import asdict, dataclass

import importlib.util
from pathlib import Path

import mlx.core as mx
import pytest

from autotuner.ladder.gates import EvalSet, LadderJob, LadderResult, run_ladder
from autotuner.ladder.static_checks import RegionContract
from autotuner.measure.session import time_once
from autotuner.regions.build import build_stretches
from autotuner.regions.price import capture_boundaries
from autotuner.regions.store import BoundaryStore, load_set
from autotuner.trace import Tracer
from autotuner.trace.replay import replay
from autotuner.trace.serialize import nodes_to_json
from autotuner_runtime.kernels import KernelSpec
from tests import cheats
from tests.conftest import require_healthy_gpu

FIXTURES = Path(__file__).parent / "fixtures"
FP32_TOL = (1e-5, 1e-6)
FP16_TOL = (1e-2, 2e-2)
TIMEOUT_S = 300.0
VALIDATE_GATES = ["static", "compile", "poison", "watchdog",
                  "smoke", "workloads", "sweep", "determinism"]


@pytest.fixture(scope="module")
def tr():
    tracer = Tracer()
    tracer.install()
    yield tracer
    tracer.uninstall()
    assert tracer.verify_restored() == []


@dataclass
class Ctx:
    """One toy region, ready for LadderJob assembly."""

    nodes_json: str
    input_ids: tuple
    output_ids: tuple
    eval_sets: list
    contract: dict


def _full_span(trace):
    stretches = build_stretches(trace, "w")
    span = next(s for s in stretches
                if (s.start_seq, s.end_seq) == (0, len(trace.nodes) - 1))
    return span


def _t_library_ms(nodes, binds, out_ids):
    def one():
        res = replay(nodes, binds, out_ids)
        return [res[i] for i in out_ids]

    mx.eval(one())  # warm
    return statistics.median(time_once(one) for _ in range(3)) * 1e3


def build_ctx(tr, store, fp, model, inputs_per_set, input_names, output_names):
    """Trace the model, take the full span as the region, capture and save k
    boundary sets, and derive the gate-1 contract from the recorded specs."""
    trace, _ = tr.trace(model, [inputs_per_set[0]])
    span = _full_span(trace)
    in_paths, ref_paths = [], []
    ids = set(span.input_ids) | set(span.output_ids)
    for j, x in enumerate(inputs_per_set):
        arrays = capture_boundaries(tr, model, [x], trace, ids)
        in_paths.append(str(store.save(
            fp, "w", j, "inputs", {a: arrays[a] for a in span.input_ids})))
        ref_paths.append(str(store.save(
            fp, "w", j, "outputs", {a: arrays[a] for a in span.output_ids})))
    nodes = trace.nodes[span.start_seq:span.end_seq + 1]
    binds0 = load_set(in_paths[0])
    specs = trace.span_specs(span.start_seq, span.end_seq)
    contract = dict(
        input_names=tuple(input_names),
        input_ranks=tuple(len(specs[a][0]) for a in span.input_ids),
        input_dtypes=tuple(specs[a][1] for a in span.input_ids),
        output_names=tuple(output_names),
        output_ranks=tuple(len(specs[a][0]) for a in span.output_ids),
        output_dtypes=tuple(specs[a][1] for a in span.output_ids),
        live_outputs=tuple(output_names),
    )
    primary = EvalSet("primary", in_paths, ref_paths,
                      t_library_ms=_t_library_ms(nodes, binds0, span.output_ids))
    return Ctx(nodes_to_json(nodes), span.input_ids, span.output_ids,
               [primary], contract), trace, span


@pytest.fixture(scope="module")
def toy(tr, tmp_path_factory):
    """The elementwise toy at (64, 8), three input sets, plus sweep instances
    at L = 1, 13, 4096 with their own spans and on-demand captures."""
    store = BoundaryStore(tmp_path_factory.mktemp("toy_store"))
    ctx, _, _ = build_ctx(
        tr, store, "toy", cheats.toy_model,
        [cheats.toy_input(300 + j) for j in range(3)], ("x",), ("y",))
    for L in (1, 13, 4096):
        x = cheats.toy_input(500 + L, L=L)
        trace, _ = tr.trace(cheats.toy_model, [x])
        span = _full_span(trace)
        ids = set(span.input_ids) | set(span.output_ids)
        arrays = capture_boundaries(tr, cheats.toy_model, [x], trace, ids)
        ip = store.save(f"toy_L{L}", "w", 0, "inputs",
                        {a: arrays[a] for a in span.input_ids})
        rp = store.save(f"toy_L{L}", "w", 0, "outputs",
                        {a: arrays[a] for a in span.output_ids})
        ctx.eval_sets.append(EvalSet(
            f"sweep@L={L}", [str(ip)], [str(rp)], correctness_only=True,
            nodes_json=nodes_to_json(trace.nodes[span.start_seq:span.end_seq + 1])))
    return ctx


@pytest.fixture(scope="module")
def red16(tr, tmp_path_factory):
    store = BoundaryStore(tmp_path_factory.mktemp("red16_store"))
    ctx, _, _ = build_ctx(
        tr, store, "red16", cheats.reduction_model,
        [cheats.reduction_input_f16(310 + j) for j in range(3)], ("x",), ("y",))
    return ctx


def overflow_model(x):
    return x * x * 20.0  # fp16: finite at ten times the data, infinite at a hundred


def overflow_input(key: int, L: int = 64) -> mx.array:
    return mx.random.uniform(low=0.5, high=1.5, shape=(L, 8),
                             key=mx.random.key(key)).astype(mx.float16)


@pytest.fixture(scope="module")
def overflow16(tr, tmp_path_factory):
    store = BoundaryStore(tmp_path_factory.mktemp("overflow_store"))
    ctx, _, _ = build_ctx(
        tr, store, "ovf", overflow_model,
        [overflow_input(700 + j) for j in range(3)], ("x",), ("y",))
    return ctx


@pytest.fixture(scope="module")
def red32(tr, tmp_path_factory):
    store = BoundaryStore(tmp_path_factory.mktemp("red32_store"))
    ctx, _, _ = build_ctx(
        tr, store, "red32", cheats.reduction_model,
        [cheats.reduction_input_f32(320 + j) for j in range(3)], ("x",), ("y",))
    return ctx


def make_job(ctx, spec, *, tag="preserving", tol=FP32_TOL, contract_over=None,
             run_clock=False, min_win_ms=0.01, changing_floor=None, baseline="plain"):
    contract = dict(ctx.contract)
    contract.update(contract_over or {})
    return LadderJob(
        baseline=baseline,
        kernel=spec,
        contract=RegionContract(**contract),
        assoc_tag=tag,
        nodes_json=ctx.nodes_json,
        input_ids=ctx.input_ids,
        output_ids=ctx.output_ids,
        eval_sets=list(ctx.eval_sets),
        tolerances=tol,
        changing_floor=changing_floor,
        min_win_ms=min_win_ms,
        run_clock=run_clock,
        clock_pairs=8,
        timeout_s=TIMEOUT_S,
    )


def test_a_library_replay_starter_passes_the_layout_variant(tr, tmp_path_factory):
    """MLX's own sum gives different bits for the same values laid out
    transposed in memory, so a starter that replays the library's ops cannot
    match the contiguous reference on the sweep's transposed variant; it is
    compared with the library on that same layout. Held to the contiguous
    reference, the fallback starter of every rolled-back reduction scaffold
    died here and closed its region (MetalBench rms_norm_linear, 2026-09-16)."""
    from autotuner.scaffold.native import reference_sequence_seed
    x = cheats.reduction_input_f32(320)
    laid_out = mx.transpose(mx.contiguous(mx.transpose(x)))
    assert mx.array_equal(x, laid_out).item()
    assert not mx.array_equal(cheats.reduction_model(x), cheats.reduction_model(laid_out)).item()  # the premise

    store = BoundaryStore(tmp_path_factory.mktemp("red32_replay_store"))
    ctx, trace, span = build_ctx(
        tr, store, "red32replay", cheats.reduction_model,
        [cheats.reduction_input_f32(320 + j) for j in range(3)], ("in0",), ("out0",))
    seed = reference_sequence_seed(trace, span)
    r = run_ladder(make_job(ctx, seed, contract_over={"input_signature": seed.input_signature}))
    assert r.failed_gate is None and "sweep" in r.gates_passed, r


# -- the cheat zoo: every cheat dies at its intended gate ---------------------
# Toy shaders fuse floating evaluation (including division) and are not
# bit-identical to MLX. Mark changing where a test needs to reach later gates;
# keep the same default tolerances, so the planted shape/value bugs still fail.


def test_partial_write_dies_at_poison(toy):
    spec, _ = cheats.partial_write()
    r = run_ladder(make_job(toy, spec))
    assert (r.outcome, r.failed_gate) == ("failed", "poison")
    assert r.gates_passed == ["static", "compile"]
    assert r.detail["non_finite_over_finite_ref"]["y"] == cheats.TOY_L * cheats.TOY_D // 2


def test_shape_hardcoded_dies_at_sweep(toy):
    spec, _ = cheats.shape_hardcoded()
    r = run_ladder(make_job(toy, spec, tag="changing"))
    assert (r.outcome, r.failed_gate) == ("failed", "sweep")
    # correct at the primary size: smoke and the k sets all passed first
    assert r.gates_passed == VALIDATE_GATES[:-2]
    assert r.detail["eval_set"] == "sweep@L=4096"
    assert "fallback_declared_but_dead" not in r.detail  # no predicate declared


def test_stride_lying_dies_at_sweep_transposed_variant(toy):
    spec, _ = cheats.stride_lying()
    r = run_ladder(make_job(toy, spec, tag="changing"))
    assert (r.outcome, r.failed_gate) == ("failed", "sweep")
    assert r.detail["kind"] == "transposed_variant"


def test_cached_by_shape_dies_at_smoke_k_set_variation(toy):
    refs = load_set(toy.eval_sets[0].reference_paths[0])
    spec, _ = cheats.cached_by_shape(refs[toy.output_ids[0]])
    r = run_ladder(make_job(toy, spec))
    assert (r.outcome, r.failed_gate) == ("failed", "smoke")
    assert r.detail["kind"] == "stored_ref"
    assert r.detail["input_set"] in (1, 2)  # exact on set 0, dead on variation


def test_eps_dropping_dies_at_smoke_tiny_scale_regime(toy):
    spec, _ = cheats.eps_dropping()
    r = run_ladder(make_job(toy, spec, tag="changing"))
    assert (r.outcome, r.failed_gate) == ("failed", "smoke")
    assert r.detail["kind"] == "regime"
    assert r.detail["regime"] == "scaled_down"


def test_fallback_declared_but_dead_dies_at_sweep(toy):
    spec, _ = cheats.fallback_declared_but_dead()
    r = run_ladder(make_job(toy, spec, tag="changing", contract_over={"requires_fallback": True}))
    assert (r.outcome, r.failed_gate) == ("failed", "sweep")
    assert r.detail["fallback_declared_but_dead"] is True
    assert not any(r.detail["fallback_engaged"].values())  # the log shows it never ran


def test_live_output_dropping_dies_at_static(toy):
    spec, _ = cheats.live_output_dropping()
    over = dict(output_names=("y", "z"), output_ranks=(2, 2),
                output_dtypes=("float32", "float32"), live_outputs=("y", "z"))
    r = run_ladder(make_job(toy, spec, contract_over=over))
    assert (r.outcome, r.failed_gate) == ("failed", "static")
    assert r.gates_passed == []
    assert "live_output_dropped" in {f["check"] for f in r.detail["failures"]}


def test_fallback_missing_dies_at_static(toy):
    spec, _ = cheats.fallback_missing()
    r = run_ladder(make_job(toy, spec, contract_over={"requires_fallback": True}))
    assert (r.outcome, r.failed_gate) == ("failed", "static")
    assert "fallback_missing" in {f["check"] for f in r.detail["failures"]}


def test_fallback_on_secondary_workload_returns_failure_instead_of_crashing(toy):
    from dataclasses import replace
    spec, _ = cheats.stride_lying()
    spec = replace(spec, fallback_predicate="in0.shape[0] != 64")
    job = make_job(toy, spec, tag="changing")
    # This shape is a required optimization workload, rather than a fallback sweep.
    job.eval_sets = [job.eval_sets[0], replace(job.eval_sets[1], correctness_only=False)]
    result = run_ladder(job)
    assert (result.outcome, result.failed_gate) == ("failed", "static"), result
    assert "fallback_on_workload" in result.detail["reason"]


@pytest.mark.xfail(strict=True, reason="known hole since 2026-09-17: the scaled-up regime was the only gate "
                   "that caught a half-precision accumulator, and it was removed because its fixed atol "
                   "at 1000x failed honest reordered sums; the outlier regime does not catch this cheat")
def test_fp16_sloppy_accumulation_dies_at_smoke_regimes(red16):
    from dataclasses import replace
    spec, _ = cheats.fp16_sloppy_accumulation()
    # Keep the intended half-accumulation error, but distribute the work so
    # this numeric regression does not depend on a serial shader's watchdog ratio.
    spec = replace(spec, source="""uint lane = thread_position_in_grid.x % 32;
uint r = thread_position_in_grid.x / 32;
uint D = (uint)x_shape[1];
half acc = half(0.0);
for (uint j = lane; j < D; j += 32) { acc += x[r * D + j]; }
float total = metal::simd_sum((float)acc);
if (lane == 0) { y[r] = (half)total; }
""", grid=("in0.shape[0] * 32", "1", "1"), threadgroup=("32", "1", "1"))
    r = run_ladder(make_job(red16, spec, tag="changing", tol=FP16_TOL))
    assert (r.outcome, r.failed_gate) == ("failed", "smoke"), r
    assert r.detail["kind"] == "regime"
    assert r.detail["regime"] == "outliers"


def test_atomic_racy_dies_at_determinism(red32):
    spec, _ = cheats.atomic_racy()
    r = run_ladder(make_job(red32, spec, tag="changing", changing_floor=1e-4))
    assert (r.outcome, r.failed_gate) == ("failed", "determinism")
    assert r.detail["nondeterministic_output"] == "y"
    assert r.gates_passed == VALIDATE_GATES[:-1]  # everything up through sweep


# the library rounds x*x to fp16 before the scale, so the kernel does too
OVF_CORRECT = """uint i = thread_position_in_grid.x;
float v = (float)x[i];
float sq = (float)(half)(v * v);
y[i] = (half)(sq * 20.0f);
"""


def test_stress_magnitude_steps_down_until_the_library_stays_finite(overflow16):
    """Injected outliers overflow this fp16 region's own library output at
    every magnitude tried, and no kernel can be written to match an
    overflowed reference. The regime steps down and, when the library never
    survives, is skipped instead of failing a correct kernel (five regions
    died this way in the 2254 run)."""
    spec = KernelSpec(
        kernel_id="ovf_ok", name="ovf_ok", input_names=("x",), output_names=("y",),
        source=OVF_CORRECT, grid=("in0.shape[0] * in0.shape[1]", "1", "1"),
        threadgroup=("32", "1", "1"), output_shapes=(("in0.shape[0]", "in0.shape[1]"),),
        output_dtypes=("float16",),
    )
    r = run_ladder(make_job(overflow16, spec, tol=FP16_TOL))
    assert r.outcome == "correct_slower" and r.failed_gate is None, r.detail
    assert "outliers" in r.detail["regime_skipped"]


# -- the tag is a claim: reordering passes only the changing gate -------------


def test_reordered_reduction_fails_the_preserving_gate(red16):
    spec, _ = cheats.fp16_fp32_accumulation()
    r = run_ladder(make_job(red16, spec, tag="preserving", tol=FP16_TOL))
    assert (r.outcome, r.failed_gate) == ("failed", "smoke")
    assert r.detail["kind"] == "stored_ref"
    assert r.detail["reason"] == "exact"


@pytest.mark.parametrize("tol, passes", [((1e-6, 1e-6), False), (FP16_TOL, True), ((0.02, 0.2), True)])
def test_reordered_reduction_respects_the_configured_tolerance(red16, tol, passes):
    # An fp32 accumulator over an fp16 sum differs from MLX by its rounding.
    # User policy decides the acceptable deviation: a tolerance tighter than
    # that rounding fails it on the recorded values, the default passes it.
    spec, _ = cheats.fp16_fp32_accumulation()
    r = run_ladder(make_job(red16, spec, tag="changing", tol=tol))
    if not passes:
        assert (r.outcome, r.failed_gate) == ("failed", "smoke")
        assert r.detail["reason"] == "tolerance"
    else:
        assert r.outcome == "correct_slower" and r.failed_gate is None, r.detail
        assert r.gates_passed == VALIDATE_GATES
        assert r.detail["correctness_rule"] == "tolerance"
        assert "changing_floor" not in r.detail


# -- positive controls on the planted win -------------------------------------

PW_HEADER = """inline float mao_max(float a, float b) { return metal::isnan(a) ? a : metal::fmax(a, b); }
inline float mao_min(float a, float b) { return metal::isnan(a) ? a : metal::fmin(a, b); }
"""

# mirrors the fixture's op chain exactly, including the library's NaN
# propagation through maximum/minimum (metal fmin/fmax drop NaN, mlx keeps it)
PW_FUSED = """uint i = thread_position_in_grid.x;
float xv = x[i];
float v = xv * 2.0f;
v = v + a[i];
v = mao_max(v, 0.0f);
v = v * xv;
v = v + b[i];
v = mao_min(v, 8.0f);
v = v - 1.0f;
y[i] = v * 0.5f;
"""

# the same math plus a data-dependent busy loop the compiler cannot elide and
# a guard that can never fire (sin(w) + 1 is in [0, 2], NaN compares false)
PW_SLOW = """uint i = thread_position_in_grid.x;
float xv = x[i];
float v = xv * 2.0f;
v = v + a[i];
v = mao_max(v, 0.0f);
v = v * xv;
v = v + b[i];
v = mao_min(v, 8.0f);
v = v - 1.0f;
float w = xv;
for (uint j = 0; j < 20000u; ++j) { w = metal::sin(w) + 1.0f; }
if (w > 1.0e30f) { v = w; }
y[i] = v * 0.5f;
"""


@pytest.fixture(scope="module")
def pw(tr, tmp_path_factory, request):
    spec = importlib.util.spec_from_file_location("fixture_lg_pw", FIXTURES / "planted_win.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.build()
    size = getattr(request, "param", 1024)
    if size != 1024:
        model.chain.a = mx.random.normal((size,), key=mx.random.key(98))
        model.chain.b = mx.random.normal((size,), key=mx.random.key(99))
    store = BoundaryStore(tmp_path_factory.mktemp("pw_store"))
    x0 = mx.random.normal((size,), key=mx.random.key(100))
    trace, _ = tr.trace(model, [x0])
    span = _full_span(trace)
    # kernel input order is the span's input_ids order; name each id's role
    # (weight paths like "chain.a" map to their leaf attribute name)
    names = tuple("x" if aid in trace.inputs else trace.weight_paths[aid].split(".")[-1]
                  for aid in span.input_ids)
    assert set(names) == {"x", "a", "b"}
    xs = [x0] + [mx.random.normal((size,), key=mx.random.key(101 + j)) for j in range(2)]
    ctx, _, _ = build_ctx(tr, store, "pw", model, xs, names, ("y",))
    return ctx


def pw_spec(ctx, name, source):
    return KernelSpec(
        kernel_id=name, name=name,
        input_names=ctx.contract["input_names"],
        output_names=("y",),
        source=source, header=PW_HEADER,
        grid=("in0.shape[0]", "1", "1"),
        threadgroup=("32", "1", "1"),
        output_shapes=(("in0.shape[0]",),),
        output_dtypes=("float32",),
    )


# A material memory-traffic saving, independent of Python replay overhead.
@pytest.mark.parametrize("pw", [262144], indirect=True)
def test_planted_win_ships(pw):
    require_healthy_gpu()
    job = make_job(pw, pw_spec(pw, "ctrl_pw_fused", PW_FUSED),
                   run_clock=True, min_win_ms=0.005)
    r = run_ladder(job)
    assert r.outcome == "tentative_ship" and r.failed_gate is None, asdict(r)
    assert r.gates_passed == VALIDATE_GATES + ["clock"]
    assert r.region_ms > 0 and r.library_ms > 0 and r.sigma_ms >= 0
    assert r.win_ms >= 0.005
    assert r.region_ms < r.library_ms


def test_correct_but_slower_climbs_with_honest_numbers(pw):
    require_healthy_gpu()
    job = make_job(pw, pw_spec(pw, "ctrl_pw_slow", PW_SLOW),
                   run_clock=True, min_win_ms=0.005)
    r = run_ladder(job)
    assert r.outcome == "correct_slower" and r.failed_gate is None
    assert "clock" in r.gates_passed
    assert r.win_ms < 0  # honestly slower, not vetoed
    assert r.region_ms > r.library_ms


def test_scaffold_run_stops_before_the_clock(pw):
    job = make_job(pw, pw_spec(pw, "ctrl_pw_scaffold", PW_FUSED), run_clock=False)
    r = run_ladder(job)
    assert r.outcome == "correct_slower" and r.failed_gate is None
    assert r.gates_passed == VALIDATE_GATES
    assert r.region_ms is None and r.win_ms is None


@pytest.mark.parametrize("bad_output", [False, True])
def test_reordered_reduction_with_scratch(red16, bad_output):
    """Scratch is not a region output, nor permission to skip a real one."""
    from dataclasses import replace

    spec, _ = cheats.fp16_fp32_accumulation()
    spec = replace(
        spec,
        kernel_id=f"scratch_reduction_{bad_output}",
        name=f"scratch_reduction_{bad_output}",
        output_names=spec.output_names + ("tmp0",),
        output_shapes=spec.output_shapes + (("1",),),
        output_dtypes=spec.output_dtypes + ("bfloat16",),
        # Deliberately unused scratch remains poisoned. Real output still matters.
        source=spec.source + ("\ny[thread_position_in_grid.x] = half(0);" if bad_output else ""),
    )
    result = run_ladder(make_job(red16, spec, tag="changing", tol=(0.02, 0.2)))
    if bad_output:
        assert (result.outcome, result.failed_gate) == ("failed", "smoke"), result
    else:
        assert result.outcome == "correct_slower" and result.failed_gate is None, result.detail
        assert result.gates_passed == VALIDATE_GATES
        # bf16 scratch must not relax the fp16 numeric tolerance.
        original, _ = cheats.fp16_fp32_accumulation()
        control = run_ladder(make_job(red16, original, tag="changing", tol=(0.02, 0.2)))
        assert result.detail["tolerances"] == control.detail["tolerances"]
