"""The subprocess sandbox. The harness's job is rejecting bad kernels, so
these tests are bad kernels: a hang, a compile error, an out-of-bounds read,
a partial write, and a too-slow kernel, each asserting WHICH structured
verdict comes back and that the parent process is never harmed. Every spec
is the real ladder spec over one recorded region, a + b."""

import dataclasses
import json

import mlx.core as mx
import pytest

from autotuner.sandbox.poison import saturate_pool
from autotuner.sandbox.protocol import EvalSetSpec, LadderSpec, Verdict, mode_env, run_job
from autotuner.trace import Tracer
from autotuner.trace.serialize import nodes_to_json
from autotuner_runtime.kernels import KernelSpec

N = 1024

ADD = "uint i = thread_position_in_grid.x;\nout0[i] = in0[i] + in1[i];"

# body line 2 is the broken one
COMPILE_ERR = "uint i = thread_position_in_grid.x;\nthis is not metal;\nout0[i] = in0[i] + in1[i];"

# volatile device accesses so the compiler cannot remove the infinite loop
# (a side-effect-free one is eliminated and returns instantly, measured)
HANG = """
uint i = thread_position_in_grid.x;
device volatile float* av = (device volatile float*)in0;
device volatile float* ov = (device volatile float*)out0;
while (av[0] > -1.0e30f) { ov[i] = ov[i] + 1.0f; }
"""

# reads far past in0's 4KB buffer: zerofilled under validation, wrong either way
OOB = "uint i = thread_position_in_grid.x;\nout0[i] = in0[i + 65536];"

# writes only the first half; init_value=nan makes the rest NaN deterministically
PARTIAL = "uint i = thread_position_in_grid.x;\nif (i < 512u) out0[i] = in0[i] + in1[i];"

# a dependent chain of five million sins per thread: about a tenth of a second
# per pass, far past twenty times the library's add even when another process
# is holding the GPU and every pass pays milliseconds of queueing
SLOW = """
uint i = thread_position_in_grid.x;
float acc = in0[i];
for (uint j = 0; j < 5000000u; ++j) { acc = metal::sin(acc) + 1.0f; }
out0[i] = acc + in1[i];
"""


def add_model(a, b):
    return a + b


@pytest.fixture(scope="module")
def region(tmp_path_factory):
    """The a + b span as the tracer records it, with one saved input set and
    the library's outputs for it."""
    tracer = Tracer()
    tracer.install()
    try:
        a = mx.random.normal((N,), key=mx.random.key(1))
        b = mx.random.normal((N,), key=mx.random.key(2))
        mx.eval(a, b)
        trace, _ = tracer.trace(add_model, [a, b])
    finally:
        tracer.uninstall()
        assert tracer.verify_restored() == []
    (node,) = trace.nodes
    ref = a + b
    mx.eval(ref)
    d = tmp_path_factory.mktemp("sandbox")
    mx.save_safetensors(str(d / "in.safetensors"),
                        {f"a{node.in_arrays[0]}": a, f"a{node.in_arrays[1]}": b})
    mx.save_safetensors(str(d / "ref.safetensors"), {f"a{node.out_arrays[0]}": ref})
    return dict(nodes_json=nodes_to_json([node]), input_ids=tuple(node.in_arrays),
                output_ids=tuple(node.out_arrays), inputs=str(d / "in.safetensors"),
                refs=str(d / "ref.safetensors"))


def spec(region, source, name, phase="validate"):
    kernel = KernelSpec(
        kernel_id=name, name=name, input_names=("in0", "in1"), output_names=("out0",),
        source=source, grid=("in0.shape[0]", "1", "1"), threadgroup=("32", "1", "1"),
        output_shapes=(("in0.shape[0]",),), output_dtypes=("float32",),
    )
    return LadderSpec(
        kernel=json.loads(kernel.to_json()), assoc_tag="preserving",
        nodes_json=region["nodes_json"], input_ids=region["input_ids"],
        output_ids=region["output_ids"],
        eval_sets=(EvalSetSpec("primary", (region["inputs"],), (region["refs"],),
                               t_library_ms=0.05, correctness_only=False, nodes_json=None),),
        tolerances={"rtol": 1e-5, "atol": 1e-6}, kappa=1.25, changing_floor=None,
        min_win_ms=0.01, phase=phase, clock_pairs=4,
    )


def test_correct_kernel_passes_validation_with_no_validation_detail(region):
    v = run_job(spec(region, ADD, "sbx_add"), "validate", timeout_s=60)
    assert v.passed and v.failed_gate is None, v.detail
    assert "validation" not in v.detail  # an in-bounds kernel is never flagged
    assert v.gates_passed[0] == "compile" and "determinism" in v.gates_passed


def test_correct_kernel_scores_with_the_clock(region):
    v = run_job(spec(region, ADD, "sbx_add_s", phase="score"), "score", timeout_s=120)
    assert v.passed, v.detail
    assert "ship" in v.detail and v.timing["library_ms"] > 0 and "win_ms" in v.timing


def test_compile_error_reports_structured_diagnostics(region):
    v = run_job(spec(region, COMPILE_ERR, "sbx_broken"), "validate", timeout_s=60)
    assert not v.passed and v.failed_gate == "compile"
    assert v.gates_passed == ()
    assert not v.detail["text"].startswith("[metal::Device]")  # preamble stripped
    # the offset counts the helper header plus the generated signature
    assert v.detail["line_offset"] is not None and v.detail["line_offset"] > 400
    errors = [d for d in v.detail["diagnostics"] if d["severity"] == "error"]
    assert errors and errors[0]["body_line"] == 2


def test_hanging_kernel_maps_to_subprocess_and_parent_survives(region):
    """An infinite loop blocks mx.eval forever; the parent wall timeout is the
    hang half of the watchdog, and killing the child is routine recovery."""
    v = run_job(spec(region, HANG, "sbx_hang"), "score", timeout_s=4)
    assert not v.passed and v.failed_gate == "subprocess"
    assert "timeout" in v.detail["reason"]
    # the parent's own GPU context still works
    x = mx.ones((64, 64)) @ mx.ones((64, 64))
    mx.eval(x)
    assert x[0, 0].item() == 64.0
    # and the next evaluation in a fresh child passes
    assert run_job(spec(region, ADD, "sbx_after_hang"), "validate", timeout_s=60).passed


def test_oob_read_fails_numerics_with_validation_evidence(region):
    """OOB reads zerofill silently: the value comparison catches the wrong
    outputs, and validate mode attaches the Invalid device load lines."""
    v = run_job(spec(region, OOB, "sbx_oob"), "validate", timeout_s=60)
    assert not v.passed and v.failed_gate == "smoke"
    assert "poison" in v.gates_passed  # zerofilled reads are finite
    assert v.detail["validation"]["counts"]["Invalid device load"] >= 1


def test_partial_write_fails_poison_deterministically(region):
    v = run_job(spec(region, PARTIAL, "sbx_partial"), "validate", timeout_s=60)
    assert not v.passed and v.failed_gate == "poison"
    assert v.gates_passed == ("compile",)
    assert v.detail["non_finite_over_finite_ref"]["out0"] == N // 2


def test_slow_kernel_fails_watchdog(region):
    v = run_job(spec(region, SLOW, "sbx_slow"), "validate", timeout_s=60)
    assert not v.passed and v.failed_gate == "watchdog"
    assert v.gates_passed == ("compile", "poison")
    assert v.detail["watchdog_factor"] == 20.0
    assert v.detail["timed_run_ms"] > 20.0 * v.detail["library_run_ms"]


def test_missing_tensor_file_maps_to_subprocess(region):
    absent = dataclasses.replace(
        spec(region, ADD, "sbx_missing"),
        eval_sets=(EvalSetSpec("primary", ("/nonexistent/absent.safetensors",), (region["refs"],),
                               t_library_ms=0.05, correctness_only=False, nodes_json=None),))
    v = run_job(absent, "validate", timeout_s=60)
    assert not v.passed and v.failed_gate == "subprocess"
    assert "child exit" in v.detail["reason"]
    assert v.detail["stderr_tail"]  # the child's traceback tail is preserved


def test_mode_env_sets_and_scrubs_metal_vars(monkeypatch):
    """Metal reads these at launch, so the mode is entirely the spawn env, and
    inherited knobs from the parent's environment must never leak through."""
    monkeypatch.setenv("MTL_SHADER_VALIDATION", "1")
    monkeypatch.setenv("MTL_CAPTURE_ENABLED", "1")
    score = mode_env("score")
    assert "MTL_SHADER_VALIDATION" not in score
    assert "MTL_SHADER_VALIDATION_REPORT_TO_STDERR" not in score
    assert "MTL_CAPTURE_ENABLED" not in score
    validate = mode_env("validate")
    assert validate["MTL_SHADER_VALIDATION"] == "1"
    assert validate["MTL_SHADER_VALIDATION_REPORT_TO_STDERR"] == "1"
    assert "MTL_CAPTURE_ENABLED" not in validate
    assert mode_env("capture")["MTL_CAPTURE_ENABLED"] == "1"
    with pytest.raises(ValueError):
        mode_env("fast")


def test_spec_and_verdict_round_trip(region):
    job = spec(region, ADD, "sbx_rt")
    assert LadderSpec.from_json(job.to_json()) == job
    v = Verdict(False, "poison", ("compile",),
                {"non_finite_over_finite_ref": {"out0": 3}}, {"t_library_ms": 1.0})
    assert Verdict.from_json(v.to_json()) == v


def test_saturate_pool_dirties_recycled_buffers():
    """A kernel that writes nothing reads back NaN from the dirtied pool;
    without saturation a stale-buffer bug could read plausible leftovers.
    Uses an unusual buffer size so earlier tests' freed buffers cannot be
    served instead of the NaN ones."""
    n = 333_333
    assert saturate_pool([n * 4]) == 4 * n * 4
    k = mx.fast.metal_kernel(
        name="sbx_nowrite", input_names=["a"], output_names=["out"],
        source="(void)a;",
    )
    out = k(inputs=[mx.ones((4,), dtype=mx.float32)], output_shapes=[(n,)],
            output_dtypes=[mx.float32], grid=(1, 1, 1), threadgroup=(1, 1, 1))
    mx.eval(out)
    assert int(mx.sum(mx.isnan(out[0])).item()) == n
