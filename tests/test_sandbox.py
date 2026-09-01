"""M5: the subprocess sandbox. The harness's job is rejecting bad kernels, so
these tests are bad kernels: a hang, a compile error, an OOB read, a partial
write, and a too-slow kernel, each asserting WHICH structured verdict comes
back and that the parent process is never harmed. Job specs are built inline
with tiny kernels and tensors saved to tmp_path via mx.save_safetensors.
"""

import dataclasses
import json

import mlx.core as mx
import pytest

from autotuner.sandbox.poison import saturate_pool
from autotuner.sandbox.protocol import (
    GATES,
    JobSpec,
    TensorSet,
    Verdict,
    mode_env,
    run_job,
)
from autotuner_runtime.kernels import KernelSpec

N = 1024
TOLERANCES = {"rtol": 1e-5, "atol": 1e-6}
GENEROUS_MS = 50.0  # watchdog limit 500ms: no tiny test kernel gets near it

ADD = "uint i = thread_position_in_grid.x;\nout[i] = a[i] + b[i];"

# body line 2 is the broken one
COMPILE_ERR = "uint i = thread_position_in_grid.x;\nthis is not metal;\nout[i] = a[i] + b[i];"

# volatile device accesses so the compiler cannot remove the infinite loop
# (a side-effect-free one is eliminated and returns instantly, measured)
HANG = """
uint i = thread_position_in_grid.x;
device volatile float* av = (device volatile float*)a;
device volatile float* ov = (device volatile float*)out;
while (av[0] > -1.0e30f) { ov[i] = ov[i] + 1.0f; }
"""

# reads far past a's 4KB buffer: zerofilled under validation, wrong either way
OOB = "uint i = thread_position_in_grid.x;\nout[i] = a[i + 65536];"

# writes only the first half; init_value=nan makes the rest NaN deterministically
PARTIAL = "uint i = thread_position_in_grid.x;\nif (i < 512u) out[i] = a[i] + b[i];"

# a dependent chain of 50k sins per thread: ~1.6ms measured, correct but slow
SLOW = """
uint i = thread_position_in_grid.x;
float acc = a[i];
for (uint j = 0; j < 50000u; ++j) { acc = metal::sin(acc) + 1.0f; }
out[i] = acc + b[i];
"""


def make_job(tmp_path, *, source, name, t_library_ms=GENEROUS_MS, gates=GATES):
    """A two-input elementwise job whose library reference is a + b."""
    a = mx.random.normal((N,)).astype(mx.float32)
    b = mx.random.normal((N,)).astype(mx.float32)
    ref = a + b
    mx.eval(a, b, ref)
    in_path = str(tmp_path / f"{name}_in.safetensors")
    ref_path = str(tmp_path / f"{name}_ref.safetensors")
    mx.save_safetensors(in_path, {"a": a, "b": b})
    mx.save_safetensors(ref_path, {"out": ref})
    kspec = KernelSpec(
        kernel_id=name,
        name=name,
        input_names=("a", "b"),
        output_names=("out",),
        source=source,
        grid=("in0.shape[0]", "1", "1"),
        threadgroup=("32", "1", "1"),
        output_shapes=(("in0.shape[0]",),),
        output_dtypes=("float32",),
    )
    return JobSpec(
        kernel=json.loads(kspec.to_json()),
        inputs=TensorSet(in_path, ("a", "b")),
        reference=TensorSet(ref_path, ("out",)),
        t_library_ms=t_library_ms,
        tolerances=TOLERANCES,
        gates=tuple(gates),
    )


def test_correct_kernel_passes_and_reports_timing(tmp_path):
    v = run_job(make_job(tmp_path, source=ADD, name="sbx_add"), "score", timeout_s=60)
    assert v.passed and v.failed_gate is None
    assert v.gates_passed == GATES
    assert v.detail == {}
    assert v.timing["first_run_ms"] > 0
    assert v.timing["timed_run_ms"] > 0
    assert v.timing["t_library_ms"] == GENEROUS_MS


def test_correct_kernel_in_validate_mode_has_no_validation_detail(tmp_path):
    """An in-bounds kernel must not be flagged by shader validation."""
    v = run_job(make_job(tmp_path, source=ADD, name="sbx_add_v"), "validate", timeout_s=60)
    assert v.passed
    assert "validation" not in v.detail


def test_compile_error_reports_structured_diagnostics(tmp_path):
    v = run_job(make_job(tmp_path, source=COMPILE_ERR, name="sbx_broken"), "validate", timeout_s=60)
    assert not v.passed and v.failed_gate == "compile"
    assert v.gates_passed == ()
    assert not v.detail["text"].startswith("[metal::Device]")  # preamble stripped
    # the offset counts utils.h plus the generated signature (spike_06: 472+)
    assert v.detail["line_offset"] is not None and v.detail["line_offset"] > 400
    errors = [d for d in v.detail["diagnostics"] if d["severity"] == "error"]
    assert errors and errors[0]["body_line"] == 2


def test_hanging_kernel_maps_to_subprocess_and_parent_survives(tmp_path):
    """An infinite loop blocks mx.eval forever (verified: no OS-side error);
    the parent wall timeout is the hang half of the watchdog, and killing the
    child is routine recovery, not an error path."""
    v = run_job(make_job(tmp_path, source=HANG, name="sbx_hang"), "score", timeout_s=10)
    assert not v.passed and v.failed_gate == "subprocess"
    assert "timeout" in v.detail["reason"]
    # the parent's own GPU context still works
    x = mx.ones((64, 64)) @ mx.ones((64, 64))
    mx.eval(x)
    assert x[0, 0].item() == 64.0
    # and the next evaluation in a fresh child passes
    good = make_job(tmp_path, source=ADD, name="sbx_after_hang")
    assert run_job(good, "score", timeout_s=60).passed


def test_oob_kernel_fails_allclose_with_validation_evidence(tmp_path):
    """OOB reads zerofill silently (spike_07): value comparison catches the
    wrong outputs, and validate mode attaches the Invalid device load lines."""
    v = run_job(make_job(tmp_path, source=OOB, name="sbx_oob"), "validate", timeout_s=60)
    assert not v.passed and v.failed_gate == "allclose"
    assert "poison" in v.gates_passed  # zerofilled reads are finite
    assert v.detail["validation"]["counts"]["Invalid device load"] >= 1
    assert not v.detail["outputs"]["out"]["allclose"]


def test_oob_kernel_in_score_mode_has_no_validation_detail(tmp_path):
    v = run_job(make_job(tmp_path, source=OOB, name="sbx_oob_s"), "score", timeout_s=60)
    assert v.failed_gate == "allclose"
    assert "validation" not in v.detail


def test_partial_write_fails_poison_deterministically(tmp_path):
    v = run_job(make_job(tmp_path, source=PARTIAL, name="sbx_partial"), "score", timeout_s=60)
    assert not v.passed and v.failed_gate == "poison"
    assert v.gates_passed == ("compile", "watchdog")
    assert v.detail["non_finite_over_finite_ref"]["out"] == N // 2


def test_same_spec_twice_gives_same_verdict(tmp_path):
    job = make_job(tmp_path, source=PARTIAL, name="sbx_det")
    v1 = run_job(job, "score", timeout_s=60)
    v2 = run_job(job, "score", timeout_s=60)
    assert v1.failed_gate == "poison"  # a real gate failure, not two crashes
    assert (v1.passed, v1.failed_gate, v1.gates_passed, v1.detail) == (
        v2.passed, v2.failed_gate, v2.gates_passed, v2.detail)


def test_slow_kernel_fails_watchdog(tmp_path):
    job = make_job(tmp_path, source=SLOW, name="sbx_slow", t_library_ms=0.05)
    v = run_job(job, "score", timeout_s=60)
    assert not v.passed and v.failed_gate == "watchdog"
    assert v.gates_passed == ("compile",)
    assert v.detail["watchdog_factor"] == 20.0
    assert v.detail["timed_run_ms"] > 20.0 * 0.05


def test_missing_tensor_file_maps_to_subprocess(tmp_path):
    job = make_job(tmp_path, source=ADD, name="sbx_missing")
    job = dataclasses.replace(
        job, inputs=TensorSet(str(tmp_path / "absent.safetensors"), ("a", "b")))
    v = run_job(job, "score", timeout_s=60)
    assert not v.passed and v.failed_gate == "subprocess"
    assert "child exit" in v.detail["reason"]
    assert v.detail["stderr_tail"]  # the child's traceback tail is preserved


def test_gate_subset_runs_compile_only(tmp_path):
    """The parent chooses the gates; a partial-write kernel passes when only
    the compile probe is requested."""
    job = make_job(tmp_path, source=PARTIAL, name="sbx_subset", gates=("compile",))
    v = run_job(job, "score", timeout_s=60)
    assert v.passed and v.gates_passed == ("compile",)


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


def test_job_spec_and_verdict_round_trip(tmp_path):
    job = make_job(tmp_path, source=ADD, name="sbx_rt")
    assert JobSpec.from_json(job.to_json()) == job
    v = Verdict(False, "poison", ("compile", "watchdog"),
                {"non_finite_over_finite_ref": {"out": 3}}, {"t_library_ms": 1.0})
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
