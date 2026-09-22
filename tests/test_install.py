"""Installing wins on one model, directly on the install path: a second
region after a first, several copies of one region inside one module, a
re-ship on the same span, the final whole-model check, and the artifact
loaded in a fresh process. No pricing and no judge: the ladder result is
handed in."""

import textwrap
from pathlib import Path

import mlx.core as mx
import pytest

from autotuner_runtime.stats import comparison_from_samples

from autotuner.ladder.gates import LadderResult
from autotuner.loop import JobRunner, RegionRun
from autotuner.measure.session import Session
from autotuner_runtime.kernels import KernelSpec

FIXTURES = Path(__file__).parent / "fixtures"
WIN = LadderResult("tentative_ship", None, {}, 1.0, 2.0, 1.0, 0.0, [])


def elementwise(kernel_id: str, body: str, inputs=("in0",)) -> KernelSpec:
    return KernelSpec(
        kernel_id=kernel_id, name=f"at_{kernel_id}", input_names=tuple(inputs),
        output_names=("out0",), source=f"uint i = thread_position_in_grid.x;\n{body}",
        grid=("in0.shape[0] * in0.shape[1]", "1", "1"), threadgroup=("256", "1", "1"),
        output_shapes=(("in0.shape[0]", "in0.shape[1]"),), output_dtypes=("float32",),
    )


def _runner(tmp_path, shape: str, extra: str = "", fixture: str = "planted_win.py",
            baseline: str = "plain"):
    """A plain baseline unless a test says otherwise: these tests are about
    the install path, and their elementwise wins are no wins against
    mx.compile, which fuses such chains itself."""
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(textwrap.dedent(f"""
        model: {FIXTURES / fixture}
        workloads:
          - inputs: [{{shape: {shape}, dtype: float32}}]
            name: main
        baseline: {baseline}
        {extra}
    """))
    r = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: None,
                  clock_pairs=4, session=Session(sleep=lambda s: None))
    r.load_model()
    r.trace_workloads()
    regions = r.build_regions()

    def region(*ops, start=None):
        return next(reg for reg in regions if tuple(reg.ops) == ops
                    and (start is None or reg.members[0].start_seq == start))

    r.region = region
    r.regions = regions
    return r


@pytest.fixture
def model_says_win(monkeypatch):
    """These tests hand in stand-in kernels to exercise install, compose,
    re-ship, and rollback. A lone elementwise kernel is really a little slower
    than the library's op, so force the whole-model win and let the install
    mechanics run; certification and retrace still run for real. Whether a
    kernel is actually faster is the whole-model decision tested in test_e2e.py
    and test_loop.py."""
    monkeypatch.setattr(JobRunner, "_model_win", lambda self, e2e: True)


@pytest.fixture
def runner(tmp_path, model_says_win):
    r = _runner(tmp_path, "[64, 1024]")
    yield r
    r.tracer.uninstall()
    assert r.tracer.verify_restored() == []


@pytest.fixture
def swept(tmp_path, model_says_win):
    """The row count is a named dim, primary 64, swept to 7."""
    r = _runner(tmp_path, "[L, 1024]", "sweep: {L: [7, 64]}\n        primary: {L: 64}")
    yield r
    r.tracer.uninstall()
    assert r.tracer.verify_restored() == []


def test_wins_compose_on_one_model(runner, tmp_path, monkeypatch):
    x = runner.tensors["main"][0]
    before = runner.model(x)
    mx.eval(before)

    # NaN keeps NaN, as mx.maximum and mx.minimum do
    k_max = elementwise("rmax_h1", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);")
    k_min = elementwise("rmin_h1", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::min(in0[i], 8.0f);")
    assert runner._bind_and_promote(RegionRun(region=runner.region("mx.maximum")), k_max, WIN)
    # the second region in the same workload used to fail verification, because
    # the check expected only its own cut against the job-start recording
    assert runner._bind_and_promote(RegionRun(region=runner.region("mx.minimum")), k_min, WIN)
    assert set(runner.cuts["main"].values()) == {"rmax_h1", "rmin_h1"}

    # two copies of the residual add inside one module: one wrapper, two cuts
    add = runner.region("array.__add__")
    assert add.copies == 2 and len({m.scope_stack for m in add.members}) == 1
    k_add = elementwise("radd_h1", "out0[i] = in0[i] + in1[i % (uint)in0_shape[1]];", ("in0", "in1"))
    assert runner._bind_and_promote(RegionRun(region=add), k_add, WIN)
    assert len(runner.cuts["main"]) == 4
    assert set(runner.emitted["chain"].kernel_ids) == {"radd_h1", "rmax_h1", "rmin_h1"}
    assert list(runner.cuts["main"].values()).count("radd_h1") == 2

    # a re-ship on the same span replaces the old kernel everywhere
    k_max2 = elementwise("rmax_h2", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);")
    assert runner._bind_and_promote(RegionRun(region=runner.region("mx.maximum")), k_max2, WIN)
    assert "rmax_h1" not in runner.emitted["chain"].kernel_ids
    assert set(runner.cuts["main"].values()) == {"rmax_h2", "rmin_h1", "radd_h1"}

    after = runner.model(x)
    mx.eval(after)
    assert mx.array_equal(before, after).item()  # elementwise kernels in the library's order

    # three single-op kernels in a 37 us step are a call-site tax, not a win,
    # and the chained step clock resolves that; this test is about composing
    # installs and exporting them, so the final veto is stubbed
    monkeypatch.setattr("autotuner.e2e.step_veto",
                        lambda *args, **kwargs: (comparison_from_samples([1.0] * 4, [0.9] * 4), True))
    runner._final_check()
    assert runner.final_ok and runner.report.final["passed"]
    art = runner.emit_artifact(tmp_path / "artifact")
    assert sorted(p.stem for p in (art / "kernels").glob("*.metal")) == ["radd_h1", "rmax_h2", "rmin_h1"]
    kinds = [row["kind"] for row in runner.log.rows()]
    assert kinds.count("shipped") == 4 and "final_e2e" in kinds and "artifact_checked" in kinds


def test_a_multi_op_cut_feeding_the_next_cut_verifies(runner):
    """A three-op cut whose output feeds the very next op, then that op as its
    own cut: the retrace must show two custom dispatches wired together."""
    x = runner.tensors["main"][0]
    before = runner.model(x)
    mx.eval(before)
    prefix = runner.region("array.__mul__", "array.__add__", "mx.maximum")
    fused = elementwise("rpre_h1", (
        "float y = in0[i] * 2.0f + in1[i % (uint)in0_shape[1]];\n"
        "out0[i] = (y != y) ? y : metal::max(y, 0.0f);"), ("in0", "in1"))
    assert runner._bind_and_promote(RegionRun(region=prefix), fused, WIN)
    times_x = runner.region("array.__mul__", start=prefix.members[0].end_seq + 1)
    k_mul = elementwise("rmul_h1", "out0[i] = in0[i] * in1[i];", ("in0", "in1"))
    assert runner._bind_and_promote(RegionRun(region=times_x), k_mul, WIN)
    after = runner.model(x)
    mx.eval(after)
    assert mx.array_equal(before, after).item()


FUSED = elementwise("rchain_h1", (
    "uint c = i % (uint)in0_shape[1];\n"
    "float x = in0[i];\n"
    "float y = x * 2.0f + in1[c];\n"
    "y = (y != y) ? y : metal::max(y, 0.0f);\n"
    "y = y * x;\n"
    "y = y + in2[c];\n"
    "y = (y != y) ? y : metal::min(y, 8.0f);\n"
    "out0[i] = (y - 1.0f) * 0.5f;"), ("in0", "in1", "in2"))


def _chain(runner):
    return RegionRun(region=runner.region("array.__mul__", "array.__add__", "mx.maximum",
                                          "array.__mul__", "array.__add__", "mx.minimum",
                                          "array.__sub__", "array.__mul__"))


def test_the_sweep_checks_every_kernel_at_the_other_sizes(swept, tmp_path):
    """A named dim is traced and captured at each sweep size, gate 7 checks
    every kernel there, and an installed wrapper hands the unrecorded size back
    to the original module, which the final whole-model check confirms."""
    from autotuner.ladder.gates import run_ladder

    assert list(swept.sweep_traces) == ["main@L=7"]
    assert swept.sweep_tensors["main@L=7"][0].shape == (7, 1024)
    chain = _chain(swept).region
    swept._capture([chain])
    span = swept.sweep_spans[(chain.fingerprint, "main@L=7")]
    assert span.end_seq - span.start_seq == 7  # the same eight ops, located at 7 rows
    assert swept.store.set_count(chain.fingerprint, "main@L=7") == 1

    chain.t_orig_ms["main"] = chain.t_rep_ms["main"] = 1.0  # pricing is not under test
    sets = swept._eval_sets(chain)
    assert [(e.label, e.correctness_only, e.nodes_json is not None) for e in sets] == [
        ("main", False, True), ("main@L=7", True, True)]

    passed = run_ladder(swept._ladder_job(chain, FUSED, "preserving", run_clock=False))
    assert "sweep" in passed.gates_passed, passed
    assert passed.detail["fallback_engaged"] == {"main@L=7": False}
    only_at_64 = elementwise("rchain_h2", FUSED.source.split("\n", 1)[1].replace(
        "float x = in0[i];", "float x = (in0_shape[0] == 64) ? in0[i] : 0.0f;"),
        ("in0", "in1", "in2"))
    caught = run_ladder(swept._ladder_job(chain, only_at_64, "preserving", run_clock=False))
    assert (caught.outcome, caught.failed_gate) == ("failed", "sweep")
    assert caught.detail["eval_set"] == "main@L=7"

    assert swept._bind_and_promote(RegionRun(region=chain), FUSED, WIN)
    x7 = swept.sweep_tensors["main@L=7"]
    assert mx.array_equal(swept.model(*x7), swept.baseline_model(*x7)).item()
    swept.tracer.install()  # the install left the patch surface down
    retrace7, _ = swept.tracer.trace(swept.model, x7)
    assert not any(n.op == "custom_kernel" for n in retrace7.nodes)
    retrace64, _ = swept.tracer.trace(swept.model, swept.tensors["main"])
    assert sum(n.op == "custom_kernel" for n in retrace64.nodes) == 1
    swept._final_check()
    assert [(c["name"], c["passed"]) for c in swept.report.final["checks"]] == [
        ("main", True), ("main@L=7", True)]


def test_a_step_that_keeps_state_gets_the_plain_baseline(tmp_path):
    """A step that writes a buffer it holds, as a KV cache does, cannot be
    compiled from outside the model: the job takes the plain baseline, says
    why, never takes the compiled clock, and leaves the model working."""
    r = _runner(tmp_path, "[4, 16]", fixture="kv_step.py", baseline="compiled")
    try:
        assert r.traces["main"].python_retained()
        x = r.tensors["main"]
        before = r.model(*x)
        mx.eval(before)
        r._clock_steps()
        assert r.baseline == "plain"
        b = r.report.baseline
        assert (b["requested"], b["choice"], b["compiled_available"]) == ("compiled", "plain", False)
        assert "Python state" in b["reason"]
        assert b["clocks_ms"]["main"]["plain"] > 0 and b["clocks_ms"]["main"]["compiled"] is None
        assert r.report.step_ms["main"]["before"] == b["clocks_ms"]["main"]["plain"]
        after = r.model(*x)
        mx.eval(after)
        assert mx.array_equal(before, after).item()  # alive, and the same step every call
        row = next(row for row in r.log.rows() if row["kind"] == "baseline")
        assert row["choice"] == "plain" and row["reason"]
    finally:
        r.tracer.uninstall()
        assert r.tracer.verify_restored() == []


def test_a_step_whose_cache_has_a_method_gets_the_plain_baseline(tmp_path):
    """The same rule through the other door. mlx_lm's KV cache writes inside a
    method, so the write records as one state call and the model keeps no
    recorded array: the arrays-kept count alone would call this step
    compilable, take a compiled clock that returns a plausible number, and
    leave the model dead for every later call."""
    r = _runner(tmp_path, "[4, 16]", fixture="state_call.py", baseline="compiled")
    try:
        assert not r.traces["main"].python_retained()  # the blind spot: nothing kept
        x = r.tensors["main"]
        before = r.model(*x)
        mx.eval(before)
        r._clock_steps()
        assert r.baseline == "plain"
        b = r.report.baseline
        assert (b["requested"], b["choice"], b["compiled_available"]) == ("compiled", "plain", False)
        assert "state calls" in b["reason"]
        assert b["clocks_ms"]["main"]["compiled"] is None
        after = r.model(*x)
        mx.eval(after)
        assert mx.array_equal(before, after).item()  # alive, and the same step every call
    finally:
        r.tracer.uninstall()
        assert r.tracer.verify_restored() == []


def test_a_compiled_clock_that_breaks_the_model_stops_the_job(tmp_path, monkeypatch):
    """The guard behind the detector. If a step keeps state neither kind of
    evidence shows, the compiled clock still returns a number and the model is
    already broken; the job must stop there rather than measure every later
    win against garbage."""
    import autotuner.loop as loop_mod

    monkeypatch.setattr(loop_mod, "_state_marks", lambda trace: "")  # blind the detector
    r = _runner(tmp_path, "[4, 16]", fixture="state_call.py", baseline="compiled")
    try:
        with pytest.raises(RuntimeError, match="compiled baseline"):
            r._clock_steps()
    finally:
        r.tracer.uninstall()


@pytest.mark.parametrize("error", [RuntimeError("[METAL] command buffer execution failed"),
                                  KeyboardInterrupt()])
def test_a_crash_mid_install_rolls_the_model_back(runner, tmp_path, monkeypatch, error):
    """An unexpected exception inside the install must leave the model, the
    artifact record, and the patch surface exactly as before, and be logged
    while the exception stops the job instead of submitting more GPU work."""
    import autotuner.loop as loop_mod
    from autotuner_runtime.swap import ReplayWrapper

    def exploding_e2e(*a, **k):
        raise error

    monkeypatch.setattr(loop_mod, "run_e2e", exploding_e2e)
    with pytest.raises(type(error)):
        runner._bind_and_promote(_chain(runner), FUSED, WIN)
    assert not isinstance(runner.model.chain, ReplayWrapper)
    assert not runner.tracer.patcher.installed
    assert runner.installed == {} and runner.emitted == {} and runner.cuts == {}
    assert runner.report.accepted == []


def test_a_win_that_fails_the_whole_model_check_leaves_no_trace(runner, tmp_path, monkeypatch):
    """A region-clock win the whole-model check rejects must vanish from the
    artifact record, or apply() would install it in a fresh process."""
    import json

    import autotuner.loop as loop_mod

    from autotuner.e2e import E2EResult, WorkloadCheck
    failure = E2EResult(checks=[WorkloadCheck(
        "main", 0.0, 1.0, 0.0, 1.0, False, reason="output changed")], veto_passed=False)
    monkeypatch.setattr(loop_mod, "run_e2e", lambda *a, **k: failure)
    monkeypatch.setattr(JobRunner, "_model_win", lambda self, e2e: e2e.passed)
    result = runner._bind_and_promote(_chain(runner), FUSED, WIN)
    assert not result and result.status == "correctness_failed"
    assert runner.emitted == {} and runner.installed == {}
    runner._final_check()
    art = runner.emit_artifact(tmp_path / "artifact")
    assert json.loads((art / "swap_table.json").read_text()) == []
    assert list((art / "kernels").glob("*.metal")) == []


def test_a_failed_certification_removes_the_patch_surface(runner, monkeypatch):
    """A scope that cannot be replayed invisibly gets no wrapper, and the
    tracing machinery must not stay wrapped around every op afterwards, or
    every later clock would be lying."""
    import autotuner.loop as loop_mod

    class FailedCert:
        ok = False
        reason = "forced by test"

    monkeypatch.setattr(loop_mod, "certify_identities", lambda **k: FailedCert())
    assert not runner._bind_and_promote(_chain(runner), FUSED, WIN)
    assert not runner.tracer.patcher.installed
    assert runner.installed == {} and "chain" not in runner.certified_scopes
    assert any(r["kind"] == "certification_failed" for r in runner.log.rows())


def test_an_unconfirmed_final_check_ends_the_job_normally(runner, monkeypatch):
    """Right outputs and no confirmed win is a measurement result, not a broken
    job: nothing raises, the report says why, and final_ok stays false so
    nothing is exported. Seven MetalBench jobs used to exit 1 here."""
    k_max = elementwise("rmax_u1", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);")
    assert runner._bind_and_promote(RegionRun(region=runner.region("mx.maximum")), k_max, WIN)
    monkeypatch.setattr("autotuner.e2e.step_veto",
                        lambda *args, **kwargs: (comparison_from_samples([1.0] * 4, [1.2] * 4), False))
    runner._final_check()
    assert runner.final_ok is False
    assert runner.report.final["passed"] is False and "slower" in runner.report.final["reason"]
    assert [r for r in runner.log.rows() if r["kind"] == "final_unconfirmed"]
    assert all(c["passed"] for c in runner.report.final["checks"])  # the outputs were right
