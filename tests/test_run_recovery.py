"""Run lifecycle failures must preserve accepted work and report honest status."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autotuner.cli import _output_paths, _run_lock
from autotuner.loop import JobRunner
from autotuner.log import RunLog
from autotuner.report import Report
from autotuner.bind.emit import EmittedWrapper
from autotuner.manifest import FinalBenchmark
from autotuner_runtime.kernels import KernelSpec

FIXTURES = Path(__file__).parent / "fixtures"


def bare_runner(tmp_path):
    runner = object.__new__(JobRunner)
    runner.work_dir = tmp_path
    runner.report = Report()
    runner.log = RunLog(tmp_path / "run.jsonl")
    runner.final_ok = False
    runner.shipped_tags = {}
    runner.emitted = {}
    runner._compiled_baseline = None
    # what a checkpoint's bundle reads: the model file, the baseline, the traced inputs
    runner.manifest = SimpleNamespace(model_path=FIXTURES / "planted_win.py", workloads=(),
                                      final_benchmark=FinalBenchmark(), tolerances=None)
    runner.baseline = "plain"
    runner.tensors, runner.sweep_tensors = {}, {}
    runner.context = None  # no context workload; set by __init__ in a real run
    runner.context_seed = runner.context_tokens = None
    return runner


def test_judge_readiness_failure_prevents_runner_creation(tmp_path, monkeypatch, capsys):
    import argparse
    from autotuner import cli, loop
    from autotuner.judge.agent import CliJudge

    def fail(self):
        raise ValueError("OAuth expired; run claude auth login")
    monkeypatch.setattr(CliJudge, "check_available", fail)
    monkeypatch.setattr(loop, "JobRunner", lambda *a, **k: pytest.fail("model runner started"))
    args = SimpleNamespace(judge_cmd=None, judge="claude-cli", judge_effort=None, model=None, work_dir=tmp_path)
    with pytest.raises(SystemExit) as error:
        cli._execute(args, argparse.ArgumentParser())
    assert error.value.code == 2
    assert "claude auth login" in capsys.readouterr().err


def test_dead_judge_stops_job_and_preserves_accepted_work(tmp_path):
    from autotuner.loop import RegionRun
    runner = bare_runner(tmp_path)
    runner.tracer = SimpleNamespace(uninstall=lambda: None)
    runner.total_hypotheses = 0
    runner.report.accepted = [{"kernel": "previous_win"}]
    checkpoint = tmp_path / "checkpoints" / "accepted.metal"
    checkpoint.parent.mkdir()
    checkpoint.write_text("saved kernel")
    run = RegionRun(SimpleNamespace(fingerprint="target"))
    runner._record_empty = lambda *a: "empty"
    def search():
        for _ in range(3):
            runner._empty_reply(run, None, "CLI login expired", None, None)
        pytest.fail("continued search with unavailable judge")
    runner._run = search
    with pytest.raises(RuntimeError, match="3 consecutive transport errors"):
        runner.run()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["session"]["status"] == "failed"
    assert report["accepted"] == [{"kernel": "previous_win"}]
    assert checkpoint.read_text() == "saved kernel"
    assert runner.total_hypotheses == run.hypotheses == 0


def test_successful_reply_resets_connection_failure_streak(tmp_path):
    from autotuner.loop import RegionRun
    from autotuner.judge.schema import JudgeBabble
    runner = bare_runner(tmp_path)
    run = RegionRun(SimpleNamespace(fingerprint="target"), errors=2)
    reply = SimpleNamespace(queue=[])
    assert runner._ask_judge(run, "seed", lambda: reply) == (reply, None)
    assert run.errors == 0
    run.errors = 2
    def malformed():
        raise JudgeBabble("not JSON")
    assert runner._ask_judge(run, "seed", malformed) == (None, "babble")
    assert run.errors == 0


def test_checkpoint_preserves_exact_code_and_model_measurements(tmp_path):
    runner = bare_runner(tmp_path)
    spec = KernelSpec("accepted", "accepted", ("in0",), ("out0",), "out0[0] = in0[0];")
    wrapper = EmittedWrapper("Saved", "layer", "class Saved: pass\n", [], ["accepted"])
    runner.model = SimpleNamespace(layer=SimpleNamespace(_specs={"accepted": spec}))
    runner.emitted = {"layer": wrapper}
    accepted = {"kernel": "accepted", "timings": {"main": {"median_ratio": 0.97}}}
    saved = runner._save_checkpoint(accepted)
    assert (saved / "kernels/accepted.metal").read_text() == spec.source
    assert wrapper.source in (saved / "patch/wrappers.py").read_text()
    report = json.loads((saved / "report.json").read_text())
    assert report["accepted"][0]["timings"] == accepted["timings"]
    assert report["session"]["status"] == "accepted_checkpoint"
    assert report["session"]["artifact_validation"] == "pending"
    assert runner.report.accepted == []  # saving does not commit live model state
    runner.report.accepted.append(accepted)
    second = runner._save_checkpoint({**accepted, "kernel": "next"})
    assert second != saved and (saved / "apply.py").exists()
    # a checkpoint is a whole bundle too: the model source travels with the code
    assert (saved / "model" / "tests" / "fixtures" / "planted_win.py").exists()
    assert json.loads((saved / "bundle.json").read_text())["patches"] == [
        {"module_path": "layer", "kernel_ids": ["accepted"]}]


@pytest.mark.parametrize("error", [RuntimeError("fresh-process mismatch"), KeyboardInterrupt()])
def test_export_failure_records_status_and_preserves_checkpoint(tmp_path, monkeypatch, error):
    runner = bare_runner(tmp_path)
    checkpoint = tmp_path / "checkpoints/accepted-0001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "keep").write_text("accepted code")
    def fail(path):
        raise error
    monkeypatch.setattr(runner, "_emit_artifact", fail)
    with pytest.raises(type(error)):
        runner.emit_artifact(tmp_path / "artifact")
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["session"]["status"] == ("interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
    assert (checkpoint / "keep").read_text() == "accepted code"
    assert runner.log.rows()[-1]["kind"] == "job_failed"


def test_export_cannot_run_before_final_check(tmp_path):
    runner = bare_runner(tmp_path)
    with pytest.raises(RuntimeError, match="final whole-model check"):
        runner.emit_artifact(tmp_path / "artifact")
    assert not (tmp_path / "artifact").exists()


def test_export_success_updates_both_reports(tmp_path, monkeypatch):
    runner = bare_runner(tmp_path)
    out = tmp_path / "artifact"
    out.mkdir()
    monkeypatch.setattr(runner, "_emit_artifact", lambda path: out)
    assert runner.emit_artifact(out) == out
    for path in [tmp_path / "report.json", out / "report.json"]:
        assert json.loads(path.read_text())["session"]["status"] == "complete"


def test_default_artifact_is_scoped_to_run(tmp_path):
    assert _output_paths(tmp_path / "work", None) == tmp_path / "work/artifact"


def test_missing_judge_is_refused_before_model_load(tmp_path, monkeypatch):
    from autotuner.cli import main
    monkeypatch.setattr("autotuner.judge.agent.shutil.which", lambda command: None)
    with pytest.raises(SystemExit) as error:
        main(["run", "not-a-model.yaml", "--work-dir", str(tmp_path / "work"),
              "--judge", "claude-cli"])
    assert error.value.code == 2
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("installed,ratio,confirmed", [(False, 0.98, False),
                                                      (True, 1.0, False), (True, 0.98, True)])
def test_final_clock_does_not_call_an_unchanged_model_a_win(tmp_path, installed, ratio, confirmed):
    from autotuner.measure.clocks import comparison_from_samples
    runner = bare_runner(tmp_path)
    runner.installed = {"layer": object()} if installed else {}
    clock = comparison_from_samples([100.0] * 10, [100.0 * ratio] * 10)
    runner._record_final_clock("main", clock)
    assert runner.report.step_ms["main"]["win_confirmed"] is confirmed
    assert runner.log.rows()[-1]["win_confirmed"] is confirmed


@pytest.mark.parametrize("confirmed", [False, True])
def test_cli_summary_requires_explicit_confirmation(tmp_path, monkeypatch, capsys, confirmed):
    from autotuner.cli import main

    report = Report(baseline={"choice": "plain"}, step_ms={"main": {
        "speedup": 1.02, "after": 98.0, "baseline_at_end": 100.0,
        "stability": 0.99, "win_confirmed": confirmed,
        "sequence_speedup": 1.03, "sequence_win_confirmed": confirmed}},
        final={"sequences": {"main": {"steps": 20, "candidate_sequence_ms": 1940.0,
                                      "baseline_sequence_ms": 2000.0}}})
    def fake_runner(*args, **kwargs):
        Path(args[1]).mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(run=lambda: report, emit_artifact=lambda path: path)
    monkeypatch.setattr("autotuner.loop.JobRunner", fake_runner)
    monkeypatch.setattr("autotuner.judge.agent.CliJudge.check_available", lambda self: None)
    assert main(["run", "unused.yaml", "--work-dir", str(tmp_path / "work"),
                 "--judge", "claude-cli"]) == 0
    message = capsys.readouterr().out
    if confirmed:
        assert "1.020x confirmed speedup" in message
        assert "20 repeated forward passes: patched 1940.0 ms vs untouched 2000.0 ms" in message
        assert "1.030x confirmed speedup" in message
    else:
        assert message.count("no confirmed speedup") == 2
        assert "1.020x" not in message and "1.030x" not in message


def test_pricing_report_keeps_rejected_region_and_largest_share(tmp_path, monkeypatch):
    from autotuner import loop
    from autotuner.regions.types import Region, Stretch, Roofline
    from autotuner.regions.price import RegionPrice
    runner = bare_runner(tmp_path)
    runner.manifest = SimpleNamespace(workloads=[SimpleNamespace(name="main")])
    runner.model, runner.session, runner.store, runner.peaks = None, None, None, None
    runner.session = SimpleNamespace(wait_ready=lambda: None)
    runner.tensors, runner.traces, runner.step_ms = {"main": []}, {"main": None}, {"main": 100}
    runner.baseline, runner.selection_wave = "plain", 1
    runner._step_fn = lambda *a: None
    runner._atomic = lambda r: True
    monkeypatch.setattr(loop, "price_group", lambda *a, **kw: None)
    monkeypatch.setattr(loop, "capture_instances", lambda r, traces: [("main", r.members[0], 1)])
    monkeypatch.setattr(loop, "stretch_roofline", lambda *a, **kw: Roofline(1, 1, 1, 1, "compute", 1.1))
    regions = []
    for name, share in [("small", 0.01), ("large", 0.516)]:
        r = Region(name, ("mx.exp",), [Stretch("main", 0, 0, (0,), (1,), ())])
        r.p = {"main": share}
        r.t_rep_ms = r.t_orig_ms = {"main": share * 100}
        r.prices = {"main": RegionPrice(share, share * 100, 1.0, 1.0)}
        regions.append(r)
    assert [r.fingerprint for r in runner._price_and_rank(regions)] == ["large"]
    rows = json.loads((tmp_path / "report.json").read_text())["pricing"]
    assert rows[0]["p"] == {"main": 0.516}
    assert rows[1]["status"] == "rejected" and "under floor" in rows[1]["reason"]


@pytest.mark.parametrize("destination", ["work", ".", "work/kernels", "work/kernels/nested",
                                        "work/checkpoints", "work/report.json"])
def test_artifact_cannot_replace_run_files(tmp_path, destination):
    with pytest.raises(ValueError):
        _output_paths(tmp_path / "work", tmp_path / destination)


def test_existing_artifact_and_partial_work_are_refused(tmp_path):
    out = tmp_path / "artifact"
    out.mkdir()
    (out / "keep").write_text("previous result")
    with pytest.raises(ValueError, match="already exists"):
        _output_paths(tmp_path / "work", out)
    work = tmp_path / "work"
    work.mkdir()
    (work / "partial").write_text("interrupted preparation")
    with pytest.raises(ValueError, match="not empty"):
        _output_paths(work, None)


def test_cli_lock_rejects_concurrent_run_and_releases_after_exception():
    with pytest.raises(ValueError, match="interrupted"):
        with _run_lock():
            with pytest.raises(RuntimeError, match="another autotune"):
                with _run_lock():
                    pytest.fail("second run entered")
            raise ValueError("interrupted")
    with _run_lock():
        pass


def test_post_export_report_failure_is_recorded(tmp_path, monkeypatch):
    runner = bare_runner(tmp_path)
    out = tmp_path / 'artifact'
    out.mkdir()
    (out / 'keep').write_text('validated artifact')
    monkeypatch.setattr(runner, '_emit_artifact', lambda path: out)
    write = runner.report.write
    def fail_artifact_report(path):
        if Path(path) == out / 'report.json':
            raise OSError('report write failed')
        return write(path)
    monkeypatch.setattr(runner.report, 'write', fail_artifact_report)
    with pytest.raises(OSError, match='report write failed'):
        runner.emit_artifact(out)
    assert (out / 'keep').exists()
    assert json.loads((tmp_path / 'report.json').read_text())['session']['status'] == 'failed'
    assert runner.log.rows()[-1]['kind'] == 'job_failed'


def test_first_changing_checkpoint_keeps_tolerance_policy_before_tag_commit(tmp_path):
    import mlx.core as mx
    runner = bare_runner(tmp_path)
    runner.model = SimpleNamespace()
    runner.emitted = {}
    runner._compiled_baseline = None
    runner.shipped_tags = {}  # accepting this first changing edit has not committed yet
    runner.tensors = {'main': [mx.array([1.0])]}
    runner.sweep_tensors = {'sweep': [mx.array([2.0])]}
    runner.manifest.tolerances = (0.001, 0.00001)
    saved = runner._save_checkpoint({'kernel': 'changing', 'assoc_tag': 'changing'})
    meta = json.loads((saved / 'bundle.json').read_text())
    assert meta['correctness_rule'] == 'baseline_tolerance'
    assert not meta['goldens']
    assert meta['tolerances'] == {'rtol': 0.001, 'atol': 0.00001}
    assert set(meta['workloads']) == {'main', 'sweep'}
    assert runner.shipped_tags == {}


def test_exact_policy_follows_accepted_edits_and_pending_candidate(tmp_path):
    runner = bare_runner(tmp_path)
    runner.traces = {}  # model op types are irrelevant to its numeric policy
    assert runner._requires_exact()
    assert runner._requires_exact("preserving")
    assert not runner._requires_exact("changing")
    runner.shipped_tags = {"prior": "changing"}
    assert not runner._requires_exact("preserving")
    assert runner._requires_exact("preserving", "prior")
    assert not runner._requires_exact("preserving", "another_region")


def test_custom_region_pricing_uses_boundary_probe(tmp_path, monkeypatch):
    from autotuner import loop
    from autotuner.regions.types import Region, Stretch, Roofline
    from autotuner.regions.price import RegionPrice
    runner = bare_runner(tmp_path)
    runner.manifest = SimpleNamespace(workloads=[SimpleNamespace(name="main")])
    runner.model = runner.session = runner.store = runner.peaks = None
    runner.session = SimpleNamespace(wait_ready=lambda: None)
    runner.tensors, runner.traces, runner.step_ms = {"main": []}, {"main": None}, {"main": 100}
    runner.selection_wave = 1
    runner._step_fn = lambda *a: None
    runner._atomic = lambda r: True
    monkeypatch.setattr(loop, "price_group", lambda *a, **kw: None)
    monkeypatch.setattr(loop, "capture_instances", lambda r, traces: [("main", r.members[0], 1)])
    floors = []
    def roof(*args, **kwargs):
        floors.append(kwargs["floor_ms"])
        return Roofline(1, 0, .001, kwargs["floor_ms"], "memory", 5)
    monkeypatch.setattr(loop, "stretch_roofline", roof)
    region = Region("custom", ("metal_kernel",), [Stretch("main", 0, 0, (0,), (1,), ())])
    region.p = {"main": .10}
    region.t_rep_ms = region.t_orig_ms = {"main": 10.0}
    region.prices = {"main": RegionPrice(share=.10, ms=10.0, floor_ms=2.0, stability=1.0)}
    assert runner._price_and_rank([region]) == [region]
    assert floors == [2.0]
    assert region.removable_p["main"] == pytest.approx(.08)
    row = runner.report.pricing[0]
    assert row["ranking_basis"] == "estimated_removable_share"
    assert row["compute_model_complete"] is False


def test_no_win_cli_finishes_without_packaging(tmp_path, monkeypatch):
    from autotuner.cli import main
    report = Report()
    work = tmp_path / "work"
    def fake_runner(*args, **kwargs):
        work.mkdir()
        def forbidden(path):
            raise AssertionError("no-win run tried to package a model")
        return SimpleNamespace(run=lambda: report, emit_artifact=forbidden)
    monkeypatch.setattr("autotuner.loop.JobRunner", fake_runner)
    monkeypatch.setattr("autotuner.judge.agent.CliJudge.check_available", lambda self: None)
    assert main(["run", "unused.yaml", "--work-dir", str(work), "--judge", "claude-cli"]) == 0
    saved = json.loads((work / "report.json").read_text())
    assert saved["session"]["status"] == "complete"
    assert saved["session"]["outcome"] == "no_improvement"
    assert saved["session"]["artifact"] is None


@pytest.mark.parametrize("accepted,confirmed", [(False, False), (True, False), (True, True)])
def test_cli_reports_export_outcome(tmp_path, monkeypatch, capsys, accepted, confirmed):
    from autotuner.cli import main
    report = Report(regions=[{"s": 1.2}] if accepted else [])
    if accepted:
        report.accepted.append({"region": "example"})
    work = tmp_path / "work"
    def fake_runner(*args, **kwargs):
        work.mkdir()
        def export(path):
            assert confirmed
            path.mkdir()
            return path
        return SimpleNamespace(run=lambda: report, final_ok=confirmed, emit_artifact=export)
    monkeypatch.setattr("autotuner.loop.JobRunner", fake_runner)
    monkeypatch.setattr("autotuner.judge.agent.CliJudge.check_available", lambda self: None)
    assert main(["run", "unused.yaml", "--work-dir", str(work), "--judge", "claude-cli"]) == 0
    message = capsys.readouterr().out
    if confirmed:
        assert "verified artifact with 1/1 regions optimized" in message
        assert f"artifact: {work / 'artifact'}" in message
    else:
        assert "no confirmed improvement; no artifact produced" in message
        assert "regions optimized" not in message
        assert ("Search accepted candidates" in message) == accepted


@pytest.mark.parametrize("stage", ["run", "export"])
def test_cli_failure_is_reported_and_reraised(tmp_path, monkeypatch, capsys, stage):
    from autotuner.cli import main
    report = Report(accepted=[{"region": "example"}])
    def fail(*args):
        raise RuntimeError("test validation failure")
    monkeypatch.setattr("autotuner.loop.JobRunner", lambda *a, **kw: SimpleNamespace(
        run=fail if stage == "run" else lambda: report, final_ok=True, emit_artifact=fail))
    monkeypatch.setattr("autotuner.judge.agent.CliJudge.check_available", lambda self: None)
    with pytest.raises(RuntimeError, match="test validation failure"):
        main(["run", "unused.yaml", "--work-dir", str(tmp_path / "work"), "--judge", "claude-cli"])
    captured = capsys.readouterr()
    assert "job failed:" in captured.err
    assert "job complete:" not in captured.out


@pytest.mark.parametrize("kind,label", [
    ("repeated_forward", "20 repeated forward passes"),
    ("advancing_cache_fixed_tokens", "20 steps with advancing cache and fixed input tokens"),
    ("library_generation", "20 generated tokens via library inference"),
])
def test_cli_sequence_label_matches_measurement(kind, label, capsys):
    from autotuner.cli import _print_result
    report = Report(step_ms={"main": {"sequence_speedup": 1.1}}, final={"sequences": {
        "main": {"workload_kind": kind, "steps": 20,
                 "candidate_sequence_ms": 90, "baseline_sequence_ms": 100}}})
    _print_result(report, None)
    message = capsys.readouterr().out
    assert label in message
    if kind == "library_generation":
        assert "baseline 200.00, optimized 222.22" in message
        assert "not decode-only throughput" in message
    else:
        assert "tokens/sec" not in message


def test_cli_generation_clock_is_not_called_forward_pass(capsys):
    from autotuner.cli import _print_result
    report = Report(constants={"measurement": {"kind": "library_generation", "generated_tokens": 10}},
                    step_ms={"main": {"speedup": 1.1, "after": 90, "baseline_at_end": 100,
                                      "stability": 1}})
    _print_result(report, None)
    message = capsys.readouterr().out
    assert "library generation (10 generated tokens)" in message
    assert "forward pass" not in message
