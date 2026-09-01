"""M10: the whole job end to end with the scripted judge.

The planted-win run must find the unfused chain, ship the scripted fused
kernel through bind and e2e, close by rule, and leave a working artifact. The
vendor-parity run must ship nothing and say so.
"""

import textwrap
from pathlib import Path

import mlx.core as mx
import pytest

from autotuner.judge.scripted import ScriptedJudge
from autotuner.loop import JobRunner
from autotuner.measure.session import Session

# whole jobs and live models: minutes, not seconds
pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent / "fixtures"

# max/min guard NaN like mx.maximum/minimum do (NaN propagates); the regime
# gate fails a kernel that quietly maps NaN to 0 the way bare metal::max does
FUSED_CHAIN_SOURCE = """\
uint i = thread_position_in_grid.x;
uint c = i % (uint)in0_shape[1];
float x = in0[i];
float y = x * 2.0f + in1[c];
y = (y != y) ? y : metal::max(y, 0.0f);
y = y * x;
y = y + in2[c];
y = (y != y) ? y : metal::min(y, 8.0f);
out0[i] = (y - 1.0f) * 0.5f;
"""


def write_manifest(tmp_path, fixture, shape):
    p = tmp_path / "manifest.yaml"
    p.write_text(textwrap.dedent(f"""
        model: {FIXTURES / fixture}
        workloads:
          - inputs: [{{shape: {list(shape)}, dtype: float32}}]
            name: main
        budget: {{per_region: 4, total: 8}}
    """))
    return p


def winning_chain_judge(region):
    if len(region.ops) != 8:
        return yielding_judge(region)  # only the full chain gets the real attempt
    return ScriptedJudge([
        {"queue": [{"id": "h1", "kind": "on-chip", "assoc_tag": "preserving",
                    "hypothesis": "keep the chain's intermediates in registers"}]},
        {"mutations": [], "kernel": {
            "source": FUSED_CHAIN_SOURCE,
            "parent_kernel_id": "scaffold",
            "grid": ["in0.shape[0] * in0.shape[1]", "1", "1"],
            "threadgroup": ["min(in0.shape[0] * in0.shape[1], 256)", "1", "1"],
            "output_shapes": [["in0.shape[0]", "in0.shape[1]"]],
        }},
        {"mutations": [], "kernel": None},
    ])


def yielding_judge(region):
    return ScriptedJudge([
        {"queue": [{"id": "h1", "kind": "retile", "assoc_tag": "preserving",
                    "hypothesis": "try a tile"}]},
        {"mutations": [], "kernel": None},
    ])


def test_planted_win_job_ships(tmp_path):
    from tests.conftest import require_healthy_gpu

    require_healthy_gpu()  # ships a real measured win; a mid-test throttle crossing can hide it
    manifest = write_manifest(tmp_path, "planted_win.py", (4096, 1024))
    next_payloads = []

    def factory(region):
        j = winning_chain_judge(region)
        orig = j.next

        def recording_next(meta, verdict):
            next_payloads.append(meta)
            return orig(meta, verdict)

        j.next = recording_next
        return j

    runner = JobRunner(manifest, tmp_path / "work", judge_factory=factory, clock_pairs=8,
                       session=Session())
    report = runner.run()

    # the judge edits a named parent, so every next call must carry the
    # lineage sources and the head id (the 2026-08-31 run starved this),
    # plus its only memory: the queue, the verdicts, and the executing item
    assert next_payloads
    for meta in next_payloads:
        assert meta["head"] in meta["kernels"]
        assert "source" in meta["kernels"][meta["head"]]
        assert "queue" in meta and "verdicts" in meta and "families" in meta
        assert meta["executing"]["id"]

    shipped = [r for r in report.regions if r.get("s")]
    assert shipped, f"nothing shipped; regions: {report.regions}"
    hyp = [h for h in report.hypotheses if h["verdict"] == "shipped"]
    assert hyp, report.hypotheses
    before = report.step_ms["main"]["before"]
    after = report.step_ms["main"]["after"]
    assert after < before, f"step did not improve: {before:.3f} -> {after:.3f}"

    # the artifact must reproduce the patched outputs in this process
    art = runner.emit_artifact(tmp_path / "artifact")
    assert (art / "kernels").exists() and (art / "patch" / "wrappers.py").exists()

    # run.jsonl must tell the whole story: every skeleton event of a shipping
    # run appears, and the report on disk is current (written per region close)
    assert all("t" in row and "wall" in row for row in runner.log.rows())
    kinds = [row["kind"] for row in runner.log.rows()]
    for expected in ("job", "model", "trace", "regions", "step_clock", "peaks",
                     "ranked", "region_open", "scaffold_ok", "judge", "verdict",
                     "shipped", "region_closed"):
        assert expected in kinds, f"missing {expected!r} in run log: {kinds}"
    assert kinds.index("region_open") < kinds.index("scaffold_ok") < kinds.index("shipped")
    judge_rows = [r for r in runner.log.rows() if r["kind"] == "judge"]
    assert {r["phase"] for r in judge_rows} == {"seed", "next"}
    assert all("latency_s" in r for r in judge_rows)
    assert (tmp_path / "work" / "report.json").exists()


def test_vendor_parity_job_ships_nothing(tmp_path):
    manifest = write_manifest(tmp_path, "vendor_parity.py", (256, 512))
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=yielding_judge, clock_pairs=8,
                       session=Session())
    report = runner.run()
    shipped = [r for r in report.regions if r.get("s")]
    assert not shipped
    assert not runner.installed and not runner.emitted
    assert report.step_ms["main"]["before"] > 0 and report.step_ms["main"]["after"] > 0
    # the headline speedup is now paired against the untouched model measured in
    # the same window, and carries the agreement of the pairs behind it. The old
    # before/after subtraction could not be asserted on at all: it read whatever
    # the machine did between job start and job end.
    step = report.step_ms["main"]
    assert step["baseline_at_end"] > 0
    assert 0 < step["stability"] <= 1
    # that an unchanged model reports no speedup is pinned deterministically in
    # test_drift; here it is only claimable when the pairs actually agreed, and
    # this fixture's step is microseconds
    if step["stability"] > 0.8:
        assert step["speedup"] == pytest.approx(1.0, rel=0.15), step


def test_used_work_dir_refused(tmp_path):
    """A reused work dir would interleave two jobs' run.jsonl rows."""
    manifest = write_manifest(tmp_path, "planted_win.py", (64, 64))
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "run.jsonl").write_text('{"kind": "job"}\n')
    with pytest.raises(RuntimeError, match="previous run"):
        JobRunner(manifest, tmp_path / "work", judge_factory=yielding_judge)


def test_ship_crash_restores_model_and_continues(tmp_path, monkeypatch):
    """Audit finding 1: an unexpected exception mid-installation must roll the
    model back, close the region with a named reason, and let the job finish,
    never die with wrappers installed."""
    import autotuner.loop as loop_mod

    def exploding_e2e(*a, **k):
        raise RuntimeError("[METAL] command buffer execution failed")

    monkeypatch.setattr(loop_mod, "run_e2e", exploding_e2e)
    manifest = write_manifest(tmp_path, "planted_win.py", (4096, 1024))
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=winning_chain_judge, clock_pairs=8,
                       session=Session())
    report = runner.run()  # must complete, not raise
    assert not [r for r in report.regions if r.get("s")]
    from autotuner_runtime.swap import ReplayWrapper
    assert not isinstance(runner.model.chain, ReplayWrapper)
    assert not runner.tracer.patcher.installed
    kinds = [row["kind"] for row in runner.log.rows()]
    assert "bind_failed" in kinds


def test_rolled_back_ship_leaves_artifact_clean(tmp_path, monkeypatch):
    """Audit finding 2: a win that fails the whole-model check must vanish
    from the artifact record, or apply() crashes in a fresh process."""
    import json as json_mod

    import autotuner.loop as loop_mod

    class FailedE2E:
        passed = False
        veto_passed = False
        checks = ()

    monkeypatch.setattr(loop_mod, "run_e2e", lambda *a, **k: FailedE2E())
    manifest = write_manifest(tmp_path, "planted_win.py", (4096, 1024))
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=winning_chain_judge, clock_pairs=8,
                       session=Session())
    runner.run()
    assert runner.emitted == {}
    art = runner.emit_artifact(tmp_path / "artifact")
    table = json_mod.loads((art / "swap_table.json").read_text())
    assert table == []


def test_certification_failure_removes_patch_surface(tmp_path, monkeypatch):
    """Audit finding 3: a failed identity certification must not leave the
    tracing machinery wrapped around every op, or every later clock lies."""
    import autotuner.loop as loop_mod

    class FailedCert:
        ok = False
        reason = "forced by test"

    monkeypatch.setattr(loop_mod, "certify_identity", lambda **k: FailedCert())
    manifest = write_manifest(tmp_path, "planted_win.py", (4096, 1024))
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=winning_chain_judge, clock_pairs=8,
                       session=Session())
    report = runner.run()
    assert not runner.tracer.patcher.installed
    assert not [r for r in report.regions if r.get("s")]


def test_judge_transport_error_costs_region_not_job(tmp_path):
    """Audit finding F1: a transport failure (CLI exit, timeout) closes the
    region with a named reason and the job completes."""
    class DeadTransport:
        def seed(self, meta):
            raise RuntimeError("claude CLI judge exited 3: no such model")

        def next(self, meta, verdict):
            raise RuntimeError("unreachable")

    manifest = write_manifest(tmp_path, "planted_win.py", (64, 1024))
    runner = JobRunner(manifest, tmp_path / "work",
                       judge_factory=lambda region: DeadTransport(),
                       session=Session())
    report = runner.run()  # must complete
    assert all("judge unavailable" in (r.get("close") or r.get("close_rule") or "")
               or not r.get("s") for r in report.regions)
    assert any(row.get("action") == "error" for row in runner.log.rows()
               if row["kind"] == "judge")
