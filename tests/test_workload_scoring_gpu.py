"""A real second-workload win must survive the entire delivery flow."""

import json
from pathlib import Path

import pytest

from autotuner.judge.scripted import ScriptedJudge
from autotuner.loop import JobRunner
from autotuner.measure.session import Session
from tests.test_loop import FUSED_CHAIN_SOURCE


@pytest.mark.integration
def test_second_workload_win_ships_and_packages(tmp_path):
    from tests.conftest import require_healthy_gpu
    require_healthy_gpu()
    fixture = Path(__file__).parent / "fixtures" / "workload_win.py"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {fixture}
baseline: plain
workloads:
  - name: small
    inputs: [{{shape: [1, 1024], dtype: float32}}]
  - name: large
    inputs: [{{shape: [4096, 1024], dtype: float32}}]
budget: {{per_region: 1, total: 1}}
final_benchmark: {{steps: 3, pairs: 8, warmup_steps: 3}}
""")

    def judge(region):
        return ScriptedJudge([
            {"queue": [{"id": "fuse", "kind": "on-chip", "assoc_tag": "preserving",
                        "hypothesis": "fuse the large input's eight memory passes"}]},
            {"mutations": [], "kernel": {
                "parent_kernel_id": "scaffold", "target_workload": "large",
                "source": FUSED_CHAIN_SOURCE,
                "grid": ["in0.shape[0] * in0.shape[1]", "1", "1"],
                "threadgroup": ["256", "1", "1"],
                "output_shapes": [["in0.shape[0]", "in0.shape[1]"]],
            }},
        ])

    runner = JobRunner(manifest, tmp_path / "work", judge, session=Session())
    # Focus this integration test on one known opportunity. Measurement,
    # correctness, confirmation, final sequences, and export are unmodified.
    discover = runner.build_regions
    runner.build_regions = lambda: [next(r for r in discover() if len(r.ops) == 8)]
    try:
        report = runner.run()
        assert report.accepted, report.final
        assert report.final["passed"], report.final
        for accepted in report.accepted:
            assert accepted["target_workload"] == "large"
            assert accepted["model_ratio"] is None
            assert set(accepted["model_ratios"]) == {"small", "large"}
        assert report.final["sequences"]["large"]["win_confirmed"]
        artifact = runner.emit_artifact(tmp_path / "artifact")
        assert (artifact / "load.py").is_file()
        assert (artifact / "runtime" / "autotuner_runtime" / "stats.py").is_file()
        # Fresh-process validation is part of emit_artifact, not mocked.
        events = runner.log.rows()
        assert any(row["kind"] == "artifact_checked" for row in events)
        (tmp_path / "measured-results.json").write_text(json.dumps({
            "accepted": report.accepted, "final": report.final,
        }, indent=2))
    finally:
        runner.tracer.uninstall()


@pytest.mark.integration
@pytest.mark.parametrize("baseline", ["plain", "compiled"])
def test_second_shape_is_timed_and_both_shapes_are_checked(tmp_path, baseline):
    from autotuner.judge.schema import validate_response
    from autotuner.loop import RegionRun
    fixture = Path(__file__).parent / "fixtures" / "planted_win.py"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {fixture}
baseline: {baseline}
workloads:
  - name: small
    inputs: [{{shape: [4, 1024], dtype: float32}}]
  - name: large
    inputs: [{{shape: [256, 1024], dtype: float32}}]
""")
    runner = JobRunner(manifest, tmp_path / "work", lambda region: None, session=Session())
    try:
        runner.load_model()
        runner.trace_workloads()
        region = next(r for r in runner.build_regions() if len(r.ops) == 8)
        assert region.workloads == ("small", "large")
        runner._capture([region])
        runner.tracer.uninstall()
        runner._clock_steps()
        # This region is the entire model's eight-op chain, at both shapes.
        region.t_orig_ms = dict(runner.step_ms)
        region.t_rep_ms = dict(runner.step_ms)
        seed = runner._build_scaffold(region)
        run = RegionRun(region, scaffold=seed, kernels={seed.kernel_id: seed})
        proposal = validate_response({"mutations": [], "kernel": {
            "parent_kernel_id": "scaffold", "target_workload": "large",
            "source": FUSED_CHAIN_SOURCE,
            "grid": ["in0.shape[0] * in0.shape[1]", "1", "1"],
            "threadgroup": ["256", "1", "1"],
            "output_shapes": [["in0.shape[0]", "in0.shape[1]"]],
        }}).kernel
        kernel = runner._kernel_from_proposal(run, region, proposal, "second_shape")
        result = runner._evaluate_kernel(region, kernel, "preserving", target_workload="large")
        assert result.outcome != "failed", result
        assert result.detail["target_workload"] == result.detail["timing_case"] == "large"
        assert result.region_ms is not None and result.library_ms is not None
    finally:
        runner.tracer.uninstall()
