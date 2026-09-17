"""Multi-workload decisions and measurement order, independent of GPU noise."""

from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from autotuner_runtime.stats import comparison_from_samples, workload_win


def comparison(delta):
    return comparison_from_samples([100.0] * 8, [100.0 - delta] * 8)


@pytest.mark.parametrize("deltas, expected", [
    ({"first": 0, "second": 5}, True),
    ({"first": 5, "second": 0}, True),
    ({"first": 0, "second": 0}, False),
    ({"first": 5, "second": -1}, False),
    ({"first": -1, "second": 5}, False),
    ({"single": 5}, True),
    ({"single": 0}, False),
    ({"single": -1}, False),
    ({}, False),
])
def test_workload_win_requires_any_win_and_no_resolved_loss(deltas, expected):
    clocks = {name: comparison(delta) for name, delta in deltas.items()}
    assert workload_win(clocks) is expected
    assert workload_win(dict(reversed(list(clocks.items())))) is expected


@pytest.mark.parametrize("target, expected", [("win", True), ("tie", False), ("missing", False)])
def test_confirmation_requires_the_nominated_workload_to_win(target, expected):
    assert workload_win({"tie": comparison(0), "win": comparison(5)}, target) is expected


def test_uncertain_other_workload_is_allowed_but_cannot_supply_the_win():
    uncertain = comparison_from_samples([100.0] * 8, [98.0, 103.0] * 4)
    assert not uncertain.wins_by(0.0) and not uncertain.loses_by(0.0)
    assert workload_win({"target": comparison(5), "other": uncertain}, "target")
    assert not workload_win({"target": uncertain})
    assert not workload_win({}, "missing")


@pytest.mark.parametrize("field", [
    "median_baseline_ms", "median_delta_ms", "spread_ms", "sigma_ms", "median_ratio", "stability",
    "baseline_ms", "candidate_ms", "deltas_ms", "ratios",
])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_invalid_measurement_on_another_workload_cannot_hide_behind_a_win(field, invalid):
    other = comparison(0)
    value = getattr(other, field)
    invalid_value = (invalid, *value[1:]) if isinstance(value, tuple) else invalid
    other = replace(other, **{field: invalid_value})
    assert not workload_win({"target": comparison(5), "other": other}, "target")
    assert not workload_win({"target": comparison(5), "other": other})


@pytest.mark.parametrize("fields", [
    {"baseline_ms": (), "candidate_ms": (), "deltas_ms": (), "n": 0},
    {"baseline_ms": (100.0,)},
    {"n": 9},
])
def test_missing_or_incomplete_samples_are_not_an_unresolved_workload(fields):
    invalid = replace(comparison(0), **fields)
    assert not workload_win({"target": comparison(5), "other": invalid}, "target")


def test_one_sample_cannot_certify_another_workload_has_no_regression():
    unmeasured_uncertainty = comparison_from_samples([100.0], [100.0])
    assert not workload_win({"target": comparison(5), "other": unmeasured_uncertainty}, "target")


@pytest.mark.parametrize("target", [None, "large", "decode-after-prefix"])
def test_proposal_parses_optional_target_workload(target):
    from autotuner.judge.schema import validate_response
    from tests.judges import proposal
    kernel = validate_response({"mutations": [], "kernel": proposal(target_workload=target)}).kernel
    assert kernel.target_workload == target
    assert validate_response({"mutations": [], "kernel": proposal()}).kernel.target_workload is None


@pytest.mark.parametrize("target", ["", " ", "\n\t", 1, False, [], {}])
def test_proposal_rejects_invalid_target_workload(target):
    from autotuner.judge.schema import MalformedResponse, validate_response
    from tests.judges import proposal
    with pytest.raises(MalformedResponse, match="target_workload"):
        validate_response({"mutations": [], "kernel": proposal(target_workload=target)})


def test_staged_proposal_carries_one_target_for_the_complete_candidate():
    from autotuner.judge.schema import validate_response
    from tests.judges import proposal
    stage = proposal()
    stage.pop("parent_kernel_id")
    stage.update(inputs=["in0"], outputs=["out0"], output_dtypes=["float32"])
    candidate = {"parent_kernel_id": "scaffold", "output_shapes": [["in0.shape[0]"]],
                 "stages": [stage], "target_workload": "large"}
    parsed = validate_response({"mutations": [], "kernel": candidate}).kernel
    assert parsed.target_workload == "large" and len(parsed.stages) == 1


@pytest.mark.parametrize("provider", ["claude-cli", "codex"])
def test_actual_provider_request_includes_the_target_workload_schema(tmp_path, provider):
    from autotuner.judge.agent import CLI_PRESETS
    from autotuner.judge.client import _NEXT_SCHEMA
    from tests.judges import proposal
    from tests.test_judge_agent import REGION_META, calls, cli_judge
    reply = json.dumps({"mutations": [], "kernel": proposal(target_workload="large")})
    judge, state = cli_judge(tmp_path, [reply], command=CLI_PRESETS[provider]()[1:])
    parsed = judge.next(REGION_META, {"outcome": "correct_slower"})
    sent, = calls(state)
    request = sent["stdin"] + "\n".join(sent["argv"])
    assert _NEXT_SCHEMA in request
    assert 'optional "target_workload": name from region.timing_workloads' in request
    assert parsed.kernel.target_workload == "large"


def setup_e2e(monkeypatch, deltas, failed_check=None):
    from autotuner import e2e
    checked, timed = [], []

    def check(baseline, patched, name, **kwargs):
        checked.append(name)
        return e2e.WorkloadCheck(name, 0, 0, 0, 1, name != failed_check)

    def veto(session, baseline, patched, **kwargs):
        timed.append(baseline)
        result = comparison(deltas[baseline])
        return result, not result.loses_by(0.0)

    monkeypatch.setattr(e2e, "preserving_check", check)
    monkeypatch.setattr(e2e, "step_veto", veto)
    session = SimpleNamespace(off_clock=lambda fn, **kwargs: fn())
    workloads = [(name, []) for name in ("small", "large", "sweep")]
    arms = {name: (name, "candidate") for name in ("small", "large")}
    return e2e, session, workloads, arms, checked, timed


def test_e2e_checks_every_shape_then_times_nominated_target_first(monkeypatch):
    e2e, session, workloads, arms, checked, timed = setup_e2e(
        monkeypatch, {"small": 0, "large": 5})
    result = e2e.run_e2e(session, object(), object(), workloads,
                         timed=arms, target_workload="large")
    assert checked == ["small", "large", "sweep"]
    assert timed == ["large", "small"]
    assert list(result.workload_vetos) == timed
    assert result.veto is result.workload_vetos["large"]
    assert result.target_workload == "large"
    assert result.passed
    assert workload_win(result.workload_vetos, result.target_workload)


@pytest.mark.parametrize("delta", [0, -1])
def test_e2e_stops_further_timing_when_target_does_not_win(monkeypatch, delta):
    e2e, session, workloads, arms, checked, timed = setup_e2e(
        monkeypatch, {"small": 5, "large": delta})
    result = e2e.run_e2e(session, object(), object(), workloads,
                         timed=arms, target_workload="large")
    assert checked == ["small", "large", "sweep"]
    assert timed == ["large"]
    assert not workload_win(result.workload_vetos, result.target_workload)


def test_e2e_still_rejects_failure_on_correctness_only_sweep(monkeypatch):
    e2e, session, workloads, arms, checked, timed = setup_e2e(
        monkeypatch, {"small": 0, "large": 5}, failed_check="sweep")
    result = e2e.run_e2e(session, object(), object(), workloads,
                         timed=arms, target_workload="large")
    assert checked == ["small", "large", "sweep"]
    assert not timed and not result.passed


@pytest.mark.parametrize("target", ["missing", "sweep"])
def test_e2e_rejects_a_target_outside_the_timed_workloads(monkeypatch, target):
    e2e, session, workloads, arms, checked, timed = setup_e2e(monkeypatch, {})
    with pytest.raises(ValueError, match="target workload"):
        e2e.run_e2e(session, object(), object(), workloads, timed=arms, target_workload=target)
    assert not checked and not timed


def test_e2e_without_nomination_still_times_all_targets(monkeypatch):
    e2e, session, workloads, arms, checked, timed = setup_e2e(
        monkeypatch, {"small": 0, "large": 5})
    result = e2e.run_e2e(session, object(), object(), workloads, timed=arms)
    assert timed == ["small", "large"]
    assert result.target_workload is None


@pytest.mark.parametrize("deltas, success", [
    ({"small": 0, "large": 5}, True),
    ({"small": -1, "large": 5}, False),
    ({"small": 0, "large": 0}, False),
    ({}, False),
])
def test_exported_benchmark_uses_the_same_workload_decision(
    monkeypatch, tmp_path, capsys, deltas, success,
):
    from autotuner.artifact import benchmark
    metadata = {"final_benchmark": {"steps": 3, "pairs": 4}}
    (tmp_path / "bundle.json").write_text(json.dumps(metadata))
    model = SimpleNamespace(model=object())
    validator = SimpleNamespace(
        validate=lambda *args: [{"passed": True}],
        describe=lambda row: "PASS outputs",
        saved_workloads=lambda metadata: [(name, [], "declared") for name in deltas],
    )
    modules = {"load.py": SimpleNamespace(load=lambda **kwargs: model), "validate.py": validator}
    monkeypatch.setattr(benchmark, "_HERE", tmp_path)
    monkeypatch.setattr(benchmark, "_module", lambda name, path: modules[path.name])
    monkeypatch.setattr(benchmark, "BenchSession", object)
    monkeypatch.setattr(benchmark, "make_sequence", lambda *args: object())

    def measure(*args, label, **kwargs):
        clock = comparison(deltas[label])
        return {"timing": asdict(clock), "baseline_sequence_ms": 100.0,
                "candidate_sequence_ms": 100.0 - deltas[label],
                "win_confirmed": clock.wins_by(0.0), "regression_confirmed": clock.loses_by(0.0)}

    monkeypatch.setattr(benchmark, "compare_sequences", measure)
    output = tmp_path / "result.json"
    assert benchmark.main(["--json", str(output)]) == (0 if success else 1)
    stdout = capsys.readouterr().out
    summary = json.loads(output.read_text())
    assert summary["win_confirmed"] is success
    assert list(summary["workloads"]) == list(deltas)
    if any(delta == 0 for delta in deltas.values()):
        assert "unresolved change" in stdout
    if any(delta < 0 for delta in deltas.values()):
        assert "regression confirmed" in stdout
    assert ("benchmark: confirmed speedup on at least one workload" in stdout) is success
