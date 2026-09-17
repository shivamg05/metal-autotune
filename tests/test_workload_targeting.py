"""Target selection and trace-local identities, with no GPU execution."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from autotuner.e2e import E2EResult
from autotuner.ladder.gates import LadderResult
from autotuner.loop import JobRunner, MIN_WIN_MS, SHIP_PAIRS
from autotuner.regions.types import Region, Roofline, Stretch
from autotuner.trace.recorder import ArrayRef
from autotuner.trace.serialize import nodes_from_json
from autotuner.trace.types import Trace, TraceNode
from autotuner_runtime.stats import comparison_from_samples


def clock(delta):
    return comparison_from_samples([100.0] * 8, [100.0 - delta] * 8)


def runner_for_targets(names=("small", "large")):
    runner = object.__new__(JobRunner)
    runner.manifest = SimpleNamespace(workloads=[SimpleNamespace(name=name) for name in names], tolerances=None)
    runner.session = object()
    runner.log = SimpleNamespace(append=lambda *args, **kwargs: None)
    return runner


def target_region(order=("small", "large")):
    return Region("targets", ("mx.add",),
                  members=[Stretch(name, 0, 0, (0, 1), (2,), ("layer@0",)) for name in order],
                  p={"small": .7, "large": .3},
                  rooflines={"small": Roofline(0, 0, 0, 0, "memory", 1.01),
                             "large": Roofline(0, 0, 0, 0, "memory", 3)})


def test_default_target_uses_headroom_independently_of_manifest_and_member_order():
    for names in (("small", "large"), ("large", "small")):
        runner = runner_for_targets(names)
        assert runner._target_workload(target_region(names)) == "large"
        assert runner._target_workload(target_region(names), "small") == "small"


def test_default_target_breaks_headroom_ties_by_share_then_name():
    runner = runner_for_targets()
    region = target_region()
    region.rooflines.clear()
    assert runner._target_workload(region) == "small"
    region.p = {"small": .3, "large": .3}
    assert runner._target_workload(region) == "large"
    region.members.reverse()
    assert runner._target_workload(region) == "large"


def test_unknown_target_is_an_ordinary_static_failure_without_a_worker(monkeypatch):
    runner = runner_for_targets()
    runner._ladder_job = lambda *args, **kwargs: pytest.fail("invalid target must not create a worker job")
    result = runner._evaluate_kernel(target_region(), object(), "preserving", target_workload="missing")
    assert result.outcome == "failed" and result.failed_gate == "static"
    assert "missing" in result.detail["reason"]
    assert "large" in result.detail["reason"] and "small" in result.detail["reason"]


def node(seq, first_id, width):
    return TraceNode(seq, "mx.add", (first_id, first_id + 1), (first_id + 2,),
                     (((width, 8), "float32"),) * 2, (((width, 8), "float32"),),
                     {"args": (ArrayRef(0), ArrayRef(1)), "kwargs": {}}, "layer@0", seq)


def trace(nodes, weights=()):
    return Trace(tuple(nodes), {}, (), frozenset(weights), frozenset(), {})


def span(workload, node):
    return Stretch(workload, node.seq, node.seq, node.in_arrays, node.out_arrays, ("layer@0",))


def test_ladder_target_keeps_nodes_ids_files_weight_mask_and_copy_margin_together():
    from autotuner.ladder.child import _project_ids
    runner = runner_for_targets()
    small = node(0, 10, 4)
    large_nodes = [node(0, 100, 1), node(1, 110, 16), node(2, 120, 16), node(3, 130, 32)]
    sweep = node(0, 200, 3)
    region = Region("same-operation", ("mx.add",), members=[span("small", small)] +
                    [span("large", n) for n in large_nodes[1:]],
                    t_orig_ms={"small": 30, "large": 56}, t_rep_ms={"small": 30, "large": 3},
                    prices={"small": SimpleNamespace(ms=30), "large": SimpleNamespace(ms=3),
                            "large@copy:3": SimpleNamespace(ms=50)},
                    rooflines={"large": Roofline(0, 2.5, 0, 2.5, "compute", 1.2)})
    runner.traces = {"small": trace([small], weights=[10]),
                     "large": trace(large_nodes, weights=[111, 121, 131])}
    runner.sweep_traces = {"sweep": trace([sweep])}
    runner.sweep_spans = {(region.fingerprint, "sweep"): span("sweep", sweep)}
    runner.store = SimpleNamespace(
        set_count=lambda *args: 2,
        _path=lambda fingerprint, label, index, kind: Path(fingerprint) / label / f"set{index}.{kind}")
    contract = object()
    runner._contract = lambda region: contract
    runner.baseline = "compiled"
    runner.clock_pairs = 8
    job = runner._ladder_job(region, object(), "preserving", True, "large")
    primary = nodes_from_json(job.nodes_json)
    assert primary == [large_nodes[1]]
    assert job.input_ids == (110, 111) and job.output_ids == (112,)
    assert job.weight_inputs == (False, True)
    assert [es.label for es in job.eval_sets] == ["large", "small", "large@copy:3", "sweep"]
    assert job.eval_sets[0].inputs_paths == ["same-operation/large/set0.inputs", "same-operation/large/set1.inputs"]
    assert job.eval_sets[0].reference_paths == ["same-operation/large/set0.outputs", "same-operation/large/set1.outputs"]
    assert job.eval_sets[0].nodes_json == job.nodes_json
    assert job.eval_sets[-1].correctness_only
    assert job.min_win_ms == MIN_WIN_MS / 2  # two copies of the selected local shape
    assert job.compute_floor_ms == 2.5
    assert job.timeout_s == 120 + .8 * 50  # still allows correctness on the largest other case
    assert job.contract is contract and job.baseline == "compiled" and job.clock_pairs == 8
    # The child projects the selected trace's array ids into each other trace.
    for es, expected_inputs, expected_outputs in [
        (job.eval_sets[1], (10, 11), (12,)),
        (job.eval_sets[2], (130, 131), (132,)),
        (job.eval_sets[3], (200, 201), (202,)),
    ]:
        other = nodes_from_json(es.nodes_json)
        assert _project_ids(primary, other, job.input_ids) == expected_inputs
        assert _project_ids(primary, other, job.output_ids) == expected_outputs


def test_evaluation_records_selected_target_and_timing_case(monkeypatch):
    from autotuner import loop
    runner = runner_for_targets()
    runner.installed = {}
    job = SimpleNamespace(eval_sets=[SimpleNamespace(label="large@copy:4")])
    seen = []
    runner._ladder_job = lambda region, kernel, assoc_tag, run_clock, target: seen.append(target) or job
    measured = LadderResult("correct_slower", None, {}, 2, 1, -1, 0, [])
    monkeypatch.setattr(loop, "run_ladder", lambda actual, session: measured)
    result = runner._evaluate_kernel(target_region(), SimpleNamespace(kernel_id="candidate"), "preserving",
                                      target_workload="large")
    assert seen == ["large"]
    assert result.detail["target_workload"] == "large"
    assert result.detail["timing_case"] == "large@copy:4"


@pytest.mark.parametrize("deltas, target, accepted", [
    ({"small": 0, "large": 5}, "large", True),
    ({"small": 5, "large": 0}, "large", False),
    ({"small": -1, "large": 5}, "large", False),
    ({"large": 5}, "large", False),
    ({"small": 0, "large": 5, "sweep": 0}, "large", False),
    ({"small": 0, "large": 5}, None, True),
    ({}, None, False),
])
def test_model_win_requires_the_complete_declared_workload_set(deltas, target, accepted):
    runner = runner_for_targets()
    comparisons = {name: clock(delta) for name, delta in deltas.items()}
    result = E2EResult(checks=[SimpleNamespace(passed=True)], veto=next(iter(comparisons.values()), None),
                       workload_vetos=comparisons, target_workload=target)
    assert runner._model_win(result) is accepted


def test_confirmation_cannot_switch_to_a_different_winning_workload(monkeypatch):
    from autotuner import loop
    runner = runner_for_targets()
    nomination = E2EResult(veto=clock(5), workload_vetos={"small": clock(0), "large": clock(5)},
                           target_workload="large")
    measured = []

    def measure(session, baseline, candidate, **kwargs):
        measured.append((baseline, kwargs["pairs"]))
        return clock({"large": 0, "small": 5}[baseline])

    monkeypatch.setattr(loop, "compare", measure)
    confirmed = runner._confirm_model_win(nomination, object(),
                                          timed={"small": ("small", None), "large": ("large", None)})
    assert measured == [("large", SHIP_PAIRS)]
    assert confirmed.target_workload == "large"
    assert not runner._model_win(confirmed)


def test_confirmation_retains_other_targets_regression_check(monkeypatch):
    from autotuner import loop
    runner = runner_for_targets()
    nomination = E2EResult(veto=clock(5), workload_vetos={"small": clock(0), "large": clock(5)},
                           target_workload="large")
    measured = []
    monkeypatch.setattr(loop, "compare", lambda session, baseline, candidate, **kwargs:
                        measured.append(baseline) or clock({"large": 5, "small": -1}[baseline]))
    confirmed = runner._confirm_model_win(nomination, object(),
                                          timed={"small": ("small", None), "large": ("large", None)})
    assert measured == ["large", "small"]
    assert not runner._model_win(confirmed)


def test_confirmation_rejects_empty_timings_without_crashing():
    runner = runner_for_targets()
    nomination = E2EResult(veto=clock(5), workload_vetos={"large": clock(5)}, target_workload="large")
    confirmed = runner._confirm_model_win(nomination, object(), timed={})
    assert not runner._model_win(confirmed)
