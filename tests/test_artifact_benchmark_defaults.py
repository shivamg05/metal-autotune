"""Exported timings must retain each workload's measured repetition count."""
from dataclasses import asdict
from types import SimpleNamespace
import json

import pytest

from autotuner.artifact import benchmark
from autotuner_runtime.stats import comparison_from_samples


@pytest.mark.parametrize("inference,override,expected", [
    (False, None, 111), (False, 9, 9), (True, None, 10), (True, 9, 9),
])
def test_workload_length(inference, override, expected):
    metadata = {"use_library_inference": inference,
                "final_benchmark": {"steps": 10, "steps_by_workload": {"short": 111}}}
    assert benchmark.workload_steps(metadata, "short", override) == expected
    assert benchmark.workload_steps(metadata, "other", override) == (override or 10)


def test_legacy_bundle_length():
    assert benchmark.workload_steps({"final_benchmark": {"steps": 7}}, "main") == 7
    assert benchmark.workload_steps({"final_benchmark": {}}, "main") == 20


@pytest.mark.parametrize("override,expected", [(None, [111, 23]), (9, [9, 9])])
def test_benchmark_times_each_declared_length(tmp_path, monkeypatch, override, expected):
    metadata = {"final_benchmark": {"steps": 10, "steps_by_workload": {"small": 111, "large": 23},
                                    "warmup_steps_per_sample": {"small": 111, "large": 5}}}
    (tmp_path / "bundle.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(benchmark, "_HERE", tmp_path)
    model = SimpleNamespace(model=None)
    loader = SimpleNamespace(load=lambda **kw: model)
    validator = SimpleNamespace(
        validate=lambda *a: [],
        saved_workloads=lambda m: [("small", [], "declared"), ("large", [], "declared"),
                                  ("edge", [], "sweep")],
    )
    monkeypatch.setattr(benchmark, "_module", lambda name, path: loader if "load" in name else validator)
    lengths = []
    def make_sequence(step, inputs, length):
        lengths.append(length)
        return lambda: None
    monkeypatch.setattr(benchmark, "make_sequence", make_sequence)
    timing = asdict(comparison_from_samples([10.] * 8, [8.] * 8))
    monkeypatch.setattr(benchmark, "compare_sequences", lambda *a, **kw: {
        "timing": timing, "baseline_sequence_ms": 10., "candidate_sequence_ms": 8.})
    output = tmp_path / "result.json"
    args = ["--json", str(output)] + ([] if override is None else ["--steps", str(override)])
    assert benchmark.main(args) == 0
    assert lengths == [expected[0]] * 2 + [111] * 2 + [expected[1]] * 2 + [5] * 2
    result = json.loads(output.read_text())
    assert result["steps_by_workload"] == dict(zip(("small", "large"), expected))
    assert result["steps"] == override

