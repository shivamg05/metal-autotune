"""Nested scopes must compose into one replay, including after export."""

from dataclasses import replace

import pytest

from autotuner.bind.emit import (
    NotReplayable, Splice, compose_scope_variants, emit_wrapper_variants,
)
from autotuner.trace.recorder import ArrayRef
from autotuner.trace.types import ScopeCall, Trace, TraceNode
from autotuner_runtime.kernels import KernelSpec


def _kernel(name):
    return KernelSpec(
        kernel_id=name, name=name, input_names=("x",), output_names=("out",),
        source="uint i = thread_position_in_grid.x; out[i] = x[i];",
        grid=("in0.shape[0] * in0.shape[1]", "1", "1"), threadgroup=("32", "1", "1"),
        output_shapes=(("in0.shape[0]", "in0.shape[1]"),), output_dtypes=("float32",),
    )


def _record(rows=4, offset=0, call=0):
    """A child multiply, two parent clamps, and another child multiply."""
    base = f"block@{call}"
    scopes = [ScopeCall(
        address=address, stack=stack, args_template=(ArrayRef(0),), kwargs_template={},
        arg_ids=(offset + first,), out_template=ArrayRef(0), out_ids=(offset + last,),
    ) for address, stack, first, last in [
        ("@0", ("@0",), 0, 4),
        (base, ("@0", base), 0, 4),
        (f"block.proj@{call}", ("@0", base, f"block.proj@{call}"), 0, 1),
        (f"block.out@{call}", ("@0", base, f"block.out@{call}"), 3, 4),
    ]]
    nodes = []
    for seq, (op, scalar, scope) in enumerate([
        ("array.__mul__", 2.0, scopes[2]),
        ("mx.maximum", 0.0, scopes[1]),
        ("mx.minimum", 8.0, scopes[1]),
        ("array.__mul__", 0.5, scopes[3]),
    ]):
        nodes.append(TraceNode(
            seq=seq, op=op, in_arrays=(offset + seq,), out_arrays=(offset + seq + 1,),
            in_specs=(((rows, 16), "float32"),), out_specs=(((rows, 16), "float32"),),
            scalar_args={"args": (ArrayRef(0), scalar), "kwargs": {}},
            module_address=scope.address, position_in_module=0, module_stack=scope.stack,
        ))
    trace = Trace(tuple(nodes), {}, (offset + 4,), frozenset(), frozenset({offset}), {},
                  scope_calls=tuple(scopes))
    return trace, scopes[1:]


def _splice(name, start, end, offset=0):
    return Splice(_kernel(name), start, end, (offset + start,), (offset + end + 1,), name)


def test_parent_absorbs_children_and_keeps_every_workload_variant():
    a, (parent_a, child_a, _) = _record()
    b, (parent_b, child_b, _) = _record(rows=7, offset=100, call=1)
    child_cuts = {
        ("a", child_a.address): [_splice("old", 0, 0)],
        ("b", child_b.address): [_splice("old", 0, 0, 100)],
    }
    installed = {"block.proj": child_cuts, "unrelated": {}}
    updates = compose_scope_variants(
        {"a": a, "b": b}, installed,
        {"block": [("a", parent_a, _splice("new", 1, 2))]},
    )
    assert set(updates) == {"block"}
    assert list(updates["block"]) == [("a", parent_a.address), ("b", parent_b.address)]
    assert [s.kernel.kernel_id for s in updates["block"][("a", parent_a.address)]] == ["old", "new"]
    assert updates["block"][("b", parent_b.address)][0].input_ids == (100,)
    assert len(child_cuts[("a", child_a.address)]) == 1  # planning is transactional
    emitted = emit_wrapper_variants([
        (a, parent_a, updates["block"][("a", parent_a.address)]),
        (b, parent_b, updates["block"][("b", parent_b.address)]),
    ], "Combined")
    assert emitted.kernel_ids == ["old", "new"]
    assert "self.wrapped.proj(" not in emitted.source  # calls stay inline


def test_child_after_parent_updates_parent_and_replaces_only_its_own_cut():
    trace, (parent, child, out) = _record()
    prior = {("a", parent.address): [_splice("old", 0, 0), _splice("parent", 1, 2)]}
    updates = compose_scope_variants(
        {"a": trace}, {"block": prior}, {
            "block.proj": [("a", child, _splice("better", 0, 0))],
            "block.out": [("a", out, _splice("out", 3, 3))],
        },
    )
    assert set(updates) == {"block"}
    assert [s.kernel.kernel_id for s in updates["block"][("a", parent.address)]] == [
        "better", "parent", "out",
    ]
    assert prior[("a", parent.address)][0].kernel.kernel_id == "old"


def test_new_parent_and_child_in_same_batch_share_one_wrapper():
    trace, (parent, child, _) = _record()
    updates = compose_scope_variants({"a": trace}, {}, {
        "block.proj": [("a", child, _splice("child", 0, 0))],
        "block": [("a", parent, _splice("parent", 1, 2))],
    })
    assert set(updates) == {"block"}
    assert len(updates["block"][("a", parent.address)]) == 2


def test_overlapping_parent_cut_cannot_hide_an_existing_child():
    trace, (parent, child, _) = _record()
    with pytest.raises(NotReplayable, match="overlapping cuts"):
        compose_scope_variants({"a": trace}, {
            "block.proj": {("a", child.address): [_splice("child", 0, 0)]},
        }, {"block": [("a", parent, _splice("overlap", 0, 2))]})


@pytest.mark.parametrize("span", [(-1, 0), (0, 4), (3, 2)])
def test_emitter_rejects_cuts_outside_its_scope(span):
    trace, (parent, _, _) = _record()
    with pytest.raises(NotReplayable, match="outside scope"):
        emit_wrapper_variants([(trace, parent, [_splice("bad", *span)])], "Bad")


def test_scope_path_is_not_enough_without_the_enclosing_call():
    trace, (parent, child, _) = _record()
    detached = replace(child, stack=("@0", child.address))
    with pytest.raises(NotReplayable, match="no enclosing call"):
        compose_scope_variants({"a": trace}, {}, {
            "block": [("a", parent, _splice("parent", 1, 2))],
            "block.proj": [("a", detached, _splice("child", 0, 0))],
        })


def _real_runner(tmp_path, monkeypatch):
    from pathlib import Path

    from autotuner.loop import JobRunner
    from autotuner.measure.session import Session

    model = Path(__file__).parent / "fixtures" / "nested_scopes.py"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {model}
baseline: plain
workloads:
  - name: small
    inputs: [{{shape: [4, 16], dtype: float32}}]
  - name: larger
    inputs: [{{shape: [7, 16], dtype: float32}}]
""")
    # Only the speed decision is replaced. Identity, correctness, live Metal
    # dispatch and literal retrace all still have to pass for these tiny ops.
    monkeypatch.setattr(JobRunner, "_model_win", lambda self, result: all(c.passed for c in result.checks))
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: None,
                       clock_pairs=4, session=Session(sleep=lambda seconds: None))
    runner.load_model()
    runner.trace_workloads()
    regions = runner.build_regions()
    child = next(r for r in regions if r.ops == ("array.__mul__",))
    parent = next(r for r in regions if r.ops == ("mx.maximum", "mx.minimum"))
    child_kernel = replace(_kernel("nested_child"), input_names=("x", "weight"), source=(
        "uint i = thread_position_in_grid.x; out[i] = x[i] * weight[i % 16];"))
    parent_kernel = replace(_kernel("nested_parent"), source=(
        "uint i = thread_position_in_grid.x; float v = x[i]; "
        "out[i] = (v != v) ? v : metal::min(metal::max(v, 0.0f), 8.0f);"))
    return runner, {"child": (child, child_kernel), "parent": (parent, parent_kernel)}


def _bind(runner, definition):
    from autotuner.ladder.gates import LadderResult
    from autotuner.loop import RegionRun

    region, kernel = definition
    return runner._bind_and_promote(
        RegionRun(region=region), kernel,
        LadderResult("tentative_ship", None, {}, 1.0, 2.0, 1.0, 0.0, []),
    )


@pytest.mark.parametrize("order", [("child", "parent"), ("parent", "child")])
def test_real_nested_installs_compose_retrace_and_export(tmp_path, monkeypatch, order):
    import mlx.core as mx

    from autotuner.artifact.emit import check_apply_many, emit_artifact
    from autotuner.bind.verify import verify_retrace
    from autotuner_runtime.swap import ReplayWrapper

    runner, definitions = _real_runner(tmp_path, monkeypatch)
    try:
        for key in order:
            assert _bind(runner, definitions[key])
        assert set(runner.installed) == set(runner.emitted) == {"block"}
        assert not isinstance(runner.model.block.wrapped.proj, ReplayWrapper)
        assert not isinstance(runner.model.block.wrapped.out, ReplayWrapper)
        runner.tracer.install()
        for workload, tensors in runner.tensors.items():
            retrace, _ = runner.tracer.trace(runner.model, tensors)
            cuts = runner.cuts[workload]
            spans = sorted(cuts)
            checked = verify_retrace(runner.traces[workload], retrace, spans,
                                     [cuts[span] for span in spans])
            assert checked.ok, checked.reasons
            assert len([n for n in retrace.nodes if n.op == "custom_kernel"]) == 3
        runner.tracer.uninstall()
        clone = runner._copy_incumbent()
        cases = []
        for name, tensors in runner.tensors.items():
            expected = runner.baseline_model(*tensors)
            assert mx.array_equal(runner.model(*tensors), expected).item()
            assert mx.array_equal(clone(*tensors), expected).item()
            cases.append((name, tensors, [expected]))
        artifact = tmp_path / "artifact"
        emit_artifact(artifact, list(runner.installed["block"][2].values()),
                      list(runner.emitted.values()), runner.report,
                      validate=lambda staged: check_apply_many(staged, runner.manifest.model_path, cases))
        assert (artifact / "apply.py").exists()
    finally:
        runner.tracer.uninstall()
        assert runner.tracer.verify_restored() == []


@pytest.mark.parametrize("order", [("child", "parent"), ("parent", "child")])
def test_real_nested_rejection_restores_live_tree_and_records(tmp_path, monkeypatch, order):
    import mlx.core as mx

    from autotuner.loop import JobRunner
    from autotuner_runtime.swap import resolve_value

    runner, definitions = _real_runner(tmp_path, monkeypatch)
    try:
        assert _bind(runner, definitions[order[0]])
        prior = dict(runner.installed)
        emitted = dict(runner.emitted)
        cuts = {name: dict(values) for name, values in runner.cuts.items()}
        occupants = {path: resolve_value(runner.model, path) for path in runner.installed}
        monkeypatch.setattr(JobRunner, "_model_win", lambda self, result: False)
        assert not _bind(runner, definitions[order[1]])
        assert runner.installed == prior and runner.emitted == emitted and runner.cuts == cuts
        assert all(resolve_value(runner.model, path) is obj for path, obj in occupants.items())
        assert len(runner.report.accepted) == 1
        for tensors in runner.tensors.values():
            assert mx.array_equal(runner.model(*tensors), runner.baseline_model(*tensors)).item()
    finally:
        runner.tracer.uninstall()
        assert runner.tracer.verify_restored() == []
