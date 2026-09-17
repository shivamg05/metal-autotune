"""Copies must agree on their public boundary, not just their operations."""
from dataclasses import replace

import mlx.core as mx
import pytest

from autotuner.loop import JobRunner
from autotuner.manifest import BOUNDARY_INPUT_SETS
from autotuner.regions.build import _boundary
from autotuner.regions.fingerprint import fingerprint, group_copies
from autotuner.regions.store import BoundaryMismatch, BoundaryStore
from autotuner.regions.sweep import SweepDivergence, locate_span
from autotuner.regions.types import Region
from autotuner.trace.types import Trace, TraceNode, Liveness, Retention


def pair(n=4, offset=0, native=False):
    spec = ((n,), 'float32')
    nodes = tuple(TraceNode(i, op, (offset + i,), (offset + i + 1,),
                           (spec,), (spec,), {'args': (), 'kwargs': {}}, 'layer', i,
                           ('root', 'layer'), {'source': 'example'} if native and i == 0 else None)
                  for i, op in enumerate(('mx.abs', 'mx.square')))
    trace = Trace(nodes, {0: (1,)}, (offset + 1, offset + 2), frozenset(),
                  frozenset((offset,)), {
                      offset + 1: Liveness(Retention.STEP_OUTPUT, (1,)),
                      offset + 2: Liveness(Retention.STEP_OUTPUT, ()),
                  })
    return trace, _boundary(trace, 'w', 0, 1)


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('change', ['drop', 'different_value', 'reorder'])
def test_output_roles_distinguish_otherwise_identical_regions(native, change):
    trace, full = pair(native=native)
    if change == 'drop':
        a, b = full, replace(full, output_ids=(2,))
    elif change == 'different_value':
        a, b = replace(full, output_ids=(1,)), replace(full, output_ids=(2,))
    else:
        a, b = full, replace(full, output_ids=(2, 1))
    assert fingerprint(trace, a) != fingerprint(trace, b)


def test_input_order_is_part_of_contract():
    trace, full = pair()
    node = replace(trace.nodes[0], in_arrays=(0, 10), in_specs=trace.nodes[0].in_specs * 2)
    trace = replace(trace, nodes=(node, trace.nodes[1]))
    a = replace(full, input_ids=(0, 10))
    assert fingerprint(trace, a) != fingerprint(trace, replace(a, input_ids=(10, 0)))


def test_compatible_sizes_and_array_ids_still_group():
    a, sa = pair(4)
    b, sb = pair(16, offset=100)
    regions = group_copies({'small': a, 'large': b},
                          {'small': [replace(sa, workload='small')],
                           'large': [replace(sb, workload='large')]})
    assert len(regions) == 1 and regions[0].copies == 2


def test_sweep_refuses_changed_live_outputs():
    trace, full = pair()
    retrace = replace(trace, step_outputs=(2,))
    with pytest.raises(SweepDivergence, match='input/output roles'):
        locate_span(trace, full, retrace, 'w')
    assert locate_span(trace, full, pair(16)[0], 'w').output_ids == (1, 2)


def prepared(tmp_path):
    trace, span = pair()
    region = Region('target', ('mx.abs', 'mx.square'), [span])
    runner = object.__new__(JobRunner)
    runner.traces = {'w': trace}
    runner.store = BoundaryStore(tmp_path)
    runner._sweep_instances = lambda r: []
    for si in range(BOUNDARY_INPUT_SETS):
        for kind, ids in (('inputs', span.input_ids), ('outputs', span.output_ids)):
            runner.store.save(region.fingerprint, 'w', si, kind,
                              {i: mx.ones((4,), stream=mx.cpu) for i in ids})
    return runner, region


@pytest.mark.parametrize('damage', ['missing_file', 'missing_output', 'extra_output', 'truncated', 'missing_set'])
def test_preflight_rejects_incomplete_saved_data(tmp_path, damage):
    runner, region = prepared(tmp_path)
    runner._validate_boundaries([region])
    path = runner.store._path('target', 'w', BOUNDARY_INPUT_SETS - 1, 'outputs')
    if damage == 'missing_file':
        path.unlink()
    elif damage == 'missing_set':
        runner.store._path('target', 'w', 1, 'inputs').unlink()
    elif damage == 'truncated':
        path.write_bytes(path.read_bytes()[:-1])
    else:
        ids = (2,) if damage == 'missing_output' else (1, 2, 3)
        runner.store.save('target', 'w', BOUNDARY_INPUT_SETS - 1, 'outputs',
                          {i: mx.ones((4,), stream=mx.cpu) for i in ids})
    with pytest.raises(BoundaryMismatch, match='Saved boundary data mismatch'):
        runner._validate_boundaries([region])


def test_preflight_rejects_bad_group_before_reading_files(tmp_path):
    runner, region = prepared(tmp_path)
    region.members.append(replace(region.members[0], output_ids=(2,)))
    with pytest.raises(BoundaryMismatch, match='incompatible input/output roles'):
        runner._validate_boundaries([region])


def test_preflight_checks_sweep_files_too(tmp_path):
    runner, region = prepared(tmp_path)
    runner.sweep_traces = {'w@L=16': runner.traces['w']}
    runner._sweep_instances = lambda r: [('w@L=16', region.members[0], {})]
    with pytest.raises(BoundaryMismatch, match='w%40L%3D16'):
        runner._validate_boundaries([region])


def test_preflight_rejects_sweep_with_changed_output_roles(tmp_path):
    runner, region = prepared(tmp_path)
    runner.sweep_traces = {'w@L=16': runner.traces['w']}
    span = replace(region.members[0], output_ids=(2,))
    runner._sweep_instances = lambda r: [('w@L=16', span, {})]
    with pytest.raises(BoundaryMismatch, match='input/output roles at sweep'):
        runner._validate_boundaries([region])


def test_actual_trace_separates_final_copy_and_replays_saved_outputs(tmp_path):
    from autotuner.trace import Tracer
    from autotuner.regions.price import capture_boundaries, capture_instances
    from autotuner.trace.replay import replay
    from autotuner.ladder.child import _project_ids, _ordered
    from autotuner.regions.store import load_set
    def model(x):
        a = x + x
        b = mx.abs(a)
        c = b + b
        d = mx.abs(c)
        return a, d
    tr = Tracer()
    tr.install()
    try:
        x = mx.array([-1., 2., -3., 4.])
        trace, _ = tr.trace(model, [x])
        spans = [_boundary(trace, 'w', 0, 1), _boundary(trace, 'w', 2, 3)]
        assert [len(s.output_ids) for s in spans] == [2, 1]
        regions = group_copies({'w': trace}, {'w': spans})
        assert len(regions) == 2
        wanted = {i for s in spans for i in s.input_ids + s.output_ids}
        arrays = capture_boundaries(tr, model, [x], trace, wanted)
    finally:
        tr.uninstall()
    store = BoundaryStore(tmp_path)
    for r in regions:
        label, s, _ = capture_instances(r, {'w': trace})[0]
        nodes = trace.nodes[s.start_seq:s.end_seq + 1]
        path = store.save(r.fingerprint, label, 0, 'outputs', {i: arrays[i] for i in s.output_ids})
        store.validate(r.fingerprint, label, 0, 'outputs', s.output_ids)
        projected = _project_ids(nodes, nodes, s.output_ids)
        refs = _ordered(load_set(path), projected, str(path))
        result = replay(nodes, {i: arrays[i] for i in s.input_ids}, projected)
        for aid, ref in zip(projected, refs, strict=True):
            assert mx.array_equal(result[aid], ref).item()
