"""Freeze semantics on synthetic call lists, no GPU."""

import pytest

from autotuner.trace.freeze import freeze
from autotuner.trace.types import Retention, TraceIncomplete, TraceNode


def node(seq, op, ins, outs, addr="model.layer"):
    return TraceNode(
        seq=seq, op=op,
        in_arrays=tuple(ins), out_arrays=tuple(outs),
        in_specs=tuple(((4, 4), "float32") for _ in ins),
        out_specs=tuple(((4, 4), "float32") for _ in outs),
        scalar_args={}, module_address=addr, position_in_module=0,
    )


def test_edges_and_liveness():
    # x(1) -> matmul -> h(10) -> relu -> y(11); h also read by a second op -> z(12)
    nodes = [
        node(0, "matmul", [1, 2], [10]),
        node(1, "maximum", [10], [11]),
        node(2, "add", [10, 11], [12]),
    ]
    t = freeze(nodes, inputs=[1], weights=[2], step_outputs=[12])
    assert t.edges[0] == (1, 2)
    assert t.edges[1] == (2,)
    assert t.liveness[10].kind is Retention.CONSUMED
    assert t.liveness[10].consumed_by == (1, 2)
    assert t.liveness[12].kind is Retention.STEP_OUTPUT
    assert next(n for n in t.nodes if 11 in n.out_arrays).op == "maximum"


def test_retained_wins_over_consumed_and_step_output():
    nodes = [node(0, "matmul", [1, 2], [10]), node(1, "add", [10], [11])]
    t = freeze(nodes, inputs=[1], weights=[2], step_outputs=[10, 11], retained=[10])
    assert t.liveness[10].kind is Retention.PYTHON_RETAINED
    assert t.liveness[10].consumed_by == (1,)
    assert 10 in t.step_outputs


def test_unknown_input_aborts_naming_the_call():
    nodes = [node(0, "matmul", [1, 99], [10])]
    with pytest.raises(TraceIncomplete, match=r"array 99 fed 'matmul' \(seq 0"):
        freeze(nodes, inputs=[1], weights=[], step_outputs=[10])


def test_out_of_order_producer_aborts():
    nodes = [node(0, "add", [10], [11]), node(1, "matmul", [1], [10])]
    with pytest.raises(TraceIncomplete, match="unwrapped"):
        freeze(nodes, inputs=[1], weights=[], step_outputs=[11])


def test_unproduced_step_output_aborts():
    nodes = [node(0, "matmul", [1, 2], [10])]
    with pytest.raises(TraceIncomplete, match="step output array 55"):
        freeze(nodes, inputs=[1], weights=[2], step_outputs=[55])


def test_step_output_may_be_workload_input_passthrough():
    # returning an input unchanged is odd but not an unwrapped entry point
    nodes = [node(0, "matmul", [1, 2], [10])]
    t = freeze(nodes, inputs=[1], weights=[2], step_outputs=[10, 1])
    assert t.step_outputs == (10, 1)
