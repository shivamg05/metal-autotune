"""Work the model records but never runs. MLX is lazy: a result nothing
returns, keeps, evaluates or reads never executes. The record marks such
calls dead, and no region, price, floor or ladder job may count them."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from autotuner.ladder.gates import LadderJob, _validate
from autotuner.ladder.static_checks import RegionContract
from autotuner.loop import JobRunner
from autotuner.measure.clocks import chained_loop
from autotuner.measure.peaks import Peaks
from autotuner.measure.session import Session
from autotuner.regions.build import build_stretches
from autotuner.regions.roofline import step_floor
from autotuner.trace import Tracer


class DeadBranch(nn.Module):
    """One result is returned; one is only evaluated; one is dropped."""

    def __init__(self, evaluate):
        super().__init__()
        self.evaluate = evaluate

    def __call__(self, x):
        y = x * 2.0
        aside = mx.exp(y + 1.0)
        if self.evaluate:
            mx.eval(aside)
        dropped = mx.tanh(y - 1.0).sum()  # never used, never evaluated
        return y * 3.0


def record(model, x):
    tracer = Tracer()
    tracer.install()
    try:
        trace, _ = tracer.trace(model, [x])
    finally:
        tracer.uninstall()
    return trace


@pytest.mark.parametrize("evaluate", [False, True])
def test_unneeded_calls_are_dead_and_evaluated_ones_are_not(evaluate):
    trace = record(DeadBranch(evaluate), mx.ones((4, 8)))
    ops = {n.seq: n.op for n in trace.nodes}
    dead = sorted(ops[s] for s in trace.dead)
    # tanh and its sum are dead either way; the exp branch lives only when evaluated
    expected = ["array.__sub__", "array.sum", "mx.tanh"] + ([] if evaluate else ["array.__add__", "mx.exp"])
    assert dead == sorted(expected)
    stretches = build_stretches(trace, "main")
    assert not any(any(trace.nodes[s].seq in trace.dead for s in range(st.start_seq, st.end_seq + 1))
                   for st in stretches)
    assert all(st.output_ids for st in stretches)
    if evaluate:
        # the evaluated result is a live output of the stretch that made it
        exp_node = next(n for n in trace.nodes if n.op == "mx.exp")
        assert any(exp_node.out_arrays[0] in st.output_ids for st in stretches)


def test_the_step_floor_counts_only_live_launches():
    trace = record(DeadBranch(False), mx.ones((4, 8)))
    peaks = Peaks(bandwidth_gbps=100.0, flops_gflops={"float32": 1000.0}, launch_us=5.0)
    floor = step_floor(trace, peaks, 1.0)
    live = [n for n in trace.nodes if n.seq not in trace.dead]
    assert floor["launches"] == len(live) == 2  # x * 2, y * 3
    assert len(trace.dead) == 5
    # the time term prices the same live work as the flop count, never the
    # dead branch (a 500-token RecurrentGemma prefill carried 792 GFLOP of
    # never-evaluated lm_head, a fifth of its floor)
    from autotuner.regions.roofline import compute_time_ms
    assert floor["t_compute_ms"] == pytest.approx(compute_time_ms(live, peaks))
    assert compute_time_ms(trace.nodes, peaks) > floor["t_compute_ms"]


def test_a_generation_task_marks_the_prefill_head_dead(tmp_path):
    """mlx-lm processes the prompt for its cache alone and drops the logits:
    that call's final norm and output projection never run, so no region may
    contain them, and every region has a live output."""
    model = Path(__file__).parent / "fixtures" / "llama_cache_model.py"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {model}
baseline: plain
use_library_inference: true
workloads:
  - name: prompt
    context: 0
    inputs: [{{shape: [1, 6], dtype: int32, high: 64}}]
final_benchmark: {{steps: 1, pairs: 4, warmup_steps: 1}}
""")
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda _: None,
                       session=Session(sleep=lambda _: None))
    try:
        runner.load_model()
        runner.trace_workloads()
        trace = runner.traces["prompt"]
        dead_ops = [trace.nodes[s].op for s in sorted(trace.dead)]
        assert "array.__matmul__" in dead_ops or "mx.quantized_matmul" in dead_ops, dead_ops
        regions = runner.build_regions()
        assert all(m.output_ids for r in regions for m in r.members)
        assert runner.report.coverage["discovery"]["never_evaluated_ops"] == len(trace.dead) > 0
    finally:
        runner.tracer.uninstall()


def test_a_ladder_job_without_live_outputs_is_refused():
    from autotuner.ladder.gates import EvalSet
    from autotuner_runtime.kernels import KernelSpec
    kernel = KernelSpec("k", "k", ("a",), ("out0",), "out0[0] = a[0];", grid=("1", "1", "1"),
                        threadgroup=("1", "1", "1"), output_shapes=(("1",),), output_dtypes=("float32",))
    contract = RegionContract(input_names=("in0",), input_ranks=(1,), input_dtypes=("float32",),
                              output_names=(), output_ranks=(), output_dtypes=(), live_outputs=(),
                              input_shapes=((1,),), output_shapes=())
    job = LadderJob(kernel=kernel, contract=contract, assoc_tag="preserving", nodes_json="[]",
                    input_ids=(0,), output_ids=(), tolerances=None,
                    eval_sets=[EvalSet("main", ["i"], ["o"], t_library_ms=1.0, correctness_only=False, nodes_json="[]")])
    with pytest.raises(ValueError, match="no live outputs"):
        _validate(job)


def test_a_timed_pass_with_no_outputs_is_a_named_error():
    loop = chained_loop(lambda binds: [], [{0: mx.ones(4)}], 3, 0)
    with pytest.raises(ValueError, match="no outputs"):
        loop()
