"""Installing wins on one model, directly on the install path: a second
region after a first, several copies of one region inside one module, a
re-ship on the same span, the final whole-model check, and the artifact
loaded in a fresh process. No pricing and no judge: the ladder result is
handed in."""

import textwrap
from pathlib import Path

import mlx.core as mx
import pytest

from autotuner.ladder.gates import LadderResult
from autotuner.loop import JobRunner, RegionRun
from autotuner.measure.session import Session
from autotuner_runtime.kernels import KernelSpec

FIXTURES = Path(__file__).parent / "fixtures"
WIN = LadderResult("tentative_ship", None, {}, 1.0, 2.0, 1.0, 0.0, [])


def elementwise(kernel_id: str, body: str, inputs=("in0",)) -> KernelSpec:
    return KernelSpec(
        kernel_id=kernel_id, name=f"at_{kernel_id}", input_names=tuple(inputs),
        output_names=("out0",), source=f"uint i = thread_position_in_grid.x;\n{body}",
        grid=("in0.shape[0] * in0.shape[1]", "1", "1"), threadgroup=("256", "1", "1"),
        output_shapes=(("in0.shape[0]", "in0.shape[1]"),), output_dtypes=("float32",),
    )


@pytest.fixture
def runner(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(textwrap.dedent(f"""
        model: {FIXTURES / "planted_win.py"}
        workloads:
          - inputs: [{{shape: [64, 1024], dtype: float32}}]
            name: main
    """))
    r = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: None,
                  clock_pairs=4, refuse_degraded=False, session=Session(sleep=lambda s: None))
    r.load_model()
    r.trace_workloads()
    regions = r.build_regions()

    def region(*ops, start=None):
        return next(reg for reg in regions if tuple(reg.ops) == ops
                    and (start is None or reg.members[0].start_seq == start))

    r.region = region
    yield r
    r.tracer.uninstall()
    assert r.tracer.verify_restored() == []


def test_wins_compose_on_one_model(runner, tmp_path):
    x = runner.tensors["main"][0]
    before = runner.model(x)
    mx.eval(before)

    # NaN keeps NaN, as mx.maximum and mx.minimum do
    k_max = elementwise("rmax_h1", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);")
    k_min = elementwise("rmin_h1", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::min(in0[i], 8.0f);")
    assert runner._bind_and_promote(RegionRun(region=runner.region("mx.maximum")), k_max, WIN)
    # the second region in the same workload used to fail verification, because
    # the check expected only its own cut against the job-start recording
    assert runner._bind_and_promote(RegionRun(region=runner.region("mx.minimum")), k_min, WIN)
    assert set(runner.cuts["main"].values()) == {"rmax_h1", "rmin_h1"}

    # two copies of the residual add inside one module: one wrapper, two cuts
    add = runner.region("array.__add__")
    assert add.copies == 2 and len({m.scope_stack for m in add.members}) == 1
    k_add = elementwise("radd_h1", "out0[i] = in0[i] + in1[i % (uint)in0_shape[1]];", ("in0", "in1"))
    assert runner._bind_and_promote(RegionRun(region=add), k_add, WIN)
    assert len(runner.cuts["main"]) == 4
    assert sorted(runner.emitted["chain"].kernel_ids) == ["radd_h1", "radd_h1", "rmax_h1", "rmin_h1"]

    # a re-ship on the same span replaces the old kernel everywhere
    k_max2 = elementwise("rmax_h2", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);")
    assert runner._bind_and_promote(RegionRun(region=runner.region("mx.maximum")), k_max2, WIN)
    assert "rmax_h1" not in runner.emitted["chain"].kernel_ids
    assert set(runner.cuts["main"].values()) == {"rmax_h2", "rmin_h1", "radd_h1"}

    after = runner.model(x)
    mx.eval(after)
    assert mx.array_equal(before, after).item()  # elementwise kernels in the library's order

    runner._final_check()
    assert runner.final_ok and runner.report.final["passed"]
    art = runner.emit_artifact(tmp_path / "artifact")
    assert sorted(p.stem for p in (art / "kernels").glob("*.metal")) == ["radd_h1", "rmax_h2", "rmin_h1"]
    kinds = [row["kind"] for row in runner.log.rows()]
    assert kinds.count("shipped") == 4 and "final_e2e" in kinds and "artifact_checked" in kinds


def test_a_multi_op_cut_feeding_the_next_cut_verifies(runner):
    """A three-op cut whose output feeds the very next op, then that op as its
    own cut: the retrace must show two custom dispatches wired together."""
    x = runner.tensors["main"][0]
    before = runner.model(x)
    mx.eval(before)
    prefix = runner.region("array.__mul__", "array.__add__", "mx.maximum")
    fused = elementwise("rpre_h1", (
        "float y = in0[i] * 2.0f + in1[i % (uint)in0_shape[1]];\n"
        "out0[i] = (y != y) ? y : metal::max(y, 0.0f);"), ("in0", "in1"))
    assert runner._bind_and_promote(RegionRun(region=prefix), fused, WIN)
    times_x = runner.region("array.__mul__", start=prefix.members[0].end_seq + 1)
    k_mul = elementwise("rmul_h1", "out0[i] = in0[i] * in1[i];", ("in0", "in1"))
    assert runner._bind_and_promote(RegionRun(region=times_x), k_mul, WIN)
    after = runner.model(x)
    mx.eval(after)
    assert mx.array_equal(before, after).item()
