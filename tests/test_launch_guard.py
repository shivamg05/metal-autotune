"""CPU-only resource checks. No kernel is compiled, launched, or allocated."""

from dataclasses import replace

import pytest

from autotuner.ladder.static_checks import (
    RegionContract, SINGLE_GROUP_ELEMENT_LIMIT, check, launch_resource_failure,
)
from autotuner.regions.types import Stretch
from autotuner.scaffold import NoScaffold, lower_naive
from autotuner.trace.recorder import ArrayRef
from autotuner.trace.types import Trace, TraceNode
from autotuner_runtime.kernels import KernelSpec


def kernel(**changes):
    base = KernelSpec(
        kernel_id="guard_test", name="guard_test", input_names=("in0",),
        output_names=("out0",), source="out0[0] = in0[0];",
        grid=("128", "1", "1"), threadgroup=("128", "1", "1"),
        output_shapes=(("in0.shape[0]",),), output_dtypes=("float32",),
    )
    return replace(base, **changes)


def test_restarted_flux_candidate_is_rejected_from_its_launch_metadata():
    # rc5d292_scafix launched immediately before the September 5 WindowServer
    # watchdog. Preserve its real IO and dispatch here, never its execution.
    shapes = ((1, 1, 3072), (1, 512, 3072), (1, 512, 3072), (1, 1, 3072),
              (1, 1, 3072), (27648, 384), (27648, 48), (27648, 48),
              (128,), (128,), (512, 64), (512, 64))
    outputs = ((1, 512, 3072), (1, 512, 18432), (1, 512, 3072),
               (1, 24, 512, 128), (1, 24, 512, 64, 2), (1572864,), (14155776,))
    names = tuple(f"in{i}" for i in range(12))
    spec = kernel(input_names=names,
                  output_names=("out0", "out1", "out2", "out3", "out4", "tmp0", "tmp1"),
                  output_shapes=tuple(tuple(map(str, s)) for s in outputs),
                  output_dtypes=("bfloat16",) * 4 + ("float32", "bfloat16", "bfloat16"))
    contract = RegionContract(names, tuple(map(len, shapes)), ("bfloat16",) * 12,
                              spec.output_names[:5], tuple(map(len, outputs[:5])),
                              spec.output_dtypes[:5], spec.output_names[:5],
                              input_shapes=shapes, output_shapes=outputs[:5])
    failures = check(spec, contract)
    assert [f.check for f in failures] == ["serial_launch"]
    assert "1,572,864" in failures[0].detail


@pytest.mark.parametrize("slot", ["input", "output", "scratch"])
def test_every_buffer_role_counts(slot):
    large = SINGLE_GROUP_ELEMENT_LIMIT + 1
    spec = kernel()
    shapes = ((large if slot == "input" else 16,),)
    if slot == "output":
        spec = replace(spec, output_shapes=((str(large),),))
    if slot == "scratch":
        spec = replace(spec, output_names=("out0", "tmp0"),
                       output_shapes=(("16",), (str(large),)),
                       output_dtypes=("float32", "float32"))
    assert launch_resource_failure(spec, shapes).check == "serial_launch"


def test_small_serial_and_large_tiled_launches_remain_available():
    assert launch_resource_failure(kernel(), ((1024,),)) is None
    tiled = kernel(grid=("in0.shape[0]", "1", "1"))
    assert launch_resource_failure(tiled, ((1 << 26,),)) is None
    # Group count is axis-wise ceil division, not total threads / group size.
    assert launch_resource_failure(kernel(grid=("128", "2", "1")), ((1 << 26,),)) is None


@pytest.mark.parametrize("inverted", [True, False])
def test_flux_fallback_polarity_is_checked_before_spawning_worker(monkeypatch, inverted):
    from types import SimpleNamespace
    from autotuner.ladder import gates

    # The failed FLUX proposal used a supported-shape predicate as a fallback.
    predicate = ("in0.ndim == 3 and in1.shape[0] % 64 == 0 and "
                 "in0.shape[1] % 64 == 0 and in0.shape[2] % 32 == 0")
    shapes = ((1, 256, 3072), (9216, 384), (9216, 48), (9216, 48))
    names = tuple(f"in{i}" for i in range(4))
    spec = kernel(input_names=names, grid=("4608", "8", "2"),
                  threadgroup=("32", "2", "2"),
                  output_shapes=(("1", "256", "9216"),),
                  fallback_predicate=predicate if inverted else f"not ({predicate})")
    contract = RegionContract(names, (3, 2, 2, 2), ("float32",) * 4,
                              ("out0",), (3,), ("float32",), ("out0",),
                              input_shapes=shapes, output_shapes=((1, 256, 9216),))
    failures = check(spec, contract)
    if not inverted:
        assert failures == []
        return
    assert [f.check for f in failures] == ["fallback_on_primary"]
    monkeypatch.setattr(gates, "_validate", lambda job: None)
    monkeypatch.setattr(gates, "run_job", lambda *a, **kw: pytest.fail("worker must not start"))
    result = gates.run_ladder(SimpleNamespace(kernel=spec, contract=contract))
    assert result.outcome == "failed" and result.failed_gate == "static"
    assert "true means run the original library" in result.detail["failures"][0]["detail"]


def test_fallback_does_not_launch_the_oversized_custom_dispatch():
    spec = kernel(fallback_predicate="in0.shape[0] > 1024")
    assert launch_resource_failure(spec, ((1 << 26,),)) is None


@pytest.mark.parametrize("changes,check_name", [
    ({"grid": ("0", "1", "1")}, "launch_extent"),
    ({"grid": ("-1", "1", "1")}, "launch_extent"),
    ({"grid": ("True", "1", "1")}, "launch_extent"),
    ({"threadgroup": ("128", "16", "1")}, "launch_extent"),
    ({"threadgroup": ("0", "1", "1")}, "launch_extent"),
    ({"threadgroup": ("128", "1")}, "launch_arity"),
    ({"output_shapes": (("-1",),)}, "output_extent"),
    ({"output_shapes": (("True",),)}, "output_extent"),
    ({"grid": ("1 // 0", "1", "1")}, "launch_grammar"),
])
def test_invalid_extents_fail_before_dispatch(changes, check_name):
    assert launch_resource_failure(kernel(**changes), ((16,),)).check == check_name


def matmul_transpose_trace(m, k, n):
    nodes = (
        TraceNode(0, "mx.matmul", (0, 1), (2,),
                  (((m, k), "float32"), ((k, n), "float32")), (((m, n), "float32"),),
                  {"args": (ArrayRef(0), ArrayRef(1)), "kwargs": {}}, "", 0),
        TraceNode(1, "array.T", (2,), (3,), (((m, n), "float32"),), (((n, m), "float32"),),
                  {"args": (ArrayRef(0),), "kwargs": {}}, "", 0),
    )
    trace = Trace(nodes, {}, (3,), frozenset({1}), frozenset({0}), {})
    return trace, Stretch("test", 0, 1, (0, 1), (3,), ())


def test_naive_scaffold_refuses_large_serial_compute_even_with_small_buffers():
    # Every buffer is below 1M elements, but the serial dot products exceed
    # 33M work units. The output transpose makes row-local staging impossible.
    trace, span = matmul_transpose_trace(256, 512, 256)
    with pytest.raises(NoScaffold, match="scalar work units") as exc:
        lower_naive(trace, span)
    assert exc.value.reason == "serial_launch"


def test_naive_scaffold_checks_later_shapes_without_allocating_tensors():
    trace, span = matmul_transpose_trace(4, 64, 64)
    assert lower_naive(trace, span) is not None
    with pytest.raises(NoScaffold, match="instance 1") as exc:
        lower_naive(trace, span, [((32768, 64), (64, 64))])
    assert exc.value.reason == "serial_launch"


def test_naive_work_estimate_checks_later_shapes_too():
    trace, span = matmul_transpose_trace(4, 512, 256)
    assert lower_naive(trace, span) is not None
    with pytest.raises(NoScaffold, match="instance 1.*scalar work units"):
        lower_naive(trace, span, [((256, 512), (512, 256))])
