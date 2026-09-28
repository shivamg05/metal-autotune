"""One kernel at many sizes of one dimension, inside one sandbox child.

The parent serves a shipped kernel at other sizes only where this job found
it right (phase "validate": shader validation on, outputs poisoned, three
runs bitwise equal to each other and to the recorded operations replayed at
that size) and faster (phase "score": the region clock's paired loop and
margin at each size of a grid).

Inputs at a size are the recorded boundary inputs cut or repeated along the
dimensions that change with it, so values stay the model's own.
"""

from __future__ import annotations

import dataclasses
import json
import math
import statistics
import time
from dataclasses import dataclass

import mlx.core as mx

from autotuner.measure.clocks import (CLOCK_TARGET_MS, chained_loop, comparison_from_samples,
                                      link_input, loop_iterations, paired_means, sample_group,
                                      timing_sets)
from autotuner.measure.session import Session
from autotuner.regions.store import load_set
from autotuner.sandbox.poison import saturate_pool
from autotuner.sandbox.protocol import Verdict
from autotuner.sandbox.watchdog import gpu_window
from autotuner.trace.replay import prepare_replay, replay
from autotuner.trace.serialize import nodes_from_json
from autotuner_runtime.exact import bitwise_equal
from autotuner_runtime.kernels import KernelSpec, try_sized

RUNS = 3  # as the determinism gate


@dataclass(frozen=True)
class SizesSpec:
    kernel: dict                      # KernelSpec fields
    nodes_json: str                   # the region's recorded operations
    input_ids: tuple[int, ...]        # kernel input order
    output_ids: tuple[int, ...]
    inputs_path: str                  # recorded boundary inputs at the primary size
    shapes: list                      # per input [[dim or [a, b]], dtype]: a * n + b
    sizes: tuple[int, ...]
    phase: str                        # "validate" | "score"
    weight_inputs: tuple[bool, ...] = ()
    clock_pairs: int = 8
    defer_cooling: bool = False
    kind: str = "sizes"

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @staticmethod
    def from_json(text: str) -> "SizesSpec":
        d = json.loads(text)
        if d.get("kind") != "sizes":
            raise ValueError(f"not a sizes spec: kind={d.get('kind')!r}")
        for key in ("input_ids", "output_ids", "sizes", "weight_inputs"):
            d[key] = tuple(d.get(key, ()))
        return SizesSpec(**d)


def shape_at(template, n: int) -> list:
    return [[[d if isinstance(d, int) else d[0] * n + d[1] for d in dims], dtype] for dims, dtype in template]


def resize(a: mx.array, shape) -> mx.array:
    """The recorded array cut, or repeated then cut, to shape."""
    for axis, (have, want) in enumerate(zip(a.shape, shape)):
        if want > have:
            a = mx.concatenate([a] * math.ceil(want / have), axis=axis)
        if want != a.shape[axis]:
            a = mx.take(a, mx.arange(want), axis=axis)
    return a


def evaluate_sizes(spec: SizesSpec) -> Verdict:
    """The correctness sweep takes no cooling pauses: those keep the chip's
    temperature steady for timing, and nothing here is timed. The clock phase
    paces like every other clock."""
    session = Session()
    started = time.perf_counter()
    try:
        return _evaluate(spec, session)
    finally:
        mx.synchronize()
        if spec.phase == "score":
            session._debt_s += max(0.0, time.perf_counter() - started - session.idled_s - session.work_s)
            session.settle()


def _evaluate(spec: SizesSpec, session: Session) -> Verdict:
    kspec = KernelSpec.from_json(json.dumps(spec.kernel))
    nodes = nodes_from_json(spec.nodes_json)
    recorded = load_set(spec.inputs_path)
    n_out = len(spec.output_ids)
    poison = 0.0 if kspec.atomic_outputs else float("nan")
    results: dict[str, object] = {}

    def binds_at(n):
        binds = {aid: resize(recorded[aid], shape) for aid, (shape, _dtype)
                 in zip(spec.input_ids, shape_at(spec.shapes, n))}
        mx.eval(list(binds.values()))
        return binds

    if spec.phase == "validate":
        largest = binds_at(max(spec.sizes))
        saturate_pool(a.nbytes for a in _reference(nodes, largest, spec.output_ids))
        for n in spec.sizes:
            binds = binds_at(n)
            inputs = [binds[i] for i in spec.input_ids]
            with gpu_window():
                reference = _reference(nodes, binds, spec.output_ids)
                runs = []
                for _ in range(RUNS):
                    outs = try_sized(kspec, inputs, init_value=poison)
                    if outs is None:
                        break
                    outs = outs[:n_out]
                    mx.eval(outs)
                    runs.append(outs)
            if not runs:
                results[str(n)] = "the kernel's own fallback covers this size"
            elif not all(bitwise_equal(a, b) for a, b in zip(runs[0], reference, strict=True)):
                results[str(n)] = "differs from the recorded operations"
            elif not all(bitwise_equal(a, b) for run in runs[1:] for a, b in zip(runs[0], run)):
                results[str(n)] = "differs between repeated runs"
            else:
                results[str(n)] = None
        return Verdict(True, None, ("sizes",), {"results": results})

    weight_ids = {i for i, w in zip(spec.input_ids, spec.weight_inputs) if w}
    for n in spec.sizes:
        binds = binds_at(n)
        sets = timing_sets([binds])
        link = link_input(binds, weight_ids)
        library = prepare_replay(nodes, binds, spec.output_ids)
        kernel = lambda b: try_sized(kspec, [b[i] for i in spec.input_ids])
        with gpu_window():
            iters = loop_iterations(session.timed, lambda k: chained_loop(library, sets, k, link), CLOCK_TARGET_MS)
        arms = {"library": chained_loop(library, sets, iters, link),
                "candidate": chained_loop(kernel, sets, iters, link)}
        rows = {k: paired_means(v) for k, v in sample_group(session, arms, pairs=spec.clock_pairs).items()}
        comp = comparison_from_samples([v / iters for v in rows["library"]],
                                       [v / iters for v in rows["candidate"]])
        library_ms = comp.median_baseline_ms
        margin = max(0.01 * library_ms, 3.0 * comp.sigma_ms)
        results[str(n)] = {"library_ms": library_ms, "kernel_ms": statistics.median(rows["candidate"]) / iters,
                           "win": bool(comp.median_delta_ms > margin),
                           "loss": bool(comp.median_delta_ms < -margin)}
    return Verdict(True, None, ("sizes",), {"results": results})


def _reference(nodes, binds, output_ids) -> list[mx.array]:
    res = replay(nodes, binds, output_ids)
    outs = [res[i] for i in output_ids]
    mx.eval(outs)
    return outs
