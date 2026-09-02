"""Pricing: boundary capture, the region clock, shares, the floor.

Capture is a recorded pass with retention: re-run the model on the workload
tensors, align the new pass node-for-node with the priced trace (fixed inputs
and seeds make the two passes identical; any divergence aborts naming the
seq), hold the arrays at the region's edges, evaluate, keep.

The region clock replays a region's recorded ops on the saved inputs, looped
to tens of milliseconds per sample, rotating input sets so the data cannot
just sit in the GPU cache, every iteration's outputs kept in the final eval.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Mapping

import mlx.core as mx

from ..measure.clocks import (CLOCK_TARGET_MS, chained_loop, compare, link_input, link_loop,
                              loop_iterations, timing_sets)
from ..measure.probe import array_specs, floor_from, stream_probe
from ..measure.session import Session
from ..trace import Tracer
from ..trace.types import Trace
from ..trace.replay import replay
from .types import Region, Stretch

PRICE_PAIRS = 8           # a share steers ranking and the floor; the ship clock decides wins


class CaptureMismatch(RuntimeError):
    """The capture pass diverged from the priced trace: the model is not
    deterministic under fixed inputs and seeds."""


def align_ids(priced: Trace, capture_nodes, capture_inputs, capture_weight_paths) -> dict[int, int]:
    """array_id map from the priced trace into a capture pass, by node
    position. Verifies (op, address, arity) per node."""
    if len(priced.nodes) != len(capture_nodes):
        raise CaptureMismatch(
            f"capture pass recorded {len(capture_nodes)} calls, priced trace has "
            f"{len(priced.nodes)}; the model diverged under fixed inputs"
        )
    mapping: dict[int, int] = {}
    for a, b in zip(priced.nodes, capture_nodes):
        if a.op != b.op or a.module_address != b.module_address or \
           len(a.in_arrays) != len(b.in_arrays) or len(a.out_arrays) != len(b.out_arrays):
            raise CaptureMismatch(
                f"capture pass diverged at seq {a.seq}: priced {a.op!r} at "
                f"{a.module_address!r}, capture {b.op!r} at {b.module_address!r}"
            )
        for x, y in zip(a.in_arrays, b.in_arrays):
            mapping.setdefault(x, y)
        for x, y in zip(a.out_arrays, b.out_arrays):
            mapping[x] = y
    by_path = {p: aid for aid, p in capture_weight_paths.items()}
    for aid, path in priced.weight_paths.items():
        if path in by_path:
            mapping.setdefault(aid, by_path[path])
    for x, y in zip(sorted(priced.inputs), sorted(capture_inputs)):
        mapping.setdefault(x, y)
    return mapping


def capture_boundaries(
    tracer: Tracer,
    model,
    tensors: list[mx.array],
    priced: Trace,
    ids: set[int],
) -> dict[int, mx.array]:
    """One capture pass: returns evaluated arrays for the requested priced-trace
    array_ids. Extra passes are off-clock and free; memory batching is the
    caller's concern."""
    rec = tracer.recorder
    paths = tracer.patcher.wrap_model(model)
    rec.arm(model, list(tensors), paths)
    try:
        rec.step(model, tuple(tensors))
    finally:
        rec.disarm()
    try:
        mapping = align_ids(priced, rec.nodes, rec.inputs, rec.weight_paths)
        missing = [a for a in ids if a not in mapping]
        if missing:
            raise CaptureMismatch(f"no capture-side arrays for ids {missing[:5]}")
        got = rec.arrays_for([mapping[a] for a in ids])
        arrays = {a: got[mapping[a]] for a in ids}
        mx.eval(list(arrays.values()))
        return arrays
    finally:
        rec.freeze_pass()


@dataclass(frozen=True)
class RegionPrice:
    """One region copy's cost, measured against the step in one window."""

    share: float      # one pass as a fraction of one step: the drift-immune number
    ms: float         # one pass in milliseconds, for logs and the ladder's floor
    stability: float  # 0..1 agreement of the paired ratios behind `share`
    floor_ms: float   # the one-launch stream probe over the same boundary, in ms's frame


def _looped_replay(
    session: Session,
    trace: Trace,
    stretch: Stretch,
    input_sets: list[dict[int, mx.array]],
    weight_bindings: dict[int, mx.array],
    target_ms: float,
    baseline: str = "plain",
):
    """The replay loop, how many passes one sample holds, the input the
    chain link rides on, and the sets the loop rotates: a cache-defeating
    working set, the passes chained so they cannot overlap. Under a compiled
    baseline the replay runs as one compiled graph, which is what the model
    itself would do with these ops."""
    nodes = trace.nodes[stretch.start_seq:stretch.end_seq + 1]
    out_ids = list(stretch.output_ids) or [nodes[-1].out_arrays[0]]
    sets = timing_sets(input_sets)
    link_id = link_input(sets[0], set(weight_bindings) | set(trace.weights))

    if baseline == "compiled":
        ids = sorted(sets[0])
        compiled = mx.compile(lambda *arrays: list(
            replay(nodes, {**weight_bindings, **dict(zip(ids, arrays))}, out_ids).values()))
        mx.eval(compiled(*[sets[0][a] for a in ids]))  # compile before anything is timed

        def one_pass(bindings):
            return compiled(*[bindings[a] for a in ids])
    else:
        def one_pass(bindings):
            return list(replay(nodes, {**weight_bindings, **bindings}, out_ids).values())

    iters = loop_iterations(session.timed, lambda n: chained_loop(one_pass, sets, n, link_id), target_ms)
    return chained_loop(one_pass, sets, iters, link_id), iters, link_id, sets


def _probe_pass(trace: Trace, stretch: Stretch, binds, weight_bindings):
    """The floor probe over this region's boundary: its inputs in order, its
    recorded output shapes."""
    nodes = trace.nodes[stretch.start_seq:stretch.end_seq + 1]
    out_ids = list(stretch.output_ids) or [nodes[-1].out_arrays[0]]
    specs = trace.span_specs(stretch.start_seq, stretch.end_seq)
    merged = {**weight_bindings, **binds}
    probe = stream_probe(array_specs([merged[a] for a in stretch.input_ids]),
                         [specs[a] for a in out_ids])

    def one_pass(bindings):
        merged = {**weight_bindings, **bindings}
        return probe([merged[a] for a in stretch.input_ids])
    return one_pass


def region_share(
    session: Session,
    trace: Trace,
    stretch: Stretch,
    input_sets: list[dict[int, mx.array]],
    weight_bindings: dict[int, mx.array],
    step_fn,
    target_ms: float = CLOCK_TARGET_MS,
    pairs: int = PRICE_PAIRS,
    warm_step: bool = True,
    baseline: str = "plain",
) -> RegionPrice:
    """The region's cost as a fraction of one step, from a paired interleaved
    comparison against the step itself.

    A share is a ratio, and dividing a region clock by a step clock taken
    minutes earlier reports whatever the machine did in between. On
    the 8B decode run that produced shares summing to 586 ms against a 52 ms
    step, and the same region priced 44.9 ms in one run and 91.5 ms in the next.
    """
    loop_fn, iters, link_id, sets = _looped_replay(
        session, trace, stretch, input_sets, weight_bindings, target_ms, baseline)
    comp = compare(session, step_fn, loop_fn, pairs=pairs, warm_baseline=warm_step)
    region_ms = statistics.median(comp.candidate_ms) / iters
    link_loop_ms = 0.0
    if link_id is not None:
        # the chain link's own cost comes out through a paired comparison
        # against the same loop around a pass that only hands its input back
        net = compare(session, link_loop(sets, iters, link_id), loop_fn, pairs=pairs)
        region_ms = max(-net.median_delta_ms / iters, 0.0)
        link_loop_ms = net.median_baseline_ms
    # the floor beside the region, in one window: the same loop around one
    # launch that streams the boundary, so region over floor is a paired ratio
    probe_loop = chained_loop(_probe_pass(trace, stretch, sets[0], weight_bindings),
                              sets, iters, link_id)
    floor = compare(session, probe_loop, loop_fn, pairs=pairs)
    return RegionPrice(
        share=region_ms / comp.median_baseline_ms,
        ms=region_ms,
        stability=comp.stability,
        floor_ms=floor_from(floor, link_loop_ms, iters, region_ms),
    )


def price_region(
    region: Region,
    session: Session,
    traces: Mapping[str, Trace],
    input_sets_for: Mapping[tuple[str, int], list[dict[int, mx.array]]],
    weight_bindings: Mapping[str, dict[int, mx.array]],
    step_fns: Mapping[str, object],
    baseline: str = "plain",
    pairs: int = PRICE_PAIRS,
    warmed_steps: set[str] | None = None,
) -> None:
    """Fill t_orig_ms, t_rep_ms and p per workload. Copies with distinct
    boundary shapes price separately; identical-shape copies share one price.
    step_fns maps a workload to a callable that runs one step of the model, so
    each region's share is measured against the step rather than divided by it.
    input_sets_for is keyed by (workload, representative member start_seq).
    warmed_steps remembers which workloads' steps are already warm across
    regions, so a slow step is warmed once per job, not once per region."""
    warmed = warmed_steps if warmed_steps is not None else set()
    per_workload_ms: dict[str, float] = {}
    per_workload_share: dict[str, float] = {}
    priced: dict[tuple, RegionPrice] = {}
    for m in region.members:
        trace = traces[m.workload]
        shape_key = (m.workload, _boundary_shapes(trace, m))
        if shape_key not in priced:
            sets = input_sets_for.get((m.workload, m.start_seq))
            if sets is None:
                # only representatives get captured sets; a same-workload copy
                # at another shape reuses the representative's price
                fallback = [v for (w, _), v in priced.items() if w == m.workload]
                if not fallback:
                    continue
                priced[shape_key] = fallback[0]
            else:
                priced[shape_key] = region_share(
                    session, trace, m, sets, weight_bindings[m.workload],
                    step_fns[m.workload], pairs=pairs, warm_step=m.workload not in warmed,
                    baseline=baseline,
                )
                warmed.add(m.workload)
        price = priced[shape_key]
        per_workload_ms[m.workload] = per_workload_ms.get(m.workload, 0.0) + price.ms
        per_workload_share[m.workload] = per_workload_share.get(m.workload, 0.0) + price.share
        region.t_rep_ms.setdefault(m.workload, price.ms)
        region.t_floor_ms.setdefault(m.workload, price.floor_ms)
        region.p_rep.setdefault(m.workload, price.share)
        region.stability.setdefault(m.workload, price.stability)
    for w, total in per_workload_ms.items():
        region.t_orig_ms[w] = total
        region.p[w] = per_workload_share[w]


def _boundary_shapes(trace: Trace, stretch: Stretch) -> tuple:
    specs = trace.span_specs(stretch.start_seq, stretch.end_seq)
    return tuple(specs.get(a) for a in stretch.input_ids + stretch.output_ids)
