"""Pricing (spec "Pricing and ranking", plan 5.7): boundary capture, the
region clock, shares, the floor.

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

from ..measure.clocks import compare
from ..measure.session import Session
from ..trace import Tracer
from ..trace.types import Trace
from ..trace.replay import replay
from .types import Region, Stretch

REGION_FLOOR_P = 0.02
ROOFLINE_HAS_ROOM = 1.2
CACHE_DEFEAT_BYTES = 128 * 1024 * 1024
MAX_TIMING_SETS = 16
CLOCK_TARGET_MS = 15.0
CLOCK_SAMPLES = 5
CLOCK_EST_ITERS = 10      # same amortizing estimate the ship clock uses
CLOCK_MIN_ITERS = 20      # enough passes that one sample's sync cost is a rounding error
CLOCK_MAX_ITERS = 2000
PRICE_PAIRS = 32          # shares are ratios: measured against the step, never divided


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


def _synthesize_like(arrays: Mapping[int, mx.array], seed: int) -> dict[int, mx.array]:
    """Timing-only input sets: matching shapes and dtypes, random values. Never
    used for correctness comparisons."""
    keys = mx.random.split(mx.random.key(seed), max(len(arrays), 1))
    out = {}
    for i, (aid, arr) in enumerate(sorted(arrays.items())):
        if arr.dtype in (mx.float16, mx.bfloat16, mx.float32):
            out[aid] = mx.random.normal(arr.shape, dtype=arr.dtype, key=keys[i])
        else:
            out[aid] = arr  # integer inputs (indices) keep real values
    mx.eval(list(out.values()))
    return out


@dataclass(frozen=True)
class RegionPrice:
    """One region copy's cost, measured against the step in one window."""

    share: float      # one pass as a fraction of one step: the drift-immune number
    ms: float         # one pass in milliseconds, for logs and the ladder's floor
    stability: float  # 0..1 agreement of the paired ratios behind `share`


def _looped_replay(
    session: Session,
    trace: Trace,
    stretch: Stretch,
    input_sets: list[dict[int, mx.array]],
    weight_bindings: dict[int, mx.array],
    target_ms: float,
):
    """The replay loop and how many passes one sample holds. Input sets rotate;
    synthesized sets are added if the rotated working set is too small to defeat
    the cache (capped, recorded by the caller)."""
    nodes = trace.nodes[stretch.start_seq:stretch.end_seq + 1]
    out_ids = list(stretch.output_ids) or [nodes[-1].out_arrays[0]]

    set_bytes = sum(a.nbytes for a in input_sets[0].values())
    sets = list(input_sets)
    seed = 0
    while set_bytes * len(sets) < CACHE_DEFEAT_BYTES and len(sets) < MAX_TIMING_SETS:
        sets.append({**sets[0], **_synthesize_like(sets[0], 7000 + seed)})
        seed += 1

    def one_pass(bindings):
        return list(replay(nodes, {**weight_bindings, **bindings}, out_ids).values())

    def est_pass_ms() -> float:
        return session.timed(
            lambda: [one_pass(sets[i % len(sets)]) for i in range(CLOCK_EST_ITERS)]
        ) / CLOCK_EST_ITERS * 1e3

    # A single evaluated pass is dominated by fixed submit-and-sync latency, so
    # sizing the loop from one picks a loop too short to amortize it back out.
    # The ship clock estimates from this same warm loop; if the two disagree,
    # every s_max built on this number is inflated by the gap.
    est_pass_ms()  # thrown away: Metal compile and the post-idle clock ramp
    t_est_ms = est_pass_ms()
    # The floor matters more than the target: a loop holding few passes keeps
    # one sample's fixed sync cost in the per-pass number, and how large that
    # cost looks depends on the chip's speed, which is the one thing a share
    # must not depend on.
    iters = int(target_ms / max(t_est_ms, 1e-3))
    iters = max(CLOCK_MIN_ITERS, min(iters, CLOCK_MAX_ITERS))

    def loop_fn():
        outs = []
        for i in range(iters):
            outs.append(one_pass(sets[i % len(sets)]))
        return outs

    return loop_fn, iters


def region_clock(
    session: Session,
    trace: Trace,
    stretch: Stretch,
    input_sets: list[dict[int, mx.array]],
    weight_bindings: dict[int, mx.array],
    target_ms: float = CLOCK_TARGET_MS,
) -> float:
    """Median time of one pass over the stretch, from a looped replay."""
    loop_fn, iters = _looped_replay(
        session, trace, stretch, input_sets, weight_bindings, target_ms)
    session.warm_until_stable(loop_fn)
    samples = []
    for _ in range(CLOCK_SAMPLES):
        session.fresh_chunk((loop_fn,))
        samples.append(session.timed(loop_fn))
    session.settle()
    return statistics.median(samples) / iters * 1e3


def region_share(
    session: Session,
    trace: Trace,
    stretch: Stretch,
    input_sets: list[dict[int, mx.array]],
    weight_bindings: dict[int, mx.array],
    step_fn,
    target_ms: float = CLOCK_TARGET_MS,
    pairs: int = PRICE_PAIRS,
) -> RegionPrice:
    """The region's cost as a fraction of one step, from a paired interleaved
    comparison against the step itself.

    A share is a ratio, so law 4 governs it: dividing a region clock by a step
    clock taken minutes earlier reports whatever the machine did in between. On
    the 8B decode run that produced shares summing to 586 ms against a 52 ms
    step, and the same region priced 44.9 ms in one run and 91.5 ms in the next.
    """
    loop_fn, iters = _looped_replay(
        session, trace, stretch, input_sets, weight_bindings, target_ms)
    comp = compare(session, step_fn, loop_fn, pairs=pairs)
    return RegionPrice(
        share=comp.median_ratio / iters,
        ms=statistics.median(comp.candidate_ms) / iters,
        stability=comp.stability,
    )


def price_region(
    region: Region,
    session: Session,
    traces: Mapping[str, Trace],
    input_sets_for: Mapping[tuple[str, int], list[dict[int, mx.array]]],
    weight_bindings: Mapping[str, dict[int, mx.array]],
    step_fns: Mapping[str, object],
    pairs: int = PRICE_PAIRS,
) -> None:
    """Fill t_orig_ms, t_rep_ms and p per workload. Copies with distinct
    boundary shapes price separately; identical-shape copies share one price.
    step_fns maps a workload to a callable that runs one step of the model, so
    each region's share is measured against the step rather than divided by it.
    input_sets_for is keyed by (workload, representative member start_seq)."""
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
                    step_fns[m.workload], pairs=pairs,
                )
        price = priced[shape_key]
        per_workload_ms[m.workload] = per_workload_ms.get(m.workload, 0.0) + price.ms
        per_workload_share[m.workload] = per_workload_share.get(m.workload, 0.0) + price.share
        region.t_rep_ms.setdefault(m.workload, price.ms)
        region.p_rep.setdefault(m.workload, price.share)
        region.stability.setdefault(m.workload, price.stability)
    for w, total in per_workload_ms.items():
        region.t_orig_ms[w] = total
        region.p[w] = per_workload_share[w]


def _boundary_shapes(trace: Trace, stretch: Stretch) -> tuple:
    specs = {}
    for node in trace.nodes[stretch.start_seq:stretch.end_seq + 1]:
        for aid, spec in zip(node.in_arrays, node.in_specs):
            specs[aid] = spec
        for aid, spec in zip(node.out_arrays, node.out_specs):
            specs[aid] = spec
    return tuple(specs.get(a) for a in stretch.input_ids + stretch.output_ids)
