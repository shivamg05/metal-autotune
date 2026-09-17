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
from dataclasses import dataclass, field
from typing import Mapping

import mlx.core as mx

from ..measure.clocks import (CLOCK_TARGET_MS, chained_loop, link_input, link_loop,
                              loop_iterations, sample_group, timing_sets)
from ..measure.probe import array_specs, compute_probe, stream_probe
from ..measure.session import Session
from ..trace import Tracer
from ..trace.types import Trace
from ..trace.replay import compile_replay, prepare_replay
from .roofline import compute_dtype
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
    rec.snapshot_seqs = {n.seq for n in priced.nodes if set(n.out_arrays) & set(ids)}
    try:
        rec.arm(model, list(tensors), paths)
        try:
            rec.step(model, tuple(tensors))
        finally:
            rec.disarm()
        mapping = align_ids(priced, rec.nodes, rec.inputs, rec.weight_paths)
        missing = [a for a in ids if a not in mapping]
        if missing:
            raise CaptureMismatch(f"no capture-side arrays for ids {missing[:5]}")
        got = rec.arrays_for([mapping[a] for a in ids])
        arrays = {a: got[mapping[a]] for a in ids}
        mx.eval(list(arrays.values()))
        return arrays
    finally:
        rec.snapshot_seqs = None
        rec.freeze_pass()


@dataclass(frozen=True)
class RegionPrice:
    """One region copy's cost, measured against the step in one window."""

    share: float      # one pass as a fraction of one step: the drift-immune number
    ms: float         # one pass in milliseconds, for logs and the ladder's floor
    stability: float  # 0..1 agreement of the paired ratios behind `share`
    floor_ms: float   # the one-launch stream probe over the same boundary, in ms's frame
    gflops: dict[str, float] = field(default_factory=dict)  # the arithmetic ceiling per dtype, from the same window


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
        one_pass = compile_replay(nodes, sets[0], out_ids, weight_bindings)
        session.timed(lambda: one_pass(sets[0]))  # compile before scoring; pace its work
    else:
        prepared = prepare_replay(nodes, {**weight_bindings, **sets[0]}, out_ids)

        def one_pass(bindings):
            return prepared({**weight_bindings, **bindings})

    iters = loop_iterations(session.timed, lambda n: chained_loop(one_pass, sets, n, link_id), target_ms)
    return chained_loop(one_pass, sets, iters, link_id), iters, link_id, sets


def _compute_arms(trace: Trace, members: list[Stretch]) -> tuple[dict, dict[str, float]]:
    """One plain-matmul arm per dtype the members' ops are priced at, and
    each arm's flops, so the ceiling is clocked in the regions' own window."""
    dtypes = {compute_dtype(n) for m in members for n in trace.nodes[m.start_seq:m.end_seq + 1]}
    arms, flops = {}, {}
    for dtype in sorted(dtypes):
        arms[f"compute:{dtype}"], flops[dtype] = compute_probe(dtype)
    return arms, flops


def _rates(rows: Mapping[str, tuple[float, ...]], flops: Mapping[str, float]) -> dict[str, float]:
    """GFLOP/s per dtype from the compute arms' median pass."""
    return {dtype: f / statistics.median(rows[f"compute:{dtype}"]) / 1e6 for dtype, f in flops.items()}


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
    baseline: str = "plain",
) -> RegionPrice:
    """The region's cost as a fraction of one step, from a paired interleaved
    comparison against the step itself.

    A share is a ratio, and dividing a region clock by a step clock taken
    minutes earlier reports whatever the machine did in between. On
    the 8B decode run that produced shares summing to 586 ms against a 52 ms
    step, and the same region priced 44.9 ms in one run and 91.5 ms in the next.
    """
    try:
        loop_fn, iters, link_id, sets = _looped_replay(
            session, trace, stretch, input_sets, weight_bindings, target_ms, baseline)
    finally:
        session.settle()
    arms = {"step": step_fn, "region:library": loop_fn}
    if link_id is not None:
        arms["region:link"] = link_loop(sets, iters, link_id)
    arms["region:probe"] = chained_loop(_probe_pass(trace, stretch, sets[0], weight_bindings),
                                        sets, iters, link_id)
    compute, flops = _compute_arms(trace, [stretch])
    arms.update(compute)
    rows = sample_group(session, arms, pairs=pairs)
    return _group_price(rows, "region", iters, link_id is not None, _rates(rows, flops))


def price_region(
    region: Region,
    session: Session,
    traces: Mapping[str, Trace],
    input_sets_for: Mapping[tuple[str, int], list[dict[int, mx.array]]],
    weight_bindings: Mapping[str, dict[int, mx.array]],
    step_fns: Mapping[str, object],
    baseline: str = "plain",
    pairs: int = PRICE_PAIRS,
) -> None:
    """Fill t_orig_ms, t_rep_ms and p per workload. Copies with distinct
    boundary shapes price separately; identical-shape copies share one price.
    step_fns maps a workload to a callable that runs one step of the model, so
    each region's share is measured against the step rather than divided by it.
    input_sets_for is keyed by (workload, representative member start_seq).
    Production pricing uses price_group to share the model arm across regions."""
    per_workload_ms: dict[str, float] = {}
    per_workload_share: dict[str, float] = {}
    priced: dict[tuple, RegionPrice] = {}
    for m in region.members:
        trace = traces[m.workload]
        shape_key = (m.workload, _boundary_shapes(trace, m))
        if shape_key not in priced:
            sets = input_sets_for.get((m.workload, m.start_seq))
            if sets is None:
                region.rejected = f"no captured inputs for {m.workload} at {m.start_seq}"
                return
            else:
                priced[shape_key] = region_share(
                    session, trace, m, sets, weight_bindings[m.workload],
                    step_fns[m.workload], pairs=pairs,
                    baseline=baseline,
                )
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


def capture_instances(region: Region, traces: Mapping[str, Trace]) -> list[tuple[str, Stretch, int]]:
    """One representative and copy count for each boundary shape in a workload.

    The first shape keeps the workload label for existing stores. Additional
    shapes get a stable label using their representative's recorded position.
    A label identifies saved tensors, not another optimization workload.
    """
    grouped: dict[tuple, tuple[str, Stretch, int]] = {}
    seen_workloads = set()
    for member in region.members:
        key = (member.workload, _boundary_shapes(traces[member.workload], member))
        if key in grouped:
            label, representative, copies = grouped[key]
            grouped[key] = (label, representative, copies + 1)
        else:
            label = member.workload if member.workload not in seen_workloads else \
                f"{member.workload}@copy:{member.start_seq}"
            grouped[key] = (label, member, 1)
            seen_workloads.add(member.workload)
    return list(grouped.values())


def _group_price(rows: Mapping[str, tuple[float, ...]], key: str, iters: int,
                 linked: bool, rates: Mapping[str, float] | None = None) -> RegionPrice:
    """Reduce complete forward/reverse blocks before taking any ratios.

    Each arm's two observations surround the same block midpoint. Averaging
    them first cancels linear drift, including the link we subtract. Dividing
    individual rows by opposite endpoints biases the median when those model
    endpoints differ. The floor/library ratio uses that same paired block.
    """
    def centered(name):
        values = rows[name]
        return [(a + b) / 2 for a, b in zip(values[::2], values[1::2])]

    library, probe, step = centered(key + ":library"), centered(key + ":probe"), centered("step")
    link = centered(key + ":link") if linked else [0.0] * len(library)
    net = [(a - b) / iters for a, b in zip(library, link)]
    shares = [n / s for n, s in zip(net, step)]
    floor_ratios = [(p - l) / (a - l) for p, l, a in zip(probe, link, library) if a > l]
    share = max(statistics.median(shares), 0.0)
    ms = max(statistics.median(net), 0.0)
    floor_ms = max(statistics.median(floor_ratios), 0.0) * ms if floor_ratios else 0.0
    spread = statistics.stdev(shares) if len(shares) > 1 else 0.0
    return RegionPrice(share, ms, share / (share + spread) if share else 0.0, floor_ms, dict(rates or {}))


def price_group(regions: list[Region], session: Session, traces: Mapping[str, Trace],
                store, step_fns: Mapping[str, object], baseline: str = "plain",
                pairs: int = PRICE_PAIRS) -> None:
    """Price a disjoint group with one shared model arm per workload.

    Library, chain-only and boundary-probe times all come from the same
    forward/reverse passes as the model. Average each symmetric block before
    subtracting the chain and dividing by the model. Distinct boundary shapes
    have distinct prices; copies share a clock only within their shape group.
    """
    for region in regions:
        for values in (region.t_orig_ms, region.t_rep_ms, region.t_floor_ms,
                       region.p, region.p_rep, region.stability, region.prices):
            values.clear()
    for workload, step_fn in step_fns.items():
        arms = {"step": step_fn}
        prepared, members = [], []
        for region in regions:
            if region.rejected:
                continue
            for label, member, copies in capture_instances(region, traces):
                if member.workload != workload:
                    continue
                sets = [store.load(region.fingerprint, label, i, "inputs")
                        for i in range(store.set_count(region.fingerprint, label))]
                if not sets:
                    region.rejected = f"no captured inputs for {label}"
                    break
                trace = traces[workload]
                session.log("prepare_price", region=region.fingerprint, workload=workload, instance=label)
                try:
                    library, iters, link_id, sets = _looped_replay(
                        session, trace, member, sets, {}, CLOCK_TARGET_MS,
                        region.library_arm(workload, baseline))
                finally:
                    session.settle()
                key = f"{region.fingerprint}:{label}"
                arms[key + ":library"] = library
                if link_id is not None:
                    arms[key + ":link"] = link_loop(sets, iters, link_id)
                arms[key + ":probe"] = chained_loop(
                    _probe_pass(trace, member, sets[0], {}), sets, iters, link_id)
                prepared.append((region, label, copies, iters, link_id))
                members.append(member)
        if not prepared:
            continue
        compute, flops = _compute_arms(traces[workload], members)
        arms.update(compute)
        rows = sample_group(session, arms, pairs=pairs)
        rates = _rates(rows, flops)
        for region, label, copies, iters, link_id in prepared:
            if region.rejected:
                continue
            price = _group_price(rows, f"{region.fingerprint}:{label}", iters, link_id is not None, rates)
            region.prices[label] = price
            region.p_rep.setdefault(workload, price.share)
            region.p[workload] = region.p.get(workload, 0.0) + price.share * copies
            region.t_rep_ms.setdefault(workload, price.ms)
            region.t_orig_ms[workload] = region.t_orig_ms.get(workload, 0.0) + price.ms * copies
            region.t_floor_ms.setdefault(workload, price.floor_ms)
            region.stability[workload] = min(region.stability.get(workload, 1.0), price.stability)
            session.log("region_price", region=region.fingerprint, workload=workload, instance=label,
                        copies=copies, share=price.share, region_ms=price.ms,
                        floor_ms=price.floor_ms, compute_gflops=price.gflops,
                        price_stability=price.stability, samples=pairs)


def _boundary_shapes(trace: Trace, stretch: Stretch) -> tuple:
    specs = trace.span_specs(stretch.start_seq, stretch.end_seq)
    return tuple(specs.get(a) for a in stretch.input_ids + stretch.output_ids)
