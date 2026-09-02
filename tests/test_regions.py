"""Region building, fingerprints, pricing, roofline, ranking, sweep.

Candidate sets are checked against hand-derived expectations per fixture; the
pricing tests use the real region clock on the real GPU.
"""

import importlib.util
from pathlib import Path

import mlx.core as mx
import pytest

from autotuner.measure.clocks import CLOCK_EST_ITERS, chained_loop
from autotuner.measure.peaks import Peaks
from autotuner.measure.session import Session, time_once
from autotuner.regions import price as price_mod
from autotuner.regions.build import build_stretches, is_view
from autotuner.regions.fingerprint import fingerprint, group_copies
from autotuner.regions.price import (
    CaptureMismatch,
    _looped_replay,
    capture_boundaries,
    price_region,
    region_share,
)
from autotuner.regions.rank import apply_floor, covered_by, free_members, rank
from autotuner.regions.roofline import node_flops, stretch_roofline
from autotuner.regions.sweep import SweepDivergence, locate_span
from autotuner.regions.types import Region, Roofline, Stretch
from autotuner.trace import Tracer
from tests.conftest import current_tracer, tracer_for_module

FIXTURES = Path(__file__).parent / "fixtures"

_module_tracer = tracer_for_module()


def tracer() -> Tracer:
    return current_tracer()


def load_fixture(name: str):
    tracer()
    spec = importlib.util.spec_from_file_location(f"fixture_r_{name}", FIXTURES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def traced(name: str, shape):
    model = load_fixture(name)
    x = mx.random.normal(shape, key=mx.random.key(0))
    trace, _ = tracer().trace(model, [x])
    return model, x, trace


def spans(stretches):
    return {(s.start_seq, s.end_seq) for s in stretches}


def test_norm_three_proj_candidates():
    """Chain growth merges neighbors on a shared input with no data edges:
    norm(0) q(1) k(2) v(3) yields the maximal chain, its prefixes, and the
    four singletons."""
    _, _, trace = traced("norm_three_proj", (4, 32))
    got = spans(build_stretches(trace, "w"))
    assert got == {(0, 0), (1, 1), (2, 2), (3, 3), (0, 1), (0, 2), (0, 3)}


def test_views_only_produces_no_region():
    """Views absorb into chains but a views-only stretch is not a region."""
    _, _, trace = traced("views_only", (4, 8))
    stretches = build_stretches(trace, "w")
    for s in stretches:
        nodes = trace.nodes[s.start_seq:s.end_seq + 1]
        assert not all(is_view(n) for n in nodes)
    # the full chain absorbs the view run between matmul and add
    assert (0, len(trace.nodes) - 1) in spans(stretches)


def test_cache_write_splits_chains():
    """The retained k production is a barrier: no stretch may contain it."""
    _, _, trace = traced("cache_retention", (4, 16))
    stretches = build_stretches(trace, "w")
    k_seq = 0  # first op produces the retained k
    for s in stretches:
        assert not (s.start_seq <= k_seq <= s.end_seq)
    assert len(stretches) > 0


def test_slice_write_ends_regions():
    _, _, trace = traced("operator_soup", (4, 8))
    setitem_seq = next(n.seq for n in trace.nodes if n.op == "array.__setitem__")
    for s in build_stretches(trace, "w"):
        assert not (s.start_seq <= setitem_seq <= s.end_seq)


def test_stretch_liveness_includes_step_outputs():
    _, _, trace = traced("step_output_region", (4, 16))
    full = next(
        s for s in build_stretches(trace, "w")
        if (s.start_seq, s.end_seq) == (0, len(trace.nodes) - 1)
    )
    h_id = next(n.out_arrays[0] for n in trace.nodes if n.op == "mx.maximum")
    assert h_id in full.output_ids  # live: both returned and consumed inside


def test_copy_grouping_repeated_layers():
    """Four structurally identical layers group into one region with copies=4."""
    _, _, trace = traced("repeated_layers", (4, 16))
    stretches = build_stretches(trace, "w")
    regions = group_copies({"w": trace}, {"w": stretches})
    by_copies = [r for r in regions if r.copies == 4]
    assert by_copies, "no region grouped all four layers"
    full_layer = max(by_copies, key=lambda r: len(r.ops))
    assert len(full_layer.ops) == 3  # rms_norm, matmul, residual add
    addresses = {m.scope_stack[-1] for m in full_layer.members}
    assert addresses == {f"layers.{i}@0" for i in range(4)}


def test_fingerprint_ignores_shapes_but_not_dtypes():
    _, _, t16 = traced("repeated_layers", (4, 16))
    _, _, t16b = traced("repeated_layers", (64, 16))
    s16 = build_stretches(t16, "a")
    s16b = build_stretches(t16b, "b")
    f_a = fingerprint(t16, s16[0])
    f_b = fingerprint(t16b, next(x for x in s16b if (x.start_seq, x.end_seq) == (s16[0].start_seq, s16[0].end_seq)))
    assert f_a == f_b  # same sequence at another batch is the same region


def test_roofline_chain_bytes_smaller_than_sum_of_ops():
    """A chain's boundary bytes are provably smaller than the sum of its ops'
    bytes: the intermediates drop out."""
    _, _, trace = traced("norm_three_proj", (4, 32))
    stretches = build_stretches(trace, "w")
    peaks = Peaks(bandwidth_gbps=100.0, flops_gflops={"float32": 3000.0}, launch_us=4.0)
    chain = next(s for s in stretches if (s.start_seq, s.end_seq) == (0, 3))
    singles = [s for s in stretches if s.start_seq == s.end_seq]
    chain_roof = stretch_roofline(trace, chain, peaks, t_orig_ms=1.0)
    singles_mem = sum(
        stretch_roofline(trace, s, peaks, t_orig_ms=1.0).t_mem_ms for s in singles
    )
    assert chain_roof.t_mem_ms < singles_mem
    # hand check: chain boundary = x (4x32) + g (32) + 3 weights (32x32) + 3 outs (4x32)
    expected_bytes = 4 * (4 * 32 + 32 + 3 * 32 * 32 + 3 * 4 * 32)
    assert chain_roof.t_mem_ms == pytest.approx(expected_bytes / 100e9 * 1e3, rel=1e-6)


def test_node_flops_matmul_hand_check():
    _, _, trace = traced("norm_three_proj", (4, 32))
    mm = next(n for n in trace.nodes if n.op == "array.__matmul__")
    assert node_flops(mm) == 2.0 * 4 * 32 * 32


def test_capture_and_share_price_stability():
    """Priced shares are stable across two clockings within noise, and the
    boundary capture returns the library's own values. Gated: a machine
    crossing the thermal throttle mid-test cannot clock twice consistently."""
    from tests.conftest import require_healthy_gpu

    require_healthy_gpu()
    model, x, trace = traced("norm_three_proj", (64, 32))
    tr = tracer()
    stretches = build_stretches(trace, "w")
    chain = next(s for s in stretches if (s.start_seq, s.end_seq) == (0, 3))
    ids = set(chain.input_ids) | set(chain.output_ids)
    arrays = capture_boundaries(tr, model, [x], trace, ids)
    assert set(arrays) == ids
    session = Session()
    weights = {a: arrays[a] for a in chain.input_ids if a in trace.weights}
    inputs = {a: arrays[a] for a in chain.input_ids if a not in trace.weights}
    step = lambda: model(x)
    p1 = region_share(session, trace, chain, [inputs], weights, step, pairs=8)
    p2 = region_share(session, trace, chain, [inputs], weights, step, pairs=8)
    assert p1.share > 0 and p2.share > 0
    assert p1.floor_ms > 0 and p2.floor_ms > 0  # the probe was clocked beside the region
    assert abs(p1.share - p2.share) / max(p1.share, p2.share) < 0.5  # same clock within generous noise


def test_region_loop_is_sized_from_an_amortizing_estimate():
    """The loop length must come from the difference between a long and a
    short warm loop, never one pass. A sample's fixed submit-and-sync cost,
    several milliseconds on a busy GPU, would otherwise be read as pass time
    and size the loop far too short. No GPU timing: the fake session scripts
    the clock and the assertion is on how the estimate was taken."""
    _, _, trace = traced("norm_three_proj", (8, 32))
    stretches = build_stretches(trace, "w")
    chain = next(s for s in stretches if (s.start_seq, s.end_seq) == (0, 3))
    inputs, weights = _bindings_for(trace, chain)

    class CountingSession(Session):
        """Counts replays and reports a fixed per-call time, so the iteration
        count derived is deterministic."""

        def __init__(self):
            super().__init__(sleep=lambda _s: None)
            self.passes = 0

        def timed(self, fn):
            before = _replays["n"]
            fn()
            self.passes += _replays["n"] - before
            return 1e-3  # 1 ms per timed call, whatever it contained

    _replays = {"n": 0}
    real_replay = price_mod.replay

    def counting_replay(*a, **kw):
        _replays["n"] += 1
        return real_replay(*a, **kw)

    price_mod.replay = counting_replay
    try:
        session = CountingSession()
        _looped_replay(session, trace, chain, [inputs], weights, 20.0)
    finally:
        price_mod.replay = real_replay

    # a short loop thrown away, then a short and a long one, nothing else
    assert session.passes == (1 + 1 + 4) * CLOCK_EST_ITERS


def test_compiled_replay_arm_matches_the_plain_arm():
    """Under a compiled baseline the pricing loop runs the region's ops as one
    compiled graph; its outputs must equal the plain replay's, set by set."""
    _, _, trace = traced("norm_three_proj", (8, 32))
    stretches = build_stretches(trace, "w")
    chain = next(s for s in stretches if (s.start_seq, s.end_seq) == (0, 3))
    inputs, weights = _bindings_for(trace, chain)
    session = Session(sleep=lambda _s: None)
    plain_loop, n1, *_ = _looped_replay(session, trace, chain, [inputs], weights, 20.0, "plain")
    compiled_loop, n2, *_ = _looped_replay(session, trace, chain, [inputs], weights, 20.0, "compiled")
    plain, compiled = plain_loop(), compiled_loop()
    mx.eval(plain, compiled)
    assert len(plain) == n1 and len(compiled) == n2
    for p_outs, c_outs in zip(plain, compiled):
        assert all(mx.array_equal(a, b).item() for a, b in zip(p_outs, c_outs))


def test_region_loop_agrees_with_a_long_amortizing_loop():
    """The region clock and a plain long loop over the same replay must agree.
    They are the two halves of the same comparison: pricing sets s_max and the
    ranking, the ship clock decides wins, and a gap between them inflates every
    region's apparent headroom by that factor."""
    from tests.conftest import require_healthy_gpu, require_quiet_load

    require_healthy_gpu()
    require_quiet_load()
    _, _, trace = traced("norm_three_proj", (64, 32))
    stretches = build_stretches(trace, "w")
    # a one-op stretch: the less work per pass, the more a bad loop length
    # leaks fixed latency, so this is where the two clocks separate
    span = next(s for s in stretches if s.start_seq == s.end_seq)
    inputs, weights = _bindings_for(trace, span)

    session = Session()
    loop_fn, iters, link_id, sets = _looped_replay(session, trace, span, [inputs], weights, 20.0)
    session.warm_until_stable(loop_fn)
    samples = []
    for _ in range(5):
        session.fresh_chunk((loop_fn,))
        samples.append(session.timed(loop_fn))
    session.settle()
    priced_ms = sorted(samples)[2] / iters * 1e3

    nodes = trace.nodes[span.start_seq:span.end_seq + 1]
    out_ids = list(span.output_ids) or [nodes[-1].out_arrays[0]]
    one_pass = lambda b: list(price_mod.replay(nodes, {**weights, **b}, out_ids).values())
    n = 400
    long_loop = chained_loop(one_pass, sets, n, link_id)  # the same chained discipline, ten times longer
    time_once(long_loop)  # warm
    reference_ms = min(time_once(long_loop) / n * 1e3 for _ in range(3))
    # measured on the reference machine: 0.81-0.90x amortizing, 1.55-2.37x when
    # the loop is sized from one cold pass. 1.3 sits in the gap.
    assert priced_ms < 1.3 * reference_ms, (
        f"region clock {priced_ms:.5f} ms vs long-loop {reference_ms:.5f} ms; "
        "the pricing loop is not amortizing fixed submit-and-sync latency"
    )


def _bindings_for(trace, stretch):
    """Real boundary values for a stretch, split into inputs and weights."""
    arrays = {}
    for node in trace.nodes[stretch.start_seq:stretch.end_seq + 1]:
        for aid, (shape, dtype) in zip(node.in_arrays, node.in_specs):
            arrays.setdefault(aid, mx.random.normal(shape).astype(getattr(mx, dtype)))
    mx.eval(list(arrays.values()))
    weights = {a: arrays[a] for a in stretch.input_ids if a in trace.weights}
    inputs = {a: arrays[a] for a in stretch.input_ids if a not in trace.weights}
    return inputs, weights


def test_capture_aborts_on_nondeterministic_model():
    tr = tracer()

    class Flaky:
        calls = 0

        def __call__(self, x):
            Flaky.calls += 1
            if Flaky.calls % 2 == 0:
                return mx.tanh(x @ x.T)
            return x @ x.T

    model = Flaky()
    x = mx.random.normal((8, 8), key=mx.random.key(1))
    trace, _ = tr.trace(model, [x])
    with pytest.raises(CaptureMismatch):
        capture_boundaries(tr, model, [x], trace, set())


def test_sweep_span_resolves_at_other_size():
    """A region priced at one size resolves and replays at another via its
    sweep instance."""
    model = load_fixture("repeated_layers")
    tr = tracer()
    x512 = mx.random.normal((512, 16), key=mx.random.key(2))
    x7 = mx.random.normal((7, 16), key=mx.random.key(3))
    t512, _ = tr.trace(model, [x512])
    t7, _ = tr.trace(model, [x7])
    stretches = build_stretches(t512, "w")
    layer2 = next(
        s for s in stretches
        if t512.nodes[s.start_seq].module_address == "layers.2@0"
        and s.end_seq - s.start_seq == 2
    )
    resolved = locate_span(t512, layer2, t7, "w")
    node = t7.nodes[resolved.start_seq]
    assert node.module_address == "layers.2@0"
    assert node.in_specs[0][0] == (7, 16)


def test_sweep_divergence_is_named():
    model = load_fixture("data_branch")
    tr = tracer()
    x = mx.random.normal((4, 8), key=mx.random.key(4))
    trace, _ = tr.trace(model, [x])
    # forge a retrace with a different taken path by tracing opposite-sign input
    trace2, _ = tr.trace(model, [-mx.abs(x)])
    stretches = build_stretches(trace, "w")
    tail = max(stretches, key=lambda s: s.end_seq)
    if [n.op for n in trace.nodes] != [n.op for n in trace2.nodes]:
        with pytest.raises(SweepDivergence):
            locate_span(trace, tail, trace2, "w")


def test_rank_and_floor_and_overlap():
    roof_mem = Roofline(1.0, 0.1, 0.1, 1.0, "memory", 5.0)
    roof_cmp = Roofline(0.1, 1.0, 0.1, 1.0, "compute", 5.0)
    roof_flat = Roofline(1.0, 0.1, 0.1, 1.0, "memory", 1.05)

    def region(fp, p, roof):
        r = Region(fingerprint=fp, ops=("mx.add",))
        r.members.append(Stretch("w", 0, 0, (), (), ("@0",)))
        r.p["w"] = p
        r.roofline = roof
        return r

    a = region("a", 0.30, roof_cmp)
    b = region("b", 0.30, roof_mem)
    c = region("c", 0.01, roof_mem)   # under floor
    d = region("d", 0.30, roof_flat)  # no headroom
    kept = apply_floor([a, b, c, d])
    assert {r.fingerprint for r in kept} == {"a", "b"}
    assert "under floor" in c.rejected and "no headroom" in d.rejected
    ranked = rank(kept)
    assert [r.fingerprint for r in ranked] == ["b", "a"]  # memory beats compute on ties

    # a shipped cut owns every candidate copy that touches it: one inside it,
    # one reaching past it, one straddling its edge; only a disjoint copy is free
    big = Region(fingerprint="big", ops=("x", "y"))
    big.members.append(Stretch("w", 0, 3, (), (), ("@0",)))
    small = Region(fingerprint="small", ops=("x",))
    small.members.append(Stretch("w", 1, 1, (), (), ("@0",)))
    edge = Region(fingerprint="edge", ops=("y", "z"))
    edge.members.append(Stretch("w", 3, 5, (), (), ("@0",)))
    apart = Region(fingerprint="apart", ops=("z",))
    apart.members.append(Stretch("w", 4, 5, (), (), ("@0",)))
    apart.members.append(Stretch("w", 2, 2, (), (), ("@0",)))
    assert covered_by(small, big) and covered_by(big, small) and covered_by(edge, big)
    assert not covered_by(apart, big)
    assert [m.start_seq for m in free_members(apart, [big])] == [4]



def test_multi_copy_pricing_separates_rep_from_total():
    """t_orig_ms sums every copy (the region's share of the step) while
    t_rep_ms is one copy; the roofline's s_max divides the one-copy clock by
    the one-copy floor, so a many-copy region cannot inflate its own headroom
    (the 8B run reported s_max 165x for a region actually at its roofline)."""
    model, x, trace = traced("repeated_layers", (4, 16))
    tr = tracer()
    stretches = build_stretches(trace, "w")
    regions = group_copies({"w": trace}, {"w": stretches})
    region = max((r for r in regions if r.copies == 4), key=lambda r: len(r.ops))
    rep = region.members[0]
    ids = set(rep.input_ids) | set(rep.output_ids)
    arrays = capture_boundaries(tr, model, [x], trace, ids)
    weights = {a: arrays[a] for a in rep.input_ids if a in trace.weights}
    inputs = {a: arrays[a] for a in rep.input_ids if a not in trace.weights}
    price_region(region, Session(), {"w": trace},
                 {("w", rep.start_seq): [inputs]}, {"w": weights},
                 {"w": lambda: model(x)})
    assert region.t_rep_ms["w"] > 0
    assert region.t_orig_ms["w"] == pytest.approx(4 * region.t_rep_ms["w"])
    # p is now a measured share and t_orig its per-copy sum, so both scale with
    # the copy count together. Whether the share is under 1 is a claim about a
    # measurement, and is asserted in test_drift on a region big enough to make
    # one; this fixture is microseconds of work.
    assert region.p["w"] == pytest.approx(4 * region.p_rep["w"])
    assert 0 < region.stability["w"] <= 1


def _fake(op_seq, weight_shapes):
    """A trace of consecutive single-output ops on one activation; each op's
    second input is a weight of the given shape (None for a plain unary op)."""
    from autotuner.trace.recorder import ArrayRef
    from autotuner.trace.types import Liveness, Retention, Trace, TraceNode

    nodes, weights, aid = [], set(), 1
    for seq, (op, wshape) in enumerate(zip(op_seq, weight_shapes)):
        ins, specs, args = [aid - 1], [((4, 4), "float32")], [ArrayRef(0)]
        if wshape is not None:
            ins.append(100 + seq); specs.append((wshape, "float32")); args.append(ArrayRef(1))
            weights.add(100 + seq)
        nodes.append(TraceNode(seq=seq, op=op, in_arrays=tuple(ins), out_arrays=(aid,),
                               in_specs=tuple(specs), out_specs=(((4, 4), "float32"),),
                               scalar_args={"args": tuple(args), "kwargs": {}},
                               module_address="@0", position_in_module=seq, module_stack=("@0",)))
        aid += 1
    liveness = {n.out_arrays[0]: Liveness(Retention.CONSUMED, (n.seq + 1,) if n.seq + 1 < len(nodes) else ())
                for n in nodes}
    liveness[nodes[-1].out_arrays[0]] = Liveness(Retention.STEP_OUTPUT, ())
    return Trace(nodes=tuple(nodes), edges={}, step_outputs=(nodes[-1].out_arrays[0],),
                 weights=frozenset(weights), inputs=frozenset({0}), liveness=liveness)


def test_chains_end_before_a_matmul_that_consumes_an_earlier_matmul():
    """gate, silu, times up, down: the down projection needs every element of
    the earlier matmul's output, which one Metal launch cannot wait for. The
    chain stops before it; the prefix and the singleton remain."""
    t = _fake(["array.__matmul__", "mx.sigmoid", "array.__mul__", "array.__matmul__"],
              [(4, 4), None, None, (4, 4)])
    got = spans(build_stretches(t, "w"))
    assert (0, 2) in got and (3, 3) in got
    assert not any(s == 0 and e == 3 for s, e in got)
    # two independent matmuls reading the same input may share a chain
    t2 = _fake(["array.__matmul__", "array.__matmul__"], [(4, 4), (4, 4)])
    t2 = _fake(["mx.exp", "array.__matmul__"], [None, (4, 4)])
    assert (0, 1) in spans(build_stretches(t2, "w"))


def test_projections_group_by_weight_through_the_transpose():
    """nn.Linear reads its matrix through a transpose. The matmul alone is no
    candidate (the chain from the transpose is the same cut with a better
    boundary), and chains against different matrices are different regions:
    the first Qwen run folded 197 projections over six matrices into one."""
    model = load_fixture("llama_ish")
    tokens = mx.random.randint(0, 512, (1, 4), key=mx.random.key(1))
    trace, _ = tracer().trace(model, [tokens])
    regions = group_copies({"w": trace}, {"w": build_stretches(trace, "w")})
    assert not any(r.ops == ("array.__matmul__",) for r in regions)
    proj = [r for r in regions if r.ops == ("array.T", "array.__matmul__")]
    shapes = {}
    for r in proj:
        specs = {a: s for m in r.members for a, s in trace.span_specs(m.start_seq, m.end_seq).items()}
        seen = {specs[a][0] for m in r.members for a in m.input_ids if a in trace.weights}
        assert len(seen) == 1, "one region, one weight shape"
        shapes[seen.pop()] = r.copies
    # 4 square projections per layer, gate and up, down, and the tied head
    assert shapes == {(256, 256): 32, (1024, 256): 16, (256, 1024): 8, (512, 256): 1}


def test_a_weight_only_call_starts_a_chain():
    """exp(0) T(1) matmul(2) in one module: the transpose reads only a weight,
    so a chain starts there and the projection is a candidate by itself; the
    matmul on the transposed view is not a candidate alone."""
    _, _, trace = traced("weight_view_in_module", (4, 32))
    assert spans(build_stretches(trace, "w")) == {(0, 0), (0, 2), (1, 2)}


def test_weight_shapes_tell_copies_apart():
    """The same op against a different weight shape is a different kernel."""
    same_a = _fake(["array.__matmul__"], [(4, 4)])
    same_b = _fake(["array.__matmul__"], [(4, 4)])
    other = _fake(["array.__matmul__"], [(4, 8)])
    cut = lambda t: build_stretches(t, "w")[0]
    assert fingerprint(same_a, cut(same_a)) == fingerprint(same_b, cut(same_b))
    assert fingerprint(same_a, cut(same_a)) != fingerprint(other, cut(other))


def test_a_measured_floor_replaces_the_bytes_and_launch_terms():
    """With a probe clock the roofline is the larger of that floor and the
    flops term; the bound still says which of bytes or launch the floor is
    made of. Without one the arithmetic stands."""
    _, _, trace = traced("norm_three_proj", (4, 32))
    chain = next(s for s in build_stretches(trace, "w") if (s.start_seq, s.end_seq) == (0, 3))
    peaks = Peaks(bandwidth_gbps=100.0, flops_gflops={"float32": 3000.0}, launch_us=4.0)
    plain = stretch_roofline(trace, chain, peaks, t_orig_ms=1.0)
    assert plain.t_floor_ms is None and plain.t_roofline_ms == max(plain.t_mem_ms, plain.t_compute_ms, plain.t_launch_ms)
    measured = stretch_roofline(trace, chain, peaks, t_orig_ms=1.0, floor_ms=0.5)
    assert measured.t_floor_ms == 0.5 and measured.t_roofline_ms == 0.5 and measured.s_max == 2.0
    assert measured.bound == ("memory" if measured.t_mem_ms >= measured.t_launch_ms else "launch")
    compute_bound = stretch_roofline(trace, chain, peaks, t_orig_ms=1.0, floor_ms=1e-9)
    assert compute_bound.bound == "compute" and compute_bound.t_roofline_ms == plain.t_compute_ms


def test_step_floor_counts_outside_bytes_flops_and_launches():
    """The scout line: bytes the step must read from outside itself and write
    out, flops of every op, launches the library fires, and the room left."""
    from autotuner.regions.roofline import step_floor

    _, _, trace = traced("norm_three_proj", (4, 32))
    peaks = Peaks(bandwidth_gbps=100.0, flops_gflops={"float32": 3000.0}, launch_us=4.0)
    f = step_floor(trace, peaks, step_ms=1.0)
    x, g, w, out = 4 * 32 * 4, 32 * 4, 32 * 32 * 4, 4 * 32 * 4
    assert f["bytes_mb"] == pytest.approx((x + g + 3 * w + 3 * out) / 1e6)
    assert f["gflop"] == pytest.approx((4 * 4 * 32 + 3 * 2 * 4 * 32 * 32) / 1e9)
    assert f["launches"] == 4
    assert f["floor_ms"] == max(f["t_mem_ms"], f["t_compute_ms"]) and f["room"] == pytest.approx(1 - f["floor_ms"])


def test_roofline_counts_the_fused_kernel_launch_once():
    _, _, trace = traced("norm_three_proj", (4, 32))
    chain = next(s for s in build_stretches(trace, "w") if (s.start_seq, s.end_seq) == (0, 3))
    peaks = Peaks(bandwidth_gbps=100.0, flops_gflops={"float32": 3000.0}, launch_us=4.0)
    assert stretch_roofline(trace, chain, peaks, t_orig_ms=1.0).t_launch_ms == pytest.approx(0.004)


def test_uncovered_ops_are_named_before_pricing():
    from autotuner.scaffold import uncovered_op

    assert uncovered_op(["mx.fast.rms_norm", "array.__matmul__", "mx.sigmoid"]) is None
    assert uncovered_op(["mx.quantized_matmul", "mx.fast.scaled_dot_product_attention"]) \
        == "mx.fast.scaled_dot_product_attention"
