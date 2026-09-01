"""Sandbox worker: one kernel evaluation per process (plan section 8).

Reads one JSON job spec on stdin, rebuilds everything from it (tensors from
safetensors paths, the kernel via KernelSpec), and prints exactly one JSON
verdict line to stdout as its last line. Any unexpected exception tracebacks
to stderr and exits nonzero; the parent maps that to a "subprocess" verdict.

Two spec kinds share this worker. The M5 JobSpec runs the minimal gate set
(compile probe, watchdog, poison, allclose). The M6 LadderSpec runs the full
ladder: phase "validate" runs gates 2-8, phase "score" re-runs smoke and
determinism on the un-instrumented pipeline and then the ship clock.

The worker is mode-agnostic: Metal read the validation env at launch, its
"Invalid device load/store" lines land on this process's stderr, and the
parent scans them after exit. A truly hanging kernel blocks mx.eval forever
(verified on this machine), so the parent wall timeout is the hang half of
the watchdog; the 10x rule here catches slow-but-terminating kernels, against
a library region time re-measured in this same process.

Poison rule: every correctness launch runs with init_value=nan, except that
atomic_outputs kernels get init_value=0.0, because atomic accumulation is
only well-defined from an identity and NaN would fail every legitimate atomic
kernel. The value gates plus the determinism gate police atomics instead.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from dataclasses import replace

import mlx.core as mx

from autotuner.ladder.golden import err as golden_err
from autotuner.ladder.golden import golden_outputs
from autotuner.ladder.golden import passes as golden_passes
from autotuner.ladder.numeric import REGIMES, max_abs_diff, value_regimes
from autotuner.ladder.numeric import compare as numeric_compare
from autotuner.measure.clocks import compare as paired_compare
from autotuner.measure.session import Session, time_once
from autotuner.regions.store import load_set
from autotuner.sandbox.poison import saturate_pool
from autotuner.sandbox.protocol import (
    GATES,
    WATCHDOG_FACTOR,
    JobSpec,
    LadderSpec,
    TensorSet,
    Verdict,
)
from autotuner.trace.replay import replay
from autotuner.trace.serialize import nodes_from_json
from autotuner_runtime.kernels import KernelSpec, call, fallback_fires

_BUILD_FAILURE = "Unable to build metal library from source"
_PREAMBLE_RE = re.compile(r"^\[metal::Device\] Unable to build metal library from source\s*")
_DIAG_RE = re.compile(r"^(\S+):(\d+):(\d+): (error|warning|note): (.*)$")
_PROBE_MARKER = "mao_line_probe"
_TEXT_CAP = 4000

# Relative machine epsilon per output dtype, for the assoc-changing floor.
_EPS_REL = {"float16": 2.0 ** -10, "bfloat16": 2.0 ** -7, "float32": 2.0 ** -23}
EPS_FLOOR_MULTIPLE = 8.0
DETERMINISM_RUNS = 3          # spec-fixed (plan 5.12)
CLOCK_SAMPLE_TARGET_S = 0.02  # tens of ms of work per sample
# the ship margin is floored by 3 * the standard error of the median win, which
# falls as sqrt(pairs). At 8 the floor was 25% of a decode region's clock and
# swallowed real wins; 32 halves it for about four extra seconds per hypothesis.
# The job sends the count it wants, so a fixture job need not pay for a
# precision it has no use for.
CLOCK_PAIRS = 32
CLOCK_MAX_ITERS = 2000
CLOCK_EST_ITERS = 10
WATCHDOG_ITERS = 4
# The magnitude regimes step down until the library's own output stays finite
# wherever it was finite on the real data: past that point the reference
# means nothing and no kernel could be written to match it.
REGIME_MAGNITUDES = {"scaled_up": (1e3, 1e2, 1e1), "outliers": (1e4, 1e3, 1e2)}
# A single evaluated pass is dominated by fixed submit-and-sync latency
# (measured 5-9ms on the reference machine for microsecond-scale work), so
# every per-pass time here comes from a small amortizing loop, both arms alike.

# Bitwise determinism compares float bits as integers (-0.0 != +0.0, NaN == NaN).
_BITS = {"float16": mx.uint16, "bfloat16": mx.uint16, "float32": mx.uint32}


def main() -> None:
    text = sys.stdin.read()
    if json.loads(text).get("kind") == "ladder":
        verdict = evaluate_ladder(LadderSpec.from_json(text))
    else:
        verdict = evaluate(JobSpec.from_json(text))
    print(verdict.to_json(), flush=True)


def evaluate(spec: JobSpec) -> Verdict:
    kspec = KernelSpec.from_json(json.dumps(spec.kernel))
    inputs = _load(spec.inputs)
    reference = _load(spec.reference)
    if len(inputs) != len(kspec.input_names):
        raise ValueError(
            f"spec has {len(inputs)} input tensors for {len(kspec.input_names)} kernel inputs"
        )
    if len(reference) != len(kspec.output_names):
        raise ValueError(
            f"spec has {len(reference)} reference tensors for {len(kspec.output_names)} kernel outputs"
        )
    if spec.saturate_pool:
        saturate_pool(a.nbytes for a in reference)

    gates = tuple(g for g in GATES if g in spec.gates or g == "compile")
    gates_passed: list[str] = []
    timing: dict[str, float] = {"t_library_ms": spec.t_library_ms}

    def fail(gate: str, detail: dict) -> Verdict:
        return Verdict(False, gate, tuple(gates_passed), detail, timing)

    # compile: the probe eval builds the pipeline; a broken build surfaces
    # only here, as a RuntimeError (plan 5.10). The probe is also the first
    # correctness launch, poisoned with init_value=nan, and its outputs feed
    # the poison and allclose gates below.
    outs: list[mx.array] = []

    def poisoned_launch() -> list[mx.array]:
        outs[:] = call(kspec, inputs, init_value=float("nan"))
        return outs

    try:
        timing["first_run_ms"] = time_once(poisoned_launch) * 1e3
    except RuntimeError as e:
        if _BUILD_FAILURE in str(e):
            return fail("compile", _compile_detail(str(e), _probe_offset(kspec, inputs)))
        return fail("compile", {"probe_eval_error": str(e)[:_TEXT_CAP]})
    gates_passed.append("compile")

    # watchdog, slow half: one timed run against the library region time the
    # parent measured. The probe above already paid the Metal compile, so this
    # times the kernel itself. No init_value on timed runs (plan 5.10).
    if "watchdog" in gates:
        timing["timed_run_ms"] = time_once(lambda: call(kspec, inputs)) * 1e3
        if timing["timed_run_ms"] > WATCHDOG_FACTOR * spec.t_library_ms:
            return fail("watchdog", {
                "timed_run_ms": timing["timed_run_ms"],
                "t_library_ms": spec.t_library_ms,
                "watchdog_factor": WATCHDOG_FACTOR,
            })
        gates_passed.append("watchdog")

    # poison: with init_value=nan every unwritten output element is NaN
    # deterministically; any non-finite value where the reference is finite
    # fails.
    if "poison" in gates:
        counts = {}
        for name, out, ref in zip(kspec.output_names, outs, reference):
            _check_contract(name, out, ref)
            bad = mx.logical_and(mx.logical_not(mx.isfinite(out)), mx.isfinite(ref))
            counts[name] = int(mx.sum(bad).item())
        if any(counts.values()):
            return fail("poison", {"non_finite_over_finite_ref": counts})
        gates_passed.append("poison")

    # allclose: the bare numeric gate against the saved library reference,
    # under harness-held tolerances.
    if "allclose" in gates:
        rtol = float(spec.tolerances["rtol"])
        atol = float(spec.tolerances["atol"])
        per_output = {}
        ok = True
        for name, out, ref in zip(kspec.output_names, outs, reference):
            _check_contract(name, out, ref)
            close = bool(mx.allclose(out, ref, rtol=rtol, atol=atol, equal_nan=True).item())
            per_output[name] = {"allclose": close, "max_abs_diff": _max_abs_diff(out, ref)}
            ok = ok and close
        if not ok:
            return fail("allclose", {"outputs": per_output})
        gates_passed.append("allclose")

    return Verdict(True, None, tuple(gates_passed), {}, timing)


def _load(ts: TensorSet) -> list[mx.array]:
    data = mx.load(ts.path)
    missing = [n for n in ts.names if n not in data]
    if missing:
        raise ValueError(f"{ts.path} is missing tensors {missing}")
    return [data[n] for n in ts.names]


def _check_contract(name: str, out: mx.array, ref: mx.array) -> None:
    # A shape or dtype mismatch against the reference is a spec-construction
    # bug in M5 (gate 1 static checks own it from M6): crash, do not grade.
    if tuple(out.shape) != tuple(ref.shape) or out.dtype != ref.dtype:
        raise ValueError(
            f"output {name!r} is {tuple(out.shape)} {out.dtype}, "
            f"reference is {tuple(ref.shape)} {ref.dtype}"
        )


def _max_abs_diff(out: mx.array, ref: mx.array) -> float:
    """Max abs difference where both sides are finite (JSON-safe)."""
    if out.size == 0:
        return 0.0
    diff = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))
    finite = mx.logical_and(mx.isfinite(out), mx.isfinite(ref))
    return float(mx.max(mx.where(finite, diff, mx.zeros_like(diff))).item())


def _probe_offset(kspec: KernelSpec, inputs: list[mx.array]) -> int | None:
    """The compile-error line offset is per kernel: it counts utils.h plus the
    generated signature, which grows with the IO list (PLATFORM.md spike_06).
    Measure it by compiling this kernel's body behind a deliberate #error on
    body line 1; the reported line minus one is the offset. Prepending a line
    changes nothing the signature generator scans for."""
    probe = replace(
        kspec,
        kernel_id=kspec.kernel_id + "_lineprobe",
        name=kspec.name + "_lineprobe",
        source=f"#error {_PROBE_MARKER}\n{kspec.source}",
    )
    try:
        mx.eval(call(probe, inputs))
    except RuntimeError as e:
        for line in str(e).splitlines():
            m = _DIAG_RE.match(line)
            if m and _PROBE_MARKER in m.group(5):
                return int(m.group(2)) - 1
    return None


def _compile_detail(message: str, offset: int | None) -> dict:
    """Strip the Metal preamble and report structured diagnostics with body
    line numbers when the offset probe succeeded, raw lines regardless."""
    text = _PREAMBLE_RE.sub("", message).strip()
    diagnostics = []
    for line in text.splitlines():
        m = _DIAG_RE.match(line)
        if not m:
            continue
        raw_line = int(m.group(2))
        diagnostics.append({
            "body_line": raw_line - offset if offset is not None else None,
            "raw_line": raw_line,
            "column": int(m.group(3)),
            "severity": m.group(4),
            "message": m.group(5),
        })
    return {"diagnostics": diagnostics, "line_offset": offset, "text": text[:_TEXT_CAP]}


def evaluate_ladder(spec: LadderSpec) -> Verdict:
    """Gates 2-8 (phase validate) or smoke + determinism + the ship clock
    (phase score), in spec order, first failure stops. Gate 1 ran in the
    parent. Every library reference here comes from replaying the span in
    this process; the pool is NaN-saturated before the first replay."""
    kspec = KernelSpec.from_json(json.dumps(spec.kernel))
    rtol = float(spec.tolerances["rtol"])
    atol = float(spec.tolerances["atol"])
    changing = spec.assoc_tag == "changing"
    primary_nodes = nodes_from_json(spec.nodes_json)
    in_ids = tuple(spec.input_ids)
    out_ids = tuple(spec.output_ids)
    out_names = tuple(kspec.output_names)
    prim = spec.eval_sets[0]
    prim_binds = [load_set(p) for p in prim.inputs_paths]
    prim_refs = [_ordered(load_set(p), out_ids, p) for p in prim.reference_paths]
    validate = spec.phase == "validate"

    gates_passed: list[str] = []
    timing: dict[str, float] = {}
    if prim.t_library_ms is not None:
        timing["t_library_ms"] = prim.t_library_ms
    info: dict = {"fallback_engaged": {}}
    poison_init = 0.0 if kspec.atomic_outputs else float("nan")

    def fail(gate: str, detail: dict) -> Verdict:
        return Verdict(False, gate, tuple(gates_passed), detail, timing)

    def launch(binds: dict, ids: tuple[int, ...]) -> list[mx.array]:
        # every correctness launch is poisoned; timed paths use cand_pass
        outs = call(kspec, [binds[i] for i in ids], init_value=poison_init)
        mx.eval(outs)
        return outs

    def library(nodes, binds: dict, ids: tuple[int, ...]) -> list[mx.array]:
        res = replay(nodes, binds, ids)
        outs = [res[i] for i in ids]
        mx.eval(outs)
        return outs

    def resolve_span(es):
        """The span and projected boundary ids for one eval set."""
        if es.nodes_json is None:
            return primary_nodes, in_ids, out_ids
        nodes = nodes_from_json(es.nodes_json)
        if len(nodes) != len(primary_nodes) or any(
            a.op != b.op for a, b in zip(primary_nodes, nodes)
        ):
            raise ValueError(f"eval set {es.label!r}: span op sequence differs from the region's")
        return (nodes, _project_ids(primary_nodes, nodes, in_ids),
                _project_ids(primary_nodes, nodes, out_ids))

    saturate_pool(a.nbytes for refs in prim_refs for a in refs)

    # gate 2, compile: the poisoned probe eval builds the pipeline; a broken
    # build surfaces only here (plan 5.10). Its outputs feed gate 3.
    outs0: list[mx.array] = []

    def probe() -> list[mx.array]:
        outs0[:] = launch(prim_binds[0], in_ids)
        return outs0

    try:
        timing["first_run_ms"] = time_once(probe) * 1e3
    except RuntimeError as e:
        if _BUILD_FAILURE in str(e):
            return fail("compile", _compile_detail(
                str(e), _probe_offset(kspec, [prim_binds[0][i] for i in in_ids])))
        return fail("compile", {"probe_eval_error": str(e)[:_TEXT_CAP]})
    gates_passed.append("compile")

    # gate 3, poison: unwritten output elements are NaN deterministically; any
    # non-finite value where the reference is finite fails.
    if validate:
        counts = {}
        for name, c, r in zip(out_names, outs0, prim_refs[0]):
            _check_contract(name, c, r)
            bad = mx.logical_and(mx.logical_not(mx.isfinite(c)), mx.isfinite(r))
            counts[name] = int(mx.sum(bad).item())
        if any(counts.values()):
            return fail("poison", {"non_finite_over_finite_ref": counts})
        gates_passed.append("poison")

    # Library warm + the two wobble replays. The first replay is the fresh-child
    # JIT warm the watchdog needs; the pair's max_abs_diff is the run-to-run
    # wobble floor for the preserving compares.
    lib_a = library(primary_nodes, prim_binds[0], out_ids)
    lib_b = library(primary_nodes, prim_binds[0], out_ids)
    prim_wobble = [max_abs_diff(a, b) for a, b in zip(lib_a, lib_b)]

    kins = [[b[i] for i in in_ids] for b in prim_binds]

    def lib_pass(i: int) -> list[mx.array]:
        res = replay(primary_nodes, prim_binds[i % len(prim_binds)], out_ids)
        return [res[o] for o in out_ids]

    def cand_pass(i: int) -> list[mx.array]:
        return call(kspec, kins[i % len(kins)])

    def per_pass_ms(fn_pass, n: int) -> float:
        return time_once(lambda: [fn_pass(i) for i in range(n)]) / n * 1e3

    # gate 4, watchdog (slow half): timed launches against the library region
    # time re-measured in this process, never the parent's number.
    if validate:
        t_lib_ms = per_pass_ms(lib_pass, WATCHDOG_ITERS)
        timing["library_run_ms"] = t_lib_ms
        timing["timed_run_ms"] = per_pass_ms(cand_pass, WATCHDOG_ITERS)
        if timing["timed_run_ms"] > WATCHDOG_FACTOR * t_lib_ms:
            return fail("watchdog", {
                "timed_run_ms": timing["timed_run_ms"],
                "library_run_ms": t_lib_ms,
                "watchdog_factor": WATCHDOG_FACTOR,
            })
        gates_passed.append("watchdog")

    # The assoc-changing floor: the spread of library-vs-golden error across
    # the primary k sets, floored by a small multiple of output-dtype epsilon
    # (relative error, so epsilon-relative is "epsilon at observed scale").
    changing_floor = spec.changing_floor
    if changing and changing_floor is None:
        errs = []
        for binds in prim_binds:
            g = golden_outputs(primary_nodes, binds, out_ids)
            goldens = [g[i] for i in out_ids]
            mx.eval(goldens)
            errs.append(golden_err(library(primary_nodes, binds, out_ids), goldens))
        eps = max(_EPS_REL.get(d, 2.0 ** -23) for d in kspec.output_dtypes)
        changing_floor = max(max(errs) - min(errs), EPS_FLOOR_MULTIPLE * eps)
        info["changing_floor"] = changing_floor

    def preserving_mismatch(kouts, refs, wobble) -> dict | None:
        for name, c, r, w in zip(out_names, kouts, refs, wobble):
            res = numeric_compare(c, r, rtol, atol, wobble_floor=w)
            if not res.passed:
                return {"output": name, "reason": res.reason,
                        "max_excess": res.max_excess, "note": res.detail}
        return None

    def changing_mismatch(nodes, binds, ids, kouts) -> dict | None:
        g = golden_outputs(nodes, binds, ids)
        goldens = [g[i] for i in ids]
        mx.eval(goldens)
        err_c = golden_err(kouts, goldens)
        err_l = golden_err(library(nodes, binds, ids), goldens)
        if not golden_passes(err_c, err_l, spec.kappa, changing_floor):
            # max_excess is the only judge-visible number; the rest is for
            # harness records and gets stripped by loop._safe_detail
            return {"err_candidate": err_c, "err_library": err_l,
                    "kappa": spec.kappa, "floor": changing_floor,
                    "max_excess": err_c - (spec.kappa * err_l + changing_floor)}
        return None

    def check_eval_set(es, nodes, s_in, s_out, wobble=None) -> dict | None:
        """The gate 5/6 body: stored-reference compare across the set's k
        input sets (kills cached-by-shape), then the value regimes with
        references computed on the spot via replay, never stored."""
        binds_list = [load_set(p) for p in es.inputs_paths]
        refs_list = [_ordered(load_set(p), s_out, p) for p in es.reference_paths]
        if wobble is None:
            wa = library(nodes, binds_list[0], s_out)
            wb = library(nodes, binds_list[0], s_out)
            wobble = [max_abs_diff(a, b) for a, b in zip(wa, wb)]
        for j, (binds, refs) in enumerate(zip(binds_list, refs_list)):
            kouts = launch(binds, s_in)
            bad = (changing_mismatch(nodes, binds, s_out, kouts) if changing
                   else preserving_mismatch(kouts, refs, wobble))
            if bad:
                bad.update({"eval_set": es.label, "input_set": j, "kind": "stored_ref"})
                return bad
        base_inputs = [binds_list[0][i] for i in s_in]
        for regime in REGIMES:
            found = None
            for magnitude in REGIME_MAGNITUDES.get(regime, (None,)):
                knobs = {} if magnitude is None else (
                    {"scale_up": magnitude} if regime == "scaled_up" else {"outlier": magnitude})
                rinputs = value_regimes(base_inputs, seed=spec.seed,
                                        weights=spec.weight_inputs, **knobs)[regime]
                rbinds = dict(zip(s_in, rinputs))
                ref = library(nodes, rbinds, s_out)
                if magnitude is None or all(_finite_where(r, b) for r, b in zip(ref, refs_list[0])):
                    found = (rbinds, ref, magnitude)
                    break
            if found is None:
                info.setdefault("regime_skipped", {})[regime] = \
                    "the library's own output overflows at every magnitude tried"
                continue
            rbinds, ref, magnitude = found
            if magnitude is not None:
                info.setdefault("regime_magnitude", {})[regime] = magnitude
            kouts = launch(rbinds, s_in)
            if changing:
                bad = changing_mismatch(nodes, rbinds, s_out, kouts)
            else:
                bad = preserving_mismatch(kouts, ref, wobble)
            if bad:
                bad.update({"eval_set": es.label, "regime": regime, "kind": "regime"})
                return bad
        return None

    # gate 5, smoke on the first eval set.
    bad = check_eval_set(prim, primary_nodes, in_ids, out_ids, wobble=prim_wobble)
    if bad:
        return fail("smoke", bad)
    gates_passed.append("smoke")

    # gate 6, the same on every other non-correctness-only eval set.
    if validate:
        for es in spec.eval_sets[1:]:
            if es.correctness_only:
                continue
            bad = check_eval_set(es, *resolve_span(es))
            if bad:
                return fail("workloads", bad)
        gates_passed.append("workloads")

    # gate 7, shape sweep: correctness only, fallback routing evaluated per
    # set and logged, plus one transposed-input variant on the primary set.
    if validate:
        for es in spec.eval_sets[1:]:
            if not es.correctness_only:
                continue
            nodes, s_in, s_out = resolve_span(es)
            binds_list = [load_set(p) for p in es.inputs_paths]
            refs_list = [_ordered(load_set(p), s_out, p) for p in es.reference_paths]
            wa = library(nodes, binds_list[0], s_out)
            wb = library(nodes, binds_list[0], s_out)
            wobble = [max_abs_diff(a, b) for a, b in zip(wa, wb)]
            for j, (binds, refs) in enumerate(zip(binds_list, refs_list)):
                fired = fallback_fires(kspec, [binds[i] for i in s_in])
                key = es.label if j == 0 else f"{es.label}#{j}"
                info["fallback_engaged"][key] = fired
                kouts = library(nodes, binds, s_out) if fired else launch(binds, s_in)
                bad = (changing_mismatch(nodes, binds, s_out, kouts) if changing
                       else preserving_mismatch(kouts, refs, wobble))
                if bad:
                    bad.update({"eval_set": es.label, "input_set": j, "kind": "sweep"})
                    if kspec.fallback_predicate is not None and not fired:
                        bad["fallback_declared_but_dead"] = True
                    bad["fallback_engaged"] = dict(info["fallback_engaged"])
                    return fail("sweep", bad)
        variant: dict[int, mx.array] = {}
        has_matrix = False
        for i in in_ids:
            a = prim_binds[0][i]
            if a.ndim >= 2:
                # logically identical, non-contiguous: the raw buffer holds the
                # transposed layout, so flat indexing under
                # ensure_row_contiguous=False reads the wrong order (spike_06)
                variant[i] = mx.transpose(mx.contiguous(mx.transpose(a)))
                has_matrix = True
            else:
                variant[i] = a
        if has_matrix and not fallback_fires(kspec, [variant[i] for i in in_ids]):
            mx.eval(list(variant.values()))
            kouts = launch(variant, in_ids)
            bad = (changing_mismatch(primary_nodes, variant, out_ids, kouts) if changing
                   else preserving_mismatch(kouts, prim_refs[0], prim_wobble))
            if bad:
                bad.update({"eval_set": prim.label, "kind": "transposed_variant"})
                return fail("sweep", bad)
        gates_passed.append("sweep")

    # gate 8, determinism first (three re-poisoned runs bitwise identical,
    # both tags: a racy kernel dies here no matter what else passes), then the
    # tag's numeric rule on the primary set.
    runs = [launch(prim_binds[0], in_ids) for _ in range(DETERMINISM_RUNS)]
    for r_i in range(1, DETERMINISM_RUNS):
        for name, a, b in zip(out_names, runs[0], runs[r_i]):
            if not _bitwise_equal(a, b):
                return fail("determinism", {"nondeterministic_output": name, "run": r_i})
    if changing:
        bad = changing_mismatch(primary_nodes, prim_binds[0], out_ids, runs[0])
    else:
        bad = preserving_mismatch(runs[0], prim_refs[0], prim_wobble)
    if bad:
        bad.update({"eval_set": prim.label, "kind": "numeric"})
        return fail("determinism", bad)
    gates_passed.append("determinism")

    # gate 9, the region ship clock: looped replay, candidate and library
    # interleaved ABBA in one session, k input sets rotating, all outputs kept
    # live in the final eval, the library re-measured now and never reused.
    if spec.phase == "score":
        session = Session()
        t_est_ms = per_pass_ms(lib_pass, CLOCK_EST_ITERS)
        iters = max(1, min(int(CLOCK_SAMPLE_TARGET_S * 1e3 / max(t_est_ms, 1e-3)),
                           CLOCK_MAX_ITERS))

        def lib_loop() -> list:
            return [lib_pass(i) for i in range(iters)]

        def cand_loop() -> list:
            return [cand_pass(i) for i in range(iters)]

        comp = paired_compare(session, lib_loop, cand_loop, pairs=spec.clock_pairs)
        library_ms = comp.median_baseline_ms / iters
        win_ms = comp.median_delta_ms / iters
        sigma_ms = comp.sigma_ms / iters
        margin_ms = max(0.01 * library_ms, 3.0 * sigma_ms)
        ship = bool(win_ms > margin_ms and win_ms >= spec.min_win_ms)
        timing.update({
            "region_ms": statistics.median(comp.candidate_ms) / iters,
            "library_ms": library_ms,
            "win_ms": win_ms,
            "sigma_ms": sigma_ms,
            "clock_iters": iters,
            "clock_deltas": comp.n,
        })
        info.update({"ship": ship, "margin_ms": margin_ms, "min_win_ms": spec.min_win_ms})
        gates_passed.append("clock")

    return Verdict(True, None, tuple(gates_passed), info, timing)


def _ordered(loaded: dict[int, mx.array], ids: tuple[int, ...], path: str) -> list[mx.array]:
    missing = [i for i in ids if i not in loaded]
    if missing:
        raise ValueError(f"{path} is missing arrays {missing}")
    return [loaded[i] for i in ids]


def _project_ids(primary, other, ids: tuple[int, ...]) -> tuple[int, ...]:
    """Map boundary ids from the primary span into a same-shaped span at
    another size, by each id's first (node, slot) occurrence."""
    pos: dict[int, tuple[str, int, int]] = {}
    for n, node in enumerate(primary):
        for s, aid in enumerate(node.in_arrays):
            pos.setdefault(aid, ("in", n, s))
        for s, aid in enumerate(node.out_arrays):
            pos.setdefault(aid, ("out", n, s))
    projected = []
    for aid in ids:
        kind, n, s = pos[aid]
        node = other[n]
        projected.append(node.in_arrays[s] if kind == "in" else node.out_arrays[s])
    return tuple(projected)


def _finite_where(candidate: mx.array, base: mx.array) -> bool:
    """candidate is finite everywhere base is."""
    return bool(mx.all(mx.isfinite(candidate) | ~mx.isfinite(base)).item())


def _bitwise_equal(a: mx.array, b: mx.array) -> bool:
    name = str(a.dtype).removeprefix("mlx.core.")
    if name in _BITS:
        a, b = mx.view(a, _BITS[name]), mx.view(b, _BITS[name])
    return bool(mx.array_equal(a, b).item())


if __name__ == "__main__":
    main()
