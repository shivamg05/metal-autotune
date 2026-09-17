"""The ladder's gates, run inside one sandbox child on one kernel.

Phase "validate" runs gates 2 to 8 with shader validation on. Phase "score"
re-runs smoke and determinism on the clean pipeline and then runs the ship
clock, because validation recompiles pipelines and the timed pipeline must
be the checked pipeline. Every library reference here comes from replaying
the span in this process, and the buffer pool is NaN-saturated before the
first replay.

Poison rule: every correctness launch runs with init_value=nan, except that
atomic_outputs kernels get init_value=0.0, because atomic accumulation is
only well-defined from an identity and NaN would fail every legitimate atomic
kernel. The value gates plus the determinism gate police atomics instead.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import replace

import mlx.core as mx

from autotuner.ladder.numeric import REGIMES, value_regimes
from autotuner_runtime.numeric import check as numeric_check, tolerance_for
from autotuner_runtime.exact import bitwise_equal
from autotuner.ladder.static_checks import launch_resource_failure
from autotuner.measure.clocks import (CLOCK_TARGET_MS, chained_loop, comparison_from_samples,
                                      link_input, link_loop, loop_iterations, paired_means,
                                      sample_group, timing_sets)
from autotuner.measure.probe import array_specs, stream_probe
from autotuner.measure.session import Session, time_once
from autotuner.regions.store import load_set
from autotuner.sandbox.poison import saturate_pool
from autotuner.sandbox.protocol import WATCHDOG_FACTOR, LadderSpec, Verdict
from autotuner.sandbox.watchdog import gpu_window
from autotuner.trace.replay import compile_replay, prepare_replay, replay
from autotuner.trace.serialize import nodes_from_json
from autotuner_runtime.kernels import KernelSpec, call, fallback_fires

_BUILD_FAILURE = "Unable to build metal library from source"
_DEVICE_FAILURE = "Command buffer execution failed"
_PREAMBLE_RE = re.compile(r"^\[metal::Device\] Unable to build metal library from source\s*")
_DIAG_RE = re.compile(r"^(\S+):(\d+):(\d+): (error|warning|note): (.*)$")
_PROBE_MARKER = "mao_line_probe"
_TEXT_CAP = 4000


class _LaunchRejected(Exception):
    pass


class _OriginalUnstable(Exception):
    pass

DETERMINISM_RUNS = 3          # spec-fixed
WATCHDOG_ITERS = 4            # passes per arm in the watchdog's timed comparison
# The magnitude regimes step down until the library's own output stays finite
# wherever it was finite on the real data: past that point the reference
# means nothing and no kernel could be written to match it.
REGIME_MAGNITUDES = {"scaled_up": (1e3, 1e2, 1e1), "outliers": (1e4, 1e3, 1e2)}


def _probe_offset(kspec: KernelSpec, inputs: list[mx.array]) -> int | None:
    """The compile-error line offset is per kernel: it counts utils.h plus the
    generated signature, which grows with the IO list.
    Measure it by compiling this kernel's body behind a deliberate #error on
    body line 1; the reported line minus one is the offset. Prepending a line
    changes nothing the signature generator scans for."""
    if kspec.stages:
        return None  # different signatures/headers per stage; retain raw compiler lines
    probe = replace(
        kspec,
        kernel_id=kspec.kernel_id + "_lineprobe",
        name=kspec.name + "_lineprobe",
        source=f"#error {_PROBE_MARKER}\n{kspec.source}",
    )
    try:
        with gpu_window():
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
    """Pace validation work too, including early failures and loop sizing."""
    session = Session()
    started = time.perf_counter()
    verdict = None
    try:
        verdict = _evaluate_ladder(spec, session)
    except _OriginalUnstable as error:
        verdict = Verdict(False, "determinism", (), {"reason": str(error)})
    except _LaunchRejected as error:
        verdict = Verdict(False, "static", (), {"reason": str(error)})
    except RuntimeError as error:
        if _DEVICE_FAILURE not in str(error):
            raise
        verdict = Verdict(False, "subprocess", (), {
            "reason": str(error)[:_TEXT_CAP], "abort_job": True,
            "kernel_id": spec.kernel.get("kernel_id"), "phase": spec.phase})
    finally:
        # Correctness/JIT work outside the timed arms still heats the chip.
        # Account for its elapsed cost without charging already-paid idles.
        mx.synchronize()
        extra = max(0.0, time.perf_counter() - started - session.idled_s - session.work_s)
        session._debt_s += extra
        # Only a completed verdict can transfer responsibility. An exception
        # must settle here because the parent will receive no deadline.
        if spec.defer_cooling and verdict is not None:
            verdict.timing["cooling_ready_at"] = session.defer_settle()
        else:
            session.settle()
        if verdict is not None:
            verdict.timing["pacing_work_s"] = session.work_s + extra
            verdict.timing["pacing_idle_s"] = session.idled_s
    return verdict


def _evaluate_ladder(spec: LadderSpec, session: Session) -> Verdict:
    """Gates 2-8 (phase validate) or smoke + determinism + the ship clock
    (phase score), in spec order, first failure stops. Gate 1 ran in the
    parent. Every library reference here comes from replaying the span in
    this process; the pool is NaN-saturated before the first replay."""
    kspec = KernelSpec.from_json(json.dumps(spec.kernel))
    tolerance_override = (float(spec.tolerances["rtol"]), float(spec.tolerances["atol"])) if spec.tolerances else None
    changing = spec.assoc_tag == "changing"
    primary_nodes = nodes_from_json(spec.nodes_json)
    in_ids = tuple(spec.input_ids)
    out_ids = tuple(spec.output_ids)
    # Static checks require real region outputs first, followed by tmp<N>
    # allocations. Scratch is GPU workspace, not part of the numeric contract.
    n_outputs = len(out_ids)
    out_names = tuple(kspec.output_names[:n_outputs])
    prim = spec.eval_sets[0]
    prim_binds = [load_set(p) for p in prim.inputs_paths]
    prim_refs = [_ordered(load_set(p), out_ids, p) for p in prim.reference_paths]
    validate = spec.phase == "validate"

    gates_passed: list[str] = []
    timing: dict[str, float] = {}
    if prim.t_library_ms is not None:
        timing["t_library_ms"] = prim.t_library_ms
    info: dict = {"fallback_engaged": {}, "correctness_rule": "tolerance" if changing else "exact"}
    if changing:
        info["tolerances"] = {name: dict(zip(("rtol", "atol"), tolerance_for(dtype, tolerance_override)))
                              for name, dtype in zip(out_names, kspec.output_dtypes)}
    poison_init = 0.0 if kspec.atomic_outputs else float("nan")

    def fail(gate: str, detail: dict) -> Verdict:
        return Verdict(False, gate, tuple(gates_passed), detail, timing)

    def launch(binds: dict, ids: tuple[int, ...]) -> list[mx.array]:
        # every correctness launch is poisoned; timed paths use cand_pass
        if fallback_fires(kspec, [binds[i] for i in ids]):
            raise _LaunchRejected("fallback_on_workload: this evaluated workload selects the "
                                  "original library instead of the proposed kernel")
        failure = launch_resource_failure(kspec, [binds[i].shape for i in ids])
        if failure is not None:
            raise _LaunchRejected(f"{failure.check}: {failure.detail}")
        with gpu_window():
            outs = call(kspec, [binds[i] for i in ids], init_value=poison_init)
            mx.eval(outs)
        return outs[:n_outputs]

    def library(nodes, binds: dict, ids: tuple[int, ...]) -> list[mx.array]:
        res = replay(nodes, binds, ids)
        outs = [res[i] for i in ids]
        mx.eval(outs)
        repeated = replay(nodes, binds, ids)
        repeated_outs = [repeated[i] for i in ids]
        mx.eval(repeated_outs)
        if not all(bitwise_equal(a, b) for a, b in zip(outs, repeated_outs, strict=True)):
            raise _OriginalUnstable("original region is not bitwise repeatable")
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
    # build surfaces only here. Its outputs feed gate 3.
    outs0: list[mx.array] = []

    def probe() -> list[mx.array]:
        outs0[:] = launch(prim_binds[0], in_ids)
        return outs0

    try:
        timing["first_run_ms"] = time_once(probe) * 1e3
    except RuntimeError as e:
        if _DEVICE_FAILURE in str(e):
            raise
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
            if tuple(c.shape) != tuple(r.shape) or c.dtype != r.dtype:
                return fail("poison", {"output": name, "reason": "shape",
                                       "note": f"{tuple(c.shape)} {c.dtype}, reference "
                                               f"{tuple(r.shape)} {r.dtype}"})
            bad = mx.logical_and(mx.logical_not(mx.isfinite(c)), mx.isfinite(r))
            counts[name] = int(mx.sum(bad).item())
        if any(counts.values()):
            return fail("poison", {"non_finite_over_finite_ref": counts})
        gates_passed.append("poison")

    # Require the original to repeat exactly. Never widen tolerance around
    # baseline nondeterminism, which can hide races or unstable state.
    library(primary_nodes, prim_binds[0], out_ids)

    # Every timed loop rotates a cache-defeating working set and chains its
    # passes, so the clock sees what a model step sees: bytes from memory,
    # one pass after another.
    timing_binds = timing_sets(prim_binds)
    weight_ids = {i for i, w in zip(in_ids, spec.weight_inputs) if w}
    link_id = link_input(prim_binds[0], weight_ids)

    if spec.baseline == "compiled":
        # the library arm as the baseline runs it: one compiled graph over the
        # span, compiled before the watchdog's first timed pass
        lib_pass = compile_replay(primary_nodes, prim_binds[0], out_ids)
        mx.eval(lib_pass(prim_binds[0]))
    else:
        lib_pass = prepare_replay(primary_nodes, prim_binds[0], out_ids)

    def cand_pass(binds: dict) -> list[mx.array]:
        return call(kspec, [binds[i] for i in in_ids])

    # A single evaluated pass is mostly submit-and-sync latency, so every
    # per-pass time here comes from a small loop, both arms alike.
    def per_pass_ms(fn_pass, n: int) -> float:
        return time_once(chained_loop(fn_pass, timing_binds, n, link_id)) / n * 1e3

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

    def output_mismatch(kouts, refs) -> dict | None:
        if len(kouts) != len(refs):
            return {"reason": "output count changed"}
        for name, c, r in zip(out_names, kouts, refs, strict=True):
            rtol, atol = tolerance_for(r.dtype, tolerance_override)
            res = numeric_check(c, r, exact=not changing, rtol=rtol, atol=atol)
            if not res.passed:
                return {"output": name, "reason": res.reason,
                        "max_excess": res.max_excess, "note": res.detail}
        return None

    def changing_mismatch(nodes, binds, ids, kouts) -> dict | None:
        # Original precision and original quantized weights are the reference.
        # No interpretation of arbitrary source or FP32 replay is required.
        return output_mismatch(kouts, library(nodes, binds, ids))

    def check_eval_set(es, nodes, s_in, s_out) -> dict | None:
        """The gate 5/6 body: stored-reference compare across the set's k
        input sets (kills cached-by-shape), then the value regimes with
        references computed on the spot via replay, never stored."""
        binds_list = [load_set(p) for p in es.inputs_paths]
        refs_list = [_ordered(load_set(p), s_out, p) for p in es.reference_paths]
        for j, (binds, refs) in enumerate(zip(binds_list, refs_list)):
            if not changing:
                library(nodes, binds, s_out)
            kouts = launch(binds, s_in)
            bad = (changing_mismatch(nodes, binds, s_out, kouts) if changing
                   else output_mismatch(kouts, refs))
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
                bad = output_mismatch(kouts, ref)
            if bad:
                bad.update({"eval_set": es.label, "regime": regime, "kind": "regime"})
                return bad
        return None

    # gate 5, smoke on the first eval set.
    bad = check_eval_set(prim, primary_nodes, in_ids, out_ids)
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
            for j, (binds, refs) in enumerate(zip(binds_list, refs_list)):
                if not changing:
                    library(nodes, binds, s_out)
                fired = fallback_fires(kspec, [binds[i] for i in s_in])
                key = es.label if j == 0 else f"{es.label}#{j}"
                info["fallback_engaged"][key] = fired
                kouts = library(nodes, binds, s_out) if fired else launch(binds, s_in)
                bad = (changing_mismatch(nodes, binds, s_out, kouts) if changing
                       else output_mismatch(kouts, refs))
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
                # ensure_row_contiguous=False reads the wrong order
                variant[i] = mx.transpose(mx.contiguous(mx.transpose(a)))
                has_matrix = True
            else:
                variant[i] = a
        if has_matrix and not fallback_fires(kspec, [variant[i] for i in in_ids]):
            mx.eval(list(variant.values()))
            kouts = launch(variant, in_ids)
            bad = (changing_mismatch(primary_nodes, variant, out_ids, kouts) if changing
                   else output_mismatch(kouts, prim_refs[0]))
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
            if not bitwise_equal(a, b):
                return fail("determinism", {"nondeterministic_output": name, "run": r_i})
    if changing:
        bad = changing_mismatch(primary_nodes, prim_binds[0], out_ids, runs[0])
    else:
        bad = output_mismatch(runs[0], prim_refs[0])
    if bad:
        bad.update({"eval_set": prim.label, "kind": "numeric"})
        return fail("determinism", bad)
    gates_passed.append("determinism")

    # gate 9, the region ship clock: looped replay, candidate and library
    # interleaved ABBA in one session, k input sets rotating, all outputs kept
    # live in the final eval, the library re-measured now and never reused.
    if spec.phase == "score":
        iters = loop_iterations(session.timed, lambda n: chained_loop(lib_pass, timing_binds, n, link_id),
                                CLOCK_TARGET_MS)
        lib_loop = chained_loop(lib_pass, timing_binds, iters, link_id)
        cand_loop = chained_loop(cand_pass, timing_binds, iters, link_id)
        probe = stream_probe(array_specs([prim_binds[0][i] for i in in_ids]), array_specs(prim_refs[0]))
        probe_loop = chained_loop(lambda b: probe([b[i] for i in in_ids]), timing_binds, iters, link_id)
        arms = {"library": lib_loop, "candidate": cand_loop, "probe": probe_loop}
        if spec.timing_incumbent is not None:
            incumbent = KernelSpec.from_json(json.dumps(spec.timing_incumbent))
            arms["incumbent"] = chained_loop(
                lambda b: call(incumbent, [b[i] for i in in_ids]), timing_binds, iters, link_id)
        if link_id is not None:
            arms["link"] = link_loop(timing_binds, iters, link_id)
        rows = {key: paired_means(values) for key, values in
                sample_group(session, arms, pairs=spec.clock_pairs,
                             **({"defer_cooling": True} if spec.defer_cooling else {})).items()}
        links = rows.get("link", [0.0] * len(rows["library"]))
        net = {key: [(value - link) / iters for value, link in zip(rows[key], links)]
               for key in arms if key != "link"}
        comp = comparison_from_samples(net["library"], net["candidate"])
        library_ms = comp.median_baseline_ms
        region_ms = statistics.median(net["candidate"])
        if library_ms <= 0 or region_ms <= 0:
            return fail("clock", {"reason": "region cost could not be separated from chain overhead"})
        win_ms, sigma_ms = comp.median_delta_ms, comp.sigma_ms
        floor_ms = max(statistics.median(net["probe"]), spec.compute_floor_ms, 0.0)
        margin_ms = max(0.01 * library_ms, 3.0 * sigma_ms)
        ship = bool(win_ms > margin_ms and win_ms >= spec.min_win_ms)
        if "incumbent" in net:
            current = comparison_from_samples(net["incumbent"], net["candidate"])
            # Reject resolved local regressions; ties still get a model test.
            # Local timing is a screening heuristic, never a model speed claim.
            info["incumbent_screen"] = {
                "kernel_id": incumbent.kernel_id,
                "incumbent_ms": current.median_baseline_ms,
                "win_ms": current.median_delta_ms, "sigma_ms": current.sigma_ms,
                "resolved_regression": current.loses_by(0.0),
            }
            ship = ship and not current.loses_by(0.0)
        timing.update({
            "region_ms": region_ms,
            "library_ms": library_ms,
            "floor_ms": floor_ms,
            "win_ms": win_ms,
            "sigma_ms": sigma_ms,
            "clock_iters": iters,
            "clock_sets": len(timing_binds),
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
