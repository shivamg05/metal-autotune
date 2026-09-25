"""Check the patched model against the original on this bundle's saved inputs.

Copied into each artifact as validate.py. Imports nothing from the optimizer.
The job uses this same check, so search and export enforce one policy. Both
models run plain (no mx.compile), as every correctness check in the job does.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import mlx.core as mx

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "runtime"))

from autotuner_runtime.swap import flatten_arrays  # noqa: E402

FLOOR_MULTIPLE = 4.0
EPS_MULTIPLE = 2.0
COSINE_EPS_MULTIPLE = 32.0
_EPS = {"float16": 9.77e-4, "bfloat16": 7.81e-3, "float32": 1.19e-7}
_FLOAT = (mx.float16, mx.bfloat16, mx.float32)


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def saved_workloads(metadata):
    """(label, inputs, role) for every saved workload, in bundle.json order."""
    out = []
    for label, entry in metadata["workloads"].items():
        data = mx.load(str(_HERE / entry["file"]))
        out.append((label, [data[f"i{k}"] for k in range(len(data))], entry["role"]))
    return out


def _flatten(tree):
    # freeze each observation: a model may return a buffer it overwrites later
    out = [mx.array(a) for a in flatten_arrays(tree)]
    mx.eval(out)
    return out


def _structure(tree):
    if isinstance(tree, mx.array):
        return ("array", tuple(tree.shape), str(tree.dtype))
    if isinstance(tree, dict):
        return ("dict", tuple((k, _structure(v)) for k, v in tree.items()))
    if isinstance(tree, (tuple, list)):
        return (type(tree).__name__, tuple(_structure(v) for v in tree))
    return (type(tree).__name__, repr(tree))


def _max_abs_diff(a, b):
    """Worst |a - b|, 0.0 where both share a non-finite pattern, inf when the
    patterns differ or a non-float output changed."""
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
        return float("inf")
    if a.size == 0:
        return 0.0
    if a.dtype not in _FLOAT:
        return 0.0 if mx.array_equal(a, b).item() else float("inf")
    af, bf = a.astype(mx.float32), b.astype(mx.float32)
    pattern = (mx.isfinite(bf) & mx.isfinite(af)) | (mx.isnan(bf) & mx.isnan(af)) | (mx.isinf(bf) & (af == bf))
    if not mx.all(pattern).item():
        return float("inf")
    return float(mx.max(mx.where(mx.isfinite(bf), mx.abs(af - bf), mx.zeros_like(bf))).item())


def _cosine(a, b):
    if len(a) != len(b):
        return -1.0
    worst = 1.0
    for x, y in zip(a, b):
        if x.shape != y.shape or x.dtype != y.dtype:
            return -1.0
        if x.size == 0:
            continue
        xf = mx.where(mx.isfinite(x), x, 0).astype(mx.float32).reshape(-1)
        yf = mx.where(mx.isfinite(y), y, 0).astype(mx.float32).reshape(-1)
        sx, sy = mx.abs(xf).max().item(), mx.abs(yf).max().item()
        if sx == 0.0 or sy == 0.0:
            worst = min(worst, 1.0 if sx == sy else 0.0)
            continue
        xf, yf = xf / sx, yf / sy
        denom = math.sqrt(float(mx.sum(xf * xf).item()) * float(mx.sum(yf * yf).item()))
        cosine = float(mx.sum(xf * yf).item()) / denom
        if not math.isfinite(cosine):
            return -1.0
        worst = min(worst, max(-1.0, min(1.0, cosine)))
    return worst


def _eps_term(x):
    if x.size == 0:
        return 0.0
    eps = _EPS.get(str(x.dtype).removeprefix("mlx.core."), 0.0)
    scale = mx.abs(mx.where(mx.isfinite(x), x, 0).astype(mx.float32)).max().item() or 1.0
    return EPS_MULTIPLE * eps * scale


def _legacy_check_outputs(original_run, patched_run, name, *, exact=False):
    """Patched vs original must sit on the original-vs-original floor, per
    output, plus one rounding step of the output dtype at its value scale; the
    output cosine is held to the same floor."""
    tree1 = original_run()
    ref1 = _flatten(tree1)
    signature = _structure(tree1)
    tree2 = original_run()
    ref2 = _flatten(tree2)
    got_tree = patched_run()
    got = _flatten(got_tree)
    result = {"name": name, "floor_max_abs": float("inf"), "max_abs": float("inf"), "allowance": 0.0,
              "cosine": -1.0, "passed": False, "reason": ""}
    if not ref1 or signature != _structure(tree2) or signature != _structure(got_tree):
        return {**result, "reason": "output structure, shape or dtype changed"}
    if exact:
        from autotuner_runtime.exact import bitwise_equal
        passed = all(bitwise_equal(a, b) and bitwise_equal(a, c)
                     for a, b, c in zip(ref1, ref2, got))
        return {**result, "passed": passed, "max_abs": 0.0 if passed else float("inf"),
                "allowance": 0.0, "rule": "exact", "reason": "" if passed else "bitwise output mismatch"}
    allowances = [max(FLOOR_MULTIPLE * _max_abs_diff(x, y), _eps_term(x)) for x, y in zip(ref1, ref2)]
    diffs = [_max_abs_diff(x, y) for x, y in zip(ref1, got)]
    floor = max(_max_abs_diff(x, y) for x, y in zip(ref1, ref2))
    floor_cosine = _cosine(ref1, ref2)
    cosine = _cosine(ref1, got)
    cosine_allowance = max(FLOOR_MULTIPLE * (1.0 - floor_cosine), COSINE_EPS_MULTIPLE * _EPS["float32"])
    finite = math.isfinite(floor) and all(math.isfinite(d) for d in diffs)
    absolute_ok = finite and all(d <= allowed for d, allowed in zip(diffs, allowances))
    cosine_ok = 1.0 - cosine <= cosine_allowance
    reason = ("non-finite output pattern changed" if not finite else
              "output error exceeds original-model floor" if not absolute_ok else
              "output cosine exceeds original-model floor" if not cosine_ok else "")
    return {**result, "floor_max_abs": floor, "max_abs": max(diffs), "allowance": max(allowances),
            "cosine": cosine, "passed": absolute_ok and cosine_ok, "reason": reason}


def _failure_detail(reference, candidate, check, output, *, exact, tolerances, comparison):
    """Describe one failing element; do not compare unrelated global maxima."""
    index = check.index
    if index is None and reference.size and reference.shape == candidate.shape and reference.dtype == candidate.dtype:
        left = mx.contiguous(reference.reshape(-1)).view(mx.uint8).reshape(reference.size, -1)
        right = mx.contiguous(candidate.reshape(-1)).view(mx.uint8).reshape(candidate.size, -1)
        different = mx.any(left != right, axis=1)
        flat = int(mx.argmax(different).item())
        parts = []
        for size in reversed(reference.shape):
            flat, offset = divmod(flat, size)
            parts.append(offset)
        index = tuple(reversed(parts))
    failure = {"comparison": comparison, "output": output,
               "index": None if index is None else list(index), "reason": check.reason,
               "detail": check.detail}
    if index is None:
        return failure
    ref, got = reference[index].item(), candidate[index].item()
    def number(value):
        return value if isinstance(value, (int, bool)) or math.isfinite(value) else str(value)
    rtol, atol = tolerances
    allowance = 0.0 if exact or reference.dtype not in _FLOAT else atol + rtol * abs(ref)
    finite = math.isfinite(ref) and math.isfinite(got)
    failure.update(reference=number(ref), candidate=number(got),
                   absolute_error=abs(got - ref) if finite else None,
                   allowance=number(allowance))
    return failure


def check_outputs(original_run, patched_run, name, *, exact=True, tolerances=None):
    """Compare every output to the untouched original under one fixed policy.

    Tolerance is elementwise atol + rtol * abs(original), with exact nonfloating
    outputs and matching nonfinite positions. Baseline repeat variance never
    widens this allowance. The same check runs in the job and exported bundle.
    """
    from autotuner_runtime.numeric import check, tolerance_for
    tree1 = original_run()
    ref1, signature = _flatten(tree1), _structure(tree1)
    tree2 = original_run()
    ref2, signature2 = _flatten(tree2), _structure(tree2)
    got_tree = patched_run()
    got, got_signature = _flatten(got_tree), _structure(got_tree)
    got_repeat_tree = patched_run()
    got_repeat, got_repeat_signature = _flatten(got_repeat_tree), _structure(got_repeat_tree)
    result = {"name": name, "floor_max_abs": float("inf"), "max_abs": float("inf"),
              "allowance": 0.0, "cosine": -1.0, "passed": False,
              "rule": "exact" if exact else "baseline_tolerance", "reason": "", "failure": None}
    if (not ref1 or signature != signature2 or signature != got_signature
            or signature != got_repeat_signature):
        return {**result, "reason": "output structure, shape or dtype changed"}
    checks, repeats, candidate_repeats, allowances = [], [], [], []
    for original, repeat, candidate, candidate_repeat in zip(ref1, ref2, got, got_repeat):
        rtol, atol = tolerance_for(original.dtype, tolerances)
        checks.append(check(candidate, original, exact=exact, rtol=rtol, atol=atol))
        repeats.append(check(repeat, original, exact=True))
        candidate_repeats.append(check(candidate_repeat, candidate, exact=True))
        finite = mx.where(mx.isfinite(original), original, 0).astype(mx.float32)
        scale = float(mx.max(mx.abs(finite)).item()) if original.size else 0.0
        allowances.append(0.0 if exact or original.dtype not in _FLOAT else atol + rtol * scale)
    stable = all(c.passed for c in repeats)
    candidate_stable = all(c.passed for c in candidate_repeats)
    passed = stable and candidate_stable and all(c.passed for c in checks)
    reason = ("original model is not bitwise repeatable" if not stable
              else "patched model is not bitwise repeatable" if not candidate_stable
              else next((c.reason for c in checks if not c.passed), ""))
    failure = None
    failures = (repeats, ref1, ref2, "original_repeat", True) if not stable else (
        (candidate_repeats, got, got_repeat, "patched_repeat", True) if not candidate_stable else
        (checks, ref1, got, "patched_vs_original", exact))
    for output, (check_result, reference, candidate) in enumerate(zip(*failures[:3])):
        if not check_result.passed:
            failure = _failure_detail(reference, candidate, check_result, output,
                                      exact=failures[4], tolerances=tolerance_for(reference.dtype, tolerances),
                                      comparison=failures[3])
            break
    return {**result, "failure": failure, "floor_max_abs": max(_max_abs_diff(a, b) for a, b in zip(ref1, ref2)),
            "max_abs": max(_max_abs_diff(a, b) for a, b in zip(ref1, got)),
            "allowance": max(allowances), "cosine": _cosine(ref1, got),
            "passed": passed, "reason": reason}


def policy_arguments(metadata):
    """The explicitly recorded policy used by current bundles."""
    rule = metadata.get("correctness_rule", "exact")
    if rule not in {"exact", "baseline_tolerance"}:
        raise ValueError(f"unknown correctness rule: {rule}")
    tolerance = metadata.get("tolerances")
    return {"exact": rule == "exact",
            "tolerances": None if tolerance is None else (tolerance["rtol"], tolerance["atol"])}


def check_changing_outputs(original_run, patched_run, golden, name):
    """The job's golden-relative rule, using its saved fp32 observations."""
    ref_tree = original_run()
    ref, signature = _flatten(ref_tree), _structure(ref_tree)
    got_tree = patched_run()
    got = _flatten(got_tree)
    result = {"name": name, "floor_max_abs": float("inf"), "max_abs": float("inf"),
              "allowance": 0.0, "cosine": -1.0, "passed": False,
              "reason": "output structure, shape or dtype changed", "rule": "fp32_relative"}
    if not ref or signature != _structure(got_tree):
        return result
    if len(ref) != len(golden) or any(r.shape != g.shape for r, g in zip(ref, golden)):
        return {**result, "reason": "fp32 golden output structure or shape changed"}
    if any(r.dtype not in _FLOAT and not mx.array_equal(r, c).item() for r, c in zip(ref, got)):
        return {**result, "reason": "non-floating output changed"}

    def error(outputs):
        worst = 0.0
        for value, reference, original in zip(outputs, golden, ref):
            if original.dtype not in _FLOAT or not value.size:
                continue
            cf, gf = value.astype(mx.float32), reference.astype(mx.float32)
            scale = max(float(mx.max(mx.abs(mx.where(mx.isfinite(gf), gf, 0))).item()), 1e-6)
            pattern = ((mx.isfinite(cf) & mx.isfinite(gf)) | (mx.isnan(cf) & mx.isnan(gf))
                       | (mx.isinf(gf) & (cf == gf)))
            rel = mx.abs(cf - gf) / mx.maximum(mx.abs(gf), scale)
            rel = mx.where(mx.isfinite(gf) & pattern, rel,
                           mx.where(pattern, mx.zeros_like(rel), mx.array(float("inf"))))
            worst = max(worst, float(mx.max(rel).item()))
        return worst

    reference_error, candidate_error = error(ref), error(got)
    allowance = 1.25 * reference_error + _EPS["float32"]
    passed = math.isfinite(reference_error) and math.isfinite(candidate_error) and candidate_error <= allowance
    return {**result, "floor_max_abs": reference_error, "max_abs": candidate_error,
            "allowance": allowance, "cosine": _cosine(ref, got), "passed": passed,
            "reason": "fp32 golden relative error"}


def paced(run):
    """Cool after each complete observation, without a short GPU-window limit."""
    from autotuner_runtime.sequence import BenchSession
    session = BenchSession()
    def call():
        result = None
        def execute():
            nonlocal result
            result = run()
            return result
        try:
            session.timed(execute)
            return result
        finally:
            session.settle()
    return call


def validate(original=None, patched=None, *, include_sequences=False):
    """One check per saved workload. Models default to the bundle's own
    measured execution mode; pass callables to check models built elsewhere."""
    metadata = json.loads((_HERE / "bundle.json").read_text())
    if metadata.get("correctness_rule") == "fp32_relative":
        missing = set(metadata["workloads"]) - set(metadata.get("goldens", {}))
        if missing:
            raise ValueError("recovery checkpoint lacks fp32 references for: "
                             + ", ".join(sorted(missing)) + "; finish validation before export")
    if original is None or patched is None:
        loader = _module("_artifact_load", _HERE / "load.py")
        if original is None:
            original = loader.load(patched=False)
        if patched is None:
            patched = loader.load(share_weights_with=original.model)
    results = []
    for label, inputs, role in saved_workloads(metadata):
        golden_file = metadata.get("goldens", {}).get(label)
        if golden_file:
            data = mx.load(str(_HERE / golden_file))
            result = check_changing_outputs(lambda: original(*inputs), lambda: patched(*inputs),
                                            [data[f"o{k}"] for k in range(len(data))], label)
        elif metadata.get("correctness_rule") == "preserving":
            result = _legacy_check_outputs(lambda: original(*inputs), lambda: patched(*inputs), label)
        else:
            def observe(step):
                from autotuner_runtime.state import ContextStep, correctness_call
                managed = getattr(step, "model", None)
                if isinstance(managed, ContextStep):
                    return correctness_call(managed, inputs)
                return correctness_call(step, inputs)
            result = check_outputs(paced(lambda: observe(original)), paced(lambda: observe(patched)), label,
                                   **policy_arguments(metadata))
        results.append({**result, "role": role})
        if (include_sequences and result["passed"] and role == "declared"
                and metadata.get("correctness_rule") in {"exact", "baseline_tolerance"}):
            from autotuner_runtime.state import sequence_observation
            steps = metadata.get("final_benchmark", {}).get("steps", 10)
            runs = [paced(lambda model=model: sequence_observation(model, inputs, steps))
                    for model in (original, patched)]
            sequence = check_outputs(*runs, label + ":sequence", **policy_arguments(metadata))
            results.append({**sequence, "role": "sequence", "steps": steps})
    return results


def describe(result):
    verdict = "PASS" if result["passed"] else f"FAIL ({result['reason']})"
    failure = result.get("failure")
    if failure:
        location = f"output {failure['output']} at {failure['index']}"
        values = (f", error {failure['absolute_error']}, allowed {failure['allowance']}"
                  if failure.get("absolute_error") is not None else "")
        return f"{verdict} {result['name']}: {failure['comparison']}, {location}{values}; {failure['detail']}"
    metric = "scaled fp32 error" if result.get("rule") == "fp32_relative" else "max abs difference"
    return f"{verdict} {result['name']}: {metric} {result['max_abs']:.3g}, cosine {result['cosine']:.6f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", action="store_true", help="also check every consecutive step and final cache state")
    args = parser.parse_args()
    results = validate(include_sequences=args.sequences)
    for result in results:
        print(describe(result))
    failed = [r for r in results if not r["passed"]]
    print(f"validate: {len(results) - len(failed)} of {len(results)} workloads match the original")
    sys.exit(1 if failed else 0)
