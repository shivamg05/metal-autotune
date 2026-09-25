"""Re-time the original and patched models on this bundle's saved inputs.

Copied into each artifact as benchmark.py. Imports nothing from the optimizer.
The protocol is the job's final check: whole runs of consecutive steps, each
run uninterrupted, the two models alternated in both orders with cooling
between runs. The correctness check from validate.py runs first.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import mlx.core as mx

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "runtime"))

from autotuner_runtime.sequence import BenchSession, compare_sequences, make_sequence, generation_throughput, throughput_text  # noqa: E402
from autotuner_runtime.stats import PairedComparison, workload_win  # noqa: E402


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def workload_steps(metadata, label, override=None):
    """Use the measured length; legacy bundles retain their manifest default."""
    if override is not None:
        return override
    final = metadata["final_benchmark"]
    if metadata.get("use_library_inference"):
        return final.get("steps", 20)
    return final.get("steps_by_workload", {}).get(label, final.get("steps", 20))


def main(argv=None):
    metadata = json.loads((_HERE / "bundle.json").read_text())
    final = metadata["final_benchmark"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=None,
                        help="override the saved length: generated tokens for library inference, otherwise consecutive forward steps")
    parser.add_argument("--pairs", type=int, default=final.get("pairs", 4), help="alternated run pairs, even and >= 4")
    parser.add_argument("--warmup-steps", type=int, default=final.get("warmup_steps", 3), help="minimum untimed steps before each run")
    parser.add_argument("--json", type=Path, help="also write the rows to this file")
    args = parser.parse_args(argv)

    loader = _module("_artifact_load", _HERE / "load.py")
    validator = _module("_artifact_validate", _HERE / "validate.py")
    settings = {"generated_tokens": workload_steps(metadata, None, args.steps)} if metadata.get("use_library_inference") else {}
    original = loader.load(patched=False, **settings)
    patched = loader.load(share_weights_with=original.model, **settings)
    checks = validator.validate(original, patched)
    for result in checks:
        print(validator.describe(result))
    if not all(result["passed"] for result in checks):
        print("benchmark: outputs differ from the original; not timing")
        return 1

    if metadata.get("use_library_inference") and metadata.get("baseline") == "compiled":
        original = loader.load(patched=False, measurement_baseline=True,
                               share_weights_with=original.model, **settings)

    session = BenchSession()
    rows = {}
    comparisons = {}
    for label, inputs, role in validator.saved_workloads(metadata):
        if role != "declared":
            continue
        steps = workload_steps(metadata, label, args.steps)
        runs = [make_sequence(step, inputs, steps) for step in (original, patched)]
        kind = "repeated_forward"
        if metadata.get("use_library_inference"):
            kind = "library_generation"
            runs = [lambda s=step: s(*inputs) for step in (original, patched)]
        elif metadata.get("context"):
            kind = "advancing_cache_fixed_tokens"
            runs = [lambda s=step: s.model.sequence(inputs, steps)
                    for step in (original, patched)]
            golden_file = metadata.get("sequence_goldens", {}).get(label)
            if metadata.get("goldens") and not golden_file:
                raise ValueError("bundle lacks the fp32 reference for advancing decode; re-export it")
            if golden_file:
                if steps != workload_steps(metadata, label):
                    raise ValueError("changing kernels need the saved sequence length for fp32 validation")
                data = mx.load(str(_HERE / golden_file))
                check = validator.check_changing_outputs(
                    *runs, [data[f"o{k}"] for k in range(len(data))], label + ":sequence")
            elif metadata.get("correctness_rule") == "preserving":
                check = validator._legacy_check_outputs(*runs, label + ":sequence")
            else:
                observations = [validator.paced(lambda s=step: s.model.sequence(inputs, steps, include_state=True))
                                for step in (original, patched)]
                check = validator.check_outputs(*observations, label + ":sequence",
                                                **validator.policy_arguments(metadata))
            print(validator.describe(check))
            if not check["passed"]:
                return 1
        elif metadata.get("correctness_rule") in {"exact", "baseline_tolerance"}:
            from autotuner_runtime.state import sequence_observation
            observations = [validator.paced(lambda s=step: sequence_observation(s, inputs, steps))
                            for step in (original, patched)]
            check = validator.check_outputs(*observations, label + ":sequence",
                                            **validator.policy_arguments(metadata))
            print(validator.describe(check))
            if not check["passed"]:
                return 1
        warm_length = final.get("warmup_steps_per_sample", {}).get(label, 1)
        warm = []
        for step in (original, patched):
            call = (make_sequence(step, inputs, warm_length) if warm_length > 1
                    else lambda s=step: s(*inputs))
            call.steps = warm_length
            warm.append(call)
        row = compare_sequences(session, *runs, *warm, pairs=args.pairs,
                                warmup_steps=args.warmup_steps, label=label)
        row["steps"] = steps
        row["workload_kind"] = kind
        row.update(generation_throughput(row))
        row["prefix_copy_included"] = bool(metadata.get("context"))
        rows[label] = row
        comparison = PairedComparison(**row["timing"])
        comparisons[label] = comparison
        verdict = ("win confirmed" if comparison.wins_by(0.0) else
                   "regression confirmed" if comparison.loses_by(0.0) else "unresolved change")
        amount = f"{steps} generated tokens" if kind == "library_generation" else f"{steps} consecutive steps"
        print(f"{label}: {amount}, original {row['baseline_sequence_ms']:.1f} ms, "
              f"patched {row['candidate_sequence_ms']:.1f} ms, ratio {row['timing']['median_ratio']:.3f} "
              f"(patched over original), {verdict}")
        if rates := throughput_text(row):
            print(f"  {rates}")
    confirmed = workload_win(comparisons)
    if args.json:
        lengths = {row["steps"] for row in rows.values()}
        args.json.write_text(json.dumps({"steps": next(iter(lengths)) if len(lengths) == 1 else None,
                                        "steps_by_workload": {label: row["steps"] for label, row in rows.items()},
                                        "pairs": args.pairs, "workloads": rows,
                                        "win_confirmed": confirmed}, indent=1) + "\n")
    if confirmed:
        print("benchmark: confirmed speedup on at least one workload; no detected regressions")
    elif any(c.loses_by(0.0) for c in comparisons.values()):
        print("benchmark: no confirmed speedup; at least one workload has a confirmed regression")
    else:
        print("benchmark: no confirmed speedup (a difference inside the measurement's uncertainty is not a win)")
    return 0 if confirmed else 1


if __name__ == "__main__":
    sys.exit(main())
