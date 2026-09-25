"""Artifact emit (spec "Leaving a region, and the artifact"): kernels, the
generated patch module, the swap table, the vendored runtime, apply(), report,
and with a ModelBundle the model source, workload recipes and the load,
validate and benchmark scripts.

The harness installs the same generated code it measured; emitting is writing
that code down, never regenerating it differently. Wrapper classes whose
bodies are identical are written once, under one name, and every module path
they served points at that class. The artifact works in a fresh process with
no autotuner import: apply.py puts the vendored runtime on sys.path under its
real package name.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import uuid
from pathlib import Path
from typing import Sequence

import mlx.core as mx

import autotuner_runtime

from ..bind.emit import MODULE_HEADER, EmittedWrapper
from ..report import Report
from .bundle import ModelBundle, write_bundle
from autotuner_runtime.kernels import KernelSpec

_CLASS_LINE = re.compile(r"^class (\w+)\(", re.M)

_APPLY_SHIM = '''\
"""Load this artifact onto a freshly loaded build() model:

    from artifact.apply import apply
    model = apply(build())
"""

import sys
from pathlib import Path

_here = Path(__file__).parent
sys.path.insert(0, str(_here / "runtime"))

from autotuner_runtime.apply import apply as _apply


def apply(model):
    return _apply(model, _here)
'''


def write_kernel(directory: str | Path, spec: KernelSpec) -> None:
    """Keep editable bodies in .metal files, including each ordered stage."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{spec.kernel_id}.metal").write_text(spec.source)
    launch = json.loads(spec.to_json())
    del launch["source"]
    if spec.stages:
        stage_dir = directory / f"{spec.kernel_id}.stages"
        stage_dir.mkdir(exist_ok=True)
        for i, stage in enumerate(launch["stages"]):
            (stage_dir / f"{i}.metal").write_text(stage["kernel"].pop("source"))
    (directory / f"{spec.kernel_id}.launch.json").write_text(json.dumps(launch, indent=1) + "\n")


def dedupe_wrappers(wrappers: Sequence[EmittedWrapper]) -> tuple[str, list[dict]]:
    """The exported patch module and swap table: one class per distinct body.

    Two wrappers share a class when their sources are identical once the class
    name is replaced, so the body shipped is byte for byte the body measured.
    The shared class is named from the kernel ids it carries, and a comment
    above it lists the module paths it serves.
    """
    groups: dict[str, list[EmittedWrapper]] = {}
    for w in wrappers:
        groups.setdefault(_CLASS_LINE.sub("class __CLASS__(", w.source, count=1), []).append(w)
    classes, table, names = [], [], set()
    for key, members in groups.items():
        if "class __CLASS__(" not in key:
            name, source = members[0].class_name, key
        else:
            ids = list(dict.fromkeys(k for w in members for k in w.kernel_ids))
            base = "W_" + "__".join(re.sub(r"\W", "_", k) for k in ids) if ids else members[0].class_name
            name, n = base, 2
            while name in names:
                name, n = f"{base}_{n}", n + 1
            paths = ", ".join(w.scope_path or "(root)" for w in members)
            source = key.replace("class __CLASS__(", f"# serves {paths}\nclass {name}(", 1)
        names.add(name)
        classes.append(source)
        # the span map says which recorded op each replayed line came from;
        # it is per module path, so it rides in the table, not the class
        table += [{"scope_path": w.scope_path, "wrapper_class": name, "kernel_ids": w.kernel_ids,
                   "span_map": [list(entry) for entry in w.span_map]}
                  for w in members]
    return MODULE_HEADER + "".join(classes), table


def emit_artifact(
    out_dir: str | Path,
    kernels: list[KernelSpec],
    wrappers: list[EmittedWrapper],
    report: Report,
    validate=None,
    bundle: ModelBundle | None = None,
) -> Path:
    final = Path(out_dir)
    if final.exists() and not final.is_dir():
        raise FileExistsError(f"artifact destination is not a directory: {final}")
    # build beside the target and swap at the end, so a mid-emit failure can
    # never destroy an existing good artifact
    final.parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=final.name + ".building-", dir=final.parent))
    try:
        (out / "kernels").mkdir(parents=True)
        (out / "patch").mkdir()
        (out / "buffers").mkdir()  # reserved for future precompute; empty in v1

        for spec in kernels:
            write_kernel(out / "kernels", spec)

        module, table = dedupe_wrappers(wrappers)
        (out / "patch" / "wrappers.py").write_text(module)
        (out / "swap_table.json").write_text(json.dumps(table, indent=1) + "\n")

        baseline_wrappers = bundle.baseline_wrappers if bundle is not None else []
        if baseline_wrappers:
            baseline_module, baseline_table = dedupe_wrappers(baseline_wrappers)
            (out / "patch" / "baseline_wrappers.py").write_text(baseline_module)
            (out / "baseline_swap_table.json").write_text(json.dumps(baseline_table, indent=1) + "\n")

        runtime_src = Path(autotuner_runtime.__file__).parent
        runtime_dst = out / "runtime" / "autotuner_runtime"
        runtime_dst.mkdir(parents=True)
        for py in runtime_src.glob("*.py"):
            shutil.copy(py, runtime_dst / py.name)
        if any("GraphWrapper" in w.source for w in [*wrappers, *baseline_wrappers]):
            from autotuner_runtime.graph_native import prepare
            native_dst = runtime_dst / "graph_native"
            shutil.copytree(runtime_src / "graph_native", native_dst,
                            ignore=shutil.ignore_patterns("__pycache__", "*.so", "build.json"))
            prepare(native_dst)  # ship the tested binary; inference needs no compiler

        (out / "apply.py").write_text(_APPLY_SHIM)
        (out / "__init__.py").write_text("from .apply import apply\n")
        report.write(out / "report.json")
        if report.manifest_path and Path(report.manifest_path).is_file():
            # the request as written: sweep, budget and baseline, not just the shapes timed
            shutil.copy(report.manifest_path, out / "manifest.yaml")

        if bundle is not None:
            write_bundle(out, bundle, table, report.to_dict(), package=final.name)
            (out / "__init__.py").write_text("from .apply import apply\nfrom .load import load\n")
        if validate is not None:
            validate(out)

        previous = final.with_name(final.name + ".previous-" + uuid.uuid4().hex)
        had_previous = final.exists()
        if had_previous:
            final.rename(previous)
        try:
            out.rename(final)
        except BaseException:
            if had_previous:
                previous.rename(final)
            raise
        if had_previous:
            shutil.rmtree(previous)
        return final
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise


_CHECK = '''
import importlib.util, json, sys
from pathlib import Path
import mlx.core as mx

mode = sys.argv[4]
apply_spec = importlib.util.spec_from_file_location("artifact_apply", sys.argv[2])
apply_mod = importlib.util.module_from_spec(apply_spec)
apply_spec.loader.exec_module(apply_mod)
spec_file = json.loads(Path(sys.argv[3]).read_text())
cases, context = spec_file["cases"], spec_file["context"]

def as_step(model):
    """The job's workload ran as a step over a filled KV cache: rebuild that step."""
    if metadata.get("use_library_inference"):
        from autotuner_runtime.inference import LibraryInference
        tokens = mx.load(context["tokens"])["tokens"] if context else None
        return LibraryInference(model, steps=metadata["final_benchmark"]["steps"], prefix_tokens=tokens)
    if context is None:
        return model
    from autotuner_runtime.state import context_step  # apply.py put the vendored runtime on sys.path
    warm = mx.load(context["warm"])
    return context_step(model, context["context"], mx.load(context["tokens"])["tokens"],
                        [warm[f"i{k}"] for k in range(len(warm))])

def check(model, how):
    for case in cases:
        inputs = mx.load(case["inputs"])
        outs = model(*[inputs[f"i{k}"] for k in range(len(inputs))])
        flat = []
        def walk(o):
            if isinstance(o, mx.array): flat.append(o)
            elif isinstance(o, (list, tuple)): [walk(v) for v in o]
            elif isinstance(o, dict): [walk(v) for v in o.values()]
        walk(outs)
        expected = mx.load(case["expected"])
        label = f"{case['name']} via {how}"
        assert len(flat) == len(expected), f"{label}: {len(flat)} outputs, expected {len(expected)}"
        for k, got in enumerate(flat):
            want = expected[f"o{k}"]
            assert got.shape == want.shape, f"{label}: output {k} shape changed"
            assert got.dtype == want.dtype, f"{label}: output {k} dtype changed"
            from autotuner_runtime.exact import bitwise_equal
            assert bitwise_equal(got, want), f"{label}: output {k} differs from the patched model"

load_path = Path(sys.argv[2]).with_name("load.py")
metadata_path = load_path.with_name("bundle.json")
metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
source_weights = metadata.get("weight_policy") == "model_source"
if mode == "apply":
    from autotuner_runtime.checkpoints import bundle_checkpoint_pins, use_checkpoint_pins
    metadata_path = load_path.with_name("bundle.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    with use_checkpoint_pins(bundle_checkpoint_pins(metadata, load_path.parent)):
        model_path = Path(sys.argv[1]).resolve()
        sys.path.insert(0, str(model_path.parent))
        spec = importlib.util.spec_from_file_location("autotune_model", model_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        if source_weights:
            from autotuner_runtime.swap import require_independent_models
            original = mod.build()
            model = mod.build()
            require_independent_models(original, model)
            if hasattr(model, "update"):
                model.update(original.parameters())
            original = as_step(original)
            model = as_step(apply_mod.apply(model))
        else:
            model = as_step(apply_mod.apply(mod.build()))
    if metadata_path.exists() and json.loads(metadata_path.read_text())["baseline"] == "compiled":
        plain = model
        model = mx.compile(lambda *args: plain(*args))
    if source_weights:
        validate_spec = importlib.util.spec_from_file_location("artifact_validate", load_path.with_name("validate.py"))
        validate_mod = importlib.util.module_from_spec(validate_spec)
        validate_spec.loader.exec_module(validate_mod)
        from autotuner_runtime.state import correctness_call
        # Use plain execution here so the cache observation includes state.
        candidate = plain if metadata["baseline"] == "compiled" else model
        for case in cases:
            data = mx.load(case["inputs"])
            inputs = [data[f"i{k}"] for k in range(len(data))]
            result = validate_mod.check_outputs(
                lambda: correctness_call(original, inputs),
                lambda: correctness_call(candidate, inputs), case["name"],
                **validate_mod.policy_arguments(metadata))
            assert result["passed"], f"apply() differs from fresh original: {result}"
    else:
        check(model, "apply()")
else:
    # A separate isolated interpreter has never imported the original model
    # or its helper modules. Only the bundle supplies the definition.
    load_spec = importlib.util.spec_from_file_location("artifact_load", load_path)
    load_mod = importlib.util.module_from_spec(load_spec)
    load_spec.loader.exec_module(load_mod)
    if not source_weights:
        check(load_mod.load(), "load()")
    validate_spec = importlib.util.spec_from_file_location("artifact_validate", load_path.with_name("validate.py"))
    validate_mod = importlib.util.module_from_spec(validate_spec)
    validate_spec.loader.exec_module(validate_mod)
    results = validate_mod.validate(include_sequences=True)
    failed = [result for result in results if not result["passed"]]
    assert not failed, f"bundle violates its recorded correctness policy: {failed}"
print("__AUTOTUNE_ARTIFACT_OK__")
'''


def check_apply(artifact_dir: str | Path, model_path: str | Path, inputs: list[mx.array],
                expected: list[mx.array], timeout_s: float = 900.0) -> None:
    """Compatibility wrapper for one workload; see check_apply_many."""
    check_apply_many(artifact_dir, model_path, [("workload", inputs, expected)], timeout_s)


def check_apply_many(
    artifact_dir: str | Path,
    model_path: str | Path,
    workloads: Sequence[tuple[str, list[mx.array], list[mx.array]]],
    timeout_s: float = 900.0,
    context: tuple[int, mx.array, list[mx.array]] | None = None,
) -> None:
    """Rebuild in a fresh process and check every workload and sweep, through
    apply() on the repo's model file and, when the artifact carries a bundle,
    through load() on the bundle's own copy of the model source.

    Source-based bundles compare freshly built original and patched models
    sharing identical weights under the recorded correctness rule. Legacy
    bundles also reproduce the saved job outputs on their fixed weights.
    context is (context, tokens, warm inputs) when the workload ran as a step
    over a filled KV cache; the child rebuilds that step around build().
    """
    if not workloads:
        raise ValueError("artifact validation requires at least one workload")
    metadata_path = Path(artifact_dir) / "bundle.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    source_weights = metadata.get("weight_policy") == "model_source"
    with tempfile.TemporaryDirectory() as tmp:
        cases = []
        for index, (name, inputs, expected) in enumerate(workloads):
            in_path = f"{tmp}/{index}_inputs.safetensors"
            out_path = f"{tmp}/{index}_expected.safetensors"
            mx.save_safetensors(in_path, {f"i{k}": t for k, t in enumerate(inputs)})
            if not source_weights:
                mx.save_safetensors(out_path, {f"o{k}": t for k, t in enumerate(expected)})
            cases.append({"name": name, "inputs": in_path, "expected": out_path})
        spec = {"cases": cases, "context": None}
        if context is not None:
            mx.save_safetensors(f"{tmp}/context.safetensors", {"tokens": context[1]})
            mx.save_safetensors(f"{tmp}/warm.safetensors", {f"i{k}": t for k, t in enumerate(context[2])})
            spec["context"] = {"context": context[0], "tokens": f"{tmp}/context.safetensors",
                               "warm": f"{tmp}/warm.safetensors"}
        case_path = Path(tmp) / "cases.json"
        case_path.write_text(json.dumps(spec))
        modes = ["apply", "bundle"] if (Path(artifact_dir) / "load.py").exists() else ["apply"]
        for mode in modes:
            effective_timeout = timeout_s
            if mode == "bundle":
                metadata = json.loads((Path(artifact_dir) / "bundle.json").read_text())
                report = json.loads((Path(artifact_dir) / "report.json").read_text())
                steps = metadata.get("final_benchmark", {}).get("steps", 10)
                clocks = report.get("step_ms", {})
                # Four observed arms, 3x cooling, plus 25% slack and startup.
                work_s = sum(max(float(clocks.get(name, {}).get("before", 0)),
                                 float(clocks.get(name, {}).get("after", 0))) / 1000
                             for name, _, _ in workloads)
                effective_timeout = max(timeout_s, 120 + 20 * (steps + 1) * work_s)
            import os
            environment = dict(os.environ)
            if mode == "bundle":
                environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                if metadata.get("weight_policy") != "model_source":
                    environment.update(HF_HOME=f"{tmp}/empty-hf", HF_HUB_CACHE=f"{tmp}/empty-hf/hub",
                                       TRANSFORMERS_CACHE=f"{tmp}/empty-hf/transformers")
            proc = subprocess.run(
                [sys.executable, "-I", "-c", textwrap.dedent(_CHECK), str(Path(model_path).resolve()),
                 str(Path(artifact_dir).resolve() / "apply.py"), str(case_path), mode],
                capture_output=True, text=True, timeout=effective_timeout, cwd=tmp, env=environment,
            )
            if proc.returncode != 0 or proc.stdout.strip().splitlines()[-1:] != ["__AUTOTUNE_ARTIFACT_OK__"]:
                raise RuntimeError(f"the artifact did not reproduce the patched model via {mode} "
                                   f"in a fresh process: {proc.stderr.strip()[-2000:]}")
