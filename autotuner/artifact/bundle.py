"""The self-contained part of the artifact: the model's source, the traced
workload inputs, and the scripts that load, verify, and re-time the result.

Model source is copied unchanged. Weights are obtained by build() as usual,
never copied implicitly or frozen from random initialization. Local imports
are copied recursively; explicit ARTIFACT_FILES resources remain opt-in.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import functools
import importlib.metadata
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import mlx.core as mx

FORMAT_VERSION = 1
_SCRIPTS = {"load.py": "loader.py", "validate.py": "validate.py", "benchmark.py": "benchmark.py"}


@dataclass
class ModelBundle:
    model_path: Path
    baseline: str = "compiled"
    workloads: dict[str, list[mx.array]] = field(default_factory=dict)  # label -> traced inputs
    workload_config: Any = ()      # the manifest's declared workloads
    final_benchmark: dict = field(default_factory=dict)   # steps, pairs, warmup_steps
    project_root: Path | None = None
    # a workload run as a step over a filled KV cache: workload, context, seed, tokens
    context: dict | None = None
    goldens: dict[str, list[mx.array]] = field(default_factory=dict)
    sequence_goldens: dict[str, list[mx.array]] = field(default_factory=dict)
    exact: bool = True
    tolerances: tuple[float, float] | None = None
    recovery: bool = False
    checkpoint_pins: list[dict] | None = None
    use_library_inference: bool = False


def _project_root(path: Path) -> Path:
    for directory in path.parents:
        if (directory / "pyproject.toml").is_file() or (directory / ".git").exists():
            return directory
    directory = path.parent
    while (directory / "__init__.py").exists():
        directory = directory.parent
    return directory


def source_files(model_path: Path, project_root: Path) -> tuple[set[Path], set[str], list[Path]]:
    """Find local Python imports and explicitly declared non-Python resources."""
    roots = [model_path.parent, project_root]
    pending = [model_path]
    for directory in model_path.parents:
        if directory == project_root:
            break
        init = directory / "__init__.py"
        if init.is_file():
            if not init.resolve().is_relative_to(project_root):
                raise ValueError(f"artifact source escapes project root: {init}")
            pending.append(init)
    files: set[Path] = set()
    packages: set[str] = {"mlx"}
    resources: list[Path] = []

    def add(path):
        path = Path(os.path.abspath(path))
        if not path.resolve().is_relative_to(project_root):
            raise ValueError(f"artifact source escapes project root: {path}")
        if path not in files:
            pending.append(path)

    def find(name, search_roots):
        for root in search_roots:
            path = root.joinpath(*name.split("."))
            for candidate in (path.with_suffix(".py"), path / "__init__.py"):
                if candidate.is_file():
                    add(candidate)
                    for parent in candidate.parents:
                        if parent == project_root:
                            break
                        init = parent / "__init__.py"
                        if init.is_file():
                            add(init)
                    return True
        return False

    while pending:
        path = pending.pop()
        if path in files:
            continue
        files.add(path)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not find(alias.name, roots):
                        packages.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = path.parent
                    for _ in range(node.level - 1):
                        base = base.parent
                    if node.module:
                        find(node.module, [base])
                    for alias in node.names:
                        if alias.name != "*":
                            find(".".join(filter(None, (node.module, alias.name))), [base])
                elif node.module:
                    if not find(node.module, roots):
                        packages.add(node.module.split(".")[0])
                    for alias in node.names:
                        if alias.name != "*":
                            find(f"{node.module}.{alias.name}", roots)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == "ARTIFACT_FILES" for t in targets):
                    try:
                        declared = ast.literal_eval(node.value)
                    except (ValueError, TypeError) as exc:
                        raise ValueError("ARTIFACT_FILES must be a literal list of relative file paths") from exc
                    if not isinstance(declared, (list, tuple)) or not all(isinstance(x, str) for x in declared):
                        raise ValueError("ARTIFACT_FILES must be a literal list of relative file paths")
                    for name in declared:
                        resource = Path(os.path.abspath(path.parent / name))
                        if (Path(name).is_absolute() or not resource.resolve().is_relative_to(project_root)
                                or not resource.is_file()):
                            raise ValueError(f"artifact resource must be an existing file inside the project: {name}")
                        resources.append(resource)
    if "autotuner" in packages or any(p.relative_to(project_root).parts[0] == "autotuner" for p in files):
        raise ValueError("a portable model must not import the autotuner harness; move shared model code out of autotuner")
    return files, packages, resources


def requirements(packages: set[str]) -> str:
    """mlx and the model's other top-level imports, pinned to the installed
    distributions. No transitive walk: pip resolves the rest."""
    providers = importlib.metadata.packages_distributions()
    pinned: dict[str, str] = {}
    unknown = []
    for package in sorted(packages):
        if package in sys.stdlib_module_names or package == "__future__":
            continue
        names = providers.get(package, [])
        if not names:
            unknown.append(package)
        for name in names:
            distribution = importlib.metadata.distribution(name)
            pinned[distribution.metadata["Name"]] = distribution.version
    if unknown:
        raise ValueError("cannot package model dependencies with no installed distribution: "
                         + ", ".join(unknown)
                         + ". Install them as packages or include their source in the model project.")
    lines = ["# The versions this result was measured with, on Apple Silicon."]
    lines += [f"{name}=={version}" for name, version in sorted(pinned.items(), key=lambda kv: kv[0].lower())]
    return "\n".join(lines) + "\n"


def check_model_dependencies(model_path: Path) -> None:
    """Fail before search when the model's source/dependencies cannot travel."""
    model_path = Path(model_path).resolve()
    _, packages, _ = source_files(model_path, _project_root(model_path))
    requirements(packages)


def _declared(config: Any) -> dict:
    """The manifest's workloads as JSON: named dims stay strings."""
    if isinstance(config, Mapping):
        return dict(config)
    return {w.name: [{"shape": list(i.shape), "dtype": i.dtype} for i in w.inputs] for w in config}


def _file_label(label: str, taken: set[str]) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", label) or "workload"
    name, n = stem, 2
    while name in taken:
        name, n = f"{stem}_{n}", n + 1
    taken.add(name)
    return name


@contextmanager
def capture_checkpoint_loads():
    """Record the snapshots actually used by mlx_lm.load while building.

    Capture at _download, which load() and load_tokenizer() both call, so even
    existing aliases of load() participate. Restore the helper on all exits.
    """
    captured = {}
    try:
        import mlx_lm.utils as utils
    except ImportError:
        # Custom model builders do not acquire an mlx-lm dependency merely
        # because the optimizer can record mlx-lm checkpoint provenance.
        yield captured
        return
    original = utils._download
    @functools.wraps(original)
    def download(path_or_hf_repo, revision=None, allow_patterns=None):
        result = original(path_or_hf_repo, revision=revision, allow_patterns=allow_patterns)
        key = (str(path_or_hf_repo), revision)
        resolved = str(Path(result).resolve())
        previous = captured.setdefault(key, resolved)
        if previous != resolved:
            raise ValueError(f"checkpoint revision changed between model builds: {key[0]}")
        return result
    utils._download = download
    try:
        yield captured
    finally:
        utils._download = original


from autotuner_runtime.checkpoints import use_checkpoint_pins


def resolve_model_checkpoints(model_path: Path, *, captured=None) -> list[dict]:
    """Pin actual loads within a job, without interpreting the builder source.

    These pins keep later comparison arms on the same snapshot. Exports only
    record provenance; their builders retain the user's loading behavior.
    """
    pins = []
    for (source, revision), loaded in (captured or {}).items():
        local = Path(loaded).resolve()
        if not local.is_dir():
            raise ValueError(f"loaded checkpoint is no longer available: {local}")
        pins.append({"source": source, "requested_revision": revision,
                     "revision": local.name if not Path(source).expanduser().is_dir() else revision,
                     "path": str(local)})
    return pins


def _tolerance_defaults():
    from autotuner_runtime.numeric import tolerance_for
    return {str(dtype).removeprefix("mlx.core."): dict(zip(("rtol", "atol"), tolerance_for(dtype)))
            for dtype in (mx.float16, mx.bfloat16, mx.float32)}


def write_bundle(out: Path, bundle: ModelBundle, patches: Sequence[Mapping], report_dict: dict) -> dict:
    """Write model/, workloads/, the scripts, bundle.json, requirements.txt
    and README.md into an artifact directory that already holds the kernels,
    wrappers, swap table and runtime. Returns the bundle.json content."""
    if bundle.baseline not in {"compiled", "plain"}:
        raise ValueError(f"unsupported artifact baseline: {bundle.baseline}")
    declared = _declared(bundle.workload_config)
    missing = set(declared) - set(bundle.workloads)
    if missing:
        raise ValueError(f"artifact missing declared workload inputs: {sorted(missing)}")
    if bundle.context is not None:
        label = bundle.context["workload"]
        if label not in bundle.workloads:
            raise ValueError(f"decode context has no saved workload: {label}")
        if not bundle.recovery and bundle.goldens and label not in bundle.sequence_goldens:
            raise ValueError("changing decode artifact needs its consecutive-step fp32 reference")
    unknown_sequences = set(bundle.sequence_goldens) - set(bundle.workloads)
    if unknown_sequences:
        raise ValueError(f"sequence golden has no saved workload: {sorted(unknown_sequences)}")
    model_path = Path(bundle.model_path).resolve()
    root = Path(bundle.project_root).resolve() if bundle.project_root else _project_root(model_path)
    if not model_path.is_relative_to(root):
        raise ValueError("artifact model must be inside its project root")
    files, packages, resources = source_files(model_path, root)
    if bundle.use_library_inference:
        packages.add("mlx_lm")
    for path in files | set(resources):
        destination = out / "model" / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    declared = _declared(bundle.workload_config)
    (out / "workloads").mkdir(exist_ok=True)
    workloads = {}
    taken: set[str] = set()
    for label, tensors in bundle.workloads.items():
        file = f"workloads/{_file_label(label, taken)}.safetensors"
        mx.save_safetensors(str(out / file), {f"i{k}": t for k, t in enumerate(tensors)})
        workloads[label] = {
            "file": file,
            "role": "declared" if label in declared else "sweep",
            "shapes": [list(t.shape) for t in tensors],
            "dtypes": [str(t.dtype).removeprefix("mlx.core.") for t in tensors],
        }

    goldens = {}
    for label, tensors in bundle.goldens.items():
        if label not in workloads:
            raise ValueError(f"golden has no saved workload: {label}")
        file = f"workloads/{_file_label(label + '.golden', taken)}.safetensors"
        mx.save_safetensors(str(out / file), {f"o{k}": t for k, t in enumerate(tensors)})
        goldens[label] = file
    if not bundle.recovery and goldens and set(goldens) != set(workloads):
        raise ValueError("changing artifacts need fp32 references for every saved workload")

    context = None
    if bundle.context is not None:
        # the tokens the cache is filled with, so load() rebuilds the same step
        file = f"workloads/{_file_label(bundle.context['workload'] + '.context', taken)}.safetensors"
        mx.save_safetensors(str(out / file), {"tokens": bundle.context["tokens"]})
        context = {k: bundle.context[k] for k in ("workload", "context", "seed")} | {"file": file}

    sequence_goldens = {}
    for label, tensors in bundle.sequence_goldens.items():
        file = f"workloads/{_file_label(label + '.sequence_golden', taken)}.safetensors"
        mx.save_safetensors(str(out / file), {f"o{k}": t for k, t in enumerate(tensors)})
        sequence_goldens[label] = file
    final_benchmark = dict(bundle.final_benchmark)
    measured_steps = {name: row["steps"]
                      for name, row in report_dict.get("final", {}).get("sequences", {}).items()
                      if name in declared and row.get("steps")}
    if measured_steps:
        final_benchmark["steps_by_workload"] = measured_steps
        final_benchmark["warmup_steps_per_sample"] = {
            name: report_dict.get("step_ms", {}).get(name, {}).get("steps_per_sample", 1)
            for name in measured_steps
        }
    metadata = {
        "format_version": FORMAT_VERSION,
        "entry": model_path.relative_to(root).as_posix(),
        "baseline": bundle.baseline,
        "use_library_inference": bundle.use_library_inference,
        "measurement": {
            "kind": "library_generation" if bundle.use_library_inference else "forward",
            "generated_tokens": bundle.final_benchmark.get("steps", 10) if bundle.use_library_inference else None,
        },
        "mlx": mx.__version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "declared_workloads": declared,
        "workloads": workloads,
        "context": context,
        "correctness_rule": "fp32_relative" if goldens else "exact" if bundle.exact else "baseline_tolerance",
        "tolerances": (None if bundle.tolerances is None else
                       {"rtol": bundle.tolerances[0], "atol": bundle.tolerances[1]}),
        "tolerance_defaults": _tolerance_defaults(),
        "goldens": goldens,
        "sequence_goldens": sequence_goldens,
        "weight_policy": "model_source",
        "weight_sources": [{key: pin.get(key) for key in ("source", "requested_revision", "revision")}
                           for pin in (bundle.checkpoint_pins or [])],
        "checkpoint_resources_included": False,
        "final_benchmark": final_benchmark,
        "patches": [{"module_path": p["scope_path"], "kernel_ids": list(p["kernel_ids"])} for p in patches],
        "source_files": sorted(p.relative_to(root).as_posix() for p in files),
        "resource_files": sorted(p.relative_to(root).as_posix() for p in resources),
    }
    (out / "bundle.json").write_text(json.dumps(metadata, indent=1) + "\n")
    (out / "requirements.txt").write_text(requirements(packages))
    for name, script in _SCRIPTS.items():
        shutil.copy(Path(__file__).with_name(script), out / name)
    (out / "README.md").write_text(render_readme(metadata, report_dict))
    return metadata


def _ms(value: float) -> str:
    return f"{value:,.1f} ms" if value >= 100 else f"{value:.3f} ms"


def _result_lines(metadata: dict, report: dict) -> list[str]:
    baseline = ("the model run under mx.compile" if metadata["baseline"] == "compiled"
                else "the plain model exactly as build() returns it")
    lines = [f"Every timing below compares the patched model against {baseline}, "
             "both measured together at the end of the job."]
    if not metadata["patches"]:
        lines.append("\nNo replacement was installed: no attempt established a correct, "
                     "confirmed whole-model speedup. The loaded model runs the original code.")
        return lines
    sequences = report.get("final", {}).get("sequences", {})
    for name, clocks in report.get("step_ms", {}).items():
        if "after" not in clocks:
            continue
        untouched = clocks.get("baseline_at_end", clocks.get("before"))
        verdict = ("confirmed above the timing noise" if clocks.get("win_confirmed")
                   else "not resolved above the timing noise, so treat it as no change")
        speedup = f", a {clocks['speedup']:.3f}x speedup" if clocks.get("speedup") else ""
        lines.append(f"\n- `{name}`: one step took {_ms(untouched)} untouched and "
                     f"{_ms(clocks['after'])} patched{speedup}, {verdict}.")
        row = sequences.get(name)
        if row and row.get("steps"):
            verdict = ("confirmed" if row.get("win_confirmed") else "not resolved above the timing noise")
            lines.append(f"  {row['steps']} consecutive steps, run whole and alternated: "
                         f"{_ms(row['baseline_sequence_ms'])} untouched, "
                         f"{_ms(row['candidate_sequence_ms'])} patched, {verdict}.")
    return lines


def render_readme(metadata: dict, report: dict) -> str:
    entry = metadata["entry"]
    rows = "\n".join(f"| `{p['module_path'] or '(the model itself)'}` | {', '.join(f'`{k}`' for k in p['kernel_ids'])} |"
                     for p in metadata["patches"]) or "| (none) | |"
    compiled_note = ("Because the job measured against the compiled model, the callable `load()` "
                     "returns already runs the forward pass under `mx.compile`; pass "
                     "`compile=False` for the plain model."
                     if metadata["baseline"] == "compiled" else
                     "The job measured against the plain model, so `load()` returns the plain "
                     "model; pass `compile=True` to run it under `mx.compile` instead.")
    workloads = ", ".join(f"`{name}` " + " ".join(str(tuple(s)) for s in w["shapes"])
                          for name, w in metadata["workloads"].items()) or "none"
    context = metadata.get("context")
    if context:
        compiled_note = "The repeatable cached step owns mutable state and runs without outer mx.compile."
    context_note = "" if not context else (
        f"\n\nThe `{context['workload']}` workload is one call over a KV cache already holding "
        f"{context['context']} tokens (saved in `{context['file']}`). `load()` builds the cache "
        "the model's own way, fills it with those tokens, and restores its full state after every call, so "
        "each call is the same step the job measured. This is a benchmark wrapper, not "
        "a growing conversation. For normal decoding use `loaded.inference_model`, where "
        "`loaded = load(compile=False)`, and pass your own `cache=` as the original "
        "model expects. That model keeps the installed kernels and advances its cache "
        "normally. Final timing advances the cache across the saved number of steps, "
        "using fixed input tokens in both arms; other lengths may "
        "use the library fallback.")
    if metadata.get("use_library_inference"):
        compiled_note = (
            "The job measured normal MLX-LM generation. Calling `loaded(*inputs)` "
            f"processes one prompt and completes {metadata['final_benchmark']['steps']} generated tokens, "
            "including sampling and cache updates. It excludes loading and tokenization. "
            "The library owns execution; the generation controller is never wrapped in `mx.compile`.")
        context_note = (
            " Each trial starts from independent cache state. "
            + (f"The saved {context['context']}-token context is prepared before timing. " if context else "")
            + "For normal use, pass `loaded.inference_model` to your library's generation function.")
    return README_TEMPLATE.format(
        entry=entry, mlx=metadata["mlx"], python=metadata["python"],
        result="\n".join(_result_lines(metadata, report)), rows=rows,
        compiled_note=compiled_note, workloads=workloads, context_note=context_note,
        steps=metadata["final_benchmark"].get("steps", "?"),
        pairs=metadata["final_benchmark"].get("pairs", "?"),
    )


README_TEMPLATE = """# Optimized `{entry}`

This folder is the result of one autotuning job on the model defined in
`model/{entry}`. It holds that model's source, the Metal kernels (GPU programs)
that replace runs of its operations, the generated Python that calls them, the
inputs the job measured on, and scripts to load, verify, and re-time the result.
It needs an Apple Silicon Mac, Python {python}, and the packages pinned in
`requirements.txt` (mlx {mlx}). It does not need the optimizer. It still needs the model's original external
resources: a Hub model may use the local cache or download, a local checkpoint
must be accessible, and random initialization remains random. `weight_sources`
in `bundle.json` records checkpoints observed during the job, without forcing
future loads to use them. Pin a revision in your model source when needed.

## Use it

Copy this folder into your project and install its `requirements.txt` in your
Python environment. Run the examples from the directory containing `artifact/`
(adjust the import if you renamed the folder).

For normal inference: load your weights, apply the optimized code, then call
the model as usual. Load weights before applying the patch, and use the model
returned by `apply()`.

For an MLX-LM model, using a compatible local checkpoint or Hub model ID:

```python
from mlx_lm import load, generate
from artifact import apply

model, tokenizer = load("/path/to/my-checkpoint")
model = apply(model)
text = generate(model, tokenizer, prompt="Explain gravity simply.")
print(text)
```

Use the same model structure returned by the original `build()`. If your
builder wraps the MLX-LM model, recreate that wrapper before calling `apply()`.
Generation handles its cache normally; `apply()` does not add the benchmark's
cache-reset behavior or compile the whole model.

For a custom MLX model, use its original builder and weight-loading API:

```python
from artifact import apply

model = build()  # your original model builder
model.load_weights("my_weights.safetensors")
model = apply(model)
outputs = model(*inputs)
```

Different weight values can use the same patch when the model's operations,
module layout, tensor shapes, dtypes, and quantization remain compatible.
Changing the architecture or quantization requires another optimization run.
Unsupported input shapes use the original implementation. The reported speedup
applies to the measured workload; check correctness and speed with your own
weights and inference inputs before relying on it.

## Load from the bundled source

`load()` uses the bundled model's original `build()`, including its weight
loading or random initialization. It has no separate checkpoint-path argument.
Use `apply()` above when you want to choose weights yourself.

```python
from artifact import load
loaded = load()
model = loaded.inference_model  # normal inference with your own cache, if needed
outputs = model(*inputs)
```

Calling `loaded(*inputs)` instead reproduces the job's benchmark interface,
including cache resets for cached workloads and its recorded compilation mode.
`loaded.inference_model` exposes the patched model without that outer benchmark
interface; use your model library's normal inference or generation API.

`load(patched=False)` gives the untouched model built the same way. For a
side-by-side comparison with identical weights, use:

```python
original = load(patched=False)
patched = load(share_weights_with=original.model)
```

The validation and benchmark scripts do this automatically, including for
randomly initialized models. {compiled_note}{context_note}

## Measured result

{result}

## What was replaced

Each row is one part of the model (named by its attribute path from the model
root) and the kernel that now runs inside it. A kernel only runs on the input
shapes and dtypes the job recorded; any other call falls back to the original code.

| module path | kernel |
|---|---|
{rows}

## Layout

- `README.md`: this file.
- `bundle.json`: the entry file, the baseline the job measured against, the mlx and Python versions, the saved workloads, and which kernels patch which module paths.
- `load.py`: `load()` builds the model from `model/`, applies the patch, and returns it ready to call.
- `apply.py`: `apply(model)` patches a model you built yourself from the same source.
- `validate.py`: checks that the patched model's outputs match the original's on the saved inputs.
- `benchmark.py`: re-times the original and patched models on the saved inputs, the way the job's final check did.
- `model/`: the model's source files, laid out as in the project they came from.
  Source is copied unchanged. `build()` loads or initializes weights exactly
  as the original builder does. External weights are not copied or redirected;
  `ARTIFACT_FILES` includes only resources the model author explicitly lists.
- `workloads/`: the input tensors the job traced and measured on, one safetensors file per workload ({workloads}).
- `kernels/`: a `.metal` body and `.launch.json` per candidate. For an ordered sequence, the actual shader bodies live in `<id>.stages/0.metal`, `1.metal`, etc.; the launch file declares their inputs, outputs, shapes and dtypes.
- `patch/wrappers.py`: the installation rules. Supported scopes construct their original computation, substitute selected graph operations, and cache the compiled result. Scopes with unsupported Python state retain certified replay wrappers.
- `swap_table.json`: which wrapper class installs at which module path, with its kernel ids.
- `runtime/`: the package that loads kernels and installs wrappers, including the matching native graph extension when used. Its MLX/Python/platform compatibility is checked before loading.
- `buffers/`: reserved for precomputed data; empty.
- `report.json`: the job's full account: every region, every attempt, every measurement.
- `requirements.txt`: the pinned packages.

## Verify it

```sh
python validate.py --sequences  # outputs and cache state, including consecutive steps
python benchmark.py    # repeat the saved experiment, original vs patched, {pairs} alternated pairs
```

`validate.py` uses the job's correctness rule on every saved workload. Edits
that preserve floating-point evaluation must match the original bit for bit.
Edits that change floating-point evaluation use the tolerances recorded in
`bundle.json`: each floating value must satisfy
`abs(patched - original) <= atol + rtol * abs(original)`. Nonfloating values,
shapes, and dtypes stay exact. Nonfinite positions must match. The complete
patched model is always compared with the untouched model, so errors from
multiple replacements share one allowance. Validation exits nonzero on a
mismatch. Older bundles with saved fp32 references retain their original rule.
`benchmark.py` runs the correctness check first, then times whole runs of
consecutive steps for both models, alternating their order and cooling between
runs, and exits nonzero unless the patched model is faster by more than the
measurement's own uncertainty. By default it uses each workload's actual final
measurement length, including extra repetitions used for very short forward calls.
Library inference uses the saved generated-token count. `--steps` overrides the
length for every workload; `--pairs` overrides the number of comparison pairs.

## Change it

- A kernel is `kernels/<id>.metal` (the body; mlx generates the signature from
  the input and output names) plus `kernels/<id>.launch.json` (the launch
  arithmetic). For a staged candidate, edit its bodies in `kernels/<id>.stages/`
  and the stage configurations in the launch file. Edit either, then run
  `validate.py` and `benchmark.py` again.
  A kernel that no longer matches the original's outputs must not be used.
- `patch/wrappers.py` is ordinary Python: each class replays one module's
  recorded operations, calling a kernel where the job spliced one in and
  falling back to the original module on shapes it did not record.
- To change the model itself, edit the source in `model/` and run the
  optimizer again: the patch is tied to the exact operations it recorded.
"""
