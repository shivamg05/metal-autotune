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
    baseline_wrappers: list = field(default_factory=list)


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


def write_bundle(out: Path, bundle: ModelBundle, patches: Sequence[Mapping], report_dict: dict,
                 package: str = "artifact") -> dict:
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
        "baseline_scopes": [w.scope_path for w in bundle.baseline_wrappers],
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
    (out / "README.md").write_text(render_readme(metadata, report_dict, package))
    return metadata


def _ms(value: float) -> str:
    return f"{value:,.1f} ms" if value >= 100 else f"{value:.3f} ms"


def _result_lines(metadata: dict, report: dict) -> list[str]:
    """One bullet per workload: the headline speedup first, then the numbers behind it."""
    from autotuner_runtime.sequence import throughput_text

    if not metadata["patches"]:
        return ["No replacement was installed: no attempt produced a correct kernel with a "
                "confirmed whole-model speedup, so the loaded model runs the original code."]
    lines = []
    sequences = report.get("final", {}).get("sequences", {})
    task = "one generation request" if metadata.get("use_library_inference") else "one step"
    for name, clocks in report.get("step_ms", {}).items():
        if "after" not in clocks:
            continue
        untouched = clocks.get("baseline_at_end", clocks.get("before"))
        single = (f"{task}: {_ms(untouched)} untouched, {_ms(clocks['after'])} patched"
                  + (f" ({clocks['speedup']:.3f}x)" if clocks.get("speedup") else "")
                  + (", confirmed above the timing noise" if clocks.get("win_confirmed")
                     else ", not resolved above the timing noise, so treat it as no change"))
        row = sequences.get(name)
        if row and row.get("steps") and row.get("candidate_sequence_ms"):
            unit = "generated tokens" if row.get("workload_kind") == "library_generation" else "consecutive steps"
            confirmed = "confirmed" if row.get("win_confirmed") else "not resolved above the timing noise"
            speedup = row.get("speedup") or row["baseline_sequence_ms"] / row["candidate_sequence_ms"]
            lines.append(f"- `{name}`: **{speedup:.3f}x faster** over {row['steps']} {unit} "
                         f"({_ms(row['baseline_sequence_ms'])} untouched, "
                         f"{_ms(row['candidate_sequence_ms'])} patched), {confirmed}. "
                         f"For {single}.")
            if rates := throughput_text(row):
                lines.append(f"  Generated tokens/sec, {rates.split(': ', 1)[1]}.")
        elif clocks.get("speedup"):
            lines.append(f"- `{name}`: **{clocks['speedup']:.3f}x faster** for {single}.")
        else:
            lines.append(f"- `{name}`: {single}.")
    return lines


def _measured_on(metadata: dict, report: dict) -> str:
    baseline = ("the same model run under `mx.compile`" if metadata["baseline"] == "compiled"
                else "the plain model exactly as build() returns it")
    chip = report.get("machine", {}).get("chip")
    where = f" on an {chip}" if chip and chip[0].upper() in "AEIOU" else (f" on a {chip}" if chip else "")
    return (f"Measured{where} against {baseline}, with both versions timed together at the end "
            "of the job. Other machines, weights and input sizes can move these numbers.")


def _compress_paths(paths: Sequence[str]) -> list[str]:
    """`blocks.0.attn` ... `blocks.19.attn` reads as `blocks.{0..19}.attn`."""
    groups: dict[tuple, list[int]] = {}
    order, loose = [], []
    for path in paths:
        parts = path.split(".")
        numbered = [i for i, p in enumerate(parts) if p.isdigit()]
        if len(numbered) != 1:
            loose.append(path)
            continue
        i = numbered[0]
        key = (tuple(parts[:i]), tuple(parts[i + 1:]))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(int(parts[i]))
    out = []
    for key in order:
        head, tail = key
        numbers = sorted(groups[key])
        if len(numbers) >= 3 and numbers == list(range(numbers[0], numbers[-1] + 1)):
            out.append(".".join([*head, f"{{{numbers[0]}..{numbers[-1]}}}", *tail]))
        else:
            out.extend(".".join([*head, str(n), *tail]) for n in numbers)
    return out + loose


def _replacement_rows(patches: Sequence[Mapping]) -> str:
    by_kernels: dict[tuple, list[str]] = {}
    for p in patches:
        by_kernels.setdefault(tuple(p["kernel_ids"]), []).append(p["module_path"] or "(the model itself)")
    rows = []
    for kernels, paths in by_kernels.items():
        shown = ", ".join(f"`{p}`" for p in _compress_paths(paths))
        rows.append(f"| {shown} | {len(paths)} | {', '.join(f'`{k}`' for k in kernels)} |")
    return "\n".join(rows) or "| (none) | 0 | |"


def _package(name: str) -> str:
    return name if name.isidentifier() else "artifact"


def render_readme(metadata: dict, report: dict, package: str = "artifact") -> str:
    entry = metadata["entry"]
    pkg = _package(package)
    compiled_note = ("Because the job measured against the compiled model, `load()` runs the "
                     "forward pass under `mx.compile`; pass `compile=False` for the plain model."
                     if metadata["baseline"] == "compiled" else
                     "The job measured against the plain model, so `load()` returns the plain "
                     "model; pass `compile=True` to run it under `mx.compile` instead.")
    workloads = "\n".join(f"  - `{name}`: " + ", ".join(str(tuple(s)) for s in w["shapes"])
                          for name, w in metadata["workloads"].items()) or "  - none"
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
        usage = USAGE_MLX_LM.format(pkg=pkg)
    else:
        usage = USAGE_CUSTOM.format(pkg=pkg)
    rename = ("" if pkg == package else
              f" The folder name `{package}` is not a valid Python name, so the examples assume "
              "you renamed it to `artifact`.")
    return README_TEMPLATE.format(
        entry=entry, mlx=metadata["mlx"], python=metadata["python"], pkg=pkg, rename=rename,
        result="\n".join(_result_lines(metadata, report)), measured_on=_measured_on(metadata, report),
        usage=usage, rows=_replacement_rows(metadata["patches"]),
        compiled_note=compiled_note, workloads=workloads, context_note=context_note,
        steps=metadata["final_benchmark"].get("steps", "?"),
        pairs=metadata["final_benchmark"].get("pairs", "?"),
    )


USAGE_MLX_LM = """Load the model the way you normally do, patch it, and use the returned model:

```python
from mlx_lm import load, generate
from {pkg} import apply

model, tokenizer = load("/path/to/checkpoint-or-hub-id")
model = apply(model)
print(generate(model, tokenizer, prompt="Hello"))
```

If your builder wraps the MLX-LM model, recreate that wrapper before calling
`apply()`. Generation handles its cache normally."""

USAGE_CUSTOM = """Build and load your model the way you normally do, patch it, and use the
returned model:

```python
from {pkg} import apply

model = build()                              # your original builder
model.load_weights("my_weights.safetensors")  # if you load weights
model = apply(model)                         # keep the returned model
outputs = model(*inputs)
```"""


README_TEMPLATE = """# Optimized `{entry}`

This folder makes the model in `model/{entry}` faster on Apple Silicon. It
comes from one metal-autotune job: parts of the model now run custom Metal GPU
kernels, and the job checked that the outputs still match the original. The
folder is self-contained, so you don't need the optimizer to use it.

## Result

{result}

{measured_on}

## Quick start

Run these from inside this folder. You need an Apple Silicon Mac and Python {python}.

```sh
pip install -r requirements.txt   # pins mlx {mlx}
python validate.py                # patched outputs match the original on the saved inputs
python benchmark.py               # re-time original vs patched, the way the job did
```

## Use it in your code

Put this folder next to your code and import it by its folder name.{rename} If
you rename the folder, change the import to match.

{usage}

To build the model from the bundle's own copy of the source instead:

```python
from {pkg} import load
model = load().inference_model
```

**What it works with.** The kernels were made for this model's exact
operations, layer layout, shapes, dtypes and quantization. Different weight
values with the same structure (for example, a fine-tune) are fine. A
different architecture or quantization needs a new optimization run. Inputs
whose shapes the job didn't measure run the original code: correct, just not
faster. Check speed and correctness on your own weights and inputs before
relying on the result.

**Weights are not included.** `build()` loads or initializes weights exactly
as the original does: a Hub model uses your cache or downloads, a local
checkpoint must be reachable, and random initialization stays random.
`bundle.json` records the checkpoints the job saw (`weight_sources`).

## Reference

### What was replaced

Each row is a set of modules (by attribute path from the model root) and the
kernels that now run inside them. A kernel runs only on the input shapes and
dtypes the job recorded; anything else falls back to the original code.

| modules | count | kernels |
|---|---|---|
{rows}

### Loading options

`load()` runs the bundled `build()` and applies the patch. Calling the returned
object (`loaded(*inputs)`) reproduces the job's benchmark interface;
`loaded.inference_model` is the patched model for normal use.
`load(patched=False)` gives the untouched model; for a side-by-side comparison
with identical weights use `original = load(patched=False, measurement_baseline=True)`
and `patched = load(share_weights_with=original.model)`.
`measurement_baseline=True` restores the job's original compiled scopes for
timing; without it, `patched=False` leaves the model untouched for correctness
checks. {compiled_note}{context_note}

### How it was checked

`validate.py` runs every saved workload through the patched and the original
model and applies the job's correctness rule. Edits that keep floating-point
evaluation unchanged must match bit for bit. Edits that change it must satisfy
`abs(patched - original) <= atol + rtol * abs(original)` with the tolerances in
`bundle.json`. Shapes, dtypes and non-floating values must match exactly, and
non-finite values must sit in the same places. The whole patched model is
compared at once, so several replacements share one allowance.
`python validate.py --sequences` also checks consecutive steps and cache state.
`validate.py` exits nonzero on a mismatch.

`benchmark.py` runs that check first, then times whole runs of {steps}
consecutive steps for both models, alternating their order across {pairs} pairs
with cooling in between. It exits nonzero unless the patched model is faster by
more than the measurement's own noise. `--steps` and `--pairs` override the
job's settings.

### Files

- `manifest.yaml`: the manifest the job ran, as written (when the job had one);
  its paths point to where the job ran.
- `model/`: the model's source, copied unchanged.
- `workloads/`: the inputs the job measured on, one file per workload:
{workloads}
- `kernels/`: one `.metal` body and `.launch.json` per kernel. A staged kernel
  keeps its shader bodies in `<id>.stages/0.metal`, `1.metal`, and so on.
- `patch/wrappers.py` and `swap_table.json`: which module each kernel installs
  into and how.
- `runtime/`: the small package that loads the kernels and installs them. It
  checks MLX, Python and platform compatibility before loading.
- `load.py`, `apply.py`, `validate.py`, `benchmark.py`: the entry points above.
- `bundle.json`: entry file, baseline, versions, workloads, tolerances and patches.
- `report.json`: the job's full record: every region, attempt and measurement.
- `requirements.txt`: the pinned packages. `buffers/` is reserved and empty.

### Changing it

A kernel is `kernels/<id>.metal` (the body; MLX generates the signature from
the input and output names) plus `kernels/<id>.launch.json` (the launch
arithmetic). Edit either, then run `validate.py` and `benchmark.py` again, and
don't use a kernel that no longer matches the original's outputs.
`patch/wrappers.py` is ordinary Python. To change the model itself, edit
`model/` and run the optimizer again: the patch is tied to the exact operations
it recorded.
"""
