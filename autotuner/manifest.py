"""Manifest: the job contract. Parse, validate, default; frozen once loaded.

The spec ("Job in: the manifest") fixes the vocabulary; plan 5.12 fixes the
defaults. Every defaulted field is listed in Manifest.defaulted so the report
can record what was actually used.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import mlx.core as mx
import yaml

DEFAULT_SWEEP_SIZES = (1, 13, 50, 4096)
DEFAULT_BUDGET_PER_REGION = 25
DEFAULT_BUDGET_TOTAL = 250
DEFAULT_INT_RANGE = (0, 100)
DEFAULT_JOB_SEED = 0x4D455441  # fixed by the harness, recorded, never supplied
BOUNDARY_INPUT_SETS = 3

# Assoc-preserving tolerances per dtype (rtol, atol); manifest may override with one pair.
DEFAULT_TOLERANCES: Mapping[str, tuple[float, float]] = MappingProxyType({
    "float32": (1e-5, 1e-6),
    "float16": (1e-2, 2e-2),
    "bfloat16": (2e-2, 4e-2),
})

_DTYPE_NAMES: Mapping[str, mx.Dtype] = MappingProxyType({
    name: getattr(mx, attr)
    for name, attr in {
        "bool": "bool_", "uint8": "uint8", "uint16": "uint16", "uint32": "uint32",
        "uint64": "uint64", "int8": "int8", "int16": "int16", "int32": "int32",
        "int64": "int64", "float16": "float16", "bfloat16": "bfloat16",
        "float32": "float32", "complex64": "complex64",
    }.items()
})

FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32"})
INT_DTYPES = frozenset({"uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"})


class ManifestError(ValueError):
    """A manifest problem the user can fix; the message says how."""


def resolve_dtype(name: str) -> mx.Dtype:
    if name not in _DTYPE_NAMES:
        raise ManifestError(
            f"unknown dtype {name!r}; use one of {sorted(_DTYPE_NAMES)}"
        )
    return _DTYPE_NAMES[name]


@dataclass(frozen=True)
class InputSpec:
    """One positional argument of the model: a shape plus a dtype.

    A str dim is a named, sweepable dim; int dims are model constants and never
    move. Integer dtypes sample from [low, high) so token ids stay in range.
    """

    shape: tuple[int | str, ...]
    dtype: str
    low: int = DEFAULT_INT_RANGE[0]
    high: int = DEFAULT_INT_RANGE[1]

    def named_dims(self) -> frozenset[str]:
        return frozenset(d for d in self.shape if isinstance(d, str))


@dataclass(frozen=True)
class Workload:
    """The real input shapes to optimize for. Name is a report label only."""

    inputs: tuple[InputSpec, ...]
    name: str

    def named_dims(self) -> frozenset[str]:
        return frozenset().union(*(i.named_dims() for i in self.inputs)) if self.inputs else frozenset()


@dataclass(frozen=True)
class Manifest:
    model_path: Path
    workloads: tuple[Workload, ...]
    sweep: Mapping[str, tuple[int, ...]]
    primary: Mapping[str, int]
    tolerances: tuple[float, float] | None
    budget_per_region: int
    budget_total: int
    seed: int
    defaulted: tuple[str, ...]

    def named_dims(self) -> frozenset[str]:
        return frozenset().union(*(w.named_dims() for w in self.workloads))

    def tolerance_for(self, dtype: str) -> tuple[float, float]:
        if self.tolerances is not None:
            return self.tolerances
        if dtype in DEFAULT_TOLERANCES:
            return DEFAULT_TOLERANCES[dtype]
        raise ManifestError(f"no tolerance default for dtype {dtype!r}; set tolerances in the manifest")


def _parse_shape(raw: Any, where: str) -> tuple[int | str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ManifestError(f"{where}: shape must be a non-empty list, got {raw!r}")
    dims: list[int | str] = []
    for d in raw:
        if isinstance(d, bool) or not isinstance(d, (int, str)):
            raise ManifestError(f"{where}: shape dims are ints or named strings, got {d!r}")
        if isinstance(d, int) and d < 1:
            raise ManifestError(f"{where}: integer dims must be >= 1, got {d}")
        if isinstance(d, str) and not d.isidentifier():
            raise ManifestError(f"{where}: named dim {d!r} must be an identifier")
        dims.append(d)
    return tuple(dims)


def _parse_input(raw: Any, where: str) -> InputSpec:
    if not isinstance(raw, dict) or "shape" not in raw or "dtype" not in raw:
        raise ManifestError(f"{where}: each input needs {{shape, dtype}}, got {raw!r}")
    unknown = set(raw) - {"shape", "dtype", "low", "high"}
    if unknown:
        raise ManifestError(f"{where}: unknown input keys {sorted(unknown)}")
    dtype = raw["dtype"]
    if not isinstance(dtype, str):
        raise ManifestError(f"{where}: dtype must be a string, got {dtype!r}")
    resolve_dtype(dtype)
    low = raw.get("low", DEFAULT_INT_RANGE[0])
    high = raw.get("high", DEFAULT_INT_RANGE[1])
    if ("low" in raw or "high" in raw) and dtype not in INT_DTYPES:
        raise ManifestError(f"{where}: low/high apply to integer dtypes only")
    if not (isinstance(low, int) and isinstance(high, int) and low < high):
        raise ManifestError(f"{where}: need integer low < high, got low={low!r} high={high!r}")
    return InputSpec(shape=_parse_shape(raw["shape"], where), dtype=dtype, low=low, high=high)


def _parse_workload(raw: Any, index: int) -> Workload:
    where = f"workloads[{index}]"
    if not isinstance(raw, dict) or "inputs" not in raw:
        raise ManifestError(f"{where}: a workload needs an inputs list, got {raw!r}")
    unknown = set(raw) - {"inputs", "name"}
    if unknown:
        raise ManifestError(f"{where}: unknown workload keys {sorted(unknown)}")
    raw_inputs = raw["inputs"]
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ManifestError(f"{where}: inputs must be a non-empty list")
    name = raw.get("name", f"workload{index}")
    if not isinstance(name, str) or not name:
        raise ManifestError(f"{where}: name must be a non-empty string")
    inputs = tuple(_parse_input(r, f"{where}.inputs[{i}]") for i, r in enumerate(raw_inputs))
    return Workload(inputs=inputs, name=name)


def _parse_positive_int(raw: Any, where: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ManifestError(f"{where}: must be a positive integer, got {raw!r}")
    return raw


def load(path: str | Path) -> Manifest:
    """Parse and validate a manifest file. Model path resolves relative to it."""
    path = Path(path).resolve()
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise ManifestError(f"manifest file not found: {path}")
    except yaml.YAMLError as e:
        raise ManifestError(f"manifest is not valid YAML: {e}")
    if not isinstance(raw, dict):
        raise ManifestError(f"manifest must be a mapping, got {type(raw).__name__}")

    known = {"model", "workloads", "sweep", "primary", "tolerances", "budget"}
    unknown = set(raw) - known
    if unknown:
        raise ManifestError(
            f"unknown manifest keys {sorted(unknown)}; dtypes and quantization are not "
            f"knobs and the chip is auto-detected (allowed: {sorted(known)})"
        )
    if "model" not in raw:
        raise ManifestError("manifest needs a model: path to a .py file defining build()")
    model_path = (path.parent / str(raw["model"])).resolve()
    if not model_path.is_file():
        raise ManifestError(f"model file does not exist: {model_path}")
    if model_path.suffix != ".py":
        raise ManifestError(f"model must be a .py file defining build(), got {model_path.name}")

    raw_workloads = raw.get("workloads")
    if not isinstance(raw_workloads, list) or not raw_workloads:
        raise ManifestError("manifest needs a non-empty workloads list")
    workloads = tuple(_parse_workload(w, i) for i, w in enumerate(raw_workloads))
    names = [w.name for w in workloads]
    if len(set(names)) != len(names):
        raise ManifestError(f"workload names must be unique, got {names}")

    named = frozenset().union(*(w.named_dims() for w in workloads))
    defaulted: list[str] = []

    raw_sweep = raw.get("sweep", {})
    if not isinstance(raw_sweep, dict):
        raise ManifestError("sweep must be a mapping of named dim to size list")
    sweep: dict[str, tuple[int, ...]] = {}
    for dim, sizes in raw_sweep.items():
        if dim not in named:
            raise ManifestError(
                f"sweep names dim {dim!r} which appears in no workload shape; "
                f"integer dims are model constants and are never swept (named dims: {sorted(named) or 'none'})"
            )
        if not isinstance(sizes, list) or not sizes:
            raise ManifestError(f"sweep.{dim}: must be a non-empty list of sizes")
        sweep[dim] = tuple(_parse_positive_int(s, f"sweep.{dim}") for s in sizes)
    for dim in sorted(named - set(sweep)):
        sweep[dim] = DEFAULT_SWEEP_SIZES
        defaulted.append(f"sweep.{dim}")

    # A named dim needs one concrete size for tracing and pricing; the sweep sizes
    # are correctness-only. Default: the largest sweep size. Overridable via primary.
    raw_primary = raw.get("primary", {})
    if not isinstance(raw_primary, dict):
        raise ManifestError("primary must be a mapping of named dim to one size")
    primary: dict[str, int] = {}
    for dim, size in raw_primary.items():
        if dim not in named:
            raise ManifestError(f"primary names dim {dim!r} which appears in no workload shape")
        primary[dim] = _parse_positive_int(size, f"primary.{dim}")
    for dim in sorted(named - set(primary)):
        primary[dim] = max(sweep[dim])
        defaulted.append(f"primary.{dim}")

    tolerances: tuple[float, float] | None = None
    if "tolerances" in raw:
        t = raw["tolerances"]
        if not isinstance(t, dict) or set(t) != {"rtol", "atol"}:
            raise ManifestError("tolerances must be {rtol, atol}")
        try:
            tolerances = (float(t["rtol"]), float(t["atol"]))
        except (TypeError, ValueError):
            raise ManifestError(f"tolerances must be numbers, got {t!r}")
        if tolerances[0] < 0 or tolerances[1] < 0:
            raise ManifestError("tolerances must be non-negative")
    else:
        defaulted.append("tolerances")

    raw_budget = raw.get("budget", {})
    if not isinstance(raw_budget, dict) or set(raw_budget) - {"per_region", "total"}:
        raise ManifestError("budget must be {per_region?, total?}")
    if "per_region" in raw_budget:
        per_region = _parse_positive_int(raw_budget["per_region"], "budget.per_region")
    else:
        per_region = DEFAULT_BUDGET_PER_REGION
        defaulted.append("budget.per_region")
    if "total" in raw_budget:
        total = _parse_positive_int(raw_budget["total"], "budget.total")
    else:
        total = DEFAULT_BUDGET_TOTAL
        defaulted.append("budget.total")

    defaulted.append("seed")
    return Manifest(
        model_path=model_path,
        workloads=workloads,
        sweep=MappingProxyType(sweep),
        primary=MappingProxyType(primary),
        tolerances=tolerances,
        budget_per_region=per_region,
        budget_total=total,
        seed=DEFAULT_JOB_SEED,
        defaulted=tuple(defaulted),
    )


_BUILD_PROBE = textwrap.dedent("""
    import importlib.util, inspect, sys
    path = sys.argv[1]
    spec = importlib.util.spec_from_file_location("autotune_model", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"import failed: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(3)
    build = getattr(mod, "build", None)
    if build is None:
        print("no build() found", file=sys.stderr)
        sys.exit(4)
    if not callable(build):
        print(f"build is not callable (it is {type(build).__name__})", file=sys.stderr)
        sys.exit(4)
    try:
        params = inspect.signature(build).parameters.values()
    except (TypeError, ValueError):
        params = ()
    required = [p.name for p in params
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    if required:
        print(f"build() must take no arguments, but requires {required}", file=sys.stderr)
        sys.exit(5)
    print("ok")
""")


def check_build(manifest: Manifest, timeout_s: float = 60.0) -> None:
    """Probe the build() contract in a fresh subprocess with a foreign CWD.

    Catches: a model file that fails to import cleanly (including import-time
    CWD-relative reads, since the probe runs elsewhere), a missing or
    non-callable build, and a build() with required arguments. Raises
    ManifestError with the fix spelled out.
    """
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.run(
            [sys.executable, "-c", _BUILD_PROBE, str(manifest.model_path)],
            capture_output=True, text=True, cwd=cwd, timeout=timeout_s,
        )
    if proc.returncode == 0:
        return
    detail = proc.stderr.strip() or f"exit code {proc.returncode}"
    fix = {
        3: "the model file must import cleanly in a fresh process with a different "
           "working directory; resolve paths relative to __file__, never the CWD",
        4: f"define build() at module level in {manifest.model_path.name}; it must "
           "return the model as a self-contained repeatable callable, not a live object",
        5: "build() takes no arguments; bake configuration into the model file",
    }.get(proc.returncode, "the model file must be importable in a fresh subprocess")
    raise ManifestError(f"model probe failed ({detail}). Fix: {fix}")
