"""Kernel loading and the one call site.

A KernelSpec is everything needed to rebuild and launch an mx.fast.metal_kernel
from serialized form: source body, header, IO names, launch expressions in the
grammar, template dtypes, output shape expressions. The harness owns this call
site: init_value, math_mode, and streams are set here, never by the judge.

call() is also what the record-mode tracer patches, so a custom dispatch shows
up as one node in retraces.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

from .grammar import Expr

MATH_MODE = "safe"  # pinned per job, harness-side; recorded in the report

_DTYPES = {
    "bool": mx.bool_, "uint8": mx.uint8, "uint16": mx.uint16, "uint32": mx.uint32,
    "uint64": mx.uint64, "int8": mx.int8, "int16": mx.int16, "int32": mx.int32,
    "int64": mx.int64, "float16": mx.float16, "bfloat16": mx.bfloat16,
    "float32": mx.float32, "complex64": mx.complex64,
}


@dataclass(frozen=True)
class KernelSpec:
    kernel_id: str
    name: str                                  # a valid C identifier
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    source: str                                # kernel BODY; mlx generates the signature
    header: str = ""
    grid: tuple[str, str, str] = ("1", "1", "1")         # launch grammar, TOTAL threads
    threadgroup: tuple[str, str, str] = ("1", "1", "1")
    output_shapes: tuple[tuple[str, ...], ...] = ()      # one expr tuple per output
    output_dtypes: tuple[str, ...] = ()
    template: tuple[tuple[str, str], ...] = ()           # (name, dtype name or "inK")
    fallback_predicate: str | None = None                # true -> use the library path
    ensure_row_contiguous: bool = True
    atomic_outputs: bool = False

    def to_json(self) -> str:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        return json.dumps(d, indent=1)

    @staticmethod
    def from_json(text: str) -> "KernelSpec":
        d = json.loads(text)
        for k in ("input_names", "output_names", "grid", "threadgroup", "output_dtypes"):
            d[k] = tuple(d[k])
        d["output_shapes"] = tuple(tuple(s) for s in d["output_shapes"])
        d["template"] = tuple(tuple(t) for t in d["template"])
        return KernelSpec(**d)


def load_spec(metal_path: str | Path) -> KernelSpec:
    """Rebuild a spec from <id>.metal + <id>.launch.json next to it."""
    metal_path = Path(metal_path)
    launch = json.loads(metal_path.with_suffix("").with_suffix(".launch.json").read_text())
    launch["source"] = metal_path.read_text()
    return KernelSpec.from_json(json.dumps(launch))


class LoadedKernel:
    """A compiled-on-first-use kernel plus its parsed launch expressions.

    The launch (output shapes, grid, threadgroup, template, fallback) is a
    function of the inputs' shapes and dtypes, so it is evaluated once per
    distinct call signature and reused: evaluating the grammar on every call
    cost a small kernel more than its GPU time (spike 13)."""

    def __init__(self, spec: KernelSpec):
        self.spec = spec
        self._kernel = mx.fast.metal_kernel(
            name=spec.name,
            input_names=list(spec.input_names),
            output_names=list(spec.output_names),
            source=spec.source,
            header=spec.header,
            ensure_row_contiguous=spec.ensure_row_contiguous,
            atomic_outputs=spec.atomic_outputs,
            compile_options={"math_mode": MATH_MODE},
        )
        self._grid = tuple(Expr(e) for e in spec.grid)
        self._tg = tuple(Expr(e) for e in spec.threadgroup)
        self._out_shapes = tuple(tuple(Expr(e) for e in s) for s in spec.output_shapes)
        self._out_dtypes = [_DTYPES[d] for d in spec.output_dtypes]
        self._fallback = Expr(spec.fallback_predicate) if spec.fallback_predicate else None
        self._launches: dict[tuple, tuple] = {}

    def _launch(self, inputs: list[mx.array]) -> tuple:
        key = tuple((tuple(a.shape), a.dtype) for a in inputs)
        launch = self._launches.get(key)
        if launch is None:
            shapes = [shape for shape, _ in key]
            template = []
            for name, dt in self.spec.template:
                # "inN" borrows input N's dtype; anything else is a dtype name
                # (the exact-match test matters: "int32" is a dtype, not input t32)
                if dt.startswith("in") and dt[2:].isdigit():
                    template.append((name, inputs[int(dt[2:])].dtype))
                else:
                    template.append((name, _DTYPES[dt]))
            launch = (
                [tuple(e.evaluate(shapes) for e in s) for s in self._out_shapes],
                tuple(e.evaluate(shapes) for e in self._grid),
                tuple(e.evaluate(shapes) for e in self._tg),
                template,
                bool(self._fallback.evaluate(shapes)) if self._fallback is not None else False,
            )
            self._launches[key] = launch
        return launch

    def fallback_fires(self, inputs: list[mx.array]) -> bool:
        return self._launch(inputs)[4]

    def __call__(self, inputs: list[mx.array], init_value: float | None = None) -> list[mx.array]:
        out_shapes, grid, threadgroup, template, _ = self._launch(inputs)
        return self._kernel(
            inputs=inputs,
            output_shapes=out_shapes,
            output_dtypes=self._out_dtypes,
            grid=grid,
            threadgroup=threadgroup,
            template=template,
            init_value=init_value,
        )


def imported(path: str):
    """The model's own compiled function, found by its import path; a
    generated wrapper calls it where the recording saw the compiled section."""
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


_cache: dict[str, LoadedKernel] = {}


def _loaded(spec: KernelSpec) -> LoadedKernel:
    loaded = _cache.get(spec.kernel_id)
    if loaded is None or loaded.spec is not spec:
        loaded = LoadedKernel(spec)
        _cache[spec.kernel_id] = loaded
    return loaded


def fallback_fires(spec: KernelSpec, inputs: list[mx.array]) -> bool:
    """True when this call's shapes are not covered and the wrapper must take
    the original op sequence instead."""
    return _loaded(spec).fallback_fires(inputs)


def call(spec: KernelSpec, inputs: list[mx.array], init_value: float | None = None) -> list[mx.array]:
    """The one kernel call site. Record mode patches this function, so the
    custom dispatch appears as a node in retraces."""
    return _loaded(spec)(inputs, init_value=init_value)
