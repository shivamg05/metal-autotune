"""Gate 1: static checks on a KernelSpec against the region's IO contract
(plan section 7). No GPU work runs here.

Tolerance secrecy is structural: the contract carries names, ranks, dtypes,
and liveness only, so this module cannot see or leak tolerance values.

Input ranks have no declaration on the kernel side; they are enforced through
the launch grammar instead, by probe-evaluating every expression against
dummy shapes of the contract's ranks so a bad input index or axis fails here,
not at dispatch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from autotuner_runtime.grammar import Expr, GrammarError
from autotuner_runtime.kernels import _DTYPES, KernelSpec

# spike_04: the kernel name is pasted into the generated signature; a bad name
# fails only at probe eval, so it is gated here.
_C_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TEMPLATE_INPUT_REF = re.compile(r"^in\d+$")


@dataclass(frozen=True)
class RegionContract:
    """The region's boundary as gate 1 sees it: input/output names, ranks,
    dtypes, and which outputs are live. No tensors, no tolerances."""

    input_names: tuple[str, ...]
    input_ranks: tuple[int, ...]
    input_dtypes: tuple[str, ...]
    output_names: tuple[str, ...]
    output_ranks: tuple[int, ...]
    output_dtypes: tuple[str, ...]
    live_outputs: tuple[str, ...]
    requires_fallback: bool = False   # the hypothesis is shape-specialized


@dataclass(frozen=True)
class Failure:
    check: str
    detail: str


def check(spec: KernelSpec, contract: RegionContract) -> list[Failure]:
    """Every static failure in the spec, named. Empty list means gate 1 passes."""
    fails: list[Failure] = []

    if not _C_IDENTIFIER.match(spec.name):
        fails.append(Failure(
            "kernel_name",
            f"{spec.name!r} is not a valid C identifier; it is pasted into the "
            "generated signature and would fail only at probe eval",
        ))

    if tuple(spec.input_names) != tuple(contract.input_names):
        fails.append(Failure(
            "input_names",
            f"kernel inputs {tuple(spec.input_names)} != region inputs "
            f"{tuple(contract.input_names)}",
        ))
    # region outputs come first; extra scratch buffers are legal only when
    # every one is named tmp<N> (naive scaffolds stage through device memory)
    n_region = len(contract.output_names)
    head = tuple(spec.output_names[:n_region])
    extras = tuple(spec.output_names[n_region:])
    if head != tuple(contract.output_names) or not all(
        e.startswith("tmp") and e[3:].isdigit() for e in extras
    ):
        fails.append(Failure(
            "output_names",
            f"kernel outputs {tuple(spec.output_names)} must start with region "
            f"outputs {tuple(contract.output_names)}, extras all tmp<N>",
        ))
    for name in contract.live_outputs:
        if name not in spec.output_names:
            fails.append(Failure(
                "live_output_dropped",
                f"live value {name!r} is neither produced nor listed as an output",
            ))

    n_out = len(spec.output_names)
    if len(spec.output_shapes) != n_out or len(spec.output_dtypes) != n_out:
        fails.append(Failure(
            "output_arity",
            f"{n_out} output names but {len(spec.output_shapes)} shape tuples "
            f"and {len(spec.output_dtypes)} dtypes",
        ))
    for name, dt, want in zip(spec.output_names, spec.output_dtypes, contract.output_dtypes):
        if dt != want:
            fails.append(Failure(
                "output_dtype",
                f"output {name!r}: {dt} != frozen boundary dtype {want}",
            ))
    # zips stop at the region outputs; tmp extras carry no boundary contract
    for name, exprs, want in zip(spec.output_names, spec.output_shapes, contract.output_ranks):
        if len(exprs) != want:
            fails.append(Failure(
                "output_rank",
                f"output {name!r}: {len(exprs)} shape expressions, region rank is {want}",
            ))

    dummy_shapes = [(4,) * r for r in contract.input_ranks]
    for axis_name, exprs in (("grid", spec.grid), ("threadgroup", spec.threadgroup)):
        if len(exprs) != 3:
            fails.append(Failure(
                "launch_arity", f"{axis_name} has {len(exprs)} expressions, needs 3"))
        for i, text in enumerate(exprs):
            _probe(f"{axis_name}[{i}]", text, dummy_shapes, fails)
    for j, exprs in enumerate(spec.output_shapes):
        for i, text in enumerate(exprs):
            _probe(f"output_shapes[{j}][{i}]", text, dummy_shapes, fails)
    if spec.fallback_predicate is not None:
        _probe("fallback_predicate", spec.fallback_predicate, dummy_shapes, fails)
    elif contract.requires_fallback:
        fails.append(Failure(
            "fallback_missing",
            "shape-specialized kernel declares no fallback predicate",
        ))

    for tname, dt in spec.template:
        if _TEMPLATE_INPUT_REF.match(dt):
            if int(dt[2:]) >= len(spec.input_names):
                fails.append(Failure(
                    "template",
                    f"template {tname!r} references {dt} but the kernel has "
                    f"{len(spec.input_names)} inputs",
                ))
        elif dt not in _DTYPES:
            fails.append(Failure(
                "template", f"template {tname!r}: unknown dtype {dt!r}"))

    return fails


def _probe(slot: str, text: str, dummy_shapes: list[tuple[int, ...]],
           fails: list[Failure]) -> None:
    """Parse, then evaluate against dummy shapes of the contract's ranks so a
    bad input index or axis fails statically."""
    try:
        expr = Expr(text)
    except GrammarError as e:
        fails.append(Failure("launch_grammar", f"{slot}: {e}"))
        return
    try:
        expr.evaluate(dummy_shapes)
    except GrammarError as e:
        fails.append(Failure("launch_grammar", f"{slot}: {e}"))
    except ArithmeticError:
        pass  # value-dependent (dummy sizes hit a zero divisor); not static
