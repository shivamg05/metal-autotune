"""E2e promotion checks: the orig-vs-orig floor, patched vs
original on that floor, and the step-time veto.

Both arms live in one process with weights shared by array identity, so memory does not double.
The baseline arm is a fresh untouched build(); the patched arm is a fresh
build() with the generated wrappers swapped in. No hooks exist in either arm.

The floor detail the spec leaves loose: a deterministic library gives an
exactly-zero orig-vs-orig difference, so the allowance is floored by a small
multiple of the output dtype's epsilon at the observed value scale, as the
assoc-changing floor is. Both knobs are recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import mlx.core as mx

from .measure.clocks import PairedComparison, compare
from .measure.session import Session
from .trace.walk import flatten_arrays

FLOOR_MULTIPLE = 4.0
EPS_MULTIPLE = 32.0
STEP_VETO_PCT = 0.005

_EPS = {"float16": 9.77e-4, "bfloat16": 7.81e-3, "float32": 1.19e-7}


@dataclass
class WorkloadCheck:
    name: str
    floor_max_abs: float
    max_abs: float
    allowance: float
    cosine: float
    passed: bool


@dataclass
class E2EResult:
    checks: list[WorkloadCheck] = field(default_factory=list)
    veto: PairedComparison | None = None
    veto_passed: bool = True

    @property
    def passed(self) -> bool:
        return self.veto_passed and all(c.passed for c in self.checks)


def _flatten(tree: object) -> list[mx.array]:
    out = flatten_arrays(tree)
    mx.eval(out)
    return out


def _max_abs(a: list[mx.array], b: list[mx.array]) -> float:
    worst = 0.0
    for x, y in zip(a, b):
        worst = max(worst, mx.abs(x.astype(mx.float32) - y.astype(mx.float32)).max().item())
    return worst


def _cosine(a: list[mx.array], b: list[mx.array]) -> float:
    worst = 1.0
    for x, y in zip(a, b):
        xf = x.astype(mx.float32).reshape(-1)
        yf = y.astype(mx.float32).reshape(-1)
        denom = (mx.linalg.norm(xf) * mx.linalg.norm(yf)).item()
        if denom == 0.0:
            continue
        worst = min(worst, ((xf @ yf).item() / denom))
    return worst


def _eps_term(outputs: list[mx.array]) -> float:
    term = 0.0
    for x in outputs:
        eps = _EPS.get(str(x.dtype).removeprefix("mlx.core."), 1.19e-7)
        scale = mx.abs(x.astype(mx.float32)).max().item() or 1.0
        term = max(term, EPS_MULTIPLE * eps * scale)
    return term


def preserving_check(
    baseline_run: Callable[[], object],
    patched_run: Callable[[], object],
    name: str,
) -> WorkloadCheck:
    """Assoc-preserving: patched vs original must sit on the orig-vs-orig
    floor. A large miss means the bind installed the wrong cut."""
    ref1 = _flatten(baseline_run())
    ref2 = _flatten(baseline_run())
    floor = _max_abs(ref1, ref2)
    got = _flatten(patched_run())
    max_abs = _max_abs(ref1, got)
    allowance = max(FLOOR_MULTIPLE * floor, _eps_term(ref1))
    return WorkloadCheck(
        name=name,
        floor_max_abs=floor,
        max_abs=max_abs,
        allowance=allowance,
        cosine=_cosine(ref1, got),
        passed=max_abs <= allowance,
    )


def step_veto(
    session: Session,
    baseline_run: Callable[[], object],
    patched_run: Callable[[], object],
    pairs: int = 16,
) -> tuple[PairedComparison, bool]:
    """The patched step must not be significantly slower: a non-regression
    veto under the interleaved paired discipline, not a detection gate."""
    result = compare(session, baseline_run, patched_run, pairs=pairs)
    margin_ms = STEP_VETO_PCT * result.median_baseline_ms
    return result, not result.loses_by(margin_ms)


def run_e2e(
    session: Session,
    baseline_model: Callable,
    patched_model: Callable,
    workloads: Sequence[tuple[str, list[mx.array]]],
    veto_pairs: int = 16,
) -> E2EResult:
    result = E2EResult()
    for name, tensors in workloads:
        result.checks.append(preserving_check(
            lambda: baseline_model(*tensors),
            lambda: patched_model(*tensors),
            name,
        ))
    first = workloads[0][1]
    result.veto, result.veto_passed = step_veto(
        session,
        lambda: baseline_model(*first),
        lambda: patched_model(*first),
        pairs=veto_pairs,
    )
    return result


def share_weights(donor, receiver) -> int:
    """Point receiver's parameters at donor's arrays and return how many
    were shared; the caller verifies the count is every parameter."""
    import mlx.nn as nn

    if not isinstance(receiver, nn.Module) or not isinstance(donor, nn.Module):
        return 0
    params = donor.parameters()
    receiver.update(params)
    shared = 0
    flat_d = _flatten_params(params)
    flat_r = _flatten_params(receiver.parameters())
    for k, v in flat_d.items():
        if k in flat_r and flat_r[k] is v:
            shared += 1
    return shared


def _flatten_params(tree, prefix="") -> dict:
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(_flatten_params(v, f"{prefix}.{k}"))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            out.update(_flatten_params(v, f"{prefix}.{i}"))
    else:
        out[prefix] = tree
    return out
