"""Deterministic fake judge (plan 5.11). All loop and ladder tests run against
this; no test depends on a live LLM.

A script is a list of raw response payloads consumed one per call, each pushed
through the same validate_response boundary as the real client: a malformed
step consumes the one re-ask (the next step answers it), and two malformed in
a row raise JudgeBabble, exactly the client's behavior.
"""

from __future__ import annotations

from typing import Sequence

from .schema import (
    JudgeBabble,
    MalformedResponse,
    NextResponse,
    SeedResponse,
    validate_response,
)


class ScriptExhausted(RuntimeError):
    """The canned script has no step left for this call."""


class ScriptedJudge:
    """Same two entry points as the real client, driven by a canned script."""

    def __init__(self, script: Sequence[object]):
        self._script = list(script)
        self._pos = 0
        self.seen: list[tuple] = []  # (entry_point, region_meta, verdict) per call

    @property
    def steps_consumed(self) -> int:
        return self._pos

    def seed(self, region_meta: dict) -> SeedResponse:
        self.seen.append(("seed", region_meta, None))
        return self._take(SeedResponse)

    def next(self, region_meta: dict, verdict: object) -> NextResponse:
        self.seen.append(("next", region_meta, verdict))
        return self._take(NextResponse)

    def _take(self, want: type):
        reason = None
        for _attempt in range(2):  # the original ask, then the one re-ask
            if self._pos >= len(self._script):
                raise ScriptExhausted(f"script ended after {self._pos} steps")
            raw = self._script[self._pos]
            self._pos += 1
            try:
                response = validate_response(raw)
                if not isinstance(response, want):
                    raise MalformedResponse(f"expected a {want.__name__} shape")
                return response
            except MalformedResponse as e:
                reason = e
        raise JudgeBabble(str(reason))


def proposal(source: str = "out0[i] = in0[i];", parent_kernel_id: str = "scaffold",
             **overrides: object) -> dict:
    """A minimal valid kernel-proposal payload for scripts and tests."""
    payload = {
        "source": source,
        "parent_kernel_id": parent_kernel_id,
        "grid": ["in0.shape[0]", "1", "1"],
        "threadgroup": ["1", "1", "1"],
        "output_shapes": [["in0.shape[0]"]],
    }
    payload.update(overrides)
    return payload


def winning_judge(kernel: dict | None = None) -> ScriptedJudge:
    """Seeds one on-chip item, proposes one winning kernel for it, then yields."""
    kernel = kernel if kernel is not None else proposal()
    return ScriptedJudge([
        {"queue": [{"id": "h1", "kind": "on-chip", "assoc_tag": "preserving",
                    "hypothesis": "keep the chain's intermediates in registers"}]},
        {"mutations": [], "kernel": kernel},
        {"mutations": [], "kernel": None},
    ])


def fix_judge(broken: dict | None = None, fixed: dict | None = None) -> ScriptedJudge:
    """Proposes a kernel that fails compile, then inserts a fix item on the
    failure branch and proposes the repaired kernel, then yields."""
    broken = broken if broken is not None else proposal(source="out0[i] = in0[i]")  # missing ;
    fixed = fixed if fixed is not None else proposal()
    return ScriptedJudge([
        {"queue": [{"id": "h1", "kind": "on-chip", "assoc_tag": "preserving",
                    "hypothesis": "keep the chain's intermediates in registers"}]},
        {"mutations": [], "kernel": broken},
        {"mutations": [{"op": "insert",
                        "item": {"id": "h2", "kind": "fix", "assoc_tag": "preserving",
                                 "hypothesis": "repair what broke",
                                 "depends_on": "h1", "condition": "failed"}}],
         "kernel": fixed},
        {"mutations": [], "kernel": None},
    ])


def babbling_judge(then: Sequence[object] = (), times: int = 1) -> ScriptedJudge:
    """Babbles malformed JSON `times` times, then continues with `then`. With
    times=1 the re-ask recovers; times=2 burns the hypothesis (JudgeBabble)."""
    return ScriptedJudge(["this is not a judge response"] * times + list(then))
