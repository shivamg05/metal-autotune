"""Canned judge scripts for the loop and judge tests: a judge that wins, one
that fixes its first attempt, and one that babbles."""

from typing import Sequence

from autotuner.judge.scripted import ScriptedJudge


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
