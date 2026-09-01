"""The judge: an LLM behind a strict JSON boundary (plan 5.11, section 10).

Metadata in, hypotheses and one kernel edit out. The queue and verdict log
are its only memory. The real client lives in judge.client; loop and ladder
tests use judge.scripted.
"""

from .prompts import render_region_state
from .queue import ABANDON_STRIKES, FamilyBook, Queue, QueueError
from .schema import (
    ASSOC_TAGS,
    CONDITIONS,
    KINDS,
    OUTCOMES,
    DeleteItem,
    InsertItem,
    JudgeBabble,
    KernelProposal,
    MalformedResponse,
    NextResponse,
    QueueItem,
    ReorderItems,
    SeedResponse,
    validate_response,
)
from .scripted import ScriptedJudge, ScriptExhausted

__all__ = [
    "ABANDON_STRIKES", "ASSOC_TAGS", "CONDITIONS", "KINDS", "OUTCOMES",
    "DeleteItem", "FamilyBook", "InsertItem", "JudgeBabble", "KernelProposal",
    "MalformedResponse", "NextResponse", "Queue", "QueueError", "QueueItem",
    "ReorderItems", "ScriptExhausted", "ScriptedJudge", "SeedResponse",
    "render_region_state", "validate_response",
]
