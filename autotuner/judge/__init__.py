"""The judge: an LLM behind a strict JSON boundary.

Metadata in, hypotheses and one kernel edit out. The queue and verdict log
are its only memory. The real client lives in judge.client; loop and ladder
tests use judge.scripted.
"""

from .prompts import render_region_state
from .queue import Queue, QueueError
from .schema import (
    ASSOC_TAGS,
    CONDITIONS,
    SUGGESTED_KINDS,
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
    "ASSOC_TAGS", "CONDITIONS", "SUGGESTED_KINDS", "OUTCOMES",
    "DeleteItem", "InsertItem", "JudgeBabble", "KernelProposal",
    "MalformedResponse", "NextResponse", "Queue", "QueueError", "QueueItem",
    "ReorderItems", "ScriptExhausted", "ScriptedJudge", "SeedResponse",
    "render_region_state", "validate_response",
]
