"""Deterministic fake judge. All loop and ladder tests run against
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
