"""Judge transports that need no API key.

ClaudeCLIJudge shells out to the local `claude` CLI in print mode, so judge
calls run on the machine's Claude Code login. Each call is one fresh process
with every tool disabled, no settings, and an empty working directory, and
the harness's system prompt replaces the default one, so the judge still
sees exactly what the harness sends and nothing else.

AgentFileJudge turns judge calls into a file handshake so a live agent (a
Claude Code session opened on this repo) can be the judge: the harness
writes NNNN.request.json into a mailbox directory and blocks until the
agent writes NNNN.response.json holding one JSON object that matches the
schema carried inside the request. AGENT_JUDGE.md is the operator guide.

Both feed the same exchange loop as the API client, so malformed replies
get the one re-ask and then JudgeBabble, identically across transports.
"""

from __future__ import annotations

import atexit
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .client import DEFAULT_MODEL, JsonJudge

CLI_TIMEOUT_S = 600.0
MAILBOX_TIMEOUT_S = 1800.0
GRACE_S = 10.0
_POLL_S = 0.5


def _flatten(messages: list[dict]) -> str:
    """One prompt document from the message list; the re-ask needs the judge
    to see its rejected reply and the rejection reason in order."""
    parts = []
    for m in messages:
        tag = "your previous response" if m["role"] == "assistant" else "user"
        parts.append(f"[{tag}]\n{m['content']}")
    return "\n\n".join(parts)


class ClaudeCLIJudge(JsonJudge):
    """One `claude -p` process per ask, on the local Claude Code login."""

    def __init__(self, model: str = DEFAULT_MODEL, command: list[str] | None = None,
                 timeout_s: float = CLI_TIMEOUT_S):
        self._model = model
        self._command = command  # test override; None builds the claude argv
        self._timeout_s = timeout_s
        self._cwd: str | None = None  # empty dir so no CLAUDE.md or project context leaks in

    def _argv(self) -> list[str]:
        if self._command is not None:
            return list(self._command)
        return ["claude", "-p", "--output-format", "text", "--tools", "",
                "--strict-mcp-config", "--setting-sources", "",
                "--disable-slash-commands", "--no-session-persistence",
                "--model", self._model]

    def _ask(self, system: str, messages: list[dict]) -> str:
        if self._cwd is None:
            self._cwd = tempfile.mkdtemp(prefix="judge_cli_")
            atexit.register(shutil.rmtree, self._cwd, ignore_errors=True)
        try:
            proc = subprocess.run(
                self._argv() + ["--system-prompt", system],
                input=_flatten(messages), capture_output=True, text=True,
                cwd=self._cwd, timeout=self._timeout_s,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"claude CLI judge gave no reply within {self._timeout_s:.0f}s")
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip()[-500:]
            raise RuntimeError(f"claude CLI judge exited {proc.returncode}: {tail}")
        return proc.stdout.strip()


class AgentFileJudge(JsonJudge):
    """Blocking file mailbox: a live agent answers each request by hand."""

    def __init__(self, mailbox: str | Path, timeout_s: float = MAILBOX_TIMEOUT_S,
                 grace_s: float = GRACE_S):
        self._dir = Path(mailbox)
        self._dir.mkdir(parents=True, exist_ok=True)
        leftover = sorted(p.name for p in self._dir.glob("*.json"))
        if leftover:
            # a reused mailbox would answer new requests with last run's mail
            raise RuntimeError(
                f"judge mailbox {self._dir} holds {len(leftover)} file(s) from a "
                f"previous run ({leftover[0]} ...); move or delete them first")
        self._seq = 0
        self._timeout_s = timeout_s
        self._grace_s = grace_s

    def _ask(self, system: str, messages: list[dict]) -> str:
        self._seq += 1
        name = f"{self._seq:04d}"
        request = {"seq": self._seq, "respond_in": f"{name}.response.json",
                   "system": system, "messages": messages}
        tmp = self._dir / f".{name}.request.tmp"
        tmp.write_text(json.dumps(request, indent=2))
        tmp.rename(self._dir / f"{name}.request.json")
        return self._await_response(self._dir / f"{name}.response.json")

    def _await_response(self, path: Path) -> str:
        """Accept the file once it parses as one JSON document; a non-JSON
        file that stops changing for the grace window is handed over as-is so
        the re-ask can say why. Reads tolerate the agent still writing or
        deleting to rewrite."""
        deadline = time.monotonic() + self._timeout_s
        last_text = None
        stable_since = 0.0
        while time.monotonic() <= deadline:
            try:
                text = path.read_text()
            except OSError:
                text = ""
            if text:
                try:
                    json.loads(text)
                    return text
                except (json.JSONDecodeError, RecursionError):
                    now = time.monotonic()
                    if text != last_text:
                        last_text, stable_since = text, now
                    elif now - stable_since >= self._grace_s:
                        return text
            time.sleep(_POLL_S)
        raise TimeoutError(
            f"no complete {path.name} after {self._timeout_s:.0f}s; "
            f"is a judge agent watching {self._dir}? (see AGENT_JUDGE.md)")
