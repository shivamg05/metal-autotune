"""The no-API-key judge transports: claude-cli subprocess and the agent
file mailbox. Both ride the shared exchange loop, so these tests pin what
each transport adds: process spawning and argv/stdin content for one, the
request/response handshake for the other, plus the re-ask and babble paths
end to end through each."""

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from autotuner.judge.agent import AgentFileJudge, ClaudeCLIJudge
from autotuner.judge.schema import JudgeBabble, NextResponse, SeedResponse

REGION_META = {"fingerprint": "abc123", "ops": ["multiply", "add", "exp"],
               "roofline": {"bound": "memory", "s_max": 3.1}}

SEED_JSON = json.dumps({"queue": [
    {"id": "h1", "kind": "on-chip", "assoc_tag": "preserving",
     "hypothesis": "fuse the three elementwise ops into one pass"},
]})

YIELD_JSON = json.dumps({"mutations": [], "kernel": None})

# the stub judge: pops the next canned reply, logs stdin and argv per call
STUB = """\
import json, sys
from pathlib import Path
state = Path(sys.argv[1])
answers = json.loads((state / "answers.json").read_text())
n = len(list(state.glob("call_*.json")))
(state / f"call_{n}.json").write_text(json.dumps(
    {"stdin": sys.stdin.read(), "argv": sys.argv[2:]}))
reply = answers[n]
if reply == "<die>":
    sys.stderr.write("no such model\\n")
    sys.exit(3)
sys.stdout.write(reply)
"""


def cli_judge(tmp_path, answers):
    state = tmp_path / "state"
    state.mkdir()
    (state / "answers.json").write_text(json.dumps(answers))
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    return ClaudeCLIJudge(command=[sys.executable, str(stub), str(state)]), state


def calls(state):
    return [json.loads(p.read_text()) for p in sorted(state.glob("call_*.json"))]


class TestClaudeCLIJudge:
    def test_seed_round_trip(self, tmp_path):
        judge, state = cli_judge(tmp_path, [SEED_JSON])
        response = judge.seed(REGION_META)
        assert isinstance(response, SeedResponse)
        assert response.queue[0].id == "h1"
        (call,) = calls(state)
        assert "abc123" in call["stdin"] and call["stdin"].startswith("[user]")
        # system prompt rides argv and carries the role plus the seed schema
        assert call["argv"][-2] == "--system-prompt"
        assert "judge" in call["argv"][-1] and '"queue"' in call["argv"][-1]

    def test_next_yield(self, tmp_path):
        judge, _ = cli_judge(tmp_path, [YIELD_JSON])
        response = judge.next(REGION_META, {"outcome": "failed"})
        assert isinstance(response, NextResponse)
        assert response.kernel is None and response.mutations == ()

    def test_malformed_gets_one_reask(self, tmp_path):
        judge, state = cli_judge(tmp_path, ["definitely not json", SEED_JSON])
        response = judge.seed(REGION_META)
        assert isinstance(response, SeedResponse)
        first, second = calls(state)
        assert "rejected" in second["stdin"]
        assert "definitely not json" in second["stdin"]

    def test_two_malformed_is_babble(self, tmp_path):
        judge, _ = cli_judge(tmp_path, ['{"queue": []}', "still wrong"])
        with pytest.raises(JudgeBabble):
            judge.seed(REGION_META)

    def test_process_failure_raises(self, tmp_path):
        judge, _ = cli_judge(tmp_path, ["<die>"])
        with pytest.raises(RuntimeError, match="no such model"):
            judge.seed(REGION_META)

    def test_default_argv_disables_tools(self):
        argv = ClaudeCLIJudge(model="claude-opus-5")._argv()
        assert argv[:2] == ["claude", "-p"]
        assert argv[argv.index("--tools") + 1] == ""
        assert argv[argv.index("--setting-sources") + 1] == ""
        assert argv[argv.index("--model") + 1] == "claude-opus-5"


def respond(mailbox: Path, replies: list[str]):
    """The agent side of the handshake, on a thread: answer each request
    file in sequence with the next canned reply."""
    def run():
        for i, reply in enumerate(replies, start=1):
            request = mailbox / f"{i:04d}.request.json"
            for _ in range(200):
                if request.exists():
                    break
                time.sleep(0.02)
            else:
                return
            body = json.loads(request.read_text())
            (mailbox / body["respond_in"]).write_text(reply)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


class TestAgentFileJudge:
    def test_seed_round_trip(self, tmp_path):
        judge = AgentFileJudge(tmp_path / "judge_io", timeout_s=20)
        respond(tmp_path / "judge_io", [SEED_JSON])
        response = judge.seed(REGION_META)
        assert isinstance(response, SeedResponse)
        request = json.loads((tmp_path / "judge_io" / "0001.request.json").read_text())
        assert request["seq"] == 1
        assert request["respond_in"] == "0001.response.json"
        assert "judge" in request["system"]
        assert "abc123" in request["messages"][0]["content"]

    def test_reask_is_a_second_request(self, tmp_path):
        judge = AgentFileJudge(tmp_path / "judge_io", timeout_s=20, grace_s=0.5)
        respond(tmp_path / "judge_io", ["not json", SEED_JSON])
        response = judge.seed(REGION_META)
        assert isinstance(response, SeedResponse)
        second = json.loads((tmp_path / "judge_io" / "0002.request.json").read_text())
        # the re-ask carries the rejected reply and the reason, in order
        roles = [m["role"] for m in second["messages"]]
        assert roles == ["user", "assistant", "user"]
        assert second["messages"][1]["content"] == "not json"
        assert "rejected" in second["messages"][2]["content"]

    def test_two_malformed_is_babble(self, tmp_path):
        judge = AgentFileJudge(tmp_path / "judge_io", timeout_s=20, grace_s=0.5)
        respond(tmp_path / "judge_io", ["nope", "still nope"])
        with pytest.raises(JudgeBabble):
            judge.seed(REGION_META)

    def test_unanswered_mailbox_times_out(self, tmp_path):
        judge = AgentFileJudge(tmp_path / "judge_io", timeout_s=0.6)
        with pytest.raises(TimeoutError, match="judge agent"):
            judge.seed(REGION_META)

    def test_leftover_mailbox_refused(self, tmp_path):
        (tmp_path / "judge_io").mkdir()
        (tmp_path / "judge_io" / "0001.response.json").write_text(SEED_JSON)
        with pytest.raises(RuntimeError, match="previous run"):
            AgentFileJudge(tmp_path / "judge_io")

    def test_chunked_write_is_not_truncated(self, tmp_path):
        judge = AgentFileJudge(tmp_path / "judge_io", timeout_s=20)
        response = tmp_path / "judge_io" / "0001.response.json"

        def slow_writer():
            while not (tmp_path / "judge_io" / "0001.request.json").exists():
                time.sleep(0.02)
            with open(response, "w") as f:
                f.write(SEED_JSON[:40])
                f.flush()
                time.sleep(1.2)  # longer than any settle heuristic
                f.write(SEED_JSON[40:])
        threading.Thread(target=slow_writer, daemon=True).start()
        assert isinstance(judge.seed(REGION_META), SeedResponse)

    def test_delete_and_rewrite_survives(self, tmp_path):
        judge = AgentFileJudge(tmp_path / "judge_io", timeout_s=20)
        response = tmp_path / "judge_io" / "0001.response.json"

        def second_thoughts():
            while not (tmp_path / "judge_io" / "0001.request.json").exists():
                time.sleep(0.02)
            response.write_text("oops, prose")
            time.sleep(0.4)
            response.unlink()
            time.sleep(0.4)
            response.write_text(SEED_JSON)
        threading.Thread(target=second_thoughts, daemon=True).start()
        assert isinstance(judge.seed(REGION_META), SeedResponse)

    def test_stable_non_json_reaches_the_reask(self, tmp_path):
        judge = AgentFileJudge(tmp_path / "judge_io", timeout_s=20, grace_s=0.5)
        respond(tmp_path / "judge_io", ["never json", SEED_JSON])
        assert isinstance(judge.seed(REGION_META), SeedResponse)

    def test_sequence_spans_calls(self, tmp_path):
        judge = AgentFileJudge(tmp_path / "judge_io", timeout_s=20)
        respond(tmp_path / "judge_io", [SEED_JSON, YIELD_JSON])
        judge.seed(REGION_META)
        judge.next(REGION_META, {"outcome": "failed"})
        names = sorted(p.name for p in (tmp_path / "judge_io").glob("*.request.json"))
        assert names == ["0001.request.json", "0002.request.json"]


def test_next_prompt_carries_the_item_schema():
    """A live decode run burned its whole budget on this: 'item: as
    in the seed schema' referenced a schema the next prompt never included,
    so every live judge invented item keys and babbled."""
    from autotuner.judge.client import _NEXT_SCHEMA, _SEED_SCHEMA, _SYSTEM

    for schema in (_SEED_SCHEMA, _NEXT_SCHEMA):
        prompt = _SYSTEM.format(schema=schema)
        for field in ('"id"', '"kind"', '"assoc_tag"', '"hypothesis"'):
            assert field in prompt, f"{field} missing from prompt"
    assert "as in the seed schema" not in _NEXT_SCHEMA


def test_safe_detail_strips_the_acceptance_envelope():
    """Hard law: the judge sees distances, never the envelope it must land
    under, wherever those keys sit in the detail tree."""
    from autotuner.loop import _safe_detail

    detail = {"err_candidate": 0.02, "err_library": 0.01, "kappa": 1.25,
              "floor": 1e-4, "max_excess": 0.007,
              "nested": {"changing_floor": 1e-4, "rtol": 1e-3, "note": "kept"},
              "outputs": [{"atol": 1e-5, "max_excess": 0.001}]}
    safe = _safe_detail(detail)
    assert safe == {"max_excess": 0.007, "nested": {"note": "kept"},
                    "outputs": [{"max_excess": 0.001}]}


@pytest.mark.skipif(not os.environ.get("CLAUDE_CLI_LIVE_TEST"),
                    reason="live claude CLI call; set CLAUDE_CLI_LIVE_TEST=1")
def test_live_claude_cli_seed():
    """One real `claude -p` seed call on the machine's Claude Code login."""
    judge = ClaudeCLIJudge(timeout_s=300)
    response = judge.seed(REGION_META)
    assert isinstance(response, SeedResponse)
    assert len(response.queue) >= 1
