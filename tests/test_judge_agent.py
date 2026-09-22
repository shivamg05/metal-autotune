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

from autotuner.judge.agent import (
    PROMPT_TOKEN, RESPONSE_TOKEN, SYSTEM_TOKEN, AgentFileJudge, CliJudge,
    claude_argv, codex_argv, gemini_argv)
from autotuner.judge.client import DEFAULT_MODEL
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
if "--output-last-message" in sys.argv:
    path = Path(sys.argv[sys.argv.index("--output-last-message") + 1])
    if reply != "<missing>":
        path.write_text(reply)
    sys.stdout.write("progress: thinking about the next kernel\\n")
else:
    sys.stdout.write(reply)
"""


def cli_judge(tmp_path, answers, command=None):
    state = tmp_path / "state"
    state.mkdir()
    (state / "answers.json").write_text(json.dumps(answers))
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    base = [sys.executable, str(stub), str(state)]
    return CliJudge(command=base + (command or [])), state


def calls(state):
    return [json.loads(p.read_text()) for p in sorted(state.glob("call_*.json"))]


class TestCliJudge:
    @pytest.mark.parametrize("command", [[], [SYSTEM_TOKEN, PROMPT_TOKEN], codex_argv()[1:]])
    def test_readiness_uses_search_transport(self, tmp_path, command):
        judge, state = cli_judge(tmp_path, ['{"ready":true}', SEED_JSON], command=command)
        judge.check_available()
        assert len(calls(state)) == 1
        assert isinstance(judge.seed(REGION_META), SeedResponse)

    @pytest.mark.parametrize("reply", ["", "not JSON", '{"ready":false}', "<die>"])
    def test_readiness_refuses_unusable_judge(self, tmp_path, reply):
        judge, state = cli_judge(tmp_path, [reply])
        with pytest.raises(ValueError, match="before model loading"):
            judge.check_available()
        assert len(calls(state)) == 1  # no silent retries or budget-consuming search

    def test_readiness_preserves_error_text_even_with_success_exit_code(self, tmp_path):
        judge, _ = cli_judge(tmp_path, ["Provider quota exhausted"])
        with pytest.raises(ValueError, match="Provider quota exhausted"):
            judge.check_available()

    def test_failure_preserves_both_streams_and_login_instruction(self, tmp_path):
        stub = tmp_path / "claude"
        stub.write_text(f"#!{sys.executable}\nimport sys\n"
                        "print('OAuth token has expired. Please login.')\n"
                        "print('diagnostic from stderr', file=sys.stderr)\nsys.exit(1)\n")
        stub.chmod(0o755)
        judge = CliJudge([str(stub)])
        with pytest.raises(ValueError) as failure:
            judge.check_available()
        message = str(failure.value)
        assert "stdout: OAuth token has expired" in message
        assert "stderr: diagnostic from stderr" in message
        assert "claude auth login" in message

    def test_readiness_timeout_is_bounded(self):
        judge = CliJudge([sys.executable, "-c", "import time; time.sleep(30)"], timeout_s=.1)
        started = time.monotonic()
        with pytest.raises(ValueError, match="no reply within"):
            judge.check_available()
        assert time.monotonic() - started < 5

    def test_seed_round_trip(self, tmp_path):
        # no {system}/{prompt} token: the system is folded onto the messages and
        # the whole prompt is delivered on stdin, the same for any agent
        judge, state = cli_judge(tmp_path, [SEED_JSON])
        response = judge.seed(REGION_META)
        assert isinstance(response, SeedResponse)
        assert response.queue[0].id == "h1"
        (call,) = calls(state)
        stdin = call["stdin"]
        assert "abc123" in stdin and "[user]" in stdin
        # the system prompt (role + seed schema) leads the prompt on stdin
        assert "judge" in stdin and '"queue"' in stdin
        assert stdin.index("judge") < stdin.index("[user]")

    def test_system_and_prompt_tokens_are_substituted(self, tmp_path):
        # a command with tokens gets the system on argv and the messages as an
        # argument, and nothing on stdin: the custom-command/gemini shape
        judge, state = cli_judge(tmp_path, [SEED_JSON], command=[SYSTEM_TOKEN, PROMPT_TOKEN])
        judge.seed(REGION_META)
        (call,) = calls(state)
        assert call["stdin"] == ""
        assert "judge" in call["argv"][-2] and '"queue"' in call["argv"][-2]
        assert call["argv"][-1].startswith("[user]") and "abc123" in call["argv"][-1]

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

    def test_claude_preset_disables_tools(self):
        argv = claude_argv("claude-opus-5")
        assert argv[:2] == ["claude", "-p"]
        assert argv[argv.index("--tools") + 1] == ""
        assert argv[argv.index("--setting-sources") + 1] == ""
        assert argv[argv.index("--model") + 1] == "claude-opus-5"
        assert argv[argv.index("--effort") + 1] == "low"  # pinned, never the CLI's own default
        assert claude_argv("claude-opus-5", "medium")[claude_argv().index("--effort") + 1] == "medium"
        with pytest.raises(ValueError, match="judge effort"):
            claude_argv("claude-opus-5", "extreme")
        # the harness system prompt rides --system-prompt, not stdin
        assert argv[-2:] == ["--system-prompt", SYSTEM_TOKEN]

    def test_codex_preset_uses_stdin_and_final_response_file(self):
        argv = codex_argv()
        assert argv[:2] == ["codex", "exec"]
        assert "--skip-git-repo-check" in argv  # the judge directory is not a checkout
        assert "--ignore-user-config" in argv and "--ephemeral" in argv
        assert "project_doc_max_bytes=0" in argv
        assert argv[argv.index("--sandbox") + 1] == "read-only"
        assert "--model" not in argv and DEFAULT_MODEL not in argv
        assert PROMPT_TOKEN not in argv  # large region sources must not hit argv size limits
        assert argv[-3:] == ["--output-last-message", RESPONSE_TOKEN, "-"]
        disabled = {argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == "--disable"}
        assert {"shell_tool", "plugins", "memories", "apps", "multi_agent"} <= disabled

    def test_provider_defaults_and_explicit_models(self):
        argv = claude_argv()
        assert argv[argv.index("--model") + 1] == DEFAULT_MODEL
        assert "--model" not in gemini_argv()
        for preset in (codex_argv, claude_argv, gemini_argv):
            argv = preset("chosen-model")
            assert argv[argv.index("--model") + 1] == "chosen-model"

    def test_codex_shaped_command_ignores_stdout_progress(self, tmp_path):
        judge, state = cli_judge(tmp_path, [SEED_JSON, YIELD_JSON], command=codex_argv()[1:])
        assert isinstance(judge.seed(REGION_META), SeedResponse)
        assert isinstance(judge.next(REGION_META, {"outcome": "failed"}), NextResponse)
        first, second = calls(state)
        for call in (first, second):
            argv = call["argv"]
            path = Path(argv[argv.index("--output-last-message") + 1])
            assert path.is_absolute() and path.parent.parent == Path(judge._cwd)
            assert "abc123" in call["stdin"] and "judge" in call["stdin"]
            assert RESPONSE_TOKEN not in argv and "abc123" not in " ".join(argv)
        paths = [call["argv"][call["argv"].index("--output-last-message") + 1]
                 for call in (first, second)]
        assert paths[0] != paths[1]  # late replies cannot overwrite the next ask's file

    def test_missing_final_file_cannot_reuse_the_previous_reply(self, tmp_path):
        judge, _ = cli_judge(tmp_path, [SEED_JSON, "<missing>"], command=codex_argv()[1:])
        assert isinstance(judge.seed(REGION_META), SeedResponse)
        with pytest.raises(RuntimeError, match="without writing its final response"):
            judge.seed(REGION_META)

    def test_malformed_final_file_gets_the_same_reask(self, tmp_path):
        judge, state = cli_judge(tmp_path, ["bad final JSON", SEED_JSON], command=codex_argv()[1:])
        assert isinstance(judge.seed(REGION_META), SeedResponse)
        first, second = calls(state)
        assert "bad final JSON" in second["stdin"] and "rejected" in second["stdin"]


@pytest.mark.parametrize("exit_mode", ["success", "error", "timeout"])
def test_cli_cleans_up_descendants_before_another_request(tmp_path, exit_mode):
    marker = tmp_path / "late_reply"
    started = tmp_path / "child_started"
    child = ("import pathlib,time;"
             f"pathlib.Path({str(started)!r}).write_text('ready');"
             "time.sleep(1);"
             f"pathlib.Path({str(marker)!r}).write_text('stale reply')")
    parent = ("import pathlib,subprocess,sys,time;"
              f"subprocess.Popen([sys.executable,'-c',{child!r}],"
              "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);\n"
              f"while not pathlib.Path({str(started)!r}).exists(): time.sleep(.01)\n")
    if exit_mode == "timeout":
        parent += "time.sleep(5)"
    elif exit_mode == "error":
        parent += "sys.exit(3)"
    else:
        parent += f"print({SEED_JSON!r})"
    judge = CliJudge([sys.executable, "-c", parent], timeout_s=.5)
    if exit_mode == "success":
        assert isinstance(judge.seed(REGION_META), SeedResponse)
    else:
        with pytest.raises(RuntimeError, match="no reply within|exited 3"):
            judge.seed(REGION_META)
    assert started.exists()  # the child was actually alive when its parent finished
    time.sleep(1.1)
    assert not marker.exists()


@pytest.mark.parametrize("provider,model", [("codex", None), ("codex", "chosen-model"),
                                           ("claude-cli", None), ("gemini", None)])
def test_cli_selects_provider_model_without_running_a_job(tmp_path, monkeypatch, provider, model):
    from types import SimpleNamespace
    from autotuner import cli, loop

    captured = {}

    class NoGpuRunner:
        def __init__(self, manifest, work_dir, judge_factory):
            captured["judge"] = judge_factory(None)

        def run(self):
            from autotuner.report import Report
            return Report(manifest_path="unused.yaml")

        def emit_artifact(self, path):
            return path

    monkeypatch.setattr(loop, "JobRunner", NoGpuRunner)
    monkeypatch.setattr("autotuner.judge.agent.shutil.which", lambda command: command)
    monkeypatch.setattr(CliJudge, "check_available", lambda self: None)
    args = ["run", "unused.yaml", "--judge", provider, "--work-dir", str(tmp_path)]
    if model:
        args += ["--model", model]
    assert cli.main(args) == 0
    argv = captured["judge"]._command
    if model or provider == "claude-cli":
        assert argv[argv.index("--model") + 1] == (model or DEFAULT_MODEL)
    else:
        assert "--model" not in argv


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
    from autotuner.judge.client import _NEXT_SCHEMA, _SEED_SCHEMA, _SYSTEM, _SOURCE_GUIDE

    for schema in (_SEED_SCHEMA, _NEXT_SCHEMA):
        for guide in ("", _SOURCE_GUIDE):
            prompt = _SYSTEM.format(schema=schema, source_guide=guide)
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
    judge = CliJudge(claude_argv(DEFAULT_MODEL), timeout_s=300)
    response = judge.seed(REGION_META)
    assert isinstance(response, SeedResponse)
    assert len(response.queue) >= 1
