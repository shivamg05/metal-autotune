"""The smaller presentation must preserve code, failures and numeric guards."""
import copy
import json

import pytest

from autotuner.judge.briefing import share_context
from autotuner.judge.client import JsonJudge
from autotuner.loop import _kernel_view
from autotuner_runtime.kernels import KernelSpec


def expand(payload):
    """Independent reader of the public briefing reference convention."""
    shared = payload.get("shared_context", {})
    def visit(value):
        if isinstance(value, dict):
            if set(value) == {"context_ref"} and value["context_ref"] in shared:
                return shared[value["context_ref"]]
            return {k: visit(v) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v) for v in value]
        return value
    return {k: visit(v) for k, v in payload.items() if k != "shared_context"}


def test_repeated_headers_and_failed_model_checks_roundtrip_exactly():
    header = "// Unicode λ, quote \" and literal backslash \\n\n" * 300
    source = "out0[thread_position_in_grid.x] = in0[thread_position_in_grid.x];\n" * 10
    failed = {"status": "correctness_failed", "reason": "nonfinite pattern mismatch",
              "checks": [{"name": f"w{i}", "passed": False, "reason": "NaN vs Infinity"}
                         for i in range(10)]}
    history = [{"id": "bad", "verdict": "rolled_back", "hypothesis": "try a new reduction"}]
    payload = {"region_state": {"kernels": {
        "a": {"source": source, "header": header, "model_check": failed},
        "b": {"source": source, "header": header, "model_check": failed}},
        "history": history, "lessons": [{"lesson": "This reduction lost accuracy."}]},
        "verdict": {"detail": {"model_check": failed}}}
    original = copy.deepcopy(payload)
    packed = share_context(payload)
    assert payload == original
    assert expand(json.loads(json.dumps(packed))) == original
    assert len(json.dumps(packed)) < len(json.dumps(original))
    assert list(packed["shared_context"].values()).count(header) == 1
    assert packed["region_state"]["history"] == history


def test_distinct_code_versions_are_never_merged():
    header = "inline float helper(float x) { return x; }\n" * 100
    payload = {"kernels": {"a": {"header": header}, "b": {"header": header + "// changed\n"}}}
    assert share_context(payload) == payload


def test_small_context_stays_inline():
    payload = {"kernels": {"a": {"source": "out0[0]=in0[0];"},
                            "b": {"source": "out0[0]=in0[0];"}}}
    assert share_context(payload) == payload


def test_model_metadata_with_literal_reference_names_stays_unambiguous():
    payload = {"region_state": {"args": {"context_ref": "context_1"}},
               "a": {"header": "x" * 1000}, "b": {"header": "x" * 1000}}
    assert share_context(payload) == payload
    assert "shared_context" not in share_context(payload)


def test_ordinary_library_header_is_visible_in_full():
    header = "// library implementation\n" * 1000
    kernel = KernelSpec("k", "k", ("in0",), ("out0",), "out0[0]=in0[0];", header=header)
    assert _kernel_view(kernel)["header"] == header
    assert kernel.header == header


def test_reference_display_never_becomes_executable_code(tmp_path):
    from autotuner.judge.schema import MalformedResponse, validate_response
    response = {"mutations": [], "kernel": {
        "source": {"context_ref": "context_1"}, "parent_kernel_id": "head",
        "grid": ["1"] * 3, "threadgroup": ["1"] * 3, "output_shapes": [["1"]]}}
    with pytest.raises(MalformedResponse):
        validate_response(response)


def test_transport_checks_secrets_before_sharing():
    class Judge(JsonJudge):
        def _ask(self, *args):
            pytest.fail("must reject before reaching the transport")
    private = {"rtol": .01, "reason": "x" * 300}
    with pytest.raises(ValueError, match="tolerance"):
        Judge().next({"a": {"model_check": private}, "b": {"model_check": private}}, {})


def test_retry_retains_full_briefing_and_reports_actual_size(tmp_path):
    class Judge(JsonJudge):
        calls = []
        def _ask(self, system, messages):
            self.calls.append((system, copy.deepcopy(messages)))
            return "not json" if len(self.calls) == 1 else '{"mutations": [], "kernel": null}'
    judge = Judge()
    judge.transcript = tmp_path / "judge.jsonl"
    header = "// full implementation\n" * 300
    metadata = {"kernels": {"a": {"header": header}, "b": {"header": header}}}
    judge.next(metadata, {"outcome": "failed", "detail": {"reason": "compile failed"}})
    assert len(judge.calls) == 2
    for system, messages in judge.calls:
        assert expand(json.loads(messages[0]["content"]))["region_state"] == metadata
        assert "failed to compile" in system
        assert "family that keeps losing" in system
    assert judge.calls[1][1][-2]["content"] == "not json"
    rows = [json.loads(line) for line in judge.transcript.read_text().splitlines()]
    requests = [row for row in rows if row["event"] == "request"]
    assert len(requests) == 2
    for row in requests:
        assert row["context_chars"] == len(row["system"]) + sum(len(m["content"]) for m in row["messages"])


@pytest.mark.parametrize("separate_system", [False, True])
def test_large_library_briefing_survives_cli_stdin(separate_system):
    """Exercise the subprocess paths used by Codex and Claude without a paid call."""
    import sys
    from autotuner.judge.agent import CliJudge, SYSTEM_TOKEN
    script = '''
import json, sys
document = sys.stdin.read()
first = document.split("[user]\\n", 1)[1].split("\\n\\n[your previous response]", 1)[0]
payload = json.loads(first)
kernels = payload["region_state"]["kernels"]
assert kernels["a"]["header"] == kernels["b"]["header"]
entry = payload["source_catalog"][kernels["a"]["header"]["source_id"]]
assert entry["characters"] == 160000
assert entry["excerpts"][0]["text"] == ("// full header\\n" * 10000)[:6000]
if "[your previous response]" not in document:
    print('{"read_source":[{"id":"source_1","start":150000,"length":8000}]}')
else:
    reply = json.loads(document.rsplit("[user]\\n", 1)[1])
    assert reply["source_reads"][0]["text"] == " " * 8000
    print('{"mutations": [], "kernel": null}')
'''
    command = [sys.executable, "-c", script]
    if separate_system:
        command.append(SYSTEM_TOKEN)
    header = "// full header\n" * 10000 + " " * 10000
    judge = CliJudge(command)
    judge.next({"kernels": {"a": {"header": header}, "b": {"header": header}}}, {})
