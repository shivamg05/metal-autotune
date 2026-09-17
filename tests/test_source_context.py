"""Source navigation must be exact, bounded, provider-independent and read-only."""
import copy
import json

import pytest

from autotuner.judge.client import JsonJudge
from autotuner.judge.schema import JudgeBabble, MalformedResponse, NextResponse
from autotuner.judge.source_context import SourceContext, MAX_READ_ROUNDS, PREVIEW_CHARS


HEADER = "// unrelated λ\n" * 1500 + "template<typename T> T focus(T x) { return helper(x); }\n" + "// tail\n" * 1500
META = {"kernels": {"head": {"source": "out0[0] = focus<T>(in0[0]);", "header": HEADER}},
        "history": [{"id": "bad", "verdict": "failed", "reason": "compile failed"}],
        "lessons": ["Repair the failed helper."]}


def test_focus_keeps_history_and_finds_entry_without_mutating_source():
    original = copy.deepcopy(META)
    context = SourceContext()
    brief = context.focus(META)
    assert META == original
    assert brief["history"] == META["history"] and brief["lessons"] == META["lessons"]
    assert brief["kernels"]["head"]["source"] == META["kernels"]["head"]["source"]
    excerpt = brief["source_catalog"]["source_1"]["excerpts"][0]
    assert "T focus(" in excerpt["text"]
    assert excerpt["text"] == HEADER[excerpt["start"]:excerpt["end"]]
    assert len(json.dumps(brief)) < len(json.dumps(META)) / 2
    # Every character, including Unicode, is still accessible, in exact order.
    parts, start = [], 0
    while start is not None:
        result = context.read({"read_source": [{"id": "source_1", "start": start}]})[0]
        parts.append(result["text"])
        start = result["next_start"]
    assert "".join(parts) == HEADER


def test_equal_sources_share_ids_distinct_versions_do_not():
    context = SourceContext()
    brief = context.focus({"a": {"header": HEADER}, "b": {"header": HEADER},
                           "c": {"header": HEADER + "\n// edit"},
                           "d": {"source": HEADER}})
    assert brief["a"]["header"] == brief["b"]["header"] == brief["d"]["source"]
    assert brief["c"]["header"] != brief["a"]["header"]
    assert len(context.sources) == 2


def test_long_history_cannot_multiply_previews_and_head_has_priority():
    kernels = {f"v{i}": {"source": "focus<float>(x);", "header": HEADER + f"// v{i}"}
               for i in range(15)}
    payload = {"region_state": {"head": "v14", "kernels": kernels,
                               "history": [{"id": f"h{i}", "verdict": "failed"} for i in range(15)]},
               "verdict": {"kernel_id": "v13", "outcome": "failed"}}
    context = SourceContext()
    brief = context.focus(payload)
    entries = brief["source_catalog"]
    assert sum(len(e["text"]) for v in entries.values() for e in v["excerpts"]) == PREVIEW_CHARS
    assert entries[brief["region_state"]["kernels"]["v14"]["header"]["source_id"]]["excerpts"]
    assert entries[brief["region_state"]["kernels"]["v13"]["header"]["source_id"]]["excerpts"]
    old_id = brief["region_state"]["kernels"]["v0"]["header"]["source_id"]
    assert entries[old_id]["excerpts"] == []
    assert context.read({"read_source": [{"id": old_id, "find": "focus("}]})[0]["matches"]
    assert brief["region_state"]["history"] == payload["region_state"]["history"]


def test_literal_search_handles_overloads_pagination_and_absent_names():
    context = SourceContext()
    source = "// padding\n" * 1000 + "float helper(float x);\nint helper(int x);\n" * 12
    context.focus({"header": source})
    hits = context.read({"read_source": [{"id": "source_1", "find": "helper("}]})[0]
    assert len(hits["matches"]) == 20
    rest = context.read({"read_source": [{"id": "source_1", "find": "helper(", "start": hits["next_start"]}]})[0]
    assert len(rest["matches"]) == 4 and rest["next_start"] is None
    assert context.read({"read_source": [{"id": "source_1", "find": ".*"}]})[0]["matches"] == []


@pytest.mark.parametrize("query", [
    {}, {"read_source": []}, {"read_source": [{}] * 5},
    {"read_source": [{"id": "/etc/passwd"}]},
    {"read_source": [{"id": "source_1", "path": "weights.safetensors"}]},
    {"read_source": [{"id": "source_1", "start": -1}]},
    {"read_source": [{"id": "source_1", "start": True}]},
    {"read_source": [{"id": "source_1", "start": 10**20}]},
    {"read_source": [{"id": "source_1", "length": 8001}]},
    {"read_source": [{"id": "source_1", "length": 0}]},
    {"read_source": [{"id": "source_1", "find": ""}]},
    {"read_source": [{"id": "source_1", "find": "x", "length": 10}]},
    {"read_source": [{"id": "source_1"}], "mutations": []},
])
def test_invalid_reads_cannot_escape_source_catalog(query):
    context = SourceContext()
    context.focus(META)
    with pytest.raises(MalformedResponse):
        context.read(query)


def test_reads_are_inside_one_proposal_and_retry_preserves_evidence(tmp_path):
    class Judge(JsonJudge):
        calls = []
        def _ask(self, system, messages):
            self.calls.append(copy.deepcopy(messages))
            return [
                '{"read_source":[{"id":"source_1","find":"focus("}]}',
                '{"read_source":[{"id":"source_1","start":22000,"length":2000}]}',
                'bad JSON', '{"mutations":[],"kernel":null}',
            ][len(self.calls)-1]
    judge = Judge()
    judge.transcript = tmp_path / "judge.jsonl"
    assert isinstance(judge.next(META, {"outcome": "failed"}), NextResponse)
    final = judge.calls[-1]
    assert json.loads(final[0]["content"])["region_state"]["history"] == META["history"]
    assert json.loads(final[2]["content"])["source_reads"][0]["matches"]
    assert json.loads(final[4]["content"])["source_reads"][0]["text"] == HEADER[22000:24000]
    requests = [json.loads(s) for s in judge.transcript.read_text().splitlines() if json.loads(s)["event"] == "request"]
    assert [r["attempt"] for r in requests] == [0, 0, 0, 1]
    assert [r["source_rounds"] for r in requests] == [0, 1, 2, 2]
    assert all(r["context_chars"] == len(r["system"]) + sum(len(m["content"]) for m in r["messages"]) for r in requests)


def test_endless_reads_are_bounded():
    class Judge(JsonJudge):
        calls = 0
        def _ask(self, *args):
            self.calls += 1
            return '{"read_source":[{"id":"source_1","length":1}]}'
    judge = Judge()
    with pytest.raises(JudgeBabble, match="lookup limit"):
        judge.next(META, {})
    assert judge.calls == MAX_READ_ROUNDS + 2


def test_lookup_instructions_only_appear_when_there_is_source_to_fetch():
    class Judge(JsonJudge):
        systems = []
        def _ask(self, system, messages):
            self.systems.append(system)
            return '{"mutations":[],"kernel":null}'
    judge = Judge()
    judge.next({"kernels": {"head": {"source": "out0[0]=in0[0];"}}}, {})
    judge.next(META, {})
    assert '"read_source"' not in judge.systems[0]
    assert '{"read_source":[{"id":"source_1","start":0,"length":8000}]}' in judge.systems[1]


def test_missing_id_uses_existing_malformed_retry_and_cannot_reset_it():
    class Judge(JsonJudge):
        calls = 0
        def _ask(self, *args):
            self.calls += 1
            if self.calls == 2:
                return '{"read_source":[{"id":"source_1","length":1}]}'
            return '{"read_source":[{"id":"unknown"}]}'
    judge = Judge()
    with pytest.raises(JudgeBabble, match="unknown source"):
        judge.next(META, {})
    assert judge.calls == 3
