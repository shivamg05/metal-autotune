"""The judge. Schema boundary, queue mechanics, family bookkeeping,
prompt rendering, the scripted judge, and the real client's re-ask path
(mocked transport; the live test runs only under ANTHROPIC_LIVE_TEST=1)."""

import inspect
import json
import os
from types import SimpleNamespace

import pytest

from autotuner.judge import prompts
from autotuner.judge.client import AnthropicJudge
from autotuner.judge.queue import ABANDON_STRIKES, FamilyBook, Queue, QueueError
from autotuner.judge.schema import (
    JudgeBabble,
    KernelProposal,
    MalformedResponse,
    NextResponse,
    QueueItem,
    SeedResponse,
    validate_response,
)
from autotuner.judge.scripted import ScriptedJudge, ScriptExhausted
from tests.judges import babbling_judge, fix_judge, proposal, winning_judge
from autotuner.regions.types import Region, Roofline, Stretch
from autotuner_runtime import grammar


def item(item_id, kind="on-chip", assoc="preserving", hypothesis="registers", **kw):
    return QueueItem(id=item_id, kind=kind, assoc_tag=assoc, hypothesis=hypothesis, **kw)


def witem(item_id="h1", **over):
    d = {"id": item_id, "kind": "on-chip", "assoc_tag": "preserving", "hypothesis": "registers"}
    d.update(over)
    return d


def rejects(payload, fragment):
    with pytest.raises(MalformedResponse) as e:
        validate_response(payload)
    assert fragment in str(e.value), f"{fragment!r} not in {e.value}"


# ---------------------------------------------------------------- schema


def test_validate_seed_round_trip():
    resp = validate_response({"queue": [
        witem("h1"),
        witem("h2", kind="retile", assoc_tag="changing", family_id="split-k",
              depends_on="h1", condition="correct"),
    ]})
    assert isinstance(resp, SeedResponse)
    assert resp.queue[0] == item("h1")
    assert resp.queue[1] == item("h2", kind="retile", assoc="changing",
                                 family_id="split-k", depends_on="h1", condition="correct")


def test_validate_next_round_trip():
    resp = validate_response({
        "mutations": [
            {"op": "insert", "item": witem("h9", kind="fix"), "before": "h2"},
            {"op": "delete", "id": "h3"},
            {"op": "reorder", "order": ["h2", "h9"]},
        ],
        "kernel": proposal(template=[["T", "in0"], ["ACC", "float32"]],
                           fallback_predicate="in0.shape[0] > 64"),
    })
    assert isinstance(resp, NextResponse)
    ins, dele, reo = resp.mutations
    assert ins.item.id == "h9" and ins.before == "h2"
    assert dele.item_id == "h3"
    assert reo.order == ("h2", "h9")
    k = resp.kernel
    assert isinstance(k, KernelProposal)
    assert k.parent_kernel_id == "scaffold"
    assert k.grid == ("in0.shape[0]", "1", "1")
    assert k.output_shapes == (("in0.shape[0]",),)
    assert k.template == (("T", "in0"), ("ACC", "float32"))
    assert k.fallback_predicate == "in0.shape[0] > 64"


def test_validate_next_yield():
    resp = validate_response({"mutations": [], "kernel": None})
    assert resp.mutations == () and resp.kernel is None


def test_validate_rejects_shapes_and_items():
    rejects("nope", "JSON object")
    rejects({"queue": []}, "non-empty")
    rejects({"queue": [witem()], "kernel": None}, "seed response")
    rejects({"mutations": []}, "keys must be")
    rejects({"queue": [witem(kind="x" * 41)]}, "short label")
    rejects({"queue": [witem(kind="two\nlines")]}, "short label")
    rejects({"queue": [witem(assoc_tag="exact")]}, "assoc_tag")
    rejects({"queue": [{"id": "h1", "kind": "fix", "assoc_tag": "preserving"}]}, "hypothesis")
    rejects({"queue": [witem(depends_on="h0")]}, "come together")
    rejects({"queue": [witem(condition="correct")]}, "come together")
    rejects({"queue": [witem(), witem("h2", depends_on="h1", condition="maybe")]}, "condition")
    rejects({"queue": [witem(), witem()]}, "duplicate")
    rejects({"queue": [witem(depends_on="h2", condition="failed"), witem("h2")]},
            "not an earlier item")
    rejects({"queue": [witem(surprise=1)]}, "unknown keys")


def test_validate_rejects_harness_owned_kernel_fields():
    # init_value, math_mode, streams, and the kernel name are the harness's
    for key in ("init_value", "math_mode", "stream", "name", "output_dtypes"):
        rejects({"mutations": [], "kernel": proposal(**{key: "x"})}, "not the judge's to set")


def test_validate_rejects_bad_launch_config():
    rejects({"mutations": [], "kernel": proposal(grid=["1", "1"])}, "exactly 3")
    rejects({"mutations": [], "kernel": proposal(grid=["open('x')", "1", "1"])}, "kernel grid")
    rejects({"mutations": [], "kernel": proposal(output_shapes=[])}, "non-empty")
    rejects({"mutations": [], "kernel": proposal(template=[["T", "float99"]])}, "dtype name")
    rejects({"mutations": [], "kernel": proposal(template=[["2T", "float32"]])}, "C identifier")
    rejects({"mutations": [], "kernel": proposal(fallback_predicate=7)}, "string expression")
    rejects({"mutations": [], "kernel": proposal(source="")}, "source")


def test_validate_rejects_bad_mutations():
    rejects({"mutations": [{"op": "explode"}], "kernel": None}, "insert, delete, or reorder")
    rejects({"mutations": [{"op": "insert"}], "kernel": None}, "must be an object")
    rejects({"mutations": [{"op": "delete"}], "kernel": None}, "'id'")
    rejects({"mutations": [{"op": "reorder", "order": [1]}], "kernel": None}, "list of item ids")
    rejects({"mutations": [{"op": "insert", "item": witem(), "after": "h2"}], "kernel": None},
            "unknown keys")


# ---------------------------------------------------------------- queue


def test_pop_ready_spec_example():
    """The spec's example queue: pop order follows depends_on conditions, and
    unsatisfied items are skipped in place, not consumed."""
    q = Queue()
    q.seed([
        item("h1"),
        item("h2", kind="retile", depends_on="h1", condition="correct"),
        item("h3", kind="launch", depends_on="h2", condition="shipped"),
        item("h4", kind="fix", depends_on="h1", condition="failed"),
    ])
    assert q.pop_ready().id == "h1"
    assert q.pop_ready() is None                # everything waits on h1's verdict
    q.record_verdict("h1", "failed")
    assert q.pop_ready().id == "h4"             # failure branch; h2 skipped, kept
    q.record_verdict("h4", "correct_slower")
    assert q.pop_ready() is None                # h1 failed, so h2's 'correct' never holds
    assert q.ids() == ("h2", "h3")


def test_pop_ready_condition_satisfaction():
    q = Queue()
    q.seed([
        item("h1"),
        item("h2", depends_on="h1", condition="correct"),
        item("h3", depends_on="h2", condition="shipped"),
    ])
    q.pop_ready()
    q.record_verdict("h1", "correct_slower")    # correct-but-slower satisfies 'correct'
    assert q.pop_ready().id == "h2"
    q.record_verdict("h2", "tentative_ship")
    assert q.pop_ready() is None                # tentative is not shipped
    q.record_verdict("h2", "shipped")
    assert q.pop_ready().id == "h3"

    q2 = Queue()
    q2.seed([item("hA"), item("hB", depends_on="hA", condition="correct"),
             item("hC", depends_on="hA", condition="failed")])
    q2.pop_ready()
    q2.record_verdict("hA", "shipped")          # shipped is correct too
    assert q2.pop_ready().id == "hB"
    q2.record_verdict("hA", "rolled_back")      # a rollback overwrites: failure branch opens
    assert q2.pop_ready().id == "hC"


def test_verdict_rules():
    q = Queue()
    q.seed([item("h1")])
    with pytest.raises(QueueError):
        q.record_verdict("h1", "failed")        # still queued
    q.pop_ready()
    with pytest.raises(QueueError):
        q.record_verdict("h1", "exploded")      # not an outcome
    q.record_verdict("h1", "shipped")
    assert q.verdicts == {"h1": "shipped"}


def test_seed_rules():
    q = Queue()
    with pytest.raises(QueueError):
        q.seed([item("h1"), item("h1")])
    q = Queue()
    with pytest.raises(QueueError):
        q.seed([item("h1", depends_on="h2", condition="failed"), item("h2")])
    q = Queue()
    q.seed([item("h1")])
    with pytest.raises(QueueError):
        q.seed([item("h2")])                    # seed is once per region


def test_mutations_insert_delete_reorder():
    q = Queue()
    q.seed([item("h1")])
    q.pop_ready()
    q.record_verdict("h1", "failed")
    with pytest.raises(QueueError):             # only schema mutation records apply
        q.apply_mutations([SimpleNamespace()])
    resp = validate_response({"mutations": [
        {"op": "insert", "item": witem("h2", depends_on="h1", condition="failed")},
        {"op": "insert", "item": witem("h3", kind="retile"), "before": "h2"},
    ], "kernel": None})
    q.apply_mutations(resp.mutations)
    assert q.ids() == ("h3", "h2")
    reorder = validate_response({"mutations": [{"op": "reorder", "order": ["h2", "h3"]}],
                                 "kernel": None})
    q.apply_mutations(reorder.mutations)
    assert q.ids() == ("h2", "h3")
    delete = validate_response({"mutations": [{"op": "delete", "id": "h3"}], "kernel": None})
    q.apply_mutations(delete.mutations)
    assert q.ids() == ("h2",)
    assert q.pop_ready().id == "h2"             # depends_on h1 failed: satisfied


def test_mutation_errors():
    q = Queue()
    q.seed([item("h1"), item("h2")])

    def muts(payload):
        return validate_response({"mutations": payload, "kernel": None}).mutations

    with pytest.raises(QueueError):             # duplicate id in region history
        q.apply_mutations(muts([{"op": "insert", "item": witem("h1")}]))
    with pytest.raises(QueueError):             # dependency does not exist yet
        q.apply_mutations(muts([{"op": "insert",
                                 "item": witem("h5", depends_on="h9", condition="correct")}]))
    with pytest.raises(QueueError):
        q.apply_mutations(muts([{"op": "insert", "item": witem("h5"), "before": "h9"}]))
    with pytest.raises(QueueError):
        q.apply_mutations(muts([{"op": "delete", "id": "h9"}]))
    with pytest.raises(QueueError):             # not an exact permutation
        q.apply_mutations(muts([{"op": "reorder", "order": ["h1"]}]))
    assert q.ids() == ("h1", "h2")              # failed batch left the queue intact


# ---------------------------------------------------------------- families


def test_family_inheritance():
    book = FamilyBook()
    book.register_scaffold("scaffold")
    f1 = book.resolve(item("h1"), "scaffold")
    assert f1 == "family1"                      # parentless item starts a new family
    book.register_kernel("k1", f1)
    assert book.resolve(item("h2"), "k1") == "family1"   # inherits the parent kernel's
    assert book.resolve(item("h3", family_id="split-k"), "k1") == "split-k"  # declared wins
    assert book.resolve(item("h4"), "scaffold") == "family2"  # a second fresh family


def test_eight_strikes_trips_on_eighth():
    book = FamilyBook()
    fam = book.resolve(item("h1"), "scaffold")
    for _ in range(ABANDON_STRIKES - 1):
        book.record_verdict(fam, "correct_slower")
    assert not book.tripped(fam)
    book.record_verdict(fam, "correct_slower")
    assert book.tripped(fam)                    # the 8th correct-but-slower
    assert book.climbing == fam
    book.abandon(fam)
    assert book.abandoned(fam)
    assert book.climbing is None


def test_strikes_reset_on_ship_and_beaten_family_never_trips():
    book = FamilyBook()
    fam = book.resolve(item("h1"), "scaffold")
    for _ in range(5):
        book.record_verdict(fam, "correct_slower")
    book.record_verdict(fam, "tentative_ship")
    assert book.state()[fam] == {"strikes": 0, "beaten_library": True, "abandoned": False}
    for _ in range(ABANDON_STRIKES):
        book.record_verdict(fam, "correct_slower")
    assert not book.tripped(fam)                # it beat the library once; never abandoned
    book.record_verdict(fam, "rolled_back")     # rollback moves no counter
    assert book.state()[fam]["strikes"] == ABANDON_STRIKES


def test_climbing_tracks_last_slow_family():
    book = FamilyBook()
    fa = book.resolve(item("h1", family_id="one-pass"), "scaffold")
    fb = book.resolve(item("h2", family_id="two-pass"), "scaffold")
    book.record_verdict(fa, "correct_slower")
    assert book.climbing == "one-pass"
    book.record_verdict(fb, "correct_slower")
    assert book.climbing == "two-pass"
    book.record_verdict(fa, "failed")           # a fail does not move the climb
    assert book.climbing == "two-pass"
    book.record_verdict(fb, "shipped")
    assert book.climbing is None


# ---------------------------------------------------------------- prompts


def fixed_state():
    """One fixed region state, shared by the snapshot test and the live test."""
    region = Region(
        fingerprint="fp-exp-sin",
        ops=("mx.exp", "mx.sin"),
        members=[
            Stretch("decode", 3, 4, (10,), (12,), ("", "block.0")),
            Stretch("prefill", 3, 4, (10,), (12,), ("", "block.0")),
        ],
        t_orig_ms={"decode": 0.5, "prefill": 1.25},
        t_rep_ms={"decode": 0.5, "prefill": 1.25},
        p={"decode": 0.04, "prefill": 0.06},
        roofline=Roofline(t_mem_ms=0.125, t_compute_ms=0.0625, t_launch_ms=0.004,
                          t_roofline_ms=0.125, bound="memory", s_max=5.0),
    )
    io_specs = {
        "decode": {"inputs": [((1, 64), "float16")], "outputs": [((1, 64), "float16")]},
        "prefill": {"inputs": [((32, 64), "float16")], "outputs": [((32, 64), "float16")]},
    }
    ops = [{"op": "mx.exp", "args": ["in0"], "kwargs": {}, "outputs": ["t3"]},
           {"op": "mx.sin", "args": ["t3"], "kwargs": {}, "outputs": ["out0"]}]
    book = FamilyBook()
    book.register_scaffold("scaffold")
    fam = book.resolve(item("h1"), "scaffold")
    book.register_kernel("k1", fam)
    book.record_verdict(fam, "correct_slower")
    queue = Queue()
    queue.seed([
        item("h2", kind="retile", assoc="changing", hypothesis="tile K, 8 per thread"),
        item("h3", kind="launch", hypothesis="then grid 256x4",
             depends_on="h2", condition="shipped"),
    ])
    kernels = {"k1": {
        "source": "out0[i] = metal::precise::sin(metal::precise::exp(in0[i]));",
        "header": "", "grid": ["in0.shape[0]", "1", "1"], "threadgroup": ["1", "1", "1"],
        "output_shapes": [["in0.shape[0]"]], "output_dtypes": ["float16"], "template": [],
        "input_names": ["in0"], "output_names": ["out0"],
        "hypothesis_id": "h1", "verdict": "correct_slower", "failed_gate": None,
        "region_ms": 0.625, "library_ms": 0.5, "win_ms": -0.125,
    }}
    last_verdict = {"hypothesis_id": "h1", "kernel_id": "k1", "outcome": "correct_slower",
                    "failed_gate": None, "detail": {}, "region_ms": 0.625,
                    "library_ms": 0.5, "win_ms": -0.125, "sigma_ms": 0.01}
    writing_for = {"id": "h2", "kind": "retile", "assoc_tag": "changing",
                   "hypothesis": "tile K, 8 per thread"}
    return dict(region=region, io_specs=io_specs, ops=ops, kernels=kernels,
                head="k1", shipped=None, head_ms=0.625, shipped_ms=None,
                assoc_tag="preserving", families=book, queue=queue,
                last_verdict=last_verdict, writing_for=writing_for)


def test_prompt_snapshot():
    """The whole contract, rendered: the spec's judge_sees block plus the
    queue, the verdicts, the kernels in play, and a legend for every key."""
    state = fixed_state()
    rendered = prompts.render_region_state(**state)
    assert rendered["region"] == {
        "fingerprint": "fp-exp-sin",
        "ops": state["ops"],
        "io": {
            "decode": {"inputs": [[[1, 64], "float16"]], "outputs": [[[1, 64], "float16"]]},
            "prefill": {"inputs": [[[32, 64], "float16"]], "outputs": [[[32, 64], "float16"]]},
        },
        "copies": 2,
        "p": {"decode": 0.04, "prefill": 0.06},
        "bound": "memory",
        "T_orig_ms": {"decode": 0.5, "prefill": 1.25},
        "T_rep_ms": {"decode": 0.5, "prefill": 1.25},
        "roofline_ms": 0.125,
        "s_max": 5.0,
        "head_ms": 0.625,
        "shipped_ms": None,
        "head_minus_roofline_ms": 0.5,
        "shipped_minus_roofline_ms": None,
    }
    assert rendered["head"] == "k1" and rendered["shipped"] is None
    assert rendered["kernels"] == state["kernels"]
    assert rendered["family"] == "assoc-preserving"
    assert rendered["families"] == {
        "per_family": {"family1": {"strikes": 1, "beaten_library": False, "abandoned": False}},
        "climbing": "family1",
    }
    assert rendered["last_verdict"] == state["last_verdict"]
    assert rendered["writing_for"] == state["writing_for"]
    assert [q["id"] for q in rendered["queue"]] == ["h2", "h3"]
    assert rendered["queue"][0]["satisfied"] and not rendered["queue"][1]["satisfied"]
    assert rendered["verdicts"] == {}
    assert rendered["menu"] == prompts.MENU and rendered["laws"] == list(prompts.LAWS)
    # the grammar doc the judge reads is the evaluator's own contract
    assert rendered["launch_grammar"] == grammar.__doc__
    assert "ceil_div" in rendered["launch_grammar"] and "in0" in rendered["launch_grammar"]
    assert "out0" in rendered["body"] and "tmp0" in rendered["body"]
    for key in ("p", "T_orig_ms", "T_rep_ms", "roofline_ms", "s_max", "head_ms",
                "shipped_ms", "library_ms", "win_ms", "sigma_ms", "bound", "writing_for"):
        assert key in rendered["legend"]


def test_prompt_is_structurally_sealed():
    """The renderer's signature is the secrecy boundary: fixed keyword-only
    parameters, no tensor or tolerance parameter to pass."""
    params = inspect.signature(prompts.render_region_state).parameters
    assert set(params) == {"region", "io_specs", "ops", "kernels", "head", "shipped",
                           "head_ms", "shipped_ms", "assoc_tag", "families", "queue",
                           "last_verdict", "writing_for", "chip"}
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())


def test_prompt_refuses_tolerance_keys():
    state = fixed_state()
    state["last_verdict"] = {"detail": {"numeric": {"rtol": 1e-5}}}
    with pytest.raises(ValueError, match="rtol"):
        prompts.render_region_state(**state)
    state = fixed_state()
    state["kernels"] = {"k1": {"verdict": [{"atol": 1e-6}]}}
    with pytest.raises(ValueError, match="atol"):
        prompts.render_region_state(**state)


def test_prompt_refuses_non_json_data():
    state = fixed_state()
    state["last_verdict"] = {"detail": object()}
    with pytest.raises(TypeError, match="plain JSON"):
        prompts.render_region_state(**state)


def test_prompt_rejects_bad_assoc_tag():
    state = fixed_state()
    state["assoc_tag"] = "assoc-preserving"     # the tag, not the rendered form
    with pytest.raises(ValueError, match="assoc_tag"):
        prompts.render_region_state(**state)


# ---------------------------------------------------------------- scripted


def test_scripted_winning_round_trip():
    judge = winning_judge()
    meta = {"stub": True}
    seeded = judge.seed(meta)
    assert isinstance(seeded, SeedResponse)
    q = Queue()
    q.seed(seeded.queue)
    front = q.pop_ready()
    assert front.id == "h1" and front.kind == "on-chip"
    first = judge.next(meta, None)
    assert first.kernel is not None and first.kernel.parent_kernel_id == "scaffold"
    q.record_verdict("h1", "shipped")
    second = judge.next(meta, {"hypothesis_id": "h1", "outcome": "shipped"})
    assert second.kernel is None and second.mutations == ()
    assert q.empty                              # the judge yielded; close by empty queue
    with pytest.raises(ScriptExhausted):
        judge.next(meta, None)
    assert [call[0] for call in judge.seen] == ["seed", "next", "next", "next"]


def test_scripted_fix_flow():
    judge = fix_judge()
    q = Queue()
    q.seed(judge.seed({}).queue)
    assert q.pop_ready().id == "h1"
    assert judge.next({}, None).kernel is not None      # the broken attempt
    q.record_verdict("h1", "failed")
    fix = judge.next({}, {"hypothesis_id": "h1", "outcome": "failed",
                          "failed_gate": "compile"})
    q.apply_mutations(fix.mutations)
    front = q.pop_ready()
    assert front.id == "h2" and front.kind == "fix"
    assert front.depends_on == "h1" and front.condition == "failed"
    assert fix.kernel is not None                       # the repaired kernel


def test_scripted_babble_once_recovers():
    judge = babbling_judge(then=[{"mutations": [], "kernel": None}])
    resp = judge.next({}, None)
    assert resp.kernel is None
    assert judge.steps_consumed == 2            # the babble burned the one re-ask


def test_scripted_babble_twice_burns_hypothesis():
    judge = babbling_judge(times=2)
    with pytest.raises(JudgeBabble):
        judge.next({}, None)


def test_scripted_wrong_shape_counts_as_malformed():
    judge = ScriptedJudge([
        {"mutations": [], "kernel": None},      # a next shape offered to seed
        {"queue": [witem()]},
    ])
    assert isinstance(judge.seed({}), SeedResponse)
    assert judge.steps_consumed == 2


# ---------------------------------------------------------------- client


class FakeMessages:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=self.texts.pop(0))],
        )


class FakeSDK:
    def __init__(self, texts):
        self.messages = FakeMessages(texts)


def test_client_valid_first_try():
    sdk = FakeSDK([json.dumps({"queue": [witem()]})])
    judge = AnthropicJudge(client=sdk)
    resp = judge.seed({"stub": True})
    assert isinstance(resp, SeedResponse) and resp.queue[0].id == "h1"
    assert len(sdk.messages.calls) == 1
    call = sdk.messages.calls[0]
    assert "temperature" not in call            # current models reject sampling params
    assert json.loads(call["messages"][0]["content"]) == {"region_state": {"stub": True}}
    assert "Response schema (seed)" in call["system"]


def test_client_malformed_burns_one_reask_then_succeeds(monkeypatch):
    sdk = FakeSDK(["```json not json```", json.dumps({"mutations": [], "kernel": None})])
    judge = AnthropicJudge()
    monkeypatch.setattr(judge, "_client", sdk)  # mock the SDK transport
    resp = judge.next({"stub": True}, {"outcome": "failed"})
    assert isinstance(resp, NextResponse) and resp.kernel is None
    assert len(sdk.messages.calls) == 2
    reask = sdk.messages.calls[1]["messages"]
    assert len(reask) == 3                      # ask, the bad answer, the rejection
    assert reask[1]["role"] == "assistant"
    assert "rejected" in reask[2]["content"]


def test_client_malformed_twice_raises_judge_babble():
    sdk = FakeSDK(["not json", "still not json"])
    judge = AnthropicJudge(client=sdk)
    with pytest.raises(JudgeBabble):
        judge.seed({"stub": True})
    assert len(sdk.messages.calls) == 2         # exactly one re-ask, then the burn


def test_client_wrong_shape_triggers_reask():
    sdk = FakeSDK([
        json.dumps({"mutations": [], "kernel": None}),  # a next shape for a seed call
        json.dumps({"queue": [witem()]}),
    ])
    judge = AnthropicJudge(client=sdk)
    resp = judge.seed({"stub": True})
    assert isinstance(resp, SeedResponse)
    assert len(sdk.messages.calls) == 2


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_LIVE_TEST"),
                    reason="live LLM contract test; set ANTHROPIC_LIVE_TEST=1")
def test_client_live_seed_contract():
    judge = AnthropicJudge()
    meta = prompts.render_region_state(**fixed_state())
    resp = judge.seed(meta)
    assert isinstance(resp, SeedResponse) and resp.queue


def _plain_item(i, depends_on=None, condition=None):
    from autotuner.judge.schema import QueueItem

    return QueueItem(id=i, kind="retile", assoc_tag="preserving",
                     hypothesis="try a tile", depends_on=depends_on,
                     condition=condition)


def test_in_flight_id_stays_taken():
    """A crash seen live: an item was popped for execution, the judge
    re-proposed its id, the duplicate was accepted, and the verdict raised.
    A popped id must stay taken until its verdict lands."""
    from autotuner.judge.schema import InsertItem

    q = Queue()
    q.seed([_plain_item("tg1024"), _plain_item("other")])
    inflight = q.pop_ready()
    assert inflight.id == "tg1024"
    with pytest.raises(QueueError, match="already exists"):
        q.apply_mutations([InsertItem(item=_plain_item("tg1024"))])
    q.record_verdict(inflight.id, "failed")  # must not raise
    assert q.verdicts["tg1024"] == "failed"


def test_new_item_may_depend_on_the_in_flight_item():
    """Planning against the outcome of the currently running attempt is a
    natural judge move and must be legal."""
    from autotuner.judge.schema import InsertItem

    q = Queue()
    q.seed([_plain_item("h1")])
    q.pop_ready()
    q.apply_mutations([InsertItem(item=_plain_item("h2", depends_on="h1",
                                                   condition="correct"))])
    assert q.ids() == ("h2",)


def test_parser_exhaustion_is_rejected_not_fatal():
    """Schema-valid content that exhausts a parser's stack must become a
    normal rejection, never an uncaught RecursionError."""
    from autotuner.judge.client import JsonJudge
    from autotuner.judge.schema import JudgeBabble
    from autotuner_runtime.grammar import Expr, GrammarError

    with pytest.raises(GrammarError):
        Expr("1" + "+1" * 40000)

    class Canned(JsonJudge):
        def _ask(self, system, messages):
            return "[" * 100000 + "]" * 100000

    with pytest.raises(JudgeBabble):
        Canned().next({}, None)


def test_delete_that_strands_dependents_is_rejected():
    """Deleting an item that queued items depend on would leave them
    unsatisfiable forever with a false close reason."""
    from autotuner.judge.schema import DeleteItem

    q = Queue()
    q.seed([_plain_item("a"), _plain_item("b", depends_on="a", condition="correct")])
    with pytest.raises(QueueError, match="strand"):
        q.apply_mutations([DeleteItem(item_id="a")])
    assert q.ids() == ("a", "b")


def test_client_transcript_records_every_ask_and_reply(tmp_path):
    """What the judge was told and what it answered must survive the run, for
    every transport, including the re-ask after a rejected reply."""
    sdk = FakeSDK(["not json", json.dumps({"queue": [witem()]})])
    judge = AnthropicJudge(client=sdk)
    judge.transcript = tmp_path / "judge.jsonl"
    judge.seed({"stub": True})
    rows = [json.loads(l) for l in (tmp_path / "judge.jsonl").read_text().splitlines()]
    assert [(r["call"], r["attempt"]) for r in rows] == [("SeedResponse", 0), ("SeedResponse", 1)]
    assert rows[0]["reply"] == "not json"
    assert json.loads(rows[0]["messages"][0]["content"]) == {"region_state": {"stub": True}}
    assert "rejected" in rows[1]["messages"][-1]["content"]
    assert "Response schema (seed)" in rows[1]["system"]


def test_item_ids_must_be_identifiers_because_they_become_kernel_names():
    rejects({"queue": [witem("fuse-silu")]}, "letters, digits, and underscores")
    rejects({"queue": [witem("h 1")]}, "kernel names")
    assert validate_response({"queue": [witem("fuse_silu_2")]}).queue[0].id == "fuse_silu_2"


def test_scratch_buffers_are_validated():
    ok = validate_response({"mutations": [], "kernel": proposal(
        scratch=[["tmp0", "float32", ["in0.shape[0]", "8"]], ["tmp1", "float16", ["1"]]])}).kernel
    assert ok.scratch == (("tmp0", "float32", ("in0.shape[0]", "8")), ("tmp1", "float16", ("1",)))
    rejects({"mutations": [], "kernel": proposal(scratch=[["buf", "float32", ["1"]]])}, "tmp0, tmp1")
    rejects({"mutations": [], "kernel": proposal(scratch=[["tmp0", "float99", ["1"]]])}, "unknown dtype")
    rejects({"mutations": [], "kernel": proposal(scratch=[["tmp0", "float32", ["x"]]])}, "scratch tmp0")
    rejects({"mutations": [], "kernel": proposal(scratch=[["tmp0", "float32"]])}, "[name, dtype")


def test_kind_is_the_judges_own_label():
    """The seven menu kinds are suggestions; any short label passes, so the
    judge can name a move the menu never listed."""
    resp = validate_response({"queue": [
        witem("h1", kind="fold rope into the matmul epilogue"),
        witem("h2", kind="split-K/2"),
    ]})
    assert [it.kind for it in resp.queue] == ["fold rope into the matmul epilogue", "split-K/2"]


def test_worked_examples_are_valid_replies_and_reach_the_prompt():
    """Every example reply must pass the same validator the judge faces, the
    fused example must be the kernel the planted-win test ships, and each call
    kind's examples must sit in that call's system prompt."""
    from autotuner.judge.client import JsonJudge
    from autotuner.judge.examples import (FUSED_CHAIN_SOURCE, NEXT_EXAMPLES, SEED_EXAMPLES,
                                          render_examples)
    from autotuner.judge.schema import NextResponse, SeedResponse
    from tests.test_loop import FUSED_CHAIN_SOURCE as SHIPPED

    for ex in SEED_EXAMPLES:
        assert isinstance(validate_response(ex["reply"]), SeedResponse), ex["title"]
    for ex in NEXT_EXAMPLES:
        assert isinstance(validate_response(ex["reply"]), NextResponse), ex["title"]
    assert FUSED_CHAIN_SOURCE == SHIPPED
    assert "Worked examples" in render_examples("seed") and "h1_fix" in render_examples("next")

    seen = {}

    class Capturing(JsonJudge):
        def _ask(self, system, messages):
            seen[len(seen)] = system
            return json.dumps({"queue": [witem()]}) if "schema (seed)" in system \
                else json.dumps({"mutations": [], "kernel": None})

    j = Capturing()
    j.seed({"region": {}})
    j.next({"region": {}}, {"outcome": "failed"})
    assert "Example 1 (seed" in seen[0] and "Example 1 (next" not in seen[0]
    assert "Example 1 (next" in seen[1] and "insert" in seen[1]


def test_chip_facts_reach_the_judge():
    """The briefing carries the machine: cores, bandwidth, launch cost, and a
    legend line saying what one threadgroup gets of them."""
    from autotuner.judge.prompts import LEGEND, render_region_state

    state = fixed_state()
    assert isinstance(state, dict)
    rendered = render_region_state(**state, chip={"gpu_cores": 10, "bandwidth_gbps": 94.0,
                                                  "launch_us": 8.0, "flops_gflops": {"bfloat16": 2588.0}})
    assert rendered["chip"]["gpu_cores"] == 10 and rendered["chip"]["bandwidth_gbps"] == 94.0
    assert "one core" in LEGEND["chip"] and "chip" in rendered["legend"]
    assert any("chip.gpu_cores" in m for m in rendered["moves"])
