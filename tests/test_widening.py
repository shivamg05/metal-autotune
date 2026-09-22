"""The widening round and the launch-floor stop (spec "Widening round" and
"Closing a region"). The harness refuses what it can check: an edit of a
non-scaffold parent or a repeated kind while openers remain, never a repair
of a failed kernel; and a launch-bound region whose head sits at the launch
floor for two attempts closes with that reason."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from autotuner.judge import prompts
from autotuner.judge.directions import DIRECTIONS, OPENERS
from autotuner.judge.scripted import ScriptedJudge
from autotuner.ladder.gates import LadderResult
from autotuner.loop import JobRunner, PromotionResult, RegionRun
from autotuner.regions.types import Roofline
from autotuner_runtime.kernels import KernelSpec
from tests.test_judge import fixed_state
from tests.test_search_feedback import region

SEED = KernelSpec("seed", "seed", ("in0",), ("out0",), "out0[0]=in0[0];")


def proposal(parent, item_id, mutations=()):
    return {"mutations": list(mutations), "kernel": {
        "source": SEED.source, "parent_kernel_id": parent, "item_id": item_id,
        "grid": ["1"] * 3, "threadgroup": ["1"] * 3, "output_shapes": [["1"]]}}


def item(i, kind, **rest):
    return {"id": i, "kind": kind, "assoc_tag": "preserving",
            "hypothesis": f"{kind}: one thread per row", **rest}


def bare(tmp_path, budget=8, failing=()):
    runner = object.__new__(JobRunner)
    runner.manifest = SimpleNamespace(budget_per_region=budget, budget_total=budget, seed=0)
    runner.total_hypotheses = 0
    runner.refusals = []
    runner._finish_requested = lambda: False
    runner._meta = lambda run, queue, item: {}
    runner._ask_judge = lambda run, phase, call: (call(), None)
    runner._note_lesson = lambda *a, **kw: None
    runner._bind_and_promote = lambda *a, **kw: PromotionResult("not_tested")
    runner.log = SimpleNamespace(append=lambda kind, **row: (
        runner.refusals.append(row["reason"]) if kind == "plan_refused" else None))
    runner._evaluate_kernel = lambda region, kernel, tag, **kw: (
        LadderResult("failed", "compile", {}, None, None, None, None) if kernel.kernel_id in failing
        else LadderResult("correct_slower", None, {}, 1., 2., 1., .01))
    runner.kernel_dir = tmp_path
    runner._kernel_from_proposal = lambda run, region, proposal, name: replace(SEED, kernel_id=name)
    runner._static_refusal = lambda *a: None  # this bare runner has no traces to build a contract from

    def record(run, hyp_id, kind, text, tag, kernel, parent, result, outcome=None, **rest):
        if kernel is not None:  # the verdict the repair exception reads
            run.attempts[kernel.kernel_id] = {"verdict": outcome or result.outcome, "parent": parent}
    runner._record_attempt = record
    return runner


def opening():
    return RegionRun(region("r", 0, 2), scaffold=SEED, head=SEED, kernels={"seed": SEED})


def test_the_briefing_offers_examples_only_on_request():
    import json
    from autotuner.judge.source_context import SourceContext
    widening = {"openers": 4, "opened": ["custom-design"], "left": 3}
    rendered = prompts.render_region_state(**fixed_state(), widening=widening)
    context = SourceContext()
    brief = context.focus(rendered)
    assert brief["widening"] == widening
    assert not {"directions", "menu", "moves"}.intersection(brief)
    reference = context.sources[brief["techniques_reference"]["source_id"]]
    assert len(json.loads(reference)["examples"]) == len(DIRECTIONS)


def test_openers_are_written_against_the_scaffold_under_new_kinds(tmp_path):
    runner = bare(tmp_path)
    run = opening()
    assert runner._openers(run) == OPENERS == 4
    judge = ScriptedJudge([
        {"queue": [item("h1", "one-dispatch"), item("h2", "simdgroup-per-row"),
                   item("h3", "one-dispatch"), item("h4", "wide-loads"),
                   item("h5", "persistent"), item("h6", "retile")]},
        proposal("scaffold", "h1"),
        proposal("head", "h2"),       # head moved to h1: refused, asked again for free
        proposal("h1", "h2"),         # refused again, charged
        proposal("scaffold", "h2"),
        proposal("scaffold", "h3"),   # a kind the round already opened
        proposal("scaffold", "h4"),   # third opener
        proposal("scaffold", "h5"),   # fourth opener closes the round
        proposal("head", "h6"),       # an edit of head is legal again
    ])
    runner.hypothesis_cycle(run, judge)
    assert run.openers == ["one-dispatch", "simdgroup-per-row", "wide-loads", "persistent"]
    assert [r.rsplit(";", 1)[0] for r in runner.refusals[:3]] == [
        "the widening round has 3 opener(s) left, so 'head' is refused as a parent: write against "
        "the scaffold under a kind no earlier opener used, or repair a kernel that failed",
        "the widening round has 3 opener(s) left, so 'h1' is refused as a parent: write against "
        "the scaffold under a kind no earlier opener used, or repair a kernel that failed",
        "the widening round already opened 'one-dispatch'; an opener needs a kind no earlier "
        "opener used (opened: one-dispatch, simdgroup-per-row)",
    ]
    assert run.attempts["h5"]["parent"] == "seed"
    assert run.attempts["h6"]["parent"] == "h1"
    assert run.close_rule == "the region's hypothesis budget is spent"


def test_a_repair_of_a_failed_opener_is_allowed_in_the_round(tmp_path):
    runner = bare(tmp_path, failing={"h1"})
    run = opening()
    judge = ScriptedJudge([
        {"queue": [item("h1", "one-dispatch"), item("h2", "wide-loads")]},
        proposal("scaffold", "h1"),
        proposal("h1", "h1_fix", mutations=[{"op": "insert", "before": "h2", "item": item(
            "h1_fix", "fix", depends_on="h1", condition="failed")}]),
        proposal("h1_fix", "h2"),     # h1_fix passed, so this is an edit, not a repair
    ])
    runner.hypothesis_cycle(run, judge)
    assert run.attempts["h1"]["verdict"] == "failed"
    assert run.attempts["h1_fix"] == {"verdict": "correct_slower", "parent": "h1"}
    assert run.openers == ["one-dispatch"]
    assert runner.refusals[0].startswith("the widening round has 3 opener(s) left, so 'h1_fix' is refused")


def test_four_designs_do_not_depend_on_a_catalogue(tmp_path):
    runner = bare(tmp_path, budget=2)
    run = RegionRun(region("r", 0, 2), scaffold=SEED, head=SEED, kernels={"seed": SEED})
    judge = ScriptedJudge([{"queue": [item("h1", "retile"), item("h2", "new-algorithm")]},
                           proposal("scaffold", "h1"), proposal("scaffold", "h2")])
    runner.hypothesis_cycle(run, judge)
    assert not runner.refusals and run.attempts["h2"]["parent"] == "seed"
    assert runner._openers(run) == 4


def launch_bound(age=2, streak=3, bound="launch"):
    run = RegionRun(region("r", 0, 0), head_age=age, floor_streak=streak, openers=["a", "b", "c", "d"])
    run.region.roofline = Roofline(t_mem_ms=0.001, t_compute_ms=0.0005, t_launch_ms=0.0068,
                                   t_roofline_ms=0.0068, bound=bound, s_max=2.0)
    return run


def test_a_launch_bound_region_at_the_floor_closes_after_two_still_attempts(tmp_path):
    runner = bare(tmp_path)
    assert runner._close_rule(launch_bound()) == (
        "no discernible headroom: the region is launch-bound, the last two attempts did not "
        "move head, and the last three kernels each sat within one sigma of the launch floor "
        "clocked beside them")
    assert runner._close_rule(launch_bound(age=5, streak=7)) is not None
    assert runner._close_rule(launch_bound(age=1)) is None       # head just moved
    assert runner._close_rule(launch_bound(streak=2)) is None    # one window said above the floor
    assert runner._close_rule(launch_bound(bound="memory")) is None
    run = launch_bound()
    run.region.roofline = None
    assert runner._close_rule(run) is None


def test_head_age_and_the_floor_streak_follow_the_verdicts(tmp_path):
    runner = bare(tmp_path)
    run = opening()
    verdict = SimpleNamespace(assoc_tag="preserving")

    def clock(kernel_ms, floor_ms, sigma_ms=.01):
        return LadderResult("correct_slower", None, {}, kernel_ms, 2., 2. - kernel_ms, sigma_ms,
                            floor_ms=floor_ms)
    runner._apply_verdict(run, verdict, replace(SEED, kernel_id="k1"),
                          LadderResult("failed", "compile", {}, None, None, None, None))
    assert (run.head_age, run.floor_streak) == (1, 0)      # a failed kernel measured nothing
    runner._apply_verdict(run, verdict, replace(SEED, kernel_id="k2"), clock(1.0, 0.995))
    assert run.head.kernel_id == "k2" and (run.head_age, run.floor_streak) == (0, 1)
    assert (run.head_sigma_ms, run.head_floor_ms) == (.01, .995)
    runner._apply_verdict(run, verdict, replace(SEED, kernel_id="k3"), clock(1.5, 1.495))
    assert run.head.kernel_id == "k2" and (run.head_age, run.floor_streak) == (1, 2)
    runner._apply_verdict(run, verdict, replace(SEED, kernel_id="k4"), clock(1.5, 1.4))
    assert (run.head_age, run.floor_streak) == (2, 0)      # 0.1 above the floor at sigma 0.01
    runner._apply_verdict(run, verdict, replace(SEED, kernel_id="k5"), clock(1.5, 1.6))
    assert (run.head_age, run.floor_streak) == (3, 1)      # below the floor counts as at it


def test_floor_stop_waits_for_all_four_opening_designs(tmp_path):
    runner = bare(tmp_path)
    run = launch_bound()
    run.openers = ['a', 'b', 'c']
    assert runner._close_rule(run) is None
    run.openers.append('d')
    assert runner._close_rule(run).startswith('no discernible headroom')


def test_after_opening_the_agent_can_revisit_or_start_fresh(tmp_path):
    runner = bare(tmp_path, budget=7, failing={'h3'})
    run = opening()
    judge = ScriptedJudge([
        {'queue': [item(f'h{i}', f'design{i}') for i in range(1, 8)]},
        *[proposal('scaffold', f'h{i}') for i in range(1, 5)],
        proposal('h2', 'h5'),       # a correct candidate that never became head
        proposal('h3', 'h6'),       # repair an older failure
        proposal('scaffold', 'h7'), # new design after the opening round
    ])
    runner.hypothesis_cycle(run, judge)
    assert not runner.refusals
    assert len(run.openers) == 4
    assert run.attempts['h5']['parent'] == 'h2'
    assert run.attempts['h6']['parent'] == 'h3'
    assert run.attempts['h7']['parent'] == 'seed'
    assert runner.total_hypotheses == run.hypotheses == 7


def test_budget_can_end_before_four_designs(tmp_path):
    runner = bare(tmp_path, budget=2)
    run = opening()
    judge = ScriptedJudge([
        {'queue': [item(f'h{i}', f'design{i}') for i in range(1, 5)]},
        proposal('scaffold', 'h1'), proposal('scaffold', 'h2'),
    ])
    runner.hypothesis_cycle(run, judge)
    assert len(run.openers) == runner.total_hypotheses == 2
    assert run.close_rule == "the region's hypothesis budget is spent"
