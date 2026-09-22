"""The whole job end to end with the scripted judge.

The planted-win run must find the unfused chain, ship the scripted fused
kernel through bind and e2e, close by rule, and leave a working artifact. The
vendor-parity run must ship nothing and say so.
"""

import json
import textwrap
from pathlib import Path

import mlx.core as mx
import pytest

from autotuner.judge.scripted import ScriptedJudge
from autotuner.loop import JobRunner
from autotuner.measure.session import Session

# whole jobs and live models: minutes, not seconds
pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent / "fixtures"

# max/min guard NaN like mx.maximum/minimum do (NaN propagates); the regime
# gate fails a kernel that quietly maps NaN to 0 the way bare metal::max does
FUSED_CHAIN_SOURCE = """\
uint i = thread_position_in_grid.x;
uint c = i % (uint)in0_shape[1];
float x = in0[i];
float y = x * 2.0f + in1[c];
y = (y != y) ? y : metal::max(y, 0.0f);
y = y * x;
y = y + in2[c];
y = (y != y) ? y : metal::min(y, 8.0f);
out0[i] = (y - 1.0f) * 0.5f;
"""


def write_manifest(tmp_path, fixture, shape, extra=""):
    p = tmp_path / "manifest.yaml"
    p.write_text(textwrap.dedent(f"""
        model: {FIXTURES / fixture}
        workloads:
          - inputs: [{{shape: {list(shape)}, dtype: float32}}]
            name: main
        budget: {{per_region: 4, total: 8}}
        {extra}
    """))
    return p


def winning_chain_judge(region):
    if len(region.ops) != 8:
        return yielding_judge(region)  # only the full chain gets the real attempt
    return ScriptedJudge([
        {"queue": [{"id": "h1", "kind": "on-chip", "assoc_tag": "preserving",
                    "hypothesis": "keep the chain's intermediates in registers"}]},
        {"mutations": [], "kernel": {
            "source": FUSED_CHAIN_SOURCE,
            "parent_kernel_id": "scaffold",
            "grid": ["in0.shape[0] * in0.shape[1]", "1", "1"],
            "threadgroup": ["min(in0.shape[0] * in0.shape[1], 256)", "1", "1"],
            "output_shapes": [["in0.shape[0]", "in0.shape[1]"]],
        }},
        {"mutations": [], "kernel": None},
    ])


def yielding_judge(region):
    return ScriptedJudge([
        {"queue": [{"id": "h1", "kind": "retile", "assoc_tag": "preserving",
                    "hypothesis": "try a tile"}]},
        {"mutations": [], "kernel": None},
    ])


def test_selection_defers_overlaps_without_gpu_prescreen(tmp_path):
    manifest = write_manifest(tmp_path, "llama_ish.py", (1, 64))
    # This fixture takes token ids rather than floating activations.
    text = manifest.read_text().replace("float32", "int32")
    manifest.write_text(text)
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: None,
                       clock_pairs=4, session=Session(sleep=lambda s: None))
    runner.load_model()
    runner.trace_workloads()
    regions = runner.build_regions()
    kept = runner.capture_and_price(regions)
    selected = {r.fingerprint for r in regions if r.t_orig_ms}
    deferred = {r.fingerprint for r in runner.pending_regions}
    rejected = {r.fingerprint for r in regions if r.rejected}
    assert selected and deferred
    assert not (selected & deferred)
    assert selected | deferred | rejected == {r.fingerprint for r in regions}
    assert all(not r.rejected and not r.t_orig_ms for r in runner.pending_regions)
    assert not any(row["kind"] == "prescreen" for row in runner.log.rows())


@pytest.mark.parametrize('combined_start', [False, True])
def test_planted_win_job_ships(tmp_path, combined_start):
    from tests.conftest import require_healthy_gpu

    require_healthy_gpu()  # ships a real measured win; a mid-test throttle crossing can hide it
    # the row count is a named dim: the job traces and captures it at 7 rows
    # too, gate 7 checks every kernel there, and the final check runs there
    # a plain baseline: this test is about the mechanics of a ship, and under
    # the compiled baseline this chain is no win at all (see the next test)
    manifest = write_manifest(tmp_path, "planted_win.py", ("L", 1024),
                              "sweep: {L: [7, 4096]}\n        primary: {L: 4096}\n"
                              "        baseline: plain")
    next_payloads = []

    def factory(region):
        j = winning_chain_judge(region)
        if combined_start:
            from autotuner.judge.client import JsonJudge
            # Exercise the real JSON client and metadata validation without
            # paying for a model call. Merge this fixture's seed and first edit.
            script = list(j._script)
            first = dict(script[1])
            first['mutations'] = [{'op': 'insert', 'item': item} for item in script[0]['queue']]
            if first['kernel'] is not None:
                first['kernel'] = dict(first['kernel'], item_id=script[0]['queue'][0]['id'])
            replies = iter([first, *script[2:]])

            class CombinedJudge(JsonJudge):
                def _ask(self, system, messages):
                    assert 'There is no separate\nplanning turn' in system
                    return json.dumps(next(replies, {'mutations': [], 'kernel': None}))

            j = CombinedJudge()
        orig = j.next

        def recording_next(meta, verdict):
            next_payloads.append(meta)
            return orig(meta, verdict)

        j.next = recording_next
        return j

    runner = JobRunner(manifest, tmp_path / "work", judge_factory=factory, clock_pairs=8,
                       session=Session())
    report = runner.run()
    # shipped against eager, and still measured against mx.compile at the end
    assert report.step_ms["main"]["speedup_vs_compiled"] > 0
    assert [r for r in runner.log.rows() if r["kind"] == "baseline_comparison" and r["against"] == "compiled"]
    # the final run of consecutive steps is never shorter than one timed sample
    chain = report.step_ms["main"]["steps_per_sample"]
    assert report.final["sequences"]["main"]["steps"] == max(runner.manifest.final_benchmark.steps, chain)

    # the judge edits a named parent, so every next call must carry the
    # lineage sources and the head id (a live run starved this),
    # plus its only memory: the queue, the verdicts, and the executing item
    assert next_payloads
    for meta in next_payloads:
        assert meta["head"] in meta["kernels"]
        assert "source" in meta["kernels"][meta["head"]]
        assert "queue" in meta and "verdicts" in meta and "budget" in meta
        assert "history" in meta and "lessons" in meta and "regions_done" in meta
        assert meta["launch_grammar"] and meta["menu"] and meta["laws"] and meta["legend"]
        assert "directions" not in meta and meta["widening"]["openers"] == 4
    # writing_for names the front ready item and is null once nothing is
    # queued; the judge's yields are refused until the region's budget is spent
    assert [m["writing_for"]["id"] for m in next_payloads if m["writing_for"]] == ([] if combined_start else ["h1"])
    assert next_payloads[-1]["writing_for"] is None
    closes = [r.get("close_rule") or "" for r in report.regions if r.get("s")]
    assert closes and all("budget is spent" in c for c in closes), closes

    shipped = [r for r in report.regions if r.get("s")]
    assert shipped, f"nothing shipped; regions: {report.regions}"
    hyp = [h for h in report.hypotheses if h["verdict"] == "shipped"]
    assert hyp, report.hypotheses
    # the headline is the paired end-of-job comparison, never before minus after
    step = report.step_ms["main"]
    assert step["speedup"] > 1.0, f"the patched step did not beat the untouched one: {step}"

    # the artifact must reproduce the patched outputs in this process
    art = runner.emit_artifact(tmp_path / "artifact")
    assert (art / "kernels").exists() and (art / "patch" / "wrappers.py").exists()

    # run.jsonl must tell the whole story: every skeleton event of a shipping
    # run appears, and the report on disk is current (written per region close)
    assert all("t" in row and "wall" in row for row in runner.log.rows())
    kinds = [row["kind"] for row in runner.log.rows()]
    for expected in ("job", "model", "trace", "regions", "step_clock", "peaks",
                     "ranked", "region_open", "scaffold_ok", "judge", "verdict",
                     "shipped", "region_closed"):
        assert expected in kinds, f"missing {expected!r} in run log: {kinds}"
    assert kinds.index("region_open") < kinds.index("scaffold_ok") < kinds.index("shipped")
    judge_rows = [r for r in runner.log.rows() if r["kind"] == "judge"]
    assert {r["phase"] for r in judge_rows} == ({"next"} if combined_start else {"seed", "next"})
    assert all("latency_s" in r for r in judge_rows)
    assert (tmp_path / "work" / "report.json").exists()

    # the sweep: traced and captured at L=7, checked by gate 7 on every
    # attempt, and at that size the installed wrapper runs the original module
    assert "main@L=7" in [row["workload"] for row in runner.log.rows() if row["kind"] == "trace"]
    assert any(key[1] == "main@L=7" for key in runner.sweep_spans)
    assert [c["name"] for c in report.final["checks"]] == ["main", "main@L=7"]
    assert all(c["passed"] for c in report.final["checks"])
    x7 = runner.sweep_tensors["main@L=7"]
    got, want = runner.model(*x7), runner.baseline_model(*x7)
    assert mx.array_equal(got, want).item()
    runner.tracer.install()
    try:
        retrace7, _ = runner.tracer.trace(runner.model, x7)
    finally:
        runner.tracer.uninstall()
    assert not any(n.op == "custom_kernel" for n in retrace7.nodes)


def test_compiled_baseline_refuses_a_fusion_compile_already_does(tmp_path):
    """Under the default baseline the library arm is the region's ops as one
    compiled graph, and mx.compile already fuses an elementwise chain into one
    kernel, so the planted fusion is no win: nothing ships, the report names
    the choice, and both step clocks are recorded."""
    from tests.conftest import require_healthy_gpu

    manifest = write_manifest(tmp_path, "planted_win.py", (4096, 1024))
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=winning_chain_judge,
                       clock_pairs=8, session=Session(sleep=lambda s: None))
    report = runner.run()
    assert report.baseline["choice"] == "compiled"
    clocks = report.baseline["clocks_ms"]["main"]
    assert clocks["plain"] > 0 and clocks["compiled"] > 0
    assert report.step_ms["main"]["before"] == clocks["compiled"]
    rows = [r for r in runner.log.rows() if r["kind"] == "step_clock" and r["phase"] == "before"]
    assert rows and rows[0]["baseline"] == "compiled" and "plain_ms" in rows[0]
    # the final model against eager, measured paired at the end rather than derived
    step = report.step_ms["main"]
    assert step["speedup_vs_plain"] > 0 and isinstance(step["plain_win_confirmed"], bool)
    assert [r for r in runner.log.rows() if r["kind"] == "baseline_comparison" and r["against"] == "plain"]
    require_healthy_gpu()  # the verdicts below are measured; the fields above are not
    assert not [r for r in report.regions if r.get("s")], report.regions
    assert not runner.installed


def test_vendor_parity_job_ships_nothing(tmp_path):
    manifest = write_manifest(tmp_path, "vendor_parity.py", (256, 512))
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=yielding_judge, clock_pairs=8,
                       session=Session(sleep=lambda s: None))
    report = runner.run()
    shipped = [r for r in report.regions if r.get("s")]
    assert not shipped
    assert not runner.installed and not runner.emitted
    assert report.step_ms["main"]["before"] > 0 and report.step_ms["main"]["after"] > 0
    # the headline speedup is now paired against the untouched model measured in
    # the same window, and carries the agreement of the pairs behind it. The old
    # before/after subtraction could not be asserted on at all: it read whatever
    # the machine did between job start and job end.
    step = report.step_ms["main"]
    assert step["baseline_at_end"] > 0
    assert 0 < step["stability"] <= 1
    # that an unchanged model reports no speedup is pinned deterministically in
    # test_drift; here it is only claimable when the pairs actually agreed, and
    # this fixture's step is microseconds
    if step["stability"] > 0.8:
        assert step["speedup"] == pytest.approx(1.0, rel=0.15), step


def test_used_work_dir_refused(tmp_path):
    """A reused work dir would interleave two jobs' run.jsonl rows."""
    manifest = write_manifest(tmp_path, "planted_win.py", (64, 64))
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "run.jsonl").write_text('{"kind": "job"}\n')
    with pytest.raises(RuntimeError, match="previous run"):
        JobRunner(manifest, tmp_path / "work", judge_factory=yielding_judge)


def only_the_chain(monkeypatch):
    """Open only the planted chain: the other thirteen candidates each cost
    a starting kernel and two child processes and prove nothing here."""
    import autotuner.loop as loop_mod

    monkeypatch.setattr(loop_mod, "apply_floor",
                        lambda regions, **k: [r for r in regions if len(r.ops) == 8])


def test_persistent_judge_transport_error_stops_job(tmp_path, monkeypatch):
    """Repeated connection failures must not keep opening new regions."""
    class DeadTransport:
        def seed(self, meta):
            raise RuntimeError("claude CLI judge exited 3: no such model")

        def next(self, meta, verdict):
            raise RuntimeError("unreachable")

    only_the_chain(monkeypatch)
    manifest = write_manifest(tmp_path, "planted_win.py", (64, 1024))
    runner = JobRunner(manifest, tmp_path / "work",
                       judge_factory=lambda region: DeadTransport(),
                       session=Session(sleep=lambda s: None))
    with pytest.raises(RuntimeError, match="judge unavailable"):
        runner.run()
    assert runner.report.session["status"] == "failed"
    assert runner.total_hypotheses == 0
    assert any(row.get("action") == "error" for row in runner.log.rows()
               if row["kind"] == "judge")


def test_failed_first_item_lets_the_judge_insert_a_fix(tmp_path, monkeypatch):
    """The spec's failure branch: the judge hears a failed verdict before the
    next item is chosen, prepends a fix conditioned on that failure, and
    writes it in the same reply. The old cycle popped first and closed the
    region with the plan untouched."""
    only_the_chain(monkeypatch)
    broken = FUSED_CHAIN_SOURCE.replace("out0[i] =", "out0[i] = this_is_not_metal +")
    calls = []

    def judge_for(region):
        if len(region.ops) != 8:
            return yielding_judge(region)
        return ScriptedJudge([
            {"queue": [
                {"id": "h1", "kind": "on-chip", "assoc_tag": "preserving",
                 "hypothesis": "keep the chain's intermediates in registers"},
                {"id": "h2", "kind": "retile", "assoc_tag": "preserving",
                 "hypothesis": "then retile", "depends_on": "h1", "condition": "correct"},
            ]},
            {"mutations": [], "kernel": {
                "source": broken, "parent_kernel_id": "scaffold",
                "grid": ["in0.shape[0] * in0.shape[1]", "1", "1"],
                "threadgroup": ["min(in0.shape[0] * in0.shape[1], 256)", "1", "1"],
                "output_shapes": [["in0.shape[0]", "in0.shape[1]"]],
            }},
            {"mutations": [{"op": "insert", "before": "h2", "item": {
                "id": "hfix", "kind": "fix", "assoc_tag": "preserving",
                "hypothesis": "repair the compile error", "depends_on": "h1",
                "condition": "failed"}}],
             "kernel": {
                "source": FUSED_CHAIN_SOURCE, "parent_kernel_id": "h1", "item_id": "hfix",
                "grid": ["in0.shape[0] * in0.shape[1]", "1", "1"],
                "threadgroup": ["min(in0.shape[0] * in0.shape[1], 256)", "1", "1"],
                "output_shapes": [["in0.shape[0]", "in0.shape[1]"]],
            }},
            {"mutations": [], "kernel": None},
        ])

    def factory(region):
        j = judge_for(region)
        orig = j.next

        def recording_next(meta, verdict):
            calls.append((meta["writing_for"], verdict))
            return orig(meta, verdict)

        j.next = recording_next
        return j

    manifest = write_manifest(tmp_path, "planted_win.py", (64, 1024))
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=factory, clock_pairs=4,
                       session=Session(sleep=lambda s: None))
    report = runner.run()
    chain = next(r for r in report.regions if len(r["ops"]) == 8)
    rows = {h["id"]: h for h in report.hypotheses if h["region"] == chain["fingerprint"]}
    assert rows["h1"]["verdict"] == "failed" and rows["h1"]["failed_gate"] == "compile"
    assert rows["hfix"]["verdict"] in ("correct_slower", "shipped"), rows["hfix"]
    assert rows["hfix"]["parent"] == rows["h1"]["kernel"]
    # the second call carried h1's verdict and was asked to write for h2, which
    # was not ready; the judge's fix was written for hfix instead
    second_writing_for, second_verdict = calls[1]
    assert second_verdict["hypothesis_id"] == "h1" and second_verdict["failed_gate"] == "compile"
    assert second_writing_for is None
    # h2 still waits on h1 succeeding and the judge has nothing more; its
    # yields are refused until the region's budget is spent, which is the
    # only way a region closes
    assert "budget is spent" in chain["close_rule"], chain["close_rule"]
    refused = [r for r in runner.log.rows() if r["kind"] == "plan_refused"]
    assert refused and "yield" in refused[0]["reason"]


def test_a_refused_seed_is_asked_again_not_closed(tmp_path, monkeypatch):
    """The budget is the only close rule, at seed too: a seed queue the
    harness refuses (here an id reserved for the starting kernel) comes back
    to the judge as plan_refused, and its next reply may plan and write."""
    only_the_chain(monkeypatch)
    seen = []

    def factory(region):
        if len(region.ops) != 8:
            return yielding_judge(region)
        j = ScriptedJudge([
            {"mutations": [{"op": "insert", "item": {
                "id": "h1", "kind": "on-chip", "assoc_tag": "preserving",
                "hypothesis": "keep the chain's intermediates in registers"}}],
             "kernel": {
                "source": FUSED_CHAIN_SOURCE, "parent_kernel_id": "scaffold", "item_id": "h1",
                "grid": ["in0.shape[0] * in0.shape[1]", "1", "1"],
                "threadgroup": ["min(in0.shape[0] * in0.shape[1], 256)", "1", "1"],
                "output_shapes": [["in0.shape[0]", "in0.shape[1]"]],
            }},
        ])
        # JSON validation now rejects reserved ids before Queue sees them.
        # Bypass that earlier guard to exercise the queue-refusal recovery path.
        from autotuner.judge.schema import QueueItem, SeedResponse
        j.seed = lambda meta: SeedResponse((QueueItem(
            'scaffold', 'on-chip', 'preserving', 'a reserved id'),))
        orig = j.next

        def recording_next(meta, verdict):
            seen.append(verdict)
            return orig(meta, verdict)

        j.next = recording_next
        return j

    manifest = write_manifest(tmp_path, "planted_win.py", (64, 1024))
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=factory, clock_pairs=4,
                       session=Session(sleep=lambda s: None))
    report = runner.run()
    chain = next(r for r in report.regions if len(r["ops"]) == 8)
    assert "reserved" in seen[0]["plan_refused"]
    rows = {h["id"]: h for h in report.hypotheses if h["region"] == chain["fingerprint"]}
    assert rows["h1"]["verdict"] in ("correct_slower", "shipped"), rows
    assert "budget is spent" in chain["close_rule"]


def test_a_busy_or_throttled_machine_is_named_not_refused(tmp_path, monkeypatch):
    """A laptop's GPU is shared with whatever else is open, and every verdict
    is a paired comparison taken in one window, so the job goes on: the busy
    reading and an implausible peak each become an env_warning the operator
    can read, never a refusal. The busy reading is the one taken when the
    job object is made, before it has loaded a model: the counter trails, so
    a later reading would report the job's own loading as another process."""
    import autotuner.loop as loop_mod
    from autotuner.measure.peaks import Peaks

    manifest = write_manifest(tmp_path, "planted_win.py", (64, 1024))
    readings = iter([100.0, 0.0, 0.0])
    monkeypatch.setattr(loop_mod, "gpu_utilization", lambda: next(readings))
    monkeypatch.setattr(loop_mod, "measure_peaks",
                        lambda session: Peaks(bandwidth_gbps=9.4, flops_gflops={"float32": 180.0}))
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=yielding_judge,
                       session=Session(sleep=lambda s: None))
    runner.measure_machine()
    warnings = [r["detail"] for r in runner.log.rows() if r["kind"] == "env_warning"]
    assert any("utilization" in w for w in warnings) and any("bandwidth" in w for w in warnings), warnings
    assert "job_refused" not in [r["kind"] for r in runner.log.rows()]
    assert runner.report.session["gpu_utilization_at_start_pct"] == 100.0


def test_the_whole_model_decides_every_ship(tmp_path):
    """A region-clock nomination must also improve the whole forward pass."""
    from tests.conftest import require_healthy_gpu

    require_healthy_gpu()  # a real measured win; a mid-test throttle can hide it
    manifest = write_manifest(tmp_path, "planted_win.py", ("L", 1024),
                              "sweep: {L: [7, 4096]}\n        primary: {L: 4096}\n"
                              "        baseline: plain")
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=winning_chain_judge,
                       clock_pairs=8, session=Session(sleep=lambda s: None))
    report = runner.run()

    kinds = {json.loads(l)["kind"] for l in
             (tmp_path / "work" / "run.jsonl").read_text().splitlines()}
    assert kinds & {"shipped", "not_shipped"}, (
        "no candidate reached the whole-model decision")
    if [r for r in report.regions if r.get("s")]:
        assert runner.model_ratio < 1.0, (
            f"a kernel shipped without the whole model getting faster "
            f"(model_ratio {runner.model_ratio})")
    else:
        # nothing shipped is a real answer here: the whole model refused it
        assert runner.model_ratio == 1.0 and not runner.installed


def test_hand_written_norm_scaffold_is_checked_within_tolerance(tmp_path):
    """A naive lowering of a mean adds in its own order. Held to bitwise it
    failed the smoke gate on every such region, and the judge's one repair was
    held to the same rule; tagged changing, it passes and the search begins."""
    manifest = write_manifest(tmp_path, "hand_norm.py", (64, 1024))
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=yielding_judge, clock_pairs=8,
                       session=Session(sleep=lambda s: None))
    runner.run()
    rows = runner.log.rows()
    kinds = [r["kind"] for r in rows if r["kind"] in ("scaffold_ok", "scaffold_failed")]
    assert kinds and "scaffold_failed" not in kinds
    scaffolds = [r for r in rows if r["kind"] == "ladder_result" and r["kernel"].endswith("scaffold")]
    changing = [r for r in scaffolds if r["assoc_tag"] == "changing"]
    assert changing, scaffolds
    assert all(r["result"]["detail"].get("correctness_rule") == "tolerance" for r in changing), changing


def test_a_small_step_is_chained_to_fill_a_timed_sample(tmp_path):
    """A step of a few hundred microseconds called on its own leaves the GPU
    idle between samples and reads several times slow; the runner sizes a
    chain of dependent steps once per workload and every whole-model arm
    carries it."""
    from autotuner.measure.clocks import CLOCK_TARGET_MS
    from autotuner.measure.session import time_once
    manifest = write_manifest(tmp_path, "planted_win.py", (4, 1024))
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=yielding_judge,
                       session=Session(sleep=lambda s: None))
    runner.load_model()
    runner.trace_workloads()
    try:
        tensors = runner.tensors["main"]
        k = runner._chain(tensors)
        assert k > 1
        step = runner._step_fn(runner.baseline_model, tensors)
        assert step.steps == k
        sample_ms = min(time_once(step) for _ in range(5)) * k * 1e3
        assert CLOCK_TARGET_MS / 4 < sample_ms < CLOCK_TARGET_MS * 4
        assert runner._step_fn(runner.model, tensors, "compiled").steps == k
    finally:
        runner.tracer.uninstall()  # a job does this itself once tracing is done


def test_a_constant_built_in_forward_runs_as_a_job(tmp_path):
    """The recorded creation call has to survive everything after the trace:
    region building, the identity wrappers delivery certifies, the ladder's
    replays, and the final check."""
    manifest = write_manifest(tmp_path, "const_scale.py", (256, 64))
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=yielding_judge, clock_pairs=8,
                       session=Session(sleep=lambda s: None))
    report = runner.run()
    assert any(n.op == "mx.array" for n in runner.traces["main"].nodes)
    assert report.session["status"] == "search_complete" and report.final["passed"]
    kinds = [r["kind"] for r in runner.log.rows()]
    assert "job_failed" not in kinds and "region_open" in kinds


def test_a_statically_refused_kernel_is_asked_again_without_costing_an_attempt(tmp_path):
    """A kernel the static checks reject never reaches the GPU. The judge hears
    why under the refusal rule and its next kernel takes the attempt; 14 of 140
    MetalBench attempts were lost to a fallback predicate that was true on the
    workload being optimized."""
    manifest = write_manifest(tmp_path, "planted_win.py", ("L", 1024),
                              "sweep: {L: [7, 4096]}\n        primary: {L: 4096}\n"
                              "        baseline: plain")
    good = {"source": FUSED_CHAIN_SOURCE, "parent_kernel_id": "scaffold",
            "grid": ["in0.shape[0] * in0.shape[1]", "1", "1"],
            "threadgroup": ["min(in0.shape[0] * in0.shape[1], 256)", "1", "1"],
            "output_shapes": [["in0.shape[0]", "in0.shape[1]"]]}
    bad = dict(good, fallback_predicate="in0.shape[0] != 7")  # true at the primary size

    def factory(region):
        if len(region.ops) != 8:
            return yielding_judge(region)
        return ScriptedJudge([
            {"queue": [{"id": "h1", "kind": "on-chip", "assoc_tag": "preserving",
                        "hypothesis": "keep the chain's intermediates in registers"}]},
            {"mutations": [], "kernel": bad},
            {"mutations": [], "kernel": good},
            {"mutations": [], "kernel": None},
        ])

    runner = JobRunner(manifest, tmp_path / "work", judge_factory=factory, clock_pairs=8,
                       session=Session(sleep=lambda s: None))
    runner.run()
    rows = runner.log.rows()
    refused = [r for r in rows if r["kind"] == "plan_refused" and "fallback_on_primary" in r["reason"]]
    assert len(refused) == 1
    h1 = [r for r in rows if r["kind"] == "verdict" and r["hypothesis"] == "h1"]
    assert len(h1) == 1 and h1[0]["outcome"] != "failed", h1  # the good kernel took the attempt
    assert not [r for r in rows if r["kind"] == "verdict" and r.get("gate") == "static"]
