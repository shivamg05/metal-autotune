"""Runtime reductions must preserve local correctness and measurement coverage."""
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from autotuner.bind.certify import certify_identities
from autotuner.measure.clocks import compare
from autotuner.measure.session import Session
from autotuner.loop import JobRunner, RegionRun, PromotionResult
from autotuner.judge.scripted import ScriptedJudge
from autotuner.ladder.gates import LadderResult
from autotuner_runtime.kernels import KernelSpec
from autotuner_runtime.swap import ReplayWrapper
from tests.test_search_feedback import region


class Identity(nn.Module):
    def __call__(self, x):
        return x


class Chain(nn.Module):
    def __init__(self, n=64):
        super().__init__()
        self.layers = [Identity() for _ in range(n)]
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        for layer in self.layers:
            x = layer(x)
        return x


class Shift(ReplayWrapper):
    def __init__(self, original, amount=0):
        super().__init__(original)
        self.amount = amount

    def __call__(self, x):
        return self.wrapped(x) + self.amount


def test_shared_identity_check_uses_six_passes_for_64_locations():
    model = Chain()
    originals = list(model.layers)
    wrappers = {f'layers.{i}': Shift(layer) for i, layer in enumerate(originals)}
    result = certify_identities(model, wrappers, [lambda: model(mx.ones((8,)))])
    assert result.ok, result.reason
    assert model.calls == 6
    assert all(a is b for a, b in zip(model.layers, originals))


def test_cancelling_wrapper_errors_do_not_pass():
    model = Chain(2)
    originals = list(model.layers)
    wrappers = {'layers.0': Shift(originals[0], 1), 'layers.1': Shift(originals[1], -1)}
    result = certify_identities(model, wrappers, [lambda: model(mx.ones((8,)))])
    assert not result.ok and 'layers.0' in result.reason
    assert all(a is b for a, b in zip(model.layers, originals))


def test_identity_observers_unwind_on_exception():
    model = Chain(2)
    original = model.layers[0]
    def broken():
        model(mx.ones((8,)))
        raise RuntimeError('fixture failure')
    with pytest.raises(RuntimeError, match='fixture failure'):
        certify_identities(model, {'layers.0': Shift(original)}, [broken])
    assert model.layers[0] is original


def test_identity_checks_nested_targets_separately():
    model = Chain(1)
    model.layers[0] = Chain(1)
    parent = model.layers[0]
    child = parent.layers[0]
    result = certify_identities(model, {'layers.0': Shift(parent), 'layers.0.layers.0': Shift(child)},
                               [lambda: model(mx.ones((8,)))])
    assert result.ok, result.reason
    assert model.calls == 12
    assert model.layers[0] is parent and parent.layers[0] is child


@pytest.mark.parametrize('change', ['scalar', 'structure', 'model_state'])
def test_identity_checks_scalar_state_and_output_structure(change):
    class WithMetadata(nn.Module):
        def __call__(self, x):
            return x, 0

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = WithMetadata()
            self.offset = 0

        def __call__(self, x):
            self.offset = 0  # rewind, as a managed correctness call does
            out = self.layer(x)
            return {'output': out[0], 'offset': self.offset}

    model = Model()
    original = model.layer

    class Changed(ReplayWrapper):
        def __call__(self, x):
            out = self.wrapped(x)
            if change == 'scalar':
                return out[0], 1
            if change == 'structure':
                return list(out)
            model.offset = 1
            return out

    result = certify_identities(model, {'layer': Changed(original)}, [lambda: model(mx.ones((8,)))])
    assert not result.ok, 'identical arrays must not hide changed metadata or state'
    assert model.layer is original


@pytest.mark.parametrize('ramped', [False, True])
def test_comparison_reuses_ramp_but_keeps_all_40_samples(ramped):
    warmed, timed = [], []
    a, b = object(), object()
    session = SimpleNamespace(fresh_chunk=lambda fn: ramped,
        warm_until_stable=lambda fn: warmed.append(fn),
        timed=lambda fn: timed.append(fn) or .001,
        settle=lambda: None, log=lambda *a, **kw: None)
    result = compare(session, a, b, pairs=20)
    assert warmed == ([b] if ramped else [a, b])
    assert timed.count(a) == timed.count(b) == 20
    assert result.n == 10


def test_first_proposal_seeds_queue_and_search_keeps_history(tmp_path):
    runner = object.__new__(JobRunner)
    runner.manifest = SimpleNamespace(budget_per_region=2, budget_total=2)
    runner.total_hypotheses = 0
    runner._finish_requested = lambda: False
    runner._meta = lambda run, queue, item: {}
    runner._ask_judge = lambda run, phase, call: (call(), None)
    runner._note_lesson = runner._record_attempt = lambda *a, **kw: None
    runner._bind_and_promote = lambda *a, **kw: PromotionResult('not_tested')
    runner._evaluate_kernel = lambda *a: LadderResult('correct_slower', None, {}, 1., 2., 1., .01)
    runner.kernel_dir = tmp_path
    seed = KernelSpec('seed', 'seed', ('in0',), ('out0',), 'out0[0]=in0[0];')
    runner._kernel_from_proposal = lambda run, region, proposal, name: replace(seed, kernel_id=name)
    run = RegionRun(region('r', 0, 2), scaffold=seed, head=seed, kernels={'seed': seed})
    item = lambda i: {'id': i, 'kind': 'retile', 'assoc_tag': 'preserving', 'hypothesis': 'try layout'}
    kernel = {'source': seed.source, 'parent_kernel_id': 'head', 'grid': ['1']*3,
              'threadgroup': ['1']*3, 'output_shapes': [['1']]}
    judge = ScriptedJudge([
        {'mutations': [{'op': 'insert', 'item': item(i)} for i in ('h1', 'h2')], 'kernel': kernel},
        {'mutations': [], 'kernel': kernel}])
    judge.combined_start = True
    runner.hypothesis_cycle(run, judge)
    assert runner.total_hypotheses == 2
    assert [entry[0] for entry in judge.seen] == ['next', 'next']
    assert judge.seen[1][2]['hypothesis_id'] == 'h1'


@pytest.mark.parametrize('thinking,remaining', [(0, 30), (12, 18), (40, 0)])
def test_cpu_work_counts_toward_cooling(thinking, remaining):
    now, slept = [100.], []
    session = Session(now=lambda: now[0], sleep=slept.append)
    session._debt_s = 10
    session.defer_settle()
    assert slept == []  # result available immediately
    now[0] += thinking
    assert session.wait_ready()
    assert slept == ([remaining] if remaining else [])
    assert not session.wait_ready()  # never pay twice


@pytest.mark.parametrize('entry', ['timed', 'off_clock'])
def test_next_gpu_call_waits_for_deferred_cooling(entry):
    order = []
    session = Session(now=lambda: 100., sleep=lambda seconds: order.append('idle'))
    session._debt_s = 1
    session.defer_settle()
    getattr(session, entry)(lambda: order.append('gpu') or mx.ones((1,)))
    assert order[:2] == ['idle', 'gpu']


def test_interrupted_cooling_keeps_remaining_deadline():
    now = [100.]
    def interrupted(seconds):
        now[0] += 4
        raise InterruptedError('interrupted sleep')
    session = Session(now=lambda: now[0], sleep=interrupted)
    session._debt_s = 10
    session.defer_settle()
    with pytest.raises(InterruptedError):
        session.wait_ready()
    slept = []
    session._sleep = slept.append
    assert session.wait_ready()
    assert slept == [26.]
    assert session.idled_s == 30.
    assert not session.wait_ready()


def test_confirmation_reuses_prepared_functions(monkeypatch):
    from autotuner import loop
    runner = object.__new__(JobRunner)
    runner.session = object()
    runner.log = SimpleNamespace(append=lambda *a, **kw: None)
    runner._model_win = lambda e: True
    runner._timed_arms = lambda *a: pytest.fail('must not compile again')
    a, b = object(), object()
    seen = []
    # asdict needs a dataclass comparison in the event log.
    from autotuner_runtime.stats import comparison_from_samples
    monkeypatch.setattr(loop, 'compare', lambda session, first, second, **kw:
                        seen.append((first, second, kw['pairs'])) or comparison_from_samples([2.,2.], [1.,1.]))
    runner._confirm_model_win(SimpleNamespace(checks=[]), object(), timed={'w': (a, b)})
    assert seen == [(a, b, 20)]


@pytest.mark.parametrize('incumbent_ms,screened', [(0.5, True), (1., False), (1.5, False)])
def test_native_incumbent_is_measured_in_same_group(tmp_path, monkeypatch, incumbent_ms, screened):
    from tests.test_native_search import runner_for, candidate
    from autotuner.ladder import child
    from autotuner.ladder.gates import _spec
    runner, region = runner_for(tmp_path)
    try:
        seed, proposed = candidate(runner, region)
        job = runner._ladder_job(region, proposed, 'preserving', True)
        job.timing_incumbent = seed
        seen = []
        def samples(session, arms, pairs):
            seen.extend(arms)
            for fn in arms.values():
                mx.eval(fn())  # all real launch paths, including the native incumbent
            ms = dict(library=2., candidate=1., incumbent=incumbent_ms, probe=.1, link=0.)
            return {key: [ms[key]] * pairs for key in arms}
        monkeypatch.setattr(child, 'sample_group', samples)
        monkeypatch.setattr(child, 'loop_iterations', lambda *a: 1)
        monkeypatch.setattr(child, 'Session', lambda: Session(sleep=lambda s: None))
        result = child.evaluate_ladder(_spec(job, 'score'))
        assert result.passed, result
        assert 'incumbent' in seen
        assert result.detail['incumbent_screen']['resolved_regression'] is screened
        assert result.detail['ship'] is not screened
    finally:
        runner.tracer.uninstall()
