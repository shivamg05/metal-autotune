"""Managed cache state is checked even when logits conceal its corruption."""
import mlx.core as mx
import pytest

from autotuner.artifact.validate import check_outputs
from autotuner_runtime.state import ContextStep, ContextSequence, correctness_call


@pytest.fixture(autouse=True)
def cpu_arrays():
    before = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(before)


class Cache:
    def __init__(self):
        self.offset = 0
        self.values = mx.array([0.0])

    @property
    def state(self):
        return self.values

    def trim(self, n):
        self.offset -= n


class Model:
    def __init__(self, error=0.0):
        self.error = error

    def __call__(self, x, *, cache):
        cache[0].offset += 1
        cache[0].values = cache[0].values + 1.0 + self.error
        return x


def step(error=0.0):
    return ContextStep(Model(error), [Cache()], 0)


def test_one_step_detects_hidden_state_corruption_without_changing_live_cache():
    original, bad = step(), step(0.125)
    inputs = [mx.array([1.0])]
    assert check_outputs(lambda: original(*inputs), lambda: bad(*inputs), 'outputs_only')['passed']
    # Ordinary calls restored all fields, so these same instances can be checked.
    result = check_outputs(lambda: correctness_call(original, inputs),
                           lambda: correctness_call(bad, inputs), 'state')
    assert not result['passed']
    assert original._cache[0].values.item() == bad._cache[0].values.item() == 0.0
    assert original._cache[0].offset == bad._cache[0].offset == 0


def test_sequence_state_uses_one_cumulative_allowance():
    original, bad = step(), step(0.006)
    inputs = [mx.array([1.0])]
    single = check_outputs(lambda: correctness_call(original, inputs),
                           lambda: correctness_call(bad, inputs), 'one',
                           exact=False, tolerances=(0, .01))
    assert single['passed']
    sequences = [ContextSequence(model, 3, include_state=True) for model in (original, bad)]
    result = check_outputs(lambda: sequences[0](*inputs), lambda: sequences[1](*inputs),
                           'three', exact=False, tolerances=(0, .01))
    assert not result['passed']


def test_timing_sequence_keeps_its_existing_output_contract():
    model = step()
    inputs = [mx.array([1.0])]
    assert len(model.sequence(inputs, 3)) == 3
    observation = model.sequence(inputs, 3, include_state=True)
    assert len(observation['outputs']) == 3
    assert observation['state'][0]['offset'] == 3
    assert observation['state'][0]['state'].item() == 3.0


def test_sequence_observation_keeps_earlier_outputs_and_mutable_containers():
    from autotuner_runtime.state import sequence_observation
    shared = {'value': mx.array([0.0]), 'index': 0}
    def model():
        shared['index'] += 1
        shared['value'] = mx.array([float(shared['index'])])
        return shared
    result = sequence_observation(model, [], 3)
    assert [item['index'] for item in result] == [1, 2, 3]
    assert [item['value'].item() for item in result] == [1, 2, 3]
