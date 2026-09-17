"""Report the actual failing element, not unrelated maximum scales."""
import mlx.core as mx
import pytest
from autotuner.artifact.validate import check_outputs, describe
from autotuner.e2e import changing_check


@pytest.fixture(autouse=True)
def cpu_arrays():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_failure_identifies_small_value_despite_large_global_allowance():
    original = lambda: {'primary': mx.array([3.0]), 'aux': mx.array([[1000., .001]])}
    candidate = lambda: {'primary': mx.array([3.0]), 'aux': mx.array([[1000., .003]])}
    result = check_outputs(original, candidate, 'w', exact=False, tolerances=(.01, .000001))
    assert not result['passed']
    failure = result['failure']
    assert failure['output'] == 1 and failure['index'] == [0, 1]
    assert failure['absolute_error'] == pytest.approx(.002)
    assert failure['allowance'] == pytest.approx(.000011)
    assert failure['comparison'] == 'patched_vs_original'
    assert 'output 1 at [0, 1]' in describe(result)
    job = changing_check(original, candidate, 'w', tolerances=(.01, .000001))
    assert job.failure == failure


def test_exact_signed_zero_failure_has_an_index_even_with_zero_numeric_error():
    result = check_outputs(lambda: mx.array([1., 0.]), lambda: mx.array([1., -0.]), 'zero')
    assert not result['passed']
    assert result['failure']['index'] == [1]
    assert result['failure']['absolute_error'] == 0
    assert result['failure']['reason'] == 'exact'


def test_nonfinite_diagnostic_is_json_safe():
    import json
    result = check_outputs(lambda: mx.array([1.]), lambda: mx.array([float('inf')]),
                           'inf', exact=False, tolerances=(0, 0))
    assert not result['passed']
    assert result['failure']['candidate'] == 'inf'
    json.dumps(result['failure'], allow_nan=False)
