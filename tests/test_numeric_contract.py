"""Exact output contracts cannot be weakened by conversion to float32."""

import mlx.core as mx
import pytest

from autotuner.e2e import changing_check, preserving_check
from autotuner.ladder.numeric import compare, max_abs_diff


@pytest.fixture(autouse=True)
def cpu_arrays():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("dtype, value", [
    (mx.int32, 2**24),
    (mx.uint32, 2**24),
    (mx.int64, 2**40),
    (mx.uint64, 2**40),
    (mx.bool_, False),
])
def test_nonfloating_corruption_is_rejected_by_every_output_gate(dtype, value):
    reference = mx.array([value], dtype=dtype)
    corrupted = mx.array([value + 1], dtype=dtype)

    # Even a caller's loose float tolerance cannot legalize an index change.
    assert not compare(corrupted, reference, rtol=1.0, atol=100.0).passed
    assert max_abs_diff(reference, corrupted) == float("inf")
    assert not preserving_check(lambda: reference, lambda: corrupted, "w").passed
    changed = changing_check(lambda: reference, lambda: corrupted, "w", tolerances=(1.0, 100.0))
    assert not changed.passed


@pytest.mark.parametrize("dtype, value", [(mx.uint32, 2**24 + 1), (mx.bool_, True)])
def test_matching_nonfloating_outputs_pass(dtype, value):
    reference = mx.array([value], dtype=dtype)
    same = mx.array(reference)
    assert compare(same, reference, rtol=0.0, atol=0.0).passed
    assert max_abs_diff(reference, same) == 0.0
    assert preserving_check(lambda: reference, lambda: same, "w").passed
    assert changing_check(lambda: reference, lambda: same, "w").passed


def test_changing_gate_keeps_integer_outputs_exact_beside_noisy_float_outputs():
    reference = [mx.array([1.0]), mx.array([2**24], dtype=mx.uint32)]
    candidate = [mx.array([1.125]), mx.array([2**24 + 1], dtype=mx.uint32)]
    assert not changing_check(lambda: reference, lambda: candidate, "mixed",
                              tolerances=(0.2, 0.0)).passed


def test_changing_gate_uses_original_without_a_golden():
    reference = [mx.array([1.0]), mx.array([7], dtype=mx.uint32)]
    result = changing_check(lambda: reference, lambda: reference, "mixed")
    assert result.passed


@pytest.mark.parametrize("reference,candidate,exact,passed", [
    ([0.0], [-0.0], True, False),
    ([0.0], [-0.0], False, True),
    ([1.0], [1.001], True, False),
    ([1.0], [1.001], False, True),
    ([float("nan")], [float("nan")], False, True),
    ([float("inf")], [-float("inf")], False, False),
    ([float("inf")], [float("inf")], False, True),
    ([1.0], [float("nan")], False, False),
])
def test_shared_policy_exact_and_tolerance(reference, candidate, exact, passed):
    from autotuner_runtime.numeric import check
    result = check(mx.array(candidate), mx.array(reference), exact=exact,
                   rtol=0.01, atol=0.001)
    assert result.passed is passed


def test_tolerance_cannot_hide_extreme_subtraction_overflow():
    from autotuner_runtime.numeric import check
    reference = mx.array([3e38], dtype=mx.float32)
    candidate = -reference
    assert not check(candidate, reference, exact=False, rtol=1.5, atol=0.0).passed
    assert check(candidate, reference, exact=False, rtol=2.5, atol=0.0).passed


def test_output_defaults_follow_each_dtype():
    from autotuner_runtime.numeric import tolerance_for
    assert tolerance_for(mx.float16) == (1e-2, 2e-2)
    assert tolerance_for(mx.float32) == (1e-5, 1e-6)
    assert tolerance_for(mx.float16, (0.01, 0.0001)) == (0.01, 0.0001)


@pytest.mark.parametrize("values", [(float("nan"), 0), (-1, 0), (True, 0), (0, float("inf"))])
def test_shared_policy_rejects_invalid_tolerances(values):
    from autotuner_runtime.numeric import check
    with pytest.raises(ValueError, match="finite non-negative"):
        check(mx.array([1.0]), mx.array([1.0]), exact=False, rtol=values[0], atol=values[1])



def test_exact_mismatch_has_no_fictitious_numeric_distance():
    from autotuner_runtime.numeric import check
    result = check(mx.array([1.0]), mx.array([2.0]))
    assert not result.passed
    assert result.reason == "exact"
    assert result.max_excess is None



@pytest.mark.parametrize("candidate, reference, rtol, atol", [
    (1.0, 0.0, 1e300, 0.0),
    (3e38, -3e38, 0.0, 4e38),
])
def test_unrepresentable_tolerances_cannot_overflow_into_a_silent_pass(candidate, reference, rtol, atol):
    from autotuner_runtime.numeric import check
    with pytest.raises(ValueError, match="fit in float32"):
        check(mx.array([candidate]), mx.array([reference]), exact=False, rtol=rtol, atol=atol)


def test_nan_comparison_error_is_never_an_acceptance(monkeypatch):
    from autotuner_runtime import numeric
    # A nonfinite internal result must fail closed, even if a backend or future
    # comparator change creates one after valid inputs pass the pattern check.
    monkeypatch.setattr(numeric.mx, "max", lambda *_: mx.array(float("nan")))
    result = numeric.check(mx.array([1.0]), mx.array([0.0]), exact=False, rtol=0.0, atol=0.0)
    assert not result.passed
    assert result.reason == "nonfinite_error"


def test_largest_finite_tolerance_still_obeys_zero_reference_rule():
    from autotuner_runtime.numeric import MAX_TOLERANCE, check
    result = check(mx.array([1.0]), mx.array([0.0]), exact=False,
                   rtol=MAX_TOLERANCE, atol=0.0)
    assert not result.passed
