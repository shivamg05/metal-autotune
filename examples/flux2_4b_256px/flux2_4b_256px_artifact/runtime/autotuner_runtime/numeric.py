"""Portable output contract shared by search, model checks and exported bundles."""
from dataclasses import dataclass
import math
from types import MappingProxyType
import mlx.core as mx
from .exact import bitwise_equal

FLOAT_DTYPES = (mx.float16, mx.bfloat16, mx.float32)
MAX_TOLERANCE = float.fromhex("0x1.fffffep+127")  # largest finite comparison-dtype value
DEFAULT_TOLERANCES = MappingProxyType({
    "float32": (1e-5, 1e-6),
    "float16": (1e-2, 2e-2),
    "bfloat16": (2e-2, 4e-2),
})


def validate_tolerances(values):
    if len(values) != 2 or any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("tolerances must be two finite non-negative numbers (rtol, atol)")
    if any(v > MAX_TOLERANCE for v in values):
        raise ValueError("tolerances must fit in float32, the comparison dtype")


def tolerance_for(dtype, override=None):
    name = str(dtype).removeprefix("mlx.core.")
    if override is not None:
        validate_tolerances(override)
        return tuple(override)
    return DEFAULT_TOLERANCES.get(name, (0.0, 0.0))


@dataclass(frozen=True)
class CompareResult:
    passed: bool
    reason: str                    # "" | shape | dtype | nonfinite_pattern | tolerance
    max_excess: float | None       # scalar excess when meaningful; None for exact mismatches
    index: tuple[int, ...] | None  # the worst offender (or first pattern violation)
    detail: str = ""


def nonfinite_pattern_ok(candidate: mx.array, reference: mx.array) -> mx.array:
    """Elementwise: the candidate is finite wherever the reference is finite,
    NaN where NaN, and the same signed inf where inf."""
    finite_ok = mx.isfinite(reference) & mx.isfinite(candidate)
    nan_ok = mx.isnan(reference) & mx.isnan(candidate)
    inf_ok = mx.isinf(reference) & (candidate == reference)
    return finite_ok | nan_ok | inf_ok


def compare(
    candidate: mx.array,
    reference: mx.array,
    rtol: float,
    atol: float,
    wobble_floor: float = 0.0,
) -> CompareResult:
    """Floating-point tolerance compare: |c - r| <= max(atol + rtol * |r|, wobble_floor)
    elementwise, after the non-finite pattern rule. wobble_floor is the
    library's own run-to-run wobble, measured on the spot by the caller."""
    validate_tolerances((rtol, atol))
    if tuple(candidate.shape) != tuple(reference.shape):
        return CompareResult(False, "shape", float("inf"), None,
                             f"{tuple(candidate.shape)} != {tuple(reference.shape)}")
    if candidate.dtype != reference.dtype:
        return CompareResult(False, "dtype", float("inf"), None,
                             f"{candidate.dtype} != {reference.dtype}")
    if candidate.size == 0:
        return CompareResult(True, "", 0.0, None)

    # Indices, masks and packed words are exact values. Converting them to
    # float32 can erase a one-unit change above 2**24 before we compare it.
    if reference.dtype not in FLOAT_DTYPES:
        different = candidate != reference
        if mx.any(different).item():
            return CompareResult(False, "tolerance", float("inf"), _first_index(different),
                                 "non-floating outputs must match exactly")
        return CompareResult(True, "", 0.0, None)

    c = candidate.astype(mx.float32)
    r = reference.astype(mx.float32)
    ok = nonfinite_pattern_ok(c, r)
    if not mx.all(ok).item():
        viol = ~ok
        return CompareResult(
            False, "nonfinite_pattern", float("inf"), _first_index(viol),
            f"{int(mx.sum(viol).item())} elements break the non-finite pattern",
        )

    allowed = mx.maximum(atol + rtol * mx.abs(r), wobble_floor)
    # non-finite positions already matched the pattern exactly: zero excess
    excess = mx.where(mx.isfinite(r), mx.abs(c - r) - allowed, mx.zeros_like(r))
    # Opposite extreme float32 values can overflow subtraction. inf - inf
    # must never turn an invalid comparison into an accidental NaN pass.
    scale = mx.maximum(mx.maximum(mx.abs(c), mx.abs(r)), 1.0)
    scaled_excess = (mx.abs(c / scale - r / scale)
                     - (atol / scale + rtol * mx.abs(r / scale)))
    excess = mx.where(mx.isfinite(r) & mx.isnan(excess), scaled_excess * scale, excess)
    worst = float(mx.max(excess).item())
    if math.isnan(worst):
        return CompareResult(False, "nonfinite_error", None, None,
                             "comparison produced a NaN error")
    index = _unravel(int(mx.argmax(excess).item()), tuple(reference.shape))
    if worst > 0:
        over = int(mx.sum(excess > 0).item())
        return CompareResult(False, "tolerance", worst, index,
                             f"{over} elements over tolerance")
    return CompareResult(True, "", worst, index)



def check(candidate, reference, *, exact=True, rtol=0.0, atol=0.0):
    """Exact by default; tolerance is opt-in for floating evaluation changes."""
    if not exact:
        return compare(candidate, reference, rtol, atol)
    if tuple(candidate.shape) != tuple(reference.shape):
        return CompareResult(False, "shape", None, None, "output shape changed")
    if candidate.dtype != reference.dtype:
        return CompareResult(False, "dtype", None, None, "output dtype changed")
    if not bitwise_equal(candidate, reference):
        return CompareResult(False, "exact", None, None,
                             "output must match the original bit-for-bit")
    return CompareResult(True, "", 0.0, None)

def _first_index(viol: mx.array) -> tuple[int, ...]:
    """Index of the first True in a violation mask, deterministically: score
    each position by how early it is, then argmax."""
    n = viol.size
    score = viol.reshape(-1).astype(mx.int64) * mx.arange(n, 0, -1)
    return _unravel(int(mx.argmax(score).item()), tuple(viol.shape))


def _unravel(flat: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    idx: list[int] = []
    for dim in reversed(shape):
        flat, r = divmod(flat, dim)
        idx.append(r)
    return tuple(reversed(idx))
