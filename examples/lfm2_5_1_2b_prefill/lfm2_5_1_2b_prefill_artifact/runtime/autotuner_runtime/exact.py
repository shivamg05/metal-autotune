"""Exact tensor comparison, including signed zeros and NaN payloads."""
import mlx.core as mx


def bitwise_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    # flatten handles scalars; contiguous permits byte views of strided tensors.
    return bool(mx.array_equal(mx.contiguous(a.reshape(-1)).view(mx.uint8),
                               mx.contiguous(b.reshape(-1)).view(mx.uint8)).item())
