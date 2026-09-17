"""Trusted worker signals around GPU work, outside the measurement clock."""

from contextlib import contextmanager
import os

GPU_WINDOW_S = 5.0
_fd: int | None = None
_depth = 0


def configure(fd: int) -> None:
    global _fd
    _fd = fd


@contextmanager
def gpu_window():
    """Nested evaluations share a deadline; none can keep extending it."""
    global _depth
    if _fd is None:
        yield
        return
    if _depth == 0:
        os.write(_fd, b"B")
    _depth += 1
    try:
        yield
    finally:
        _depth -= 1
        if _depth == 0:
            os.write(_fd, b"E")
