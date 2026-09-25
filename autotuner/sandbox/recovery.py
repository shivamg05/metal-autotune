"""One bounded, known-good GPU check after a candidate worker has exited.

Killing a worker does not cancel a kernel it already put on the GPU: measured
on an M4, a killed worker's kernel kept a 2048-square matmul 2.4x slow until it
finished 8 s later. A correct answer only proves the GPU responds. So the check
also waits until that matmul is back near its quiet time, recorded once before
the first candidate ran, and fails if the GPU stays busy.
"""

import json
import os
import statistics
import sys
import time

from autotuner.sandbox.watchdog import configure, gpu_window

QUIET_RATIO = 1.5      # busy if the matmul reads more than this multiple of quiet
QUIET_WAIT_S = 120.0   # how long a leftover kernel may take to finish
_WINDOW_S = 1.0        # back-to-back samples per reading, so clocks stay up
_SIZE = 2048


def _known_good_sum(mx):
    values = mx.arange(1024, dtype=mx.int32)
    result = mx.sum((values + 7) * 3)
    mx.eval(result)
    mx.synchronize()
    if result.item() != 3 * (1024 * 1023 // 2 + 7 * 1024):
        raise RuntimeError("GPU recovery check returned an incorrect result")


def _matmul_ms(mx, a) -> float:
    """Median time of back-to-back matmuls over one window."""
    samples = []
    end = time.monotonic() + _WINDOW_S
    while time.monotonic() < end:
        with gpu_window():
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(a @ a)
            mx.synchronize()
            samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples)


def main():
    configure(int(os.environ["AUTOTUNER_WATCHDOG_FD"]))
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    with gpu_window():
        _known_good_sum(mx)
    a = mx.random.normal((_SIZE, _SIZE), key=mx.random.key(0))
    mx.eval(a)
    _matmul_ms(mx, a)  # warm the clocks; a reading after idle runs slow
    request = json.loads(sys.stdin.read() or "{}")
    quiet_ms = request.get("quiet_ms")
    if quiet_ms is None:
        print(f"GPU_QUIET_MS {_matmul_ms(mx, a):.4f}", flush=True)
        return
    limit = QUIET_RATIO * quiet_ms
    wait_s = request["wait_s"]
    deadline = time.monotonic() + wait_s
    while (ms := _matmul_ms(mx, a)) > limit:
        if time.monotonic() >= deadline:
            print(f"GPU still busy after {wait_s:.0f}s: matmul {ms:.2f} ms, "
                  f"quiet {quiet_ms:.2f} ms", file=sys.stderr, flush=True)
            sys.exit(1)
    print("GPU_CHECK_OK", flush=True)


if __name__ == "__main__":
    main()
