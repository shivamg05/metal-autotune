"""Spike 12: is a one-launch streaming kernel a fair floor for MLX's own ops?

Runs MLX's bf16 matvec over 2, 4 and 6 MB matrices beside the floor probe
(autotuner/measure/probe.py), each pair chained, cache-cold, and interleaved
in one window per the clock laws, and prints the library's time over the
probe's. Also prints the arithmetic roofline both ways (the larger of stream
and launch, and their sum) against the library's absolute time, and MLX's own
row reduction unchained, as a bandwidth reference. Findings are recorded in
PLATFORM.md under spike_12.

    uv run python spikes/spike_12_stream_probe.py
"""

import re
import subprocess

import mlx.core as mx

from autotuner.measure.clocks import (CLOCK_TARGET_MS, chained_loop, compare, link_input,
                                      link_loop, loop_iterations, timing_sets)
from autotuner.measure.peaks import measure_peaks
from autotuner.measure.probe import floor_from, stream_probe
from autotuner.measure.session import Session, time_once


def main() -> None:
    busy = re.search(r'"Device Utilization %"=(\d+)', subprocess.run(
        ["ioreg", "-r", "-c", "IOAccelerator", "-d", "4"], capture_output=True, text=True).stdout)
    print("GPU busy before the spike:", busy.group(1) + "%" if busy else "unknown")
    session = Session()
    peaks = measure_peaks(session, dtypes=(mx.bfloat16,))
    print(f"peaks: {peaks.bandwidth_gbps:.1f} GB/s, launch {peaks.launch_us:.2f} us")
    for name, n_out, k in [("2 MB", 1024, 1024), ("4 MB", 2048, 1024), ("6 MB", 3072, 1024),
                           ("4 MB again", 2048, 1024)]:
        x = mx.random.normal((1, 1, k)).astype(mx.bfloat16)
        w = mx.random.normal((n_out, k)).astype(mx.bfloat16)
        mx.eval(x, w)
        sets = timing_sets([{0: x, 1: w}])
        link_id = link_input(sets[0], {1})
        lib = lambda b: [b[0] @ b[1].T]
        iters = loop_iterations(time_once, lambda n: chained_loop(lib, sets, n, link_id), CLOCK_TARGET_MS)
        lib_loop = chained_loop(lib, sets, iters, link_id)
        probe = stream_probe([((1, 1, k), "bfloat16"), ((n_out, k), "bfloat16")], [((1, 1, n_out), "bfloat16")])
        probe_loop = chained_loop(lambda b: probe([b[0], b[1]]), sets, iters, link_id)
        net = compare(session, link_loop(sets, iters, link_id), lib_loop, pairs=8)
        library_ms = -net.median_delta_ms / iters
        floor = compare(session, probe_loop, lib_loop, pairs=8)
        floor_ms = floor_from(floor, net.median_baseline_ms, iters, library_ms)
        stream_us = w.nbytes / peaks.bandwidth_gbps / 1e3
        red = chained_loop(lambda b: [mx.sum(b[1], axis=1)], sets, iters, None)
        t_red = min(time_once(red) for _ in range(3)) / iters
        print(f"{name:11} library {library_ms * 1e3:6.1f} us | probe {floor_ms * 1e3:6.1f} us "
              f"| library over probe {library_ms / floor_ms:.2f} (stability {floor.stability:.2f}) "
              f"| arithmetic: max(stream, launch) {max(stream_us, peaks.launch_us):6.1f} us, "
              f"stream + launch {stream_us + peaks.launch_us:6.1f} us "
              f"| mx.sum(w, axis=1) unchained {t_red * 1e6:6.1f} us")


if __name__ == "__main__":
    main()
