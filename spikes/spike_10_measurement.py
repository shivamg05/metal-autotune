"""Spike 10: measurement laws (plan section 6, M0 "Measurement").

Measures on the actual machine: the timing recipe, the A/A session floor cool
and hot, the GPU clock ramp after pacing idles, thermal drift and recovery,
duty-cycle pacing in a blocked design, warm-until-stable convergence on a
first-ever kernel, memory-pressure floor inflation, and the bandwidth/flops
peaks the roofline will use.

Design note proven while building this spike: after a duty-cycle idle the GPU
runs at ramped-down clocks, so the first sample of a paced chunk reads far
slow (the pacing-clock-ramp fact). All timed paths here therefore work in
chunks: a few unmeasured warm samples, then the timed samples back to back,
then idle 3x the chunk's work (law 2 at chunk granularity).

Modes: --quick (default, mechanical check of every phase at tiny durations)
and --full (the real numbers, roughly 9 minutes). FACT slugs are stable
across modes. Output contract: one line per fact,
`FACT <slug>: PASS|FAIL|INFO - <detail>`. Exit 0 if the script ran.
"""

import argparse
import os
import statistics
import sys
import time
import traceback

import mlx.core as mx

# durations and counts per mode; --full values are the real run, tune here
QUICK = {
    "load_s": 5.0,       # sustained-load duration before the hot A/A
    "idle_s": 2.0,       # idle before the recovery A/A
    "aa_pairs": 9,       # ABAB pairs per A/A run
    "duty_work_s": 4.0,  # GPU work per duty block
    "duty_cool_s": 2.0,  # idle after each duty block
    "peak_samples": 6,   # timed reps per peaks microbench
    "ballast_gib": 6.0,  # extra resident memory for the pressure probe
}
FULL = {
    "load_s": 120.0,
    "idle_s": 90.0,
    "aa_pairs": 30,
    "duty_work_s": 45.0,
    "duty_cool_s": 30.0,
    "peak_samples": 18,
    "ballast_gib": 6.0,
}
TARGET_SAMPLE_MS = 25.0  # chain workload is calibrated to about this much GPU work
PACE_RATIO = 3.0         # after a chunk of w seconds of work idle 3w (law 2)
CHUNK_WARM = 2           # unmeasured clock-ramp warm samples per chunk
PAIRS_PER_CHUNK = 3      # ABAB pairs measured per chunk
WARM_CAP = 30            # warm-until-stable iteration cap (law 3)


def fact(slug, status, detail):
    print(f"FACT {slug}: {status} - {detail}", flush=True)


def note(msg):
    print(f"[spike10] {msg}", file=sys.stderr, flush=True)


def sample(fn):
    """The timing recipe: synchronize, perf_counter, eval outputs, synchronize."""
    mx.synchronize()
    t0 = time.perf_counter()
    out = fn()
    mx.eval(out)
    mx.synchronize()
    return time.perf_counter() - t0


def chunk(fn, n_body, n_warm=CHUNK_WARM, paced=True):
    """One paced chunk: warm samples (discarded), timed body, then idle 3x work."""
    warm = [sample(fn) for _ in range(n_warm)]
    body = [sample(fn) for _ in range(n_body)]
    if paced:
        time.sleep(PACE_RATIO * sum(warm + body))
    return body


def median(xs):
    return statistics.median(xs)


def iqr(xs):
    qs = statistics.quantiles(xs, n=4, method="inclusive")
    return qs[2] - qs[0]


def make_chain(depth, n=1024, k=3, seed=0):
    """A matmul chain workload; inputs rotate through k sets to defeat caching."""
    mx.random.seed(seed)
    ws = [mx.random.uniform(-0.01, 0.01, (n, n)) for _ in range(depth)]
    xs = [mx.random.uniform(-0.5, 0.5, (n, n)) for _ in range(k)]
    mx.eval(ws, xs)
    counter = [0]

    def fn():
        x = xs[counter[0] % k]
        counter[0] += 1
        for w in ws:
            x = x @ w
        return x

    return fn


def calibrated_chain():
    """Pick a chain depth giving about TARGET_SAMPLE_MS of GPU work per sample.

    Two probe depths, differenced, so fixed per-sample overhead does not
    inflate the per-matmul estimate.
    """
    t8 = median(chunk(make_chain(8), 3, paced=False))
    t16 = median(chunk(make_chain(16), 3, paced=False))
    per = max((t16 - t8) / 8, 1e-5)
    overhead = max(t8 - 8 * per, 0.0)
    depth = max(4, min(64, round((TARGET_SAMPLE_MS / 1000.0 - overhead) / per)))
    fn = make_chain(depth)
    sample(fn)
    return fn, depth


def aa_run(fn, pairs):
    """A/A null: same callable both arms, ABAB pairs inside paced chunks.

    Returns (median sample seconds, median paired delta %, IQR of deltas %).
    """
    deltas, all_ts = [], []
    while len(deltas) < pairs:
        n = min(PAIRS_PER_CHUNK, pairs - len(deltas))
        body = chunk(fn, 2 * n)
        for i in range(0, 2 * n, 2):
            deltas.append(body[i] - body[i + 1])
        all_ts.extend(body)
    med = median(all_ts)
    rel = [100.0 * d / med for d in deltas]
    return med, median(rel), iqr(rel)


def sustained_load(fn, seconds):
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        mx.eval(fn())
    mx.synchronize()


def phase_recipe_and_aa(cfg):
    note("calibrating chain workload")
    fn, depth = calibrated_chain()
    ts = chunk(fn, 5)
    env = os.environ.get("MLX_MAX_OPS_PER_BUFFER", "unset")
    fact(
        "timing-recipe",
        "INFO",
        f"sync/perf_counter/eval/sync on a depth-{depth} 1024x1024 fp32 matmul chain: "
        f"median {median(ts) * 1e3:.2f} ms per sample over {len(ts)} samples "
        f"(MLX_MAX_OPS_PER_BUFFER={env})",
    )

    # law 7 evidence: building the graph without eval observes nothing
    mx.synchronize()
    t0 = time.perf_counter()
    out = fn()
    lazy_t = time.perf_counter() - t0
    mx.eval(out)
    evaled = median(ts)
    fact(
        "lazy-graph-build",
        "PASS" if lazy_t < 0.1 * evaled else "FAIL",
        f"graph construction without eval took {lazy_t * 1e3:.3f} ms vs "
        f"{evaled * 1e3:.2f} ms evaluated; a timed loop must eval its outputs",
    )

    # the clock ramp after a pacing idle: per-position medians across chunks
    note("pacing clock ramp")
    rows = [chunk(fn, 8, n_warm=0) for _ in range(6)]
    pos_med = [median([r[p] for r in rows]) for p in range(8)]
    steady = median(pos_med[2:])
    slow = [p for p in range(8) if pos_med[p] > 1.01 * steady]
    fact(
        "pacing-clock-ramp",
        "INFO",
        f"after a 3w pacing idle the first sample reads {pos_med[0] / steady:.2f}x "
        f"steady state (positions >1% slow: {slow if slow else 'none'}); paced "
        f"chunks must warm {CHUNK_WARM} samples before timing, refines law 2",
    )

    note("A/A null, cool")
    cool = aa_run(fn, cfg["aa_pairs"])
    fact(
        "aa-null-cool",
        "INFO",
        f"median paired delta {cool[1]:+.3f}% of median, IQR {cool[2]:.3f}% "
        f"({cfg['aa_pairs']} ABAB pairs in warmed chunks at "
        f"{cool[0] * 1e3:.2f} ms/sample); the IQR is the session floor",
    )
    return fn, cool


def phase_thermal(cfg, fn, cool, quick):
    weak = "quick mode, load too short for real thermal signal; " if quick else ""
    note(f"sustained load {cfg['load_s']}s")
    sustained_load(fn, cfg["load_s"])
    hot = aa_run(fn, cfg["aa_pairs"])
    fact(
        "thermal-hot-latency",
        "INFO",
        f"{weak}after {cfg['load_s']:.0f}s continuous work latency is "
        f"{100.0 * (hot[0] / cool[0] - 1):+.1f}% vs cool "
        f"({hot[0] * 1e3:.2f} ms vs {cool[0] * 1e3:.2f} ms)",
    )
    fact(
        "thermal-hot-floor",
        "INFO",
        f"{weak}A/A floor hot/cool = {hot[2] / max(cool[2], 1e-9):.2f}x "
        f"(IQR {hot[2]:.3f}% vs {cool[2]:.3f}%)",
    )
    note(f"idle {cfg['idle_s']}s for recovery")
    time.sleep(cfg["idle_s"])
    rec = aa_run(fn, cfg["aa_pairs"])
    fact(
        "thermal-recovery",
        "INFO",
        f"{weak}after {cfg['idle_s']:.0f}s idle latency is "
        f"{100.0 * (rec[0] / cool[0] - 1):+.1f}% vs cool, floor "
        f"{rec[2] / max(cool[2], 1e-9):.2f}x cool (IQR {rec[2]:.3f}%)",
    )


def duty_block(fn, n_chunks, paced):
    """One block of chunks; returns per-chunk medians of the body samples."""
    return [median(chunk(fn, 8, paced=paced)) for _ in range(n_chunks)]


def block_drift(meds):
    q = max(1, len(meds) // 4)
    return median(meds[-q:]) / median(meds[:q])


def phase_duty(cfg, fn, quick):
    weak = "quick mode, blocks too short to accumulate heat; " if quick else ""
    per = sample(fn)
    n_chunks = max(4, int(cfg["duty_work_s"] / max((CHUNK_WARM + 8) * per, 1e-3)))
    note(f"duty blocked design, {n_chunks} chunks per block")
    t0 = time.perf_counter()
    paced = duty_block(fn, n_chunks, paced=True)
    wall_p = time.perf_counter() - t0
    fact(
        "duty-paced-drift",
        "INFO",
        f"{weak}paced block (idle 3w after each chunk), {n_chunks} chunks over "
        f"{wall_p:.0f}s wall: last-quarter/first-quarter chunk median = "
        f"{block_drift(paced):.4f} at {median(paced) * 1e3:.2f} ms/sample",
    )
    time.sleep(cfg["duty_cool_s"])
    t0 = time.perf_counter()
    unpaced = duty_block(fn, n_chunks, paced=False)
    wall_u = time.perf_counter() - t0
    fact(
        "duty-unpaced-drift",
        "INFO",
        f"{weak}unpaced back-to-back block, same work over {wall_u:.0f}s wall: "
        f"last-quarter/first-quarter chunk median = {block_drift(unpaced):.4f} "
        f"at {median(unpaced) * 1e3:.2f} ms/sample",
    )
    time.sleep(cfg["duty_cool_s"])


def phase_warm_kernel():
    # unique name and source so this kernel has never been compiled anywhere
    token = f"{time.time_ns():x}"
    src = (
        f"uint tok = 0x{token[-6:]}u;\n"
        "uint elem = thread_position_in_grid.x;\n"
        "out[elem] = inp[elem] + T(tok - tok);\n"
    )
    kernel = mx.fast.metal_kernel(
        name=f"spike10_warm_{token}",
        input_names=["inp"],
        output_names=["out"],
        source=src,
    )
    mx.random.seed(1)
    a = mx.random.uniform(shape=(4 * 1024 * 1024,))
    mx.eval(a)

    def call():
        return kernel(
            inputs=[a],
            template=[("T", mx.float32)],
            grid=(a.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[a.shape],
            output_dtypes=[a.dtype],
        )[0]

    ts = [sample(call) for _ in range(WARM_CAP)]
    converged_at = None
    for i in range(1, len(ts)):
        if abs(ts[i] - ts[i - 1]) <= 0.01 * min(ts[i], ts[i - 1]):
            converged_at = i + 1
            break
    steady = median(ts[-5:])
    conv = (
        f"{converged_at} calls until two consecutive timings agree within 1%"
        if converged_at is not None
        else f"cap {WARM_CAP} hit with no two consecutive timings within 1% "
        f"(1% of {steady * 1e3:.2f} ms is under the dispatch jitter)"
    )
    fact(
        "warm-until-stable",
        "INFO",
        f"first-ever kernel: {conv}; first call {ts[0] * 1e3:.2f} ms is "
        f"{ts[0] / steady:.1f}x steady state {steady * 1e3:.3f} ms",
    )


def phase_memory(cfg, fn):
    note("memory-pressure probe")
    base = aa_run(fn, cfg["aa_pairs"])
    chunks = int(cfg["ballast_gib"] * 2)
    ballast = []
    for i in range(chunks):
        mx.random.seed(100 + i)
        b = mx.random.uniform(shape=(128 * 1024 * 1024,))  # 512 MiB fp32
        mx.eval(b)
        ballast.append(b)
    active_gib = mx.get_active_memory() / 2**30
    press = aa_run(fn, cfg["aa_pairs"])
    del ballast
    mx.clear_cache()
    fact(
        "mem-pressure-floor",
        "INFO",
        f"with {cfg['ballast_gib']:.0f} GiB extra resident ({active_gib:.1f} GiB "
        f"active) A/A floor = {press[2] / max(base[2], 1e-9):.2f}x baseline "
        f"(IQR {press[2]:.3f}% vs {base[2]:.3f}%), latency "
        f"{100.0 * (press[0] / base[0] - 1):+.1f}%",
    )


def peak_run(fn, work_per_rep, reps):
    """Running max over timed reps inside warmed paced chunks (law 5)."""
    best = 0.0
    while reps > 0:
        n = min(6, reps)
        for t in chunk(fn, n):
            best = max(best, work_per_rep / t)
        reps -= n
    return best


def phase_peaks(cfg):
    note("peaks microbench")
    # bandwidth: a single eager add kernel, reads 2 arrays and writes 1 (3N);
    # a 2x+y saxpy runs as two eager kernels (5N) and reads high, avoid it
    n = 64 * 1024 * 1024  # 256 MiB per fp32 array
    mx.random.seed(2)
    x = mx.random.uniform(shape=(n,))
    y = mx.random.uniform(shape=(n,))
    mx.eval(x, y)
    bw = peak_run(lambda: x + y, 3 * n * 4, cfg["peak_samples"])
    fact(
        "peak-bandwidth",
        "INFO",
        f"eager add on 256 MiB arrays (reads 2, writes 1): running max "
        f"{bw / 1e9:.1f} GB/s over {cfg['peak_samples']} reps",
    )
    del x, y
    mx.clear_cache()

    N = 3072  # within ~1% of the N=4096 rate, measured while building this spike
    flops = 2 * N**3
    for dtype, slug in ((mx.float16, "fp16"), (mx.bfloat16, "bf16"), (mx.float32, "fp32")):
        mx.random.seed(3)
        a = mx.random.uniform(-0.01, 0.01, (N, N)).astype(dtype)
        b = mx.random.uniform(-0.01, 0.01, (N, N)).astype(dtype)
        mx.eval(a, b)
        best = peak_run(lambda: a @ b, flops, cfg["peak_samples"])
        fact(
            f"peak-flops-{slug}",
            "INFO",
            f"{N}x{N} matmul: running max {best / 1e9:.0f} GFLOP/s "
            f"over {cfg['peak_samples']} reps",
        )
        del a, b
        mx.clear_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny durations (default)")
    parser.add_argument("--full", action="store_true", help="real durations, ~9 min")
    args = parser.parse_args()
    quick = not args.full
    cfg = QUICK if quick else FULL

    if not mx.metal.is_available():
        raise RuntimeError("Metal is not available; this spike cannot measure")

    fn, cool = phase_recipe_and_aa(cfg)
    phase_thermal(cfg, fn, cool, quick)
    phase_duty(cfg, fn, quick)
    phase_warm_kernel()
    phase_memory(cfg, fn)
    phase_peaks(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)
