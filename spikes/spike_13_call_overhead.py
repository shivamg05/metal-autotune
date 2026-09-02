"""What a custom kernel pays at the call site that a library op does not.

Every kernel the judge writes reaches the GPU through autotuner_runtime.kernels.call:
Python that evaluates the launch grammar (grid, threadgroup, output shapes) and
builds the template list on every call, then mx.fast.metal_kernel's own binding.
A library op like mx.fast.rms_norm is one C++ call. On a region whose GPU time
is a few microseconds, that Python could be the whole race.

Three arms over the same rms_norm region as the 13:54 Qwen run (1024 bf16
values, one copy per call), and the same for a 4-bit matvec (1024 -> 2048):
  library   the MLX op
  direct    a hand-written kernel called through mx.fast.metal_kernel with
            literal launch arguments
  harness   the same kernel through autotuner_runtime.kernels.call, launch
            arguments as grammar expressions, as a shipped kernel runs
For each: CPU cost per call (graph building, no eval), and GPU-side per-pass
time under the harness's own chained, cache-cold, paired clock.

Run: uv run python spikes/spike_13_call_overhead.py
"""

import time

import mlx.core as mx

from autotuner.measure.clocks import (CLOCK_TARGET_MS, chained_loop, compare, link_input,
                                      link_loop, loop_iterations, timing_sets)
from autotuner.measure.session import Session, time_once
from autotuner_runtime.kernels import KernelSpec, call

N = 1024
EPS = 1e-6

RMS_BODY = f"""
uint tid = thread_position_in_threadgroup.x;
float acc = 0.0f;
for (uint i = tid; i < {N}u; i += 256u) {{ float v = float(in0[i]); acc += v * v; }}
float s = simd_sum(acc);
threadgroup float partial[8];
if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = s;
threadgroup_barrier(mem_flags::mem_threadgroup);
float total = 0.0f;
for (uint j = 0; j < 8u; j++) total += partial[j];
float inv = metal::precise::rsqrt(total / {N}.0f + {EPS}f);
for (uint i = tid; i < {N}u; i += 256u) out0[i] = T(float(in0[i]) * inv * float(in1[i]));
"""


def cpu_us_per_call(fn, reps=2000):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e6


def gpu_us_per_pass(session, pass_fn, sets, link_id, label):
    iters = loop_iterations(session.timed, lambda n: chained_loop(pass_fn, sets, n, link_id), CLOCK_TARGET_MS)
    loop = chained_loop(pass_fn, sets, iters, link_id)
    net = compare(session, link_loop(sets, iters, link_id), loop, pairs=8)
    per_pass = -net.median_delta_ms / iters * 1e3
    print(f"  {label:8s} GPU per pass {per_pass:7.2f} us   (iters {iters}, stability {net.stability:.2f})")
    return per_pass


def main():
    mx.random.seed(0)
    x = mx.random.normal((1, 1, N)).astype(mx.bfloat16)
    w = mx.random.normal((N,)).astype(mx.bfloat16)
    mx.eval(x, w)

    kernel = mx.fast.metal_kernel(name="spike13_rms", input_names=["in0", "in1"], output_names=["out0"],
                                  source=RMS_BODY, compile_options={"math_mode": "safe"})
    spec = KernelSpec(kernel_id="spike13", name="spike13_rms", input_names=("in0", "in1"),
                      output_names=("out0",), source=RMS_BODY, grid=("256", "1", "1"),
                      threadgroup=("256", "1", "1"),
                      output_shapes=(("in0.shape[0]", "in0.shape[1]", "in0.shape[2]"),),
                      output_dtypes=("bfloat16",), template=(("T", "in0"),))

    def library(b):
        return [mx.fast.rms_norm(b[0], b[1], EPS)]

    def direct(b):
        return kernel(inputs=[b[0], b[1]], output_shapes=[(1, 1, N)], output_dtypes=[mx.bfloat16],
                      grid=(256, 1, 1), threadgroup=(256, 1, 1), template=[("T", mx.bfloat16)])

    def harness(b):
        return call(spec, [b[0], b[1]])

    ref = library({0: x, 1: w})[0]
    for name, fn in (("direct", direct), ("harness", harness)):
        got = fn({0: x, 1: w})[0]
        print(f"{name} max abs diff vs library: {mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item():.3g}")

    print("\nCPU cost of one call, graph building only (no eval):")
    for name, fn in (("library", library), ("direct", direct), ("harness", harness)):
        print(f"  {name:8s} {cpu_us_per_call(lambda: fn({0: x, 1: w})):7.1f} us per call")

    print("\nGPU per pass under the harness's chained, cache-cold, paired clock:")
    session = Session()
    sets = timing_sets([{0: x, 1: w}])
    link_id = link_input(sets[0], {1})
    for name, fn in (("library", library), ("direct", direct), ("harness", harness)):
        gpu_us_per_pass(session, fn, sets, link_id, name)


if __name__ == "__main__":
    main()
