"""Quick check of this Mac's raw GPU speed, in about ten seconds.

Times three standard operations and prints how fast the GPU ran them:

- a 4096 x 4096 x 4096 fp16 matrix multiply (raw compute, in TFLOPS)
- a 4-bit quantized matrix multiply shaped like a language-model layer
- a sum over 256 MB of memory (memory bandwidth, in GB/s)

Use it to see whether the GPU is free and running at full speed before timing
anything. An Apple M4 reads about 3.3 TFLOPS, 2.8 TFLOPS and 91 GB/s when idle.
Numbers well below your chip's usual reading mean something else is using the
GPU (an optimization job, a benchmark, a video call) or the chip is hot and
throttling. Run it again once the GPU is quiet.

Usage: uv run python tools/gpu_speed_check.py
"""

import time

import mlx.core as mx


def clock(fn, n=5):
    """Median time of n calls, after one call to compile and warm up."""
    mx.eval(fn())
    mx.synchronize()
    times = []
    for _ in range(n):
        mx.synchronize()
        start = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        times.append(time.perf_counter() - start)
    return sorted(times)[len(times) // 2]


def main():
    print("device:", mx.default_device())

    m = k = n = 4096
    a = mx.random.normal((m, k)).astype(mx.float16)
    b = mx.random.normal((k, n)).astype(mx.float16)
    mx.eval(a, b)
    t = clock(lambda: a @ b)
    print(f"fp16 matmul 4096^3:                  {t * 1e3:7.2f} ms -> {2 * m * k * n / t / 1e12:.2f} TFLOPS")

    x = mx.random.normal((512, 4096)).astype(mx.float16)
    w = mx.random.normal((4096, 4096)).astype(mx.float16)
    wq, scales, biases = mx.quantize(w, bits=4, group_size=64)
    mx.eval(x, wq, scales, biases)
    t = clock(lambda: mx.quantized_matmul(x, wq, scales, biases, transpose=True, bits=4, group_size=64))
    print(f"4-bit matmul [512,4096]x[4096,4096]: {t * 1e3:7.2f} ms -> {2 * 512 * 4096 * 4096 / t / 1e12:.2f} TFLOPS")

    big = mx.zeros((256 * 1024 * 1024 // 2,), dtype=mx.float16)  # 256 MB
    mx.eval(big)
    t = clock(lambda: big.sum())
    print(f"sum over 256 MB:                     {t * 1e3:7.2f} ms -> {256 / 1024 / t:.1f} GB/s")


if __name__ == "__main__":
    main()
