"""Where does Llama 3 8B (4-bit) actually have headroom on this machine?

Measures the real step time of prefill at several lengths and of cached
single-token decode, then places each against the chip's measured peaks:
compute peak for prefill, weight-bytes bandwidth floor and launch floor for
decode. Headroom = measured / floor. Run on a quiet machine."""

import os
import sys
import time

import mlx.core as mx

MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
PREFILL_LS = (32, 128, 512, 1024, 2048)
DECODE_TOKENS = 24
DECODE_CONTEXTS = (128, 1024)

# measured by the harness on this machine, 2026-08-31 (work/report.json)
PEAK_TFLOPS = 3.226
PEAK_GBPS = 95.3
LAUNCH_US = 7.0


def step(fn):
    mx.synchronize()
    t0 = time.perf_counter()
    mx.eval(fn())
    mx.synchronize()
    return time.perf_counter() - t0


def median_step(fn, n=3):
    times = [step(fn) for _ in range(n + 1)][1:]  # first is warm-up
    return sorted(times)[len(times) // 2]


def main() -> int:
    print(f"load average: {os.getloadavg()[0]:.1f}")
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, _ = load(MODEL)
    flat = tree_flatten(model.parameters())
    n_params = sum(v.size * (8 if v.dtype == mx.uint32 else 1)
                   for name, v in flat if "scales" not in name and "biases" not in name)
    weight_bytes = sum(v.nbytes for _, v in flat)
    print(f"params ~{n_params/1e9:.2f}B, weight bytes {weight_bytes/1e9:.2f} GB")

    print("\nprefill (no cache): compute-bound; headroom vs compute peak")
    print(f"{'L':>6} {'ms':>10} {'TFLOP/s':>9} {'% peak':>7} {'headroom':>9}")
    for L in PREFILL_LS:
        tokens = mx.random.randint(0, 128000, (1, L), key=mx.random.key(0))
        t = median_step(lambda: model(tokens))
        flops = 2 * n_params * L
        eff = flops / t / 1e12
        floor = flops / (PEAK_TFLOPS * 1e12)
        print(f"{L:>6} {t*1e3:>10.1f} {eff:>9.2f} {eff/PEAK_TFLOPS*100:>6.0f}% "
              f"{t/floor:>8.2f}x")

    print("\ndecode (cached, 1 token/step): memory- and launch-bound")
    print(f"{'ctx':>6} {'ms/tok':>8} {'GB/s eff':>9} {'mem floor':>10} "
          f"{'launch floor':>13} {'headroom':>9}")
    for ctx in DECODE_CONTEXTS:
        cache = make_prompt_cache(model)
        prompt = mx.random.randint(0, 128000, (1, ctx), key=mx.random.key(1))
        mx.eval(model(prompt, cache=cache))
        tok = mx.random.randint(0, 128000, (1, 1), key=mx.random.key(2))
        times = []
        for _ in range(DECODE_TOKENS):
            times.append(step(lambda: model(tok, cache=cache)))
        t = sorted(times)[len(times) // 2]
        # per token: every weight byte read once, plus the KV cache once
        kv_bytes = 2 * 32 * 8 * 128 * (ctx + DECODE_TOKENS) * 2  # layers*kv_heads*head_dim*fp16
        mem_floor = (weight_bytes + kv_bytes) / (PEAK_GBPS * 1e9)
        launch_floor = 700 * LAUNCH_US / 1e6  # ~700 kernel launches per step
        floor = max(mem_floor, launch_floor)
        eff = (weight_bytes + kv_bytes) / t / 1e9
        print(f"{ctx:>6} {t*1e3:>8.2f} {eff:>9.1f} {mem_floor*1e3:>9.1f}ms "
              f"{launch_floor*1e3:>12.1f}ms {t/floor:>8.2f}x")

    print("\nheadroom > ~1.3x is worth hunting; ~1.0x is already at the roofline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
