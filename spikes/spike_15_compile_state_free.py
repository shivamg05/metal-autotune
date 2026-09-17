"""Can the state-free parts of a decode step be compiled, and does it pay?

`mx.compile` refuses the whole step: the KV cache reallocates its arrays as it
grows, so the compiler cannot capture what the function will read. But the
cache is only 28 of the 734 ops in a token (3.8%), and cutting at every cache
write leaves runs of consecutive ops that touch no state at all.

This compiles one such run per layer, the MLP, straight on the model's own
module. No wrapper, no kernel: just the question of whether compiling a
state-free chunk is allowed here and what it is worth.

One model, one cache, one quantization. The two arms differ only in whether
each layer's `.mlp` attribute points at the compiled forward or the original,
so neither arm carries memory or allocation state the other lacks. Both module
lists are built once, before timing, so a toggle never re-enters `mx.compile`.

    uv run python spikes/spike_15_compile_state_free.py
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import mlx.nn as nn

from autotuner.measure.peaks import measure_bandwidth
from autotuner.measure.session import Session

VOCAB, CONTEXT = 151936, 512
HEALTHY_GBPS = 90.0          # a cold chip ramping mid-run reads ~88; the settled band is 92-96
STABLE_FRAC = 0.03           # two consecutive blocks within 3% means the clock has settled


class Compiled(nn.Module):
    """The wrapped module with its forward compiled once. Weights are captured
    as constants, which is what they are: the job never changes them."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        object.__setattr__(self, "_fn", mx.compile(lambda x: inner(x)))

    def __call__(self, x):
        return self._fn(x)


def build(bits: int = 4, group_size: int = 64):
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, _ = load("Qwen/Qwen3-0.6B-Base")
    nn.quantize(model, group_size=group_size, bits=bits)
    cache = make_prompt_cache(model)
    mx.eval(model(mx.random.randint(0, VOCAB, (1, CONTEXT), key=mx.random.key(7)), cache=cache))
    return model, cache


def block(fn, n: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    return (time.perf_counter() - t0) * 1000 / n


def warm_until_stable(run_block, n: int, limit: int = 30) -> int:
    """Run blocks until two in a row agree. Fixed warm-up rounds only hope the
    clock has settled; this checks."""
    prev = run_block(n)
    for i in range(limit):
        cur = run_block(n)
        if abs(cur - prev) / prev < STABLE_FRAC:
            return i + 2
        prev = cur
    raise SystemExit(f"block time never settled within {STABLE_FRAC:.0%} over {limit} rounds; "
                     "the machine is not quiet enough to measure on")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--block", type=int, default=40)
    args = ap.parse_args()

    bw = measure_bandwidth(Session(), samples=3)
    print(f"GPU bandwidth {bw:.0f} GB/s")
    if bw < HEALTHY_GBPS:
        raise SystemExit(f"chip is not in its settled band ({bw:.0f} GB/s, need {HEALTHY_GBPS:.0f}); "
                         "a cold or throttled chip moves under the measurement")

    model, cache = build()
    tok = mx.random.randint(0, VOCAB, (1, 1), key=mx.random.key(8))
    layers = model.model.layers

    def step():
        out = model(tok, cache=cache)
        for c in cache:
            c.offset = CONTEXT       # rewind: every call is the same step
        return out

    originals = [layer.mlp for layer in layers]
    try:
        compiled = [Compiled(m) for m in originals]     # built once, never rebuilt on a toggle
    except Exception as e:
        raise SystemExit(f"compiling the MLP run failed: {type(e).__name__}: {e}")

    def use_compiled(on: bool):
        for layer, orig, comp in zip(layers, originals, compiled):
            layer.mlp = comp if on else orig

    mx.eval(step())
    use_compiled(False); want = step(); mx.eval(want)
    use_compiled(True);  got = step();  mx.eval(got)
    print(f"compiled {len(layers)} MLP runs (one per layer)")
    print(f"outputs match: {bool(mx.allclose(got, want, rtol=1e-4, atol=1e-4).item())}"
          f"   max abs diff {float(mx.max(mx.abs(got - want))):.2e}")

    def timed(on: bool, n: int) -> float:
        use_compiled(on)          # toggled once per block, never inside the timer
        return block(step, n)

    rounds = warm_until_stable(lambda n: timed(False, n), args.block)
    print(f"\nclock settled after {rounds} warm-up blocks")

    ratios, a_times, b_times = [], [], []
    for _ in range(args.pairs // 2):        # ABBA, no idling anywhere
        a1, b1 = timed(False, args.block), timed(True, args.block)
        b2, a2 = timed(True, args.block), timed(False, args.block)
        ratios += [b1 / a1, b2 / a2]
        a_times += [a1, a2]; b_times += [b1, b2]

    r = st.median(ratios)
    print(f"plain step   {st.median(a_times):.3f} ms   (paired)")
    print(f"compiled MLP {st.median(b_times):.3f} ms   (paired)")
    print(f"\nratio compiled/plain {r:.4f}  ({1/r:.3f}x)   spread "
          f"{min(ratios):.3f}-{max(ratios):.3f}")
    print(f"pairs in order: {[round(x, 3) for x in ratios]}")
    print(f"post-run bandwidth {measure_bandwidth(Session(), samples=3):.0f} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
