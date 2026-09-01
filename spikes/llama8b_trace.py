"""Trace Llama 3 8B (4-bit) prefill through the tracer and region machinery.

Not a job run: this proves completeness, addressing, copy grouping, and scope
screening on a real 8B model, and reports what the region builder sees. Run
serially on a quiet-ish machine; the trace itself is lazy and cheap."""

import sys
import time
from collections import Counter

import mlx.core as mx

from autotuner.trace import Tracer
from autotuner.regions.build import build_stretches, is_view
from autotuner.regions.fingerprint import group_copies
from autotuner.bind.certify import screen_scope

MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
L = 256


def main() -> int:
    tracer = Tracer()
    tracer.install()

    from mlx_lm import load  # after install: the tracer must see mlx_lm's ops

    t0 = time.perf_counter()
    model, _tok = load(MODEL)
    print(f"loaded {MODEL} in {time.perf_counter() - t0:.1f}s")

    tokens = mx.random.randint(0, 128000, (1, L), key=mx.random.key(0))
    t0 = time.perf_counter()
    trace, outs = tracer.trace(model, [tokens])
    print(f"pass 1 recorded {len(trace.nodes)} nodes in {time.perf_counter() - t0:.1f}s "
          f"(lazy, no eval)")
    print(f"in_pass_evaluation: {trace.in_pass_evaluation}")
    print(f"weights registered: {len(trace.weights)}, step outputs: {len(trace.step_outputs)}")

    ops = Counter(n.op for n in trace.nodes)
    print("top ops:", ops.most_common(12))
    opaque = [n for n in trace.nodes if n.op == "compiled_fn"]
    print(f"opaque compiled calls: {len(opaque)}")
    views = sum(1 for n in trace.nodes if is_view(n))
    print(f"view nodes: {views}/{len(trace.nodes)}")

    addresses = Counter(n.module_address.split("@")[0].rsplit(".", 1)[0]
                        for n in trace.nodes if n.module_address)
    layer_addrs = {a for a in addresses if ".layers." in a or a.startswith("layers")}
    print(f"distinct op-bearing module paths: {len(addresses)}")

    t0 = time.perf_counter()
    stretches = build_stretches(trace, "prefill")
    print(f"stretches: {len(stretches)} in {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    regions = group_copies({"prefill": trace}, {"prefill": stretches})
    print(f"regions after copy grouping: {len(regions)} in {time.perf_counter() - t0:.1f}s")

    by_copies = Counter(r.copies for r in regions)
    print("copy histogram (copies -> regions):", dict(sorted(by_copies.items())))
    many = sorted((r for r in regions if r.copies >= 16), key=lambda r: -len(r.ops))[:5]
    for r in many:
        print(f"  region {r.fingerprint} copies={r.copies} ops={len(r.ops)}: "
              f"{list(r.ops)[:6]}{'...' if len(r.ops) > 6 else ''}")

    screened = 0
    reasons = Counter()
    for r in regions[:400]:
        reason = screen_scope(trace, r.members[0].scope_stack)
        if reason is None:
            screened += 1
        else:
            reasons[reason.split("(")[0][:60]] += 1
    print(f"screen (first 400 regions): {screened} pass")
    for reason, count in reasons.most_common(5):
        print(f"  strand x{count}: {reason}")

    tracer.uninstall()
    bad = tracer.verify_restored()
    print(f"uninstall clean: {not bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
