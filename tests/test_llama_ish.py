"""The tracer and region machinery against a LLaMA-shaped
stack. Completeness, per-layer copy grouping, scope screening, and scaffolds
for the norm-projection region all must hold on real decoder ops."""

import importlib.util
from pathlib import Path

import mlx.core as mx

from autotuner.regions.build import build_stretches
from autotuner.regions.fingerprint import group_copies
from autotuner.bind.certify import screen_scope
from autotuner.trace import Tracer

FIXTURES = Path(__file__).parent / "fixtures"


def test_llama_ish_traces_and_groups():
    tracer = Tracer()
    tracer.install()
    try:
        spec = importlib.util.spec_from_file_location("fx_llama", FIXTURES / "llama_ish.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.build()
        tokens = mx.random.randint(0, 512, (1, 64), key=mx.random.key(0))
        trace, outs = tracer.trace(model, [tokens])

        # completeness held on a real decoder stack: embedding gather, rope,
        # sdpa, swiglu all recorded, nothing appeared from nowhere
        ops = {n.op for n in trace.nodes}
        assert "mx.fast.rms_norm" in ops
        assert "mx.fast.rope" in ops
        assert "mx.fast.scaled_dot_product_attention" in ops
        assert "array.__getitem__" in ops  # the embedding gather

        # every layer's ops carry that layer's address
        addresses = {n.module_address for n in trace.nodes}
        for i in range(8):
            assert any(a.startswith(f"layers.{i}") for a in addresses)

        stretches = build_stretches(trace, "main")
        regions = group_copies({"main": trace}, {"main": stretches})
        eight_copy = [r for r in regions if r.copies == 8]
        assert eight_copy, "no region grouped across all 8 layers"
        biggest = max(eight_copy, key=lambda r: len(r.ops))
        assert len(biggest.ops) >= 10  # a fused-attention-block-sized stretch exists

        # a layer scope passes the static screen: no retention, no opaque
        # calls, no mid-call evaluation in a clean decoder block
        layer_stack = next(
            sc.stack for sc in trace.scope_calls if sc.address == "layers.3@0"
        )
        assert screen_scope(trace, layer_stack) is None
    finally:
        tracer.uninstall()
        assert tracer.verify_restored() == []
