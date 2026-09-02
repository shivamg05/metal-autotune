"""Does mx.compile accept a model patched with a generated wrapper and a custom kernel,
bitwise, and does a fresh closure after a swap see the new module? Also: compiled replay
of a region's recorded ops vs plain replay, fp32 and fp16, and the speed of both."""
import sys, tempfile, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import mlx.core as mx
from tests.test_install import _runner, _chain, FUSED, WIN
from autotuner.loop import RegionRun
from autotuner.trace.replay import replay
from autotuner.measure.session import time_once

tmp = pathlib.Path(tempfile.mkdtemp())
r = _runner(tmp, "[64, 1024]")
x = r.tensors["main"]
plain_before = r.model(*x); mx.eval(plain_before)
c0 = mx.compile(lambda *t: r.model(*t))
comp_before = c0(*x); mx.eval(comp_before)
print("compiled untouched == plain untouched:", mx.array_equal(plain_before, comp_before).item())

chain = _chain(r).region
chain.t_orig_ms["main"] = chain.t_rep_ms["main"] = 1.0
r._capture([chain])
print("promote:", r._bind_and_promote(RegionRun(region=chain), FUSED, WIN))
plain_after = r.model(*x); mx.eval(plain_after)
stale = c0(*x); mx.eval(stale)
print("old compiled closure after swap still equals untouched:", mx.array_equal(stale, plain_before).item())
c1 = mx.compile(lambda *t: r.model(*t))
comp_after = c1(*x); mx.eval(comp_after)
print("fresh compiled closure == plain patched:", mx.array_equal(comp_after, plain_after).item())
print("patched vs untouched max abs:", mx.abs(plain_after - plain_before).max().item())

# compiled replay of the chain's recorded ops vs plain replay
trace = r.traces["main"]; m = chain.members[0]
nodes = trace.nodes[m.start_seq:m.end_seq + 1]
ids = list(m.input_ids)
sets = [r.store.load(chain.fingerprint, "main", i, "inputs") for i in range(r.store.set_count(chain.fingerprint, "main"))]
binds = sets[0]
weights = {a: arr for a, arr in binds.items() if a in trace.weights}
def plain_pass(b):
    return list(replay(nodes, b, list(m.output_ids)).values())
arg_ids = [a for a in ids if a not in trace.weights]
def f(*arrays):
    b = {**{a: binds[a] for a in ids if a in trace.weights}, **dict(zip(arg_ids, arrays))}
    return list(replay(nodes, b, list(m.output_ids)).values())
cf = mx.compile(f)
outs_p = plain_pass(binds); outs_c = cf(*[binds[a] for a in arg_ids]); mx.eval(outs_p, outs_c)
print("compiled replay == plain replay (fp32):", all(mx.array_equal(a, b).item() for a, b in zip(outs_p, outs_c)))
# fp16 variant
binds16 = {a: (v.astype(mx.float16) if v.dtype == mx.float32 else v) for a, v in binds.items()}
def f16(*arrays):
    b = {**{a: binds16[a] for a in ids if a in trace.weights}, **dict(zip(arg_ids, arrays))}
    return list(replay(nodes, b, list(m.output_ids)).values())
cf16 = mx.compile(f16)
p16 = f16(*[binds16[a] for a in arg_ids]); c16 = cf16(*[binds16[a] for a in arg_ids]); mx.eval(p16, c16)
print("compiled replay == plain replay (fp16):", all(mx.array_equal(a, b).item() for a, b in zip(p16, c16)),
      "| max abs diff:", max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item() for a, b in zip(p16, c16)))
# speed: plain vs compiled untouched step, and the plain replay vs compiled replay of the chain
r2 = r.baseline_model
cb = mx.compile(lambda *t: r2(*t))
for _ in range(3): time_once(lambda: r2(*x)); time_once(lambda: cb(*x))
tp = sorted(time_once(lambda: [r2(*x) for _ in range(20)]) for _ in range(5))[2] / 20 * 1e3
tc = sorted(time_once(lambda: [cb(*x) for _ in range(20)]) for _ in range(5))[2] / 20 * 1e3
print(f"untouched step: plain {tp:.4f} ms, compiled {tc:.4f} ms")
for _ in range(3): time_once(lambda: plain_pass(binds)); time_once(lambda: cf(*[binds[a] for a in arg_ids]))
rp = sorted(time_once(lambda: [plain_pass(binds) for _ in range(20)]) for _ in range(5))[2] / 20 * 1e3
rc = sorted(time_once(lambda: [cf(*[binds[a] for a in arg_ids]) for _ in range(20)]) for _ in range(5))[2] / 20 * 1e3
print(f"chain replay: plain {rp:.4f} ms, compiled {rc:.4f} ms")
r.tracer.uninstall()
