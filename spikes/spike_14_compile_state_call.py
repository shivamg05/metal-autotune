"""What mx.compile does to a step whose cache is written inside a method.

Two questions, answered on the real Qwen3 0.6B 4-bit decode step:

1. Does its trace report any python_retained arrays? That count was the whole
   detector for "this step keeps state, do not compile it", and a cache write
   recorded as one state call leaves nothing on the model to count.
2. Does compiling this step break it, and on which compile? The answer decides
   whether a missed detection is loud or silent.

Run from the repo root: uv run python spikes/spike_14_compile_state_call.py
"""
import importlib.util, sys
import mlx.core as mx

sys.path.insert(0, "/Users/shivamgarg/dev/metal-autotune")
from autotuner.trace import Tracer
from autotuner.trace.recorder import is_opaque

tracer = Tracer()
tracer.install(model_module_name="qwen4bit")
spec = importlib.util.spec_from_file_location("qwen4bit", "models/qwen3_0.6b_4bit_decode.py")
mod = importlib.util.module_from_spec(spec); sys.modules["qwen4bit"] = mod
spec.loader.exec_module(mod)
model = mod.build()
tok = mx.random.randint(0, 151936, (1, 1), key=mx.random.key(3))

trace, _ = tracer.trace(model, [tok])
tracer.uninstall()

retained = trace.python_retained()
opaque = [n.op for n in trace.nodes if is_opaque(n.op)]
state = [o for o in opaque if o.startswith("state:")]
print(f"Q1 nodes={len(trace.nodes)} python_retained={len(retained)} "
      f"opaque={len(opaque)} state_calls={len(state)} kinds={sorted(set(state))}")

# Q2: compile the step the way loop._step_fn does, twice, exactly as the job did.
def clock(fn, n=3):
    for _ in range(n):
        mx.eval(fn())

print("Q2 plain call:", end=" ", flush=True)
mx.eval(model(tok)); print("ok")

for attempt in (1, 2):
    try:
        compiled = mx.compile(lambda *t: model(*t))
        mx.eval(compiled(tok))
        print(f"Q2 compile #{attempt}: ok")
    except Exception as e:
        print(f"Q2 compile #{attempt}: {type(e).__name__}: {str(e).splitlines()[0]}")

print("Q2 plain call after compiles:", end=" ", flush=True)
try:
    mx.eval(model(tok)); print("ok")
except Exception as e:
    print(f"{type(e).__name__}: {str(e).splitlines()[0]}")
