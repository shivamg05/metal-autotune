"""Spike 04: delivery micro-spike, the whole bind mechanism in miniature (plan 5.2, M0).

Proves by experiment, on a two-child toy module with a stateful step:
- a hand-written replay wrapper swapped in by parent setattr is bitwise invisible
  in identity form, outputs and state, across repeated calls
- a fused mx.fast.metal_kernel spliced over two adjacent elementwise ops matches
  the library within fp tolerance
- a retrace shows the two cut ops gone, neighbors intact, one custom kernel call
- rollback by parent setattr restores baseline behavior exactly

Output: one line per fact, `FACT <slug>: PASS|FAIL|INFO - <detail>`. Exit 0 if the
script ran to completion, even with FAILs.
"""

import sys
import traceback

import mlx.core as mx
import mlx.nn as nn

DIM = 16
BATCH = 8
N_CALLS = 3
RTOL, ATOL = 1e-5, 1e-6  # plan 5.12 assoc-preserving fp32 tolerance


def fact(slug, status, detail):
    print(f"FACT {slug}: {status} - {detail}")


# ---- minimal call log via temporarily patched mx functions (not the real tracer)

RECORDED_OPS = ("multiply", "add", "exp", "sin", "matmul", "sum")
LOG = []
_SAVED = {}


def arm_recorder():
    LOG.clear()
    for name in RECORDED_OPS:
        orig = getattr(mx, name)
        _SAVED[name] = orig

        def make(name=name, orig=orig):
            def patched(*args, **kwargs):
                LOG.append(name)
                return orig(*args, **kwargs)

            return patched

        setattr(mx, name, make())


def disarm_recorder():
    for name, orig in _SAVED.items():
        setattr(mx, name, orig)
    _SAVED.clear()


# ---- kernel shim: the wrapper calls kernels through this so a retrace can log them


def _kernel_call_impl(kernel, **kwargs):
    return kernel(**kwargs)


kernel_call = _kernel_call_impl


def arm_shim():
    global kernel_call

    def logged(kernel, **kwargs):
        LOG.append("custom_kernel")
        return _kernel_call_impl(kernel, **kwargs)

    kernel_call = logged


def disarm_shim():
    global kernel_call
    kernel_call = _kernel_call_impl


# ---- the toy model


class ChildA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = mx.random.normal((dim,))

    def __call__(self, x):
        return mx.add(mx.multiply(x, 2.0), self.bias)


class ChildB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = mx.random.normal((dim, dim))

    def __call__(self, x):
        return mx.matmul(x, self.weight)


class Block(nn.Module):
    # the toy parent: two children, two adjacent elementwise ops between, a state update
    def __init__(self, dim):
        super().__init__()
        self.a = ChildA(dim)
        self.b = ChildB(dim)
        self.acc = mx.zeros((dim,))

    def __call__(self, x):
        h = self.a(x)
        h = mx.exp(h)
        h = mx.sin(h)
        y = self.b(h)
        self.acc = mx.add(self.acc, mx.sum(y, axis=0))
        return mx.add(y, self.acc)


class Root(nn.Module):
    # exists so the block can be swapped by setattr on its parent
    def __init__(self, dim):
        super().__init__()
        self.block = Block(dim)

    def __call__(self, x):
        return self.block(x)


# flat recorded stream of one Block call, hand-derived
EXPECTED_BASE_LOG = ["multiply", "add", "exp", "sin", "matmul", "sum", "add", "add"]
EXPECTED_FUSED_LOG = ["multiply", "add", "custom_kernel", "matmul", "sum", "add", "add"]


# ---- the replay wrapper: re-executes the recorded op sequence, delegating
# attribute and weight access to the wrapped original module


class ReplayWrapper:
    def __init__(self, wrapped, fused_kernel=None):
        self._wrapped = wrapped
        self._fused_kernel = fused_kernel
        self._calls = 0

    def __getattr__(self, name):
        return getattr(self.__dict__["_wrapped"], name)

    def __call__(self, x):
        self._calls += 1
        w = self.__dict__["_wrapped"]
        # ops called by live mx attribute lookup so a retrace records them
        h = mx.multiply(x, 2.0)
        h = mx.add(h, w.a.bias)
        if self._fused_kernel is None:
            h = mx.exp(h)
            h = mx.sin(h)
        else:
            outs = kernel_call(
                self._fused_kernel,
                inputs=[h],
                output_shapes=[h.shape],
                output_dtypes=[h.dtype],
                grid=(h.size, 1, 1),
                threadgroup=(64, 1, 1),
            )
            h = outs[0]
        y = mx.matmul(h, w.b.weight)
        new_state = mx.add(w.acc, mx.sum(y, axis=0))
        w.acc = new_state
        return mx.add(y, new_state)


# ---- helpers


def run_trajectory(root, block, xs):
    block.acc = mx.zeros((DIM,))
    outs, states = [], []
    for x in xs:
        y = root(x)
        mx.eval(y, block.acc)
        outs.append(y)
        states.append(block.acc)
    return outs, states


def flat(traj):
    return traj[0] + traj[1]


def traj_bitwise_equal(t1, t2):
    return all(bool(mx.array_equal(p, q)) for p, q in zip(flat(t1), flat(t2)))


def traj_allclose(t1, t2):
    return all(bool(mx.allclose(p, q, rtol=RTOL, atol=ATOL)) for p, q in zip(flat(t1), flat(t2)))


def traj_max_abs_diff(t1, t2):
    return max(float(mx.max(mx.abs(p - q))) for p, q in zip(flat(t1), flat(t2)))


def single_retrace_log(root, block, x, with_shim=False):
    block.acc = mx.zeros((DIM,))
    arm_recorder()
    if with_shim:
        arm_shim()
    try:
        y = root(x)
        mx.eval(y, block.acc)
    finally:
        disarm_recorder()
        if with_shim:
            disarm_shim()
    return list(LOG)


def main():
    # found while building this spike: nn.Module reserves `state` as a property with
    # no deleter, so a model attribute named `state` breaks Module.__setattr__
    reserved = isinstance(getattr(nn.Module, "state", None), property)
    fact("module-state-name-reserved", "INFO",
         f"nn.Module.state is a property on 0.32.2: {reserved}; assigning self.state "
         "in a subclass raises AttributeError, so the toy's state attribute is named acc")

    mx.random.seed(42)
    root = Root(DIM)
    block = root.block
    xs = [mx.random.normal((BATCH, DIM)) for _ in range(N_CALLS)]
    mx.eval(root.parameters(), xs)

    # 1. record the op stream by hand and check it against the hand-derived expectation
    base_log = single_retrace_log(root, block, xs[0])
    if base_log == EXPECTED_BASE_LOG:
        fact("record-op-stream", "PASS",
             f"one Block call recorded as {base_log}, matching the hand-derived sequence")
    else:
        fact("record-op-stream", "FAIL",
             f"recorded {base_log}, expected {EXPECTED_BASE_LOG}")

    # 2. baseline trajectory: outputs and state across repeated stateful calls
    base = run_trajectory(root, block, xs)

    # 3. identity wrapper swap-in by parent setattr
    ident = ReplayWrapper(block)
    root.block = ident
    swapped = getattr(root, "block") is ident
    in_dunder = "block" in root.__dict__
    in_moddict = "block" in root  # nn.Module is a dict of its children
    if swapped:
        fact("parent-setattr-swap", "PASS",
             f"parent setattr installed a plain-object wrapper (stored in __dict__={in_dunder}, "
             f"module dict entry popped={not in_moddict}); lookup returns the wrapper")
    else:
        fact("parent-setattr-swap", "FAIL",
             f"after setattr, getattr returned {type(getattr(root, 'block')).__name__}, not the wrapper")

    ident_traj = run_trajectory(root, block, xs)
    routed = ident._calls == N_CALLS
    out_bitwise = all(bool(mx.array_equal(p, q)) for p, q in zip(base[0], ident_traj[0]))
    if routed and out_bitwise:
        fact("identity-wrapper-bitwise", "PASS",
             f"{N_CALLS} calls routed through the wrapper; every output mx.array_equal to baseline")
    else:
        fact("identity-wrapper-bitwise", "FAIL",
             f"wrapper calls={ident._calls}/{N_CALLS}, outputs bitwise equal={out_bitwise}")

    state_bitwise = all(bool(mx.array_equal(p, q)) for p, q in zip(base[1], ident_traj[1]))
    if state_bitwise:
        fact("identity-wrapper-state", "PASS",
             f"state array after each of {N_CALLS} calls mx.array_equal to baseline "
             "(state updates flowed through the replayed ops)")
    else:
        fact("identity-wrapper-state", "FAIL", "replayed state trajectory diverged from baseline")

    # 4. attribute/weight delegation through __getattr__
    deleg = (root.block.a is block.a
             and root.block.b.weight is block.b.weight
             and bool(mx.array_equal(root.block.acc, block.acc)))
    fact("wrapper-getattr-delegation", "PASS" if deleg else "FAIL",
         "wrapper.a, wrapper.b.weight, wrapper.acc resolve to the wrapped original's own objects"
         if deleg else "delegated attribute did not resolve to the original module's object")

    # 5. identity retrace must equal the baseline stream (identity certification in miniature)
    ident_log = single_retrace_log(root, block, xs[0])
    if ident_log == base_log:
        fact("identity-retrace-matches", "PASS",
             f"retrace of the identity wrapper recorded {ident_log}, identical to baseline")
    else:
        fact("identity-retrace-matches", "FAIL",
             f"identity retrace {ident_log} != baseline {base_log}")

    # 6. fused form: one metal kernel over the two adjacent elementwise ops.
    # metal::precise:: variants reproduce the library's unary kernels bitwise;
    # plain metal:: ones do not (probed below), so the fused kernel uses precise.
    kernel = None
    source = """
        uint i = thread_position_in_grid.x;
        out[i] = metal::precise::sin(metal::precise::exp(inp[i]));
    """
    try:
        kernel = mx.fast.metal_kernel(
            name="spike04_fused_exp_sin",
            input_names=["inp"],
            output_names=["out"],
            source=source,
            compile_options={"math_mode": "safe"},
        )
        fact("metal-kernel-signature", "PASS",
             "constructed from body-only source with compile_options math_mode=safe; call is "
             "keyword-only with inputs/output_shapes/output_dtypes/grid/threadgroup, grid in total threads")
    except Exception as e:
        fact("metal-kernel-signature", "FAIL", f"construction raised {type(e).__name__}: {e}")

    # probe: plain metal:: vs metal::precise:: against library exp-then-sin bits
    try:
        probe_in = xs[0]
        probe_ref = mx.sin(mx.exp(probe_in))
        diffs = {}
        for tag, body in (
            ("metal", "uint i = thread_position_in_grid.x;\nout[i] = metal::sin(metal::exp(inp[i]));"),
            ("precise", "uint i = thread_position_in_grid.x;\nout[i] = metal::precise::sin(metal::precise::exp(inp[i]));"),
        ):
            for mode in ("safe", "fast"):
                k = mx.fast.metal_kernel(
                    name=f"spike04_probe_{tag}_{mode}", input_names=["inp"], output_names=["out"],
                    source=body, compile_options={"math_mode": mode})
                (o,) = k(inputs=[probe_in], output_shapes=[probe_in.shape],
                         output_dtypes=[probe_in.dtype], grid=(probe_in.size, 1, 1),
                         threadgroup=(64, 1, 1))
                mx.eval(o)
                diffs[f"{tag}/{mode}"] = float(mx.max(mx.abs(o - probe_ref)))
        fact("transcendental-precision", "INFO",
             f"max abs diff vs library sin(exp(x)): {diffs}; precision comes from the "
             "metal:: vs metal::precise:: namespace, not from math_mode")
    except Exception as e:
        fact("transcendental-precision", "INFO", f"probe failed: {type(e).__name__}: {e}")

    # plan 5.10: compile errors surface at eval of a probe output, not at construction
    # or call (an invalid C identifier in the kernel name breaks the generated signature)
    try:
        bad = mx.fast.metal_kernel(
            name="spike04-bad-name", input_names=["inp"], output_names=["out"],
            source="uint i = thread_position_in_grid.x;\nout[i] = inp[i];")
        (bo,) = bad(inputs=[xs[0]], output_shapes=[xs[0].shape], output_dtypes=[xs[0].dtype],
                    grid=(xs[0].size, 1, 1), threadgroup=(64, 1, 1))
        try:
            mx.eval(bo)
            fact("compile-error-at-eval", "FAIL",
                 "kernel with invalid C-identifier name compiled and ran; expected a build error")
        except RuntimeError as e:
            fact("compile-error-at-eval", "PASS",
                 "construction and call both accepted a broken kernel (hyphenated name); "
                 f"RuntimeError surfaced only at mx.eval of the probe output: {str(e).splitlines()[0]}")
    except Exception as e:
        fact("compile-error-at-eval", "FAIL",
             f"error surfaced before eval, at {type(e).__name__}: {e}")

    fused_ok = False
    if kernel is not None:
        fused = ReplayWrapper(block, fused_kernel=kernel)
        root.block = fused
        try:
            fused_traj = run_trajectory(root, block, xs)
            fused_ok = True
        except Exception as e:
            fact("fused-kernel-matches", "FAIL",
                 f"fused wrapper raised at eval: {type(e).__name__}: {e}")
        if fused_ok:
            close = traj_allclose(base, fused_traj)
            diff = traj_max_abs_diff(base, fused_traj)
            if close:
                fact("fused-kernel-matches", "PASS",
                     f"fused sin(exp(x)) kernel over {N_CALLS} stateful calls within fp32 "
                     f"rtol={RTOL} atol={ATOL}; max abs diff {diff:.3g}")
            else:
                fact("fused-kernel-matches", "FAIL",
                     f"fused output outside fp32 tolerance; max abs diff {diff:.3g}")
            bitwise = traj_bitwise_equal(base, fused_traj)
            fact("fused-kernel-bitwise", "INFO",
                 f"fused kernel bitwise-equal to library exp-then-sin: {bitwise} "
                 f"(max abs diff {diff:.3g}); order-preserving math {'did' if bitwise else 'did not'} "
                 "reproduce library bits")

            # 7. retrace of the fused model: cut ops gone, neighbors intact, one kernel call
            fused_log = single_retrace_log(root, block, xs[0], with_shim=True)
            cut_gone = "exp" not in fused_log and "sin" not in fused_log
            one_kernel = fused_log.count("custom_kernel") == 1
            neighbors = all(op in fused_log for op in ("multiply", "add", "matmul", "sum"))
            if fused_log == EXPECTED_FUSED_LOG:
                fact("retrace-cut-ops-gone", "PASS",
                     f"fused retrace recorded {fused_log}: exp and sin gone, one custom kernel "
                     "call in their place, neighbors intact and in order")
            else:
                fact("retrace-cut-ops-gone", "FAIL",
                     f"fused retrace {fused_log} (cut gone={cut_gone}, one kernel={one_kernel}, "
                     f"neighbors={neighbors}), expected {EXPECTED_FUSED_LOG}")
    else:
        fact("fused-kernel-matches", "FAIL", "skipped: kernel construction failed")
        fact("retrace-cut-ops-gone", "FAIL", "skipped: kernel construction failed")

    # 8. rollback: swap the original child back by parent setattr
    root.block = block
    restored = getattr(root, "block") is block
    rb_traj = run_trajectory(root, block, xs)
    rb_bitwise = traj_bitwise_equal(base, rb_traj)
    rb_log = single_retrace_log(root, block, xs[0])
    if restored and rb_bitwise and rb_log == base_log:
        fact("rollback-restores", "PASS",
             "parent setattr restored the original module; trajectory bitwise equal to "
             "baseline and retrace stream identical")
    else:
        fact("rollback-restores", "FAIL",
             f"restored={restored}, trajectory bitwise={rb_bitwise}, retrace matches={rb_log == base_log}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)
