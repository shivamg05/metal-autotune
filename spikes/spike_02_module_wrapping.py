"""Spike 02: module wrapping facts (plan 5.1 "Module calls", M0 "Module wrapping").

Proves by experiment, on the pinned mlx, how mlx.nn module calls dispatch and
therefore where the tracer's patch surface must sit. Output contract: one line
per fact, `FACT <slug>: PASS|FAIL|INFO - <detail>`. Exit 0 if the script ran.
"""

import sys

import mlx.core as mx
import mlx.nn as nn

facts = []


def fact(slug, status, detail):
    line = f"FACT {slug}: {status} - {detail}"
    facts.append(line)
    print(line, flush=True)


def check(slug, ok, detail):
    fact(slug, "PASS" if ok else "FAIL", detail)


def bitwise_equal(out, expected):
    if not isinstance(out, mx.array):
        return False
    mx.eval(out)
    return mx.array_equal(out, expected).item()


def main():
    mx.random.seed(0)
    x = mx.arange(8, dtype=mx.float32).reshape(2, 4)

    # ---- fact 1: nn.Module defines no __call__ of its own ----
    base_has_call = "__call__" in vars(nn.Module)
    check(
        "module-base-no-own-call",
        not base_has_call,
        f"'__call__' in vars(nn.Module) is {base_has_call}; MRO {[c.__name__ for c in nn.Module.__mro__]}",
    )

    lin = nn.Linear(4, 4)
    expected_lin = lin(x)
    mx.eval(expected_lin)

    # ---- fact 2: patching the base class does not fire for nn.Linear ----
    base_fired = []

    def base_wrapper(self, *args, **kwargs):
        # a sentinel, not a raise, so contrary dispatch shows up as FAIL, not a crash
        base_fired.append(id(self))
        return "base-wrapper-sentinel"

    nn.Module.__call__ = base_wrapper
    try:
        out = lin(x)
        eq = bitwise_equal(out, expected_lin)
        check(
            "base-class-patch-does-not-fire",
            not base_fired and eq,
            f"patched nn.Module.__call__, called nn.Linear: wrapper fired {len(base_fired)} times, output bitwise equal {eq}",
        )
    finally:
        del nn.Module.__call__
    check(
        "base-class-unpatch-clean",
        "__call__" not in vars(nn.Module),
        f"after del, '__call__' in vars(nn.Module) is {'__call__' in vars(nn.Module)}",
    )

    # ---- fact 3: per-subclass type(m).__call__ patch intercepts, two layer types ----
    ln = nn.LayerNorm(4)
    expected_ln = ln(x)
    mx.eval(expected_ln)

    calls = []
    originals = {}

    def patch_class(cls):
        orig = vars(cls)["__call__"]
        originals[cls] = orig

        def wrapper(self, *args, **kwargs):
            calls.append((cls.__name__, id(self)))
            return orig(self, *args, **kwargs)

        cls.__call__ = wrapper

    for m in (lin, ln):
        if type(m) not in originals:
            patch_class(type(m))

    out_lin = lin(x)
    out_ln = ln(x)
    lin_hits = [c for c in calls if c[0] == "Linear"]
    ln_hits = [c for c in calls if c[0] == "LayerNorm"]
    eq_lin = bitwise_equal(out_lin, expected_lin)
    eq_ln = bitwise_equal(out_ln, expected_ln)
    check(
        "subclass-patch-intercepts-linear",
        len(lin_hits) == 1 and eq_lin,
        f"wrapper fired {len(lin_hits)}x for Linear, output bitwise equal {eq_lin}",
    )
    check(
        "subclass-patch-intercepts-layernorm",
        len(ln_hits) == 1 and eq_ln,
        f"wrapper fired {len(ln_hits)}x for LayerNorm, output bitwise equal {eq_ln}",
    )

    # ---- fact 4: self identity distinguishes two instances of one class ----
    lin2 = nn.Linear(4, 4)
    calls.clear()
    a1 = lin(x)
    a2 = lin2(x)
    mx.eval(a1, a2)
    seen_ids = [c[1] for c in calls if c[0] == "Linear"]
    identity_ok = seen_ids == [id(lin), id(lin2)] and id(lin) != id(lin2)
    outputs_differ = not mx.array_equal(a1, a2).item()
    check(
        "self-identity-dispatch",
        identity_ok and outputs_differ,
        f"recorded ids {seen_ids} match [id(lin), id(lin2)]={identity_ok}; instance outputs differ {outputs_differ}",
    )

    # ---- fact 5: unpatching restores each class exactly ----
    for cls, orig in originals.items():
        cls.__call__ = orig
    restored = all(vars(cls)["__call__"] is orig for cls, orig in originals.items())
    calls.clear()
    out = lin(x)
    eq = bitwise_equal(out, expected_lin)
    check(
        "unpatch-restores-exactly",
        restored and not calls and eq,
        f"class dict entries identical to originals {restored}; wrapper fired {len(calls)}x after restore; output bitwise equal {eq}",
    )

    # ---- fact 6: per-instance __call__ assignment does not intercept ----
    inst_fired = []

    def inst_wrapper(*args, **kwargs):
        inst_fired.append(1)
        return "instance-wrapper-sentinel"

    lin.__call__ = inst_wrapper
    out = lin(x)
    eq = bitwise_equal(out, expected_lin)
    check(
        "instance-assign-does-not-intercept",
        not inst_fired and eq,
        f"assigned lin.__call__ on the instance: fired {len(inst_fired)}x, call still dispatched via type, output bitwise equal {eq}",
    )
    landed = "__dict__" if "__call__" in object.__getattribute__(lin, "__dict__") else (
        "dict-storage" if "__call__" in lin else "nowhere-found"
    )
    fact(
        "instance-assign-attr-location",
        "INFO",
        f"the assigned attribute landed in the instance {landed}; dunder lookup bypasses it",
    )
    del lin.__call__

    # ---- fact 7: wrapping the top-level build() callable, nn.Module model ----
    def build_module_model():
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
        mx.eval(model.parameters())
        return model

    def build_function_model():
        w = mx.random.normal((4, 4))
        mx.eval(w)

        def step(inp):
            return mx.tanh(inp @ w)

        return step

    def wrap_top_level(fn):
        count = []

        def stepped(*args, **kwargs):
            count.append(1)
            return fn(*args, **kwargs)

        return stepped, count

    for slug, builder in (
        ("wrap-top-level-module-model", build_module_model),
        ("wrap-top-level-function-model", build_function_model),
    ):
        top = builder()
        expected = top(x)
        mx.eval(expected)
        wrapped, count = wrap_top_level(top)
        out = wrapped(x)
        eq = bitwise_equal(out, expected)
        check(
            slug,
            len(count) == 1 and eq,
            f"wrapper fired {len(count)}x, output bitwise equal {eq} ({type(top).__name__} callable)",
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"SPIKE BROKE: {type(e).__name__}: {e}", file=sys.stderr)
        raise
    sys.exit(0)
