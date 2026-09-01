"""Spike 05: module swap semantics on mlx (plan 5.3, M0 "Module swap semantics").

Proves by experiment, on the installed mlx version:
- parent setattr swaps a named child and the model uses the replacement
- update_modules rejects numeric path segments as dict keys, needs list-form spec
- list-index assignment swaps a child living in a plain python list attribute
- a wrapper delegating __getattr__ survives parent code reaching through it
- what parameters()/named_modules() report after each kind of swap
"""

import sys
import traceback

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

mx.random.seed(0)


def fact(slug, status, detail):
    print(f"FACT {slug}: {status} - {detail}")


class Affine(nn.Module):
    """y = x * weight, one scalar parameter."""

    def __init__(self, scale):
        super().__init__()
        self.weight = mx.array(float(scale))

    def __call__(self, x):
        return x * self.weight


class Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = Affine(2.0)

    def __call__(self, x):
        return self.child(x)


class ListModel(nn.Module):
    """Four Affine children in a plain python list; product of scales is 24."""

    def __init__(self):
        super().__init__()
        self.layers = [Affine(float(i + 1)) for i in range(4)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def scalar_out(model):
    y = model(mx.array(1.0))
    mx.eval(y)
    return y.item()


def tree_ids(model):
    return [id(mod) for _, mod in model.named_modules()]


def check_named_setattr():
    m = Parent()
    x = mx.array(1.0)
    before = m(x).item()
    old = m.child
    new = Affine(3.0)
    m.child = new
    after = m(x).item()
    in_tree = id(new) in tree_ids(m)
    old_gone = id(old) not in tree_ids(m)
    ok = before == 2.0 and after == 3.0 and in_tree and old_gone
    fact(
        "parent-setattr-named-child",
        "PASS" if ok else "FAIL",
        f"output {before}->{after} (want 2.0->3.0), new in named_modules={in_tree}, "
        f"old removed={old_gone}",
    )


def check_update_modules_numeric_keys():
    m = ListModel()
    before = scalar_out(m)
    errors = []
    for key in ("3", 3):
        try:
            m.update_modules({"layers": {key: Affine(10.0)}})
            errors.append(f"key {key!r}: accepted")
        except ValueError as e:
            errors.append(f"key {key!r}: ValueError({e})")
        except Exception as e:
            errors.append(f"key {key!r}: {type(e).__name__}({e})")
    after = scalar_out(m)
    rejected = all("ValueError" in e for e in errors)
    untouched = before == after == 24.0
    fact(
        "update-modules-rejects-numeric-dict-keys",
        "PASS" if rejected and untouched else "FAIL",
        f"{'; '.join(errors)}; output before/after attempts {before}/{after}",
    )

    # non-strict mode with the same bad spec: record what it does
    m2 = ListModel()
    try:
        m2.update_modules({"layers": {"3": Affine(10.0)}}, strict=False)
        fact(
            "update-modules-numeric-keys-nonstrict",
            "INFO",
            f"strict=False silently no-ops on dict key '3': output stays {scalar_out(m2)}",
        )
    except Exception as e:
        fact(
            "update-modules-numeric-keys-nonstrict",
            "INFO",
            f"strict=False raised {type(e).__name__}({e})",
        )


def check_update_modules_list_form():
    m = ListModel()
    new = Affine(10.0)
    m.update_modules({"layers": [{}, {}, {}, new]})
    out = scalar_out(m)
    swapped = m.layers[3] is new
    ok = out == 60.0 and swapped
    fact(
        "update-modules-list-form",
        "PASS" if ok else "FAIL",
        f"spec {{'layers': [{{}}, {{}}, {{}}, new]}} gives output {out} (want 60.0), "
        f"layers[3] is new={swapped}",
    )


def check_list_index_assignment():
    m = ListModel()
    new = Affine(10.0)
    old = m.layers[3]
    m.layers[3] = new
    out = scalar_out(m)
    in_tree = id(new) in tree_ids(m)
    old_gone = id(old) not in tree_ids(m)
    ok = out == 60.0 and in_tree and old_gone
    fact(
        "list-index-assignment",
        "PASS" if ok else "FAIL",
        f"model.layers[3] = new gives output {out} (want 60.0), "
        f"new in named_modules={in_tree}, old removed={old_gone}",
    )


class Sub(nn.Module):
    def __init__(self):
        super().__init__()
        self.attr = mx.array(7.0)


class DeepChild(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.array(2.0)
        self.sub = Sub()

    def __call__(self, x):
        return x * self.weight


class Reacher(nn.Module):
    """Parent that reaches through the child and also calls it."""

    def __init__(self):
        super().__init__()
        self.child = DeepChild()

    def __call__(self, x):
        return self.child(x) + x * self.child.weight + self.child.sub.attr


class Wrapper(nn.Module):
    """Delegates calls and unknown attribute reads to the wrapped module."""

    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def __call__(self, *args, **kwargs):
        return self["wrapped"](*args, **kwargs)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        if "wrapped" not in self:
            raise AttributeError(name)
        return getattr(self["wrapped"], name)


def check_wrapper_delegation():
    m = Reacher()
    x = mx.array(1.0)
    before = m(x)
    mx.eval(before)
    m.child = Wrapper(m.child)
    after = m(x)
    mx.eval(after)
    bitwise = bool(mx.array_equal(before, after).item())
    one_level = m.child.weight.item()
    two_level = m.child.sub.attr.item()
    ok = bitwise and one_level == 2.0 and two_level == 7.0
    fact(
        "wrapper-getattr-delegation",
        "PASS" if ok else "FAIL",
        f"output bitwise equal after swap={bitwise} ({before.item()} vs {after.item()}), "
        f"child.weight={one_level} (want 2.0), child.sub.attr={two_level} (want 7.0)",
    )

    names = sorted(name for name, _ in m.named_modules() if name)
    params = sorted(k for k, _ in tree_flatten(m.parameters()))
    fact(
        "wrapped-tree-report",
        "INFO",
        f"after wrapper swap named_modules={names}, parameters={params}",
    )


def check_plain_object_swap():
    # a wrapper that is NOT an nn.Module: __setattr__ pops it from the module dict
    class PlainWrapper:
        def __init__(self, wrapped):
            object.__setattr__(self, "wrapped", wrapped)
            object.__setattr__(self, "calls", 0)

        def __call__(self, *args, **kwargs):
            object.__setattr__(self, "calls", self.calls + 1)
            return self.wrapped(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "wrapped"), name)

    m = Parent()
    wrapper = PlainWrapper(m.child)
    m.child = wrapper
    out = scalar_out(m)
    names = sorted(name for name, _ in m.named_modules() if name)
    params = sorted(k for k, _ in tree_flatten(m.parameters()))
    fact(
        "plain-object-swap",
        "INFO",
        f"non-Module wrapper via setattr: model output {out} through wrapper "
        f"(calls={wrapper.calls}), but named_modules={names} and parameters={params} "
        f"(dropped from module tree)",
    )


def main():
    fact("mlx-version", "INFO", f"mlx {mx.__version__}, device {mx.default_device()}")
    check_named_setattr()
    check_update_modules_numeric_keys()
    check_update_modules_list_form()
    check_list_index_assignment()
    check_wrapper_delegation()
    check_plain_object_swap()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
