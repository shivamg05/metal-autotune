"""Pinned platform facts from spike_04 (delivery) and spike_05 (module swap).

Bind installs a generated wrapper on the live model by parent setattr and rolls
back the same way. These tests pin the mlx swap semantics that
mechanism relies on, so an mlx upgrade fails here instead of silently breaking
bind. Behavior only, no timing.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten


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


def out(model):
    y = model(mx.array(1.0))
    mx.eval(y)
    return y.item()


def module_names(model):
    return sorted(name for name, _ in model.named_modules() if name)


def param_names(model):
    return sorted(k for k, _ in tree_flatten(model.parameters()))


def test_parent_setattr_swaps_named_child_and_rollback_is_bitwise():
    """Bind's install and its miss path are both parent setattr: the design
    assumes the replacement is used and enters the module tree, and that
    swapping the original back restores baseline output bitwise, or a bind
    miss could not guarantee an unpatched model."""
    m = Parent()
    x = mx.array(1.0)
    baseline = m(x)
    mx.eval(baseline)
    old = m.child

    new = Affine(3.0)
    m.child = new
    assert out(m) == 3.0
    assert any(mod is new for _, mod in m.named_modules())
    assert all(mod is not old for _, mod in m.named_modules())

    m.child = old
    assert m.child is old
    restored = m(x)
    mx.eval(restored)
    assert bool(mx.array_equal(baseline, restored))


def test_update_modules_strict_rejects_numeric_dict_keys():
    """The swap table must address children in plain python lists with
    list-form specs: update_modules refuses a numeric path segment given as a
    dict key, string or int, and leaves the tree untouched."""
    m = ListModel()
    for key in ("3", 3):
        with pytest.raises(ValueError):
            m.update_modules({"layers": {key: Affine(10.0)}})
    assert out(m) == 24.0


def test_update_modules_accepts_list_form_spec():
    """The installer's mechanism for list children: a list spec with {}
    placeholders swaps exactly the one addressed index."""
    m = ListModel()
    new = Affine(10.0)
    m.update_modules({"layers": [{}, {}, {}, new]})
    assert m.layers[3] is new
    assert out(m) == 60.0


def test_update_modules_nonstrict_silently_noops_on_numeric_keys():
    """Hazard pin: strict=False neither raises nor swaps on a numeric dict
    key. Install code must never rely on non-strict mode to surface a bad
    swap-table address, because a bad address would silently leave the
    original module in place."""
    m = ListModel()
    new = Affine(10.0)
    m.update_modules({"layers": {"3": new}}, strict=False)
    assert all(layer is not new for layer in m.layers)
    assert out(m) == 24.0


def test_non_module_wrapper_called_but_vanishes_from_tree():
    """Why generated bind wrappers must subclass nn.Module: a plain-object
    wrapper installed by parent setattr still receives every call, but
    Module.__setattr__ pops the child from the module dict, so the wrapped
    subtree disappears from named_modules() and parameters() and any
    post-install tree walk goes blind. Both halves pinned."""

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
    assert out(m) == 2.0
    assert wrapper.calls == 1
    assert module_names(m) == []
    assert param_names(m) == []


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
    """Parent that calls the child and also reaches through it."""

    def __init__(self):
        super().__init__()
        self.child = DeepChild()

    def __call__(self, x):
        return self.child(x) + x * self.child.weight + self.child.sub.attr


class ModuleWrapper(nn.Module):
    """The generated-wrapper delegation pattern from spike_05."""

    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def __call__(self, *args, **kwargs):
        return self["wrapped"](*args, **kwargs)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        # AttributeError, not KeyError, before 'wrapped' exists: Module.
        # __setattr__ probes attributes during __init__ and must see a clean miss
        if "wrapped" not in self:
            raise AttributeError(name)
        return getattr(self["wrapped"], name)


def test_module_wrapper_survives_reach_through_and_stays_in_tree():
    """Generated bind wrappers subclass nn.Module and delegate unknown
    attributes to the wrapped module. Pins that the pattern constructs at all
    (the __getattr__ raises AttributeError before 'wrapped' exists, keeping
    Module.__setattr__'s probe alive during __init__), that parent code
    reaching through one and two levels still resolves, that output is
    bitwise unchanged, and that the wrapped subtree stays visible to
    named_modules() and parameters()."""
    m = Reacher()
    x = mx.array(1.0)
    before = m(x)
    mx.eval(before)

    m.child = ModuleWrapper(m.child)
    after = m(x)
    mx.eval(after)
    assert bool(mx.array_equal(before, after))
    assert m.child.weight.item() == 2.0
    assert m.child.sub.attr.item() == 7.0
    assert module_names(m) == ["child", "child.wrapped", "child.wrapped.sub"]
    assert param_names(m) == ["child.wrapped.sub.attr", "child.wrapped.weight"]
