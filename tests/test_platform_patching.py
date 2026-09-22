"""Pinned platform facts from spike_01 (patchability) and spike_02 (module
wrapping). Each test names the design argument it protects; a failure here
means an mlx upgrade changed dispatch mechanics under the tracer, not that
the harness has a bug. Behavior only, never timing. Every patch is restored
in a finally block so the file leaves mx and mlx.nn exactly as found.
"""

import mlx.core as mx
import mlx.nn as nn


def _a():
    return mx.array([1.0, 2.0, 3.0, 4.0])


def _b():
    return mx.array([4.0, 3.0, 2.0, 1.0])


def _spy_module_fn(name, calls):
    orig = getattr(mx, name)

    def wrapper(*args, __orig=orig, __name=name, **kwargs):
        calls.append(__name)
        return __orig(*args, **kwargs)

    setattr(mx, name, wrapper)
    return orig


def _spy_dunder(name, calls):
    orig = vars(mx.array)[name]

    def wrapper(self, *args, __orig=orig, __name=name, **kwargs):
        calls.append(__name)
        return __orig(self, *args, **kwargs)

    setattr(mx.array, name, wrapper)
    return orig


def test_module_setattr_seen_by_nn_at_call_time():
    """The tracer records module-level ops by plain setattr on mx. That only
    traces anything because mlx.nn layers resolve mx functions at call time,
    not import time: a patched mx.addmm must be hit by nn.Linear(bias=True)."""
    calls = []
    lin = nn.Linear(4, 3, bias=True)
    x = mx.arange(8, dtype=mx.float32).reshape(2, 4)
    expected = lin(x)
    mx.eval(expected)
    orig = _spy_module_fn("addmm", calls)
    try:
        out = lin(x)
        mx.eval(out)
        assert "addmm" in calls
        assert mx.array_equal(out, expected).item()
    finally:
        mx.addmm = orig
    assert mx.addmm is orig


def test_patching_mx_add_does_not_intercept_plus():
    """a + b dispatches through the array C slot and never calls mx.add, so
    module-level patching alone cannot record operator expressions. This is
    why the array dunder patches are load-bearing."""
    calls = []
    orig = _spy_module_fn("add", calls)
    try:
        r = _a() + _b()
        mx.eval(r)
        assert calls == []
        assert r.tolist() == [5.0] * 4
    finally:
        mx.add = orig
    assert mx.add is orig


def test_array_dunder_setattr_propagates_to_c_slots():
    """mx.array is a mutable nanobind heaptype: setattr of a Python wrapper on
    a dunder updates the C slot the interpreter dispatches through. a + b,
    2.0 * x (reflected), a @ b, a[i], and a[i] = v are all interceptable, and
    the wrappers delegate correctly. Operator recording rests entirely on this."""
    calls = []
    names = ["__add__", "__rmul__", "__matmul__", "__getitem__", "__setitem__"]
    originals = {n: _spy_dunder(n, calls) for n in names}
    try:
        m = mx.array([[1.0, 2.0], [3.0, 4.0]])
        r_add = _a() + _b()
        r_rmul = 2.0 * _a()
        r_mat = m @ m
        r_get = _a()[1:3]
        wrote = _a()
        wrote[0] = 9.0
        mx.eval(r_add, r_rmul, r_mat, r_get, wrote)
        for n in names:
            assert n in calls, f"{n} not intercepted via C slot"
        assert r_add.tolist() == [5.0] * 4
        assert r_rmul.tolist() == [2.0, 4.0, 6.0, 8.0]
        assert wrote.tolist()[0] == 9.0
    finally:
        for n, orig in originals.items():
            setattr(mx.array, n, orig)
    for n, orig in originals.items():
        assert vars(mx.array)[n] is orig


def test_augmented_assignment_hits_iadd_not_add():
    """a += b dispatches to __iadd__ and never folds to __add__, so the tracer
    must patch the in-place dunders too or in-place ops vanish from the trace."""
    calls = []
    originals = {n: _spy_dunder(n, calls) for n in ("__iadd__", "__add__")}
    try:
        x = _a()
        x += _b()
        mx.eval(x)
        assert "__iadd__" in calls
        assert "__add__" not in calls
        assert x.tolist() == [5.0] * 4
    finally:
        for n, orig in originals.items():
            setattr(mx.array, n, orig)
    for n, orig in originals.items():
        assert vars(mx.array)[n] is orig


def test_reflected_arithmetic_dunders_exist_and_intercept():
    """scalar-op-array expressions (2.0 + x) dispatch through reflected
    dunders, which need their own patches. All 7 arithmetic reflected dunders
    exist on mx.array and intercept when patched; if an upgrade drops one,
    scalar-left expressions become untraceable."""
    triggers = {
        "__radd__": lambda: 2.0 + _a(),
        "__rsub__": lambda: 2.0 - _a(),
        "__rmul__": lambda: 2.0 * _a(),
        "__rtruediv__": lambda: 2.0 / _a(),
        "__rfloordiv__": lambda: 2.0 // _a(),
        "__rmod__": lambda: 2.0 % _a(),
        "__rpow__": lambda: 2.0 ** _a(),
    }
    for n in triggers:
        assert n in vars(mx.array), f"{n} missing from mx.array"
    calls = []
    originals = {n: _spy_dunder(n, calls) for n in triggers}
    try:
        for n, trig in triggers.items():
            mx.eval(trig())
            assert n in calls, f"{n} not intercepted"
    finally:
        for n, orig in originals.items():
            setattr(mx.array, n, orig)
    for n, orig in originals.items():
        assert vars(mx.array)[n] is orig


def test_base_module_call_patch_does_not_fire():
    """nn.Module defines no __call__ of its own; each layer class defines it.
    There is no single base-class hook, so the tracer's module wrap surface
    must be per-subclass type(m).__call__ (spike_02)."""
    assert "__call__" not in vars(nn.Module)
    lin = nn.Linear(4, 3)
    x = mx.arange(8, dtype=mx.float32).reshape(2, 4)
    expected = lin(x)
    mx.eval(expected)
    fired = []
    nn.Module.__call__ = lambda self, *a, **k: fired.append(1)
    try:
        out = lin(x)
        assert fired == []
        assert mx.array_equal(out, expected).item()
    finally:
        del nn.Module.__call__
    assert "__call__" not in vars(nn.Module)


def test_subclass_call_patch_intercepts_instance_assignment_does_not():
    """Dunder lookup goes to the type and skips the instance: the module
    wrapper must install on type(m).__call__, where it intercepts exactly
    once per call, and a per-instance __call__ assignment silently fails."""
    lin = nn.Linear(4, 3)
    x = mx.arange(8, dtype=mx.float32).reshape(2, 4)
    expected = lin(x)
    mx.eval(expected)
    calls = []
    orig = vars(nn.Linear)["__call__"]

    def wrapper(self, *args, **kwargs):
        calls.append(id(self))
        return orig(self, *args, **kwargs)

    nn.Linear.__call__ = wrapper
    try:
        out = lin(x)
        assert calls == [id(lin)]
        assert mx.array_equal(out, expected).item()
    finally:
        nn.Linear.__call__ = orig
    assert vars(nn.Linear)["__call__"] is orig

    inst_fired = []
    lin.__call__ = lambda *a, **k: inst_fired.append(1)
    out = lin(x)
    assert inst_fired == []
    assert mx.array_equal(out, expected).item()


def test_uninstall_restores_identity_on_every_surface():
    """Pass 2 (the step clock) runs with the recorder fully removed: after
    restore, the mx function, the array dunder, and the layer __call__ must be
    the identical original objects, wrappers must stay silent, and outputs
    must be unchanged. Anything less contaminates the baseline clock."""
    lin = nn.Linear(4, 3)
    x = mx.arange(8, dtype=mx.float32).reshape(2, 4)
    expected = lin(x)
    mx.eval(expected)
    calls = []
    orig_exp = _spy_module_fn("exp", calls)
    orig_add = _spy_dunder("__add__", calls)
    orig_call = vars(nn.Linear)["__call__"]

    def call_wrapper(self, *args, **kwargs):
        calls.append("__call__")
        return orig_call(self, *args, **kwargs)

    nn.Linear.__call__ = call_wrapper
    try:
        mx.eval(mx.exp(_a()) + _b(), lin(x))
        assert set(calls) == {"exp", "__add__", "__call__"}
    finally:
        mx.exp = orig_exp
        setattr(mx.array, "__add__", orig_add)
        nn.Linear.__call__ = orig_call
    assert mx.exp is orig_exp
    assert vars(mx.array)["__add__"] is orig_add
    assert vars(nn.Linear)["__call__"] is orig_call
    n = len(calls)
    r = mx.exp(_a()) + _b()
    out = lin(x)
    mx.eval(r, out)
    assert len(calls) == n
    assert mx.array_equal(out, expected).item()


def test_array_T_is_a_data_descriptor():
    """a.T is a read through a property, not a call: the callable-method patch
    pass never sees it, so the tracer patches array properties (T, real, imag)
    as replacement properties in their own patch kind. A data descriptor also
    guarantees the type-level replacement cannot be shadowed per instance."""
    for name in ("T", "real", "imag"):
        d = vars(mx.array)[name]
        assert not callable(d), f"{name} became callable"
        assert hasattr(type(d), "__get__") and hasattr(type(d), "__set__"), \
            f"{name} is not a data descriptor"

    calls = []
    orig = vars(mx.array)["T"]

    def getter(self):
        calls.append("T")
        return orig.__get__(self, mx.array)

    setattr(mx.array, "T", property(getter))
    try:
        m = mx.array([[1.0, 2.0], [3.0, 4.0]])
        t = m.T
        mx.eval(t)
        assert calls == ["T"]
        assert t.tolist() == [[1.0, 3.0], [2.0, 4.0]]
    finally:
        setattr(mx.array, "T", orig)
    assert vars(mx.array)["T"] is orig
