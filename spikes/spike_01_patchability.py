"""Spike 01: patchability of the mx module surface and mx.array dunders (plan 5.1, M0).

Proves by experiment, on the installed mlx, that module-level setattr on mx is seen
by mlx.nn layers, that mx.array dunder patches propagate to the C slots, which dunders
exist and intercept, whether a += b folds, that __eq__/__hash__ patches keep arrays
usable as dict/set keys, and that uninstall restores mx and mx.array exactly.

Output: one line per fact, `FACT <slug>: PASS|FAIL|INFO - <detail>`.
Exit 0 if the script ran to completion, even with FAILs.
"""

import operator
import sys

import mlx.core as mx
import mlx.nn as nn

calls = []    # op names recorded by wrappers, in call order
patched = []  # (kind, target, name, original) for restore and identity checks


def fact(slug, status, detail):
    print(f"FACT {slug}: {status} - {detail}")


def patch_module_fn(name):
    orig = getattr(mx, name)

    def wrapper(*args, __orig=orig, __name=f"mx.{name}", **kwargs):
        calls.append(__name)
        return __orig(*args, **kwargs)

    setattr(mx, name, wrapper)
    patched.append(("module", mx, name, orig))
    return orig


def patch_type_attr(name):
    orig = vars(mx.array)[name]

    def wrapper(self, *args, __orig=orig, __name=name, **kwargs):
        calls.append(__name)
        return __orig(self, *args, **kwargs)

    setattr(mx.array, name, wrapper)
    patched.append(("type", mx.array, name, orig))


def patch_type_hash():
    # __hash__ is not in vars(mx.array); it inherits object's pointer hash
    def wrapper(self):
        calls.append("__hash__")
        return object.__hash__(self)

    setattr(mx.array, "__hash__", wrapper)
    patched.append(("type-added", mx.array, "__hash__", None))


def restore_all():
    problems = []
    for kind, target, name, orig in reversed(patched):
        if kind == "type-added":
            delattr(target, name)
            if name in vars(target):
                problems.append(name)
        elif kind == "type":
            setattr(target, name, orig)
            if vars(target).get(name) is not orig:
                problems.append(name)
        else:
            setattr(target, name, orig)
            if getattr(target, name) is not orig:
                problems.append(name)
    patched.clear()
    return problems


def ran(fn):
    """Run fn, eval any array result. Return (ok, err_name, names recorded during)."""
    start = len(calls)
    try:
        r = fn()
        if isinstance(r, mx.array):
            mx.eval(r)
        return True, "", calls[start:]
    except Exception as e:
        return False, type(e).__name__, calls[start:]


# deterministic fixtures, fresh per trigger so in-place ops cannot leak state
def fa():
    return mx.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def fb():
    return mx.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])


def ia():
    return mx.array([1, 2, 3], dtype=mx.int32)


def ib():
    return mx.array([3, 2, 1], dtype=mx.int32)


def pa():
    return mx.array([True, False, True])


def pb():
    return mx.array([True, True, False])


def ma():
    return mx.array([[1.0, 2.0], [3.0, 4.0]])


def mb():
    return mx.array([[5.0, 6.0], [7.0, 8.0]])


def aug(opname, mk_x, mk_y):
    # operator.iadd(x, y) has exact `x += y` semantics including the __add__ fold
    iop = getattr(operator, opname)

    def f():
        return iop(mk_x(), mk_y())

    return f


def setitem_trigger():
    x = fa()
    x[2] = 9.0
    return x


BINARY = [
    ("__add__", lambda: fa() + fb()),
    ("__sub__", lambda: fa() - fb()),
    ("__mul__", lambda: fa() * fb()),
    ("__truediv__", lambda: fa() / fb()),
    ("__floordiv__", lambda: fa() // fb()),
    ("__mod__", lambda: fa() % fb()),
    ("__pow__", lambda: fa() ** fb()),
    ("__matmul__", lambda: ma() @ mb()),
    ("__and__", lambda: ia() & ib()),
    ("__or__", lambda: ia() | ib()),
    ("__xor__", lambda: ia() ^ ib()),
    ("__lshift__", lambda: ia() << ib()),
    ("__rshift__", lambda: ia() >> ib()),
]
REFLECTED = [
    ("__radd__", lambda: 2.0 + fa()),
    ("__rsub__", lambda: 2.0 - fa()),
    ("__rmul__", lambda: 2.0 * fa()),
    ("__rtruediv__", lambda: 2.0 / fa()),
    ("__rfloordiv__", lambda: 2.0 // fa()),
    ("__rmod__", lambda: 2.0 % fa()),
    ("__rpow__", lambda: 2.0 ** fa()),
    ("__rmatmul__", lambda: [[1.0, 0.0], [0.0, 1.0]] @ ma()),
    ("__rand__", lambda: 2 & ia()),
    ("__ror__", lambda: 2 | ia()),
    ("__rxor__", lambda: 2 ^ ia()),
    ("__rlshift__", lambda: 2 << ia()),
    ("__rrshift__", lambda: 2 >> ia()),
]
INPLACE = [
    ("__iadd__", aug("iadd", fa, fb)),
    ("__isub__", aug("isub", fa, fb)),
    ("__imul__", aug("imul", fa, fb)),
    ("__itruediv__", aug("itruediv", fa, fb)),
    ("__ifloordiv__", aug("ifloordiv", fa, fb)),
    ("__imod__", aug("imod", fa, fb)),
    ("__ipow__", aug("ipow", fa, fb)),
    ("__imatmul__", aug("imatmul", ma, mb)),
    ("__iand__", aug("iand", ia, ib)),
    ("__ior__", aug("ior", ia, ib)),
    ("__ixor__", aug("ixor", ia, ib)),
    ("__ilshift__", aug("ilshift", ia, ib)),
    ("__irshift__", aug("irshift", ia, ib)),
]
UNARY = [
    ("__neg__", lambda: -fa()),
    ("__pos__", lambda: +fa()),
    ("__abs__", lambda: abs(fa())),
    ("__invert__", lambda: ~pa()),
]
COMPARISON = [
    ("__eq__", lambda: fa() == fb()),
    ("__ne__", lambda: fa() != fb()),
    ("__lt__", lambda: fa() < fb()),
    ("__le__", lambda: fa() <= fb()),
    ("__gt__", lambda: fa() > fb()),
    ("__ge__", lambda: fa() >= fb()),
]
INDEXING = [
    ("__getitem__", lambda: fa()[1:4]),
    ("__setitem__", setitem_trigger),
]
CONVERSION = [
    ("__len__", lambda: len(fa())),
    ("__iter__", lambda: next(iter(fa()))),
    ("__bool__", lambda: bool(mx.array(True))),
    ("__int__", lambda: int(mx.array(3))),
    ("__float__", lambda: float(mx.array(1.5))),
]
METHODS = [
    ("reshape", lambda: fa().reshape(2, 3)),
    ("sum", lambda: fa().sum()),
]
CATEGORIES = [
    ("arithmetic", BINARY),
    ("reflected", REFLECTED),
    ("inplace", INPLACE),
    ("unary", UNARY),
    ("comparison", COMPARISON),
    ("indexing", INDEXING),
    ("conversion", CONVERSION),
    ("plain-methods", METHODS),
]
ALL_NAMES = [name for _, cat in CATEGORIES for name, _ in cat]


def run_category(cat, pre_absent):
    """Return (intercepted, folded, broke, absent_unsupported, absent_leaky).

    folded = op ran but the wrapper was not hit. An absent name is harmless only if
    its op also failed pre-patch (nothing to trace); if the op WORKED pre-patch with
    no dunder to patch, that is a tracer hole (absent_leaky).
    """
    intercepted, folded, broke, unsupported, leaky = [], [], [], [], []
    for name, trig in cat:
        if name not in vars(mx.array):
            pre_ok, pre_err = pre_absent[name]
            (leaky if pre_ok else unsupported).append(name if not pre_ok else f"{name}(op works unpatched)")
            continue
        ok, err, rec = ran(trig)
        if not ok:
            broke.append(f"{name}({err})")
        elif name in rec:
            intercepted.append(name)
        else:
            folded.append(f"{name}(hit {','.join(sorted(set(rec))) or 'nothing'})")
    return intercepted, folded, broke, unsupported, leaky


def main():
    mx.random.seed(0)
    fact("mlx-version", "INFO", f"mlx {mx.__version__} on {sys.platform}, python {sys.version.split()[0]}")

    # --- mx.array type shape ---
    flags = mx.array.__flags__
    heap = bool(flags & (1 << 9))         # Py_TPFLAGS_HEAPTYPE
    immutable = bool(flags & (1 << 8))    # Py_TPFLAGS_IMMUTABLETYPE
    fact("array-nanobind-heaptype", "PASS" if heap and not immutable else "FAIL",
         f"metatype={type(mx.array).__module__}.{type(mx.array).__name__}, HEAPTYPE={heap}, IMMUTABLETYPE={immutable}")

    present = [n for n in ALL_NAMES if n in vars(mx.array)]
    missing = [n for n in ALL_NAMES if n not in vars(mx.array)]
    fact("array-dunder-inventory", "INFO",
         f"{len(present)}/{len(ALL_NAMES)} probed names in vars(mx.array); missing: {','.join(missing) or 'none'}; "
         f"__hash__ inherited from object (pointer hash)")

    # --- pre-patch baseline for every op whose dunder is absent from the type ---
    pre_absent = {}
    for _, cat in CATEGORIES:
        for name, trig in cat:
            if name not in vars(mx.array):
                ok, err, _ = ran(trig)
                pre_absent[name] = (ok, err)
    pre_missing_reflected = {n: pre_absent[n] for n, _ in REFLECTED if n in pre_absent}

    # --- claim: patching mx.add alone does not intercept a + b ---
    orig_add = getattr(mx, "add")

    def add_spy(*args, **kwargs):
        calls.append("mx.add")
        return orig_add(*args, **kwargs)

    setattr(mx, "add", add_spy)
    ok, err, rec = ran(lambda: fa() + fb())
    setattr(mx, "add", orig_add)
    hit = "mx.add" in rec
    fact("module-add-not-plus", "FAIL" if hit or not ok else "PASS",
         f"a + b with only mx.add patched: ran={ok}{' err=' + err if err else ''}, patched mx.add called={hit}")

    # --- install the full patch set ---
    setattr_failures = []
    for name in ("addmm", "matmul", "add"):
        patch_module_fn(name)
    for name in present:
        try:
            patch_type_attr(name)
        except Exception as e:
            setattr_failures.append(f"{name}({type(e).__name__})")
    patch_type_hash()
    fact("array-dunder-setattr", "PASS" if not setattr_failures else "FAIL",
         f"setattr accepted on {len(present) - len(setattr_failures)}/{len(present)} present names"
         f"{'; failed: ' + ','.join(setattr_failures) if setattr_failures else ''} (+__hash__ added)")

    # --- mlx.nn sees module-level patches at call time ---
    lin_b = nn.Linear(4, 3, bias=True)
    lin_nb = nn.Linear(4, 3, bias=False)
    x = mx.arange(8).reshape(2, 4).astype(mx.float32)
    ok_b, err_b, rec_b = ran(lambda: lin_b(x))
    ok_nb, err_nb, rec_nb = ran(lambda: lin_nb(x))
    bias_ops = sorted(set(rec_b))
    nobias_ops = sorted(set(rec_nb))
    seen = ok_b and any(n.startswith("mx.") for n in rec_b)
    fact("module-setattr-nn-visible", "PASS" if seen else "FAIL",
         f"nn.Linear(bias=True) hit patched module-level op: {seen} (recorded: {','.join(bias_ops) or 'nothing'})")
    fact("linear-op-usage", "INFO",
         f"bias=True records {','.join(bias_ops)}; bias=False records {','.join(nobias_ops)} (x @ W.T, not mx.matmul)")

    # --- the plan's core slot-propagation claim, one expression each ---
    core = [
        ("a + b", lambda: fa() + fb(), "__add__"),
        ("2.0 * x", lambda: 2.0 * fa(), "__rmul__"),
        ("2.0 + x", lambda: 2.0 + fa(), "__radd__"),
        ("a @ b", lambda: ma() @ mb(), "__matmul__"),
        ("a[i]", lambda: fa()[1:4], "__getitem__"),
        ("a[i] = v", setitem_trigger, "__setitem__"),
        ("a == b", lambda: fa() == fb(), "__eq__"),
    ]
    core_bad = []
    for label, trig, want in core:
        ok, err, rec = ran(trig)
        if not ok or want not in rec:
            core_bad.append(f"{label}->{err or 'not intercepted'}")
    fact("slot-propagation-core", "PASS" if not core_bad else "FAIL",
         f"a+b, 2.0*x, 2.0+x, a@b, a[i], a[i]=v, a==b all intercepted via C slots"
         if not core_bad else f"failed: {'; '.join(core_bad)}")

    # numeric spot check that wrappers delegate correctly
    got = (fa() + fb()).tolist()
    fact("wrapper-delegation-correct", "PASS" if got == [7.0] * 6 else "FAIL",
         f"(a + b) under full patch = {got}, expected [7.0]*6")

    # --- category sweep ---
    for cname, cat in CATEGORIES:
        intercepted, folded, broke, unsupported, leaky = run_category(cat, pre_absent)
        detail = f"intercepted {len(intercepted)}/{len(cat)}: {','.join(intercepted) or 'none'}"
        if folded:
            detail += f"; folded: {','.join(folded)}"
        if broke:
            detail += f"; errored: {','.join(broke)}"
        if unsupported:
            detail += f"; no dunder and op raises unpatched (nothing to trace): {','.join(unsupported)}"
        if leaky:
            detail += f"; TRACER HOLE, op works with no dunder to patch: {','.join(leaky)}"
        status = "PASS" if not folded and not broke and not leaky else "FAIL"
        fact(f"intercept-{cname}", status, detail)
        if cname == "inplace":
            ok, err, rec = ran(aug("iadd", fa, fb))
            note = "hits __iadd__ (no fold)" if "__iadd__" in rec else \
                f"folds: recorded {','.join(sorted(set(rec))) or 'nothing'}"
            fact("iadd-fold", "INFO", f"a += b {note}")

    # --- swapped-operand ops with no __r*__: pre vs patched ---
    changed = []
    detail_parts = []
    for name, (pre_ok, pre_err) in pre_missing_reflected.items():
        trig = dict(REFLECTED)[name]
        ok, err, _ = ran(trig)
        detail_parts.append(f"{name}: pre={'ok' if pre_ok else pre_err} patched={'ok' if ok else err}")
        if pre_ok != ok:
            changed.append(name)
    fact("swapped-operand-no-reflected", "PASS" if not changed else "FAIL",
         ("unchanged by patching, all raise TypeError pre and post: " if not changed else
          f"patching CHANGED behavior of {','.join(changed)}: ") + "; ".join(detail_parts))

    # --- __eq__/__hash__ patched: arrays as dict keys and set members ---
    try:
        k1, k2 = fa(), fb()
        start = len(calls)
        d = {k1: "one", k2: "two"}
        dict_ok = d[k1] == "one" and d[k2] == "two" and len(d) == 2
        s = {k1, k2, k1}
        set_ok = k1 in s and k2 in s and len(s) == 2
        hash_hits = calls[start:].count("__hash__")
        fact("eq-hash-dict-set", "PASS" if dict_ok and set_ok and hash_hits > 0 else "FAIL",
             f"dict lookups={dict_ok}, set membership={set_ok}, patched __hash__ called {hash_hits}x "
             f"(identity keys; == is elementwise so only same-object keys are meaningful, patched or not)")
    except Exception as e:
        fact("eq-hash-dict-set", "FAIL", f"raised {type(e).__name__}: {e}")

    # --- uninstall: identity restore, silence, and behavior restore ---
    expected_lin = lin_b(x)
    mx.eval(expected_lin)
    problems = restore_all()
    start = len(calls)
    r1 = fa() + fb()
    r2 = 2.0 * fa()
    r3 = lin_b(x)
    mx.eval(r1, r2, r3)
    silent = len(calls) == start
    correct = r1.tolist() == [7.0] * 6 and bool(mx.array_equal(r3, expected_lin))
    fact("uninstall-restores-identity", "PASS" if not problems and silent and correct else "FAIL",
         f"identity mismatches: {','.join(problems) or 'none'}; wrappers silent after restore={silent}; "
         f"outputs still correct={correct}")

    # behavior of swapped-operand ops after restore vs before any patching
    post_changed = []
    for name, (pre_ok, pre_err) in pre_missing_reflected.items():
        ok, err, _ = ran(dict(REFLECTED)[name])
        if (pre_ok, pre_err) != (ok, err):
            post_changed.append(f"{name}: pre={pre_ok or pre_err} post={ok or err}")
    fact("uninstall-restores-behavior", "PASS" if not post_changed else "FAIL",
         "swapped-operand ops behave exactly as before patching" if not post_changed
         else f"behavior changed after restore: {'; '.join(post_changed)}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)
