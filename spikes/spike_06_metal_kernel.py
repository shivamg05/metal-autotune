"""Spike 06: mx.fast.metal_kernel facts from plan 5.10, proved by experiment on this machine.

Output contract: one line per fact, `FACT <slug>: PASS|FAIL|INFO - <detail>`.
Exit 0 if the script ran to completion (FAILs included); nonzero only on script breakage.
"""

import os
import re
import struct
import subprocess
import sys
import tempfile

import mlx.core as mx

SEED = 0


def fact(slug, status, detail):
    sys.stdout.flush()
    print(f"FACT {slug}: {status} - {detail}")
    sys.stdout.flush()


def capture_fd1(fn):
    # metal_kernel verbose prints from C++ to fd 1; redirect the fd, not sys.stdout
    sys.stdout.flush()
    saved = os.dup(1)
    with tempfile.TemporaryFile(mode="w+") as f:
        os.dup2(f.fileno(), 1)
        try:
            fn()
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)
        f.seek(0)
        return f.read()


def check_construction_signature():
    doc = mx.fast.metal_kernel.__doc__ or ""
    sig_line = doc.strip().splitlines()[0]
    expected = [
        "name: str",
        "input_names:",
        "output_names:",
        "source: str",
        "header: str = ''",
        "ensure_row_contiguous: bool = True",
        "atomic_outputs: bool = False",
        "compile_options:",
    ]
    missing = [e for e in expected if e not in sig_line]
    if missing:
        fact("construction_signature", "FAIL", f"doc signature missing {missing}; got: {sig_line}")
    else:
        fact("construction_signature", "PASS", f"doc signature matches plan 5.10: {sig_line}")


def check_call_keyword_only():
    src = "uint elem = thread_position_in_grid.x;\nout[elem] = static_cast<T>(inp[elem]) * T(2);\n"
    k = mx.fast.metal_kernel(name="spike06_call", input_names=["inp"], output_names=["out"], source=src)
    a = mx.arange(8, dtype=mx.float32)
    try:
        k([a], [(8,)], [mx.float32], (8, 1, 1), (8, 1, 1))
        positional_rejected = False
    except TypeError:
        positional_rejected = True
    out = k(
        inputs=[a],
        output_shapes=[(8,)],
        output_dtypes=[mx.float32],
        grid=(8, 1, 1),
        threadgroup=(8, 1, 1),
        template=[("T", mx.float32)],
        init_value=0.0,
        verbose=False,
    )
    mx.eval(out)
    kw_ok = bool(mx.all(out[0] == a * 2).item())
    if positional_rejected and kw_ok:
        fact("call_keyword_only", "PASS",
             "positional call raises TypeError; keyword call with inputs/output_shapes/output_dtypes/"
             "grid/threadgroup/template/init_value/verbose runs correctly")
    else:
        fact("call_keyword_only", "FAIL",
             f"positional_rejected={positional_rejected} keyword_call_correct={kw_ok}")


def check_grid_total_threads():
    # 50 threads cannot be expressed as whole 32-wide threadgroups, so
    # threads_per_grid.x == 50 is only possible under dispatchThreads semantics
    src = (
        "uint elem = thread_position_in_grid.x;\n"
        "out[elem] = (float)elem;\n"
        "tpg[0] = (float)threads_per_grid.x;\n"
    )
    k = mx.fast.metal_kernel(name="spike06_grid", input_names=["inp"], output_names=["out", "tpg"], source=src)
    a = mx.zeros((50,), dtype=mx.float32)
    out, tpg = k(
        inputs=[a],
        output_shapes=[(50,), (1,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(50, 1, 1),
        threadgroup=(32, 1, 1),
        init_value=float("nan"),
    )
    mx.eval(out, tpg)
    ids_ok = bool(mx.all(out == mx.arange(50, dtype=mx.float32)).item())
    tpg_val = tpg[0].item()
    if ids_ok and tpg_val == 50.0:
        fact("grid_total_threads", "PASS",
             f"grid=(50,1,1) tg=(32,1,1): thread ids cover 0..49 and threads_per_grid.x={tpg_val:.0f} "
             "(dispatchThreads semantics, grid is TOTAL THREADS)")
    else:
        fact("grid_total_threads", "FAIL",
             f"ids_cover_0_49={ids_ok} threads_per_grid.x={tpg_val} (would be 1600 under dispatchThreadgroups)")


def build_broken(name, body):
    k = mx.fast.metal_kernel(name=name, input_names=["inp"], output_names=["out"], source=body)
    a = mx.zeros((8,), dtype=mx.float32)
    return k(inputs=[a], output_shapes=[(8,)], output_dtypes=[mx.float32], grid=(8, 1, 1), threadgroup=(8, 1, 1))


def check_compile_error_and_offset():
    # error on body line 3
    bad3 = "uint elem = thread_position_in_grid.x;\nout[elem] = inp[elem];\nSPIKE06_BROKEN_A;\n"
    try:
        out = build_broken("spike06_broken_a", bad3)
    except Exception as e:
        fact("compile_error_at_eval", "FAIL", f"error at construction/call time: {type(e).__name__}: {e}")
        return
    try:
        mx.eval(out)
        fact("compile_error_at_eval", "FAIL", "broken kernel evaluated without raising")
        return
    except RuntimeError as e:
        msg = str(e)
    except Exception as e:
        fact("compile_error_at_eval", "FAIL", f"eval raised {type(e).__name__}, not RuntimeError: {e}")
        return
    m = re.search(r":(\d+):\d+: error:", msg)
    if not m:
        fact("compile_error_at_eval", "PASS",
             "RuntimeError at mx.eval, none at construction or call; no line number in message")
        fact("error_line_offset", "INFO", f"no parsable line number; message head: {msg[:120]!r}")
        return
    line3 = int(m.group(1))
    fact("compile_error_at_eval", "PASS",
         f"RuntimeError at mx.eval only; message reports line {line3} for an error on body line 3")

    # same signature shape, error on body line 6: offset must be constant
    bad6 = ("uint elem = thread_position_in_grid.x;\n" "out[elem] = inp[elem];\n" "float x = 1.0f;\n"
            "float y = 2.0f;\n" "float z = x + y;\n" "SPIKE06_BROKEN_B;\n")
    line6 = None
    try:
        mx.eval(build_broken("spike06_broken_b", bad6))
    except RuntimeError as e:
        m2 = re.search(r":(\d+):\d+: error:", str(e))
        line6 = int(m2.group(1)) if m2 else None

    # one extra input adds one signature line: does the offset shift?
    bad_2in = "uint elem = thread_position_in_grid.x;\nout[elem] = inp[elem] + inp2[elem];\nSPIKE06_BROKEN_C;\n"
    k2 = mx.fast.metal_kernel(name="spike06_broken_c", input_names=["inp", "inp2"], output_names=["out"],
                              source=bad_2in)
    a = mx.zeros((8,), dtype=mx.float32)
    line_2in = None
    try:
        mx.eval(k2(inputs=[a, a], output_shapes=[(8,)], output_dtypes=[mx.float32],
                   grid=(8, 1, 1), threadgroup=(8, 1, 1)))
    except RuntimeError as e:
        m3 = re.search(r":(\d+):\d+: error:", str(e))
        line_2in = int(m3.group(1)) if m3 else None

    off3 = line3 - 3
    off6 = (line6 - 6) if line6 is not None else None
    off_2in = (line_2in - 3) if line_2in is not None else None
    constant = off6 == off3
    fact("error_line_offset", "INFO",
         f"offset={off3} lines (reported minus body line) for 1-in/1-out kernel, "
         f"constant across error positions={constant}; with 2 inputs offset={off_2in} "
         "(offset depends on generated signature length, so the harness must compute it per kernel)")


def check_math_mode():
    results = []
    for mode in ("safe", "relaxed", "fast"):
        src = "uint elem = thread_position_in_grid.x;\nout[elem] = metal::exp(inp[elem]);\n"
        k = mx.fast.metal_kernel(name=f"spike06_math_{mode}", input_names=["inp"], output_names=["out"],
                                 source=src, compile_options={"math_mode": mode})
        a = mx.arange(8, dtype=mx.float32) * 0.1
        try:
            out = k(inputs=[a], output_shapes=[(8,)], output_dtypes=[mx.float32],
                    grid=(8, 1, 1), threadgroup=(8, 1, 1))
            mx.eval(out)
            ok = bool(mx.allclose(out[0], mx.exp(a)).item())
            results.append((mode, "ok" if ok else "wrong-values"))
        except Exception as e:
            results.append((mode, f"{type(e).__name__}"))
    detail = ", ".join(f"{m}={r}" for m, r in results)
    if all(r == "ok" for _, r in results):
        fact("math_mode_options", "PASS", f"compile_options math_mode accepts and runs: {detail}")
    else:
        fact("math_mode_options", "FAIL", detail)


def check_init_value_nan():
    src = "uint elem = thread_position_in_grid.x;\nif (elem < 512) { out[elem] = inp[elem]; }\n"
    k = mx.fast.metal_kernel(name="spike06_poison", input_names=["inp"], output_names=["out"], source=src)
    mx.random.seed(SEED)
    a = mx.random.normal((1024,)).astype(mx.float32)
    mx.eval(a)
    runs_ok = 0
    for _ in range(5):
        # dirty the buffer pool so a recycled non-NaN buffer would be visible
        garbage = mx.full((1024,), 7.0, dtype=mx.float32)
        mx.eval(garbage)
        del garbage
        out = k(inputs=[a], output_shapes=[(1024,)], output_dtypes=[mx.float32],
                grid=(1024, 1, 1), threadgroup=(256, 1, 1), init_value=float("nan"))
        mx.eval(out)
        written_ok = bool(mx.all(out[0][:512] == a[:512]).item())
        unwritten_nan = bool(mx.all(mx.isnan(out[0][512:])).item())
        if written_ok and unwritten_nan:
            runs_ok += 1
    if runs_ok == 5:
        fact("init_value_nan_poison", "PASS",
             "init_value=nan: unwritten half reads back all-NaN and written half is exact, 5/5 runs "
             "with the pool pre-dirtied")
    else:
        fact("init_value_nan_poison", "FAIL", f"only {runs_ok}/5 runs showed NaN poison in the unwritten half")


def check_shape_injection():
    src = (
        "uint elem = thread_position_in_grid.x;\n"
        "if (elem == 0) {\n"
        "  meta[0] = (float)inp_ndim;\n"
        "  meta[1] = (float)inp_shape[0];\n"
        "  meta[2] = (float)inp_shape[1];\n"
        "  meta[3] = (float)inp_strides[0];\n"
        "  meta[4] = (float)inp_strides[1];\n"
        "}\n"
    )
    k = mx.fast.metal_kernel(name="spike06_shapes", input_names=["inp"], output_names=["meta"], source=src)
    a = mx.arange(12, dtype=mx.float32).reshape(3, 4)
    mx.eval(a)
    generated = {}

    def run():
        out = k(inputs=[a], output_shapes=[(5,)], output_dtypes=[mx.float32],
                grid=(1, 1, 1), threadgroup=(1, 1, 1), verbose=True)
        mx.eval(out)
        generated["meta"] = [v.item() for v in out[0]]

    cap = capture_fd1(run)
    vals = generated.get("meta")
    expect = [2.0, 3.0, 4.0, 4.0, 1.0]
    injected_in_sig = all(s in cap for s in ("inp_shape", "inp_strides", "inp_ndim"))
    if vals == expect and injected_in_sig:
        fact("shape_buffer_injection", "PASS",
             f"referencing inp_shape/inp_strides/inp_ndim auto-injects buffers (present in generated "
             f"signature) and values are correct: ndim/shape/strides={vals}")
    else:
        fact("shape_buffer_injection", "FAIL",
             f"values={vals} expected={expect}; names in generated signature={injected_in_sig}")

    # a kernel that mentions no built-ins and no shape buffers must get none in its signature
    src_bare = "out[0] = inp[0] + 1.0f;\n"
    k2 = mx.fast.metal_kernel(name="spike06_bare", input_names=["inp"], output_names=["out"], source=src_bare)

    def run2():
        out = k2(inputs=[a], output_shapes=[(1,)], output_dtypes=[mx.float32],
                 grid=(1, 1, 1), threadgroup=(1, 1, 1), verbose=True)
        mx.eval(out)

    cap2 = capture_fd1(run2)
    bare_clean = not any(s in cap2 for s in ("thread_position_in_grid", "threads_per_grid",
                                             "inp_shape", "inp_strides", "inp_ndim"))
    mentioned_present = "thread_position_in_grid" in cap and "[[thread_position_in_grid]]" in cap
    if bare_clean and mentioned_present:
        fact("grid_builtins_conditional", "PASS",
             "generated signature includes thread_position_in_grid only when the source mentions it; "
             "a bare kernel's signature has no grid built-ins and no shape buffers")
    else:
        fact("grid_builtins_conditional", "FAIL",
             f"bare_kernel_signature_clean={bare_clean} mentioned_builtin_present={mentioned_present}")


def check_row_contiguous():
    a = mx.arange(12, dtype=mx.float32).reshape(3, 4)
    at = mx.transpose(a)
    mx.eval(a, at)
    logical = mx.reshape(at, (12,))  # row-major flatten of the transposed view
    raw = mx.reshape(a, (12,))       # underlying buffer order
    mx.eval(logical, raw)
    src = "uint elem = thread_position_in_grid.x;\nout[elem] = inp[elem];\n"

    k_true = mx.fast.metal_kernel(name="spike06_rc_true", input_names=["inp"], output_names=["out"],
                                  source=src, ensure_row_contiguous=True)
    out_t = k_true(inputs=[at], output_shapes=[(12,)], output_dtypes=[mx.float32],
                   grid=(12, 1, 1), threadgroup=(12, 1, 1))
    mx.eval(out_t)
    true_sees_logical = bool(mx.all(out_t[0] == logical).item())
    if true_sees_logical:
        fact("row_contiguous_true_copy", "PASS",
             f"ensure_row_contiguous=True on a transposed view: kernel sees a contiguous row-major copy, "
             f"flat read equals the transposed logical order {[int(v.item()) for v in out_t[0]]}")
    else:
        fact("row_contiguous_true_copy", "FAIL",
             f"flat read {[v.item() for v in out_t[0]]} != transposed logical order")

    k_false = mx.fast.metal_kernel(name="spike06_rc_false", input_names=["inp"], output_names=["out"],
                                   source=src, ensure_row_contiguous=False)
    out_f = k_false(inputs=[at], output_shapes=[(12,)], output_dtypes=[mx.float32],
                    grid=(12, 1, 1), threadgroup=(12, 1, 1))
    mx.eval(out_f)
    false_wrong = not bool(mx.all(out_f[0] == logical).item())
    false_sees_raw = bool(mx.all(out_f[0] == raw).item())
    if false_wrong:
        fact("row_contiguous_false_wrong", "PASS",
             f"ensure_row_contiguous=False with naive flat indexing is silently wrong on the transposed "
             f"view (kernel read the raw untransposed buffer: {false_sees_raw}); no error was raised")
    else:
        fact("row_contiguous_false_wrong", "FAIL",
             "naive flat indexing on a transposed view still produced the logical order under "
             "ensure_row_contiguous=False")


def check_same_name_one_process():
    src1 = "uint elem = thread_position_in_grid.x;\nout[elem] = 1.0f;\n"
    src2 = "uint elem = thread_position_in_grid.x;\nout[elem] = 2.0f;\n"
    k1 = mx.fast.metal_kernel(name="spike06_samename", input_names=["inp"], output_names=["out"], source=src1)
    k2 = mx.fast.metal_kernel(name="spike06_samename", input_names=["inp"], output_names=["out"], source=src2)
    a = mx.zeros((4,), dtype=mx.float32)

    def run(k):
        out = k(inputs=[a], output_shapes=[(4,)], output_dtypes=[mx.float32],
                grid=(4, 1, 1), threadgroup=(4, 1, 1))
        mx.eval(out)
        return out[0][0].item()

    seq = [run(k1), run(k2), run(k1), run(k2)]
    if seq == [1.0, 2.0, 1.0, 2.0]:
        fact("same_name_one_process", "PASS",
             f"two kernels with the same name and different source behave independently in one process, "
             f"interleaved results {seq}")
    else:
        fact("same_name_one_process", "FAIL", f"interleaved results {seq}, expected [1.0, 2.0, 1.0, 2.0]")


CHILD_SCRIPT = """
import sys
import mlx.core as mx
val = sys.argv[1]
src = "uint elem = thread_position_in_grid.x;\\nout[elem] = " + val + "f;\\n"
k = mx.fast.metal_kernel(name="spike06_xproc", input_names=["inp"], output_names=["out"], source=src)
a = mx.zeros((16,), dtype=mx.float32)
out = k(inputs=[a], output_shapes=[(16,)], output_dtypes=[mx.float32],
        grid=(16, 1, 1), threadgroup=(16, 1, 1))
mx.eval(out)
print("RESULT", out[0][0].item())
"""


def check_cross_process_cache():
    # same kernel name, sources differing only in one constant: worst case for a
    # name-keyed on-disk cache serving a stale binary
    magics = ["111.0", "222.0", "111.0", "333.0"]
    got = []
    for magic in magics:
        r = subprocess.run([sys.executable, "-c", CHILD_SCRIPT, magic],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            fact("cross_process_cache", "FAIL",
                 f"child for {magic} failed rc={r.returncode}: {r.stderr.strip()[:200]}")
            return
        m = re.search(r"RESULT ([0-9.]+)", r.stdout)
        got.append(m.group(1) if m else "no-output")
    expected = [f"{float(m)}" for m in magics]
    if got == expected:
        fact("cross_process_cache", "PASS",
             f"4 sequential processes, same kernel name, source differs only in a constant: each read "
             f"back its own value {got}; on-disk shader cache served no stale binary")
    else:
        fact("cross_process_cache", "FAIL",
             f"STALE BINARY HAZARD: expected {expected}, got {got}")


def check_float_atomics():
    src = ("uint elem = thread_position_in_grid.x;\n"
           "atomic_fetch_add_explicit(&out[0], inp[elem], memory_order_relaxed);\n")
    k = mx.fast.metal_kernel(name="spike06_atomic", input_names=["inp"], output_names=["out"],
                             source=src, atomic_outputs=True)
    mx.random.seed(SEED)
    a = mx.random.normal((1 << 20,)).astype(mx.float32)
    mx.eval(a)
    bits = set()
    vals = []
    for _ in range(10):
        out = k(inputs=[a], output_shapes=[(1,)], output_dtypes=[mx.float32],
                grid=(a.size, 1, 1), threadgroup=(256, 1, 1), init_value=0.0)
        mx.eval(out)
        v = out[0][0].item()
        vals.append(v)
        bits.add(struct.pack("<f", v))
    spread = max(vals) - min(vals)
    if len(bits) > 1:
        fact("float_atomics_nondeterminism", "PASS",
             f"1M-element atomic float sum on fixed input: {len(bits)}/10 distinct bit patterns, "
             f"spread {spread:.6g} (nondeterministic across runs, as the plan expects)")
    else:
        fact("float_atomics_nondeterminism", "FAIL",
             f"10/10 runs bit-identical ({vals[0]!r}); plan expects float atomics to be nondeterministic")


def main():
    fact("mlx_version", "INFO", f"mlx {mx.__version__} on {sys.platform}, default device {mx.default_device()}")
    check_construction_signature()
    check_call_keyword_only()
    check_grid_total_threads()
    check_compile_error_and_offset()
    check_math_mode()
    check_init_value_nan()
    check_shape_injection()
    check_row_contiguous()
    check_same_name_one_process()
    check_cross_process_cache()
    check_float_atomics()


if __name__ == "__main__":
    main()
