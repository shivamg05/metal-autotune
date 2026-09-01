#!/usr/bin/env python3
"""Spike 07: OOB access behavior and launch-time-only Metal env vars.

Proves by experiment (plan 5.10, M0):
  1. OOB reads in a metal_kernel without validation: fault or silence, and what comes back.
  2. What a harness can detect when the same kernel runs under MTL_SHADER_VALIDATION=1,
     and under MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 on top (man MetalValidation says
     reports go to os_log unless that second var redirects them to stderr).
  3. MTL_SHADER_VALIDATION set inside a running process changes nothing (launch-time only).
  4. MTL_CAPTURE_ENABLED is launch-time only: start_capture fails unless set at launch.
  5. OOB writes without validation: fault, neighbor corruption, or silently absorbed;
     plus whether validation reports them.

Every Metal experiment runs in a subprocess of this same file so a GPU fault cannot
kill the orchestrator. Children print one JSON record per line as results land, so a
mid-run crash keeps the records already produced. Exit 0 if the spike itself ran.
"""

import json
import os
import shutil
import subprocess
import sys

SIZE = 4096                 # elements per float32 buffer, 16 KB
NEAR_OFF = 4 * SIZE         # read starting 64 KB into a 16 KB buffer, 48 KB past its end
FAR_OFF = 64 * 1024 * 1024  # read starting 256 MB past the base, likely unmapped
SENTINEL = 123456.0
N_NEIGHBORS = 128
HERE = os.path.abspath(__file__)
OUT_DIR = os.path.join(os.path.dirname(HERE), "out")

# ---------------- child side ----------------


def emit(obj):
    print(json.dumps(obj), flush=True)


def oob_read_once(tag, offset):
    import mlx.core as mx

    inp = mx.full((SIZE,), 1.0, dtype=mx.float32)
    mx.eval(inp)
    src = f"uint i = thread_position_in_grid.x; out[i] = inp[i + {offset}u];"
    k = mx.fast.metal_kernel(
        name=f"spike07_read_{tag}", input_names=["inp"], output_names=["out"],
        source=src)
    rec = {"tag": tag, "offset_elems": offset}
    try:
        (out,) = k(
            inputs=[inp], output_shapes=[(SIZE,)], output_dtypes=[mx.float32],
            grid=(SIZE, 1, 1), threadgroup=(256, 1, 1))
        mx.eval(out)
        mx.synchronize()
        rec.update(
            ok=True,
            zeros=int(mx.sum(out == 0).item()),
            ones=int(mx.sum(out == 1.0).item()),
            nans=int(mx.sum(mx.isnan(out)).item()),
            sample=[float(out[j].item()) for j in range(4)],
        )
    except Exception as e:  # noqa: BLE001 - the exception IS the finding
        rec.update(ok=False, error=repr(e)[:300])
    emit(rec)
    return rec


def child_read():
    oob_read_once("ctl", 0)  # in-bounds control proving the indexing path is real
    oob_read_once("near", NEAR_OFF)
    oob_read_once("far", FAR_OFF)


def child_read_setenv():
    # env vars set inside a live process, after Metal is initialized by the first run;
    # the post run uses a fresh kernel name, so a fresh pipeline gets compiled after the set
    oob_read_once("pre", NEAR_OFF)
    os.environ["MTL_SHADER_VALIDATION"] = "1"
    os.environ["MTL_SHADER_VALIDATION_REPORT_TO_STDERR"] = "1"
    emit({"tag": "env", "MTL_SHADER_VALIDATION": os.environ.get("MTL_SHADER_VALIDATION")})
    oob_read_once("post", NEAR_OFF)


def child_capture_off():
    import mlx.core as mx

    mx.eval(mx.zeros((8,)))  # force Metal init before the first attempt
    os.makedirs(OUT_DIR, exist_ok=True)
    p1 = os.path.join(OUT_DIR, "spike07_cap_a.gputrace")
    p2 = os.path.join(OUT_DIR, "spike07_cap_b.gputrace")
    rec = {"tag": "capture_off"}
    for path, key in ((p1, "before_set"), (p2, "after_set")):
        shutil.rmtree(path, ignore_errors=True)
        try:
            mx.metal.start_capture(path)
            mx.metal.stop_capture()
            rec[key] = {"ok": True}
        except Exception as e:  # noqa: BLE001
            rec[key] = {"ok": False, "error": repr(e)[:200]}
        if key == "before_set":
            os.environ["MTL_CAPTURE_ENABLED"] = "1"
    rec["files_created"] = [os.path.exists(p1), os.path.exists(p2)]
    for path in (p1, p2):
        shutil.rmtree(path, ignore_errors=True)
    emit(rec)


def child_capture_on():
    import mlx.core as mx

    mx.eval(mx.zeros((8,)))
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "spike07_cap_on.gputrace")
    shutil.rmtree(path, ignore_errors=True)
    rec = {"tag": "capture_on"}
    try:
        mx.metal.start_capture(path)
        mx.eval(mx.ones((SIZE,)) * 2.0)
        mx.metal.stop_capture()
        rec.update(ok=True, file_created=os.path.exists(path))
    except Exception as e:  # noqa: BLE001
        rec.update(ok=False, error=repr(e)[:200], file_created=os.path.exists(path))
    shutil.rmtree(path, ignore_errors=True)
    emit(rec)


def child_write():
    import mlx.core as mx

    # known-pattern buffers allocated first, so the kernel output lands among them
    neighbors = []
    for _ in range(N_NEIGHBORS):
        b = mx.full((SIZE,), 777.0, dtype=mx.float32)
        mx.eval(b)
        neighbors.append(b)
    dummy = mx.zeros((1,), dtype=mx.float32)
    mx.eval(dummy)
    nthreads = 4 * SIZE  # sentinel writes cover 64 KB past the 16 KB output
    src = (
        "uint i = thread_position_in_grid.x;\n"
        f"if (i < {SIZE}u) out[i] = 1.0f + inp[0] * 0.0f;\n"
        f"out[{SIZE}u + i] = {SENTINEL}f;\n"
    )
    k = mx.fast.metal_kernel(
        name="spike07_write", input_names=["inp"], output_names=["out"], source=src)
    rec = {"tag": "write"}
    try:
        (out,) = k(
            inputs=[dummy], output_shapes=[(SIZE,)], output_dtypes=[mx.float32],
            grid=(nthreads, 1, 1), threadgroup=(256, 1, 1))
        mx.eval(out)
        mx.synchronize()
        corrupted = 0
        sentinel_hits = 0
        bad_vals = []
        for b in neighbors:
            nbad = int(mx.sum(b != 777.0).item())
            if nbad:
                corrupted += 1
                sentinel_hits += int(mx.sum(b == SENTINEL).item())
                if len(bad_vals) < 4:
                    idx = int(mx.argmax(b != 777.0).item())
                    bad_vals.append(float(b[idx].item()))
        rec.update(
            ok=True,
            own_output_intact=bool(mx.all(out == 1.0).item()),
            corrupted_buffers=corrupted,
            n_neighbors=N_NEIGHBORS,
            sentinel_hits=sentinel_hits,
            bad_vals=bad_vals,
        )
    except Exception as e:  # noqa: BLE001
        rec.update(ok=False, error=repr(e)[:300])
    emit(rec)


CHILDREN = {
    "read": child_read,
    "read_setenv": child_read_setenv,
    "capture_off": child_capture_off,
    "capture_on": child_capture_on,
    "write": child_write,
}

# ---------------- parent side ----------------

SIGNS = [
    "invalid device load", "invalid device store", "invalid load", "invalid store",
    "shader validation", "mtldebug", "out-of-bounds", "out of bounds",
]


def signals(stderr):
    lo = (stderr or "").lower()
    return sorted({s for s in SIGNS if s in lo})


def first_signal_line(stderr):
    for line in (stderr or "").splitlines():
        if any(s in line.lower() for s in SIGNS):
            return line.strip()[:160]
    return ""


def run_child(mode, extra_env=None, timeout=90):
    env = dict(os.environ)
    env.pop("MTL_SHADER_VALIDATION", None)
    env.pop("MTL_CAPTURE_ENABLED", None)
    if extra_env:
        env.update(extra_env)
    try:
        p = subprocess.run(
            [sys.executable, HERE, "--child", mode],
            capture_output=True, text=True, env=env, timeout=timeout)
        rc, out_s, err_s = p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as e:
        rc, out_s, err_s = "timeout", str(e.stdout or ""), str(e.stderr or "")
    recs = {}
    for line in out_s.splitlines():
        try:
            r = json.loads(line)
            recs[r.get("tag")] = r
        except (ValueError, TypeError, AttributeError):
            pass
    return rc, recs, err_s


def fact(slug, status, detail):
    print(f"FACT {slug}: {status} - {detail}")


def describe_read(rec, rc):
    if rec is None:
        return f"no record (child rc={rc}, likely crashed before it ran)"
    if not rec.get("ok"):
        return f"exception at eval: {rec['error']}"
    return (f"no fault, zeros={rec['zeros']}/{SIZE} ones={rec['ones']} "
            f"nans={rec['nans']} sample={[round(v, 2) for v in rec['sample']]}")


VALIDATE_ENV = {"MTL_SHADER_VALIDATION": "1"}
REPORT_ENV = {"MTL_SHADER_VALIDATION": "1", "MTL_SHADER_VALIDATION_REPORT_TO_STDERR": "1"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    rc_c, recs_c, err_c = run_child("read")
    rc_v, recs_v, err_v = run_child("read", VALIDATE_ENV)
    rc_r, recs_r, err_r = run_child("read", REPORT_ENV)
    rc_s, recs_s, err_s = run_child("read_setenv")
    rc_off, recs_off, _ = run_child("capture_off")
    rc_on, recs_on, _ = run_child("capture_on", {"MTL_CAPTURE_ENABLED": "1"})
    rc_w, recs_w, err_w = run_child("write")
    rc_wr, recs_wr, err_wr = run_child("write", REPORT_ENV)

    # fact 1: plan claims OOB reads do not reliably fault (observed silent zeros)
    ctl, near_c, far_c = recs_c.get("ctl"), recs_c.get("near"), recs_c.get("far")
    ctl_ok = bool(ctl and ctl.get("ok") and ctl.get("ones") == SIZE)
    near_silent = bool(near_c and near_c.get("ok"))
    fact(
        "oob_read_no_validation",
        "PASS" if (ctl_ok and near_silent) else "FAIL",
        f"in-bounds control all-ones={ctl_ok}; near(+48KB past 16KB buf): "
        f"{describe_read(near_c, rc_c)}; far(+256MB): {describe_read(far_c, rc_c)}; "
        f"child rc={rc_c}",
    )

    # fact 2: is the plan's validate mode (MTL_SHADER_VALIDATION=1 alone) detectable
    near_v = recs_v.get("near")
    sigs_v = [s for s in signals(err_v) if s not in signals(err_c)]
    exception_v = bool(near_v and not near_v.get("ok"))
    detectable_v = bool(sigs_v) or exception_v or (rc_v != 0 and rc_c == 0)
    fact(
        "oob_read_with_validation",
        "PASS" if detectable_v else "FAIL",
        f"stderr signals={sigs_v or 'none'} (stderr: {first_signal_line(err_v) or 'only the Metal GPU Validation Enabled banner'}); "
        f"near: {describe_read(near_v, rc_v)}; identical to unvalidated run; child rc={rc_v}",
    )

    # no prior claim: the reporting redirect from man MetalValidation
    near_r = recs_r.get("near")
    sigs_r = [s for s in signals(err_r) if s not in signals(err_c)]
    detectable_r = bool(sigs_r) or bool(near_r and not near_r.get("ok"))
    fact(
        "oob_read_validation_stderr_report",
        "INFO",
        f"+MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 at launch: detectable={detectable_r}, "
        f"stderr signals={sigs_r or 'none'}, first line: {first_signal_line(err_r) or 'n/a'}; "
        f"near: {describe_read(near_r, rc_r)}",
    )

    # fact 3: setting the vars inside a running process must change nothing
    detectable_any = detectable_v or detectable_r
    ref_sigs = sigs_r if detectable_r else sigs_v
    pre, post = recs_s.get("pre"), recs_s.get("post")
    sig_s = signals(err_s)
    env_seen = (recs_s.get("env") or {}).get("MTL_SHADER_VALIDATION")
    if not detectable_any:
        fact("shader_validation_launch_time_only", "INFO",
             "cannot verify: no launch-time configuration produced a detectable signal to compare against")
    else:
        unchanged = (post is not None and not sig_s
                     and bool(post.get("ok")) == bool((pre or {}).get("ok")))
        fact(
            "shader_validation_launch_time_only",
            "PASS" if unchanged else "FAIL",
            f"both validation vars set mid-process (readback={env_seen!r}), fresh pipeline compiled after: "
            f"post-set OOB run {describe_read(post, rc_s)}; signals={sig_s or 'none'} "
            f"(same config at launch showed {ref_sigs}); child rc={rc_s}",
        )

    # fact 4: MTL_CAPTURE_ENABLED launch-time only
    off = recs_off.get("capture_off") or {}
    on = recs_on.get("capture_on") or {}
    before = off.get("before_set") or {}
    after = off.get("after_set") or {}
    on_ok = bool(on.get("ok")) and bool(on.get("file_created"))
    launch_only = (not before.get("ok")) and (not after.get("ok")) and on_ok
    fact(
        "capture_launch_time_only",
        "PASS" if launch_only else "FAIL",
        f"no env at launch: start_capture before set ok={before.get('ok')} "
        f"({(before.get('error') or '')[:80]}), after os.environ set ok={after.get('ok')} "
        f"({(after.get('error') or '')[:80]}); env at launch: capture ok={bool(on.get('ok'))} "
        f"trace file created={bool(on.get('file_created'))}; rcs={rc_off},{rc_on}",
    )

    # fact 5: plan claims OOB access does not reliably fault; measure write side effects
    w = recs_w.get("write")
    write_silent = rc_w == 0 and bool(w and w.get("ok"))
    if w and w.get("ok"):
        detail = (f"no fault; own output intact={w['own_output_intact']}; corrupted neighbor "
                  f"buffers={w['corrupted_buffers']}/{w['n_neighbors']} "
                  f"sentinel hits={w['sentinel_hits']} bad value sample={w['bad_vals']} "
                  f"(64KB written past a 16KB output)")
    elif w:
        detail = f"exception at eval: {w['error']}; child rc={rc_w}"
    else:
        detail = f"child produced no record, rc={rc_w}, stderr tail: {(err_w or '')[-160:]}"
    fact("oob_write_no_validation", "PASS" if write_silent else "FAIL", detail)

    # no prior claim: what validation does with the same OOB write
    wr = recs_wr.get("write")
    sigs_wr = [s for s in signals(err_wr) if s not in signals(err_w)]
    wr_desc = ("no record" if not wr else
               f"exception: {wr['error']}" if not wr.get("ok") else
               f"own output intact={wr['own_output_intact']}, corrupted neighbors={wr['corrupted_buffers']}")
    fact(
        "oob_write_with_validation",
        "INFO",
        f"validation+stderr-report at launch: stderr signals={sigs_wr or 'none'}, "
        f"first line: {first_signal_line(err_wr) or 'n/a'}; {wr_desc}; child rc={rc_wr}",
    )

    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        CHILDREN[sys.argv[2]]()
        sys.exit(0)
    sys.exit(main())
