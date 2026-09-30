"""Pause and resume: a job's state saved at safe points, restored in a new process.

A job saves its state after setup, before every judge call and between
regions: points where no GPU work is in flight and every verdict so far is
recorded. `autotune pause` asks the job to stop at the next such point;
three unreachable judge calls in a row stop it there too. `autotune resume`
loads the model and records it again (deterministic, seconds), checks the
recording still yields every region the saved state refers to, restores the
state, reinstalls the saved wrappers, checks the model's outputs against the
untouched model, and continues from the saved point, the open region's queue
and history included. Nothing measured during setup is measured again.

The live model, the recorder, logs and the saved boundary files are rebuilt
or reopened; everything in SAVED is restored as saved.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
from pathlib import Path

STATE = "resume_state.pkl"
PAUSE_REQUEST = "pause.request"
JOB = "job.json"

# runner attributes a resumed job restores as saved
SAVED = ("report", "baseline", "step_ms", "step_chain", "peaks", "total_hypotheses", "lessons",
         "scaffold_overrides", "cuts", "certified_scopes", "replay_scopes", "emitted", "model_ratio",
         "model_ratios", "pending_regions", "selection_wave", "shipped_tags", "shipped_regions",
         "sweep_spans", "gpu_busy_at_start", "baseline_wrappers")


class Paused(Exception):
    """The job stopped at a safe point and can be resumed; the message says why."""


def save(runner, where: str) -> None:
    """Write the runner's state and search position, atomically."""
    state = {name: getattr(runner, name) for name in SAVED if hasattr(runner, name)}
    # the modules a wrapper replaced are live objects: a resumed job finds them again
    state["installed"] = {path: (variants, kernels) for path, (_original, variants, kernels)
                          in getattr(runner, "installed", {}).items()}
    state["compiled_baseline"] = getattr(runner, "_compiled_baseline", None) is not None
    state["search"] = getattr(runner, "_search_state", None)
    state["where"] = where
    path = Path(runner.work_dir) / STATE
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def load(work_dir) -> dict | None:
    path = Path(work_dir) / STATE
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def pause_requested(work_dir) -> bool:
    """A pending pause request, consumed so the resumed job does not stop again."""
    request = Path(work_dir) / PAUSE_REQUEST
    if not request.is_file():
        return False
    request.unlink(missing_ok=True)
    return True


def code_identity() -> str:
    """The checked-out commit plus a digest of uncommitted changes, when git can say."""
    root = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=root, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{head}+{hashlib.sha256(diff).hexdigest()[:12]}" if diff else head


def file_digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_job(work_dir, manifest, judge: dict, artifact) -> None:
    """What `autotune resume` needs to start the same job again."""
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    digest = file_digest(manifest) if Path(manifest).is_file() else None
    (Path(work_dir) / JOB).write_text(json.dumps({
        "manifest": str(Path(manifest).resolve()), "manifest_sha256": digest,
        "code": code_identity(), "judge": judge, "artifact": str(artifact)}, indent=1) + "\n")


def read_job(work_dir) -> dict | None:
    path = Path(work_dir) / JOB
    return json.loads(path.read_text()) if path.is_file() else None
