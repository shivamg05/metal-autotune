"""autotune pause and resume on the command line: the request, what a
resume needs, and when it refuses."""

import json

import pytest

from autotuner.cli import main
from autotuner.resume import JOB, PAUSE_REQUEST, STATE, file_digest, pause_requested, write_job


def test_pause_asks_a_running_job_to_stop_and_the_request_is_used_once(tmp_path):
    with pytest.raises(SystemExit):
        main(["pause", "--work-dir", str(tmp_path)])  # no job has run here
    (tmp_path / "run.jsonl").touch()
    assert main(["pause", "--work-dir", str(tmp_path)]) == 0
    assert (tmp_path / PAUSE_REQUEST).is_file()
    assert pause_requested(tmp_path) and not pause_requested(tmp_path)


def test_resume_needs_a_saved_state(tmp_path):
    (tmp_path / "run.jsonl").touch()
    with pytest.raises(SystemExit):
        main(["resume", "--work-dir", str(tmp_path)])


def test_resume_refuses_a_changed_manifest(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("model: m.py\n")
    write_job(tmp_path, manifest, {"judge": "claude-cli", "model": None, "judge_effort": None,
                                   "judge_cmd": None}, tmp_path / "artifact")
    (tmp_path / STATE).write_bytes(b"")
    job = json.loads((tmp_path / JOB).read_text())
    assert job["manifest_sha256"] == file_digest(manifest) and job["judge"]["judge"] == "claude-cli"
    manifest.write_text("model: other.py\n")
    with pytest.raises(SystemExit):
        main(["resume", "--work-dir", str(tmp_path)])
