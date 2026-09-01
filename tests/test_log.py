"""The run log and the plain-text log."""


def test_run_log_writes_infinity_as_a_string(tmp_path):
    """A gate detail can hold infinity, which is not JSON; every row must parse."""
    import json

    from autotuner.log import RunLog

    log = RunLog(tmp_path / "run.jsonl")
    log.append("verdict", detail={"max_excess": float("inf"), "nested": [float("nan")]})
    row = json.loads((tmp_path / "run.jsonl").read_text())
    assert row["detail"] == {"max_excess": "inf", "nested": ["nan"]}
    assert log.rows() == [row]


def test_text_log_appends_one_line_per_call(tmp_path):
    from autotuner.log import TextLog

    log = TextLog(tmp_path / "deep" / "candidates.log")
    log.append("first")
    log.append("second\n")
    assert (tmp_path / "deep" / "candidates.log").read_text() == "first\nsecond\n"
