"""The sign convention: internal deltas are positive-means-faster, human-facing
strings are signed ms where negative means faster. fmt_signed_ms is the one
converter, and the CLI's per-workload line goes through it."""

from autotuner.log import fmt_signed_ms


def test_fmt_signed_ms_flips_the_sign():
    assert fmt_signed_ms(4.556) == "-4.556ms"   # 4.556 ms faster -> negative
    assert fmt_signed_ms(-4.556) == "+4.556ms"  # slower -> positive
    assert fmt_signed_ms(0.0) == "+0.000ms"


def test_cli_delta_line_uses_the_formatter():
    import inspect

    from autotuner import cli

    assert "fmt_signed_ms" in inspect.getsource(cli.main)


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
