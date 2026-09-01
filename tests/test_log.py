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
