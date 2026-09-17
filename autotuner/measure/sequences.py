"""Whole sequences are the samples here; the protocol lives in the runtime so an
exported bundle's benchmark runs the same code the job's final check ran."""

from autotuner_runtime.sequence import compare_sequences  # noqa: F401
