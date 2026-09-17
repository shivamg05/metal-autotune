"""Tracing: the recorder, the patch surface, freeze, and replay.

Usage (the spec's stage-as-code):

    tracer = Tracer()
    tracer.install(model_module_name="model")   # BEFORE the model file imports
    model = import_and_build()
    trace, outs = tracer.trace(model, workload_tensors)   # pass 1: lazy
    tracer.uninstall()                          # pass 2 times the bare model
"""

from __future__ import annotations

from .patches import Patcher
from .recorder import Recorder
from .types import Trace, TraceIncomplete, TraceNode


class Tracer:
    _trace_internal = True

    def __init__(self) -> None:
        self.recorder = Recorder()
        self.patcher = Patcher(self.recorder)

    def install(self, model_module_name: str | None = None) -> None:
        self.patcher.install(model_module_name)

    def trace(self, model, inputs) -> tuple[Trace, object]:
        """One recorded pass: lazy, no eval, completeness-checked. Returns the
        frozen trace and the model's (lazy) outputs."""
        paths = self.patcher.wrap_model(model)
        self.recorder.snapshot_seqs = set()  # a trace wants shapes and ids, never values
        try:
            self.recorder.arm(model, list(inputs), paths)
            try:
                outs = self.recorder.step(model, tuple(inputs))
            finally:
                self.recorder.disarm()
            return self.recorder.freeze_pass(), outs
        finally:
            self.recorder.snapshot_seqs = None

    def uninstall(self) -> None:
        self.patcher.uninstall()

    def verify_restored(self) -> list[str]:
        return self.patcher.verify_restored()
