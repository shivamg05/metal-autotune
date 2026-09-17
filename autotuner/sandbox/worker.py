"""Sandbox worker: the entry point of one child process, one kernel
evaluation. It reads one ladder spec as JSON on stdin, runs the gates in
autotuner.ladder.child, and prints exactly one JSON verdict line to stdout
as its last line. Any unexpected exception tracebacks to stderr and exits
nonzero; the parent maps that to a "subprocess" verdict.

Metal read the shader-validation variables at this process's launch, so the
mode was decided by the parent's spawn environment; the "Invalid device
load/store" lines validation writes land on this process's stderr, and the
parent scans them after exit.
"""

import os
import sys

from autotuner.sandbox.protocol import LadderSpec
from autotuner.sandbox.watchdog import configure


def main() -> None:
    configure(int(os.environ["AUTOTUNER_WATCHDOG_FD"]))
    from autotuner.ladder.child import evaluate_ladder

    verdict = evaluate_ladder(LadderSpec.from_json(sys.stdin.read()))
    print(verdict.to_json(), flush=True)


if __name__ == "__main__":
    main()
