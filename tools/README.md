# Measurement diagnostics

These scripts investigate GPU timing, cooling and replacement overhead. They are
optional developer tools, not prerequisites for running an optimization.
Read a script's module docstring and `--help` before running it. Some accept an
existing run or model path; they may depend on that model's optional libraries.
Run them serially on a quiet GPU, never alongside a benchmark or optimization.

Write new output under `runs/diagnostics/` or an explicit external path. A timing
observation on one machine is not a portable constant or a confirmed shipped win.
