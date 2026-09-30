# Run an optimization

Start with [README.md](README.md) for installation and a runnable example.
Detailed model and manifest options are in [docs/usage.md](docs/usage.md).

If you are an agent operating a run for someone, read
[docs/operating.md](docs/operating.md) before starting. It defines preflight,
event-driven progress updates, failure reporting, and how to finish early
without losing final validation. Follow the requested manifest and judge;
do not inspect other runs unless asked.

```sh
uv run autotune run path/to/manifest.yaml --judge claude-cli
```

The default output is a fresh timestamped folder under `runs/`. You can specify
`--work-dir runs/YOUR_LABEL` or a location outside this checkout. Never reuse a
nonempty directory or run competing GPU benchmarks alongside an optimization.

## Pause and resume

A run can stop and continue later, in a new process, where it left off:

```sh
uv run autotune pause --work-dir <work-dir>     # from another terminal, while it runs
uv run autotune resume --work-dir <work-dir>    # later, from this checkout
```

- **Pause before the Mac sleeps, restarts or needs its GPU.** The job stops at
  its next safe point, after the attempt it is on, with nothing left running
  on the GPU, and exits with status `paused`. An attempt can take a while (on
  FLUX, half take over 10 minutes and the longest took an hour). If the
  operator can't wait, press Ctrl-C instead and resume later: only the attempt
  in progress is lost, and it is tried again. A sleeping Mac can also drop the
  job's network; don't rely on the job surviving a sleep.
- **Resume** reuses the manifest and judge settings the job started with
  (`--judge`, `--model`, `--judge-effort`, `--judge-cmd` override them). It
  records the model again (seconds), restores everything setup measured,
  reinstalls the accepted kernels, checks the model's outputs still match the
  original, and continues the same region with the same history and budget. It
  refuses if the manifest changed, and warns if the tool's code changed.
- **The job pauses itself** when the judge can't be reached three times in a
  row (no network, an expired login). Those failed calls don't use attempts.
  Fix the cause, then resume.
- **Ctrl-C, a crash or a killed process** can be resumed too, from the last
  safe point: at most the one attempt in progress is repeated. A win accepted
  in that attempt is found again; its checkpoint folder stays, and the repeat
  is saved beside it.
- Runs started before pause and resume existed have no saved state and can't
  be resumed.

After completion, report the confirmed whole-workload result and artifact path,
or explain why nothing shipped. See [artifact usage](docs/artifacts.md).
For a benchmark suite, follow [MetalBench's guide](metalbench/README.md).
