# Limits and interpreting results

- This project targets MLX on Apple Silicon. Other GPU backends are unsupported.
- Declaring a model does not guarantee that every operation can be traced, lowered
  or safely replaced. The report records skipped regions and the reasons.
- Kernels are verified for declared workloads and applicable runtime guards.
  Untested shapes or cache layouts can fall back to the original implementation.
  One result does not establish a speedup at every shape or on another machine.
- Small gains may remain inconclusive under measurement noise. Background GPU
  activity, temperature and power state affect timing. More attempts do not
  guarantee a confirmed improvement.
- Cache handling and library-inference integration depend on supported model APIs.
  Unsupported explicitly requested behavior should fail preflight, not silently
  benchmark a different task. See the model-specific notes in
  [models/README.md](../models/README.md).
- Final artifacts do not automatically bundle weights. They require compatible
  architecture, dtypes, quantization, dependencies and the builder's external
  resources. See [artifact compatibility](artifacts.md).
- A budget controls candidate attempts, not wall-clock time. Worker failures and
  timeouts can stop a job; preserve its report, console output and checkpoints.

Only final confirmed workload measurements establish a shipped improvement.
Region estimates and faster isolated kernels are search evidence, not that result.
