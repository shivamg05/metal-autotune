# MetalBench standard: Apple M4, September 19–20, 2026

**19 of 35 workloads improved over compiled MLX. Best speedup: 2.344×.**

This is one completed run. All 35 jobs completed; 19 exported validated kernels.
Speedup = compiled original wall-clock time / compiled optimized wall-clock time.
Timings cover the whole problem, using warmed, dependent repeated calls with
the same protocol for both arms. They do not represent isolated dispatch latency.

## Run configuration

- Apple M4, 24 GiB unified memory; MLX 0.32.2; Python 3.12.13.
- MetalBench standard set: 35 problems, shapes and tolerances in each saved manifest.
- Judge: `claude-fable-5-1`, low effort, through `claude-cli`.
- Budget: 10 attempts per region, 30 per problem.
- Baseline: compiled. Final checks compare original and optimized workloads.
- Benchmark definitions: `Lazarus-931/MetalBench@ebe9c790a4fbb397e9f2af1906396a0adbc9e6b2`.

```sh
uv run python metalbench/run.py --set standard --baseline compiled \
  --budget-per-region 10 --budget-total 30 --judge claude-cli \
  --judge-model claude-fable-5-1 --judge-effort low --rerun --tag comparison
```

The provider must still offer the recorded judge model ID to reproduce these settings.

## Results

| Problem | vs compiled | vs eager | Shipped regions |
|---|---:|---:|---:|
| [matmul_gelu_softmax](problems/matmul_gelu_softmax/report.json) | 1.269× | 1.305× | 1 |
| [rms_norm_linear](problems/rms_norm_linear/report.json) | 1.121× | 1.136× | 1 |
| [silu_linear](problems/silu_linear/report.json) | 1.029× | 1.029× | 1 |
| [gelu_linear](problems/gelu_linear/report.json) | 1.000× | 1.009× | 0 |
| [add_norm](problems/add_norm/report.json) | 1.179× | 1.185× | 1 |
| [rope_embedding](problems/rope_embedding/report.json) | 1.000× | 1.550× | 0 |
| [swiglu](problems/swiglu/report.json) | 1.388× | 1.449× | 1 |
| [softmax_attention](problems/softmax_attention/report.json) | 1.244× | 1.318× | 2 |
| [residual_add](problems/residual_add/report.json) | 1.000× | 1.343× | 0 |
| [dropout](problems/dropout/report.json) | 1.000× | 1.171× | 0 |
| [instance_norm](problems/instance_norm/report.json) | 2.231× | 3.063× | 1 |
| [group_norm](problems/group_norm/report.json) | 2.344× | 3.118× | 1 |
| [cross_entropy_loss](problems/cross_entropy_loss/report.json) | 1.535× | 1.826× | 1 |
| [log_softmax](problems/log_softmax/report.json) | 1.143× | 1.152× | 1 |
| [masked_softmax](problems/masked_softmax/report.json) | 1.171× | 1.156× | 1 |
| [bias_add](problems/bias_add/report.json) | 1.000× | 1.018× | 0 |
| [fused_add_rms_norm](problems/fused_add_rms_norm/report.json) | 1.163× | 1.173× | 1 |
| [linear_bias](problems/linear_bias/report.json) | 1.250× | 1.245× | 1 |
| [bias_gelu](problems/bias_gelu/report.json) | 1.000× | 2.131× | 0 |
| [fused_qkv_projection](problems/fused_qkv_projection/report.json) | 1.102× | 1.088× | 1 |
| [attention_scores](problems/attention_scores/report.json) | 1.226× | 1.295× | 1 |
| [nll_loss](problems/nll_loss/report.json) | 1.346× | 1.339× | 1 |
| [log_softmax_cross_entropy](problems/log_softmax_cross_entropy/report.json) | 1.533× | 1.838× | 1 |
| [scaled_dot_product](problems/scaled_dot_product/report.json) | 1.472× | 1.473× | 2 |
| [llama_attention](problems/llama_attention/report.json) | 1.060× | 1.672× | 2 |
| [silu_residual](problems/silu_residual/report.json) | 1.000× | 1.220× | 0 |
| [gelu_residual](problems/gelu_residual/report.json) | 1.000× | 2.504× | 0 |
| [add_silu](problems/add_silu/report.json) | 1.000× | 1.516× | 0 |
| [mul_silu](problems/mul_silu/report.json) | 1.000× | 1.510× | 0 |
| [add_gelu](problems/add_gelu/report.json) | 1.000× | 2.465× | 0 |
| [mul_gelu](problems/mul_gelu/report.json) | 1.000× | 2.511× | 0 |
| [add_relu](problems/add_relu/report.json) | 1.000× | 1.156× | 0 |
| [mul_relu](problems/mul_relu/report.json) | 1.000× | 1.211× | 0 |
| [add_swish](problems/add_swish/report.json) | 1.000× | 1.494× | 0 |
| [residual_tanh](problems/residual_tanh/report.json) | 1.000× | 1.043× | 0 |

## Reading the evidence

`summary.json` holds the scoreboard. Each problem includes its report with raw
final paired samples, workload manifest, generated model definition, and final
kernel source when an artifact shipped. Machine-local repository paths are replaced
with `<repo>`; measurements and verdicts are unchanged. These are evidence snapshots,
not deployable artifacts. The saved model definitions document the original run;
use the command above to generate new runnable jobs.

The eager column includes compilation gains and must not be credited entirely to
generated kernels. No-kernel cases score 1.000× against the compiled baseline.
A region clock and its share are diagnostics, not additive pieces of compiled latency.

## Verification and scope

A separate September 20 check recomputed all 19 headline ratios from raw samples,
and all 19 exported artifacts passed correctness again. With matching sequence
lengths, 14 reconfirmed wins; five were inconclusive under fresh timing noise.
The benchmark results above remain the original run, not selectively replaced reruns.

The exported benchmark currently defaults to the manifest sequence length, while
the job can lengthen tiny workloads automatically. To repeat the same protocol,
use the length in `final.sequences[workload].steps`. This does not change which
execution modes the original scoreboard compares.

Full transient run directories, generated tensors, native build caches, and judge
transcripts are excluded. No model weights are needed for these synthetic problems.
