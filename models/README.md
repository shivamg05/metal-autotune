# Model targets

Each Python file exposes `build()` and loads pretrained weights unless explicitly
marked otherwise. Workload manifests select input shapes; they do not change the
model's precision during optimization. The first build downloads weights.

| Target | MLX-LM support | Example workloads |
| --- | --- | --- |
| Qwen3-4B, 4-bit | Yes, `qwen3` | `workloads/qwen3_4b_prefill_128.yaml`, `qwen3_4b_prefill_512.yaml` |
| Qwen3.5-4B, 4-bit | Yes, `qwen3_5` | Define input shapes in your manifest; see the usage guide |
| Mamba-370M, fp16 | Yes, `mamba` | `workloads/mamba_370m_prefill_32.yaml`, `mamba_370m_prefill_128.yaml` |
| RecurrentGemma 2B, checkpoint precision | Yes, `recurrent_gemma` | `workloads/recurrentgemma_2b_prefill_512.yaml`, `recurrentgemma_2b_prefill_2048.yaml` |
| Whisper small, fp16 encoder | No; MLX-Whisper | `workloads/whisper_small_encoder_30s.yaml` |
| Stable Diffusion 2.1-base, fp16 U-Net | No; MLX Examples | `workloads/sd21_unet_512.yaml`, `sd21_unet_768.yaml` |

Each manifest is a separate job with a modest
24-attempt budget. From the repo root, follow RUNNING.md and substitute:

```sh
uv run autotune run models/workloads/qwen3_4b_prefill_128.yaml --judge codex --work-dir runs/work-qwen4b-prefill-128
```

Use a fresh work directory each time. These are target definitions, not evidence
of a speedup or a completed optimization run. Unsupported operations can still
limit which regions the tool searches.

## Mamba

This is an older research language model, useful for testing optimization
headroom rather than current chat-model quality. Its checkpoint is
`mlx-community/mamba-370m-hf-f16`; loading preserves the pretrained FP16 weights.

Start with one 32-token prompt, batch 1, with a real empty cache. The 128-token
variant is also available but produces a much larger trace.
Its final benchmark uses eight paired comparisons of one complete prompt each.
Additional calls that advance the cache would measure continuation chunks,
which are a different workload. The budget is eight attempts per region,
24 total. From the repo root, follow RUNNING.md with:

```sh
uv run autotune run models/workloads/mamba_370m_prefill_32.yaml --judge claude-cli --work-dir runs/work-mamba-prefill-32
```

Choose a fresh work directory. This target's many small state-update operations
offer fusion opportunities, but also produce a large trace. Do not infer total
search duration from forward-pass latency alone.

## RecurrentGemma

The checkpoint is `google/recurrentgemma-2b`. Accept Google's license on Hugging
Face and authenticate with `hf auth login` before building it. This target keeps
its checkpoint precision; it is not the 4-bit variant.

Use prefill only. The harness does not currently support rewinding this model's
recurrent state for decode. Repeated final benchmark calls repeat independent
prefills, not consecutive generated tokens.

## Whisper

Install the optional dependency into the project environment:

```sh
uv pip install mlx-whisper
uv run --no-sync autotune run models/workloads/whisper_small_encoder_30s.yaml --judge codex --work-dir runs/work-whisper-small
```

The input is a synthetic mel tensor in MLX layout `[1, 3000, 80]`, representing
the shape of a 30-second audio window. This times the encoder only, excluding
feature extraction, token decoding and audio I/O. It is not a transcription
quality benchmark.

## Stable Diffusion

Use Apple's official source implementation instead of copying it into this repo:

```sh
git clone --depth 1 https://github.com/ml-explore/mlx-examples.git ../mlx-examples
uv pip install -r ../mlx-examples/stable_diffusion/requirements.txt
PYTHONPATH="$(cd ../mlx-examples/stable_diffusion && pwd)" uv run --no-sync python -c "import runpy; runpy.run_path('models/sd21_unet.py')['build']()"
```

This setup allows loading the adapter. Full optimizer runs now fail preflight
until the external SD source is included in the model project or installed as
a distribution that the artifact can reproduce.

The U-Net accepts `(latent, timestep, text_embeddings)`, with channels last.
Batch two represents the conditional and unconditional branches for one image
with classifier-free guidance. The 512-pixel workload is native to the selected
2.1-base checkpoint; 768 is a larger shape experiment.

The final benchmark repeats 20 U-Net calls with synthetic inputs. It does not
run a diffusion scheduler or generate an image, and excludes the text encoder
and VAE. Keep the source checkout available to worker processes.

## Deployment limits

Artifacts bundle code and dependency information, not model weights by default.
The original builder retains its loading behavior, so downloads, local paths and
optional libraries must also work on the deployment machine. External SD source
must be packaged or installed as described above. See
[artifact compatibility](../docs/artifacts.md).
