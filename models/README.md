# Model targets

Each file here is a ready-to-use `build()` for one model. Point a manifest's
`model:` at it and you have an optimization target. Pretrained weights download
the first time you build the model; they're never stored in this repo. The tool
keeps each model's precision as is: a 4-bit model stays 4-bit.

New to the tool? Start with [the tiny MLP](../examples/tiny_mlp.yaml). It
downloads nothing.

Some models come with ready-made manifests in `workloads/`. For the rest, copy
an example from [Write a manifest](../docs/manifest.md) and change its `model:`
path.

| Model | Runs through | Ready-made manifests |
| --- | --- | --- |
| [Qwen3-0.6B Base, bf16](qwen3_0.6b.py) | MLX-LM (`qwen3`) | none |
| [Qwen3-0.6B Base, 4-bit](qwen3_0.6b_4bit.py) | MLX-LM (`qwen3`), quantized after loading | none |
| [Qwen3-4B, 4-bit](qwen3_4b_4bit.py) | MLX-LM (`qwen3`) | `qwen3_4b_prefill_128.yaml`, `qwen3_4b_prefill_512.yaml` |
| [Qwen3.5-4B, 4-bit](qwen3_5_4b_4bit.py) | MLX-LM (`qwen3_5`), text only | none |
| [Qwen3.5-4B, 8-bit](qwen3_5_4b_8bit.py) ¹ | MLX-LM (`qwen3_5`), text only | none |
| [Qwen3.5-9B, 4-bit](qwen3_5_9b_4bit.py) ¹ | MLX-LM (`qwen3_5`), text only | none |
| [Qwen3.5-9B, 8-bit](qwen3_5_9b_8bit.py) ¹ | MLX-LM (`qwen3_5`), text only | none |
| [Qwen3.8-27B, 4-bit](qwen3_8_27b_4bit.py) ¹ | MLX-LM (`qwen3_5`), text only | none |
| [Llama 3 8B Instruct, 4-bit](llama8b.py) | MLX-LM (`llama`) | none |
| [LFM2.5-1.2B-Thinking, 4-bit](lfm2_5_1_2b_thinking_4bit.py) ¹ | MLX-LM (`lfm2`) | none |
| [LFM2.5-1.2B-Thinking, 8-bit](lfm2_5_1_2b_thinking_8bit.py) ¹ | MLX-LM (`lfm2`) | none |
| [LFM2.5-2.6B, 4-bit](lfm2_5_2_6b_4bit.py) ¹ | MLX-LM (`lfm2`) | none |
| [LFM2.5-2.6B, 8-bit](lfm2_5_2_6b_8bit.py) ¹ | MLX-LM (`lfm2`) | none |
| [Muse-Glimmer-30B, 4-bit](muse_glimmer_30b_4bit.py) ¹ | MLX-VLM, direct text-model calls only | none |
| [Mamba-370M, fp16](mamba_370m.py) | MLX-LM (`mamba`) | `mamba_370m_prefill_32.yaml`, `mamba_370m_prefill_128.yaml` |
| [RecurrentGemma 2B, checkpoint precision](recurrentgemma_2b.py) | MLX-LM (`recurrent_gemma`) | `recurrentgemma_2b_prefill_512.yaml`, `recurrentgemma_2b_prefill_2048.yaml` |
| [Whisper small, fp16 encoder](whisper_small_encoder.py) | MLX-Whisper, encoder calls | `whisper_small_encoder_30s.yaml` |
| [FLUX.2 transformer, random weights](flux2_4b.py) | Standalone MLX; one denoiser call | none; see [`manifest_flux.yaml`](../manifest_flux.yaml) for its inputs. No download, no image generation |

¹ New targets. Their checkpoint configs have been checked, but they haven't yet
been fully loaded and optimized end to end. See
[newer language-model targets](#newer-language-model-targets).

## Running a ready-made manifest

`uv sync` already installs MLX-LM, MLX-VLM and MLX-Whisper, so the manifests in
`workloads/` run as soon as the model's checkpoint is reachable. Two
exceptions: RecurrentGemma needs you to accept a license first, and Stable
Diffusion needs [extra packaging](#stable-diffusion-extra-setup) before it can
be optimized.

What they all have in common:

- **Beat compiled MLX** (`baseline: compiled`).
- **24 attempts per job**, at most 8 per region.
- **The language-model ones time a full MLX-LM request for one output token**
  (`use_library_inference: true`, `final_benchmark.steps: 1`). That's reading a
  made-up prompt and producing one token, starting from an empty cache every
  time. Sampling and any extra work the library queues are included, so it's
  not a bare forward pass. Loading the model and tokenizing aren't timed.
- **Whisper and Stable Diffusion time direct model calls** instead (see their
  sections below).

Run one from the repo root, using a fresh work folder each time. If an agent is
running it for you, have it follow [RUNNING.md](../RUNNING.md):

```sh
uv run autotune run models/workloads/qwen3_4b_prefill_128.yaml --judge codex --work-dir runs/work-qwen4b-prefill-128
```

These manifests define targets. They aren't evidence that a speedup exists, or
that a run has been completed. Operations the tool doesn't support yet can
also limit which regions it can search.

## Newer language-model targets

**Status:** these targets' checkpoint configs have been checked, but none has
been fully loaded and optimized end to end yet.

- **LFM2.5 and Qwen3.5 (4B and 9B)** each come in separate 4-bit and 8-bit
  files. Each keeps its checkpoint's quantization. One quirk: Liquid's 2.6B
  4-bit checkpoint uses 6-bit embeddings, and its other layers keep their
  published precision.
- **The Qwen3.5 and Qwen3.8 targets load only the text model.** Image input
  isn't available.
- **Muse Glimmer 30B** loads through MLX-VLM and exposes just its text model.
  - Set `use_library_inference: false`. The tool can't yet time full MLX-VLM
    generation.
  - Token inputs are shaped `[batch, tokens]`. Use `context: 0` for an empty
    cache, or a positive `context` for a prefilled one.
  - The adapter returns the model's original logits and lets the library
    create the cache. No image input.
  - GPU tests on a small model cover the adapter and the cache handling, but
    the full 30B checkpoint hasn't been loaded here.
- **The big ones need a lot of memory.** Muse 30B's weights are about 19.4 GB
  and Qwen3.8 27B's about 16.1 GB, before you count caches and intermediate
  results. Check your memory headroom before starting a long run.

## Mamba

An older research model (`mlx-community/mamba-370m-hf-f16`, pretrained FP16
weights, kept as is). It's not here for chat quality. It's here because it's
full of small state-update operations that are good candidates for fusing
together, which makes it a useful test of how much the tool can find.

Start with the 32-token prompt, batch 1, from a real empty cache:

```sh
uv run autotune run models/workloads/mamba_370m_prefill_32.yaml --judge claude-cli --work-dir runs/work-mamba-prefill-32
```

- The final benchmark runs 8 paired comparisons of a one-token request, each
  starting from an empty cache. The budget is the usual 8 per region, 24 total.
- A 128-token version is available too, but it produces a much bigger trace.
- Those many small operations make even the 32-token trace large. A fast forward
  pass doesn't mean a quick search.

## RecurrentGemma

The checkpoint is `google/recurrentgemma-2b`. Before building it, accept
Google's license on Hugging Face and sign in with `hf auth login`. This is the
full checkpoint precision, not the 4-bit variant.

The manifests time reading a prompt plus one output token, starting from an
empty cache each time. They say nothing about decode steps that start from an
already-filled recurrent state.

## Whisper

MLX-Whisper comes with the default `uv sync`. Run:

```sh
uv run autotune run models/workloads/whisper_small_encoder_30s.yaml --judge codex --work-dir runs/work-whisper-small
```

The input is a made-up mel spectrogram shaped `[1, 3000, 80]` (MLX layout), the
size of a 30-second audio window. **Only the encoder is timed.** Feature
extraction, token decoding and audio I/O aren't included, and this says nothing
about transcription quality.

## Stable Diffusion: extra setup

[The SD 2.1 adapter](sd21_unet.py) and its
[512-pixel](workloads/sd21_unet_512.yaml) and
[768-pixel](workloads/sd21_unet_768.yaml) manifests show how to target a U-Net.
**It isn't ready to optimize out of the box.** It depends on Apple's Stable
Diffusion source code, which this repo doesn't copy in. To load the adapter:

```sh
git clone --depth 1 https://github.com/ml-explore/mlx-examples.git ../mlx-examples
uv pip install -r ../mlx-examples/stable_diffusion/requirements.txt
PYTHONPATH="$(cd ../mlx-examples/stable_diffusion && pwd)" uv run --no-sync python -c "import runpy; runpy.run_path('models/sd21_unet.py')['build']()"
```

That gets the adapter loading, but a full optimization run will still refuse
to start. The artifact has to be able to reproduce the SD code, so that code
has to live inside the model project or be installed as a package first. Keep
the source checkout available to the tool's worker processes.

About the workload:

- The U-Net takes `(latent, timestep, text_embeddings)`, channels last.
- Batch size 2 is one image with classifier-free guidance: a conditional and an
  unconditional pass.
- 512 pixels is the native size for the 2.1-base checkpoint. 768 is a
  larger-shape experiment.
- The final benchmark repeats 20 U-Net calls on made-up inputs. There's no
  diffusion scheduler, no text encoder, no VAE, and no image is generated.

## Deploying an optimized model

The artifact bundles code and dependency info, not weights (by default). Its
`build()` loads the model exactly as the original did, so the downloads, local
paths and optional libraries it uses have to work on the machine you deploy to.
For Stable Diffusion, that includes the packaged SD source described above. See
[artifact compatibility](../docs/artifacts.md#weights-and-other-files).
