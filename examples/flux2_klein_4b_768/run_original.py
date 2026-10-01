"""Generate images with FLUX.2 Klein 4B in mflux, as people run it today.

Loads the published FLUX.2 Klein 4B with mflux, quantized to 4 bits, and
generates three 768 x 768 images, printing how long each took. The first
image also pays for mflux compiling its denoising step (on an M3 or newer).
run_optimized.py is this file plus one line.

The printed times show the run working, not the speedup: on a fanless Mac the
same image slows by a third as the chip heats over three images, far more than
the gain. For the speedup, run flux2_klein_4b_768_artifact/benchmark.py, which
alternates the original and the optimized model.

Needs mflux 0.20.0; the weights (about 8.5 GB) download on first use.
Usage, from this folder:  python run_original.py
"""

import time

from mflux.models.common.config import ModelConfig
from mflux.models.flux2.variants import Flux2Klein

pipe = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_4b())

prompts = ["a lighthouse on a rocky coast at dusk, oil painting",
           "a red fox asleep in fresh snow, morning light",
           "a busy night market lit by paper lanterns"]
for i, prompt in enumerate(prompts):
    start = time.perf_counter()
    image = pipe.generate_image(seed=42 + i, prompt=prompt, width=768, height=768)
    print(f"image {i}: {time.perf_counter() - start:.1f} s{'  (includes compiling)' if i == 0 else ''}")
    image.save(f"original_{i}.png")
