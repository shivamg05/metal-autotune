"""Generate images with FLUX.2 Klein 4B in mflux, with the optimization loaded.

Identical to run_original.py plus one line: apply() loads the optimized
kernels from flux2_klein_4b_768_artifact/ into mflux's transformer. Four of
them run at 768 x 768 only, where they were checked; one also runs at 512 x
512, 768 x 1,024 and 1,024 x 1,024. Elsewhere the transformer runs its
original code. The images are not bit-identical to run_original.py's: four
kernels keep intermediate results in 16-bit and were accepted within the
model's numeric tolerance.

The printed times show the run working, not the speedup: on a fanless Mac the
same image slows by a third as the chip heats over three images, far more than
the gain. For the speedup, run flux2_klein_4b_768_artifact/benchmark.py, which
alternates the original and the optimized model.

Needs mflux 0.20.0; the weights (about 8.5 GB) download on first use.
Usage, from this folder:  python run_optimized.py
"""

import time

from mflux.models.common.config import ModelConfig
from mflux.models.flux2.variants import Flux2Klein

from flux2_klein_4b_768_artifact import apply

pipe = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_4b())
pipe.transformer = apply(pipe.transformer)  # the only difference from run_original.py

prompts = ["a lighthouse on a rocky coast at dusk, oil painting",
           "a red fox asleep in fresh snow, morning light",
           "a busy night market lit by paper lanterns"]
for i, prompt in enumerate(prompts):
    start = time.perf_counter()
    image = pipe.generate_image(seed=42 + i, prompt=prompt, width=768, height=768)
    print(f"image {i}: {time.perf_counter() - start:.1f} s{'  (includes compiling)' if i == 0 else ''}")
    image.save(f"optimized_{i}.png")
