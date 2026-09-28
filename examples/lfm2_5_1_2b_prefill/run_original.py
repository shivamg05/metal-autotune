"""Summarize documents with LFM2.5-1.2B-Instruct, as released: no optimization.

Loads Liquid's 4-bit model with mlx-lm and asks it to summarize four of this
repository's guides, prompts of about 550 to 2,300 tokens. For each it prints
the prompt length, the wait before the first word, the decoding speed and the
answer's first line. run_optimized.py is this script plus one line.

Usage, from this folder:  python run_original.py
"""

import time
from pathlib import Path

from mlx_lm import load, stream_generate

model, tokenizer = load("LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit")

docs = Path(__file__).resolve().parents[2] / "docs"
jobs = [["limitations.md"], ["architecture.md"], ["architecture.md", "limitations.md"], ["artifacts.md"]]


def prompt_for(names):
    document = "\n\n".join((docs / name).read_text() for name in names)
    messages = [{"role": "user", "content": "Summarize the main ideas of this document in three sentences.\n\n" + document}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


for _ in stream_generate(model, tokenizer, prompt_for(["limitations.md"]), max_tokens=1):
    pass  # warm-up: loads the weights and GPU programs once

for names in jobs:
    prompt = prompt_for(names)
    start, first, answer = time.perf_counter(), None, ""
    for response in stream_generate(model, tokenizer, prompt, max_tokens=120):
        first = first or time.perf_counter() - start
        answer += response.text
    print(f"{' + '.join(names)}: {len(prompt)} prompt tokens, first word after {first:.2f} s, "
          f"then {response.generation_tps:.0f} tokens/s")
    print("   ", answer.strip().splitlines()[0][:160])
