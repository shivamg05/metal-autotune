"""Vendor MetalBench's problem definitions at a pinned commit.

MetalBench (Lazarus-931/MetalBench, MIT) is a KernelBench-shaped set of MLX
modules: single ops, fused pairs, and small full models, each with registered
input shapes. Only the problems are taken; its timing harness and leaderboard
are not (they divide GPU-only kernel time by wall-clock MLX time).

Usage: python metalbench/fetch.py   # writes metalbench/problems/<set>/*.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

REPO = "Lazarus-931/MetalBench"
COMMIT = "ebe9c790a4fbb397e9f2af1906396a0adbc9e6b2"  # 2026-06-02
SETS = ("common", "standard", "full")
HERE = Path(__file__).resolve().parent
PROBLEMS = HERE / "problems"


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def main() -> int:
    tree = json.loads(_get(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"))
    wanted = [p["path"] for p in tree["tree"]
              if p["type"] == "blob" and p["path"].startswith("mlx/kernels/")
              and p["path"].split("/")[2] in SETS and p["path"].endswith(".py")]
    wanted.append("LICENSE")
    for path in wanted:
        raw = _get(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}")
        target = PROBLEMS / (path.removeprefix("mlx/kernels/") if path != "LICENSE" else "LICENSE")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    (PROBLEMS / "PIN").write_text(f"{REPO}@{COMMIT}\n")
    print(f"vendored {len(wanted)} files from {REPO}@{COMMIT[:12]} into {PROBLEMS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
