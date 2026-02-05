#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test: local neighbor cache -> local multihop dataset -> validate.")
    parser.add_argument(
        "--python",
        default="/home/toolkit/.mamba/envs/vllm013/bin/python",
        help="Python executable to use (default: vllm013 python)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Neighbors per chunk (default: 20)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=5,
        help="Number of tasks to generate (default: 5)",
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=4,
        help="Hop count (default: 4)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    chunks = outputs / "chunks_local.jsonl"
    secrets = outputs / "secret_inventory.jsonl"
    if not chunks.exists():
        raise FileNotFoundError(f"Missing {chunks} (run data_prep/chunk_local.py first)")
    if not secrets.exists():
        raise FileNotFoundError(f"Missing {secrets} (run generate/privacy_tagger.py first)")

    neighbors = outputs / "local_neighbors.smoke.jsonl"
    dataset = outputs / "local_only.smoke.jsonl"

    cmd_neighbors = [
        args.python,
        str(root / "edges" / "build_local_neighbors.py"),
        "--chunks",
        str(chunks),
        "--output",
        str(neighbors),
        "--k",
        str(args.k),
        "--query-max-chars",
        "800",
    ]
    subprocess.run(cmd_neighbors, check=True)

    cmd_gen = [
        args.python,
        str(root / "generate" / "local_multihop_dataset.py"),
        "--chunks",
        str(chunks),
        "--neighbors",
        str(neighbors),
        "--secrets",
        str(secrets),
        "--output",
        str(dataset),
        "--num-tasks",
        str(args.num_tasks),
        "--hops",
        str(args.hops),
        "--seed",
        "0",
    ]
    subprocess.run(cmd_gen, check=True)

    cmd_val = [
        args.python,
        str(root / "validate" / "local_dataset_validate.py"),
        "--dataset",
        str(dataset),
        "--chunks",
        str(chunks),
    ]
    subprocess.run(cmd_val, check=True)

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

