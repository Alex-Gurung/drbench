#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main() -> int:
    if not os.getenv("VLLM_API_URL"):
        raise ValueError("VLLM_API_URL must be set for this smoke test.")

    out = Path("/home/toolkit/nice_code/drbench/making_dataset/outputs/mixed_entity.smoke.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    from making_dataset.generate.mixed_entity_dataset_poc import main as gen_main  # noqa: WPS433

    argv0 = sys.argv[:]
    try:
        sys.argv = [
            argv0[0],
            "--num-tasks",
            "2",
            "--hops",
            "4",
            "--output",
            str(out),
            "--web-backend",
            "bm25",  # keep smoke cheaper
            "--max-tokens",
            "128",
            "--no-rewrite-question",
        ]
        gen_main()
    finally:
        sys.argv = argv0

    if _count_lines(out) != 2:
        raise ValueError("Expected 2 tasks in smoke output")

    with out.open("r", encoding="utf-8") as f:
        first = json.loads(next(f))
    assert first["mode"] == "mixed"
    assert first["answer_type"] == "entity_string"
    assert len(first["tree"]["hops"]) == 4

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

