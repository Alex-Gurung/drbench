#!/usr/bin/env python3
from __future__ import annotations

import json
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
    out = Path("/home/toolkit/nice_code/drbench/making_dataset/outputs/web_only.smoke.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    from making_dataset.generate.web_only_dataset import main as gen_main  # noqa: WPS433

    # Invoke generator by simulating argv (simple + avoids subprocess noise).
    argv0 = sys.argv[:]
    try:
        sys.argv = [
            argv0[0],
            "--num-tasks",
            "5",
            "--hops",
            "4",
            "--output",
            str(out),
            "--limit-input",
            "50",
        ]
        gen_main()
    finally:
        sys.argv = argv0

    if _count_lines(out) != 5:
        raise ValueError("Expected 5 tasks in smoke output")

    # Minimal structural checks.
    with out.open("r", encoding="utf-8") as f:
        first = json.loads(next(f))
    assert first["mode"] == "web_only"
    assert "tree" in first and "hops" in first["tree"]
    assert len(first["tree"]["hops"]) == 4
    assert first["answer"]

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

