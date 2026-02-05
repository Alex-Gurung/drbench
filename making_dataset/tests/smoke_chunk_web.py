#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test: build a small web chunk file from corpus cache.")
    parser.add_argument(
        "--python",
        default="/home/toolkit/.mamba/envs/vllm013/bin/python",
        help="Python executable to use (default: vllm013 python)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Docs to write for the smoke file (default: 200)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    out_path = root / "outputs" / "chunks_web.smoke.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        str(root / "data_prep" / "chunk_web.py"),
        "--source",
        "corpus_cache",
        "--output",
        str(out_path),
        "--limit",
        str(args.limit),
    ]
    subprocess.run(cmd, check=True)

    seen = set()
    n = 0
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            assert obj.get("source_type") == "web"
            cid = obj.get("chunk_id")
            did = obj.get("doc_id")
            text = obj.get("text") or ""
            assert cid and did
            assert cid.startswith("web/") and did.startswith("web/")
            assert text.strip()
            assert did + "#0001" == cid
            assert did not in seen
            seen.add(did)
            n += 1

    assert n == args.limit, f"Expected {args.limit} records, got {n}"
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

