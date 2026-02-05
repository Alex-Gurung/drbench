#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from making_dataset.index.web_dense import WebDenseSearcher  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test: web dense retrieval (FAISS).")
    parser.add_argument(
        "--index-glob",
        default="/home/toolkit/BrowseComp-Plus/indexes/qwen3-embedding-0.6b/corpus.shard1_of_4.pkl",
        help="Dense index shards glob (default: shard1 only for faster smoke).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Embedding model (HF id; resolved via $TRANSFORMERS_CACHE).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    s = WebDenseSearcher(index_glob=args.index_glob, model_name_or_path=args.model)

    q = "Arwa University cultural activities"
    print("Query:", q)
    hits = s.search(q, k=5)
    for i, h in enumerate(hits, 1):
        print(f"{i}. docid={h.docid} score={h.score:.4f}")

    # Extremely weak assertion: we just want this to run and return something.
    if not hits:
        raise SystemExit("No hits returned")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

