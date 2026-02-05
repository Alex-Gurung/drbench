#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from making_dataset.index.unified_searcher import UnifiedSearcher  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test: UnifiedSearcher with web dense retrieval enabled."
    )
    parser.add_argument(
        "--local-chunks",
        default="/home/toolkit/nice_code/drbench/making_dataset/outputs/chunks_local.jsonl",
        help="Local chunks JSONL",
    )
    parser.add_argument(
        "--web-bm25-index",
        default="/home/toolkit/BrowseComp-Plus/indexes/bm25",
        help="BrowseComp-Plus BM25 index path",
    )
    parser.add_argument(
        "--web-dense-index-glob",
        default="/home/toolkit/BrowseComp-Plus/indexes/qwen3-embedding-0.6b/corpus.shard1_of_4.pkl",
        help="Dense index glob (default: shard1 only for fast smoke).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    s = UnifiedSearcher(
        local_chunks_path=args.local_chunks,
        web_bm25_index_path=args.web_bm25_index,
        web_dense_index_glob=args.web_dense_index_glob,
    )

    q = "Arwa University cultural activities"

    print("WEB dense query:", q)
    hits = s.search(q, corpus="web", k=5, web_backend="dense")
    for i, h in enumerate(hits, 1):
        print(f"{i}. {h.source_type} {h.chunk_id} score={h.score:.4f}")

    print("\nWEB bm25->dense rerank query:", q)
    hits = s.search(q, corpus="web", k=5, web_backend="bm25_rerank_dense", web_bm25_candidates_k=200)
    for i, h in enumerate(hits, 1):
        print(f"{i}. {h.source_type} {h.chunk_id} score={h.score:.4f}")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

