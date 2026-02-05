#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from making_dataset.index.unified_searcher import UnifiedSearcher  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test: UnifiedSearcher over local BM25 + web BM25.")
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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    s = UnifiedSearcher(
        local_chunks_path=args.local_chunks,
        web_bm25_index_path=args.web_bm25_index,
    )

    q_web = "Arwa University cultural activities"
    q_local = "employee engagement survey participation"

    print("WEB query:", q_web)
    hits = s.search(q_web, corpus="web", k=3)
    for i, h in enumerate(hits, 1):
        print(f"{i}. {h.source_type} {h.chunk_id} score={h.score:.4f}")

    print("\nLOCAL query:", q_local)
    hits = s.search(q_local, corpus="local", k=3)
    for i, h in enumerate(hits, 1):
        preview = h.text[:80].replace("\n", "\\n")
        print(f"{i}. {h.source_type} {h.chunk_id} score={h.score:.4f} preview={preview}")

    print("\nALL (RRF fuse) query:", q_web)
    hits = s.search(q_web, corpus="all", k=5)
    for i, h in enumerate(hits, 1):
        print(f"{i}. {h.source_type} {h.chunk_id}")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

