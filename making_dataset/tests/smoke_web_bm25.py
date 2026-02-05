#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from making_dataset.index.web_bm25 import WebBM25Searcher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test: open BrowseComp-Plus BM25 index and run a query.")
    parser.add_argument(
        "--index",
        default="/home/toolkit/BrowseComp-Plus/indexes/bm25",
        help="Path to BrowseComp-Plus BM25 Lucene index",
    )
    parser.add_argument(
        "--query",
        default="Arwa University cultural activities",
        help="Query string",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k hits to fetch (default: 5)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    s = WebBM25Searcher(args.index)
    hits = s.search(args.query, k=args.k)
    if not hits:
        raise ValueError("No hits returned (unexpected for smoke test).")
    print(f"Index docs: {s.num_docs}")
    print(f"Query: {args.query}")
    print("\nTop hits:")
    for i, h in enumerate(hits, 1):
        preview = h.text[:120].replace("\n", "\\n")
        print(f"{i}. docid={h.docid} score={h.score:.4f} text_preview={preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
