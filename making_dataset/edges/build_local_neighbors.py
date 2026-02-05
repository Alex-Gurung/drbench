#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from making_dataset.index.local_bm25 import BM25Index  # noqa: E402
from making_dataset.utils.progress import progress  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local chunk->chunk neighbor cache using a simple BM25 index."
    )
    parser.add_argument(
        "--chunks",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "chunks_local.jsonl"),
        help="Path to chunks_local.jsonl",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "local_neighbors.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of neighbors to store per chunk (default: 20)",
    )
    parser.add_argument(
        "--query-max-chars",
        type=int,
        default=2000,
        help="Truncate chunk text to this many chars for doc-as-query (default: 2000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit of chunks to index (for smoke tests)",
    )
    return parser.parse_args()


def _load_chunks(path: Path, limit: int | None) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("source_type") != "local":
                continue
            if not (obj.get("text") or "").strip():
                continue
            chunks.append(obj)
            if limit is not None and len(chunks) >= limit:
                break
    return chunks


def main() -> int:
    args = _parse_args()
    chunks_path = Path(args.chunks)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks not found: {chunks_path}")

    chunks = _load_chunks(chunks_path, args.limit)
    if not chunks:
        raise ValueError("No local chunks loaded; check input path or filters.")

    texts: List[str] = [c["text"] for c in chunks]
    index = BM25Index(texts)

    records: list[dict[str, Any]] = []
    for i, chunk in progress(enumerate(chunks), total=len(chunks), desc="Neighbors"):
        query = (chunk.get("text") or "")[: max(0, args.query_max_chars)]
        hits = index.search(query, k=args.k, exclude_doc_idx=i)
        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk.get("doc_id"),
                "neighbors": [
                    {"chunk_id": chunks[h.doc_idx]["chunk_id"], "score": h.score}
                    for h in hits
                ],
            }
        )

    with out_path.open("w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

