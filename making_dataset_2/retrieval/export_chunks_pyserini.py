#!/usr/bin/env python3
from __future__ import annotations

"""Export a chunks JSONL file to a Pyserini JsonCollection.

BrowseComp(-Plus) BM25 retrieval uses a Lucene index built by Pyserini over a
JsonCollection, where each record has:
  - "id": docid
  - "contents": text

For DRBench seed URLs, our retrieval units are *passage chunks*. This exporter
writes one JsonCollection record per chunk, with:
  - id = chunk_id (stable)
  - contents = chunk text
  - extra metadata fields copied from chunk["meta"] (kept in the stored "raw")

Indexing (requires Java):
  bash -lc 'source ~/initmamba.sh && /home/toolkit/.mamba/envs/vllm013/bin/python -m pyserini.index.lucene \
    --collection JsonCollection --input <out_dir> --index <index_dir> \
    --generator DefaultLuceneDocumentGenerator --threads 4 --storeRaw'
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from making_dataset.utils.progress import progress  # noqa: E402
from making_dataset_2.retrieval.chunks import iter_chunks  # noqa: E402

DEFAULT_CHUNKS = ROOT_DIR / "making_dataset_2" / "outputs" / "chunks_web_drbench_urls.jsonl"
DEFAULT_OUT_DIR = ROOT_DIR / "making_dataset_2" / "outputs" / "indexes" / "drbench_urls_bm25_collection"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export chunk JSONL to a Pyserini JsonCollection directory.")
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS), help="Input chunks JSONL")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for JsonCollection files")
    parser.add_argument(
        "--out-file",
        default="chunks.jsonl",
        help="Output filename within --out-dir (default: chunks.jsonl)",
    )
    parser.add_argument("--min-chars", type=int, default=50, help="Skip chunks with shorter contents")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on exported chunks")
    return parser.parse_args()


def _pick_meta(rec: dict[str, Any]) -> dict[str, Any]:
    meta = rec.get("meta")
    if not isinstance(meta, dict):
        return {}
    # Keep a conservative, stable set of fields.
    keep: dict[str, Any] = {}
    for k in ("web_pool", "url", "industry", "domain", "seed_date", "task_ids", "title"):
        if k in meta:
            keep[k] = meta[k]
    return keep


def main() -> int:
    args = _parse_args()
    chunks_path = Path(args.chunks)
    out_dir = Path(args.out_dir)
    out_path = out_dir / str(args.out_file)

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rec in progress(iter_chunks(chunks_path), total=None, desc="Export chunks"):
            chunk_id = str(rec.get("chunk_id") or "").strip()
            doc_id = str(rec.get("doc_id") or "").strip()
            text = str(rec.get("text") or "").strip()
            if not chunk_id or not doc_id or len(text) < int(args.min_chars):
                continue

            obj: dict[str, Any] = {
                "id": chunk_id,
                "contents": text,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
            }
            obj.update(_pick_meta(rec))

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_written += 1
            if args.limit is not None and n_written >= int(args.limit):
                break

    print(f"Wrote Pyserini JsonCollection file: {out_path} (n={n_written})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
