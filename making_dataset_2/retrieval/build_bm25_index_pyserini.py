#!/usr/bin/env python3
from __future__ import annotations

"""Build a Pyserini/Lucene BM25 index from a JsonCollection directory.

This is a thin wrapper around:
  python -m pyserini.index.lucene ...

It exists mostly to keep the DRBench URL pool pipeline self-contained and to
make the expected arguments obvious.

NOTE: Pyserini requires a working Java toolchain (javac). In this environment,
run via:
  bash -lc 'source ~/initmamba.sh && /home/toolkit/.mamba/envs/vllm013/bin/python -m making_dataset_2.retrieval.build_bm25_index_pyserini ...'
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

DEFAULT_COLLECTION_DIR = ROOT_DIR / "making_dataset_2" / "outputs" / "indexes" / "drbench_urls_bm25_collection"
DEFAULT_INDEX_DIR = ROOT_DIR / "making_dataset_2" / "outputs" / "indexes" / "drbench_urls_bm25"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Pyserini Lucene index (BM25) from a JsonCollection dir.")
    parser.add_argument(
        "--collection-dir",
        default=str(DEFAULT_COLLECTION_DIR),
        help="Directory containing JsonCollection JSONL files (records with id/contents).",
    )
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="Output index directory.")
    parser.add_argument("--threads", type=int, default=4, help="Indexing threads (default: 4)")
    parser.add_argument(
        "--store-positions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Store term positions (not required for basic BM25).",
    )
    parser.add_argument(
        "--store-docvectors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Store doc vectors (not required for basic BM25).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing index dir first (DANGEROUS).",
    )
    return parser.parse_args()


def _require_javac() -> None:
    if shutil.which("javac") is not None:
        return
    raise RuntimeError(
        "javac not found on PATH. Run this via:\n"
        "  bash -lc 'source ~/initmamba.sh && /home/toolkit/.mamba/envs/vllm013/bin/python -m making_dataset_2.retrieval.build_bm25_index_pyserini ...'\n"
    )


def main() -> int:
    args = _parse_args()
    _require_javac()

    collection_dir = Path(args.collection_dir)
    index_dir = Path(args.index_dir)

    if not collection_dir.exists():
        raise FileNotFoundError(f"Collection dir not found: {collection_dir}")

    if index_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Index dir already exists: {index_dir} (pass --overwrite to delete it)")
        shutil.rmtree(index_dir)

    cmd: list[str] = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(collection_dir),
        "--index",
        str(index_dir),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(int(args.threads)),
        "--storeRaw",
    ]
    if args.store_positions:
        cmd.append("--storePositions")
    if args.store_docvectors:
        cmd.append("--storeDocvectors")

    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

