#!/usr/bin/env python3
from __future__ import annotations

"""Build a dense embedding index for a chunk pool.

This writes a `.npz` with:
- `embeddings`: float32 matrix (n_chunks, dim), L2-normalized
- `chunk_ids`: array of chunk ids
- `doc_ids`: array of doc ids

If you want to *mirror BrowseComp* (pickle shards + `WebDenseSearcher`), use:
  `making_dataset_2/retrieval/build_qwen3_dense_shards.py`

IMPORTANT: Use the same embedding model across all pools (drbench_urls,
BrowseComp, local) so that dense cosine scores are comparable when merging
results. Default: Qwen/Qwen3-Embedding-4B.

Backends:
- `local_qwen3` (recommended): loads Qwen3-Embedding-4B locally via transformers.
  Same model used at search time by LocalQwen3Backend and BrowseCompSearcher.
  Use --device cpu to avoid GPU contention with vLLM.
- `openai_compatible`: calls a vLLM or OpenAI-compatible endpoint.
- `sentence_transformers`: local inference via sentence-transformers.
- `hash`: deterministic offline (low quality, for smoke tests only).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from making_dataset.utils.progress import progress  # noqa: E402
from making_dataset_2.retrieval.chunks import iter_chunks  # noqa: E402
from making_dataset_2.retrieval.dense import (  # noqa: E402
    DenseIndex,
    HashEmbeddingBackend,
    OpenAICompatibleBackend,
    SentenceTransformerBackend,
)


DEFAULT_CHUNKS = ROOT_DIR / "making_dataset_2" / "outputs" / "chunks_web_drbench_urls.jsonl"
DEFAULT_OUTPUT = ROOT_DIR / "making_dataset_2" / "outputs" / "indexes" / "drbench_urls_dense.npz"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dense embedding index for drbench URL chunks.")
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS), help="Chunks JSONL to embed")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output .npz path")
    parser.add_argument(
        "--backend",
        choices=["local_qwen3", "openai_compatible", "sentence_transformers", "hash"],
        default="local_qwen3",
        help=(
            "Embedding backend. 'local_qwen3' (default) loads Qwen3-Embedding-4B "
            "locally. 'openai_compatible' calls a vLLM endpoint. "
            "'sentence_transformers' runs locally. "
            "'hash' is deterministic offline (low quality, for smoke tests)."
        ),
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Embedding-4B",
        help="Embedding model name/path (default: Qwen3-Embedding-4B).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base_url (e.g., http://127.0.0.1:8000/v1 for vLLM).",
    )
    parser.add_argument("--api-key", default=None, help="API key (optional; falls back to env vars).")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for API embedding calls.")
    parser.add_argument("--hash-dim", type=int, default=256, help="Dimensionality for --backend hash")
    parser.add_argument("--device", default=None, help="Device for local_qwen3 backend (cpu, cuda)")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    chunks_path = Path(args.chunks)
    output_path = Path(args.output)

    chunk_ids: list[str] = []
    doc_ids: list[str] = []
    texts: list[str] = []
    for rec in progress(iter_chunks(chunks_path), total=None, desc="Load chunks"):
        text = (rec.get("text") or "").strip()
        cid = rec.get("chunk_id")
        did = rec.get("doc_id")
        if not text or not cid or not did:
            continue
        chunk_ids.append(str(cid))
        doc_ids.append(str(did))
        texts.append(text)

    if not texts:
        raise ValueError(f"No chunk texts found in {chunks_path}")

    if args.backend == "local_qwen3":
        from making_dataset.index.web_dense import Qwen3EosEmbedder
        from making_dataset_2.retrieval.dense import QWEN3_QUERY_PREFIX, LocalQwen3Backend
        qwen3 = Qwen3EosEmbedder(
            model_name_or_path=args.model,
            query_prefix=QWEN3_QUERY_PREFIX,
            device=args.device,
        )
        backend = LocalQwen3Backend(qwen3)
        print(f"Using {args.model} on {qwen3.device}")
    elif args.backend == "hash":
        backend = HashEmbeddingBackend(dim=int(args.hash_dim))
    elif args.backend == "sentence_transformers":
        backend = SentenceTransformerBackend(args.model)
    else:
        backend = OpenAICompatibleBackend(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            batch_size=int(args.batch_size),
        )

    embeddings = backend.embed_texts(texts)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    index = DenseIndex(embeddings=embeddings, chunk_ids=chunk_ids, doc_ids=doc_ids)
    index.save_npz(output_path)
    print(f"Wrote dense index: {output_path} (n={index.size}, dim={index.dim}, backend={args.backend})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
