#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate retrieval quality for the DRBench seed-URL pool.

Gold labels are derived automatically from `drbench/data/contexts/urls.json`:
- Each `example_dr_questions[]` entry is treated as a query.
- The corresponding URL doc is treated as relevant.

Metrics:
- recall@1/5/10: hit if any retrieved chunk belongs to the relevant doc_id
- MRR@10
- Breakdown by (industry, domain)
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Literal, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from making_dataset.utils.progress import progress  # noqa: E402
from making_dataset_2.drbench_urls import doc_id_for_url, load_seed_urls  # noqa: E402
from making_dataset_2.retrieval.dense import (  # noqa: E402
    HashEmbeddingBackend,
    OpenAICompatibleBackend,
    SentenceTransformerBackend,
)
from making_dataset_2.retrieval.hybrid import HybridSearcher, Mode  # noqa: E402


DEFAULT_URLS_JSON = ROOT_DIR / "drbench" / "data" / "contexts" / "urls.json"
DEFAULT_CHUNKS = ROOT_DIR / "making_dataset_2" / "outputs" / "chunks_web_drbench_urls.jsonl"
DEFAULT_DENSE_INDEX = ROOT_DIR / "making_dataset_2" / "outputs" / "indexes" / "drbench_urls_dense.npz"


@dataclass(frozen=True)
class QueryCase:
    query: str
    target_doc_id: str
    industry: str
    domain: str
    url: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on drbench seed URL pool.")
    parser.add_argument("--urls-json", default=str(DEFAULT_URLS_JSON), help="drbench/data/contexts/urls.json")
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS), help="Chunks JSONL for drbench URL pool")
    parser.add_argument(
        "--dense-index",
        default=str(DEFAULT_DENSE_INDEX),
        help="Dense index .npz (required for --mode dense/hybrid)",
    )
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--k", type=int, default=10, help="Top-k to retrieve/evaluate (default: 10)")
    parser.add_argument("--bm25-k", type=int, default=200, help="BM25 candidates for hybrid (default: 200)")
    parser.add_argument("--dense-k", type=int, default=200, help="Dense candidates for hybrid (default: 200)")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of query cases")
    parser.add_argument("--verbose", action="store_true", help="Print per-query ranks")

    parser.add_argument(
        "--embedder-backend",
        choices=["openai_compatible", "sentence_transformers", "hash"],
        default="openai_compatible",
        help="Embedder backend for query embeddings (dense/hybrid only).",
    )
    parser.add_argument(
        "--embedder-model",
        default="Qwen/Qwen3-Embedding-4B",
        help="Embedder model name/path (default: Qwen3-Embedding-4B).",
    )
    parser.add_argument("--embedder-base-url", default=None, help="OpenAI-compatible base_url (for vLLM/proxies).")
    parser.add_argument("--embedder-api-key", default=None, help="API key (optional; falls back to env vars).")
    parser.add_argument("--hash-dim", type=int, default=256, help="Dimensionality for --embedder-backend hash")

    return parser.parse_args()


def _build_cases(urls_json: Path) -> list[QueryCase]:
    seeds = load_seed_urls(urls_json)
    cases: list[QueryCase] = []
    for row in seeds:
        url = (row.get("url") or "").strip()
        if not url:
            continue
        target_doc_id = doc_id_for_url(url)
        industry = str(row.get("industry") or "")
        domain = str(row.get("domain") or "")
        for q in row.get("example_dr_questions") or []:
            q = str(q).strip()
            if not q:
                continue
            cases.append(
                QueryCase(
                    query=q,
                    target_doc_id=target_doc_id,
                    industry=industry,
                    domain=domain,
                    url=url,
                )
            )
    return cases


def _build_embedder(args) -> Any:
    if args.embedder_backend == "hash":
        return HashEmbeddingBackend(dim=int(args.hash_dim))
    if args.embedder_backend == "sentence_transformers":
        return SentenceTransformerBackend(args.embedder_model)
    return OpenAICompatibleBackend(
        model=args.embedder_model,
        api_key=args.embedder_api_key,
        base_url=args.embedder_base_url,
    )


def main() -> int:
    args = _parse_args()
    urls_json = Path(args.urls_json)
    chunks_path = Path(args.chunks)
    dense_index_path = Path(args.dense_index)

    cases = _build_cases(urls_json)
    if args.limit is not None:
        cases = cases[: max(0, int(args.limit))]

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    mode: Mode = args.mode  # type: ignore[assignment]
    embedder = None
    dense_path = None
    if mode in {"dense", "hybrid"}:
        if not dense_index_path.exists():
            raise FileNotFoundError(f"Dense index not found: {dense_index_path}")
        dense_path = dense_index_path
        embedder = _build_embedder(args)

    searcher = HybridSearcher(chunks_path=chunks_path, dense_index_path=dense_path)

    cutoffs = [1, 5, 10]
    hits_at: dict[int, int] = {c: 0 for c in cutoffs}
    mrr10_sum = 0.0

    # Breakdown by (industry, domain)
    bucket_hits_at: DefaultDict[tuple[str, str], dict[int, int]] = defaultdict(lambda: {c: 0 for c in cutoffs})
    bucket_mrr10_sum: DefaultDict[tuple[str, str], float] = defaultdict(float)
    bucket_total: DefaultDict[tuple[str, str], int] = defaultdict(int)

    for case in progress(cases, total=len(cases), desc=f"Eval {mode}"):
        hits = searcher.search(
            case.query,
            k=int(args.k),
            mode=mode,
            embedder=embedder,
            bm25_k=int(args.bm25_k),
            dense_k=int(args.dense_k),
        )

        rank: Optional[int] = None
        for i, h in enumerate(hits, start=1):
            if h.doc_id == case.target_doc_id:
                rank = i
                break

        key = (case.industry, case.domain)
        bucket_total[key] += 1

        if rank is not None:
            for c in cutoffs:
                if rank <= c:
                    hits_at[c] += 1
                    bucket_hits_at[key][c] += 1
            if rank <= 10:
                mrr10_sum += 1.0 / rank
                bucket_mrr10_sum[key] += 1.0 / rank

        if args.verbose:
            print(
                json.dumps(
                    {
                        "query": case.query,
                        "url": case.url,
                        "target_doc_id": case.target_doc_id,
                        "rank": rank,
                        "top_doc_ids": [h.doc_id for h in hits[:10]],
                    },
                    ensure_ascii=False,
                )
            )

    n = max(1, len(cases))
    print(f"Total queries: {len(cases)}")
    for c in cutoffs:
        print(f"recall@{c}: {hits_at[c] / n:.3f} ({hits_at[c]}/{n})")
    print(f"MRR@10: {mrr10_sum / n:.3f}")

    # Breakdown
    print("\nBreakdown by industry/domain:")
    for (industry, domain), total in sorted(bucket_total.items(), key=lambda kv: kv[0]):
        if total <= 0:
            continue
        recs = {c: bucket_hits_at[(industry, domain)][c] / total for c in cutoffs}
        mrr = bucket_mrr10_sum[(industry, domain)] / total
        print(
            f"- {industry}/{domain}: n={total} "
            + " ".join([f"r@{c}={recs[c]:.3f}" for c in cutoffs])
            + f" mrr10={mrr:.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
