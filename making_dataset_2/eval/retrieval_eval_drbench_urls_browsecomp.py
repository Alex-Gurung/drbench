#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate DRBench seed-URL retrieval using BrowseComp-mirrored indexes.

This evaluates retrieval over *chunk-level* docids, where each indexed docid is a
`chunk_id` string. The relevant label is the parent `doc_id` (URL doc). A query
is counted as a hit if any retrieved chunk belongs to the relevant doc.

Backends:
- bm25: Pyserini/Lucene BM25 index (requires Java)
- dense: FAISS flat IP over precomputed Qwen3 embeddings (BrowseComp shard format)
- bm25_rerank_dense: BM25 recall then dense rerank those candidates
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Literal, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from making_dataset.utils.progress import progress  # noqa: E402
from making_dataset.index.web_bm25 import WebBM25Searcher  # noqa: E402
from making_dataset.index.web_dense import WebDenseSearcher  # noqa: E402
from making_dataset_2.drbench_urls import doc_id_for_url, load_seed_urls  # noqa: E402

Mode = Literal["bm25", "dense", "bm25_rerank_dense"]


DEFAULT_URLS_JSON = ROOT_DIR / "drbench" / "data" / "contexts" / "urls.json"
DEFAULT_BM25_INDEX = ROOT_DIR / "making_dataset_2" / "outputs" / "indexes" / "drbench_urls_bm25"
DEFAULT_DENSE_GLOB = (
    ROOT_DIR
    / "making_dataset_2"
    / "outputs"
    / "indexes"
    / "drbench_urls_qwen3_dense"
    / "corpus.shard*_of_*.pkl"
)


@dataclass(frozen=True)
class QueryCase:
    query: str
    target_doc_id: str
    industry: str
    domain: str
    url: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate drbench seed URL retrieval (BrowseComp-mirrored).")
    parser.add_argument("--urls-json", default=str(DEFAULT_URLS_JSON), help="drbench/data/contexts/urls.json")
    parser.add_argument(
        "--bm25-index",
        default=str(DEFAULT_BM25_INDEX),
        help="Pyserini Lucene index directory for drbench URL chunks",
    )
    parser.add_argument(
        "--dense-index-glob",
        default=str(DEFAULT_DENSE_GLOB),
        help="Dense shard glob (BrowseComp shard format)",
    )
    parser.add_argument(
        "--dense-model",
        default="Qwen/Qwen3-Embedding-4B",
        help="Dense embedding model (HF id or local snapshot path).",
    )
    parser.add_argument("--mode", choices=["bm25", "dense", "bm25_rerank_dense"], default="bm25_rerank_dense")
    parser.add_argument("--k", type=int, default=10, help="Top-k to retrieve/evaluate (default: 10)")
    parser.add_argument(
        "--bm25-candidates-k",
        type=int,
        default=200,
        help="BM25 candidates to dense-rerank (bm25_rerank_dense only).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of query cases")
    parser.add_argument("--verbose", action="store_true", help="Print per-query ranks")
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


def _doc_id_from_chunk_docid(docid: str) -> str:
    # Our BM25/dense docids are chunk_ids like: "<doc_id>#0007".
    return str(docid).split("#", 1)[0]


def main() -> int:
    args = _parse_args()
    urls_json = Path(args.urls_json)

    cases = _build_cases(urls_json)
    if args.limit is not None:
        cases = cases[: max(0, int(args.limit))]

    mode: Mode = args.mode  # type: ignore[assignment]

    bm25 = None
    dense = None
    if mode in {"bm25", "bm25_rerank_dense"}:
        bm25 = WebBM25Searcher(str(args.bm25_index))
    if mode in {"dense", "bm25_rerank_dense"}:
        dense = WebDenseSearcher(
            index_glob=str(args.dense_index_glob),
            model_name_or_path=str(args.dense_model),
            encoder_batch_size=8,
            doc_store=bm25,
        )

    cutoffs = [1, 5, 10]
    hits_at: dict[int, int] = {c: 0 for c in cutoffs}
    mrr10_sum = 0.0

    bucket_hits_at: DefaultDict[tuple[str, str], dict[int, int]] = defaultdict(lambda: {c: 0 for c in cutoffs})
    bucket_mrr10_sum: DefaultDict[tuple[str, str], float] = defaultdict(float)
    bucket_total: DefaultDict[tuple[str, str], int] = defaultdict(int)

    for case in progress(cases, total=len(cases), desc=f"Eval {mode}"):
        retrieved_docids: list[str] = []
        if mode == "bm25":
            assert bm25 is not None
            retrieved_docids = [h.docid for h in bm25.search(case.query, k=int(args.k))]
        elif mode == "dense":
            assert dense is not None
            retrieved_docids = [h.docid for h in dense.search(case.query, k=int(args.k))]
        else:
            assert bm25 is not None and dense is not None
            cand = bm25.search(case.query, k=max(int(args.bm25_candidates_k), int(args.k)))
            cand_docids = [h.docid for h in cand]
            reranked = dense.rerank_docids(case.query, cand_docids)[: int(args.k)]
            retrieved_docids = [h.docid for h in reranked]

        rank: Optional[int] = None
        for i, docid in enumerate(retrieved_docids, start=1):
            if _doc_id_from_chunk_docid(docid) == case.target_doc_id:
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
                        "top_chunk_docids": retrieved_docids[:10],
                    },
                    ensure_ascii=False,
                )
            )

    n = max(1, len(cases))
    print(f"Total queries: {len(cases)}")
    for c in cutoffs:
        print(f"recall@{c}: {hits_at[c] / n:.3f} ({hits_at[c]}/{n})")
    print(f"MRR@10: {mrr10_sum / n:.3f}")

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

