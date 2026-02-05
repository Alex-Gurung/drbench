#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from making_dataset.index.web_bm25 import WebBM25Searcher
from making_dataset.index.web_dense import WebDenseSearcher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare BrowseComp-Plus BM25 vs Dense retrieval using qrels (offline)."
    )
    parser.add_argument(
        "--queries",
        default="/home/toolkit/BrowseComp-Plus/topics-qrels/queries.tsv",
        help="TSV: qid<TAB>query",
    )
    parser.add_argument(
        "--qrels",
        default="/home/toolkit/BrowseComp-Plus/topics-qrels/qrel_golds.txt",
        help="TREC qrels file",
    )
    parser.add_argument(
        "--bm25-index",
        default="/home/toolkit/BrowseComp-Plus/indexes/bm25",
        help="BrowseComp-Plus BM25 (Pyserini/Lucene) index",
    )
    parser.add_argument(
        "--dense-index-glob",
        default="/home/toolkit/BrowseComp-Plus/indexes/qwen3-embedding-0.6b/corpus.shard*_of_4.pkl",
        help="Dense index shards glob",
    )
    parser.add_argument(
        "--dense-model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Dense embedding model (HF id; resolved via $TRANSFORMERS_CACHE)",
    )
    parser.add_argument(
        "--ks",
        default="5,10,100",
        help="Comma-separated recall cutoffs (e.g., 5,10,100,1000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Evaluate on first N queries (default 50 for a quick run)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle queries before taking --limit",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for --shuffle",
    )
    parser.add_argument(
        "--no-bm25",
        action="store_true",
        help="Skip BM25 (useful if you don't want to source initmamba.sh)",
    )
    parser.add_argument(
        "--no-dense",
        action="store_true",
        help="Skip Dense (useful if you only want BM25 metrics)",
    )
    return parser.parse_args()


def _load_queries(path: Path) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        qid, q = line.split("\t", 1)
        out.append((qid.strip(), q.strip()))
    return out


def _load_qrels(path: Path) -> Dict[str, set[str]]:
    rel: Dict[str, set[str]] = defaultdict(set)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # qid Q0 docid rel
        parts = line.split()
        if len(parts) < 4:
            continue
        qid, _, docid, score = parts[:4]
        if int(score) > 0:
            rel[qid].add(str(docid))
    return rel


def _rrf(docid_lists: List[List[str]], *, k: int, k0: int = 60) -> List[str]:
    acc: Dict[str, float] = {}
    for docids in docid_lists:
        for rank, docid in enumerate(docids, 1):
            acc[docid] = acc.get(docid, 0.0) + 1.0 / (k0 + rank)
    merged = sorted(acc.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [docid for docid, _ in merged]


def _recall_at_k(relevant: set[str], retrieved: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant) / float(len(relevant))


def main() -> int:
    args = _parse_args()
    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    if not ks:
        raise ValueError("--ks must be non-empty")
    max_k = max(ks)

    queries_path = Path(args.queries)
    qrels_path = Path(args.qrels)
    if not queries_path.exists():
        raise FileNotFoundError(f"queries not found: {queries_path}")
    if not qrels_path.exists():
        raise FileNotFoundError(f"qrels not found: {qrels_path}")

    queries = _load_queries(queries_path)
    qrels = _load_qrels(qrels_path)
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(queries)
    if args.limit is not None:
        queries = queries[: args.limit]

    bm25 = None if args.no_bm25 else WebBM25Searcher(args.bm25_index)
    dense = None
    if not args.no_dense:
        dense = WebDenseSearcher(
            index_glob=args.dense_index_glob,
            model_name_or_path=args.dense_model,
            encoder_batch_size=8,
        )

    totals = {m: {k: 0.0 for k in ks} for m in ("bm25", "dense", "hybrid_rrf")}
    counts = 0

    t0 = time.time()
    for qid, query in tqdm(queries, desc="Evaluating", unit="q"):
        relevant = qrels.get(qid, set())
        if not relevant:
            continue

        bm25_docids: List[str] = []
        dense_docids: List[str] = []

        if bm25 is not None:
            bm25_docids = [h.docid for h in bm25.search(query, k=max_k)]
            for k in ks:
                totals["bm25"][k] += _recall_at_k(relevant, bm25_docids, k)

        if dense is not None:
            dense_docids = [h.docid for h in dense.search(query, k=max_k)]
            for k in ks:
                totals["dense"][k] += _recall_at_k(relevant, dense_docids, k)

        if bm25 is not None and dense is not None:
            fused = _rrf([bm25_docids, dense_docids], k=max_k)
            for k in ks:
                totals["hybrid_rrf"][k] += _recall_at_k(relevant, fused, k)

        counts += 1

    dt = time.time() - t0
    if counts == 0:
        raise ValueError("No evaluated queries had qrels.")

    print(f"Evaluated queries: {counts} (elapsed {dt:.1f}s)")
    print()
    print("| Method | " + " | ".join([f"recall@{k}" for k in ks]) + " |")
    print("| --- | " + " | ".join(["---:"] * len(ks)) + " |")
    for method in ("bm25", "dense", "hybrid_rrf"):
        if method == "bm25" and bm25 is None:
            continue
        if method == "dense" and dense is None:
            continue
        if method == "hybrid_rrf" and (bm25 is None or dense is None):
            continue
        row = [f"{totals[method][k] / counts:.4f}" for k in ks]
        print(f"| {method} | " + " | ".join(row) + " |")

    print()
    print("Notes:")
    print("- This script is for quick offline sanity-checks; full IR eval uses pyserini/tevatron tooling.")
    print("- BM25 requires Java/Pyserini; run via: bash -lc 'source ~/initmamba.sh && ...'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
