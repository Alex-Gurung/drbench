#!/usr/bin/env python3
"""Test Step 2: Query Generation Quality.

For N random seeds, generate search queries and check if retrieval
returns topically relevant documents. Reports per-query results and
aggregate stats.

Usage:
    python -m making_dataset_2.eval.test_step2_queries \
        --model Qwen/Qwen3-30B \
        --base-url http://127.0.0.1:8000/v1 \
        --n 20 --verbose
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from making_dataset_2.data_loading import (
    build_doc_lookup,
    filter_seed_secrets,
    load_chunks_local,
    load_eval_seeds,
    load_secrets,
)
from making_dataset_2.llm import LLMClient
from making_dataset_2.pipeline.step1_seed import select_seed
from making_dataset_2.pipeline.step2_query import generate_query
from making_dataset_2.pipeline.step3_retrieve import retrieve_candidates
from making_dataset_2.retrieval.dense import (
    QWEN3_QUERY_PREFIX,
    EmbeddingBackend,
    LocalQwen3Backend,
)
from making_dataset_2.retrieval.hybrid import HybridSearcher

DEFAULT_CHUNKS_LOCAL = ROOT_DIR / "making_dataset" / "outputs" / "chunks_local.jsonl"
DEFAULT_CHUNKS_WEB = ROOT_DIR / "making_dataset_2" / "outputs" / "chunks_web_drbench_urls.jsonl"
DEFAULT_DENSE_LOCAL = ROOT_DIR / "making_dataset_2" / "outputs" / "indexes" / "local_dense.npz"
DEFAULT_DENSE_WEB = ROOT_DIR / "making_dataset_2" / "outputs" / "indexes" / "drbench_urls_dense.npz"

# BrowseComp defaults
DEFAULT_BROWSECOMP_BM25 = "/home/toolkit/BrowseComp-Plus/indexes/bm25/"
DEFAULT_BROWSECOMP_DENSE = "/home/toolkit/BrowseComp-Plus/indexes/qwen3-embedding-4b/corpus.shard*.pkl"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test Step 2: query generation quality.")
    p.add_argument("--model", required=True, help="LLM model name")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    p.add_argument("--api-key", default=None)
    p.add_argument("--n", type=int, default=20, help="Number of seeds to test")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--pattern", default="LW", help="Chain pattern (first hop is always L)")
    p.add_argument("--task", default=None, help="Filter by task ID")
    p.add_argument("--company", default=None, help="Filter by company")
    p.add_argument("--seed-source", choices=["eval", "inventory"], default="eval",
                    help="Seed source: eval (hand-picked) or inventory (LLM-generated)")

    # Retrieval
    p.add_argument("--retrieval-mode", choices=["bm25", "dense", "hybrid"], default="bm25")
    p.add_argument("--chunks-local", default=str(DEFAULT_CHUNKS_LOCAL))
    p.add_argument("--chunks-web", default=str(DEFAULT_CHUNKS_WEB))
    p.add_argument("--dense-local", default=str(DEFAULT_DENSE_LOCAL))
    p.add_argument("--dense-web", default=str(DEFAULT_DENSE_WEB))
    p.add_argument("--embedder-model", default="Qwen/Qwen3-Embedding-4B")
    p.add_argument("--embedder-device", default=None, help="Device for embedding model (cpu, cuda)")

    # BrowseComp web pool
    p.add_argument("--browsecomp-bm25", default=DEFAULT_BROWSECOMP_BM25, help="BrowseComp BM25 index path")
    p.add_argument("--browsecomp-dense", default=DEFAULT_BROWSECOMP_DENSE, help="BrowseComp dense shard glob")
    p.add_argument("--no-browsecomp", action="store_true", help="Disable BrowseComp pool")

    p.add_argument("--k", type=int, default=10, help="Top-k retrieval results")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", default=None, help="Optional JSONL output file")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    rng = random.Random(args.seed)

    # Load data
    chunks = load_chunks_local(Path(args.chunks_local))
    doc_lookup = build_doc_lookup(chunks)

    if args.seed_source == "eval":
        eligible = load_eval_seeds(doc_lookup, task_id=args.task, company=args.company)
        print(f"Eval seeds: {len(eligible)}")
    else:
        secrets = load_secrets()
        eligible = filter_seed_secrets(secrets, doc_lookup, task_id=args.task, company=args.company)
        print(f"Inventory seeds: {len(eligible)}")

    # LLM
    llm = LLMClient(model=args.model, base_url=args.base_url, api_key=args.api_key)

    # Shared Qwen3 embedder (one model instance for all sources)
    qwen3_embedder = None
    embedder: EmbeddingBackend | None = None
    if args.retrieval_mode in ("dense", "hybrid"):
        from making_dataset.index.web_dense import Qwen3EosEmbedder
        query_prefix = QWEN3_QUERY_PREFIX if "qwen3" in args.embedder_model.lower() else ""
        qwen3_embedder = Qwen3EosEmbedder(
            model_name_or_path=args.embedder_model,
            query_prefix=query_prefix,
            device=args.embedder_device,
        )
        embedder = LocalQwen3Backend(qwen3_embedder)
        print(f"Loaded {args.embedder_model} on {qwen3_embedder.device}")

    # Searchers — test both directions
    local_searchers: list = []
    web_searchers: list = []

    cl = Path(args.chunks_local)
    dl = Path(args.dense_local)
    if cl.exists():
        local_searchers.append(HybridSearcher(
            chunks_path=cl,
            dense_index_path=dl if dl.exists() else None,
        ))
        print(f"Local pool: {local_searchers[0].size} chunks")

    cw = Path(args.chunks_web)
    dw = Path(args.dense_web)
    if cw.exists():
        s = HybridSearcher(
            chunks_path=cw,
            dense_index_path=dw if dw.exists() else None,
        )
        web_searchers.append(s)
        print(f"DRBench-URLs pool: {s.size} chunks")

    # BrowseComp pool (shares the same Qwen3 embedder)
    if not args.no_browsecomp:
        bm25_path = Path(args.browsecomp_bm25)
        if bm25_path.exists():
            from making_dataset_2.retrieval.browsecomp import BrowseCompSearcher
            dense_glob = args.browsecomp_dense if args.retrieval_mode != "bm25" else None
            bc = BrowseCompSearcher(
                bm25_index=args.browsecomp_bm25,
                dense_shard_glob=dense_glob,
                model=args.embedder_model,
                embedder=qwen3_embedder,
            )
            web_searchers.append(bc)
            print(f"BrowseComp pool: {bc.size} docs")
        else:
            print(f"WARNING: BrowseComp BM25 index not found at {bm25_path}")

    # Determine which direction to test based on pattern
    # First hop is always L (seed), so next hop type tells us search direction
    next_hop = args.pattern[1] if len(args.pattern) > 1 else "W"
    target_corpus = "local" if next_hop == "L" else "web"
    searchers = local_searchers if next_hop == "L" else web_searchers
    print(f"Testing: L-seed → {next_hop}-search (target_corpus={target_corpus})")

    if not searchers:
        print(f"ERROR: No {target_corpus} searcher available")
        return 1

    # Run tests
    results = []
    rng_seeds = rng.sample(range(len(eligible)), min(args.n, len(eligible)))

    for i, idx in enumerate(rng_seeds):
        secret = eligible[idx]
        doc = doc_lookup[secret.doc_id]

        from making_dataset_2.types import ChainState, HopRecord
        state = ChainState(
            pattern=args.pattern,
            hop_history=[HopRecord(
                hop_number=1, hop_type="L",
                question=secret.question, answer=secret.answer,
                doc_id=secret.doc_id, doc_text=doc.text,
            )],
            global_question=secret.question,
            global_answer=secret.answer,
            used_doc_ids={secret.doc_id},
            task_id=doc.meta.get("task_id"),
            company=doc.meta.get("company_name"),
        )

        # Step 2: generate query
        query, raw_reasoning = generate_query(state, llm, target_corpus=target_corpus)

        # Step 3: retrieve
        hits = retrieve_candidates(
            state, query,
            searchers=searchers,
            embedder=embedder,
            mode=args.retrieval_mode,
            k=args.k,
        )

        row = {
            "idx": i,
            "seed_q": secret.question,
            "seed_a": secret.answer,
            "company": state.company,
            "task_id": state.task_id,
            "query": query,
            "reasoning": raw_reasoning,
            "n_hits": len(hits),
            "top_doc_ids": [h.doc_id for h in hits[:5]],
            "top_scores": [round(h.score, 4) for h in hits[:5]],
            "top_snippets": [(h.text or "")[:120] for h in hits[:3]],
        }
        results.append(row)

        # Pretty output
        ok = len(hits) > 0
        tag = f"\033[32m OK \033[0m" if ok else f"\033[31mEMPT\033[0m"
        print(f"\n\033[1m[{i+1}/{args.n}]\033[0m {tag}  \033[36m{state.task_id or '?'}\033[0m / {state.company or '?'}")
        print(f"  seed:  {secret.question}")
        print(f"  ans:   \033[33m{secret.answer}\033[0m")
        print(f"  query: \033[1m{query}\033[0m")
        print(f"  hits:  {len(hits)}")

        if args.verbose and hits:
            for j, h in enumerate(hits[:5]):
                snippet = (h.text or "")[:120].replace("\n", " ")
                src = h.doc_id[:40]
                print(f"    {j+1}. [{h.score:.3f}] \033[2m{src}\033[0m")
                print(f"       {snippet}")

    # Summary
    n = len(results)
    has_hits = sum(1 for r in results if r["n_hits"] > 0)
    avg_hits = sum(r["n_hits"] for r in results) / max(n, 1)

    # Unique doc_ids across all results
    all_doc_ids = set()
    for r in results:
        all_doc_ids.update(r["top_doc_ids"])

    print(f"\n\033[1m{'='*70}\033[0m")
    print(f"\033[1mSummary\033[0m")
    print(f"{'='*70}")
    print(f"  Model:          {args.model}")
    print(f"  Retrieval:      {args.retrieval_mode} | L->{next_hop} | k={args.k}")
    print(f"  Queries w/hits: {has_hits}/{n} ({has_hits/max(n,1)*100:.0f}%)")
    print(f"  Avg hits/query: {avg_hits:.1f}")
    print(f"  Unique doc_ids: {len(all_doc_ids)}")

    # Per-company breakdown
    by_company: dict[str, list] = {}
    for r in results:
        c = r.get("company") or "unknown"
        by_company.setdefault(c, []).append(r)
    if len(by_company) > 1:
        print(f"\n  Per company:")
        for company, rows in sorted(by_company.items()):
            co_hits = sum(1 for r in rows if r["n_hits"] > 0)
            print(f"    {company:30s} {co_hits}/{len(rows)} with hits")

    print(f"{'='*70}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
