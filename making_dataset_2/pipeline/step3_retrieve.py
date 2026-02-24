"""Step 3: Retrieve Candidate Documents.

Two-stage hybrid retrieval. Searches one or more pools and merges results.
Filters out already-used doc_ids and short documents.
"""

from __future__ import annotations

from typing import Any

from making_dataset_2.retrieval.dense import EmbeddingBackend
from making_dataset_2.retrieval.hybrid import Mode, merge_results
from making_dataset_2.retrieval.types import RetrievalHit
from making_dataset_2.types import ChainState


def retrieve_candidates(
    state: ChainState,
    query: str,
    *,
    searchers: list[Any],  # HybridSearcher or BrowseCompSearcher (duck typed)
    embedder: EmbeddingBackend | None = None,
    mode: Mode = "hybrid",
    k: int = 20,
    bm25_k: int = 200,
    dense_k: int = 200,
    min_doc_len: int = 200,
) -> tuple[list[RetrievalHit], dict]:
    """Retrieve candidates from one or more pools, merge, and filter.

    Returns:
        (hits, trace_entry) — filtered hits and trace dict with query/counts/top hits.
    """
    per_pool: list[list[RetrievalHit]] = []
    for searcher in searchers:
        hits = searcher.search(
            query,
            k=k,
            mode=mode,
            embedder=embedder,
            bm25_k=bm25_k,
            dense_k=dense_k,
        )
        per_pool.append(hits)

    # Merge across pools (dedup by doc_id, keep highest score)
    merged = merge_results(*per_pool, k=k * 2)

    # Filter out used docs and short docs
    filtered: list[RetrievalHit] = []
    for hit in merged:
        if hit.doc_id in state.used_doc_ids:
            continue
        if len(hit.text or "") < min_doc_len:
            continue
        filtered.append(hit)
        if len(filtered) >= k:
            break

    trace = {
        "step": "step3_retrieve",
        "hop": state.current_hop + 1,
        "query": query,
        "n_merged": len(merged),
        "n_filtered": len(filtered),
        "top_hits": [
            {
                "doc_id": h.doc_id,
                "score": round(h.score, 4),
                "snippet": (h.text or "")[:200],
            }
            for h in filtered[:10]
        ],
    }
    return filtered, trace
