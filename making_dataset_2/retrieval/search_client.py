"""HTTP client for the BM25 search server.

Drop-in replacement for HybridSearcher — has the same .search() interface
so it can be passed directly to find_bridge as the searcher argument.

Usage:
    client = SearchClient("http://localhost:8100", pool="web")
    hits = client.search("some query", k=50, mode="bm25")
    # hits is list[RetrievalHit], same as HybridSearcher.search()
"""

from __future__ import annotations

import logging

import httpx

from making_dataset_2.retrieval.types import RetrievalHit

logger = logging.getLogger(__name__)


class SearchClient:
    """HTTP client that matches the HybridSearcher.search() interface."""

    def __init__(self, base_url: str, *, pool: str = "web", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.pool = pool
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        mode: str = "bm25",
        doc_ids: list[str] | None = None,
        **_kwargs,
    ) -> list[RetrievalHit]:
        body = {"query": query, "k": k, "mode": mode, "pool": self.pool}
        if doc_ids is not None:
            body["doc_ids"] = doc_ids
        resp = self._client.post("/search", json=body)
        resp.raise_for_status()
        data = resp.json()
        return [
            RetrievalHit(
                chunk_id=h["chunk_id"],
                doc_id=h["doc_id"],
                score=h["score"],
                text=h["text"],
                meta=h.get("meta"),
            )
            for h in data["hits"]
        ]

    def health(self) -> dict:
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self._client.close()

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass
