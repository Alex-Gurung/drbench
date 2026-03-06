"""BM25 search server — holds indexes in one process, serves over HTTP.

Avoids duplicating 10-20GB of BM25 indexes across parallel chain builder
workers or RL training jobs. Supports optional doc_id filtering.

Usage:
    python -m making_dataset_2.retrieval.search_server \
        --chunks-local making_dataset/outputs/chunks_local.jsonl \
        --chunks-web making_dataset/outputs/chunks_web.jsonl \
                     making_dataset_2/outputs/chunks_web_drbench_urls.jsonl \
        --port 8100

    # Then in chain_builder:
    python -m making_dataset_2.pipeline.chain_builder \
        --search-url http://localhost:8100 ...
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from making_dataset_2.retrieval.hybrid import HybridSearcher

logger = logging.getLogger(__name__)

app = FastAPI(title="BM25 Search Server")

# Populated at startup
_searchers: dict[str, HybridSearcher] = {}


class SearchRequest(BaseModel):
    query: str
    k: int = 50
    mode: str = "bm25"
    pool: str = "web"  # "local" or "web"
    doc_ids: list[str] | None = None  # optional post-filter (not used by current pipeline)


class HitResponse(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    text: str
    meta: dict | None = None


class SearchResponse(BaseModel):
    hits: list[HitResponse]
    pool: str
    query: str
    elapsed_ms: float


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    searcher = _searchers.get(req.pool)
    if searcher is None:
        return SearchResponse(hits=[], pool=req.pool, query=req.query, elapsed_ms=0)

    t0 = time.time()
    # Over-retrieve if filtering, so we still get k results after filter.
    # Note: this is a heuristic — if the allowed docs are rare in the global
    # ranking, valid hits can still be missed. For small pools, consider
    # building per-pool sub-indexes instead.
    fetch_k = req.k * 10 if req.doc_ids else req.k
    raw_hits = searcher.search(req.query, k=fetch_k, mode=req.mode)

    if req.doc_ids:
        allowed = set(req.doc_ids)
        raw_hits = [h for h in raw_hits if h.doc_id in allowed][:req.k]

    elapsed_ms = round((time.time() - t0) * 1000, 1)
    hits = [
        HitResponse(
            chunk_id=h.chunk_id,
            doc_id=h.doc_id,
            score=h.score,
            text=h.text,
            meta=h.meta,
        )
        for h in raw_hits[:req.k]
    ]
    return SearchResponse(hits=hits, pool=req.pool, query=req.query, elapsed_ms=elapsed_ms)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pools": {name: s.size for name, s in _searchers.items()},
    }


def _parse_args():
    p = argparse.ArgumentParser(description="BM25 search server")
    p.add_argument("--chunks-local", type=str, default=None)
    p.add_argument("--chunks-web", nargs="+", default=[])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8100)
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.chunks_local:
        path = Path(args.chunks_local)
        if path.exists():
            logger.info("Loading local chunks from %s ...", path)
            t0 = time.time()
            _searchers["local"] = HybridSearcher(chunks_path=path)
            logger.info("Local searcher ready: %d chunks (%.1fs)",
                        _searchers["local"].size, time.time() - t0)

    if args.chunks_web:
        paths = [Path(p) for p in args.chunks_web if Path(p).exists()]
        if paths:
            logger.info("Loading web chunks from %d files ...", len(paths))
            t0 = time.time()
            _searchers["web"] = HybridSearcher.from_paths(paths)
            logger.info("Web searcher ready: %d chunks (%.1fs)",
                        _searchers["web"].size, time.time() - t0)

    logger.info("Starting server on %s:%d with pools: %s",
                args.host, args.port, list(_searchers.keys()))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
