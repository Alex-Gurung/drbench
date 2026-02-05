from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from .local_bm25 import BM25Index
from .web_bm25 import WebBM25Searcher

Corpus = Literal["local", "web", "all"]
WebBackend = Literal["bm25", "dense", "bm25_rerank_dense"]


@dataclass(frozen=True)
class UnifiedHit:
    chunk_id: str
    doc_id: str
    source_type: Literal["local", "web"]
    score: float
    text: str


def _rrf_fuse(
    ranked_lists: list[list[UnifiedHit]],
    *,
    k: int,
    k0: int = 60,
) -> list[UnifiedHit]:
    # Reciprocal Rank Fusion; avoids comparing BM25 scores across corpora.
    acc: dict[str, float] = {}
    best: dict[str, UnifiedHit] = {}
    for hits in ranked_lists:
        for rank, h in enumerate(hits, 1):
            key = h.chunk_id
            acc[key] = acc.get(key, 0.0) + 1.0 / (k0 + rank)
            best.setdefault(key, h)
    merged = sorted(acc.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [best[cid] for cid, _ in merged]


class UnifiedSearcher:
    def __init__(
        self,
        *,
        local_chunks_path: str,
        web_bm25_index_path: Optional[str] = None,
        web_dense_index_glob: Optional[str] = None,
        web_dense_model_name_or_path: str = "Qwen/Qwen3-Embedding-0.6B",
        web_dense_query_max_len: int = 512,
        web_dense_encoder_batch_size: int = 8,
        local_bm25_k1: float = 1.5,
        local_bm25_b: float = 0.75,
    ) -> None:
        self.local_chunks_path = Path(local_chunks_path)
        if not self.local_chunks_path.exists():
            raise FileNotFoundError(f"Local chunks not found: {self.local_chunks_path}")

        self._local_chunks: list[dict[str, Any]] = []
        self._local_texts: list[str] = []
        self._local_chunk_id_by_idx: list[str] = []
        self._local_doc_id_by_idx: list[str] = []

        with self.local_chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("source_type") != "local":
                    continue
                text = (obj.get("text") or "").strip()
                if not text:
                    continue
                chunk_id = obj.get("chunk_id")
                doc_id = obj.get("doc_id")
                if not chunk_id or not doc_id:
                    raise ValueError("Local chunk missing chunk_id/doc_id")
                self._local_chunks.append(obj)
                self._local_texts.append(text)
                self._local_chunk_id_by_idx.append(chunk_id)
                self._local_doc_id_by_idx.append(doc_id)

        if not self._local_texts:
            raise ValueError("No local chunks loaded (empty index).")

        self.local_bm25 = BM25Index(self._local_texts, k1=local_bm25_k1, b=local_bm25_b)

        # Web search is optional (for local_only mode)
        self.web_bm25 = None
        self.web_dense = None
        if web_bm25_index_path is not None:
            self.web_bm25 = WebBM25Searcher(web_bm25_index_path)
        if web_dense_index_glob is not None:
            from .web_dense import WebDenseSearcher  # local import (heavy)

            # Use BM25 index as a doc store for dense hits (docid -> contents).
            self.web_dense = WebDenseSearcher(
                index_glob=web_dense_index_glob,
                model_name_or_path=web_dense_model_name_or_path,
                query_max_len=web_dense_query_max_len,
                encoder_batch_size=web_dense_encoder_batch_size,
                doc_store=self.web_bm25,
            )

    def search(
        self,
        query: str,
        *,
        corpus: Corpus = "all",
        k: int = 10,
        web_backend: WebBackend = "bm25",
        web_bm25_candidates_k: int = 200,
    ) -> list[UnifiedHit]:
        if corpus not in {"local", "web", "all"}:
            raise ValueError(f"Invalid corpus: {corpus}")
        if k <= 0:
            return []
        if web_backend not in {"bm25", "dense", "bm25_rerank_dense"}:
            raise ValueError(f"Invalid web_backend: {web_backend}")

        local_hits: list[UnifiedHit] = []
        web_hits: list[UnifiedHit] = []

        if corpus in {"local", "all"}:
            for hit in self.local_bm25.search(query, k=k):
                idx = hit.doc_idx
                local_hits.append(
                    UnifiedHit(
                        chunk_id=self._local_chunk_id_by_idx[idx],
                        doc_id=self._local_doc_id_by_idx[idx],
                        source_type="local",
                        score=hit.score,
                        text=self._local_texts[idx],
                    )
                )

        if corpus in {"web", "all"}:
            if self.web_bm25 is None:
                if corpus == "web":
                    raise ValueError("Web search requested but web_bm25 not initialized")
                # For "all" corpus, just skip web search silently
            elif web_backend == "bm25":
                for hit in self.web_bm25.search(query, k=k):
                    docid = hit.docid
                    doc_id = f"web/{docid}"
                    web_hits.append(
                        UnifiedHit(
                            chunk_id=f"{doc_id}#0001",
                            doc_id=doc_id,
                            source_type="web",
                            score=hit.score,
                            text=hit.text,
                        )
                    )
            else:
                if self.web_dense is None:
                    raise ValueError(
                        "web_dense is not initialized. Pass web_dense_index_glob=... when creating UnifiedSearcher."
                    )
                if web_backend == "dense":
                    dense_hits = self.web_dense.search(query, k=k)
                else:
                    bm25_candidates = self.web_bm25.search(query, k=max(web_bm25_candidates_k, k))
                    cand_docids = [h.docid for h in bm25_candidates]
                    dense_hits = self.web_dense.rerank_docids(query, cand_docids)[:k]

                for hit in dense_hits:
                    docid = hit.docid
                    doc_id = f"web/{docid}"
                    doc = self.web_bm25.get_document(docid)
                    if doc is None:
                        raise ValueError(f"BM25 doc store missing web docid={docid}")
                    web_hits.append(
                        UnifiedHit(
                            chunk_id=f"{doc_id}#0001",
                            doc_id=doc_id,
                            source_type="web",
                            score=hit.score,
                            text=str(doc["text"]),
                        )
                    )

        if corpus == "local":
            return local_hits
        if corpus == "web":
            return web_hits

        # For all: fuse local+web ranked lists.
        return _rrf_fuse([local_hits, web_hits], k=k)
