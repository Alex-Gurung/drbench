"""BrowseComp adapter — wraps existing Pyserini BM25 + Qwen3 dense searchers.

Provides a .search() interface matching HybridSearcher so Step 3 can use
both interchangeably via duck typing.

Requires: pyserini (for BM25), faiss + torch (for dense). Run `source ~/initmamba.sh`
if you see Java errors.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Add making_dataset to path for WebBM25Searcher / WebDenseSearcher
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from making_dataset.index.web_bm25 import WebBM25Searcher
from making_dataset_2.retrieval.types import RetrievalHit

# Default paths
DEFAULT_BM25_INDEX = "/home/toolkit/BrowseComp-Plus/indexes/bm25/"
DEFAULT_DENSE_GLOB = "/home/toolkit/BrowseComp-Plus/indexes/qwen3-embedding-4b/corpus.shard*.pkl"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"


class BrowseCompSearcher:
    """Adapts BrowseComp WebBM25Searcher + WebDenseSearcher to HybridSearcher interface.

    BrowseComp is doc-level (not chunk-level), so chunk_id == doc_id.
    Text comes from Lucene stored fields.

    Pass a pre-created Qwen3EosEmbedder via `embedder` to share a single model
    instance across all searchers. If not provided, WebDenseSearcher creates its own.
    The `embedder` parameter in .search() is ignored — uses the model from __init__.
    """

    def __init__(
        self,
        *,
        bm25_index: str = DEFAULT_BM25_INDEX,
        dense_shard_glob: str | None = DEFAULT_DENSE_GLOB,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        embedder: Any = None,
    ) -> None:
        self.bm25 = WebBM25Searcher(bm25_index)
        self.dense = None
        if dense_shard_glob:
            from making_dataset.index.web_dense import WebDenseSearcher
            self.dense = WebDenseSearcher(
                index_glob=dense_shard_glob,
                model_name_or_path=model,
                doc_store=self.bm25,
                device=device,
                embedder=embedder,
            )

    @property
    def size(self) -> int:
        return self.bm25.num_docs

    def _to_hit(self, docid: str, score: float, text: str | None = None) -> RetrievalHit:
        if text is None:
            doc = self.bm25.get_document(docid)
            text = doc.get("contents", "") if doc else ""
        return RetrievalHit(
            chunk_id=docid,
            doc_id=docid,
            score=float(score),
            text=text or "",
            meta=None,
        )

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        mode: str = "hybrid",
        embedder: Any = None,  # ignored — uses own Qwen3EosEmbedder
        bm25_k: int = 200,
        dense_k: int = 200,
    ) -> list[RetrievalHit]:
        """Search BrowseComp. Interface matches HybridSearcher.search()."""
        if mode == "bm25" or self.dense is None:
            hits = self.bm25.search(query, k=k)
            return [self._to_hit(h.docid, h.score, h.text) for h in hits]

        if mode == "dense":
            hits = self.dense.search(query, k=k)
            return [self._to_hit(h.docid, h.score) for h in hits]

        # hybrid: BM25 broad recall → dense rerank
        bm25_hits = self.bm25.search(query, k=bm25_k)
        cand_docids = [h.docid for h in bm25_hits]
        reranked = self.dense.rerank_docids(query, cand_docids)

        # Build text lookup from BM25 hits (avoids extra Lucene lookups)
        text_by_id = {h.docid: h.text for h in bm25_hits}
        return [
            self._to_hit(h.docid, h.score, text_by_id.get(h.docid))
            for h in reranked[:k]
        ]
