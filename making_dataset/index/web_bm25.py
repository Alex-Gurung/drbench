from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class WebBM25Hit:
    docid: str
    score: float
    text: str


class WebBM25Searcher:
    """
    Thin wrapper over a prebuilt Pyserini/Lucene index (BrowseComp-Plus).

    Note: Pyserini requires a working Java toolchain. If you see
    "Unable to find javac", run:
      source ~/initmamba.sh
    before invoking any script that imports pyserini.
    """

    def __init__(self, index_path: str) -> None:
        try:
            from pyserini.search.lucene import LuceneSearcher
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to import Pyserini LuceneSearcher. If you see 'Unable to find javac', "
                "run: source ~/initmamba.sh"
            ) from exc

        self.index_path = index_path
        self.searcher = LuceneSearcher(index_path)

    @property
    def num_docs(self) -> int:
        return int(self.searcher.num_docs)

    def search(self, query: str, k: int = 10) -> list[WebBM25Hit]:
        hits = self.searcher.search(query, k)
        out: list[WebBM25Hit] = []
        for hit in hits:
            raw = json.loads(hit.lucene_document.get("raw"))
            out.append(WebBM25Hit(docid=str(hit.docid), score=float(hit.score), text=raw["contents"]))
        return out

    def get_document(self, docid: str) -> Optional[dict[str, Any]]:
        doc = self.searcher.doc(docid)
        if doc is None:
            return None
        raw = json.loads(doc.raw())
        return {"docid": str(docid), "text": raw["contents"]}

