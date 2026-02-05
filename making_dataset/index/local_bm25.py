from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    # Simple, deterministic tokenizer for BM25 on English-ish corpora.
    # Keep it dependency-free to make pipeline tests easy to run.
    toks = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in toks if len(t) >= 2]


@dataclass(frozen=True)
class SearchHit:
    doc_idx: int
    score: float


class BM25Index:
    """
    Minimal BM25 implementation with an inverted index (sufficient for local chunks).

    This is intentionally simple: it is not meant to exactly match Lucene/Pyserini.
    It is used for local-only experiments and small pipeline tests.
    """

    def __init__(
        self,
        docs: List[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not docs:
            raise ValueError("BM25Index requires at least one doc")
        self._docs = docs
        self._k1 = float(k1)
        self._b = float(b)

        self._N = len(docs)
        self._doc_len: list[int] = []
        self._avgdl: float = 0.0

        # term -> df
        self._df: dict[str, int] = defaultdict(int)
        # term -> postings of (doc_idx, tf)
        self._inv: dict[str, list[tuple[int, int]]] = defaultdict(list)

        self._build()

    def _build(self) -> None:
        total_len = 0
        for idx, text in enumerate(self._docs):
            tf = Counter(tokenize(text))
            dl = sum(tf.values())
            self._doc_len.append(dl)
            total_len += dl
            for term, f in tf.items():
                self._df[term] += 1
                self._inv[term].append((idx, f))

        self._avgdl = (total_len / self._N) if self._N else 0.0

    def _idf(self, df: int) -> float:
        # Okapi BM25 IDF variant; stable + non-negative.
        return math.log(1.0 + (self._N - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        exclude_doc_idx: int | None = None,
    ) -> list[SearchHit]:
        if k <= 0:
            return []

        qtf = Counter(tokenize(query))
        if not qtf:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in qtf.keys():
            df = self._df.get(term)
            if not df:
                continue
            idf = self._idf(df)
            postings = self._inv.get(term, [])
            for doc_idx, f in postings:
                if exclude_doc_idx is not None and doc_idx == exclude_doc_idx:
                    continue
                dl = self._doc_len[doc_idx]
                denom = f + self._k1 * (1.0 - self._b + self._b * (dl / (self._avgdl or 1.0)))
                scores[doc_idx] += idf * (f * (self._k1 + 1.0) / denom)

        if not scores:
            return []

        # Top-k by score.
        top = heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])
        return [SearchHit(doc_idx=doc_idx, score=float(score)) for doc_idx, score in top]

    def doc_text(self, doc_idx: int) -> str:
        return self._docs[doc_idx]

    @property
    def size(self) -> int:
        return self._N

