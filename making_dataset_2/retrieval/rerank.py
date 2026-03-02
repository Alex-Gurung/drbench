"""Reranking protocol and utilities for retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


class Reranker(Protocol):
    def rerank(self, query: str, texts: Sequence[str]) -> list[int]:
        """Return indices of texts in descending relevance order."""


@dataclass(frozen=True)
class NoopReranker:
    def rerank(self, query: str, texts: Sequence[str]) -> list[int]:  # noqa: ARG002
        return list(range(len(texts)))


class CrossEncoderReranker:
    def __init__(self, model_name_or_path: str) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required for CrossEncoder reranking. "
                "Install it or omit --reranker-model."
            ) from exc
        self.model = CrossEncoder(model_name_or_path)

    def rerank(self, query: str, texts: Sequence[str]) -> list[int]:
        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs)
        # Sort indices by score desc
        return sorted(range(len(texts)), key=lambda i: float(scores[i]), reverse=True)


def build_reranker(model_name_or_path: str | None) -> Reranker | None:
    if not model_name_or_path:
        return None
    return CrossEncoderReranker(model_name_or_path)

