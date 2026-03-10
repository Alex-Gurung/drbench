"""Find bridge: the core atomic operation for document transitions.

Given the current document, current answer, and a pool of candidate next docs,
find (target_doc, bridge_entity, needs_intra) tuples.

When a BM25 searcher is available (the normal case), candidates come only from
topically-relevant docs — BM25 narrows the pool first, then we look for bridges.
This ensures web documents are thematically related to the source context rather
than random substring matches on generic values like "25%".

Without a searcher, falls back to exhaustive substring + NER entity matching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from making_dataset_2.pipeline.entity_index import EntityIndex

logger = logging.getLogger(__name__)


@dataclass
class BridgeCandidate:
    target_doc_id: str
    bridge_entity: str
    needs_intra: bool  # True if bridge_entity != current_answer
    score: float = 0.0


def _min_distance(text_lower: str, a: str, b: str) -> int:
    """Minimum character distance between any occurrence of a and any occurrence of b in text."""
    a_lower, b_lower = a.lower(), b.lower()
    a_positions: list[int] = []
    b_positions: list[int] = []
    start = 0
    while True:
        idx = text_lower.find(a_lower, start)
        if idx == -1:
            break
        a_positions.append(idx)
        start = idx + 1
    start = 0
    while True:
        idx = text_lower.find(b_lower, start)
        if idx == -1:
            break
        b_positions.append(idx)
        start = idx + 1

    if not a_positions or not b_positions:
        return float('inf')

    best = float('inf')
    for ap in a_positions:
        a_end = ap + len(a_lower)
        for bp in b_positions:
            b_end = bp + len(b_lower)
            if a_end <= bp:
                d = bp - a_end
            elif b_end <= ap:
                d = ap - b_end
            else:
                d = 0
            best = min(best, d)
            if best == 0:
                return 0
    return best


def _retrieval_query(doc_text: str, answer: str, window: int = 400) -> str:
    """Build a BM25 query from context around the answer in the source doc."""
    idx = doc_text.lower().find(answer.lower())
    if idx == -1:
        return answer
    half = window // 2
    start = max(0, idx - half)
    end = min(len(doc_text), idx + len(answer) + half)
    return doc_text[start:end]


def _find_in_pool(
    entity_index: EntityIndex,
    current_doc_id: str,
    current_answer: str,
    pool: set[str],
    used: set[str],
) -> tuple[list[BridgeCandidate], list[BridgeCandidate]]:
    """Find fast-path and entity-path candidates within a doc pool.

    Returns (fast_candidates, entity_candidates).
    """
    answer_lower = current_answer.strip().lower()
    current_doc_text_lower = entity_index.doc_text(current_doc_id).lower()
    current_entities = entity_index.entities_in_doc(current_doc_id)
    seen_keys: set[tuple[str, str]] = set()

    fast_candidates: list[BridgeCandidate] = []
    entity_candidates: list[BridgeCandidate] = []

    # Fast path: current_answer appears in pool docs
    for doc_id in entity_index.docs_containing_text(current_answer, pool=pool):
        key = (doc_id, answer_lower)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        fast_candidates.append(BridgeCandidate(
            target_doc_id=doc_id,
            bridge_entity=current_answer.strip(),
            needs_intra=False,
        ))

    # Entity path: entities in current_doc that appear in pool docs
    for ent in current_entities:
        ent_lower = ent.lower()
        if ent_lower == answer_lower or len(ent_lower) < 3 or ent_lower in used:
            continue
        for doc_id in entity_index.docs_containing_text(ent, pool=pool):
            key = (doc_id, ent_lower)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            proximity = _min_distance(current_doc_text_lower, current_answer, ent)
            entity_candidates.append(BridgeCandidate(
                target_doc_id=doc_id,
                bridge_entity=ent,
                needs_intra=True,
                score=-proximity,
            ))

    entity_candidates.sort(key=lambda c: c.score, reverse=True)
    return fast_candidates, entity_candidates


def find_bridge(
    entity_index: EntityIndex,
    current_doc_id: str,
    current_answer: str,
    next_pool: set[str],
    seen_docs: set[str] | None = None,
    used_answers: set[str] | None = None,
    searcher: Any = None,
    retrieval_k: int = 50,
) -> list[BridgeCandidate]:
    """Find how to get from current_doc to a doc in next_pool.

    When searcher is available: BM25 retrieves topically-relevant docs first,
    then bridges are found only within those docs. This ensures target documents
    are thematically related to the source context.

    When searcher is None: exhaustive fast-path + entity-path over the full pool.
    """
    seen = seen_docs or set()
    used = {a.lower() for a in (used_answers or set())}
    search_pool = next_pool - seen

    if not search_pool:
        return []

    if searcher is not None:
        # BM25-first: narrow to topically relevant docs, then find bridges
        query = _retrieval_query(entity_index.doc_text(current_doc_id), current_answer)
        hits = searcher.search(query, k=retrieval_k, mode="bm25")
        bm25_pool = {h.doc_id for h in hits} & search_pool

        fast_candidates, entity_candidates = _find_in_pool(
            entity_index, current_doc_id, current_answer, bm25_pool, used,
        )
        candidates = fast_candidates + entity_candidates

        logger.info(
            "find_bridge: %d candidates (fast=%d, entity=%d, bm25_pool=%d) from %s with answer=%r",
            len(candidates), len(fast_candidates), len(entity_candidates),
            len(bm25_pool), current_doc_id[:40], current_answer,
        )
    else:
        # No searcher: exhaustive search over full pool (capped for entity path)
        _EXHAUSTIVE_ENTITY_LIMIT = 1000
        answer_lower = current_answer.strip().lower()
        current_doc_text_lower = entity_index.doc_text(current_doc_id).lower()
        current_entities = entity_index.entities_in_doc(current_doc_id)
        seen_keys: set[tuple[str, str]] = set()

        fast_candidates: list[BridgeCandidate] = []
        entity_candidates: list[BridgeCandidate] = []

        for doc_id in entity_index.docs_containing_text(current_answer, pool=search_pool):
            key = (doc_id, answer_lower)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            fast_candidates.append(BridgeCandidate(
                target_doc_id=doc_id,
                bridge_entity=current_answer.strip(),
                needs_intra=False,
            ))

        if len(search_pool) <= _EXHAUSTIVE_ENTITY_LIMIT:
            for ent in current_entities:
                ent_lower = ent.lower()
                if ent_lower == answer_lower or len(ent_lower) < 3 or ent_lower in used:
                    continue
                for doc_id in entity_index.docs_containing_text(ent, pool=search_pool):
                    key = (doc_id, ent_lower)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    proximity = _min_distance(current_doc_text_lower, current_answer, ent)
                    entity_candidates.append(BridgeCandidate(
                        target_doc_id=doc_id,
                        bridge_entity=ent,
                        needs_intra=True,
                        score=-proximity,
                    ))

        entity_candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = fast_candidates + entity_candidates

        logger.info(
            "find_bridge (no searcher): %d candidates (fast=%d, entity=%d) from %s with answer=%r",
            len(candidates), len(fast_candidates), len(entity_candidates),
            current_doc_id[:40], current_answer,
        )

    return candidates
