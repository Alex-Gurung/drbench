"""Find bridge: the core atomic operation for document transitions.

Given the current document, current answer, and a pool of candidate next docs,
find (target_doc, bridge_entity, needs_intra) tuples.

Fast path: current_answer already appears in a next-pool doc → no intra needed.
Entity path: find entities in current_doc (spaCy NER) that appear in next-pool docs.

Scoring: proximity between current_answer and bridge_entity in the current doc text.
Entities that appear closer together are more likely to have a strong factual connection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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
    # Collect all start positions of a and b
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

    # Min distance between any pair of positions (end of one to start of other)
    best = float('inf')
    for ap in a_positions:
        a_end = ap + len(a_lower)
        for bp in b_positions:
            b_end = bp + len(b_lower)
            # Distance = gap between the two spans (0 if overlapping)
            if a_end <= bp:
                d = bp - a_end
            elif b_end <= ap:
                d = ap - b_end
            else:
                d = 0  # overlapping
            best = min(best, d)
            if best == 0:
                return 0
    return best


def find_bridge(
    entity_index: EntityIndex,
    current_doc_id: str,
    current_answer: str,
    next_pool: set[str],
    seen_docs: set[str] | None = None,
    used_answers: set[str] | None = None,
) -> list[BridgeCandidate]:
    """Find how to get from current_doc to a doc in next_pool.

    Returns ranked list of BridgeCandidate.
    Fast-path candidates (no intra needed) come first.
    Entity-path candidates are sorted by proximity to current_answer in the source doc.
    """
    seen = seen_docs or set()
    used = {a.lower() for a in (used_answers or set())}
    search_pool = next_pool - seen
    answer_lower = current_answer.strip().lower()

    if not search_pool:
        return []

    fast_candidates: list[BridgeCandidate] = []
    entity_candidates: list[BridgeCandidate] = []
    seen_keys: set[tuple[str, str]] = set()

    # --- Fast path: current_answer appears in next-pool docs ---
    direct_matches = entity_index.docs_containing_text(current_answer, pool=search_pool)
    for doc_id in direct_matches:
        key = (doc_id, answer_lower)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        fast_candidates.append(BridgeCandidate(
            target_doc_id=doc_id,
            bridge_entity=current_answer.strip(),
            needs_intra=False,
        ))

    # --- Entity path: entities in current_doc that appear in next-pool docs ---
    current_doc_text_lower = entity_index.doc_text(current_doc_id).lower()
    current_entities = entity_index.entities_in_doc(current_doc_id)

    for ent in current_entities:
        ent_lower = ent.lower()
        if ent_lower == answer_lower:
            continue  # Already handled in fast path
        if len(ent_lower) < 3:
            continue
        if ent_lower in used:
            continue  # Skip already-used answers

        matching_docs = entity_index.docs_containing_text(ent, pool=search_pool)
        for doc_id in matching_docs:
            key = (doc_id, ent_lower)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            proximity = _min_distance(current_doc_text_lower, current_answer, ent)
            # Negate so closer = higher score (sorted descending)
            entity_candidates.append(BridgeCandidate(
                target_doc_id=doc_id,
                bridge_entity=ent,
                needs_intra=True,
                score=-proximity,
            ))

    entity_candidates.sort(key=lambda c: c.score, reverse=True)
    candidates = fast_candidates + entity_candidates

    logger.info(
        "find_bridge: %d candidates (fast=%d, entity=%d) from %s with answer=%r",
        len(candidates),
        len(fast_candidates),
        len(entity_candidates),
        current_doc_id[:40],
        current_answer,
    )

    return candidates
