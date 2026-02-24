"""Step 3: Select the shared entity for a document transition.

Deterministic entity selection between two documents. No LLM calls.
Picks the best shared entity to serve as the bridge between documents.
"""

from __future__ import annotations

import logging

from making_dataset_2.pipeline.entity_index import EntityIndex, is_entity

logger = logging.getLogger(__name__)


def select_shared_entity(
    entity_index: EntityIndex,
    doc_current: str,
    doc_next: str,
    *,
    current_answer: str,
    used_answers: set[str] | None = None,
    prefer_different: bool = True,
) -> tuple[str, dict]:
    """Pick the best shared entity for a document transition.

    Selection priority:
    1. Prefer entity != current_answer (forces non-trivial intra question)
    2. Prefer named entities over numbers/dates (better for bridging)
    3. Prefer longer entities (more specific)
    4. Avoid entities already used as answers in the chain

    If prefer_different=True and no different entity exists, falls back to
    current_answer as the shared entity (trivial intra, skip it).

    Returns:
        (entity, trace)

    Raises:
        ValueError: If no shared entities found at all.
    """
    used = {a.lower() for a in (used_answers or set())}
    current_lower = current_answer.strip().lower()

    # Get shared entities from NER
    shared = entity_index.shared_entities(doc_current, doc_next)

    # Also check if current_answer is in next doc text (NER may miss it)
    next_text_lower = entity_index.doc_text(doc_next).lower()
    if current_lower in next_text_lower:
        if not any(e.lower() == current_lower for e in shared):
            shared.append(current_answer.strip())

    if not shared:
        raise ValueError(
            f"No shared entities between {doc_current} and {doc_next}"
        )

    # Score and sort candidates
    scored: list[tuple[float, str]] = []
    for ent in shared:
        ent_lower = ent.lower()
        score = 0.0

        # Prefer different from current answer
        if ent_lower != current_lower:
            score += 100

        # Prefer named entities over pure numbers
        if is_entity(ent):
            score += 50

        # Prefer longer (more specific)
        score += len(ent)

        # Penalize already-used answers
        if ent_lower in used:
            score -= 200

        scored.append((score, ent))

    scored.sort(key=lambda x: x[0], reverse=True)

    # If prefer_different but only current_answer available, use it anyway
    selected = scored[0][1]

    trace = {
        "step": "select_entity",
        "jump": None,  # Caller fills this in
        "doc_current": doc_current,
        "doc_next": doc_next,
        "current_answer": current_answer,
        "all_shared": [e for _, e in scored[:10]],
        "selected": selected,
        "is_trivial": selected.lower() == current_lower,
    }

    logger.info(
        "Shared entities: %d total, selected %r (trivial=%s)",
        len(shared), selected, selected.lower() == current_lower,
    )

    return selected, trace
