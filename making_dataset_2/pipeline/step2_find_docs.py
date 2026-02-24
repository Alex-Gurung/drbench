"""Step 2: Find candidate documents for the next jump.

Entity-based document selection. No LLM calls.
Finds documents containing the current answer entity via exact substring match.
For W->L transitions, additionally filters for privacy question overlap.
"""

from __future__ import annotations

import logging
from typing import Sequence

from making_dataset_2.data_loading import Secret
from making_dataset_2.pipeline.entity_index import EntityIndex
from making_dataset_2.types import ChainState, DocCandidate

logger = logging.getLogger(__name__)

DOC_TEXT_LIMIT = 10_000  # Truncate doc text for prompts


def find_next_docs(
    state: ChainState,
    entity_index: EntityIndex,
    *,
    target_pool: str,  # "local" or "web"
    pool_doc_ids: set[str],
    task_doc_ids: set[str] | None = None,  # Restrict to same task
    privacy_questions: Sequence[Secret] | None = None,
    use_verbatim_privacy: bool = True,
) -> tuple[list[DocCandidate], dict]:
    """Find candidate documents for the next jump.

    For any transition: find docs in target pool containing current_answer entity.
    For W->L with privacy_questions: additionally require overlap with privacy Qs.

    Returns:
        (candidates, trace) sorted by number of shared entities (descending).

    Raises:
        ValueError: If no candidate documents found.
    """
    current_answer = state.global_answer.strip()
    search_pool = pool_doc_ids - state.used_doc_ids

    if task_doc_ids is not None:
        search_pool = search_pool & task_doc_ids

    # Find docs containing the current answer entity
    matching = entity_index.docs_containing_text(current_answer, pool=search_pool)

    if not matching:
        raise ValueError(
            f"No docs in {target_pool} pool contain entity {current_answer!r} "
            f"(searched {len(search_pool)} docs)"
        )

    # Build candidates with shared entity info
    current_doc_id = state.hop_history[-1].doc_id if state.hop_history else None
    candidates: list[DocCandidate] = []

    for doc_id in matching:
        if current_doc_id:
            shared = entity_index.shared_entities(current_doc_id, doc_id)
        else:
            shared = [current_answer]

        # Ensure current_answer is in shared list even if NER missed it
        if not any(e.lower() == current_answer.lower() for e in shared):
            shared = [current_answer] + shared

        doc_text = entity_index.doc_text(doc_id)

        # For W->L: check privacy question overlap
        privacy_q = None
        privacy_a = None
        if target_pool == "local" and privacy_questions and use_verbatim_privacy:
            doc_entities = entity_index.entities_in_doc(doc_id)
            doc_text_lower = doc_text.lower()
            for secret in privacy_questions:
                # Check if any entity from the web doc appears in this privacy Q
                # AND the privacy Q's answer doc matches this local doc
                if secret.doc_id == doc_id:
                    # Check if any shared entity appears in the privacy question text
                    for ent in shared:
                        if ent.lower() in secret.question.lower():
                            privacy_q = secret.question
                            privacy_a = secret.answer
                            break
                if privacy_q:
                    break

        candidates.append(DocCandidate(
            doc_id=doc_id,
            doc_text=doc_text[:DOC_TEXT_LIMIT],
            shared_entities=shared,
            privacy_question=privacy_q,
            privacy_answer=privacy_a,
        ))

    # Sort: prefer candidates with privacy questions, then by shared entity count
    candidates.sort(key=lambda c: (c.privacy_question is not None, len(c.shared_entities)), reverse=True)

    trace = {
        "step": "find_docs",
        "jump": state.jumps_completed + 1,
        "current_answer": current_answer,
        "target_pool": target_pool,
        "pool_size": len(search_pool),
        "n_matching": len(matching),
        "candidates": [
            {
                "doc_id": c.doc_id,
                "n_shared_entities": len(c.shared_entities),
                "shared_entities": c.shared_entities[:10],
                "has_privacy_q": c.privacy_question is not None,
                "snippet": c.doc_text[:200],
            }
            for c in candidates[:5]
        ],
    }

    logger.info(
        "Jump %d: found %d candidate %s docs containing %r",
        state.jumps_completed + 1, len(candidates), target_pool, current_answer,
    )

    return candidates, trace
