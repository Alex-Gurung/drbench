"""Step 1: Select Seed Secret.

For L seeds: deterministic, no LLM. Picks from the secret inventory.
For W seeds: picks a web doc + entity, uses LLM to generate seed Q/A.
"""

from __future__ import annotations

import logging
import random
from typing import Sequence

from making_dataset_2.data_loading import LocalDoc, Secret
from making_dataset_2.llm import LLMClient
from making_dataset_2.parsing import parse_qa_with_quote
from making_dataset_2.pipeline.entity_index import EntityIndex, is_entity
from making_dataset_2.types import ChainState, HopRecord

logger = logging.getLogger(__name__)


def select_seed(
    secrets: Sequence[Secret],
    doc_lookup: dict[str, LocalDoc],
    *,
    pattern: str = "LW",
    task_id: str | None = None,
    company: str | None = None,
    rng: random.Random | None = None,
    require_entity: bool = True,
) -> ChainState:
    """Pick a random seed secret and initialize ChainState.

    Assumes `secrets` is already filtered by `filter_seed_secrets()`.
    If require_entity=True, filters to secrets whose answer is an entity
    (not a pure number, email, or date) since entities are needed for
    entity-based document matching.

    Raises ValueError if no secrets available.
    """
    candidates = list(secrets)
    if require_entity:
        candidates = [s for s in candidates if is_entity(s.answer)]
    if not candidates:
        raise ValueError("No seed secrets available after filtering.")

    rng = rng or random.Random()
    secret = rng.choice(candidates)

    doc = doc_lookup[secret.doc_id]

    hop = HopRecord(
        hop_number=1,
        hop_type="L",
        question=secret.question,
        answer=secret.answer,
        doc_id=secret.doc_id,
        doc_text=doc.text,
    )

    state = ChainState(
        pattern=pattern,
        hop_history=[hop],
        global_question=secret.question,
        global_answer=secret.answer,
        used_doc_ids={secret.doc_id},
        task_id=task_id or doc.meta.get("task_id"),
        company=company or doc.meta.get("company_name"),
    )
    return state


# ---------------------------------------------------------------------------
# Web (W) seeds
# ---------------------------------------------------------------------------

PROMPT_WEB_SEED = '''\
Given this document, write a factual question whose answer is exactly "{entity}".
The question should be specific and answerable from the document text alone.
The answer "{entity}" must NOT appear in the question text.

<document>
{doc_text}
</document>

<answer>
QUOTE: <exact sentence from document containing "{entity}">
QUESTION: ... (must NOT contain "{entity}")
ANSWER: {entity}
</answer>'''


def select_web_seed(
    entity_index: EntityIndex,
    web_doc_ids: set[str],
    local_doc_ids: set[str],
    llm: LLMClient,
    *,
    pattern: str = "WL",
    task_id: str | None = None,
    company: str | None = None,
    rng: random.Random | None = None,
) -> ChainState:
    """Pick a web doc seed with an entity that can bridge to local docs.

    Pre-filters to (web_doc, entity) pairs where the entity appears in at
    least one local doc, ensuring the W→L transition is feasible.

    Raises ValueError if no viable seed found.
    """
    rng = rng or random.Random()

    # Collect bridgeable (web_doc_id, entity) pairs
    candidates: list[tuple[str, str]] = []
    for web_id in web_doc_ids:
        entities = entity_index.entities_in_doc(web_id)
        for ent in entities:
            if not is_entity(ent):
                continue
            if entity_index.docs_containing_text(ent, pool=local_doc_ids):
                candidates.append((web_id, ent))

    if not candidates:
        raise ValueError("No web docs with bridgeable entities found.")

    rng.shuffle(candidates)
    logger.info("Web seed: %d bridgeable (doc, entity) candidates", len(candidates))

    for web_doc_id, entity in candidates[:20]:
        doc_text = entity_index.doc_text(web_doc_id)
        context = doc_text[:8000]

        prompt = PROMPT_WEB_SEED.format(entity=entity, doc_text=context)
        try:
            raw = llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
            )
        except Exception as e:
            logger.debug("Web seed LLM call failed: %s", e)
            continue

        result = parse_qa_with_quote(raw)
        if result is None:
            logger.debug("Web seed parse failed for entity=%r", entity)
            continue

        question, answer, quote = result

        # Verify answer matches expected entity
        if answer.lower().strip() != entity.lower().strip():
            logger.debug("Web seed answer mismatch: got %r, expected %r", answer, entity)
            continue

        # Answer should not appear in question
        if entity.lower() in question.lower():
            logger.debug("Web seed: entity %r leaked into question", entity)
            continue

        logger.info("Web seed OK: Q=%r A=%r doc=%s", question[:60], answer, web_doc_id[:40])

        hop = HopRecord(
            hop_number=1,
            hop_type="W",
            question=question,
            answer=answer,
            doc_id=web_doc_id,
            doc_text=doc_text,
            quote=quote,
        )

        return ChainState(
            pattern=pattern,
            hop_history=[hop],
            global_question=question,
            global_answer=answer,
            used_doc_ids={web_doc_id},
            task_id=task_id,
            company=company,
        )

    raise ValueError("Failed to generate web seed after 20 attempts.")
