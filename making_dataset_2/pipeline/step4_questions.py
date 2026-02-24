"""Atomic question generation: constrained, pick-from-list, and LLM ranking.

Two question types:
  - constrained: both prev_answer and target_answer known (for intra-doc bridging)
  - pick: prev_answer known, model picks answer from provided entity list (for inter-doc)

Plus an LLM judge that ranks multiple candidate questions by quality.
"""

from __future__ import annotations

import logging
import re
import time

from making_dataset_2.llm import LLMClient
from making_dataset_2.parsing import parse_qa, parse_qa_with_quote

logger = logging.getLogger(__name__)

DOC_TEXT_LIMIT = 8_000

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPT_CONSTRAINED = '''\
You are writing one question in a multi-hop QA chain.
The previous answer was "{prev_answer}". Write a question where:
- "{prev_answer}" is the subject or a key qualifier — not buried in a relative clause.
  BAD: "What year is the report that found {prev_answer}..." (locates a document)
  BAD: "What company has X, which is growing by {prev_answer}..." (prev_answer is a side descriptor)
  GOOD: "What regulation applies to systems that achieved {prev_answer}..." (prev_answer drives the question)
- The answer is "{target_answer}" (1-5 words, no source references)

First find the exact sentence in the document that connects "{prev_answer}" to \
"{target_answer}", then write the question based on that sentence.

Example:
Previous answer: "Salesforce"
Target answer: "Health Cloud"
QUOTE: Salesforce's Health Cloud platform enables healthcare organizations to manage patient relationships.
REASONING: The sentence says Salesforce offers Health Cloud. I can ask what platform Salesforce offers.
QUESTION: What healthcare platform does Salesforce offer?
ANSWER: Health Cloud

<document>
{doc_text}
</document>

<answer>
QUOTE: <exact sentence from document containing both "{prev_answer}" and "{target_answer}">
REASONING: ...
QUESTION: ... (must contain "{prev_answer}")
ANSWER: {target_answer}
</answer>'''

PROMPT_PICK = '''\
You are writing one question in a multi-hop QA chain.
The previous answer was "{prev_answer}". Write a question where:
- "{prev_answer}" is the subject or a key qualifier — not buried in a relative clause.
  BAD: "What year is the report that found {prev_answer}..." (locates a document)
  BAD: "What company has X, which is growing by {prev_answer}..." (prev_answer is a side descriptor)
  GOOD: "What consequence leads {prev_answer} of retailers to..." (prev_answer drives the question)
- The answer is one of the entities below (match exactly)
- Pick the entity with the strongest factual link to "{prev_answer}" in the document

ENTITIES:
{entity_list}

First find the exact sentence in the document that connects "{prev_answer}" to \
your chosen entity, then write the question based on that sentence.

Example:
Previous answer: "Health Cloud"
Entities: FDA, HIPAA, CMS
QUOTE: Protecting sensitive patient data through HIPAA-compliant features in Health Cloud is essential for maintaining privacy.
REASONING: The sentence says Health Cloud must be HIPAA-compliant. HIPAA has the strongest link.
QUESTION: What regulation must Health Cloud comply with?
ANSWER: HIPAA

<document>
{doc_text}
</document>

<answer>
QUOTE: <exact sentence from document containing "{prev_answer}" and your chosen entity>
REASONING: ...
QUESTION: ... (must contain "{prev_answer}")
ANSWER: ...
</answer>'''


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_question_constrained(
    doc_text: str,
    prev_answer: str,
    target_answer: str,
    llm: LLMClient,
) -> tuple[str, str, dict]:
    """Generate a question where both sides are known.

    Used for intra-doc bridging: prev_answer → target_answer within one doc.

    Returns (question, answer, trace). Raises ValueError on parse failure.
    """
    prompt = PROMPT_CONSTRAINED.format(
        prev_answer=prev_answer,
        target_answer=target_answer,
        doc_text=doc_text[:DOC_TEXT_LIMIT],
    )
    t0 = time.time()
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4096,
    )
    duration = round(time.time() - t0, 2)

    result = parse_qa_with_quote(raw)
    if result is None:
        raise ValueError(f"Constrained question parse failed. Raw: {raw[:500]}")

    question, answer, quote = result
    trace = {
        "step": "question_constrained",
        "prompt": prompt,
        "raw_output": raw,
        "question": question,
        "answer": answer,
        "quote": quote,
        "prev_answer": prev_answer,
        "target_answer": target_answer,
        "duration": duration,
    }
    return question, answer, trace


def generate_question_pick(
    doc_text: str,
    prev_answer: str,
    entity_list: list[str],
    llm: LLMClient,
) -> tuple[str, str, dict]:
    """Generate a question where the model picks from an entity list.

    Used for inter-doc questions: prev_answer → one of entity_list.
    The model chooses the entity with the most natural connection.

    Returns (question, answer, trace). Raises ValueError on parse failure.
    """
    entity_str = "\n".join(f"- {e}" for e in entity_list)
    prompt = PROMPT_PICK.format(
        prev_answer=prev_answer,
        entity_list=entity_str,
        doc_text=doc_text[:DOC_TEXT_LIMIT],
    )
    t0 = time.time()
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4096,
    )
    duration = round(time.time() - t0, 2)

    result = parse_qa_with_quote(raw)
    if result is None:
        raise ValueError(f"Pick question parse failed. Raw: {raw[:500]}")

    question, answer, quote = result
    trace = {
        "step": "question_pick",
        "prompt": prompt,
        "raw_output": raw,
        "question": question,
        "answer": answer,
        "quote": quote,
        "prev_answer": prev_answer,
        "entity_list": entity_list,
        "duration": duration,
    }
    return question, answer, trace


# ---------------------------------------------------------------------------
# LLM Judge: rank candidate questions
# ---------------------------------------------------------------------------

PROMPT_RANK = '''\
You are evaluating questions for a multi-hop QA chain.
The previous answer was "{prev_answer}". Each candidate question should:
- Use "{prev_answer}" as the subject or key qualifier, not buried in a relative clause \
(e.g. "...which is growing by {prev_answer}" is bad — it's a side descriptor)
- Have a clear, specific, factual connection between the question and answer
- Sound natural and well-formed

Rank these candidates from best to worst. Output ONLY the ranking as a \
comma-separated list of numbers (e.g. "2, 1, 3").

CANDIDATES:
{candidates}

RANKING (best to worst):'''


def rank_questions(
    candidates: list[dict],
    prev_answer: str,
    llm: LLMClient,
) -> list[int]:
    """Rank candidate questions using an LLM judge.

    Args:
        candidates: [{"question": str, "answer": str}, ...]
        prev_answer: The previous answer that should appear in each question.
        llm: LLM client.

    Returns:
        Indices into candidates, ordered best-first.
        Falls back to original order on parse failure.
    """
    if len(candidates) <= 1:
        return list(range(len(candidates)))

    cand_str = "\n".join(
        f"{i + 1}. Q: {c['question']}\n   A: {c['answer']}"
        for i, c in enumerate(candidates)
    )
    prompt = PROMPT_RANK.format(prev_answer=prev_answer, candidates=cand_str)

    t0 = time.time()
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    duration = round(time.time() - t0, 2)

    # Parse ranking: extract numbers from the response
    numbers = [int(x) for x in re.findall(r'\d+', raw)]
    # Convert 1-indexed to 0-indexed, filter valid indices
    indices = [n - 1 for n in numbers if 1 <= n <= len(candidates)]
    # Deduplicate while preserving order
    seen: set[int] = set()
    unique: list[int] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    # Append any missing indices at the end
    for i in range(len(candidates)):
        if i not in seen:
            unique.append(i)

    logger.info("rank_questions: %s (%.1fs)", unique, duration)
    return unique
