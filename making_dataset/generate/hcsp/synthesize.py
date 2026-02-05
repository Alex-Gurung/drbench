"""
Question synthesis from constraints using LLM.

Takes a set of fuzzy constraints and generates a natural question
that requires satisfying all constraints to answer.

Key principles:
- NEVER include banned terms (answer, intermediate secrets, doc_ids)
- Weave constraints naturally (no "clue A / clue B" phrasing)
- No corpus hints (local/web/document/excerpt)
- Sound like a puzzle, not a template
"""
from __future__ import annotations

import re
from typing import Any, List, Literal, Optional, Tuple, Union

from .schema import Constraint, LinkageType


SYNTHESIZE_QUESTION_TEMPLATE = """You are writing a multi-hop research question.

You have a set of constraints that together identify a unique answer.
Your job is to write a natural-sounding question that incorporates these constraints.

SEED QUESTION (the question this task is based on):
{seed_question}

The answer to your question should be the same as the answer to the seed question.
Your generated question should ask about the same type of information, but
incorporate the constraints to make it require multiple hops to answer.

CONSTRAINTS (facts that must be woven into the question):
{constraints_text}

{entity_blurring}

ANSWER TYPE: {answer_type}
(The question should ask for this type of answer)

{linkage_guidance}

BANNED TERMS (MUST NOT appear in your question):
{banned_terms}

RULES:
1. DO NOT use any of the banned terms
2. DO NOT use words like: local, web, internal, external, document, excerpt, chunk, corpus, snippet, evidence
3. DO NOT use phrasing like "clue A and clue B" or "according to X and Y"
4. DO write a natural, puzzle-like question that sounds like something a researcher would ask
5. Weave the constraints together into ONE coherent question (not a list of conditions)
6. Keep it under 60 words if possible
7. CRITICAL: The answer to your question must be the same type as the seed question answer
8. CRITICAL: Do NOT use the entity names directly - use the web constraints to describe them instead

Write ONLY the question, nothing else.
"""


# Linkage-specific guidance for question synthesis
LINKAGE_GUIDANCE = {
    "entity_chain": """STYLE GUIDANCE (Entity Chain):
The question should describe an entity using indirect constraints rather than naming it directly.
Example: "What was the retention rate for the inventory system that won the 2024 Award?"
(The system name is discovered via web, the rate is found in local data)""",

    "computational": """STYLE GUIDANCE (Computational):
The question should ask for a value that requires combining facts from multiple sources.
Example: "How many percentage points above industry average was the company's retention rate?"
(Industry average from one source, company rate from another)""",

    "selection": """STYLE GUIDANCE (Selection):
The question should use an external criterion to select from multiple options.
Example: "What was the retention for the product line ranked #1 in the Sustainability Index?"
(The ranking comes from one source, the metrics from another)""",

    "definitional": """STYLE GUIDANCE (Definitional):
The question should ask about a threshold or requirement defined externally.
Example: "What minimum retention threshold was required for Gold certification?"
(The certification is mentioned in one source, the requirements in another)""",

    "creative": """STYLE GUIDANCE:
Write a natural research question that requires information from multiple sources to answer.
The question should not be answerable from either source alone.""",
}


REWRITE_QUESTION_TEMPLATE = """Rewrite this question to sound more natural and puzzle-like.

ORIGINAL QUESTION:
{original_question}

BANNED TERMS (must NOT appear):
{banned_terms}

RULES:
1. Keep the same meaning and constraints
2. Make it sound like a natural research question
3. Don't use words like: local, web, internal, external, document, excerpt, chunk, corpus
4. Don't number the constraints or use "clue A / clue B" style
5. Keep it concise (under 60 words)

Write ONLY the rewritten question, nothing else.
"""


def _format_constraints_for_prompt(constraints: List[Constraint]) -> str:
    """Format constraints as a numbered list for the prompt."""
    lines = []
    for i, c in enumerate(constraints, 1):
        # Avoid adding any corpus hints like "web"/"local" into the prompt, since we
        # explicitly ban those words in the question output and the model may copy them.
        lines.append(f"{i}. {c.text}")
    return "\n".join(lines)


def _check_question_for_banned(question: str, banned_terms: List[str]) -> List[str]:
    """Check if question contains any banned terms. Returns list of violations."""
    violations = []
    question_lower = question.lower()
    for term in banned_terms:
        if term.lower() in question_lower:
            violations.append(term)
    return violations


def _check_question_for_corpus_hints(question: str) -> List[str]:
    """Check for corpus hints that shouldn't be in the question."""
    corpus_hints = [
        "local", "web", "internal", "external", "document", "excerpt",
        "chunk", "corpus", "snippet", "evidence", "file", "database",
    ]
    violations = []
    question_lower = question.lower()
    for hint in corpus_hints:
        # Check for word boundaries to avoid false positives
        if re.search(rf"\b{hint}\b", question_lower):
            violations.append(hint)
    return violations


def synthesize_question(
    constraints: List[Constraint],
    answer_type: Literal["entity", "metric", "date", "fact"],
    banned_terms: List[str],
    client: Any,  # VLLMClient
    seed_question: str = "",
    max_tokens: int = 256,
    temperature: float = 0.7,
    linkage_type: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """
    Generate a natural question from constraints.

    Args:
        constraints: List of fuzzy constraints to incorporate
        answer_type: What type of answer the question asks for
        banned_terms: Terms that must not appear in the question
        client: VLLMClient for making API calls
        seed_question: Original question from secret (anchor for synthesis)
        max_tokens: Max tokens for response
        temperature: Sampling temperature
        linkage_type: Optional linkage type for style guidance

    Returns:
        Tuple of (question, list_of_violations)
        Violations list is empty if question is valid
    """
    if not constraints:
        return "", ["no_constraints"]

    # Format inputs
    constraints_text = _format_constraints_for_prompt(constraints)
    banned_str = ", ".join(f'"{t}"' for t in banned_terms) if banned_terms else "(none)"

    # Build entity mappings for blurring guidance
    entity_mappings = build_entity_mappings(constraints)
    entity_blurring = format_entity_blurring_for_prompt(entity_mappings)

    # Get linkage-specific guidance
    linkage_guidance = ""
    if linkage_type and linkage_type in LINKAGE_GUIDANCE:
        linkage_guidance = LINKAGE_GUIDANCE[linkage_type]
    elif linkage_type:
        linkage_guidance = LINKAGE_GUIDANCE.get("creative", "")

    prompt = SYNTHESIZE_QUESTION_TEMPLATE.format(
        seed_question=seed_question or "(not provided)",
        constraints_text=constraints_text,
        entity_blurring=entity_blurring,
        answer_type=answer_type,
        banned_terms=banned_str,
        linkage_guidance=linkage_guidance,
    )

    resp = client.chat(
        messages=[{"role": "user", "content": prompt}],
        stage="hcsp_question_synthesis",
        extra={"num_constraints": len(constraints), "answer_type": answer_type},
        max_tokens=max_tokens,
        temperature=temperature,
    )
    question = (resp.choices[0].message.content or "").strip()

    # Clean up the question
    question = question.strip()
    if question.startswith('"') and question.endswith('"'):
        question = question[1:-1]
    if question.startswith("Question:"):
        question = question[9:].strip()

    # Check for violations
    violations = []
    violations.extend(_check_question_for_banned(question, banned_terms))
    violations.extend(_check_question_for_corpus_hints(question))

    # Check for unblurred entities (entity names appearing directly)
    if entity_mappings:
        unblurred = check_entities_blurred(question, entity_mappings)
        violations.extend([f"unblurred:{e}" for e in unblurred])

    return question, violations


def rewrite_question(
    original_question: str,
    banned_terms: List[str],
    client: Any,  # VLLMClient
    max_tokens: int = 256,
    temperature: float = 0.5,
) -> Tuple[str, List[str]]:
    """
    Rewrite a question to sound more natural.

    Args:
        original_question: The question to rewrite
        banned_terms: Terms that must not appear
        client: VLLMClient for making API calls
        max_tokens: Max tokens for response
        temperature: Sampling temperature

    Returns:
        Tuple of (rewritten_question, list_of_violations)
    """
    banned_str = ", ".join(f'"{t}"' for t in banned_terms) if banned_terms else "(none)"

    prompt = REWRITE_QUESTION_TEMPLATE.format(
        original_question=original_question,
        banned_terms=banned_str,
    )

    resp = client.chat(
        messages=[{"role": "user", "content": prompt}],
        stage="hcsp_question_rewrite",
        extra={},
        max_tokens=max_tokens,
        temperature=temperature,
    )
    question = (resp.choices[0].message.content or "").strip()

    # Clean up
    question = question.strip()
    if question.startswith('"') and question.endswith('"'):
        question = question[1:-1]

    # Check for violations
    violations = []
    violations.extend(_check_question_for_banned(question, banned_terms))
    violations.extend(_check_question_for_corpus_hints(question))

    return question, violations


def synthesize_with_retry(
    constraints: List[Constraint],
    answer_type: Literal["entity", "metric", "date", "fact"],
    banned_terms: List[str],
    client: Any,
    seed_question: str = "",
    max_retries: int = 3,
    max_tokens: int = 256,
    temperature: float = 0.7,
    linkage_type: Optional[Union[str, LinkageType]] = None,
) -> Tuple[Optional[str], List[str]]:
    """
    Synthesize question with retries on violation.

    Args:
        seed_question: Original question from secret (anchor for synthesis)

    Returns the first valid question, or the last attempt with violations list.
    """
    last_question = ""
    last_violations: List[str] = []

    # Convert LinkageType enum to string
    linkage_str = None
    if linkage_type:
        if isinstance(linkage_type, LinkageType):
            linkage_str = linkage_type.value
        else:
            linkage_str = str(linkage_type)

    for attempt in range(max_retries):
        # Increase temperature slightly on retries
        temp = temperature + (attempt * 0.1)
        question, violations = synthesize_question(
            constraints=constraints,
            answer_type=answer_type,
            banned_terms=banned_terms,
            client=client,
            seed_question=seed_question,
            max_tokens=max_tokens,
            temperature=temp,
            linkage_type=linkage_str,
        )

        last_question = question
        last_violations = violations

        if question and not violations:
            return question, []

        # If we got a question but it violates constraints, try one rewrite pass.
        # This is often more effective than resampling from scratch.
        if question:
            rewritten, rewrite_violations = rewrite_question(
                original_question=question,
                banned_terms=banned_terms,
                client=client,
                max_tokens=max_tokens,
                temperature=max(0.2, temp - 0.2),
            )
            last_question = rewritten
            last_violations = rewrite_violations
            if rewritten and not rewrite_violations:
                return rewritten, []

    # All retries failed, return last attempt
    return last_question if last_question else None, last_violations


# Utility for building banned terms list

def build_banned_terms(
    answer: str,
    intermediate_secrets: List[str],
    doc_ids: List[str],
    filenames: List[str],
    extra_terms: Optional[List[str]] = None,
) -> List[str]:
    """
    Build a comprehensive banned terms list.

    Args:
        answer: The final answer string
        intermediate_secrets: Intermediate values that must be discovered
        doc_ids: Document IDs that shouldn't appear
        filenames: Filenames that shouldn't appear
        extra_terms: Any additional terms to ban

    Returns:
        Deduplicated list of banned terms
    """
    terms = set()

    # Add answer (and common variants)
    if answer:
        terms.add(answer)
        # Also ban without common punctuation
        terms.add(answer.strip().rstrip(".").rstrip(","))

    # Add intermediate secrets
    for secret in intermediate_secrets:
        if secret:
            terms.add(secret)

    # Add doc_ids
    for doc_id in doc_ids:
        if doc_id:
            terms.add(doc_id)
            # Also extract filename from doc_id path
            if "/" in doc_id:
                terms.add(doc_id.split("/")[-1])

    # Add filenames
    for fname in filenames:
        if fname:
            terms.add(fname)
            # Without extension
            if "." in fname:
                terms.add(fname.rsplit(".", 1)[0])

    # Add extra terms
    if extra_terms:
        for term in extra_terms:
            if term:
                terms.add(term)

    # Remove empty strings and very short terms (likely false positives)
    terms = {t for t in terms if len(t) >= 3}

    return sorted(terms)


# Entity blurring functions for InfoSeek-style question synthesis

def build_entity_mappings(constraints: List[Constraint]) -> dict:
    """
    Group constraints by the entity they describe.

    Returns: {
        "FreshTrack": {
            "web": [Constraint("won 2024 Award"), ...],
            "local": [Constraint("Q3 retention was 85%"), ...]
        }
    }
    """
    mappings: dict = {}
    for c in constraints:
        if c.entity:
            if c.entity not in mappings:
                mappings[c.entity] = {"web": [], "local": []}
            mappings[c.entity][c.corpus].append(c)
    return mappings


def format_entity_blurring_for_prompt(entity_mappings: dict) -> str:
    """Format entity mappings as guidance for question synthesis."""
    if not entity_mappings:
        return ""

    lines = ["ENTITY BLURRING (use web constraints to identify entities, don't name them directly):"]
    for entity, by_corpus in entity_mappings.items():
        web_constraints = by_corpus.get("web", [])
        if web_constraints:
            constraints_text = "; ".join(c.text for c in web_constraints[:3])
            lines.append(f"  - Instead of '{entity}', describe as: {constraints_text}")

    if len(lines) == 1:
        return ""  # No blurring guidance needed

    return "\n".join(lines)


def check_entities_blurred(question: str, entity_mappings: dict) -> List[str]:
    """
    Check that entity names don't appear directly in the question.

    Returns list of entity names that appear unblurred (should be empty for good questions).
    """
    unblurred = []
    question_lower = question.lower()
    for entity in entity_mappings.keys():
        # Check if entity name appears in question
        if entity.lower() in question_lower:
            unblurred.append(entity)
    return unblurred
