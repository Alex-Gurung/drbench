"""
Constraint extraction from chunks using LLM.

Extracts "fuzzy constraints" - factual statements that describe an entity
without naming it directly. Each constraint must be grounded in a specific
text span with character offsets.

Key principle (from InfoSeek):
"Find a factual, but not strongly directional, attribute...
this fact alone is not enough for a user to directly search for the root entity"
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from .schema import Constraint, EvidencePointer


EXTRACT_CONSTRAINTS_TEMPLATE = """You are extracting fuzzy constraints from a text excerpt.

A "fuzzy constraint" is a factual statement that describes something WITHOUT naming it directly.
The constraint should be:
1. TRUE - verifiable from the text
2. FUZZY - not enough alone to uniquely identify the entity/fact
3. GROUNDED - you must quote the exact text span that supports it

BANNED TERMS (must NOT appear in your constraints):
{banned_terms}

TEXT EXCERPT:
{chunk_text}

TARGET: Extract 1-{max_constraints} constraints that describe facts from this text.
Each constraint should be about something DIFFERENT (don't repeat the same fact).

For each constraint, output in this exact format:
CONSTRAINT: <a short factual statement, 5-20 words>
QUOTE: <the exact text span from the excerpt that supports this>
TYPE: <one of: attribute, relation, temporal, other>

Output {max_constraints} constraints if possible, fewer if the text doesn't support more.
"""


def _find_quote_span(text: str, quote: str) -> Optional[Tuple[int, int]]:
    """
    Find the character offsets of a quote in the text.
    Returns (start, end) or None if not found.

    Uses progressively more lenient matching:
    1. Exact match
    2. Case-insensitive match
    3. Whitespace-normalized match
    4. Substring match (first 30+ chars)
    5. Key phrase match
    """
    quote = quote.strip()
    if not quote:
        return None

    text_lower = text.lower()
    quote_lower = quote.lower()

    # Try exact match first
    idx = text.find(quote)
    if idx >= 0:
        return idx, idx + len(quote)

    # Try case-insensitive match
    idx = text_lower.find(quote_lower)
    if idx >= 0:
        return idx, idx + len(quote)

    # Try fuzzy match with normalized whitespace
    pattern = re.escape(quote).replace(r"\ ", r"\s+")
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if m:
        return m.start(), m.end()

    # Try matching a substring of the quote (for partial matches)
    # This handles cases where LLM adds/removes words from the actual quote
    if len(quote) >= 40:
        # Try first 30 chars
        substr = quote[:30]
        idx = text_lower.find(substr.lower())
        if idx >= 0:
            # Find end of sentence or phrase
            end_idx = idx + len(substr)
            # Extend to end of sentence if possible
            for end_char in [".", ",", "\n", ";"]:
                next_end = text.find(end_char, end_idx)
                if next_end >= 0 and next_end < end_idx + 100:
                    end_idx = next_end + 1
                    break
            return idx, min(end_idx, len(text))

    # Try matching key words from the quote
    # Extract significant words (4+ chars, not common words)
    stop_words = {"that", "this", "with", "from", "have", "been", "were", "they", "their", "which", "about", "would", "there", "could"}
    words = [w for w in re.findall(r'\b\w{4,}\b', quote_lower) if w not in stop_words]

    if len(words) >= 3:
        # Find a span containing at least 3 key words
        for start_idx in range(0, len(text) - 50, 20):
            window = text_lower[start_idx:start_idx + 200]
            matches = sum(1 for w in words[:5] if w in window)
            if matches >= 3:
                # Found a good window, return it
                return start_idx, min(start_idx + 200, len(text))

    return None


def _parse_constraint_output(output: str) -> List[Dict[str, str]]:
    """
    Parse the LLM output into a list of constraint dictionaries.
    Each dict has: constraint, quote, type
    """
    results = []
    current: Dict[str, str] = {}

    for line in output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.upper().startswith("CONSTRAINT:"):
            if current.get("constraint"):
                results.append(current)
            current = {"constraint": line.split(":", 1)[1].strip()}
        elif line.upper().startswith("QUOTE:"):
            current["quote"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("TYPE:"):
            current["type"] = line.split(":", 1)[1].strip().lower()

    if current.get("constraint"):
        results.append(current)

    return results


def _contains_banned(text: str, banned_terms: List[str]) -> bool:
    """Check if text contains any banned terms (case-insensitive)."""
    text_lower = text.lower()
    for term in banned_terms:
        if term.lower() in text_lower:
            return True
    return False


def extract_constraints(
    chunk_text: str,
    chunk_id: str,
    corpus: Literal["local", "web"],
    banned_terms: List[str],
    client: Any,  # VLLMClient
    max_constraints: int = 3,
    max_tokens: int = 512,
    temperature: float = 0.3,
    entity_context: Optional[str] = None,
) -> List[Constraint]:
    """
    Extract fuzzy constraints from a chunk using LLM.

    Args:
        chunk_text: The text to extract constraints from
        chunk_id: ID of the chunk (for evidence pointer)
        corpus: Whether this is a local or web chunk
        banned_terms: Terms that must not appear in constraints
        client: VLLMClient for making API calls
        max_constraints: Maximum number of constraints to extract
        max_tokens: Max tokens for LLM response
        temperature: Sampling temperature
        entity_context: If set, all constraints describe this entity (for blurring)

    Returns:
        List of Constraint objects with grounded evidence
    """
    if not chunk_text.strip():
        return []

    # Format banned terms for prompt
    if banned_terms:
        banned_str = ", ".join(f'"{t}"' for t in banned_terms)
    else:
        banned_str = "(none)"

    prompt = EXTRACT_CONSTRAINTS_TEMPLATE.format(
        banned_terms=banned_str,
        chunk_text=chunk_text[:8000],  # Truncate very long chunks
        max_constraints=max_constraints,
    )

    try:
        resp = client.chat(
            messages=[{"role": "user", "content": prompt}],
            stage="hcsp_constraint_extraction",
            extra={"chunk_id": chunk_id, "corpus": corpus},
            max_tokens=max_tokens,
            temperature=temperature,
        )
        output = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"Constraint extraction failed for {chunk_id}: {e}")
        return []

    # Parse the output
    parsed = _parse_constraint_output(output)

    # Convert to Constraint objects, filtering invalid ones
    constraints = []
    for p in parsed:
        constraint_text = p.get("constraint", "").strip()
        quote = p.get("quote", "").strip()
        ctype = p.get("type", "other").strip()

        # Validate constraint type
        if ctype not in ("attribute", "relation", "temporal", "other"):
            ctype = "other"

        # Skip if constraint text is empty or too short
        if len(constraint_text) < 10:
            continue

        # Skip if contains banned terms
        if _contains_banned(constraint_text, banned_terms):
            continue

        # Find the quote span in the original text
        span = _find_quote_span(chunk_text, quote)
        if span is None:
            # Try fallback: search for the constraint text itself
            span = _find_quote_span(chunk_text, constraint_text)
        if span is None:
            # Last resort: find any sentence containing key terms
            # Extract key terms from constraint (numbers, proper nouns, technical terms)
            key_terms = re.findall(r'\b(?:\d+[%$,.\d]*|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', constraint_text)
            if key_terms:
                for term in key_terms:
                    idx = chunk_text.find(term)
                    if idx >= 0:
                        # Found a key term, use surrounding context
                        start = max(0, idx - 50)
                        end = min(len(chunk_text), idx + len(term) + 100)
                        span = (start, end)
                        break
        if span is None:
            # Can't ground this constraint, skip it
            continue

        char_start, char_end = span
        evidence = EvidencePointer(
            chunk_id=chunk_id,
            char_start=char_start,
            char_end=char_end,
            text=chunk_text[char_start:char_end],
        )

        constraints.append(Constraint(
            text=constraint_text,
            evidence=evidence,
            constraint_type=ctype,
            corpus=corpus,
            entity=entity_context,  # Track which entity this constraint describes
        ))

        if len(constraints) >= max_constraints:
            break

    return constraints


def extract_constraints_batch(
    chunks: List[Dict[str, Any]],
    banned_terms: List[str],
    client: Any,
    max_constraints_per_chunk: int = 2,
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> Dict[str, List[Constraint]]:
    """
    Extract constraints from multiple chunks.

    Args:
        chunks: List of chunk dicts with 'chunk_id', 'text', 'source_type'
        banned_terms: Terms that must not appear in constraints
        client: VLLMClient for making API calls
        max_constraints_per_chunk: Max constraints to extract per chunk

    Returns:
        Dict mapping chunk_id -> list of constraints
    """
    results: Dict[str, List[Constraint]] = {}

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        chunk_text = chunk.get("text", "")
        source_type = chunk.get("source_type", "local")

        if source_type not in ("local", "web"):
            source_type = "local"

        constraints = extract_constraints(
            chunk_text=chunk_text,
            chunk_id=chunk_id,
            corpus=source_type,
            banned_terms=banned_terms,
            client=client,
            max_constraints=max_constraints_per_chunk,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        results[chunk_id] = constraints

    return results


# Utility functions for working with constraints

def filter_constraints_by_corpus(
    constraints: List[Constraint],
    corpus: Literal["local", "web"],
) -> List[Constraint]:
    """Filter constraints to only those from a specific corpus."""
    return [c for c in constraints if c.corpus == corpus]


def check_constraints_for_banned(
    constraints: List[Constraint],
    banned_terms: List[str],
) -> List[Tuple[Constraint, str]]:
    """
    Check constraints for banned terms.
    Returns list of (constraint, banned_term) for any violations.
    """
    violations = []
    for c in constraints:
        for term in banned_terms:
            if term.lower() in c.text.lower():
                violations.append((c, term))
                break
    return violations
