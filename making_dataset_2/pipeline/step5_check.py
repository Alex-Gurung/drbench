"""Deterministic validation of generated questions. No LLM calls."""

from __future__ import annotations


def check_question(
    question: str,
    answer: str,
    *,
    required_phrase: str,
    expected_answer: str | None = None,
    prev_answers: list[str] | None = None,
    quote: str | None = None,
    doc_text: str | None = None,
) -> str | None:
    """Validate a generated question/answer pair.

    Returns None if OK, or an error string if validation fails.

    Checks:
    1. required_phrase appears in question (case-insensitive)
    2. Answer is 1-5 words
    3. If expected_answer set, answer matches (case-insensitive)
    4. Answer differs from all previous answers
    5. Answer differs from required_phrase
    6. If quote and doc_text provided, quote must appear in doc_text
    7. If quote provided, answer must appear in quote
    """
    q = question.strip()
    a = answer.strip()

    if not q:
        return "empty question"
    if not a:
        return "empty answer"
    if required_phrase.lower() not in q.lower():
        return f"required phrase {required_phrase!r} not in question"
    words = a.split()
    if len(words) > 5:
        return f"answer too long ({len(words)} words): {a!r}"
    if expected_answer is not None:
        if a.lower() != expected_answer.strip().lower():
            return f"answer {a!r} != expected {expected_answer!r}"
    if prev_answers:
        for prev in prev_answers:
            if a.lower() == prev.strip().lower():
                return f"answer {a!r} duplicates previous answer"
    if a.lower() == required_phrase.strip().lower():
        return "answer same as required phrase"
    if quote and doc_text:
        # Fuzzy match: normalize whitespace and check case-insensitive
        norm_quote = ' '.join(quote.lower().split())
        norm_doc = ' '.join(doc_text.lower().split())
        if norm_quote not in norm_doc:
            return f"quote not found in document: {quote[:100]!r}"
        # Answer must appear in the quote (ensures grounding)
        if a.lower() not in norm_quote:
            return f"answer {a!r} not in quote: {quote[:100]!r}"
    return None
