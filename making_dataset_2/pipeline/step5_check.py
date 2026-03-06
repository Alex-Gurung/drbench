"""Validation of generated questions.

check_question: deterministic (no LLM).
check_answerable_without_doc: LLM-based trivial-answerability gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ANSWER_WITHOUT_DOC_PROMPT = """Answer the question WITHOUT access to any document.

Rules:
- If you cannot answer with high confidence, set the answer to NOT_ANSWERABLE.
- Do not guess.
- You MUST end your response with exactly these two lines:
  Answer: <your answer or NOT_ANSWERABLE>
  Justification: <brief reason>

Question:
{question}"""


@dataclass
class TrivialCheckResult:
    trivial: bool  # True = answerable without doc (bad)
    answer: str | None  # The model's answer, if trivial
    justification: str  # Why it can/can't answer


def _parse_answer_justification(raw: str) -> tuple[str, str]:
    """Extract Answer: and Justification: fields from LLM response."""
    answer_match = re.search(r"^Answer:\s*(.+)$", raw, re.MULTILINE | re.IGNORECASE)
    just_match = re.search(r"^Justification:\s*(.+)$", raw, re.MULTILINE | re.IGNORECASE)
    answer_line = answer_match.group(1).strip() if answer_match else ""
    justification = just_match.group(1).strip() if just_match else ""
    return answer_line, justification


def check_answerable_without_doc(question: str, llm) -> TrivialCheckResult:
    """Check if a question is trivially answerable without any document.

    Returns TrivialCheckResult. If .trivial is True, the question should be rejected.
    """
    raw = llm.chat(
        [{"role": "user", "content": ANSWER_WITHOUT_DOC_PROMPT.format(question=question)}],
        temperature=0.0,
        max_tokens=150,
    )
    answer_line, justification = _parse_answer_justification(raw)

    if answer_line.upper() == "NOT_ANSWERABLE":
        return TrivialCheckResult(trivial=False, answer=None, justification=justification)
    return TrivialCheckResult(trivial=True, answer=answer_line, justification=justification)


BACKREF_CHECK_PROMPT = """Can you answer this question WITHOUT knowing what "{placeholder}" refers to?

Question: {question}

Rules:
- "{placeholder}" is an unknown value — you don't know what it is.
- Try HARD to answer. Use reasoning, world knowledge, and process of elimination.
  For example, if only one entity fits regardless of the unknown value, you CAN answer.
- If you can determine or confidently guess the answer despite not knowing "{placeholder}",
  output the answer.
- ONLY output NOT_ANSWERABLE if changing "{placeholder}" to different values would
  genuinely change the answer.
- You MUST end your response with exactly these two lines:
  Answer: <your answer or NOT_ANSWERABLE>
  Justification: <brief reason>"""


@dataclass
class BackrefCheckResult:
    independent: bool  # True = answerable without the backref (bad)
    answer: str | None
    justification: str


def check_answer_needs_backref(question: str, prev_answer: str, llm) -> BackrefCheckResult:
    """Check if a question can be answered without knowing the back-referenced value.

    Replaces prev_answer with a placeholder and asks the LLM to answer.
    If the LLM can answer → the back-reference is decorative → reject.
    """
    placeholder = "an unknown entity"
    blanked = question.replace(prev_answer, placeholder)
    # If replacement didn't change anything, the prev_answer isn't in the question
    if blanked == question:
        return BackrefCheckResult(independent=False, answer=None, justification="prev_answer not in question")

    raw = llm.chat(
        [{"role": "user", "content": BACKREF_CHECK_PROMPT.format(
            question=blanked, placeholder=placeholder)}],
        temperature=0.0,
        max_tokens=150,
    )
    answer_line, justification = _parse_answer_justification(raw)

    if answer_line.upper() == "NOT_ANSWERABLE":
        return BackrefCheckResult(independent=False, answer=None, justification=justification)
    return BackrefCheckResult(independent=True, answer=answer_line, justification=justification)


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
        # Answer must appear somewhere in the document (not necessarily in the quote,
        # since for emails the answer is often the sender/subject while the quote
        # is the evidence sentence from the body)
        if a.lower() not in norm_doc:
            return f"answer {a!r} not found in document"
    return None
