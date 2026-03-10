"""Step 7: Verify Complete Chain.

Tests answerability under 4 conditions to ensure the chain truly requires
all hops. 4 LLM calls total.

Uses quotes (exact sentences from docs) as focused context when available,
falling back to truncated doc text.
"""

from __future__ import annotations

import concurrent.futures
import time

from making_dataset_2.llm import LLMClient
from making_dataset_2.parsing import parse_verification
from making_dataset_2.prompts import build_verification_prompt
from making_dataset_2.types import ChainState, HopRecord, VerificationResult

# Fallback: truncate each doc if no quote available
_VERIFY_DOC_LIMIT = 6_000

# Characters of context to include around a quote from the doc
_QUOTE_CONTEXT = 2000


def _context_around_quote(hop: HopRecord) -> str:
    """Get the quote + surrounding context from the doc.

    If the quote exists in the doc, expand to ~500 chars on each side.
    Falls back to truncated doc text if quote not found.
    """
    if not hop.quote:
        return hop.doc_text[:_VERIFY_DOC_LIMIT]

    needle = hop.quote.lower()
    haystack = hop.doc_text.lower()
    idx = haystack.find(needle)
    if idx == -1:
        # Quote not found verbatim — try normalized whitespace
        norm_needle = ' '.join(needle.split())
        norm_haystack = ' '.join(haystack.split())
        idx = norm_haystack.find(norm_needle)
        if idx == -1:
            return hop.doc_text[:_VERIFY_DOC_LIMIT]
        # Use normalized text for extraction
        start = max(0, idx - _QUOTE_CONTEXT)
        end = min(len(norm_haystack), idx + len(norm_needle) + _QUOTE_CONTEXT)
        return ' '.join(hop.doc_text.split())[start:end]

    start = max(0, idx - _QUOTE_CONTEXT)
    end = min(len(hop.doc_text), idx + len(hop.quote) + _QUOTE_CONTEXT)
    return hop.doc_text[start:end]


def _verify_condition(
    condition_name: str, question: str, context: str, llm: LLMClient
) -> tuple[str, bool, str, dict]:
    """Run one verification condition. Returns (name, answerable, content, trace)."""
    prompt = build_verification_prompt(context, question)
    t0 = time.time()
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=4096,
    )
    duration = round(time.time() - t0, 2)
    result = parse_verification(raw)

    trace = {
        "step": "step7_verify",
        "condition": condition_name,
        "prompt": prompt,
        "raw_output": raw,
        "parsed": {"answerable": result.answerable, "content": result.content},
        "duration": duration,
    }
    return condition_name, result.answerable, result.content, trace


def verify_chain(state: ChainState, llm: LLMClient) -> tuple[VerificationResult, list[dict]]:
    """Verify a complete chain with 4 answerability tests. Returns (result, trace_entries).

    Tests:
    1. No docs → should be NOT_ANSWERABLE
    2. First hop doc only → should be NOT_ANSWERABLE
    3. Last hop doc only → should be NOT_ANSWERABLE
    4. All docs → should be ANSWERABLE with correct answer
    """
    question = state.global_question
    hops = state.hop_history

    # Build contexts using quotes + surrounding context, with numbered labels
    all_contexts = [_context_around_quote(h) for h in hops]
    all_docs_text = "\n\n".join(
        f"=== Document {i+1} ===\n{ctx}" for i, ctx in enumerate(all_contexts)
    )
    first_doc_text = f"=== Document 1 ===\n{all_contexts[0]}" if all_contexts else ""
    last_doc_text = f"=== Document 1 ===\n{all_contexts[-1]}" if all_contexts else ""

    conditions = [
        ("no_docs", ""),
        ("first_only", first_doc_text),
        ("last_only", last_doc_text),
        ("all_docs", all_docs_text),
    ]

    # Run all 4 in parallel
    results: dict[str, tuple[bool, str]] = {}
    traces: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_verify_condition, name, question, ctx, llm): name
            for name, ctx in conditions
        }
        for fut in concurrent.futures.as_completed(futures):
            name, answerable, content, trace = fut.result()
            results[name] = (answerable, content)
            traces.append(trace)

    no_docs_answerable, _ = results["no_docs"]
    first_answerable, _ = results["first_only"]
    last_answerable, _ = results["last_only"]
    all_answerable, all_answer = results["all_docs"]

    # Valid chain: first 3 should be NOT answerable, last should be answerable
    is_valid = (
        not no_docs_answerable
        and not first_answerable
        and not last_answerable
        and all_answerable
    )

    verification = VerificationResult(
        no_docs_pass=not no_docs_answerable,
        first_only_pass=not first_answerable,
        last_only_pass=not last_answerable,
        all_docs_pass=all_answerable,
        all_docs_answer=all_answer,
        is_valid=is_valid,
        details={
            name: {"answerable": ans, "content": content}
            for name, (ans, content) in results.items()
        },
    )
    return verification, traces
