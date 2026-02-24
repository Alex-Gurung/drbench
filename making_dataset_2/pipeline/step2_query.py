"""Step 2: Generate Search Query.

One LLM call. Turns chain context into a corpus-aware search query.
"""

from __future__ import annotations

import time

from making_dataset_2.llm import LLMClient
from making_dataset_2.parsing import extract_answer_or_fallback
from making_dataset_2.prompts import build_query_generation_prompt
from making_dataset_2.types import ChainState


def generate_query(
    state: ChainState,
    llm: LLMClient,
    *,
    target_corpus: str,
) -> tuple[str, dict]:
    """Generate a search query for the next hop.

    Returns:
        (query, trace_entry) — extracted query and trace dict with prompt/raw/parsed.

    Raises:
        ValueError: If the LLM returns an empty query.
    """
    prompt = build_query_generation_prompt(state, target_corpus)
    t0 = time.time()
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=4096,
    )
    duration = round(time.time() - t0, 2)
    query = extract_answer_or_fallback(raw).strip()
    if not query:
        raise ValueError(f"Empty query from LLM. Raw output: {raw[:500]}")

    trace = {
        "step": "step2_query",
        "hop": state.current_hop + 1,
        "prompt": prompt,
        "raw_output": raw,
        "parsed_query": query,
        "duration": duration,
    }
    return query, trace
