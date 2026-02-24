"""Step 4: Compose Bridges (Parallel).

For each candidate document, ask the LLM to compose a bridge extending the chain.
Uses two-doc P1 approach: direct questions with literal previous answers (for (N) formatting).
No P2 blend — the numbered question format replaces blending.
"""

from __future__ import annotations

import concurrent.futures
import time
from typing import Sequence

from making_dataset_2.llm import LLMClient
from making_dataset_2.parsing import parse_bridge
from making_dataset_2.prompts import build_bridge_composition_prompt
from making_dataset_2.prompts import build_bridge_two_documents_composition_prompt
from making_dataset_2.retrieval.types import RetrievalHit
from making_dataset_2.types import Bridge, ChainState

# Max chars of document text to include in the prompt
DOC_TEXT_LIMIT = 10000


def _compose_one(
    state: ChainState,
    hit: RetrievalHit,
    candidate_index: int,
    llm: LLMClient,
) -> tuple[Bridge | None, dict]:
    """Compose a bridge for a single candidate (single-doc prompt)."""
    doc_text = (hit.text or "")[:DOC_TEXT_LIMIT]
    prompt = build_bridge_composition_prompt(state, doc_text)
    t0 = time.time()
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=8192,
    )
    duration = round(time.time() - t0, 2)
    parsed = parse_bridge(raw)

    trace = {
        "step": "step4_bridge",
        "hop": state.current_hop + 1,
        "candidate": candidate_index,
        "doc_id": hit.doc_id,
        "prompt": prompt,
        "raw_output": raw,
        "parsed": {
            "question": parsed.question,
            "answer": parsed.answer,
            "bridge_phrase": parsed.bridge_phrase,
            "justification": parsed.justification,
        } if parsed else None,
        "duration": duration,
    }

    if parsed is None:
        return None, trace
    return Bridge(
        bridge_phrase=parsed.bridge_phrase,
        question=parsed.question,
        answer=parsed.answer,
        doc_id=hit.doc_id,
        doc_text=hit.text or "",
        candidate_index=candidate_index,
        justification=parsed.justification,
    ), trace


def _compose_two_doc_one(
    state: ChainState,
    hit: RetrievalHit,
    candidate_index: int,
    llm: LLMClient,
) -> tuple[Bridge | None, dict]:
    """Compose a bridge using two-doc P1 only (direct question with literal answer).

    No P2 blend — the numbered question format handles indirection via (N) references.
    """
    doc_text = (hit.text or "")[:DOC_TEXT_LIMIT]
    prompt = build_bridge_two_documents_composition_prompt(state, doc_text)
    t0 = time.time()
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=8192,
    )
    duration = round(time.time() - t0, 2)
    parsed = parse_bridge(raw)

    trace = {
        "step": "step4_bridge",
        "hop": state.current_hop + 1,
        "candidate": candidate_index,
        "doc_id": hit.doc_id,
        "prompt": prompt,
        "raw_output": raw,
        "parsed": {
            "question": parsed.question,
            "answer": parsed.answer,
            "justification": parsed.justification,
        } if parsed else None,
        "duration": duration,
    }

    if parsed is None:
        return None, trace
    return Bridge(
        bridge_phrase="",
        question=parsed.question,
        answer=parsed.answer,
        doc_id=hit.doc_id,
        doc_text=hit.text or "",
        candidate_index=candidate_index,
        justification=parsed.justification,
    ), trace


def compose_bridges(
    state: ChainState,
    candidates: Sequence[RetrievalHit],
    llm: LLMClient,
    *,
    max_parallel: int = 10,
    two_documents: bool = True,
) -> tuple[list[Bridge], list[dict]]:
    """Compose bridges for top-N candidates in parallel.

    Returns (bridges, trace_entries) — successfully parsed bridges and all trace dicts.
    """
    n = min(len(candidates), max_parallel)
    if n == 0:
        return [], []

    compose_fn = _compose_two_doc_one if two_documents else _compose_one

    bridges: list[Bridge] = []
    traces: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futures = {
            pool.submit(compose_fn, state, candidates[i], i, llm): i
            for i in range(n)
        }
        for fut in concurrent.futures.as_completed(futures):
            bridge, trace = fut.result()
            traces.append(trace)
            if bridge is not None:
                bridges.append(bridge)

    return bridges, traces
