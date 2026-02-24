"""Step 5: Validate & Select Best Bridge.

5a: Deterministic pre-checks (fast, no LLM).
5b: LLM validation (1 call per survivor).
5c: LLM selection (1 call if multiple survivors).
"""

from __future__ import annotations

import concurrent.futures
import logging
import time

from making_dataset_2.llm import LLMClient
from making_dataset_2.parsing import parse_selection, parse_validation
from making_dataset_2.prompts import build_bridge_selection_prompt, build_bridge_validation_prompt
from making_dataset_2.types import Bridge, ChainState, ValidationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 5a: Deterministic pre-checks
# ---------------------------------------------------------------------------

def precheck(bridge: Bridge, state: ChainState) -> str | None:
    """Return None if bridge passes, or a reason string if it fails.

    NOTE: We do NOT reject for previous answer appearing in the question —
    we need literal answers in questions for the (N) numbered format.
    """
    ans = bridge.answer.strip()
    prev = state.global_answer.strip()

    words = ans.split()
    if not words or len(words) > 5:
        return f"answer too long ({len(words)} words): {ans!r}"

    if ans.lower() == prev.lower():
        return f"answer same as previous: {ans!r} == {prev!r}"

    if prev.lower() in ans.lower() or ans.lower() in prev.lower():
        return f"substring containment: {ans!r} vs {prev!r}"

    return None


# ---------------------------------------------------------------------------
# Step 5b: LLM validation
# ---------------------------------------------------------------------------

def _validate_one(
    state: ChainState, bridge: Bridge, llm: LLMClient
) -> tuple[Bridge, ValidationResult | None, dict]:
    """Validate one bridge. Returns (bridge, validation_result, trace_entry)."""
    prompt = build_bridge_validation_prompt(state, bridge)
    t0 = time.time()
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=8192,
    )
    duration = round(time.time() - t0, 2)
    parsed = parse_validation(raw)

    trace = {
        "step": "step5b_validate",
        "hop": state.current_hop + 1,
        "candidate": bridge.candidate_index,
        "prompt": prompt,
        "raw_output": raw,
        "parsed": {
            "verdict": parsed.verdict,
            "hops_required": parsed.hops_required,
            "reason": parsed.reason,
        } if parsed else None,
        "duration": duration,
    }

    if parsed is None:
        return bridge, None, trace
    return bridge, ValidationResult(
        verdict=parsed.verdict,
        hops_required=parsed.hops_required,
        reason=parsed.reason,
    ), trace


def validate_bridges(
    state: ChainState,
    bridges: list[Bridge],
    llm: LLMClient,
    *,
    expected_hops: int | None = None,
) -> tuple[list[Bridge], list[dict]]:
    """Run LLM validation on all bridges, return those that pass + trace entries."""
    if not bridges:
        return [], []

    results: list[tuple[Bridge, ValidationResult | None, dict]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(bridges)) as pool:
        futures = [
            pool.submit(_validate_one, state, b, llm)
            for b in bridges
        ]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    accepted: list[Bridge] = []
    traces: list[dict] = []
    for bridge, vr, trace in results:
        traces.append(trace)
        if vr is None:
            logger.info("  5b [%d]: PARSE_FAIL", bridge.candidate_index)
            continue
        if vr.verdict != "ACCEPT":
            logger.info("  5b [%d]: %s — %s", bridge.candidate_index, vr.verdict, vr.reason[:100])
            continue
        if expected_hops is not None and vr.hops_required < expected_hops:
            logger.info("  5b [%d]: ACCEPT but hops_required=%d < expected=%d — %s", bridge.candidate_index, vr.hops_required, expected_hops, vr.reason[:80])
            continue
        logger.info("  5b [%d]: ACCEPT (hops=%d) — %s", bridge.candidate_index, vr.hops_required, vr.reason[:80])
        accepted.append(bridge)
    return accepted, traces


# ---------------------------------------------------------------------------
# Step 5c: LLM selection
# ---------------------------------------------------------------------------

def select_best(
    state: ChainState,
    bridges: list[Bridge],
    llm: LLMClient,
) -> tuple[Bridge | None, dict | None]:
    """Select the best bridge from validated survivors. Returns (bridge, trace_entry)."""
    if not bridges:
        return None, None
    if len(bridges) == 1:
        return bridges[0], None  # No LLM call needed, no trace

    prompt = build_bridge_selection_prompt(state, bridges)
    t0 = time.time()
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=8192,
    )
    duration = round(time.time() - t0, 2)
    idx = parse_selection(raw)

    trace = {
        "step": "step5c_select",
        "hop": state.current_hop + 1,
        "prompt": prompt,
        "raw_output": raw,
        "parsed_selection": idx,
        "duration": duration,
    }

    if idx is not None and 1 <= idx <= len(bridges):
        return bridges[idx - 1], trace
    logger.warning("  5c: selection parse failed (got %s), returning first bridge", idx)
    return bridges[0], trace


# ---------------------------------------------------------------------------
# Combined: precheck → validate → select
# ---------------------------------------------------------------------------

def validate_and_select(
    state: ChainState,
    bridges: list[Bridge],
    llm: LLMClient,
    *,
    expected_hops: int | None = None,
) -> tuple[Bridge | None, list[dict]]:
    """Full Step 5: precheck → LLM validate → LLM select. Returns (bridge, trace_entries)."""
    all_traces: list[dict] = []

    # 5a: deterministic pre-checks
    survivors: list[Bridge] = []
    for b in bridges:
        reason = precheck(b, state)
        trace_5a = {
            "step": "step5a_precheck",
            "hop": state.current_hop + 1,
            "candidate": b.candidate_index,
            "passed": reason is None,
            "reason": reason,
            "question": b.question,
            "answer": b.answer,
        }
        all_traces.append(trace_5a)
        if reason is None:
            survivors.append(b)
        else:
            logger.info("  5a REJECT [%d]: %s  (Q: %s → A: %s)", b.candidate_index, reason, b.question[:60], b.answer)

    logger.info("  5a: %d/%d pass precheck", len(survivors), len(bridges))
    for b in survivors:
        logger.info("  5a PASS [%d]: Q=%r A=%r", b.candidate_index, b.question[:80], b.answer)

    if not survivors:
        return None, all_traces

    # 5b: LLM validation
    validated, traces_5b = validate_bridges(state, survivors, llm, expected_hops=expected_hops)
    all_traces.extend(traces_5b)
    logger.info("  5b: %d/%d pass LLM validation (expected_hops=%s)", len(validated), len(survivors), expected_hops)
    if not validated:
        return None, all_traces

    # 5c: LLM selection
    selected, trace_5c = select_best(state, validated, llm)
    if trace_5c:
        all_traces.append(trace_5c)
    if selected:
        logger.info("  5c: selected [%d] Q=%r A=%r", selected.candidate_index, selected.question[:80], selected.answer)
    return selected, all_traces
