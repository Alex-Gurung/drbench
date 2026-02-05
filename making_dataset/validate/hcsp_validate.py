"""
HCSP Task Validation - Validates generated tasks.

Validation checks:
1. Deterministic: answer in evidence, no banned terms leaked, valid spans
2. Shape: Chain structure validation (web on critical path for mixed)
3. LLM-based: ablation tests (full info, no info, partial info)
4. Retrieval-based: question should retrieve answer chunk in top-K
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

# Prompt for answering with evidence
ANSWER_WITH_EVIDENCE_TEMPLATE = """Answer the question using ONLY the provided evidence excerpts.

EVIDENCE:
{evidence_text}

QUESTION: {question}

RULES:
- If you can answer using ONLY the evidence, output the answer on a single line
- If you CANNOT answer using the evidence, output exactly: NOT_ANSWERABLE
- Do not guess or use outside knowledge

OUTPUT:
"""


def _check_banned_terms(text: str, banned_terms: List[str]) -> List[str]:
    """Check if text contains banned terms. Returns list of violations."""
    violations = []
    text_lower = text.lower()
    for term in banned_terms:
        if term.lower() in text_lower:
            violations.append(term)
    return violations


def _check_corpus_hints(text: str) -> List[str]:
    """Check for corpus hints that shouldn't appear."""
    corpus_hints = [
        "local", "web", "internal", "external", "document", "excerpt",
        "chunk", "corpus", "snippet", "evidence",
    ]
    violations = []
    text_lower = text.lower()
    for hint in corpus_hints:
        if re.search(rf"\b{hint}\b", text_lower):
            violations.append(hint)
    return violations


def validate_shape(
    task: Dict[str, Any],
    mode: str,
) -> Tuple[bool, List[str]]:
    """
    Validate tree shape / chain structure.

    For mixed mode:
    - Must have at least one web vertex
    - Must have at least one local vertex
    - Web must be on critical path (validated via ablations)

    Returns:
        Tuple of (passed, list_of_issues)
    """
    issues = []

    tree = task.get("tree", {})
    hops = tree.get("hops", [])

    if not hops:
        issues.append("no_hops")
        return False, issues

    # Count by source type
    local_count = sum(1 for h in hops if h.get("source_type") == "local")
    web_count = sum(1 for h in hops if h.get("source_type") == "web")

    if mode == "local_only":
        if web_count > 0:
            issues.append("local_only_has_web_evidence")
        if local_count == 0:
            issues.append("no_local_evidence")

    elif mode == "mixed":
        if local_count == 0:
            issues.append("mixed_missing_local_evidence")
        if web_count == 0:
            issues.append("mixed_missing_web_evidence")

    elif mode == "web_only":
        if local_count > 0:
            issues.append("web_only_has_local_evidence")
        if web_count == 0:
            issues.append("no_web_evidence")

    # Check constraints exist
    hcsp = tree.get("hcsp", {})
    nodes = hcsp.get("nodes", {})
    constraint_count = sum(1 for n in nodes.values() if n.get("kind") == "constraint")

    if constraint_count == 0:
        issues.append("no_constraints")

    # Check diversity metadata is present
    diversity = task.get("diversity", {})
    if not diversity:
        issues.append("missing_diversity_metadata")
    else:
        hop_pattern = diversity.get("hop_pattern", "")
        if mode == "mixed" and "W" not in hop_pattern:
            issues.append("hop_pattern_missing_web")
        if mode == "mixed" and "L" not in hop_pattern:
            issues.append("hop_pattern_missing_local")

    passed = len(issues) == 0
    return passed, issues


def validate_deterministic(
    task: Dict[str, Any],
    chunk_map: Dict[str, Dict[str, Any]],
    banned_terms: List[str],
) -> Tuple[bool, List[str]]:
    """
    Run deterministic validation checks.

    Checks:
    1. Answer string exists
    2. Question doesn't contain banned terms
    3. Question doesn't contain corpus hints
    4. All evidence pointers are valid (chunk exists, span is valid)

    Returns:
        Tuple of (passed, list_of_issues)
    """
    issues = []

    # Check answer exists
    answer = task.get("answer", "")
    if not answer:
        issues.append("missing_answer")

    # Check question
    question = task.get("question", "")
    if not question:
        issues.append("missing_question")
    else:
        # Check banned terms
        banned_violations = _check_banned_terms(question, banned_terms)
        for v in banned_violations:
            issues.append(f"banned_term_in_question: {v}")

        # Check corpus hints
        corpus_violations = _check_corpus_hints(question)
        for v in corpus_violations:
            issues.append(f"corpus_hint_in_question: {v}")

        # Check answer not in question
        if answer and answer.lower() in question.lower():
            issues.append("answer_in_question")

    # Validate evidence pointers in gold
    # Note: Web chunks (starting with "web/") are not in chunk_map, so skip validation for them
    gold = task.get("gold", {})
    if "answer_evidence" in gold:
        ev = gold["answer_evidence"]
        chunk_id = ev.get("chunk_id", "")
        char_start = ev.get("char_start", 0)
        char_end = ev.get("char_end", 0)

        is_web_chunk = chunk_id.startswith("web/")
        if not is_web_chunk and chunk_id not in chunk_map:
            issues.append(f"invalid_evidence_chunk: {chunk_id}")
        elif not is_web_chunk and chunk_id in chunk_map:
            chunk_text = chunk_map[chunk_id].get("text", "")
            if char_end > len(chunk_text):
                issues.append(f"evidence_span_out_of_bounds: {chunk_id}")

    # Validate constraint evidence
    tree = task.get("tree", {})
    hcsp = tree.get("hcsp", {})
    nodes = hcsp.get("nodes", {})
    for node_id, node in nodes.items():
        if node.get("kind") == "constraint":
            constraint = node.get("constraint", {})
            evidence = constraint.get("evidence", {})
            chunk_id = evidence.get("chunk_id", "")
            # Skip validation for web chunks (not in chunk_map)
            is_web_chunk = chunk_id.startswith("web/")
            if chunk_id and not is_web_chunk and chunk_id not in chunk_map:
                issues.append(f"invalid_constraint_chunk: {chunk_id}")

    passed = len(issues) == 0
    return passed, issues


def validate_ablation_full_info(
    task: Dict[str, Any],
    chunk_map: Dict[str, Dict[str, Any]],
    client: Any,  # VLLMClient
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> Tuple[bool, str]:
    """
    Validate that question is answerable with full evidence.

    Returns:
        Tuple of (passed, model_answer)
    """
    question = task.get("question", "")
    expected_answer = task.get("answer", "")

    # Gather evidence from all hops
    tree = task.get("tree", {})
    hops = tree.get("hops", [])

    evidence_texts = []
    for hop in hops:
        chunk_id = hop.get("chunk_id", "")
        if chunk_id in chunk_map:
            text = chunk_map[chunk_id].get("text", "")
            if text:
                evidence_texts.append(text[:2000])  # Truncate long chunks

    evidence_text = "\n\n---\n\n".join(evidence_texts)

    prompt = ANSWER_WITH_EVIDENCE_TEMPLATE.format(
        evidence_text=evidence_text,
        question=question,
    )

    try:
        resp = client.chat(
            messages=[{"role": "user", "content": prompt}],
            stage="hcsp_validate_full_info",
            extra={},
            max_tokens=max_tokens,
            temperature=temperature,
        )
        model_answer = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return False, f"LLM_ERROR: {e}"

    # Check if model found the answer
    if "NOT_ANSWERABLE" in model_answer.upper():
        return False, model_answer

    # Check if answer is approximately correct (fuzzy match)
    model_answer_lower = model_answer.lower()
    expected_lower = expected_answer.lower()

    # Exact match or containment
    if expected_lower in model_answer_lower or model_answer_lower in expected_lower:
        return True, model_answer

    # For numeric answers, try parsing
    try:
        model_nums = re.findall(r"[\d.]+", model_answer)
        expected_nums = re.findall(r"[\d.]+", expected_answer)
        if model_nums and expected_nums:
            model_val = float(model_nums[0])
            expected_val = float(expected_nums[0])
            if abs(model_val - expected_val) < 0.1:
                return True, model_answer
    except Exception:
        pass

    # Consider partial match as success (first few words match)
    if model_answer_lower[:20] == expected_lower[:20]:
        return True, model_answer

    return False, model_answer


def validate_ablation_no_info(
    task: Dict[str, Any],
    client: Any,  # VLLMClient
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> Tuple[bool, str]:
    """
    Validate that question is NOT answerable without evidence.

    Returns:
        Tuple of (passed, model_answer)
        passed=True if model correctly says NOT_ANSWERABLE
    """
    question = task.get("question", "")

    prompt = ANSWER_WITH_EVIDENCE_TEMPLATE.format(
        evidence_text="(No evidence provided)",
        question=question,
    )

    try:
        resp = client.chat(
            messages=[{"role": "user", "content": prompt}],
            stage="hcsp_validate_no_info",
            extra={},
            max_tokens=max_tokens,
            temperature=temperature,
        )
        model_answer = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return False, f"LLM_ERROR: {e}"

    # Should be NOT_ANSWERABLE
    if "NOT_ANSWERABLE" in model_answer.upper():
        return True, model_answer

    return False, model_answer


def validate_ablation_corpus(
    task: Dict[str, Any],
    chunk_map: Dict[str, Dict[str, Any]],
    corpus: Literal["local", "web"],
    client: Any,
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> Tuple[bool, str]:
    """
    Validate with only one corpus's evidence (for mixed mode).

    For mixed tasks, should be NOT_ANSWERABLE with only one corpus.

    Returns:
        Tuple of (passed, model_answer)
        passed=True if model correctly says NOT_ANSWERABLE
    """
    question = task.get("question", "")

    # Gather evidence from only the specified corpus
    tree = task.get("tree", {})
    hops = tree.get("hops", [])

    evidence_texts = []
    for hop in hops:
        source_type = hop.get("source_type", "local")
        if source_type == corpus:
            chunk_id = hop.get("chunk_id", "")
            if chunk_id in chunk_map:
                text = chunk_map[chunk_id].get("text", "")
                if text:
                    evidence_texts.append(text[:2000])

    if not evidence_texts:
        evidence_text = "(No evidence from this corpus)"
    else:
        evidence_text = "\n\n---\n\n".join(evidence_texts)

    prompt = ANSWER_WITH_EVIDENCE_TEMPLATE.format(
        evidence_text=evidence_text,
        question=question,
    )

    try:
        resp = client.chat(
            messages=[{"role": "user", "content": prompt}],
            stage=f"hcsp_validate_{corpus}_only",
            extra={},
            max_tokens=max_tokens,
            temperature=temperature,
        )
        model_answer = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return False, f"LLM_ERROR: {e}"

    # For mixed mode, should be NOT_ANSWERABLE with only one corpus
    if "NOT_ANSWERABLE" in model_answer.upper():
        return True, model_answer

    return False, model_answer


def validate_retrieval(
    task: Dict[str, Any],
    searcher: Any,  # UnifiedSearcher
    top_k: int = 20,
) -> Tuple[bool, int]:
    """
    Validate that the question retrieves the answer chunk.

    Returns:
        Tuple of (passed, rank)
        passed=True if answer chunk is in top-K
    """
    question = task.get("question", "")
    gold = task.get("gold", {})

    # Get answer chunk ID
    answer_evidence = gold.get("answer_evidence", {})
    answer_chunk_id = answer_evidence.get("chunk_id", "")

    if not answer_chunk_id:
        # Try to get from first hop
        tree = task.get("tree", {})
        hops = tree.get("hops", [])
        if hops:
            answer_chunk_id = hops[0].get("chunk_id", "")

    if not answer_chunk_id:
        return False, -1

    # Determine corpus from chunk_id
    if answer_chunk_id.startswith("web/") or "/web/" in answer_chunk_id:
        corpus = "web"
    else:
        corpus = "local"

    # Search
    try:
        hits = searcher.search(question, corpus=corpus, k=top_k)
    except Exception as e:
        print(f"Retrieval validation failed: {e}")
        return False, -1

    # Find rank of answer chunk
    for i, hit in enumerate(hits):
        if hit.chunk_id == answer_chunk_id:
            return True, i + 1

    return False, -1


def validate_task(
    task: Dict[str, Any],
    chunk_map: Dict[str, Dict[str, Any]],
    banned_terms: List[str],
    client: Optional[Any] = None,
    searcher: Optional[Any] = None,
    run_ablations: bool = True,
    run_retrieval: bool = True,
) -> Dict[str, Any]:
    """
    Run full validation on a task.

    Returns dict with validation results.
    """
    results: Dict[str, Any] = {
        "deterministic_pass": False,
        "deterministic_issues": [],
        "shape_pass": False,
        "shape_issues": [],
        "ablation_full_info": None,
        "ablation_full_info_pass": None,
        "ablation_no_info": None,
        "ablation_no_info_pass": None,
        "ablation_local_only": None,
        "ablation_local_only_pass": None,
        "ablation_web_only": None,
        "ablation_web_only_pass": None,
        "retrieval_rank": None,
        "retrieval_pass": None,
    }

    # Shape validation (chain structure)
    mode = task.get("mode", "local_only")
    shape_pass, shape_issues = validate_shape(task, mode)
    results["shape_pass"] = shape_pass
    results["shape_issues"] = shape_issues

    # Deterministic checks
    det_pass, det_issues = validate_deterministic(task, chunk_map, banned_terms)
    results["deterministic_pass"] = det_pass
    results["deterministic_issues"] = det_issues

    # LLM ablations
    if run_ablations and client is not None:
        # Full info
        full_pass, full_answer = validate_ablation_full_info(task, chunk_map, client)
        results["ablation_full_info"] = full_answer
        results["ablation_full_info_pass"] = full_pass

        # No info
        no_pass, no_answer = validate_ablation_no_info(task, client)
        results["ablation_no_info"] = no_answer
        results["ablation_no_info_pass"] = no_pass

        # Corpus-specific (for mixed mode)
        mode = task.get("mode", "")
        if mode == "mixed":
            local_pass, local_answer = validate_ablation_corpus(
                task, chunk_map, "local", client
            )
            results["ablation_local_only"] = local_answer
            results["ablation_local_only_pass"] = local_pass

            web_pass, web_answer = validate_ablation_corpus(
                task, chunk_map, "web", client
            )
            results["ablation_web_only"] = web_answer
            results["ablation_web_only_pass"] = web_pass

    # Retrieval check
    if run_retrieval and searcher is not None:
        ret_pass, ret_rank = validate_retrieval(task, searcher)
        results["retrieval_rank"] = ret_rank
        results["retrieval_pass"] = ret_pass

    return results


def summarize_validation_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize validation results across multiple tasks."""
    total = len(results)
    if total == 0:
        return {"total": 0}

    summary = {
        "total": total,
        "shape_pass": sum(1 for r in results if r.get("shape_pass")),
        "deterministic_pass": sum(1 for r in results if r.get("deterministic_pass")),
        "ablation_full_info_pass": sum(
            1 for r in results if r.get("ablation_full_info_pass") is True
        ),
        "ablation_no_info_pass": sum(
            1 for r in results if r.get("ablation_no_info_pass") is True
        ),
        "ablation_local_only_pass": sum(
            1 for r in results if r.get("ablation_local_only_pass") is True
        ),
        "ablation_web_only_pass": sum(
            1 for r in results if r.get("ablation_web_only_pass") is True
        ),
        "retrieval_pass": sum(1 for r in results if r.get("retrieval_pass") is True),
    }

    # Compute rates
    for key in list(summary.keys()):
        if key != "total":
            summary[f"{key}_rate"] = summary[key] / total

    # Common issues
    all_issues = []
    for r in results:
        all_issues.extend(r.get("deterministic_issues", []))
    if all_issues:
        from collections import Counter
        summary["common_issues"] = dict(Counter(all_issues).most_common(10))

    return summary
