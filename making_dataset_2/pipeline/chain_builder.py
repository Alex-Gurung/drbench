"""Atomic entity-bridge chain builder.

Each document transition is handled by two atomic operations:
  1. find_bridge: find (target_doc, bridge_entity) via entity matching
  2. generate_question: 1 or 2 LLM calls depending on whether intra bridging needed

Usage:
    python -m making_dataset_2.pipeline.chain_builder \
        --pattern LWL \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --base-url http://127.0.0.1:8000/v1 \
        --n 5 --output chains.jsonl --verbose
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import logging
import random
import sys
import threading
import time
import uuid
from pathlib import Path

from making_dataset_2.data_loading import (
    build_doc_lookup,
    filter_seed_secrets,
    load_chunks_local,
    load_secrets,
)
from making_dataset_2.format_questions import format_numbered_questions
from making_dataset_2.llm import LLMClient
from making_dataset_2.pipeline.entity_index import EntityIndex
from making_dataset_2.pipeline.find_bridge import find_bridge
from making_dataset_2.retrieval.hybrid import HybridSearcher
from making_dataset_2.retrieval.search_client import SearchClient
from making_dataset_2.pipeline.step1_seed import select_seed, select_web_seed
from making_dataset_2.pipeline.step4_questions import (
    generate_question_constrained,
    generate_question_pick,
    rank_questions,
)
from making_dataset_2.pipeline.step5_check import (
    check_answer_needs_backref,
    check_answerable_without_doc,
    check_question,
    check_search_leaks_bridge,
)
from making_dataset_2.pipeline.step7_verify import verify_chain
from making_dataset_2.types import Chain, ChainState, HopRecord

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]

DEFAULT_SECRETS = ROOT_DIR / "making_dataset_2" / "outputs" / "secret_inventory.jsonl"
DEFAULT_CHUNKS_LOCAL = Path("/home/toolkit/nice_code/making_dataset/outputs/chunks_local.jsonl")
DEFAULT_CHUNKS_WEB = [
    Path("/home/toolkit/nice_code/making_dataset/outputs/chunks_web.jsonl"),  # BrowseComp-Plus (100K docs)
    ROOT_DIR / "making_dataset_2" / "outputs" / "chunks_web_drbench_urls.jsonl",  # curated DrBench URLs
]

MAX_CANDIDATES = 10  # Bridge candidates to generate questions for
MAX_ENTITY_LIST = 20  # Max entities to show in pick prompt
CONTEXT_WINDOW = 4000  # Chars of context around bridge entity for question generation


def _focused_context(doc_text: str, term: str, window: int = CONTEXT_WINDOW) -> str:
    """Extract context centered on the first occurrence of term in doc_text."""
    idx = doc_text.lower().find(term.lower())
    if idx == -1:
        return doc_text[:window]
    half = window // 2
    start = max(0, idx - half)
    end = min(len(doc_text), idx + len(term) + half)
    return doc_text[start:end]


def build_one_chain(
    *,
    state: ChainState,
    llm: LLMClient,
    entity_index: EntityIndex,
    local_doc_ids: set[str],
    web_doc_ids: set[str],
    record_trace: bool = True,
    local_searcher: HybridSearcher | None = None,
    web_searcher: HybridSearcher | None = None,
    retrieval_k: int = 50,
) -> Chain:
    """Build a single chain from an initialized seed state.

    For each transition in the pattern, runs the atomic loop:
      1. find_bridge → (target_doc, bridge_entity, needs_intra)
      2. if needs_intra: generate_question_constrained (current_doc)
      3. generate_question_pick (target_doc)
    """
    t0 = time.time()
    llm_calls = 0
    prev_answers: list[str] = [state.global_answer]
    used_answers: set[str] = {state.global_answer.lower()}

    pattern = state.pattern
    current_doc_id = state.hop_history[-1].doc_id
    current_answer = state.global_answer

    for jump_idx in range(len(pattern) - 1):
        src_type = pattern[jump_idx]
        dst_type = pattern[jump_idx + 1]
        next_pool = local_doc_ids if dst_type == "L" else web_doc_ids

        logger.info(
            "=== Transition %d/%d (%s→%s) answer=%r ===",
            jump_idx + 1, len(pattern) - 1, src_type, dst_type, current_answer,
        )

        dst_searcher = local_searcher if dst_type == "L" else web_searcher
        candidates = find_bridge(
            entity_index, current_doc_id, current_answer,
            next_pool, state.used_doc_ids, used_answers,
            searcher=dst_searcher, retrieval_k=retrieval_k,
        )

        if record_trace:
            state.trace.append({
                "step": "find_bridge",
                "transition": jump_idx + 1,
                "current_answer": current_answer,
                "current_doc": current_doc_id[:60],
                "n_candidates": len(candidates),
                "top_candidates": [
                    {"doc": c.target_doc_id[:60], "entity": c.bridge_entity,
                     "needs_intra": c.needs_intra, "score": c.score}
                    for c in candidates[:5]
                ],
            })

        if not candidates:
            logger.error("No bridge candidates for transition %d", jump_idx + 1)
            break

        # Generate questions for top candidates, collect valid ones, rank with LLM
        valid_options: list[dict] = []  # Each: {cand, intra_q/a (if needed), inter_q/a, target_doc_text}
        # Cache intra results per bridge_entity to avoid duplicate LLM calls
        intra_cache: dict[str, dict | None] = {}  # entity -> result dict or None if failed

        for cand in candidates[:MAX_CANDIDATES]:
            bridge_entity = cand.bridge_entity
            target_doc_id = cand.target_doc_id
            target_doc_text = entity_index.doc_text(target_doc_id)

            logger.info(
                "  Trying: %s via %r (intra=%s)",
                target_doc_id[:50], bridge_entity, cand.needs_intra,
            )

            option: dict = {"cand": cand, "target_doc_text": target_doc_text}

            # --- Intra question (if needed) ---
            if cand.needs_intra:
                be_key = bridge_entity.lower()
                if be_key in intra_cache:
                    cached = intra_cache[be_key]
                    if cached is None:
                        logger.info("  Skipping (intra already failed for %r)", bridge_entity)
                        continue
                    option.update(cached)
                    logger.info("  Intra cached: %r → %r", cached["intra_q"][:60], cached["intra_a"])
                else:
                    current_doc_text = entity_index.doc_text(current_doc_id)
                    intra_context = _focused_context(current_doc_text, current_answer)
                    try:
                        q_intra, a_intra, trace_intra = generate_question_constrained(
                            intra_context,
                            current_answer, bridge_entity, llm,
                        )
                        llm_calls += 1
                        if record_trace:
                            trace_intra["transition"] = jump_idx + 1
                            state.trace.append(trace_intra)
                    except ValueError as e:
                        logger.info("  Intra question failed: %s", e)
                        intra_cache[be_key] = None
                        continue

                    quote_intra = trace_intra.get("quote", "")
                    err = check_question(
                        q_intra, a_intra,
                        required_phrase=current_answer,
                        expected_answer=bridge_entity,
                        prev_answers=prev_answers,
                        quote=quote_intra,
                        doc_text=entity_index.doc_text(current_doc_id),
                    )
                    if record_trace:
                        state.trace.append({
                            "step": "check_intra", "transition": jump_idx + 1,
                            "passed": err is None, "error": err,
                            "question": q_intra, "answer": a_intra,
                            "quote": quote_intra,
                        })
                    if err:
                        logger.info("  Intra FAIL: %s | Q=%r A=%r", err, q_intra[:80], a_intra)
                        intra_cache[be_key] = None
                        continue

                    trivial = check_answerable_without_doc(q_intra, llm)
                    llm_calls += 1
                    if record_trace:
                        state.trace.append({
                            "step": "check_trivial_intra", "transition": jump_idx + 1,
                            "question": q_intra, "trivial": trivial.trivial,
                            "model_answer": trivial.answer, "justification": trivial.justification,
                        })
                    if trivial.trivial:
                        logger.info("  Intra TRIVIAL: %r → %r (%s)", q_intra[:60], trivial.answer, trivial.justification)
                        intra_cache[be_key] = None
                        continue

                    backref = check_answer_needs_backref(q_intra, current_answer, llm)
                    llm_calls += 1
                    if record_trace:
                        state.trace.append({
                            "step": "check_backref_intra", "transition": jump_idx + 1,
                            "question": q_intra, "prev_answer": current_answer,
                            "independent": backref.independent,
                            "model_answer": backref.answer, "justification": backref.justification,
                        })
                    if backref.independent:
                        logger.info("  Intra BACKREF-INDEPENDENT: %r answerable without %r → %r (%s)",
                                    q_intra[:60], current_answer, backref.answer, backref.justification)
                        intra_cache[be_key] = None
                        continue

                    intra_result = {"intra_q": q_intra, "intra_a": a_intra, "intra_quote": quote_intra}
                    intra_cache[be_key] = intra_result
                    option.update(intra_result)
                    logger.info("  Intra OK: %r → %r", q_intra[:60], a_intra)

            # --- Inter question (pick from entity list) ---
            target_entities = entity_index.entities_in_doc(target_doc_id)
            entity_list = [
                e for e in target_entities
                if e.lower() != bridge_entity.lower()
                and e.lower() not in used_answers
                and len(e) >= 3
            ][:MAX_ENTITY_LIST]

            if not entity_list:
                logger.info("  No viable entities in target doc, skipping")
                continue

            inter_context = _focused_context(target_doc_text, bridge_entity)
            try:
                q_inter, a_inter, trace_inter = generate_question_pick(
                    inter_context, bridge_entity, entity_list, llm,
                )
                llm_calls += 1
                if record_trace:
                    trace_inter["transition"] = jump_idx + 1
                    state.trace.append(trace_inter)
            except ValueError as e:
                logger.info("  Inter question failed: %s", e)
                continue

            quote_inter = trace_inter.get("quote", "")
            err = check_question(
                q_inter, a_inter,
                required_phrase=bridge_entity,
                prev_answers=prev_answers,
                quote=quote_inter,
                doc_text=target_doc_text,
            )
            if record_trace:
                state.trace.append({
                    "step": "check_inter", "transition": jump_idx + 1,
                    "passed": err is None, "error": err,
                    "question": q_inter, "answer": a_inter,
                    "quote": quote_inter,
                })
            if err:
                logger.info("  Inter FAIL: %s | Q=%r A=%r", err, q_inter[:80], a_inter)
                continue

            trivial = check_answerable_without_doc(q_inter, llm)
            llm_calls += 1
            if record_trace:
                state.trace.append({
                    "step": "check_trivial_inter", "transition": jump_idx + 1,
                    "question": q_inter, "trivial": trivial.trivial,
                    "model_answer": trivial.answer, "justification": trivial.justification,
                })
            if trivial.trivial:
                logger.info("  Inter TRIVIAL: %r → %r (%s)", q_inter[:60], trivial.answer, trivial.justification)
                continue

            backref_inter = check_answer_needs_backref(q_inter, bridge_entity, llm)
            llm_calls += 1
            if record_trace:
                state.trace.append({
                    "step": "check_backref_inter", "transition": jump_idx + 1,
                    "question": q_inter, "prev_answer": bridge_entity,
                    "independent": backref_inter.independent,
                    "model_answer": backref_inter.answer, "justification": backref_inter.justification,
                })
            if backref_inter.independent:
                logger.info("  Inter BACKREF-INDEPENDENT: %r answerable without %r → %r (%s)",
                            q_inter[:60], bridge_entity, backref_inter.answer, backref_inter.justification)
                continue

            # Check if web search for this question would leak the bridge value
            search_leak = check_search_leaks_bridge(q_inter, bridge_entity, llm)
            llm_calls += 1
            if record_trace:
                state.trace.append({
                    "step": "check_search_leak", "transition": jump_idx + 1,
                    "question": q_inter, "bridge_value": bridge_entity,
                    "would_leak": search_leak.would_leak,
                    "queries": search_leak.queries, "justification": search_leak.justification,
                })
            if not search_leak.would_leak:
                logger.info("  Inter NO-LEAK: bridge %r not in search queries %s",
                            bridge_entity, search_leak.queries)
                continue

            option["inter_q"] = q_inter
            option["inter_a"] = a_inter
            option["inter_quote"] = quote_inter
            logger.info("  Inter OK (leaks bridge): %r → %r  queries=%s", q_inter[:60], a_inter, search_leak.queries)
            valid_options.append(option)

        if not valid_options:
            logger.error("Failed transition %d (%s→%s): no valid options", jump_idx + 1, src_type, dst_type)
            break

        # --- LLM judge: rank valid options ---
        if len(valid_options) > 1:
            judge_candidates = [
                {"question": opt["inter_q"], "answer": opt["inter_a"]}
                for opt in valid_options
            ]
            ranking = rank_questions(judge_candidates, current_answer, llm)
            llm_calls += 1
            best_idx = ranking[0]
            logger.info("  LLM judge picked option %d/%d", best_idx + 1, len(valid_options))
        else:
            best_idx = 0

        best = valid_options[best_idx]
        cand = best["cand"]

        # Commit the chosen option to state
        if cand.needs_intra:
            hop_intra = HopRecord(
                hop_number=state.current_hop + 1,
                hop_type=src_type,
                question=best["intra_q"],
                answer=best["intra_a"],
                doc_id=current_doc_id,
                doc_text=entity_index.doc_text(current_doc_id),
                quote=best.get("intra_quote", ""),
            )
            state.hop_history.append(hop_intra)
            prev_answers.append(best["intra_a"])
            used_answers.add(best["intra_a"].lower())

        hop_inter = HopRecord(
            hop_number=state.current_hop + 1,
            hop_type=dst_type,
            question=best["inter_q"],
            answer=best["inter_a"],
            doc_id=cand.target_doc_id,
            doc_text=best["target_doc_text"],
            quote=best.get("inter_quote", ""),
        )
        state.hop_history.append(hop_inter)
        state.used_doc_ids.add(cand.target_doc_id)
        state.jumps_completed += 1
        prev_answers.append(best["inter_a"])
        used_answers.add(best["inter_a"].lower())
        current_doc_id = cand.target_doc_id
        current_answer = best["inter_a"]
        state.global_answer = best["inter_a"]

        # For W-starting chains, fill in task_id from first local doc
        # Local doc_ids follow pattern: local/{task_id}/subdir/filename
        if dst_type == "L" and not state.task_id:
            parts = cand.target_doc_id.split("/")
            if len(parts) >= 2 and parts[0] == "local":
                state.task_id = parts[1]

        logger.info(
            "  Committed: %r → %r (doc=%s)",
            best["inter_q"][:60], best["inter_a"], cand.target_doc_id[:40],
        )

    # Build the numbered chain question for verification and output
    hops_for_format = [
        {"question": h.question, "answer": h.answer, "hop_number": h.hop_number}
        for h in state.hop_history
    ]
    try:
        numbered_q = format_numbered_questions(hops_for_format)
    except ValueError:
        numbered_q = state.hop_history[0].question if state.hop_history else ""
    state.global_question = numbered_q

    # Step 7: Verify (only if chain is complete)
    verification = None
    if state.is_complete:
        logger.info("Chain complete (%d hops), verifying...", state.current_hop)
        verification, traces_s7 = verify_chain(state, llm)
        if record_trace:
            state.trace.extend(traces_s7)
        llm_calls += 4
        logger.info("Verification: valid=%s", verification.is_valid)

    elapsed = time.time() - t0

    return Chain(
        chain_id=str(uuid.uuid4())[:8],
        pattern=state.pattern,
        hop_history=state.hop_history,
        global_question=numbered_q,
        global_answer=state.global_answer,
        verification=verification,
        metadata={
            "task_id": state.task_id,
            "company": state.company,
            "llm_calls": llm_calls,
            "elapsed_seconds": round(elapsed, 1),
            "complete": state.is_complete,
            "n_hops": state.current_hop,
            "n_jumps": state.jumps_completed,
            "trace": state.trace if record_trace else [],
        },
    )


DOC_TEXT_LIMIT_OUTPUT = 5000

# Characters of context to show around the answer in evidence snippets
_EVIDENCE_WINDOW = 120


def _extract_evidence(doc_text: str, answer: str, max_snippets: int = 2) -> list[str]:
    """Find sentences containing the answer in doc_text."""
    lower_text = doc_text.lower()
    lower_answer = answer.strip().lower()
    snippets: list[str] = []
    start = 0
    while len(snippets) < max_snippets:
        idx = lower_text.find(lower_answer, start)
        if idx == -1:
            break
        # Expand to sentence boundaries
        sent_start = max(0, doc_text.rfind('.', max(0, idx - _EVIDENCE_WINDOW), idx) + 1)
        if sent_start == 0 and idx > _EVIDENCE_WINDOW:
            sent_start = idx - _EVIDENCE_WINDOW
        sent_end = doc_text.find('.', idx + len(lower_answer))
        if sent_end == -1 or sent_end > idx + len(lower_answer) + _EVIDENCE_WINDOW:
            sent_end = min(len(doc_text), idx + len(lower_answer) + _EVIDENCE_WINDOW)
        else:
            sent_end += 1  # include the period
        snippet = doc_text[sent_start:sent_end].strip()
        # Clean up newlines
        snippet = ' '.join(snippet.split())
        if snippet and snippet not in snippets:
            snippets.append(snippet)
        start = idx + len(lower_answer)
    return snippets


def _pretty_print_chain(chain: Chain) -> None:
    """Print chain with answers and evidence for each hop."""
    status = "VALID" if chain.verification and chain.verification.is_valid else "INCOMPLETE"
    print(f"\n  Chain {chain.chain_id} [{status}]")
    print(f"  {'─' * 70}")
    for i, hop in enumerate(chain.hop_history):
        is_final = i == len(chain.hop_history) - 1
        label = f"  Q{hop.hop_number}" + (" (final)" if is_final else "")
        print(f"{label}: {hop.question}")
        print(f"  A{hop.hop_number}: {hop.answer}")
        doc_short = hop.doc_id.split('/')[-1][:40]
        if hop.quote:
            quote_display = hop.quote[:200] + "..." if len(hop.quote) > 200 else hop.quote
            print(f"  Quote ({doc_short}): \"{quote_display}\"")
        else:
            # Fallback to extracted evidence for seed hops (no quote)
            evidence = _extract_evidence(hop.doc_text, hop.answer)
            if evidence:
                snip = evidence[0][:200] + "..." if len(evidence[0]) > 200 else evidence[0]
                print(f"  Evidence ({doc_short}): \"{snip}\"")
            else:
                print(f"  Evidence ({doc_short}): (answer not found in text)")
        if not is_final:
            print()
    print(f"  {'─' * 70}")
    # Print the numbered chain with (N) back-references
    if chain.global_question:
        print(f"\n  Numbered chain:")
        for line in chain.global_question.splitlines():
            print(f"    {line}")


def _chain_to_dict(chain: Chain) -> dict:
    """Convert Chain to JSON-serializable dict."""
    hops_for_format = [
        {"question": h.question, "answer": h.answer, "hop_number": h.hop_number}
        for h in chain.hop_history
    ]
    try:
        numbered = format_numbered_questions(hops_for_format)
    except ValueError as e:
        numbered = f"ERROR: {e}"

    d = {
        "chain_id": chain.chain_id,
        "pattern": chain.pattern,
        "numbered_questions": numbered,
        "global_question": chain.global_question,
        "global_answer": chain.global_answer,
        "hops": [
            {
                "hop_number": h.hop_number,
                "hop_type": h.hop_type,
                "question": h.question,
                "answer": h.answer,
                "doc_id": h.doc_id,
                "doc_text": h.doc_text[:DOC_TEXT_LIMIT_OUTPUT],
                "quote": h.quote,
            }
            for h in chain.hop_history
        ],
        "valid": chain.verification.is_valid if chain.verification else False,
        "metadata": chain.metadata,
    }
    if chain.verification is not None:
        d["verification"] = {
            "is_valid": chain.verification.is_valid,
            "no_docs_pass": chain.verification.no_docs_pass,
            "first_only_pass": chain.verification.first_only_pass,
            "last_only_pass": chain.verification.last_only_pass,
            "all_docs_pass": chain.verification.all_docs_pass,
            "all_docs_answer": chain.verification.all_docs_answer,
        }
    return d


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Atomic entity-bridge chain builder.")
    p.add_argument("--pattern", default=None, help="Single chain pattern (e.g. LWL)")
    p.add_argument("--patterns", nargs="+", default=None, help="Multiple patterns (e.g. LW WL LWL)")
    p.add_argument("--n", type=int, default=5, help="Number of chains per pattern")
    p.add_argument("--output", default="chains.jsonl", help="Output JSONL file")

    p.add_argument("--model", required=True, help="LLM model name")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    p.add_argument("--api-key", default=None, help="API key")

    p.add_argument("--secrets", default=str(DEFAULT_SECRETS))
    p.add_argument("--chunks-local", default=str(DEFAULT_CHUNKS_LOCAL))
    p.add_argument("--chunks-web", nargs="+", default=[str(p) for p in DEFAULT_CHUNKS_WEB])

    p.add_argument("--task", default=None, help="Filter by task ID (e.g. DR0001)")
    p.add_argument("--company", default=None, help="Filter by company name")
    p.add_argument("--spacy-model", default="en_core_web_trf")
    p.add_argument("--retrieval-mode", choices=["none", "bm25"], default="none",
                   help="Retrieval to expand bridge candidates (default: none = entity-only)")
    p.add_argument("--retrieval-k", type=int, default=50,
                   help="Number of chunks to retrieve for bridge expansion")
    p.add_argument("--search-url", default=None,
                   help="URL of BM25 search server (avoids loading indexes in-process)")
    p.add_argument("--workers", type=int, default=1, help="Parallel chain builders (default 1)")
    p.add_argument("--min-max-tokens", type=int, default=None,
                   help="Floor for max_tokens on all LLM calls (useful for reasoning models)")
    p.add_argument("--no-trace", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing output file, skipping already-completed chains")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _load_web_docs(chunks_web_paths: list[Path]) -> dict[str, str]:
    """Load web chunks from one or more JSONL files and build doc_id -> full text mapping."""
    docs: dict[str, list[tuple[int, str]]] = {}
    for path in chunks_web_paths:
        if not path.exists():
            logger.warning("Web chunks file not found: %s", path)
            continue
        logger.info("Loading web chunks from %s ...", path.name)
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                doc_id = row.get("doc_id", row.get("chunk_id", ""))
                text = row.get("text", "")
                offset = (row.get("offsets") or {}).get("start", 0)
                if doc_id and text:
                    docs.setdefault(doc_id, []).append((offset, text))

    result: dict[str, str] = {}
    for doc_id, chunks in docs.items():
        chunks.sort(key=lambda x: x[0])
        result[doc_id] = "\n\n".join(t for _, t in chunks)
    return result


def main() -> int:
    args = _parse_args()

    # Verbose: full logs to stderr. Quiet: only warnings to stderr, detailed to log file.
    output_path = Path(args.output)
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    else:
        log_file = output_path.with_suffix(".log")
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            filename=str(log_file),
            filemode="w",
        )
        # Show only warnings on stderr
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(logging.WARNING)
        logging.getLogger().addHandler(console)
        print(f"Detailed logs: {log_file}", file=sys.stderr)
    for _lib in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(_lib).setLevel(logging.WARNING)

    rng = random.Random(args.seed)

    # Determine patterns to run
    if args.patterns:
        patterns = [p.upper() for p in args.patterns]
    elif args.pattern:
        patterns = [args.pattern.upper()]
    else:
        patterns = ["LWL"]

    for p in patterns:
        if not all(c in "LW" for c in p) or len(p) < 2:
            logger.error("Invalid pattern %r: must be 2+ chars of L and W", p)
            return 1

    # Load local docs
    logger.info("Loading local chunks...")
    chunks = load_chunks_local(Path(args.chunks_local))
    doc_lookup = build_doc_lookup(chunks)
    local_docs = {doc_id: doc.text for doc_id, doc in doc_lookup.items()}
    local_doc_ids = set(local_docs.keys())
    logger.info("Loaded %d local documents", len(local_docs))

    # Load seeds
    secrets = load_secrets(Path(args.secrets))
    eligible = filter_seed_secrets(secrets, doc_lookup, task_id=args.task, company=args.company)
    logger.info("%d secrets, %d eligible seeds", len(secrets), len(eligible))

    # Group local docs and secrets by task_id (each task is an isolated doc set)
    task_local_doc_ids: dict[str, set[str]] = {}
    for doc_id in local_doc_ids:
        doc = doc_lookup.get(doc_id)
        tid = doc.meta.get("task_id", "") if doc else ""
        if tid:
            task_local_doc_ids.setdefault(tid, set()).add(doc_id)

    task_secrets: dict[str, list] = {}
    for s in eligible:
        doc = doc_lookup.get(s.doc_id)
        tid = doc.meta.get("task_id", "") if doc else ""
        if tid:
            task_secrets.setdefault(tid, []).append(s)

    # Filter by --task or --company
    if args.task:
        task_local_doc_ids = {t: ids for t, ids in task_local_doc_ids.items() if t == args.task}
        task_secrets = {t: s for t, s in task_secrets.items() if t == args.task}
    elif args.company:
        keep = {
            doc.meta.get("task_id", "")
            for doc in doc_lookup.values()
            if args.company.lower() in (doc.meta.get("company_name") or "").lower()
        }
        task_local_doc_ids = {t: ids for t, ids in task_local_doc_ids.items() if t in keep}
        task_secrets = {t: s for t, s in task_secrets.items() if t in keep}

    if not task_local_doc_ids:
        logger.error("No tasks matched after filtering. Check --task/--company flags.")
        return 1

    for tid in sorted(task_local_doc_ids):
        doc0 = next(iter(task_local_doc_ids[tid]))
        company = (doc_lookup[doc0].meta.get("company_name") or "?") if doc0 in doc_lookup else "?"
        logger.info("  %s (%s): %d docs, %d secrets",
                     tid, company, len(task_local_doc_ids[tid]), len(task_secrets.get(tid, [])))

    # Load web docs
    chunks_web_paths = [Path(p) for p in args.chunks_web]
    web_docs = _load_web_docs(chunks_web_paths)
    logger.info("Loaded %d web documents from %d files", len(web_docs), len(chunks_web_paths))
    web_doc_ids = set(web_docs.keys())

    # Build or load entity index
    all_docs = {**local_docs, **web_docs}
    cache_dir = ROOT_DIR / "making_dataset_2" / "outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"entity_index_{args.spacy_model}.json"

    # Always load spaCy for lazy NER on uncached docs (e.g., BrowseComp web docs)
    import spacy
    logger.info("Loading spaCy model %s...", args.spacy_model)
    nlp = spacy.load(args.spacy_model)

    if cache_path.exists():
        logger.info("Loading cached entity index from %s", cache_path)
        entity_index = EntityIndex.load(cache_path, all_docs, nlp=nlp)
    else:
        logger.info("Building entity index (spaCy NER: %s)...", args.spacy_model)
        entity_index = EntityIndex(nlp, all_docs)
        entity_index.save(cache_path)
        logger.info("Saved entity index to %s", cache_path)

    # Build retrieval searchers (optional, for bridge expansion)
    local_searcher = None
    web_searcher = None
    if args.retrieval_mode != "none":
        if args.search_url:
            # Use remote search server (saves ~15-20GB of in-process memory)
            logger.info("Using search server at %s", args.search_url)
            local_searcher = SearchClient(args.search_url, pool="local")
            web_searcher = SearchClient(args.search_url, pool="web")
            health = web_searcher.health()
            logger.info("Search server pools: %s", health.get("pools", {}))
        else:
            logger.info("Building in-process retrieval searchers (mode=%s, k=%d)...",
                        args.retrieval_mode, args.retrieval_k)
            chunks_local_path = Path(args.chunks_local)
            if chunks_local_path.exists():
                local_searcher = HybridSearcher(chunks_path=chunks_local_path)
                logger.info("Local searcher: %d chunks", local_searcher.size)
            existing_web_paths = [p for p in chunks_web_paths if p.exists()]
            if existing_web_paths:
                web_searcher = HybridSearcher.from_paths(existing_web_paths)
                logger.info("Web searcher: %d chunks from %d files", web_searcher.size, len(existing_web_paths))

    # Free data structures no longer needed (entity_index holds doc texts from here)
    del local_docs, web_docs, all_docs, chunks
    gc.collect()
    logger.info("Freed intermediate data, running gc")

    # Build LLM client
    llm = LLMClient(model=args.model, base_url=args.base_url, api_key=args.api_key,
                     min_max_tokens=args.min_max_tokens)
    record_trace = not args.no_trace

    # Build chains
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-plan all (pattern, index, rng_seed) tasks for deterministic parallel execution
    tasks: list[tuple[str, int, int]] = []
    for pattern in patterns:
        for i in range(args.n):
            tasks.append((pattern, i, rng.randint(0, 2**31)))

    total = len(tasks)
    results: list[dict] = [None] * total  # type: ignore[list-item]

    # Shared progress state
    lock = threading.Lock()
    completed = [0]
    n_valid = [0]
    n_complete = [0]
    t_start = time.time()
    pattern_stats: dict[str, dict[str, int]] = {p: {"valid": 0, "complete": 0, "done": 0} for p in patterns}

    def _progress_line() -> str:
        done = completed[0]
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 and done > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        parts = " | ".join(f"{p} {s['valid']}/{s['done']}" for p, s in pattern_stats.items() if s["done"])
        return (
            f"[{done}/{total}] "
            f"{n_valid[0]} valid, {n_complete[0] - n_valid[0]} complete-invalid, {done - n_complete[0]} incomplete  "
            f"ETA {int(eta // 60)}m{int(eta % 60):02d}s  "
            f"({parts})"
        )

    def _build_one(task_idx: int, pattern: str, idx: int, task_seed: int) -> None:
        task_rng = random.Random(task_seed)

        # Randomly pick a task for this chain
        task_ids = sorted(task_local_doc_ids.keys())
        chain_task = task_rng.choice(task_ids)
        chain_local_ids = task_local_doc_ids[chain_task]
        chain_secrets = task_secrets.get(chain_task, [])
        doc0 = next(iter(chain_local_ids))
        chain_company = (doc_lookup[doc0].meta.get("company_name") or "?") if doc0 in doc_lookup else "?"

        try:
            if pattern[0] == "L":
                state = select_seed(
                    chain_secrets, doc_lookup,
                    pattern=pattern,
                    task_id=chain_task, company=chain_company,
                    rng=task_rng,
                )
            else:
                state = select_web_seed(
                    entity_index, web_doc_ids, chain_local_ids, llm,
                    pattern=pattern,
                    task_id=chain_task, company=chain_company,
                    rng=task_rng,
                )
        except ValueError as e:
            logger.error("Seed selection failed for %s #%d (%s/%s): %s",
                         pattern, idx + 1, chain_task, chain_company, e)
            return

        chain = build_one_chain(
            state=state,
            llm=llm,
            entity_index=entity_index,
            local_doc_ids=chain_local_ids,
            web_doc_ids=web_doc_ids,
            record_trace=record_trace,
            local_searcher=local_searcher,
            web_searcher=web_searcher,
            retrieval_k=args.retrieval_k,
        )

        d = _chain_to_dict(chain)
        d["metadata"]["task_idx"] = task_idx
        is_valid = chain.verification and chain.verification.is_valid
        is_complete = chain.metadata.get("complete", False)

        with lock:
            results[task_idx] = d
            completed[0] += 1
            if is_valid:
                n_valid[0] += 1
            if is_complete:
                n_complete[0] += 1
            pattern_stats[pattern]["done"] += 1
            if is_valid:
                pattern_stats[pattern]["valid"] += 1
            if is_complete:
                pattern_stats[pattern]["complete"] += 1

            status = "VALID" if is_valid else ("COMPLETE" if is_complete else "INCOMPLETE")
            secs = chain.metadata.get("elapsed_seconds", 0)
            n_hops = len(chain.hop_history)
            print(
                f"  {status:10s} {pattern} #{idx+1:2d}  "
                f"{n_hops} hops  {secs:5.1f}s  "
                f"{chain.chain_id}  [{chain_task} {chain_company}]",
                file=sys.stderr,
            )
            print(f"  {_progress_line()}", file=sys.stderr)

            # Write incrementally
            with open(output_path, "a") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Resume: load existing results and skip completed task indices
    skip_indices: set[int] = set()
    if args.resume and output_path.exists() and output_path.stat().st_size > 0:
        for line in open(output_path):
            try:
                d = json.loads(line)
                tidx = d.get("metadata", {}).get("task_idx")
                if tidx is not None:
                    skip_indices.add(tidx)
                    results[tidx] = d
                    # Restore counters
                    p = d["pattern"]
                    completed[0] += 1
                    is_v = d.get("verification", {}).get("is_valid", False)
                    is_c = d.get("metadata", {}).get("complete", False)
                    if is_v:
                        n_valid[0] += 1
                    if is_c:
                        n_complete[0] += 1
                    pattern_stats[p]["done"] += 1
                    if is_v:
                        pattern_stats[p]["valid"] += 1
                    if is_c:
                        pattern_stats[p]["complete"] += 1
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"Resumed: {len(skip_indices)} chains already done, "
              f"{total - len(skip_indices)} remaining\n", file=sys.stderr)
    else:
        output_path.write_text("")

    workers = max(1, args.workers)
    remaining = total - len(skip_indices)
    print(f"Building {remaining} chains ({', '.join(f'{p}x{args.n}' for p in patterns)}) "
          f"with {workers} workers\n", file=sys.stderr)

    if workers == 1:
        for task_idx, (pattern, idx, task_seed) in enumerate(tasks):
            if task_idx in skip_indices:
                continue
            _build_one(task_idx, pattern, idx, task_seed)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for task_idx, (pattern, idx, task_seed) in enumerate(tasks):
                if task_idx in skip_indices:
                    continue
                futures.append(pool.submit(_build_one, task_idx, pattern, idx, task_seed))
            concurrent.futures.wait(futures)
            # Collect worker exceptions
            worker_errors = []
            for f in futures:
                exc = f.exception()
                if exc:
                    logger.error("Worker failed: %s", exc)
                    worker_errors.append(exc)
            if worker_errors:
                print(f"\nWARNING: {len(worker_errors)} worker(s) failed with exceptions",
                      file=sys.stderr)

    # Re-write in deterministic order (parallel may have written out-of-order)
    final_results = [r for r in results if r is not None]
    n_missing = total - len(skip_indices) - (len(final_results) - len(skip_indices))
    if n_missing > 0:
        print(f"\nWARNING: {n_missing} chain(s) missing from output (worker crashes or seed failures)",
              file=sys.stderr)
    with open(output_path, "w") as f:
        for d in final_results:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Final summary
    valid = sum(1 for d in final_results if d.get("verification", {}).get("is_valid"))
    complete = sum(1 for d in final_results if d.get("metadata", {}).get("complete"))
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Done in {int(elapsed // 60)}m{int(elapsed % 60):02d}s — "
          f"{valid}/{len(final_results)} valid, {complete}/{len(final_results)} complete", file=sys.stderr)
    for p in patterns:
        s = pattern_stats[p]
        print(f"  {p}: {s['valid']}/{s['done']} valid, {s['complete']}/{s['done']} complete", file=sys.stderr)
    print(f"\nWrote {len(final_results)} chains to {output_path}", file=sys.stderr)

    if n_missing > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
