#!/usr/bin/env python3
"""
HCSP Dataset Generator - InfoSeek-style task generation.

Generates multi-hop QA tasks with fuzzy constraints that require
traversing multiple hops (local and/or web) to answer.

Usage:
    # Local-only (per-task scope)
    python hcsp_dataset.py --mode local_only --task-id DR0001 --num-tasks 20

    # Mixed mode
    python hcsp_dataset.py --mode mixed --task-id DR0001 --num-tasks 20

    # Web-only
    python hcsp_dataset.py --mode web_only --num-tasks 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from making_dataset.generate.hcsp.schema import (
    EvidencePointer,
    HCSPTask,
    RequiredSecret,
)
from making_dataset.generate.hcsp.tree_builder import (
    build_hop_tree,
    build_hcsp_tree,
    build_research_tree,
    research_tree_to_hops,
    research_tree_to_hcsp_tree,
    compute_diversity_metadata,
    find_intermediate_secrets,
    get_all_constraints_from_tree,
)
from making_dataset.generate.hcsp.synthesize import (
    build_banned_terms,
    synthesize_with_retry,
)
from making_dataset.index.unified_searcher import UnifiedSearcher
from making_dataset.utils.run_context import get_log_dir
from making_dataset.utils.vllm_client import VLLMClient
from making_dataset.validate.hcsp_validate import validate_task


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HCSP (InfoSeek-style) multi-hop QA tasks."
    )

    # Mode and scope
    parser.add_argument(
        "--mode",
        choices=["local_only", "web_only", "mixed"],
        default="local_only",
        help="Task mode: local_only, web_only, or mixed (default: local_only)",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="DRBench task ID for scoped local vault (e.g., DR0001). Required for local/mixed modes.",
    )

    # Data paths
    parser.add_argument(
        "--local-chunks",
        default=None,
        help="Path to local chunks JSONL. If --task-id is set, defaults to scoped path.",
    )
    parser.add_argument(
        "--secrets",
        default=None,
        help="Path to secret inventory JSONL. If --task-id is set, defaults to scoped path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output dataset JSONL path.",
    )

    # Web retrieval
    parser.add_argument(
        "--web-bm25-index",
        default="/home/toolkit/BrowseComp-Plus/indexes/bm25",
        help="BrowseComp-Plus BM25 index path.",
    )
    parser.add_argument(
        "--web-dense-index-glob",
        default="/home/toolkit/BrowseComp-Plus/indexes/qwen3-embedding-0.6b/corpus.shard*_of_4.pkl",
        help="Web dense index shards glob.",
    )
    parser.add_argument(
        "--web-backend",
        default="bm25_rerank_dense",
        choices=["bm25", "dense", "bm25_rerank_dense"],
        help="Web retrieval backend.",
    )
    parser.add_argument(
        "--web-bm25-candidates-k",
        type=int,
        default=200,
        help="BM25 candidates for dense reranking.",
    )

    # Generation parameters
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=20,
        help="Number of tasks to generate.",
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=4,
        help="Target number of hops per task.",
    )
    parser.add_argument(
        "--constraints-per-hop",
        type=int,
        default=2,
        help="Max constraints to extract per hop.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Max generation attempts (default: num_tasks * 10).",
    )

    # LLM
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="vLLM model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature for question synthesis.",
    )

    # Validation
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run validation on generated tasks.",
    )
    parser.add_argument(
        "--ablations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run LLM ablation tests during validation.",
    )

    # Workspace ID
    parser.add_argument(
        "--workspace-id",
        default=None,
        help="Workspace identifier. Defaults to drbench_<task_id>_hcsp_v1.",
    )

    return parser.parse_args()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_chunks_map(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load chunks as a map from chunk_id -> chunk."""
    chunk_map = {}
    for rec in _load_jsonl(path):
        chunk_id = rec.get("chunk_id", "")
        if chunk_id:
            chunk_map[chunk_id] = rec
    return chunk_map


def _load_secrets_by_chunk(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load secrets as a map from chunk_id -> list of secrets."""
    secrets_by_chunk: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in _load_jsonl(path):
        chunk_id = rec.get("chunk_id", "")
        secrets = rec.get("secrets", [])
        if chunk_id and secrets:
            secrets_by_chunk[chunk_id].extend(secrets)
    return dict(secrets_by_chunk)


def _find_answer_span(chunk_text: str, answer: str) -> Optional[EvidencePointer]:
    """Find the answer string in chunk text and return evidence pointer."""
    import re

    # Try exact match
    idx = chunk_text.find(answer)
    if idx >= 0:
        return EvidencePointer(
            chunk_id="",  # Will be filled in
            char_start=idx,
            char_end=idx + len(answer),
            text=answer,
        )

    # Try case-insensitive
    idx = chunk_text.lower().find(answer.lower())
    if idx >= 0:
        found_text = chunk_text[idx:idx + len(answer)]
        return EvidencePointer(
            chunk_id="",
            char_start=idx,
            char_end=idx + len(answer),
            text=found_text,
        )

    # Try fuzzy whitespace match
    pattern = re.escape(answer).replace(r"\ ", r"\s+")
    m = re.search(pattern, chunk_text, flags=re.IGNORECASE)
    if m:
        return EvidencePointer(
            chunk_id="",
            char_start=m.start(),
            char_end=m.end(),
            text=chunk_text[m.start():m.end()],
        )

    return None


def generate_task(
    secret: Dict[str, Any],
    answer_chunk: Dict[str, Any],
    chunk_map: Dict[str, Dict[str, Any]],
    secrets_by_chunk: Dict[str, List[Dict[str, Any]]],
    searcher: UnifiedSearcher,
    client: VLLMClient,
    mode: Literal["local_only", "web_only", "mixed"],
    workspace_id: str,
    target_hops: int,
    constraints_per_hop: int,
    web_backend: str,
    web_bm25_candidates_k: int,
    temperature: float,
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    """
    Generate a single HCSP task.

    Returns task dict or None if generation failed.
    """
    # Extract answer info
    answer = (secret.get("answer") or "").strip()
    question_seed = (secret.get("question") or "").strip()
    secret_type = secret.get("secret_type", "other")
    chunk_id = answer_chunk.get("chunk_id", "")
    chunk_text = answer_chunk.get("text", "")

    if not answer or not chunk_text:
        return None

    # Find answer span in chunk
    answer_evidence = _find_answer_span(chunk_text, answer)
    if answer_evidence is None:
        return None
    answer_evidence.chunk_id = chunk_id

    # Build research tree (InfoSeek-style with planner/browser loop)
    research_tree, tree_issues = build_research_tree(
        answer_chunk=answer_chunk,
        answer=answer,
        seed_question=question_seed,
        seed_secret_type=secret_type,
        searcher=searcher,
        client=client,
        chunk_map=chunk_map,
        mode=mode,
        target_evidence=target_hops,
        target_constraints=target_hops * constraints_per_hop,
        web_backend=web_backend,
        web_bm25_candidates_k=web_bm25_candidates_k,
        rng=rng,
    )

    # Convert to legacy formats for compatibility
    hops = research_tree_to_hops(research_tree)
    hcsp_tree = research_tree_to_hcsp_tree(research_tree, answer_evidence)

    if len(hops) < 1:
        return None

    # Build chunk_dicts from research tree
    chunk_dicts = []
    for vertex in research_tree.get_evidence_vertices():
        chunk_dict = {
            "chunk_id": vertex.chunk_id,
            "doc_id": vertex.doc_id,
            "text": vertex.text or chunk_map.get(vertex.chunk_id or "", {}).get("text", ""),
            "source_type": vertex.source_type,
        }
        chunk_dicts.append(chunk_dict)

    # Find intermediate secrets
    intermediate_secrets = find_intermediate_secrets(
        hops, chunk_dicts, secrets_by_chunk
    )

    # Build banned terms - only ban the final answer (not intermediate entities)
    # This is a key insight from researcher feedback
    banned_terms = build_banned_terms(
        answer=answer,
        intermediate_secrets=[],  # Don't ban intermediate entities
        doc_ids=[h.doc_id for h in hops if h.doc_id],
        filenames=[],
    )

    # Get constraints from research tree (extracted during tree building)
    constraints = get_all_constraints_from_tree(research_tree)
    if not constraints:
        return None

    # Build HCSP tree from research tree
    hcsp_tree = build_hcsp_tree(research_tree, answer_evidence)

    # Get linkage type for question synthesis guidance
    linkage_type = research_tree.linkage_type

    # Determine answer type
    if secret_type in ("kpi_numeric", "metrics"):
        answer_type = "metric"
    elif secret_type in ("dates", "temporal"):
        answer_type = "date"
    elif secret_type in ("names", "entities"):
        answer_type = "entity"
    else:
        answer_type = "fact"

    # Synthesize question with linkage-aware guidance
    # Pass seed question as anchor so generated question matches answer type
    question, violations = synthesize_with_retry(
        constraints=constraints,
        answer_type=answer_type,
        banned_terms=banned_terms,
        client=client,
        seed_question=question_seed,
        max_retries=3,
        temperature=temperature,
        linkage_type=linkage_type,
    )

    if not question:
        return None

    # Check for violations
    if violations:
        # Log but don't fail - we'll track prompt_leaks
        pass

    # Build required secrets
    final_secret = RequiredSecret(
        chunk_id=chunk_id,
        question=question_seed,
        answer=answer,
        secret_type=secret_type,
        is_intermediate=False,
    )
    required_secrets = [final_secret] + intermediate_secrets

    # Compute diversity metadata
    answer_corpus = "local" if answer_chunk.get("source_type", "local") == "local" else "web"
    diversity = compute_diversity_metadata(
        hops=hops,
        hcsp_tree=hcsp_tree,
        required_secrets=required_secrets,
        answer_corpus=answer_corpus,
        research_tree=research_tree,
    )

    # Build gold dict with linkage info
    gold = {
        "answer_evidence": answer_evidence.to_dict(),
        "source_secret": {
            "question": question_seed,
            "answer": answer,
            "secret_type": secret_type,
        },
    }
    if linkage_type:
        gold["linkage_type"] = linkage_type.value if hasattr(linkage_type, 'value') else str(linkage_type)

    # Build task
    task = HCSPTask(
        workspace_id=workspace_id,
        mode=mode,
        question=question,
        answer=answer,
        answer_type=answer_type,
        hops=hops,
        hcsp=hcsp_tree,
        gold=gold,
        required_secrets=required_secrets,
        unnecessary_secrets=[],  # TODO: find from other hops
        prompt_leaks=violations,
        diversity=diversity,
    )

    return task.to_dict()


def main() -> int:
    args = _parse_args()

    # Resolve paths based on task-id
    base_outputs = ROOT_DIR / "making_dataset" / "outputs"

    if args.mode == "web_only":
        print("ERROR: web_only mode is not yet implemented for HCSP tasks.")
        print("       HCSP requires local secrets as answer sources.")
        print("       Use local_only or mixed mode instead.")
        return 1

    if args.mode in ("local_only", "mixed"):
        if args.task_id is None:
            print("ERROR: --task-id is required for local_only and mixed modes")
            return 1

        scoped_dir = base_outputs / "scopes" / "task" / args.task_id

        if args.local_chunks is None:
            args.local_chunks = str(scoped_dir / "chunks_local.jsonl")
        if args.secrets is None:
            args.secrets = str(scoped_dir / "secret_inventory.jsonl")
        if args.output is None:
            args.output = str(scoped_dir / f"{args.mode}.hcsp.jsonl")
        if args.workspace_id is None:
            args.workspace_id = f"drbench_{args.task_id}_hcsp_v1"

    # Validate paths exist
    local_chunks_path = Path(args.local_chunks) if args.local_chunks else None
    secrets_path = Path(args.secrets) if args.secrets else None
    output_path = Path(args.output)

    if args.mode in ("local_only", "mixed"):
        if local_chunks_path is None or not local_chunks_path.exists():
            print(f"ERROR: Local chunks not found: {local_chunks_path}")
            print("Run split_scopes.py first to create per-task chunks.")
            return 1
        if secrets_path is None or not secrets_path.exists():
            print(f"ERROR: Secrets not found: {secrets_path}")
            print("Run split_scopes.py first to create per-task secrets.")
            return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Mode: {args.mode}")
    print(f"Task ID: {args.task_id}")

    chunk_map: Dict[str, Dict[str, Any]] = {}
    secrets_by_chunk: Dict[str, List[Dict[str, Any]]] = {}

    if args.mode in ("local_only", "mixed"):
        print(f"Loading local chunks from {local_chunks_path}...")
        chunk_map = _load_chunks_map(local_chunks_path)
        print(f"  Loaded {len(chunk_map)} chunks")

        print(f"Loading secrets from {secrets_path}...")
        secrets_by_chunk = _load_secrets_by_chunk(secrets_path)
        print(f"  Loaded secrets for {len(secrets_by_chunk)} chunks")

    # Initialize searcher
    print("Initializing searcher...")
    searcher = UnifiedSearcher(
        local_chunks_path=str(local_chunks_path) if local_chunks_path else None,
        web_bm25_index_path=args.web_bm25_index if args.mode != "local_only" else None,
        web_dense_index_glob=args.web_dense_index_glob if args.mode != "local_only" else None,
    )

    # Initialize LLM client
    log_dir = get_log_dir()
    client = VLLMClient(model=args.model, log_dir=log_dir)
    print(f"Logs: {log_dir}")

    # Build candidate pool
    candidates = []
    for chunk_id, secrets in secrets_by_chunk.items():
        if chunk_id not in chunk_map:
            continue
        chunk = chunk_map[chunk_id]
        for secret in secrets:
            if secret.get("question") and secret.get("answer"):
                candidates.append((chunk, secret))

    if not candidates:
        print("ERROR: No valid secret candidates found")
        return 1

    print(f"Found {len(candidates)} candidate secrets")

    # Generate tasks
    rng = random.Random(args.seed)
    max_attempts = args.max_attempts or args.num_tasks * 10

    tasks: List[Dict[str, Any]] = []
    skip_counts: Dict[str, int] = defaultdict(int)

    for attempt in tqdm(range(max_attempts), desc="Generating tasks"):
        if len(tasks) >= args.num_tasks:
            break

        chunk, secret = rng.choice(candidates)

        try:
            task = generate_task(
                secret=secret,
                answer_chunk=chunk,
                chunk_map=chunk_map,
                secrets_by_chunk=secrets_by_chunk,
                searcher=searcher,
                client=client,
                mode=args.mode,
                workspace_id=args.workspace_id,
                target_hops=args.hops,
                constraints_per_hop=args.constraints_per_hop,
                web_backend=args.web_backend,
                web_bm25_candidates_k=args.web_bm25_candidates_k,
                temperature=args.temperature,
                rng=rng,
            )

            if task is None:
                skip_counts["generation_failed"] += 1
                continue

            # Validate if requested
            if args.validate:
                # Build banned terms for validation
                banned_terms = build_banned_terms(
                    answer=task.get("answer", ""),
                    intermediate_secrets=[
                        s.get("answer", "")
                        for s in task.get("privacy", {}).get("required_secrets", [])
                        if s.get("is_intermediate")
                    ],
                    doc_ids=[h.get("doc_id", "") for h in task.get("tree", {}).get("hops", [])],
                    filenames=[],
                )

                validation_result = validate_task(
                    task=task,
                    chunk_map=chunk_map,
                    banned_terms=banned_terms,
                    client=client if args.ablations else None,
                    searcher=None,  # Skip retrieval validation for now
                    run_ablations=args.ablations,
                    run_retrieval=False,
                )

                if not validation_result.get("deterministic_pass"):
                    issues = validation_result.get("deterministic_issues", [])
                    for issue in issues:
                        skip_counts[f"validation_{issue}"] += 1
                    continue

                # Add validation results to task
                task["quality"] = {
                    "deterministic_pass": validation_result["deterministic_pass"],
                }
                if args.ablations:
                    task["quality"]["ablation_full_info"] = validation_result.get("ablation_full_info")
                    task["quality"]["ablation_full_info_pass"] = validation_result.get("ablation_full_info_pass")
                    task["quality"]["ablation_no_info"] = validation_result.get("ablation_no_info")
                    task["quality"]["ablation_no_info_pass"] = validation_result.get("ablation_no_info_pass")

            tasks.append(task)

        except Exception as e:
            skip_counts[f"exception_{type(e).__name__}"] += 1
            continue

    # Write output
    with output_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"\nWrote {output_path} ({len(tasks)} tasks)")
    print(f"Logs: {log_dir}")

    if skip_counts:
        print("\nSkip reasons:")
        for reason, count in sorted(skip_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    # Summary stats
    if tasks:
        hop_patterns = [t.get("diversity", {}).get("hop_pattern", "") for t in tasks]
        from collections import Counter
        print("\nHop pattern distribution:")
        for pattern, count in Counter(hop_patterns).most_common(10):
            print(f"  {pattern}: {count}")

        required_secrets_counts = [
            t.get("diversity", {}).get("required_local_secrets", 0) for t in tasks
        ]
        print(f"\nRequired local secrets: min={min(required_secrets_counts)}, max={max(required_secrets_counts)}, avg={sum(required_secrets_counts)/len(required_secrets_counts):.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
