"""
HCSP Tree Builder - Constructs research trees via planner/browser loop.

Implements InfoSeek-style incremental tree construction:
1. Planner decides WHAT action to take
2. Browser executes the action
3. Loop until termination criteria met

Key changes from original (based on researcher feedback):
- Root is a QUESTION node, not an answer node
- Answer value stored only in gold.answer, not in tree
- Minimal banning (only final answer, not intermediate entities)
- Shape-based validation for chain structure
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Literal, Optional, Tuple

from .schema import (
    Action,
    Constraint,
    EvidencePointer,
    HCSPNode,
    HCSPTree,
    Hop,
    RequiredSecret,
    ResearchTree,
    TaskDiversity,
    TreeContext,
    Vertex,
)
from .planner import Planner, PlannerConfig
from .browser import Browser
from .linkage import detect_linkage_type


def build_research_tree(
    answer_chunk: Dict[str, Any],
    answer: str,
    seed_question: str,
    seed_secret_type: str,
    searcher: Any,  # UnifiedSearcher
    client: Any,    # VLLMClient
    chunk_map: Dict[str, Dict[str, Any]],
    mode: Literal["local_only", "web_only", "mixed"],
    target_evidence: int = 3,
    target_constraints: int = 4,
    web_backend: str = "bm25_rerank_dense",
    web_bm25_candidates_k: int = 200,
    rng: Optional[random.Random] = None,
    max_actions: int = 20,
) -> Tuple[ResearchTree, List[str]]:
    """
    Build a research tree using planner/browser loop.

    Args:
        answer_chunk: The chunk containing the answer
        answer: The answer string (stored in gold, not tree)
        seed_question: The seed question from secret
        seed_secret_type: Type of secret (for answer_type inference)
        searcher: UnifiedSearcher for retrieval
        client: VLLMClient for LLM calls
        chunk_map: Map of chunk_id -> chunk dict
        mode: local_only, web_only, or mixed
        target_evidence: Target number of evidence vertices
        target_constraints: Target total constraints
        web_backend: Backend for web search
        web_bm25_candidates_k: BM25 candidates for reranking
        rng: Random number generator
        max_actions: Maximum actions before forced termination

    Returns:
        Tuple of (ResearchTree, list_of_issues)
    """
    if rng is None:
        rng = random.Random()

    issues: List[str] = []

    # Initialize tree and context
    tree = ResearchTree()
    context = TreeContext(
        tree=tree,
        mode=mode,
        target_constraints=target_constraints,
        target_evidence=target_evidence,
        gold_answer=answer,
        seed_question=seed_question,
        seed_secret_type=seed_secret_type,
    )

    # Initialize planner and browser
    planner_config = PlannerConfig(
        target_evidence=target_evidence,
        target_constraints=target_constraints,
    )
    planner = Planner(mode=mode, config=planner_config)
    browser = Browser(
        searcher=searcher,
        client=client,
        chunk_map=chunk_map,
        web_backend=web_backend,
        web_bm25_candidates_k=web_bm25_candidates_k,
        rng=rng,
    )

    # Main loop
    for action_num in range(max_actions):
        action, target_vertex_id = planner.next_action(context)

        if action == Action.TERMINATE:
            break

        # Special handling for first local extension - use answer chunk
        if action == Action.EXTEND_LOCAL and len(tree.get_local_vertices()) == 0:
            result = browser.add_answer_evidence(context, answer_chunk, answer)
        else:
            result = browser.execute(action, context, target_vertex_id)

        planner.update_state(action, result.success)

        if not result.success:
            planner.handle_failure(action, result.message)
            issues.append(f"{action.name}: {result.message}")

    # Detect linkage type for mixed mode
    if mode == "mixed":
        tree.linkage_type = detect_linkage_type(tree, answer)

    # Validate tree shape
    if mode == "mixed" and not tree.has_web_vertex():
        issues.append("mixed mode requires web evidence")
    if not tree.has_local_vertex():
        issues.append("no local evidence")
    if tree.total_constraints() == 0:
        issues.append("no constraints extracted")

    return tree, issues


def research_tree_to_hops(tree: ResearchTree) -> List[Hop]:
    """
    Convert ResearchTree vertices to legacy Hop format.

    The Hop format is used for compatibility with existing code.
    """
    hops: List[Hop] = []
    hop_id = 1

    for vertex in tree.get_evidence_vertices():
        hop = Hop(
            hop_id=hop_id,
            chunk_id=vertex.chunk_id or "",
            doc_id=vertex.doc_id,
            source_type=vertex.source_type or "local",
            edge={
                "query": vertex.bridge_query,
                "vertex_id": vertex.id,
            },
        )
        hops.append(hop)
        hop_id += 1

    return hops


def research_tree_to_hcsp_tree(
    tree: ResearchTree,
    answer_evidence: EvidencePointer,
) -> HCSPTree:
    """
    Convert ResearchTree to legacy HCSPTree format.

    Creates HCSPNode for each constraint and an answer node.
    """
    nodes: Dict[str, HCSPNode] = {}
    constraint_ids: List[str] = []

    # Create constraint nodes from each evidence vertex
    for vertex in tree.get_evidence_vertices():
        for i, constraint in enumerate(vertex.constraints):
            node_id = f"{vertex.id}_C{i}"
            nodes[node_id] = HCSPNode(
                id=node_id,
                kind="constraint",
                constraint=constraint,
            )
            constraint_ids.append(node_id)

    # Create answer node
    answer_node = HCSPNode(
        id="A0",
        kind="answer",
        op="INTERSECT" if len(constraint_ids) > 1 else "EXTRACT",
        inputs=constraint_ids,
        answer_evidence=answer_evidence,
    )
    nodes["A0"] = answer_node

    return HCSPTree(root_id="A0", nodes=nodes)


def build_hop_tree(
    answer_chunk: Dict[str, Any],
    answer: str,
    answer_evidence: EvidencePointer,
    searcher: Any,  # UnifiedSearcher
    mode: Literal["local_only", "web_only", "mixed"],
    task_id: str,
    target_hops: int = 4,
    client: Any = None,  # VLLMClient - required
    web_backend: str = "bm25_rerank_dense",
    web_bm25_candidates_k: int = 200,
    rng: Optional[random.Random] = None,
    chunk_map: Optional[Dict[str, Dict[str, Any]]] = None,
    seed_question: str = "",
    seed_secret_type: str = "",
) -> Tuple[List[Hop], List[Dict[str, Any]], str]:
    """
    Build a hop tree starting from the answer chunk.

    Returns:
        Tuple of (hops, chunk_dicts, hop_pattern)
    """
    if chunk_map is None:
        chunk_map = {}
        if answer_chunk.get("chunk_id"):
            chunk_map[answer_chunk["chunk_id"]] = answer_chunk

    # Build research tree
    tree, issues = build_research_tree(
        answer_chunk=answer_chunk,
        answer=answer,
        seed_question=seed_question or "",
        seed_secret_type=seed_secret_type or "",
        searcher=searcher,
        client=client,
        chunk_map=chunk_map,
        mode=mode,
        target_evidence=target_hops,
        target_constraints=target_hops * 2,
        web_backend=web_backend,
        web_bm25_candidates_k=web_bm25_candidates_k,
        rng=rng,
    )

    # Convert to hops
    hops = research_tree_to_hops(tree)

    # Build chunk_dicts
    chunk_dicts: List[Dict[str, Any]] = []
    for vertex in tree.get_evidence_vertices():
        chunk_dict = {
            "chunk_id": vertex.chunk_id,
            "doc_id": vertex.doc_id,
            "text": vertex.text or chunk_map.get(vertex.chunk_id or "", {}).get("text", ""),
            "source_type": vertex.source_type,
        }
        chunk_dicts.append(chunk_dict)

    hop_pattern = tree.hop_pattern()

    return hops, chunk_dicts, hop_pattern


def build_hcsp_tree(
    research_tree: ResearchTree,
    answer_evidence: EvidencePointer,
) -> HCSPTree:
    """
    Build an HCSP tree from a research tree's constraints.
    """
    return research_tree_to_hcsp_tree(research_tree, answer_evidence)


def compute_diversity_metadata(
    hops: List[Hop],
    hcsp_tree: HCSPTree,
    required_secrets: List[RequiredSecret],
    answer_corpus: Literal["local", "web"],
    research_tree: Optional[ResearchTree] = None,
) -> TaskDiversity:
    """
    Compute diversity metadata for a task.
    """
    # Compute hop pattern
    if research_tree is not None:
        hop_pattern = research_tree.hop_pattern()
    else:
        hop_pattern = "".join("L" if h.source_type == "local" else "W" for h in hops)

    # Count constraints by corpus
    local_constraints = len(hcsp_tree.get_local_constraints())
    web_constraints = len(hcsp_tree.get_web_constraints())

    # Count required local secrets
    required_local = sum(1 for s in required_secrets if not s.is_intermediate)
    required_local += sum(1 for s in required_secrets if s.is_intermediate)

    return TaskDiversity(
        required_local_secrets=required_local,
        hop_pattern=hop_pattern,
        answer_corpus=answer_corpus,
        local_constraints=local_constraints,
        web_constraints=web_constraints,
        total_hops=len(hops),
    )


def find_intermediate_secrets(
    hops: List[Hop],
    chunk_dicts: List[Dict[str, Any]],
    secrets_by_chunk: Dict[str, List[Dict[str, Any]]],
) -> List[RequiredSecret]:
    """
    Find secrets in intermediate hops that could be required for retrieval.
    """
    intermediate_secrets: List[RequiredSecret] = []

    # Skip the answer hop (first hop), look at intermediate hops
    for i, (hop, chunk_dict) in enumerate(zip(hops[1:], chunk_dicts[1:]), start=1):
        if hop.source_type != "local":
            continue

        chunk_id = hop.chunk_id
        chunk_secrets = secrets_by_chunk.get(chunk_id, [])

        for secret in chunk_secrets:
            question = secret.get("question", "")
            answer = secret.get("answer", "")
            secret_type = secret.get("secret_type")

            if not question or not answer:
                continue

            intermediate_secrets.append(RequiredSecret(
                chunk_id=chunk_id,
                question=question,
                answer=answer,
                secret_type=secret_type,
                is_intermediate=True,
            ))

    return intermediate_secrets


def get_all_constraints_from_tree(tree: ResearchTree) -> List[Constraint]:
    """
    Get all constraints from a research tree.

    Helper for question synthesis.
    """
    return tree.all_constraints()
