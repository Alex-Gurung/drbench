"""
Browser Agent - Executes actions for research tree construction.

The Browser executes 5 action types:
1. INIT_ROOT: Create question node from secret
2. ADD_CONSTRAINTS: Extract constraints from vertex's chunk
3. EXTEND_LOCAL: Retrieve local chunk, add as evidence
4. EXTEND_WEB: Retrieve web chunk, add as evidence
5. TERMINATE: (no-op, handled by planner)

Key insight from researcher feedback:
- Root is a QUESTION node, not an answer node
- Answer value lives only in gold.answer, not in the tree
- Only ban the final answer string, not intermediate entities
"""
from __future__ import annotations

import re
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .schema import (
    Action,
    Constraint,
    EvidencePointer,
    ResearchTree,
    TreeContext,
    Vertex,
)
from .constraints import extract_constraints


# Template for extracting ENTITIES for entity-based web search (InfoSeek blurring)
# Key: Extract specific named entities that we can search for and describe via constraints
EXTRACT_ENTITIES_TEMPLATE = """Extract specific named entities from this text that might have information available on the PUBLIC web.

TEXT:
{chunk_text}

Extract these types of entities:
- Product/system names (e.g., "FreshTrack", "OrderHub", "RetentionPlus")
- Regulatory frameworks mentioned (e.g., "FSMA 204", "FDA Food Traceability Rule")
- Industry certifications (e.g., "Retail Excellence Gold", "ISO 9001")
- Awards mentioned (e.g., "2024 Innovation Award", "Best Workplace Award")
- External organizations (e.g., "National Retail Federation", "FDA")
- Industry standards or programs (e.g., "HACCP", "Six Sigma")

DO NOT extract:
- Generic concepts (e.g., "employee engagement", "customer retention")
- Fictional person names (e.g., "Emily Patel", "David Chen")
- The company name itself (e.g., "Lee's Market")

Output each entity name on a new line, nothing else. Just the entities.
If no suitable entities are found, output NONE.
"""


@dataclass
class ActionResult:
    """Result of a browser action."""
    success: bool
    vertex_id: Optional[str] = None
    message: str = ""


class Browser:
    """
    Executes actions determined by Planner.

    Has access to:
    - Local chunks (via searcher)
    - Web chunks (via searcher)
    - LLM (via client)
    """

    def __init__(
        self,
        searcher: Any,  # UnifiedSearcher
        client: Any,    # VLLMClient
        chunk_map: Dict[str, Dict[str, Any]],
        web_backend: str = "bm25_rerank_dense",
        web_bm25_candidates_k: int = 200,
        rng: Optional[random.Random] = None,
    ):
        self.searcher = searcher
        self.client = client
        self.chunk_map = chunk_map
        self.web_backend = web_backend
        self.web_bm25_candidates_k = web_bm25_candidates_k
        self.rng = rng or random.Random()
        self._visited_chunks: Set[str] = set()

    def execute(
        self,
        action: Action,
        context: TreeContext,
        target_vertex_id: Optional[str] = None,
    ) -> ActionResult:
        """Execute an action on the tree."""
        if action == Action.INIT_ROOT:
            return self._init_root(context)
        elif action == Action.ADD_CONSTRAINTS:
            return self._add_constraints(context, target_vertex_id)
        elif action == Action.EXTEND_LOCAL:
            return self._extend_local(context, target_vertex_id)
        elif action == Action.EXTEND_WEB:
            return self._extend_web(context, target_vertex_id)
        elif action == Action.TERMINATE:
            return ActionResult(success=True, message="terminated")
        else:
            return ActionResult(success=False, message=f"unknown action: {action}")

    def _init_root(self, context: TreeContext) -> ActionResult:
        """
        Create root representing the UNKNOWN we're solving for.

        Key insight: Root does NOT contain the answer value.
        The answer lives only in gold.answer (passed via context).

        The root is a QUESTION node with:
        - variable: what we're asking about
        - answer_type: the type of answer expected
        """
        root = Vertex(
            id="root",
            kind="question",
            variable=context.seed_question,
            answer_type=self._infer_answer_type(context.gold_answer, context.seed_secret_type),
        )

        context.tree.vertices["root"] = root
        return ActionResult(success=True, vertex_id="root", message="initialized root")

    def _infer_answer_type(self, answer: str, secret_type: str) -> str:
        """Infer answer type from answer string and secret type."""
        if secret_type in ("kpi_numeric", "metrics"):
            return "metric"
        if secret_type in ("dates", "temporal"):
            return "date"
        if secret_type in ("names", "entities"):
            return "entity"

        # Infer from answer format
        if "%" in answer:
            return "metric"
        if re.search(r"\b\d{4}\b", answer):  # Year-like
            return "date"
        if re.search(r"\$[\d,]+", answer):  # Currency
            return "metric"

        return "fact"

    def _add_constraints(
        self,
        context: TreeContext,
        target_vertex_id: Optional[str],
    ) -> ActionResult:
        """
        Extract constraints from a vertex's chunk.

        Key insight from researcher feedback: We do NOT aggressively ban
        intermediate entities. Only ban the final answer string.
        """
        if not target_vertex_id:
            return ActionResult(success=False, message="no target vertex")

        vertex = context.tree.get_vertex_by_id(target_vertex_id)
        if not vertex:
            return ActionResult(success=False, message=f"vertex not found: {target_vertex_id}")

        if vertex.kind != "evidence":
            return ActionResult(success=False, message="can only add constraints to evidence")

        chunk_id = vertex.chunk_id
        if not chunk_id:
            return ActionResult(success=False, message="vertex has no chunk")

        # Get chunk text
        if chunk_id in self.chunk_map:
            chunk_text = self.chunk_map[chunk_id].get("text", "")
        elif vertex.text:
            chunk_text = vertex.text
        else:
            return ActionResult(success=False, message=f"chunk not found: {chunk_id}")

        if not chunk_text:
            return ActionResult(success=False, message="empty chunk text")

        # Only ban the final answer string (not intermediate entities)
        banned_terms = [context.gold_answer] if context.gold_answer else []

        # For web vertices with blurs_entity, pass entity_context so constraints
        # are tagged as describing that entity (InfoSeek-style blurring)
        entity_context = vertex.blurs_entity if vertex.source_type == "web" else None

        # Extract constraints
        constraints = extract_constraints(
            chunk_text=chunk_text,
            chunk_id=chunk_id,
            corpus=vertex.source_type or "local",
            banned_terms=banned_terms,
            client=self.client,
            max_constraints=2,
            entity_context=entity_context,
        )

        if not constraints:
            return ActionResult(success=False, message="no constraints extracted")

        vertex.constraints.extend(constraints)
        return ActionResult(
            success=True,
            vertex_id=target_vertex_id,
            message=f"added {len(constraints)} constraints",
        )

    def _extend_local(
        self,
        context: TreeContext,
        target_vertex_id: Optional[str],
    ) -> ActionResult:
        """
        Retrieve a related local chunk and create evidence vertex.

        1. Extract bridge terms from target vertex (or use seed question)
        2. Query local corpus
        3. Select best hit not already visited
        4. Create evidence vertex
        """
        tree = context.tree

        # Determine query
        if target_vertex_id and target_vertex_id != "root":
            vertex = tree.get_vertex_by_id(target_vertex_id)
            if vertex and vertex.text:
                bridge_terms = self._extract_bridge_terms(vertex.text)
                query = " ".join(bridge_terms[:3]) if bridge_terms else vertex.text[:100]
            else:
                query = context.seed_question
        else:
            # First extension - use seed question
            query = context.seed_question

        if not query:
            return ActionResult(success=False, message="no query for local search")

        # Search local corpus
        try:
            hits = self.searcher.search(query, corpus="local", k=20)
        except Exception as e:
            return ActionResult(success=False, message=f"local search failed: {e}")

        # Select hit not already visited
        selected = None
        for hit in hits:
            if hit.chunk_id not in self._visited_chunks:
                selected = hit
                break

        if not selected:
            return ActionResult(success=False, message="no new local chunks found")

        # Create evidence vertex
        vertex_id = f"L{len(tree.get_local_vertices()) + 1}"
        vertex = Vertex(
            id=vertex_id,
            kind="evidence",
            chunk_id=selected.chunk_id,
            doc_id=selected.doc_id,
            source_type="local",
            text=selected.text,
            bridge_query=query,
        )

        tree.vertices[vertex_id] = vertex
        self._visited_chunks.add(selected.chunk_id)

        # Add as child of root
        root = tree.get_root()
        if root:
            root.children.append(vertex_id)

        return ActionResult(
            success=True,
            vertex_id=vertex_id,
            message=f"added local evidence: {selected.chunk_id}",
        )

    def _extend_web(
        self,
        context: TreeContext,
        target_vertex_id: Optional[str],
    ) -> ActionResult:
        """
        Retrieve a web chunk that describes a specific entity from local data.

        InfoSeek-style entity blurring:
        1. Extract entities from local vertex
        2. Search web for each entity until we find relevant content
        3. Track which entity this web vertex describes (blurs_entity)
        """
        tree = context.tree
        blurs_entity = None  # Track which entity this web chunk describes

        # Get target vertex for entity extraction
        if target_vertex_id and target_vertex_id != "root":
            vertex = tree.get_vertex_by_id(target_vertex_id)
        else:
            # Use first local vertex if no target specified
            local_vertices = tree.get_local_vertices()
            vertex = local_vertices[0] if local_vertices else None

        # Extract entities from local vertex
        entities = []
        if vertex and vertex.text:
            entities = self._extract_entities(vertex.text)

        # Try searching for each entity in order
        selected = None
        used_query = None

        for entity in entities:
            try:
                hits = self.searcher.search(
                    entity,
                    corpus="web",
                    k=20,
                    web_backend=self.web_backend,
                    web_bm25_candidates_k=self.web_bm25_candidates_k,
                )
                # Select hit not already visited
                for hit in hits:
                    if hit.chunk_id not in self._visited_chunks:
                        selected = hit
                        blurs_entity = entity
                        used_query = entity
                        break
                if selected:
                    break
            except Exception:
                continue

        # Fallback: use seed question if no entities found content
        if not selected:
            query = context.seed_question
            if query:
                try:
                    hits = self.searcher.search(
                        query,
                        corpus="web",
                        k=20,
                        web_backend=self.web_backend,
                        web_bm25_candidates_k=self.web_bm25_candidates_k,
                    )
                    for hit in hits:
                        if hit.chunk_id not in self._visited_chunks:
                            selected = hit
                            used_query = query
                            break
                except Exception as e:
                    return ActionResult(success=False, message=f"web search failed: {e}")

        if not selected:
            return ActionResult(success=False, message="no new web chunks found")

        # Create evidence vertex with blurs_entity tracking
        vertex_id = f"W{len(tree.get_web_vertices()) + 1}"
        web_vertex = Vertex(
            id=vertex_id,
            kind="evidence",
            chunk_id=selected.chunk_id,
            doc_id=selected.doc_id,
            source_type="web",
            text=selected.text,
            bridge_query=used_query,
            blurs_entity=blurs_entity,  # Track which entity this describes
        )

        tree.vertices[vertex_id] = web_vertex
        self._visited_chunks.add(selected.chunk_id)

        # Add as child of root
        root = tree.get_root()
        if root:
            root.children.append(vertex_id)

        return ActionResult(
            success=True,
            vertex_id=vertex_id,
            message=f"added web evidence for entity '{blurs_entity}': {selected.chunk_id}" if blurs_entity else f"added web evidence: {selected.chunk_id}",
        )

    def _extract_entities(self, chunk_text: str) -> List[str]:
        """Extract specific named entities from chunk text for entity-based web search."""
        prompt = EXTRACT_ENTITIES_TEMPLATE.format(chunk_text=chunk_text[:4000])

        resp = self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            stage="hcsp_entity_extraction",
            extra={},
            max_tokens=128,
            temperature=0.3,
        )
        output = (resp.choices[0].message.content or "").strip()

        # Handle "NONE" response
        if output.upper() == "NONE":
            return []

        entities = [line.strip() for line in output.split("\n") if line.strip()]
        entities = [e for e in entities if 3 <= len(e) <= 80 and e.upper() != "NONE"]
        return entities[:5]

    def _extract_bridge_terms(self, chunk_text: str) -> List[str]:
        """Backwards compatibility wrapper."""
        return self._extract_entities(chunk_text)

    def add_answer_evidence(
        self,
        context: TreeContext,
        answer_chunk: Dict[str, Any],
        answer: str,
    ) -> ActionResult:
        """
        Add the answer chunk as the first local evidence.

        This is called after INIT_ROOT to establish the answer source.
        The answer VALUE is not stored in the vertex - it's in gold.answer.
        """
        tree = context.tree
        chunk_id = answer_chunk.get("chunk_id", "")
        chunk_text = answer_chunk.get("text", "")

        if not chunk_id or not chunk_text:
            return ActionResult(success=False, message="invalid answer chunk")

        # Create evidence vertex
        vertex_id = "L1"
        vertex = Vertex(
            id=vertex_id,
            kind="evidence",
            chunk_id=chunk_id,
            doc_id=answer_chunk.get("doc_id"),
            source_type="local",
            text=chunk_text,
            bridge_query=None,  # Answer chunk - no bridge
        )

        tree.vertices[vertex_id] = vertex
        self._visited_chunks.add(chunk_id)

        # Add as child of root
        root = tree.get_root()
        if root:
            root.children.append(vertex_id)

        return ActionResult(
            success=True,
            vertex_id=vertex_id,
            message=f"added answer evidence: {chunk_id}",
        )

    def reset_visited(self) -> None:
        """Reset visited chunks for new tree building."""
        self._visited_chunks.clear()
