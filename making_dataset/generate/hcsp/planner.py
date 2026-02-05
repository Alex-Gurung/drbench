"""
Planner Agent - Strategic control over research tree construction.

The Planner decides WHAT to do next based on:
- Current tree state
- Mode (local_only, mixed)
- Complexity targets (depth, constraints)

Inspired by InfoSeek's dual-agent architecture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .schema import Action, LinkageType, ResearchTree, TreeContext, Vertex


@dataclass
class PlannerConfig:
    """Configuration for the planner."""
    target_evidence: int = 3           # Target number of evidence vertices
    target_constraints: int = 4        # Target total constraints
    min_web_for_mixed: int = 1         # Minimum web vertices for mixed mode
    min_local_for_mixed: int = 1       # Minimum local vertices for mixed mode
    max_consecutive_same: int = 2      # Max consecutive same-corpus extensions


class Planner:
    """
    Decides WHAT action to take next in tree construction.

    Strategy varies by mode:
    - local_only: Extend local until target met
    - mixed: Ensure both local and web evidence, web must be necessary
    """

    def __init__(
        self,
        mode: str,
        config: Optional[PlannerConfig] = None,
    ):
        self.mode = mode
        self.config = config or PlannerConfig()
        self._consecutive_local = 0
        self._consecutive_web = 0
        self._last_action: Optional[Action] = None
        self._action_count = 0

    def next_action(self, context: TreeContext) -> Tuple[Action, Optional[str]]:
        """
        Decide next action based on current tree state.

        Returns:
            Tuple of (action, target_vertex_id)
            target_vertex_id is the vertex to act on (for ADD_CONSTRAINTS)
        """
        tree = context.tree

        # Step 1: Initialize if empty
        if tree.is_empty():
            return Action.INIT_ROOT, None

        # Step 2: Check if we need evidence
        evidence_count = len(tree.get_evidence_vertices())

        if evidence_count == 0:
            # No evidence yet - add first evidence (local for answer)
            return Action.EXTEND_LOCAL, "root"

        # Step 3: Mode-specific strategy
        if self.mode == "local_only":
            return self._next_action_local_only(context)
        elif self.mode == "mixed":
            return self._next_action_mixed(context)
        else:
            return Action.TERMINATE, None

    def _next_action_local_only(self, context: TreeContext) -> Tuple[Action, Optional[str]]:
        """Strategy for local-only mode."""
        tree = context.tree

        # Check constraints target
        if tree.total_constraints() < context.target_constraints:
            # Find vertex needing constraints
            vertex = self._find_vertex_needing_constraints(tree)
            if vertex:
                return Action.ADD_CONSTRAINTS, vertex.id

        # Check evidence target
        if len(tree.get_evidence_vertices()) < context.target_evidence:
            return Action.EXTEND_LOCAL, tree.get_evidence_vertices()[-1].id

        # Targets met
        return Action.TERMINATE, None

    def _next_action_mixed(self, context: TreeContext) -> Tuple[Action, Optional[str]]:
        """
        Strategy for mixed mode - ensure web is NECESSARY.

        Key insight: For web to be necessary, we need a chain like:
        Question -> Web (provides context/entity) -> Local (provides answer)

        The web must be on the critical path to answering.
        """
        tree = context.tree
        evidence = tree.get_evidence_vertices()
        has_web = tree.has_web_vertex()
        has_local = tree.has_local_vertex()

        # Must have at least one local (for the answer)
        if not has_local:
            return Action.EXTEND_LOCAL, "root"

        # Must have at least one web (for mixed mode)
        if not has_web and len(evidence) >= 1:
            # Extend from last local to find web
            last_local = tree.get_local_vertices()[-1]
            return Action.EXTEND_WEB, last_local.id

        # Add constraints to evidence vertices
        if tree.total_constraints() < context.target_constraints:
            vertex = self._find_vertex_needing_constraints(tree)
            if vertex:
                return Action.ADD_CONSTRAINTS, vertex.id

        # Add more evidence if needed (alternating)
        if len(evidence) < context.target_evidence:
            last_corpus = tree.last_corpus()
            if last_corpus == "local" and self._consecutive_local < self.config.max_consecutive_same:
                # Try web next for variety
                last_vertex = evidence[-1]
                self._consecutive_web = 0
                return Action.EXTEND_WEB, last_vertex.id
            elif last_corpus == "web" and self._consecutive_web < self.config.max_consecutive_same:
                # Try local next
                last_vertex = evidence[-1]
                self._consecutive_local = 0
                return Action.EXTEND_LOCAL, last_vertex.id
            else:
                # Alternate
                if last_corpus == "local":
                    return Action.EXTEND_WEB, evidence[-1].id
                else:
                    return Action.EXTEND_LOCAL, evidence[-1].id

        # Ensure minimum constraints exist
        if tree.total_constraints() < 2:
            vertex = self._find_vertex_needing_constraints(tree)
            if vertex:
                return Action.ADD_CONSTRAINTS, vertex.id

        return Action.TERMINATE, None

    def _find_vertex_needing_constraints(self, tree: ResearchTree) -> Optional[Vertex]:
        """Find evidence vertex that needs constraints extracted."""
        for v in tree.get_evidence_vertices():
            if len(v.constraints) == 0:
                return v
        # All have at least one, find one with fewer than 2
        for v in tree.get_evidence_vertices():
            if len(v.constraints) < 2:
                return v
        return None

    def update_state(self, action: Action, success: bool) -> None:
        """Update planner state after action execution."""
        self._last_action = action
        self._action_count += 1

        if action == Action.EXTEND_LOCAL:
            if success:
                self._consecutive_local += 1
                self._consecutive_web = 0
        elif action == Action.EXTEND_WEB:
            if success:
                self._consecutive_web += 1
                self._consecutive_local = 0

    def handle_failure(self, action: Action, reason: str) -> None:
        """Handle action failure - adjust strategy if needed."""
        # For now, just track. Could implement retry logic.
        pass

    def suggest_linkage_type(self, tree: ResearchTree) -> Optional[LinkageType]:
        """
        Suggest a linkage type based on tree structure.

        This is heuristic - the actual linkage emerges from the evidence.
        """
        if not tree.has_web_vertex():
            return None  # No linkage for local-only

        local_vertices = tree.get_local_vertices()
        web_vertices = tree.get_web_vertices()

        if not local_vertices or not web_vertices:
            return None

        # Check if web has entity constraints
        web_has_entity = any(
            c.constraint_type in ("relation", "attribute")
            for v in web_vertices
            for c in v.constraints
        )

        # Check if local has numerical constraints
        local_has_numeric = any(
            "%" in c.text or any(char.isdigit() for char in c.text)
            for v in local_vertices
            for c in v.constraints
        )

        if web_has_entity and local_has_numeric:
            return LinkageType.ENTITY_CHAIN
        elif local_has_numeric and len(web_vertices) > 0:
            return LinkageType.COMPUTATIONAL

        return LinkageType.CREATIVE
