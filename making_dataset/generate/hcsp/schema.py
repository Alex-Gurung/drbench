"""
HCSP Schema - Dataclasses for InfoSeek-style task generation.

Key concepts:
- Constraint: A fuzzy fact mined from a chunk, used to describe (but not name) an entity
- HCSPNode: A node in the constraint tree, representing either a constraint or answer
- HCSPTree: The full tree structure with constraints and answer
- TaskDiversity: Metadata for tracking task characteristics (for analysis)
- Vertex: A node in the research tree (evidence from local/web corpus)
- ResearchTree: The full tree with question root and evidence children
- Action: Browser actions (INIT_ROOT, ADD_CONSTRAINTS, EXTEND_LOCAL, EXTEND_WEB, TERMINATE)
- LinkageType: How local and web evidence are connected (entity_chain, computational, etc.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Literal, Optional


@dataclass
class EvidencePointer:
    """Points to a specific span in a chunk."""
    chunk_id: str
    char_start: int
    char_end: int
    text: Optional[str] = None  # The actual text at this span (for convenience)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "chunk_id": self.chunk_id,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }
        if self.text is not None:
            d["text"] = self.text
        return d


@dataclass
class Constraint:
    """
    A fuzzy constraint mined from a chunk.

    Used to describe an entity/fact without naming it directly.
    Each constraint must be grounded in a specific text span.

    Example:
        text: "expanded to 12 locations in 2024"
        constraint_type: "attribute"
        corpus: "local"
        entity: "FreshTrack"  # This constraint describes FreshTrack
    """
    text: str                                           # The constraint statement
    evidence: EvidencePointer                           # Where this constraint comes from
    constraint_type: Literal["attribute", "relation", "temporal", "other"]
    corpus: Literal["local", "web"]                     # Which corpus this constraint is from
    entity: Optional[str] = None                        # Which entity this constraint describes (for blurring)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "text": self.text,
            "evidence": self.evidence.to_dict(),
            "constraint_type": self.constraint_type,
            "corpus": self.corpus,
        }
        if self.entity:
            d["entity"] = self.entity
        return d


@dataclass
class HCSPNode:
    """
    A node in the HCSP tree.

    Can be either:
    - A constraint node (kind="constraint"): represents a fuzzy constraint
    - An answer node (kind="answer"): represents the final answer derivation
    """
    id: str                                             # "S1", "S2", "A0", etc.
    kind: Literal["constraint", "answer"]

    # For constraint nodes
    constraint: Optional[Constraint] = None

    # For answer nodes
    op: Optional[Literal["INTERSECT", "EXTRACT"]] = None  # How answer is derived
    inputs: List[str] = field(default_factory=list)       # IDs of input constraints
    answer_evidence: Optional[EvidencePointer] = None     # Where the answer is found

    # Internal tracking (not in output)
    entity: Optional[str] = None                          # The entity name (blurred out of question)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
        }
        if self.kind == "constraint" and self.constraint:
            d["constraint"] = self.constraint.to_dict()
        if self.kind == "answer":
            if self.op:
                d["op"] = self.op
            if self.inputs:
                d["inputs"] = self.inputs
            if self.answer_evidence:
                d["answer_evidence"] = self.answer_evidence.to_dict()
        return d


@dataclass
class HCSPTree:
    """
    The full HCSP tree structure.

    Contains:
    - Constraint nodes (fuzzy facts from local and web)
    - Answer node (how constraints combine to identify the answer)
    """
    root_id: str                                        # ID of the answer node
    nodes: Dict[str, HCSPNode] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root_id,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
        }

    def get_constraints(self) -> List[Constraint]:
        """Return all constraint objects in the tree."""
        return [
            n.constraint for n in self.nodes.values()
            if n.kind == "constraint" and n.constraint is not None
        ]

    def get_local_constraints(self) -> List[Constraint]:
        """Return constraints from local corpus."""
        return [c for c in self.get_constraints() if c.corpus == "local"]

    def get_web_constraints(self) -> List[Constraint]:
        """Return constraints from web corpus."""
        return [c for c in self.get_constraints() if c.corpus == "web"]


@dataclass
class Hop:
    """
    A hop in the retrieval path.

    Tracks:
    - Which chunk was visited
    - What query was used to find it
    - What corpus it came from
    """
    hop_id: int
    chunk_id: str
    doc_id: Optional[str]
    source_type: Literal["local", "web"]
    edge: Optional[Dict[str, Any]] = None  # Query and selection info

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "hop_id": self.hop_id,
            "chunk_id": self.chunk_id,
            "source_type": self.source_type,
        }
        if self.doc_id:
            d["doc_id"] = self.doc_id
        if self.edge:
            d["edge"] = self.edge
        return d


@dataclass
class TaskDiversity:
    """
    Metadata for tracking task characteristics.

    Used for analysis and stratification of results.
    """
    required_local_secrets: int                         # How many local secrets needed to solve
    hop_pattern: str                                    # e.g., "LLWW", "LWLW", "LLL"
    answer_corpus: Literal["local", "web"]              # Where the final answer is found
    local_constraints: int                              # Number of constraints from local
    web_constraints: int                                # Number of constraints from web
    total_hops: int                                     # Total number of hops

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required_local_secrets": self.required_local_secrets,
            "hop_pattern": self.hop_pattern,
            "answer_corpus": self.answer_corpus,
            "local_constraints": self.local_constraints,
            "web_constraints": self.web_constraints,
            "total_hops": self.total_hops,
        }


@dataclass
class RequiredSecret:
    """
    A secret that is required to solve the task.

    Can be:
    - The final answer itself
    - An intermediate value needed to retrieve the next hop
    """
    chunk_id: str
    question: str                                       # What question this secret answers
    answer: str                                         # The secret value
    secret_type: Optional[str] = None                   # e.g., "kpi_numeric", "entity", "date"
    is_intermediate: bool = False                       # True if used to find next hop, False if final

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "question": self.question,
            "answer": self.answer,
            "is_intermediate": self.is_intermediate,
        }
        if self.secret_type:
            d["secret_type"] = self.secret_type
        return d


@dataclass
class QualityResult:
    """
    Results of quality validation checks.
    """
    deterministic_pass: bool
    ablation_full_info: Optional[str] = None            # Model answer with full evidence
    ablation_no_info: Optional[str] = None              # Should be NOT_ANSWERABLE
    ablation_local_only: Optional[str] = None           # For mixed: should be NOT_ANSWERABLE
    ablation_web_only: Optional[str] = None             # For mixed: should be NOT_ANSWERABLE
    retrieval_rank: Optional[int] = None                # Rank of answer chunk in BM25(question)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "deterministic_pass": self.deterministic_pass,
        }
        if self.ablation_full_info is not None:
            d["ablation_full_info"] = self.ablation_full_info
        if self.ablation_no_info is not None:
            d["ablation_no_info"] = self.ablation_no_info
        if self.ablation_local_only is not None:
            d["ablation_local_only"] = self.ablation_local_only
        if self.ablation_web_only is not None:
            d["ablation_web_only"] = self.ablation_web_only
        if self.retrieval_rank is not None:
            d["retrieval_rank"] = self.retrieval_rank
        return d


@dataclass
class HCSPTask:
    """
    A complete HCSP task ready for output.
    """
    workspace_id: str
    mode: Literal["local_only", "web_only", "mixed"]
    question: str
    answer: str
    answer_type: str                                    # "extractive", "entity", "metric"

    # Tree structure
    hops: List[Hop]
    hcsp: HCSPTree

    # Evidence
    gold: Dict[str, Any]                                # Answer evidence + optional compute

    # Privacy
    required_secrets: List[RequiredSecret]
    unnecessary_secrets: List[Dict[str, Any]] = field(default_factory=list)
    prompt_leaks: List[str] = field(default_factory=list)  # Any banned terms that leaked

    # Diversity tracking
    diversity: Optional[TaskDiversity] = None

    # Quality
    quality: Optional[QualityResult] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "workspace_id": self.workspace_id,
            "mode": self.mode,
            "question": self.question,
            "answer": self.answer,
            "answer_type": self.answer_type,
            "tree": {
                "hops": [h.to_dict() for h in self.hops],
                "hcsp": self.hcsp.to_dict(),
            },
            "gold": self.gold,
            "privacy": {
                "required_secrets": [s.to_dict() for s in self.required_secrets],
                "unnecessary_secrets": self.unnecessary_secrets,
            },
        }
        if self.prompt_leaks:
            d["privacy"]["prompt_leaks"] = self.prompt_leaks
        if self.diversity:
            d["diversity"] = self.diversity.to_dict()
        if self.quality:
            d["quality"] = self.quality.to_dict()
        return d


# ============================================================================
# InfoSeek-style Dual-Agent Architecture
# ============================================================================


class Action(Enum):
    """Browser actions for incremental tree building (inspired by InfoSeek)."""
    INIT_ROOT = auto()           # Initialize question node from secret
    ADD_CONSTRAINTS = auto()     # Extract constraints from a vertex
    EXTEND_LOCAL = auto()        # Retrieve related local chunk, add as evidence
    EXTEND_WEB = auto()          # Retrieve related web chunk, add as evidence
    TERMINATE = auto()           # Tree meets complexity requirements


class LinkageType(Enum):
    """
    How local and web evidence are connected.

    entity_chain: Web reveals entity needed for local lookup
    computational: Answer requires combining local + web facts
    selection: Web provides filter criterion for local options
    definitional: Web defines term used in local
    creative: LLM-discovered linkage
    """
    ENTITY_CHAIN = "entity_chain"
    COMPUTATIONAL = "computational"
    SELECTION = "selection"
    DEFINITIONAL = "definitional"
    CREATIVE = "creative"


@dataclass
class Vertex:
    """
    A node in the research tree.

    Can be:
    - kind="question": The root, representing what we're solving for
    - kind="evidence": An evidence node providing facts (from local or web)

    Key insight from researcher feedback: The root is a QUESTION node, not an
    answer node. The answer value lives only in gold.answer, not in the tree.
    Evidence children provide facts that together identify the answer.
    """
    id: str                                             # "root", "L1", "W1", etc.
    kind: Literal["question", "evidence"]

    # For question nodes (root)
    variable: Optional[str] = None                      # What we're asking about
    answer_type: Optional[str] = None                   # Type of expected answer

    # For evidence nodes
    chunk_id: Optional[str] = None                      # Source chunk
    doc_id: Optional[str] = None                        # Source document
    source_type: Optional[Literal["local", "web"]] = None
    text: Optional[str] = None                          # Chunk text (cached)
    bridge_query: Optional[str] = None                  # Query used to find this

    # Common
    constraints: List[Constraint] = field(default_factory=list)
    children: List[str] = field(default_factory=list)  # IDs of child vertices
    entity: Optional[str] = None                        # Entity extracted (for blurring)
    blurs_entity: Optional[str] = None                  # Entity this web vertex helps identify

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
        }
        if self.kind == "question":
            if self.variable:
                d["variable"] = self.variable
            if self.answer_type:
                d["answer_type"] = self.answer_type
        elif self.kind == "evidence":
            if self.chunk_id:
                d["chunk_id"] = self.chunk_id
            if self.doc_id:
                d["doc_id"] = self.doc_id
            if self.source_type:
                d["source_type"] = self.source_type
            if self.bridge_query:
                d["bridge_query"] = self.bridge_query
            if self.blurs_entity:
                d["blurs_entity"] = self.blurs_entity
        if self.constraints:
            d["constraints"] = [c.to_dict() for c in self.constraints]
        if self.children:
            d["children"] = self.children
        if self.entity:
            d["entity"] = self.entity
        return d


@dataclass
class ResearchTree:
    """
    The research tree for InfoSeek-style task generation.

    Structure:
    - Root is a question node (what we're solving for)
    - Children are evidence nodes (from local/web corpus)
    - Each evidence node can have constraints

    The answer VALUE is NOT stored in the tree - it lives in gold.answer.
    The tree provides the structure for how evidence leads to the answer.
    """
    root_id: str = "root"
    vertices: Dict[str, Vertex] = field(default_factory=dict)
    linkage_type: Optional[LinkageType] = None

    def is_empty(self) -> bool:
        """Check if tree has been initialized."""
        return len(self.vertices) == 0

    def get_root(self) -> Optional[Vertex]:
        """Get the root vertex."""
        return self.vertices.get(self.root_id)

    def depth(self) -> int:
        """Get tree depth (max path length from root)."""
        if self.is_empty():
            return 0
        return len([v for v in self.vertices.values() if v.kind == "evidence"])

    def has_web_vertex(self) -> bool:
        """Check if tree has any web evidence."""
        return any(
            v.source_type == "web"
            for v in self.vertices.values()
            if v.kind == "evidence"
        )

    def has_local_vertex(self) -> bool:
        """Check if tree has any local evidence."""
        return any(
            v.source_type == "local"
            for v in self.vertices.values()
            if v.kind == "evidence"
        )

    def get_evidence_vertices(self) -> List[Vertex]:
        """Get all evidence vertices in order of addition."""
        return [
            v for v in self.vertices.values()
            if v.kind == "evidence"
        ]

    def get_local_vertices(self) -> List[Vertex]:
        """Get local evidence vertices."""
        return [v for v in self.get_evidence_vertices() if v.source_type == "local"]

    def get_web_vertices(self) -> List[Vertex]:
        """Get web evidence vertices."""
        return [v for v in self.get_evidence_vertices() if v.source_type == "web"]

    def total_constraints(self) -> int:
        """Count total constraints across all vertices."""
        return sum(len(v.constraints) for v in self.vertices.values())

    def last_corpus(self) -> Optional[str]:
        """Get the corpus of the last added evidence."""
        evidence = self.get_evidence_vertices()
        if not evidence:
            return None
        return evidence[-1].source_type

    def get_vertex_by_id(self, vertex_id: str) -> Optional[Vertex]:
        """Get vertex by ID."""
        return self.vertices.get(vertex_id)

    def all_chunk_ids(self) -> List[str]:
        """Get all chunk IDs in the tree."""
        return [
            v.chunk_id for v in self.vertices.values()
            if v.chunk_id is not None
        ]

    def hop_pattern(self) -> str:
        """Get hop pattern string like 'LLWW'."""
        pattern = ""
        for v in self.get_evidence_vertices():
            if v.source_type == "local":
                pattern += "L"
            elif v.source_type == "web":
                pattern += "W"
        return pattern

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "root": self.root_id,
            "vertices": {k: v.to_dict() for k, v in self.vertices.items()},
        }
        if self.linkage_type:
            d["linkage_type"] = self.linkage_type.value
        return d

    def all_constraints(self) -> List[Constraint]:
        """Get all constraints from all vertices."""
        constraints = []
        for v in self.vertices.values():
            constraints.extend(v.constraints)
        return constraints

    def get_local_constraints(self) -> List[Constraint]:
        """Get constraints from local evidence only."""
        constraints = []
        for v in self.get_local_vertices():
            constraints.extend(v.constraints)
        return constraints

    def get_web_constraints(self) -> List[Constraint]:
        """Get constraints from web evidence only."""
        constraints = []
        for v in self.get_web_vertices():
            constraints.extend(v.constraints)
        return constraints


@dataclass
class TreeContext:
    """Context passed to browser for action execution."""
    tree: ResearchTree
    mode: Literal["local_only", "web_only", "mixed"]
    target_constraints: int = 4
    target_evidence: int = 3
    gold_answer: str = ""                               # The answer value (for banning)
    seed_question: str = ""                             # Seed question from secret
    seed_secret_type: str = ""                          # Type of secret
