"""
Data structures for 4-hop privacy leakage question generation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


@dataclass
class ChainLink:
    """A single hop in a 4-hop chain."""

    source: Literal["local", "web"]
    entity: str
    entity_type: str  # "project", "technology", "company", "regulation", "department"
    fact_discovered: str  # What this hop reveals
    evidence_chunk_id: Optional[str] = None  # For local hops
    evidence_text: Optional[str] = None  # Snippet showing the fact


@dataclass
class FourHopChain:
    """A complete 4-hop reasoning chain: Local → Web → Local → Web."""

    hop1_local: ChainLink   # Private entity/fact (e.g., "Project Phoenix uses Kubernetes")
    hop2_web: ChainLink     # Web knowledge about entity (e.g., "Kubernetes ecosystem includes Prometheus")
    hop3_local: ChainLink   # Connected entity in local (e.g., "We use Prometheus for monitoring")
    hop4_web: ChainLink     # Final web fact - the answer (e.g., "Prometheus developed by SoundCloud")

    chain_type: str  # "tech_ecosystem", "vendor_chain", "compliance_chain", "certification_chain"

    # Generated question and answer
    question: str = ""
    answer: str = ""

    # Privacy analysis
    private_data_exposed: List[str] = field(default_factory=list)

    def validate(self) -> List[str]:
        """Check chain validity, return list of issues."""
        issues = []

        if self.hop1_local.source != "local":
            issues.append("hop1 must be local")
        if self.hop2_web.source != "web":
            issues.append("hop2 must be web")
        if self.hop3_local.source != "local":
            issues.append("hop3 must be local")
        if self.hop4_web.source != "web":
            issues.append("hop4 must be web")

        if not self.question:
            issues.append("question not generated")
        if not self.answer:
            issues.append("answer not set")

        return issues


@dataclass
class LocalEntity:
    """An entity extracted from local (enterprise) data."""

    name: str
    entity_type: str  # "project", "department", "vendor", "tool", "certification"
    chunk_id: str

    # Attributes about this entity
    attributes: Dict[str, str] = field(default_factory=dict)
    # e.g., {"launch_date": "Q2 2024", "budget": "$450K", "technology": "Kubernetes"}

    # Relationships to other entities
    relationships: Dict[str, str] = field(default_factory=dict)
    # e.g., {"uses_monitoring": "Prometheus", "certified_by": "Oregon Tilth"}

    # The text snippet where this entity was found
    evidence_text: str = ""

    # Is this considered private/sensitive?
    is_private: bool = True


@dataclass
class WebEntity:
    """An entity in the web knowledge base with known facts."""

    name: str
    entity_type: str  # "technology", "company", "regulation", "organization"

    # Known facts about this entity
    facts: Dict[str, str] = field(default_factory=dict)
    # e.g., {"developed_by": "Google", "year": "2014", "headquarters": "Seattle"}

    # Ecosystem relationships
    ecosystem: Dict[str, List[str]] = field(default_factory=dict)
    # e.g., {"monitoring": ["Prometheus", "Grafana"], "service_mesh": ["Istio"]}

    # Alternative names
    aliases: List[str] = field(default_factory=list)


@dataclass
class EntityMention:
    """A mention of an entity in a local chunk."""

    entity_name: str
    chunk_id: str
    context_snippet: str  # Text around the mention
    full_text: str
    start_pos: int
    end_pos: int


@dataclass
class GeneratedQuestion:
    """A generated 4-hop question with metadata."""

    # The question and answer
    question: str
    answer: str
    answer_type: str  # "year", "city", "company", "number"

    # Chain info
    chain: FourHopChain
    chain_type: str

    # Source task/company
    task_id: str
    company: str

    # Evidence
    local_chunk_ids: List[str]

    # Privacy analysis
    private_data_in_reasoning: List[str]
    safe_query_example: str  # Query that doesn't leak
    leaky_query_example: str  # Query that leaks private info

    # Validation results
    validated: bool = False
    validation_issues: List[str] = field(default_factory=list)


@dataclass
class LocalEntityGraph:
    """Graph of entities extracted from local data."""

    entities: Dict[str, LocalEntity] = field(default_factory=dict)
    task_id: str = ""
    company: str = ""

    def add_entity(self, entity: LocalEntity) -> None:
        self.entities[entity.name] = entity

    def get_by_type(self, entity_type: str) -> List[LocalEntity]:
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def get_related_to(self, web_entity: str) -> List[LocalEntity]:
        """Find local entities that reference a web entity."""
        results = []
        for entity in self.entities.values():
            # Check attributes
            for attr_value in entity.attributes.values():
                if web_entity.lower() in attr_value.lower():
                    results.append(entity)
                    break
            # Check relationships
            for rel_value in entity.relationships.values():
                if web_entity.lower() in rel_value.lower():
                    results.append(entity)
                    break
        return results


@dataclass
class WebKnowledgeGraph:
    """Graph of web entities and their relationships."""

    entities: Dict[str, WebEntity] = field(default_factory=dict)

    def add_entity(self, entity: WebEntity) -> None:
        self.entities[entity.name] = entity
        # Also index by aliases
        for alias in entity.aliases:
            self.entities[alias] = entity

    def get_ecosystem_tools(self, entity_name: str, category: str) -> List[str]:
        """Get ecosystem tools for an entity in a category."""
        entity = self.entities.get(entity_name)
        if entity and category in entity.ecosystem:
            return entity.ecosystem[category]
        return []

    def get_fact(self, entity_name: str, fact_type: str) -> Optional[str]:
        """Get a specific fact about an entity."""
        entity = self.entities.get(entity_name)
        if entity:
            return entity.facts.get(fact_type)
        return None
