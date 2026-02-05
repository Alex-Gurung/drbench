"""
Entity extraction from local (enterprise) chunks.

Finds entities like projects, vendors, tools, certifications mentioned
in the local data and extracts their relationships.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .schema import LocalEntity, LocalEntityGraph, EntityMention
from .web_knowledge_base import get_web_knowledge


# Patterns to find entity mentions
ENTITY_PATTERNS = {
    # Technologies
    "technology": [
        r"\b(Kubernetes|K8s|Docker|Docker Swarm)\b",
        r"\b(Prometheus|Grafana|Datadog|New Relic|CloudWatch)\b",
        r"\b(AWS|Azure|Google Cloud|GCP)\b",
        r"\b(PostgreSQL|Postgres|MongoDB|Redis|MySQL)\b",
        r"\b(Kafka|RabbitMQ|Redis)\b",
        r"\b(Salesforce|SAP|Workday|Oracle)\b",
        r"\b(Epic|Cerner|Meditech)\b",
        r"\b(FoodLogiQ|TraceLink)\b",
        r"\b(Terraform|Ansible|Jenkins|GitLab)\b",
    ],
    # Certifications/Standards
    "certification": [
        r"\b(ISO\s*27001|ISO\s*9001|ISO\s*14001|ISO\s*22000)\b",
        r"\b(SOC\s*2|SOC\s*2\s*Type\s*II)\b",
        r"\b(LEED|LEED\s*(?:Platinum|Gold|Silver|Certified))\b",
        r"\b(CMMI|CMMI\s*Level\s*\d)\b",
    ],
    # Regulations
    "regulation": [
        r"\b(HIPAA|GDPR|SOX|PCI\s*DSS|PCI)\b",
        r"\b(FSMA|FSMA\s*204|Food\s*Traceability)\b",
    ],
    # Organic certifiers
    "certifier": [
        r"\b(Oregon\s*Tilth|CCOF|QAI)\b",
        r"\b(USDA\s*Organic)\b",
    ],
}

# Patterns for private enterprise data
PRIVATE_DATA_PATTERNS = {
    "budget": r"\$[\d,]+(?:K|M|B)?",
    "percentage": r"\d+(?:\.\d+)?%",
    "employee_count": r"(\d+)\s*(?:employees?|staff|team\s*members?)",
    "revenue": r"(?:revenue|sales)\s*(?:of\s*)?\$[\d,]+(?:K|M|B)?",
    "date": r"(?:Q[1-4]\s*)?20\d{2}",
    "project_name": r"Project\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?",
}


def extract_entities_from_chunks(
    chunks: List[Dict[str, Any]],
    task_id: str = "",
    company: str = "",
) -> LocalEntityGraph:
    """
    Extract entities from local chunks and build an entity graph.

    Args:
        chunks: List of chunk dicts with 'chunk_id' and 'text' keys
        task_id: Task identifier (e.g., "DR0001")
        company: Company name (e.g., "Lee's Market")

    Returns:
        LocalEntityGraph with extracted entities
    """
    graph = LocalEntityGraph(task_id=task_id, company=company)
    web_knowledge = get_web_knowledge()

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        text = chunk.get("text", "")

        if not text:
            continue

        # Find entity mentions
        mentions = find_entity_mentions(text, chunk_id)

        for mention in mentions:
            # Check if entity exists in web knowledge base
            if mention.entity_name in web_knowledge.entities:
                # Create or update local entity
                entity = _create_or_update_entity(
                    graph, mention, text, chunk_id
                )

                # Extract relationships and attributes
                _extract_relationships(entity, text, web_knowledge)
                _extract_private_attributes(entity, text)

    return graph


def find_entity_mentions(
    text: str,
    chunk_id: str,
) -> List[EntityMention]:
    """Find all entity mentions in a text."""
    mentions = []

    for entity_type, patterns in ENTITY_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity_name = _normalize_entity_name(match.group(0))

                # Get context window around mention
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]

                mentions.append(EntityMention(
                    entity_name=entity_name,
                    chunk_id=chunk_id,
                    context_snippet=context,
                    full_text=text,
                    start_pos=match.start(),
                    end_pos=match.end(),
                ))

    return mentions


def _normalize_entity_name(name: str) -> str:
    """Normalize entity name to canonical form."""
    name = name.strip()

    # Normalize variations
    normalizations = {
        "K8s": "Kubernetes",
        "k8s": "Kubernetes",
        "Postgres": "PostgreSQL",
        "GCP": "Google Cloud",
        "SOC 2 Type II": "SOC 2",
        "SOC2": "SOC 2",
        "ISO 27001": "ISO 27001",
        "ISO27001": "ISO 27001",
        "FSMA 204": "FSMA 204",
        "FSMA204": "FSMA 204",
    }

    # Handle ISO standards with spaces
    iso_match = re.match(r"ISO\s*(\d+)", name, re.IGNORECASE)
    if iso_match:
        return f"ISO {iso_match.group(1)}"

    return normalizations.get(name, name)


def _create_or_update_entity(
    graph: LocalEntityGraph,
    mention: EntityMention,
    text: str,
    chunk_id: str,
) -> LocalEntity:
    """Create a new entity or update existing one."""
    entity_name = mention.entity_name

    if entity_name in graph.entities:
        entity = graph.entities[entity_name]
        # Add additional evidence
        if mention.context_snippet not in entity.evidence_text:
            entity.evidence_text += f"\n---\n{mention.context_snippet}"
    else:
        # Determine entity type
        entity_type = _determine_entity_type(entity_name)

        entity = LocalEntity(
            name=entity_name,
            entity_type=entity_type,
            chunk_id=chunk_id,
            evidence_text=mention.context_snippet,
        )
        graph.add_entity(entity)

    return entity


def _determine_entity_type(entity_name: str) -> str:
    """Determine the type of an entity based on its name."""
    for entity_type, patterns in ENTITY_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, entity_name, re.IGNORECASE):
                return entity_type

    return "unknown"


def _extract_relationships(
    entity: LocalEntity,
    text: str,
    web_knowledge,
) -> None:
    """Extract relationships between entities from text."""

    # Check for ecosystem relationships
    web_entity = web_knowledge.entities.get(entity.name)
    if web_entity and web_entity.ecosystem:
        for category, tools in web_entity.ecosystem.items():
            for tool in tools:
                if tool.lower() in text.lower():
                    entity.relationships[f"uses_{category}"] = tool

    # Check for "uses", "implements", "integrated with" patterns
    use_patterns = [
        rf"{entity.name}.*?(?:uses?|integrated?\s*with|implements?)\s+(\w+)",
        rf"(\w+).*?(?:uses?|integrated?\s*with|implements?)\s+{entity.name}",
    ]

    for pattern in use_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            related = match.group(1)
            if related in web_knowledge.entities:
                entity.relationships["uses"] = related


def _extract_private_attributes(entity: LocalEntity, text: str) -> None:
    """Extract private attributes like budgets, dates, metrics from context."""

    # Look for budget near entity mention
    budget_match = re.search(
        rf"{entity.name}.*?(\$[\d,]+(?:K|M|B)?)",
        text,
        re.IGNORECASE,
    )
    if budget_match:
        entity.attributes["budget"] = budget_match.group(1)

    # Look for date/quarter
    date_match = re.search(
        rf"{entity.name}.*?(Q[1-4]\s*20\d{{2}}|20\d{{2}})",
        text,
        re.IGNORECASE,
    )
    if date_match:
        entity.attributes["date"] = date_match.group(1)

    # Look for percentage metrics
    pct_match = re.search(
        rf"(\d+(?:\.\d+)?%).*?{entity.name}|{entity.name}.*?(\d+(?:\.\d+)?%)",
        text,
        re.IGNORECASE,
    )
    if pct_match:
        pct = pct_match.group(1) or pct_match.group(2)
        entity.attributes["metric"] = pct


def extract_projects(text: str, chunk_id: str) -> List[LocalEntity]:
    """Extract project mentions from text."""
    projects = []

    # Pattern: "Project X" or specific project names
    project_pattern = r"Project\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"

    for match in re.finditer(project_pattern, text):
        project_name = f"Project {match.group(1)}"

        # Get context
        start = max(0, match.start() - 150)
        end = min(len(text), match.end() + 150)
        context = text[start:end]

        entity = LocalEntity(
            name=project_name,
            entity_type="project",
            chunk_id=chunk_id,
            evidence_text=context,
            is_private=True,
        )

        # Extract attributes from context
        _extract_private_attributes(entity, context)

        # Look for technology mentions in context
        tech_mentions = find_entity_mentions(context, chunk_id)
        for mention in tech_mentions:
            if mention.entity_name != project_name:
                entity.relationships["uses"] = mention.entity_name

        projects.append(entity)

    return projects


def extract_departments(text: str, chunk_id: str) -> List[LocalEntity]:
    """Extract department mentions with metrics from text."""
    departments = []

    # Common department names
    dept_names = [
        "Produce", "Grocery", "Bakery", "Deli", "Prepared Foods",
        "Meat", "Seafood", "Dairy", "Frozen", "IT", "HR", "Finance",
        "Marketing", "Sales", "Engineering", "Operations",
    ]

    for dept in dept_names:
        # Look for department with metrics
        pattern = rf"({dept})(?:\s+department)?.*?(\d+(?:\.\d+)?%)"
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            dept_name = match.group(1)
            metric = match.group(2)

            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end]

            entity = LocalEntity(
                name=f"{dept_name} department",
                entity_type="department",
                chunk_id=chunk_id,
                evidence_text=context,
                is_private=True,
                attributes={"retention": metric} if "retention" in text.lower() else {"metric": metric},
            )

            # Check for certifications
            cert_mentions = find_entity_mentions(context, chunk_id)
            for mention in cert_mentions:
                if mention.entity_name.startswith("ISO") or mention.entity_name == "LEED":
                    entity.relationships["certified_by"] = mention.entity_name

            departments.append(entity)

    return departments


def find_tech_ecosystem_connections(
    local_graph: LocalEntityGraph,
) -> List[Tuple[LocalEntity, str, str]]:
    """
    Find connections between local entities via web ecosystem knowledge.

    Returns list of (local_entity, web_entity, ecosystem_tool) tuples.
    """
    web_knowledge = get_web_knowledge()
    connections = []

    for entity in local_graph.entities.values():
        # Check if this entity has ecosystem info
        web_entity = web_knowledge.entities.get(entity.name)
        if not web_entity or not web_entity.ecosystem:
            continue

        # Look for ecosystem tools that also appear in local data
        for category, tools in web_entity.ecosystem.items():
            for tool in tools:
                if tool in local_graph.entities:
                    connections.append((entity, entity.name, tool))

    return connections
