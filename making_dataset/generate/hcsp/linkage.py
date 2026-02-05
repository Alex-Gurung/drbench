"""
Linkage Type Detection - Identifies how local and web evidence are connected.

Linkage types (from the plan):
- ENTITY_CHAIN: Web reveals entity needed for local lookup
- COMPUTATIONAL: Answer requires combining local + web facts
- SELECTION: Web provides filter criterion for local options
- DEFINITIONAL: Web defines term used in local
- CREATIVE: LLM-discovered linkage

The linkage type affects:
1. How the question is phrased
2. What the solving chain looks like
3. Validation criteria
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .schema import Constraint, LinkageType, ResearchTree, Vertex


def detect_linkage_type(
    tree: ResearchTree,
    gold_answer: str,
) -> Optional[LinkageType]:
    """
    Detect the linkage type from the tree structure and constraints.

    Returns None if the tree doesn't have both local and web evidence.
    """
    if not tree.has_local_vertex() or not tree.has_web_vertex():
        return None

    local_vertices = tree.get_local_vertices()
    web_vertices = tree.get_web_vertices()
    local_constraints = tree.get_local_constraints()
    web_constraints = tree.get_web_constraints()

    # Check for COMPUTATIONAL: both have numeric/metric constraints
    local_has_numeric = _has_numeric_content(local_constraints)
    web_has_numeric = _has_numeric_content(web_constraints)

    if local_has_numeric and web_has_numeric:
        return LinkageType.COMPUTATIONAL

    # Check for SELECTION: web has ranking/criteria, local has multiple options
    web_has_ranking = _has_ranking_content(web_constraints)
    if web_has_ranking:
        return LinkageType.SELECTION

    # Check for DEFINITIONAL: web defines a term/certification
    web_has_definition = _has_definition_content(web_constraints)
    if web_has_definition:
        return LinkageType.DEFINITIONAL

    # Check for ENTITY_CHAIN: web has entity reference that appears in local
    web_entities = _extract_entities(web_constraints)
    local_text = " ".join(v.text or "" for v in local_vertices)

    for entity in web_entities:
        if entity.lower() in local_text.lower():
            return LinkageType.ENTITY_CHAIN

    # Default to CREATIVE if we have both corpora but can't classify
    return LinkageType.CREATIVE


def _has_numeric_content(constraints: List[Constraint]) -> bool:
    """Check if constraints contain numeric content."""
    numeric_patterns = [
        r"\d+%",           # Percentages
        r"\$[\d,]+",       # Currency
        r"\d+\s*(?:million|billion|thousand)",  # Large numbers
        r"\d+\.\d+",       # Decimals
        r"\d+\s+(?:units?|items?|products?)",   # Counts
    ]

    for c in constraints:
        for pattern in numeric_patterns:
            if re.search(pattern, c.text, re.IGNORECASE):
                return True
    return False


def _has_ranking_content(constraints: List[Constraint]) -> bool:
    """Check if constraints contain ranking/criteria content."""
    ranking_terms = [
        "ranked", "ranking", "top", "best", "winner", "award",
        "#1", "first place", "highest", "leading", "index",
        "certified", "accredited", "rated",
    ]

    for c in constraints:
        text_lower = c.text.lower()
        for term in ranking_terms:
            if term in text_lower:
                return True
    return False


def _has_definition_content(constraints: List[Constraint]) -> bool:
    """Check if constraints contain definitional content."""
    definition_patterns = [
        r"requires?\s+(?:at least|minimum|over|above)",
        r"defined as",
        r"certification\s+(?:requires?|means?)",
        r"threshold\s+(?:of|is|for)",
        r"standard\s+(?:of|is|for)",
        r"criteria\s+(?:include|is|are)",
    ]

    for c in constraints:
        for pattern in definition_patterns:
            if re.search(pattern, c.text, re.IGNORECASE):
                return True
    return False


def _extract_entities(constraints: List[Constraint]) -> List[str]:
    """Extract named entities from constraints."""
    entities = set()

    for c in constraints:
        # Extract capitalized phrases
        caps = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", c.text)
        entities.update(caps)

        # Extract quoted terms
        quoted = re.findall(r'"([^"]+)"', c.text)
        entities.update(quoted)

    # Filter out common words that happen to be capitalized
    common = {"The", "A", "An", "In", "On", "At", "For", "With", "To", "From"}
    entities = {e for e in entities if e not in common and len(e) > 2}

    return list(entities)


def get_linkage_description(linkage_type: LinkageType) -> str:
    """Get human-readable description of linkage type."""
    descriptions = {
        LinkageType.ENTITY_CHAIN: "Web reveals entity needed for local lookup",
        LinkageType.COMPUTATIONAL: "Answer requires combining local + web facts",
        LinkageType.SELECTION: "Web provides filter criterion for local options",
        LinkageType.DEFINITIONAL: "Web defines term used in local",
        LinkageType.CREATIVE: "LLM-discovered cross-corpus linkage",
    }
    return descriptions.get(linkage_type, "Unknown linkage type")


def suggest_question_style(linkage_type: LinkageType) -> Dict[str, Any]:
    """
    Suggest question phrasing style based on linkage type.

    Returns dict with guidance for question synthesis.
    """
    if linkage_type == LinkageType.ENTITY_CHAIN:
        return {
            "pattern": "What is [metric] for [entity described by web constraints]?",
            "focus": "Use web constraints to describe the entity indirectly",
            "example": "What was the Q3 retention rate for the inventory system that won the 2024 Award?",
        }
    elif linkage_type == LinkageType.COMPUTATIONAL:
        return {
            "pattern": "How many [units] above/below [web baseline] was [local value]?",
            "focus": "Frame as a comparison requiring both values",
            "example": "How many percentage points above industry average was Lee's Market's retention?",
        }
    elif linkage_type == LinkageType.SELECTION:
        return {
            "pattern": "What was [metric] for [local item matching web criterion]?",
            "focus": "Use web ranking/criterion to select from local options",
            "example": "What was the retention for the product line ranked #1 in the Sustainability Index?",
        }
    elif linkage_type == LinkageType.DEFINITIONAL:
        return {
            "pattern": "What [threshold/requirement] did [local achievement] require?",
            "focus": "Ask about the definition/threshold from web",
            "example": "What minimum retention threshold did the Gold certification require?",
        }
    else:
        return {
            "pattern": "Open-ended research question requiring both sources",
            "focus": "Natural question that cannot be answered from either source alone",
            "example": "What factors contributed to [outcome]?",
        }


def validate_linkage_necessity(
    tree: ResearchTree,
    local_only_answerable: bool,
    web_only_answerable: bool,
) -> Tuple[bool, str]:
    """
    Validate that the linkage is necessary (web is required).

    For mixed mode, the question should NOT be answerable with only
    local evidence OR only web evidence.

    Returns:
        Tuple of (is_valid, reason)
    """
    if local_only_answerable and web_only_answerable:
        return False, "answerable from either corpus alone"
    if local_only_answerable:
        return False, "web not necessary (local-only works)"
    if web_only_answerable:
        return False, "local not necessary (web-only works)"
    return True, "both corpora required"


def get_solving_chain(tree: ResearchTree, linkage_type: LinkageType) -> List[str]:
    """
    Generate the expected solving chain for this tree.

    Returns list of steps an agent would take to solve.
    """
    evidence = tree.get_evidence_vertices()
    steps = ["Read question"]

    if linkage_type == LinkageType.ENTITY_CHAIN:
        # Web first to identify entity, then local for answer
        for v in evidence:
            if v.source_type == "web":
                steps.append(f"Search web → identify entity from constraints")
            else:
                steps.append(f"Search local with entity → find answer")

    elif linkage_type == LinkageType.COMPUTATIONAL:
        # Need both values, order depends on tree structure
        for v in evidence:
            corpus = "web" if v.source_type == "web" else "local"
            steps.append(f"Search {corpus} → get {'baseline' if corpus == 'web' else 'company'} value")
        steps.append("Compute difference/ratio")

    elif linkage_type == LinkageType.SELECTION:
        # Web first for criterion, then local for selected value
        steps.append("Search web → identify selection criterion")
        steps.append("Search local → find matching item's value")

    elif linkage_type == LinkageType.DEFINITIONAL:
        # Could be either order
        steps.append("Search local → find achievement")
        steps.append("Search web → find definition/threshold")

    else:
        # Generic chain
        for v in evidence:
            corpus = "web" if v.source_type == "web" else "local"
            steps.append(f"Search {corpus} → gather evidence")

    steps.append("Extract answer")
    return steps
