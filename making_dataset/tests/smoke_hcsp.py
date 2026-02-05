#!/usr/bin/env python3
"""
Smoke test for HCSP dataset generation.

Tests the schema, constraint extraction, and question synthesis modules
without requiring vLLM or external indexes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))


def test_schema():
    """Test schema dataclasses."""
    from making_dataset.generate.hcsp.schema import (
        Constraint,
        EvidencePointer,
        Hop,
        HCSPNode,
        HCSPTree,
        HCSPTask,
        RequiredSecret,
        TaskDiversity,
    )

    # Test EvidencePointer
    evidence = EvidencePointer(
        chunk_id="local/DR0001/test.md#0001",
        char_start=10,
        char_end=50,
        text="expanded to 12 locations in 2024",
    )
    assert evidence.to_dict()["chunk_id"] == "local/DR0001/test.md#0001"
    print("  EvidencePointer: OK")

    # Test Constraint
    constraint = Constraint(
        text="expanded to 12 locations in 2024",
        evidence=evidence,
        constraint_type="attribute",
        corpus="local",
    )
    d = constraint.to_dict()
    assert d["corpus"] == "local"
    assert d["constraint_type"] == "attribute"
    print("  Constraint: OK")

    # Test HCSPNode (constraint)
    node = HCSPNode(
        id="S1",
        kind="constraint",
        constraint=constraint,
    )
    assert node.to_dict()["kind"] == "constraint"
    print("  HCSPNode (constraint): OK")

    # Test HCSPNode (answer)
    answer_evidence = EvidencePointer(
        chunk_id="local/DR0001/test.md#0001",
        char_start=100,
        char_end=103,
        text="85%",
    )
    answer_node = HCSPNode(
        id="A0",
        kind="answer",
        op="INTERSECT",
        inputs=["S1", "S2"],
        answer_evidence=answer_evidence,
    )
    assert answer_node.to_dict()["op"] == "INTERSECT"
    print("  HCSPNode (answer): OK")

    # Test HCSPTree
    tree = HCSPTree(
        root_id="A0",
        nodes={
            "S1": node,
            "A0": answer_node,
        },
    )
    assert tree.root_id == "A0"
    assert len(tree.get_constraints()) == 1
    print("  HCSPTree: OK")

    # Test Hop
    hop = Hop(
        hop_id=1,
        chunk_id="local/DR0001/test.md#0001",
        doc_id="local/DR0001/test.md",
        source_type="local",
        edge={"query": "grocery chain expansion", "corpus": "local"},
    )
    assert hop.to_dict()["source_type"] == "local"
    print("  Hop: OK")

    # Test TaskDiversity
    diversity = TaskDiversity(
        required_local_secrets=2,
        hop_pattern="LLWW",
        answer_corpus="local",
        local_constraints=3,
        web_constraints=2,
        total_hops=4,
    )
    assert diversity.to_dict()["hop_pattern"] == "LLWW"
    print("  TaskDiversity: OK")

    # Test RequiredSecret
    secret = RequiredSecret(
        chunk_id="local/DR0001/test.md#0001",
        question="What is the customer retention rate?",
        answer="85%",
        secret_type="kpi_numeric",
        is_intermediate=False,
    )
    assert secret.to_dict()["is_intermediate"] == False
    print("  RequiredSecret: OK")

    print("Schema tests passed!")


def test_synthesize_utils():
    """Test synthesis utilities (no LLM required)."""
    from making_dataset.generate.hcsp.synthesize import (
        build_banned_terms,
        _check_question_for_banned,
        _check_question_for_corpus_hints,
    )

    # Test build_banned_terms
    banned = build_banned_terms(
        answer="85%",
        intermediate_secrets=["FreshTrack", "Q3 Report"],
        doc_ids=["local/DR0001/files/report.md"],
        filenames=["report.md"],
    )
    assert "85%" in banned
    assert "FreshTrack" in banned
    assert "report.md" in banned or "report" in banned
    print("  build_banned_terms: OK")

    # Test banned term checking
    violations = _check_question_for_banned(
        "What is the 85% rate for FreshTrack?",
        ["85%", "FreshTrack"],
    )
    assert "85%" in violations
    assert "FreshTrack" in violations
    print("  _check_question_for_banned: OK")

    # Test corpus hint checking
    violations = _check_question_for_corpus_hints(
        "What does the local document say about web evidence?"
    )
    assert "local" in violations
    assert "document" in violations
    assert "web" in violations
    print("  _check_question_for_corpus_hints: OK")

    # Good question should have no violations
    good_violations = _check_question_for_corpus_hints(
        "What was the customer loyalty metric for a grocery chain that expanded in 2024?"
    )
    assert len(good_violations) == 0
    print("  Good question check: OK")

    print("Synthesis utility tests passed!")


def test_constraints_utils():
    """Test constraint utilities (no LLM required)."""
    from making_dataset.generate.hcsp.constraints import (
        _find_quote_span,
        _parse_constraint_output,
        _contains_banned,
    )

    # Test quote span finding
    text = "Lee's Market expanded to 12 locations in 2024 across the Pacific Northwest."
    span = _find_quote_span(text, "12 locations in 2024")
    assert span is not None
    assert text[span[0]:span[1]] == "12 locations in 2024"
    print("  _find_quote_span (exact): OK")

    # Case insensitive
    span = _find_quote_span(text, "PACIFIC NORTHWEST")
    assert span is not None
    print("  _find_quote_span (case insensitive): OK")

    # Test output parsing
    output = """CONSTRAINT: expanded to 12 locations in 2024
QUOTE: expanded to 12 locations in 2024
TYPE: attribute

CONSTRAINT: operates in the Pacific Northwest
QUOTE: Pacific Northwest
TYPE: attribute"""
    parsed = _parse_constraint_output(output)
    assert len(parsed) == 2
    assert parsed[0]["constraint"] == "expanded to 12 locations in 2024"
    assert parsed[1]["type"] == "attribute"
    print("  _parse_constraint_output: OK")

    # Test banned checking
    assert _contains_banned("Lee's Market expanded", ["Lee's Market"]) == True
    assert _contains_banned("A grocery chain expanded", ["Lee's Market"]) == False
    print("  _contains_banned: OK")

    print("Constraint utility tests passed!")


def test_tree_builder_utils():
    """Test tree builder utilities (no LLM required)."""
    from making_dataset.generate.hcsp.tree_builder import _extract_bridge_terms_simple

    text = """Lee's Market Q3 2024 Financial Report

The company expanded to 12 locations across the Pacific Northwest region.
Customer retention rate reached 85% this quarter.
The FreshTrack inventory system was successfully deployed."""

    terms = _extract_bridge_terms_simple(text)
    assert len(terms) > 0
    # Should extract capitalized entities
    assert any("Market" in t or "Lee" in t for t in terms)
    print("  _extract_bridge_terms_simple: OK")

    print("Tree builder utility tests passed!")


def main():
    print("Running HCSP smoke tests...\n")

    print("1. Testing schema...")
    test_schema()
    print()

    print("2. Testing synthesis utilities...")
    test_synthesize_utils()
    print()

    print("3. Testing constraint utilities...")
    test_constraints_utils()
    print()

    print("4. Testing tree builder utilities...")
    test_tree_builder_utils()
    print()

    print("=" * 50)
    print("All smoke tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
