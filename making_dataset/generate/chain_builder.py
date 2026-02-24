#!/usr/bin/env python3
"""
Flexible Multi-Hop Chain Builder.

Builds question chains with arbitrary hop sequences (e.g., LW, LWLW, WWL, LLW).
Each hop is either Local (L) enterprise document or Web (W) public document.

Usage:
    python -m making_dataset.generate.chain_builder \
        --pattern LW --n 5 --vllm-url http://127.0.0.1:8000

    python -m making_dataset.generate.chain_builder \
        --patterns LW LWLW LLW --n 10 --output chains.jsonl --verify
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from making_dataset.index.unified_searcher import UnifiedSearcher, UnifiedHit
from making_dataset.utils.vllm_client import VLLMClient

# Default paths
DEFAULT_CHUNKS_LOCAL = ROOT_DIR / "making_dataset" / "outputs" / "chunks_local.jsonl"
DEFAULT_SECRET_INVENTORY = ROOT_DIR / "making_dataset" / "outputs" / "secret_inventory.jsonl"
DEFAULT_WEB_BM25_INDEX = "/home/toolkit/BrowseComp-Plus/indexes/bm25"

HopType = Literal["L", "W"]


@dataclass
class Secret:
    """A secret extracted from enterprise documents."""
    chunk_id: str
    doc_id: str
    question: str
    answer: str
    secret_type: str
    text: str  # Full chunk text
    company: str
    task_id: str


@dataclass
class Hop:
    """A single hop in a chain."""
    hop_type: HopType
    doc_id: str
    text: str
    fact: str  # The key fact from this hop
    company: str | None = None
    task_id: str | None = None  # Task ID for local hops (e.g., "DR0001")
    secret_question: str | None = None
    secret_answer: str | None = None


@dataclass
class Bridge:
    """Connection between two hops via entity-relation-answer triple."""
    link: str  # Entity/concept connecting docs
    relation: str  # Predicate (e.g., "founded in", "acquired by")
    from_idx: int
    to_idx: int
    bridge_type: str  # Keep for tracking, unified prompt handles all types
    question: str = ""  # Question requiring both docs
    answer: str = ""  # New info from target doc (1-5 words)
    reasoning: str = ""  # Model's step-by-step thinking


@dataclass
class Chain:
    """A complete multi-hop chain."""
    chain_id: str
    pattern: str
    hops: list[Hop]
    bridges: list[Bridge]
    question: str
    answer: str
    verification: dict | None = None


@dataclass
class ChainState:
    """State passed between rounds in chain building."""
    doc: str  # Document text
    doc_id: str
    question: str  # Question answerable from this doc
    answer: str  # Answer (1-3 words, specific entity/number)
    hop_type: HopType
    company: str | None = None
    task_id: str | None = None


# =============================================================================
# LLM Prompts
# =============================================================================

BRIDGE_PROMPT = """Create a bridge from the current question to a new document.

You are building a multi-hop question iteratively. Given the current question state and a new document, compose the next question that wraps the current one.

CURRENT STATE:
- Question: {current_question}
- Answer: "{current_answer}"

NEW DOCUMENT:
{new_doc}

YOUR TASK:
1. Describe "{current_answer}" using context from the current question (WITHOUT naming it directly)
2. Find a new fact about "{current_answer}" in the new document
3. Create a composed question that wraps the current question

EXAMPLE 1 (entity):
Current Q: "What HR platform does Lee's Market use for onboarding?"
Current A: "Workday"
New doc: "Workday, Inc. was founded in 2005 by Dave Duffield and Aneel Bhusri..."

ANSWER_SLOT: the HR platform Lee's Market uses for onboarding
QUESTION: When was the HR platform Lee's Market uses for onboarding founded?
ANSWER: 2005

EXAMPLE 2 (number):
Current Q: "What is Lee's Market's employee retention rate?"
Current A: "82%"
New doc: "The retail industry average employee retention rate is 65%..."

ANSWER_SLOT: Lee's Market's employee retention rate
QUESTION: How many percentage points above the retail industry average is Lee's Market's employee retention rate?
ANSWER: 17 percentage points

EXAMPLE 3 (platform):
Current Q: "What container platform does MediConn use for processing?"
Current A: "Kubernetes"
New doc: "Kubernetes was originally developed by Google engineers in 2014..."

ANSWER_SLOT: the container platform MediConn uses for processing
QUESTION: Who originally developed the container platform MediConn uses for processing?
ANSWER: Google

REQUIREMENTS:
- ANSWER_SLOT describes "{current_answer}" using the current question context, WITHOUT naming it
- QUESTION wraps the current question (answering requires knowing what {current_answer} is first)
- ANSWER is 1-3 words extracted from the new document
- Do NOT include "{current_answer}" literally in QUESTION
- The new document must actually contain information about "{current_answer}"

BAD PATTERNS - REJECT:
- Question includes "{current_answer}" literally → NO_BRIDGE
- Answer not found in new document → NO_BRIDGE
- No meaningful connection between current answer and new doc → NO_BRIDGE
- Compound questions ("... and what...") → NO_BRIDGE
- Number coincidence only (35% → 35th district) → NO_BRIDGE
- Generic answers (improved, increased, data) → NO_BRIDGE

If no valid bridge exists: NO_BRIDGE: <reason>

OUTPUT:
ANSWER_SLOT: <description of "{current_answer}" using question context>
QUESTION: <composed question that wraps current question>
ANSWER: <1-3 words from new document>
"""

COMPOSE_QUESTION_PROMPT = """You have a {n}-hop reasoning chain. Generate a single question that requires traversing all hops to answer.

CHAIN:
{chain_description}

FINAL ANSWER: {final_answer}

Requirements:
1. Question must reference information from the FIRST hop
2. Answer comes from the LAST hop
3. Answering requires ALL intermediate hops
4. Do NOT mention company names directly - use indirect references ("the company", "the retailer", etc.)
5. Question should be natural and clear

Output ONLY:
QUESTION: <your question>
"""

QUERY_GENERATION_PROMPT = """Generate ONE detailed search query to find documents about the topic in this question.

CURRENT QUESTION: {current_question}
CURRENT ANSWER: "{current_answer}"

Generate a single, detailed search query that would find documents with additional facts about the topic.
Focus on the SUBJECT of the question, not the raw answer value.

EXAMPLES:

Question: "What percentage of new hires achieved full productivity within 90 days?"
Answer: "80%"
QUERY: new hire productivity benchmarks time to productivity onboarding research

Question: "What HR platform does Lee's Market use for onboarding?"
Answer: "Workday"
QUERY: Workday company history founded features enterprise HR

Question: "What is Lee's Market's employee retention rate?"
Answer: "82%"
QUERY: employee retention rate benchmarks retail industry turnover statistics

Question: "What container orchestration platform does MediConn use?"
Answer: "Kubernetes"
QUERY: Kubernetes company history development Google container orchestration

Question: "What is the average delivery time for Lee's Market orders?"
Answer: "2.3 days"
QUERY: retail delivery time benchmarks ecommerce shipping speed statistics

RULES:
- If answer is a named entity (Workday, Kubernetes, SAP), include it in the query
- If answer is a number/metric, search for benchmarks and industry statistics on the TOPIC
- DO NOT include fictional company names (Lee's Market, MediConn, etc.)
- Output just the query, nothing else

QUERY:"""

CAN_ANSWER_PROMPT = """Given ONLY the context below, can you answer this question?

CONTEXT:
{context}

QUESTION: {question}

EXPECTED ANSWER: {expected_answer}

If the context contains enough information to arrive at the expected answer (or something close to it), output:
ANSWERABLE: YES
ANSWER: <your derived answer>

If the context is missing key information needed to answer, output:
ANSWERABLE: NO
REASON: <what specific information is missing>
"""

EXTEND_QUESTION_PROMPT = """You are building a multi-hop question incrementally. Given the current question and a new document, extend the question to incorporate the new information.

COMPANY: {company_name}

CURRENT QUESTION: {current_question}
CURRENT ANSWER: {current_answer}

NEW DOCUMENT (Doc {hop_num}):
{new_doc}

BRIDGE FOUND:
- Link: {bridge_link}
- Relation: {bridge_relation}
- New answer from this doc: {bridge_answer}

Your task: Extend the current question so that:
1. The question now requires BOTH the previous context AND this new document to answer
2. The answer to the extended question is: {bridge_answer}
3. Keep reference to the original company/context

Example:
  Current: "What is Lee's Market's CAC reduction percentage?"
  Current answer: 20%
  Bridge: 20% is half of Exon's 40% energy reduction
  Extended: "What cost metric had a 2024 reduction that is twice Lee's Market's CAC decrease?"
  New answer: energy costs

OUTPUT:
EXTENDED_QUESTION: <the new extended question>
"""

BRIDGE_SCORING_PROMPT = """Evaluate these bridge candidates for a multi-hop QA chain.

SOURCE DOCUMENT (Doc A):
{source_doc}

SOURCE Q&A: "{source_q}" → "{source_a}"

CANDIDATE BRIDGES:
{candidates}

REQUIREMENTS FOR A GOOD BRIDGE:
1. REQUIRES BOTH DOCS: Cannot answer with only source or only target
2. SPECIFIC ANSWER: 1-3 words, named entity or number (not generic like "improved")
3. INDIRECT REFERENCE: Question describes target answer through source info, not by naming it directly
4. CORRECT DIRECTION: Must resolve source info first to get target answer
5. NO ANSWER LEAKAGE: Can't compute answer from question alone

GOLD EXAMPLE:
Source: "Lee's Market uses Workday for HR onboarding"
Target: "Workday was founded in 2005 by Dave Duffield"
Question: "When was the HR platform used by Lee's Market founded?"
Answer: "2005"
Why good: Describes entity indirectly ("HR platform used by Lee's Market"), specific answer, requires both docs.

BAD EXAMPLES - REJECT THESE PATTERNS:

1. Leaks both numbers:
   Source: "Lee's Market retention is 82%"
   Target: "Industry average is 65%"
   Question: "How many points above 65% is 82%?"
   Why bad: Both 82% and 65% in question - can compute 17 without docs

2. Temporal coincidence only:
   Source: "Meeting on July 20, 2023"
   Target: "FAFSA 2023-24 Award Year schedule"
   Question: "Which award year does July 20, 2023 fall in?"
   Why bad: Just "same time period" - no meaningful relationship to source content

3. Generic/vague answer:
   Source: "Employee recognition program"
   Target: "Servant leadership improves engagement"
   Question: "What leadership style aligns with recognition?"
   Why bad: "servant leadership" is generic, not uniquely derivable from these docs

4. Names entity directly:
   Source: "Lee's Market uses Workday"
   Target: "Workday founded in 2005"
   Question: "When was Workday founded?"
   Why bad: Names "Workday" directly - doesn't need source doc

5. Answer restates question info:
   Source: "25% increase in premium sales"
   Target: "13% digital grocery share"
   Question: "What is the digital grocery share that Lee's Market's 25% compares to?"
   Why bad: Answer is just info already implied in question structure

6. Compound questions with multiple parts:
   Question: "How does X compare to Y, and what technology enables Z?"
   Why bad: Should ask ONE thing, not multiple unrelated questions

7. Number/text coincidence only:
   Source: "35% of budget to East Coast"
   Target: "35th Senate District"
   Why bad: Same number but NO semantic relationship

8. Topically incoherent:
   Source: "Ancient currency 3000 BC"
   Target: "Modern scheduling software"
   Why bad: No logical thematic connection between topics

9. Jargon or phrase answers:
   Answers like: "Natural Language-to-SQL", "iPads in stores", "AI-hospital assistants"
   Why bad: Answer should be clear entity (name, number, proper noun), not jargon

OUTPUT:
Best candidate number (1-indexed), or 0 if none meet ALL requirements.
Brief explanation of why it's good or why all fail.
"""

WEB_INIT_PROMPT = """Generate a factual question-answer pair from this document.

DOCUMENT:
{doc_text}

Requirements:
- Question should be answerable ONLY from this document
- Answer must be 1-3 words, specific (entity, number, proper noun)
- Answer must be extractable from the document text
- Avoid subjective or opinion questions

OUTPUT:
QUESTION: <your question>
ANSWER: <1-3 word answer from the document>
"""


# =============================================================================
# Data Loading
# =============================================================================

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def task_to_company(task_id: str) -> str:
    """DR0001-DR0005: Lee's Market, DR0006-DR0010: MediConn, DR0011-DR0015: Elexion"""
    num = int(task_id[2:])
    if num <= 5:
        return "Lee's Market"
    elif num <= 10:
        return "MediConn Solutions"
    return "Elexion Automotive"


def load_doc_texts(chunks_path: Path) -> dict[str, str]:
    """Load document texts from chunks, keyed by doc_id."""
    doc_texts: dict[str, list[str]] = {}
    for rec in load_jsonl(chunks_path):
        doc_id = rec.get("doc_id", "")
        text = rec.get("text", "")
        if doc_id and text:
            if doc_id not in doc_texts:
                doc_texts[doc_id] = []
            doc_texts[doc_id].append(text)
    # Concatenate all chunks for each doc
    return {doc_id: "\n".join(chunks) for doc_id, chunks in doc_texts.items()}


def load_secrets(path: Path, chunks_path: Path | None = None) -> list[Secret]:
    """Load secrets from inventory, optionally enriching with text from chunks."""
    # Load doc texts if chunks path provided
    doc_texts = load_doc_texts(chunks_path) if chunks_path else {}

    secrets = []
    for rec in load_jsonl(path):
        chunk_id = rec["chunk_id"]
        doc_id = rec.get("doc_id", chunk_id)
        task_id = chunk_id.split("/")[1]
        company = task_to_company(task_id)
        # Get text from secret inventory, or fall back to chunks
        text = rec.get("text", "") or doc_texts.get(doc_id, "")
        for s in rec.get("secrets", []):
            secrets.append(Secret(
                chunk_id=chunk_id,
                doc_id=doc_id,
                question=s.get("question", ""),
                answer=s.get("answer", ""),
                secret_type=s.get("secret_type", ""),
                text=text,
                company=company,
                task_id=task_id,
            ))
    return secrets


# =============================================================================
# Query Generation (LLM-based)
# =============================================================================

def generate_queries_from_state(
    state: ChainState,
    client: VLLMClient,
) -> list[str]:
    """Generate search queries based on the current chain state.

    Args:
        state: Current chain state (doc, question, answer)
        client: LLM client

    Returns:
        List of search queries (3-5)
    """
    # Create a summary of the document (first 500 chars)
    doc_summary = state.doc[:500] + "..." if len(state.doc) > 500 else state.doc

    prompt = QUERY_GENERATION_PROMPT.format(
        company_name=state.company or "",
        current_question=state.question,
        current_answer=state.answer,
        doc_summary=doc_summary,
    )

    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        stage="query_generation",
        max_tokens=200,
        temperature=0.5,
    )

    # Parse queries (one per line, skip empty/comment lines)
    raw_output = response.choices[0].message.content
    queries = []
    for line in raw_output.split('\n'):
        line = line.strip()
        if line and not line.startswith(('#', '-', '*', '1', '2', '3', '4', '5')):
            # Remove any leading numbers like "1." or "1)"
            line = re.sub(r'^\d+[.)\s]+', '', line)
            if line:
                queries.append(line)

    return queries[:5] or ["general information"]


def generate_queries_llm(
    current_hop: Hop,
    chain_context: list[Bridge],
    corpus: str,
    client: VLLMClient,
) -> list[str]:
    """Legacy function - convert Hop to ChainState and call new function."""
    state = ChainState(
        doc=current_hop.text,
        doc_id=current_hop.doc_id,
        question=current_hop.secret_question or "What is the key information?",
        answer=current_hop.secret_answer or current_hop.fact,
        hop_type=current_hop.hop_type,
        company=current_hop.company,
        task_id=current_hop.task_id,
    )
    return generate_queries_from_state(state, client)


# =============================================================================
# Bridge Discovery
# =============================================================================

def _extract_field(field: str, output: str, multiline: bool = False) -> str:
    """Extract a field value from LLM output.

    Args:
        field: Field name to extract (e.g., "QUESTION")
        output: Raw LLM output
        multiline: If True, allow multi-line values. If False, stop at newline.
    """
    if multiline:
        # For REASONING - allow multiple lines until next field
        m = re.search(rf"{field}:\s*(.+?)(?:\n[A-Z_]+:|$)", output, re.DOTALL)
        return m.group(1).strip() if m else ""
    else:
        # For QUESTION, ANSWER, LINK, RELATION - single line only
        m = re.search(rf"{field}:\s*([^\n]+)", output)
        return m.group(1).strip() if m else ""


def generate_search_query(
    current_question: str,
    current_answer: str,
    client: VLLMClient,
) -> str:
    """Generate a context-aware search query from question context.

    Instead of searching for raw answer values like "80%" or "12%",
    this generates a meaningful query based on the question topic.

    For named entities (like "Workday", "SAP"), use simpler direct queries
    to find general information about the entity.
    """
    # Check if answer looks like a named entity (capitalized, no digits)
    is_entity = (
        len(current_answer.split()) <= 3 and  # Short
        not any(c.isdigit() for c in current_answer) and  # No numbers
        not any(c in current_answer for c in '%$@') and  # No special chars
        current_answer[0].isupper() if current_answer else False  # Capitalized
    )

    if is_entity:
        # For entities, use targeted search with multiple strategies
        # Extract key context words from question
        context_words = []
        for word in current_question.lower().split():
            if word in ['vendor', 'platform', 'software', 'system', 'tool', 'company', 'provider']:
                context_words.append(word)
        context = ' '.join(context_words) if context_words else 'software company'
        return f"{current_answer} {context} founded headquarters"

    # For non-entities (numbers, metrics, etc.), use LLM to generate topical query
    prompt = QUERY_GENERATION_PROMPT.format(
        current_question=current_question,
        current_answer=current_answer,
    )
    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        stage="query_generation",
        temperature=0.3,
        max_tokens=100,
    )
    query = response.choices[0].message.content.strip()

    # Clean up - take first line if multiple
    if "\n" in query:
        query = query.split("\n")[0].strip()

    # Remove any "QUERY:" prefix if model included it
    if query.upper().startswith("QUERY:"):
        query = query[6:].strip()

    return query


def score_bridge(
    bridge: Bridge,
    current_answer: str,
    new_doc: str,
) -> float:
    """Score a bridge composition. Higher is better."""
    score = 0.0

    # CRITICAL: Question must contain the slot description (multi-hop requirement)
    slot_in_question = False
    if bridge.link:
        slot_words = set(w.lower() for w in bridge.link.split() if len(w) > 3)
        question_words = set(w.lower() for w in bridge.question.split())
        overlap = slot_words & question_words
        if len(overlap) >= max(1, len(slot_words) // 2):
            slot_in_question = True
            score += 5.0  # Big bonus for true multi-hop
        else:
            score -= 10.0  # Heavy penalty - not multi-hop

    # 1. Answer appears in document (+2)
    if bridge.answer.lower() in new_doc.lower():
        score += 2.0

    # 2. Slot description is substantive (+1 per 3 words, max 2)
    slot_word_count = len(bridge.link.split())
    score += min(slot_word_count / 3, 2.0)

    # 3. Answer is short and specific (+1 if 1-3 words)
    ans_words = len(bridge.answer.split())
    if 1 <= ans_words <= 3:
        score += 1.0

    # 4. Penalty: answer same as input (-3)
    if bridge.answer.lower() == current_answer.lower():
        score -= 3.0

    # 5. PROXIMITY: Answer should be near current_answer in the document (+3 if close)
    # This helps when doc has multiple entities - prefer facts from the right section
    doc_lower = new_doc.lower()
    answer_pos = doc_lower.find(bridge.answer.lower())
    entity_pos = doc_lower.find(current_answer.lower())
    if answer_pos != -1 and entity_pos != -1:
        distance = abs(answer_pos - entity_pos)
        if distance < 500:  # Within ~500 chars = same section
            score += 3.0
        elif distance < 1500:  # Within ~1500 chars = nearby
            score += 1.0
        else:
            score -= 2.0  # Too far - probably wrong section

    return score


def discover_bridge(
    current_question: str,
    current_answer: str,
    new_doc: str,
    client: VLLMClient,
    bridge_type: str = "L→W",
) -> Bridge | None:
    """Discover bridge by composing current Q+A with new document.

    This is iterative question composition - the bridge wraps the current
    question to create a new question that requires resolving the current
    answer first.

    Args:
        current_question: The current question in the chain
        current_answer: The answer to the current question
        new_doc: Document containing new info about current_answer
        client: LLM client
        bridge_type: For tracking ("L→W", "W→L", etc.)

    Returns:
        Bridge with composed question and new answer, or None if no valid bridge
    """
    prompt = BRIDGE_PROMPT.format(
        current_question=current_question,
        current_answer=current_answer,
        new_doc=new_doc[:2000],
    )
    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        stage="bridge_discovery",
        max_tokens=500,
        temperature=0.3,
    )
    output = response.choices[0].message.content

    if "NO_BRIDGE" in output:
        return None

    answer_slot = _extract_field("ANSWER_SLOT", output)
    question = _extract_field("QUESTION", output)
    answer = _extract_field("ANSWER", output)

    if not question or not answer:
        return None

    # Validate: current_answer should NOT appear literally in question
    if current_answer.lower() in question.lower():
        return None

    # Validate: answer should be different from current_answer (must learn something new)
    if answer.lower().strip() == current_answer.lower().strip():
        return None
    # Also check if answer is just current_answer with minor variations
    if current_answer.lower() in answer.lower() or answer.lower() in current_answer.lower():
        return None

    return Bridge(
        link=answer_slot or "",
        relation="wraps",
        from_idx=-1,
        to_idx=-1,
        bridge_type=bridge_type,
        question=question,
        answer=answer,
    )


def score_bridges(
    source_state: ChainState,
    candidates: list[tuple[Hop, Bridge]],
    client: VLLMClient,
    verbose: bool = False,
) -> tuple[Hop, Bridge] | None:
    """Score candidate bridges and return the best one.

    Args:
        source_state: Current chain state (doc, question, answer)
        candidates: List of (hop, bridge) tuples to evaluate
        client: LLM client
        verbose: Print debug info

    Returns:
        Best (hop, bridge) tuple, or None if no valid candidates
    """
    if not candidates:
        return None

    if len(candidates) == 1:
        # Only one candidate - return it directly
        return candidates[0]

    # Format candidates for prompt
    candidates_text = ""
    for i, (hop, bridge) in enumerate(candidates, 1):
        candidates_text += f"""
Candidate {i}:
  Target doc snippet: {hop.text[:400]}...
  Question: {bridge.question}
  Answer: {bridge.answer}
"""

    prompt = BRIDGE_SCORING_PROMPT.format(
        source_doc=source_state.doc[:1500],
        source_q=source_state.question,
        source_a=source_state.answer,
        candidates=candidates_text,
    )

    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        stage="bridge_scoring",
        max_tokens=300,
        temperature=0.2,
    )

    output = response.choices[0].message.content

    if verbose:
        print(f"    Bridge scoring response: {output[:200]}...")

    # Parse response for best candidate number
    match = re.search(r'\b([0-9]+)\b', output)
    if match:
        idx = int(match.group(1))
        if idx == 0:
            # Model said none are valid
            return None
        if 1 <= idx <= len(candidates):
            return candidates[idx - 1]

    # Fallback: return first candidate if parsing fails
    return candidates[0]


def initialize_web_start(
    searcher: UnifiedSearcher,
    company: str,
    client: VLLMClient,
    web_backend: str = "bm25",
    verbose: bool = False,
) -> ChainState | None:
    """Initialize chain state from a random web document.

    For W→L patterns, we need to start from a web document with a QA pair.
    """
    # Domain queries by company
    domain_queries = {
        "Lee's Market": [
            "retail industry trends 2024",
            "grocery market statistics",
            "food retail technology",
            "supermarket supply chain",
            "retail employee retention",
        ],
        "MediConn Solutions": [
            "healthcare technology trends",
            "telehealth statistics 2024",
            "medical software market",
            "EHR adoption rates",
            "healthcare cybersecurity",
        ],
        "Elexion Automotive": [
            "automotive industry trends",
            "EV manufacturing statistics",
            "automotive supply chain 2024",
            "electric vehicle market",
            "automotive technology",
        ],
    }

    queries = domain_queries.get(company, ["industry trends 2024"])
    random.shuffle(queries)

    for query in queries[:3]:
        if verbose:
            print(f"  Web init query: {query}")

        hits = searcher.search(query, corpus="web", k=20, web_backend=web_backend)
        random.shuffle(hits)

        for hit in hits[:10]:
            if len(hit.text) < 200:
                continue  # Skip short docs

            prompt = WEB_INIT_PROMPT.format(doc_text=hit.text[:2000])
            response = client.chat(
                messages=[{"role": "user", "content": prompt}],
                stage="web_init",
                max_tokens=200,
                temperature=0.3,
            )
            output = response.choices[0].message.content

            question = _extract_field("QUESTION", output)
            answer = _extract_field("ANSWER", output)

            if question and answer and len(answer.split()) <= 5:
                if verbose:
                    print(f"  Web init success: Q={question[:50]}... A={answer}")
                return ChainState(
                    doc=hit.text,
                    doc_id=hit.doc_id,
                    question=question,
                    answer=answer,
                    hop_type="W",
                    company=company,
                )

    if verbose:
        print("  Web init failed: no valid QA pairs found")
    return None


# =============================================================================
# Core Chain Building
# =============================================================================

def find_next_hop(
    current_hop: Hop,
    current_question: str,
    current_answer: str,
    next_type: HopType,
    searcher: UnifiedSearcher,
    secrets: list[Secret],
    client: VLLMClient,
    task_filter: str | None = None,
    used_doc_ids: set[str] | None = None,
    web_backend: str = "bm25",
    k_hits: int = 10,
    verbose: bool = False,
) -> tuple[Hop, Bridge] | None:
    """Find the next hop in a chain using iterative question composition.

    Args:
        current_hop: Current hop to extend from
        current_question: The current question in the chain
        current_answer: The answer to the current question (entity to bridge on)
        next_type: Type of next hop ("L" or "W")
        searcher: Unified searcher for local/web
        secrets: Available secrets for local hops
        client: LLM client
        task_filter: Task ID to filter local docs (e.g., "DR0001")
        used_doc_ids: Doc IDs already used in chain
        web_backend: "bm25", "dense", or "bm25_rerank_dense"
        k_hits: Number of hits per query
        verbose: Print debug info
    """
    used_doc_ids = used_doc_ids or set()
    bridge_type = f"{current_hop.hop_type}→{next_type}"

    if verbose:
        print(f"  Finding {bridge_type} hop...{f' (task={task_filter})' if task_filter and next_type == 'L' else ''}")
        print(f"    Current Q: {current_question[:60]}...")
        print(f"    Current A: {current_answer}")

    if bridge_type == "L→W":
        # Local → Web: generate context-aware query from question
        query = generate_search_query(
            current_question=current_question,
            current_answer=current_answer,
            client=client,
        )

        if verbose:
            print(f"    Generated query: {query}")

        # Search with the generated query
        hits = searcher.search(query, corpus="web", k=k_hits, web_backend=web_backend)
        hits = [h for h in hits if h.doc_id not in used_doc_ids]

        # Also try direct entity search if answer is not purely numeric
        if current_answer and not current_answer.replace("%", "").replace(".", "").replace("$", "").replace(",", "").isdigit():
            entity_hits = searcher.search(current_answer, corpus="web", k=k_hits, web_backend=web_backend)
            entity_hits = [h for h in entity_hits if h.doc_id not in used_doc_ids]
            # Dedupe
            seen_ids = {h.doc_id for h in hits}
            for h in entity_hits:
                if h.doc_id not in seen_ids:
                    hits.append(h)
                    seen_ids.add(h.doc_id)

        # Try composing with each candidate doc
        bridge_candidates: list[tuple[Hop, Bridge, float]] = []
        for hit in hits[:10]:
            bridge = discover_bridge(
                current_question=current_question,
                current_answer=current_answer,
                new_doc=hit.text,
                client=client,
                bridge_type="L→W",
            )
            if bridge:
                score = score_bridge(bridge, current_answer, hit.text)
                new_hop = Hop(
                    hop_type="W",
                    doc_id=hit.doc_id,
                    text=hit.text,
                    fact=bridge.answer,
                )
                bridge_candidates.append((new_hop, bridge, score))
                if verbose:
                    print(f"    Candidate (score={score:.1f}): Q={bridge.question[:50]}... A={bridge.answer}")

        # Score and pick best bridge
        if bridge_candidates:
            bridge_candidates.sort(key=lambda x: x[2], reverse=True)
            if verbose:
                print(f"    Found {len(bridge_candidates)} valid compositions, best score={bridge_candidates[0][2]:.1f}")
            best_hop, best_bridge, _ = bridge_candidates[0]
            return best_hop, best_bridge

    elif bridge_type == "W→L":
        # Web → Local: generate context-aware query from question
        query = generate_search_query(
            current_question=current_question,
            current_answer=current_answer,
            client=client,
        )

        if verbose:
            print(f"    Generated query: {query}")

        # Search local corpus
        hits = searcher.search(query, corpus="local", k=k_hits)
        hits = [h for h in hits if h.doc_id not in used_doc_ids]

        # Also try direct entity search if answer is not purely numeric
        if current_answer and not current_answer.replace("%", "").replace(".", "").replace("$", "").replace(",", "").isdigit():
            entity_hits = searcher.search(current_answer, corpus="local", k=k_hits)
            entity_hits = [h for h in entity_hits if h.doc_id not in used_doc_ids]
            seen_ids = {h.doc_id for h in hits}
            for h in entity_hits:
                if h.doc_id not in seen_ids:
                    hits.append(h)
                    seen_ids.add(h.doc_id)

        # Find secrets from relevant doc_ids
        relevant_doc_ids = {h.doc_id for h in hits[:20]}
        candidate_secrets = [s for s in secrets if s.doc_id in relevant_doc_ids]
        if task_filter:
            candidate_secrets = [s for s in candidate_secrets if s.task_id == task_filter]

        if verbose:
            print(f"    Found {len(candidate_secrets)} relevant secrets")

        # Try composing with each candidate
        bridge_candidates: list[tuple[Hop, Bridge, float]] = []
        for secret in candidate_secrets[:10]:
            if secret.doc_id in used_doc_ids:
                continue
            bridge = discover_bridge(
                current_question=current_question,
                current_answer=current_answer,
                new_doc=secret.text,
                client=client,
                bridge_type="W→L",
            )
            if bridge:
                score = score_bridge(bridge, current_answer, secret.text)
                new_hop = Hop(
                    hop_type="L",
                    doc_id=secret.doc_id,
                    text=secret.text,
                    fact=bridge.answer,
                    company=secret.company,
                    task_id=secret.task_id,
                    secret_question=bridge.question,
                    secret_answer=bridge.answer,
                )
                bridge_candidates.append((new_hop, bridge, score))
                if verbose:
                    print(f"    Candidate (score={score:.1f}): Q={bridge.question[:50]}... A={bridge.answer}")

        # Score and pick best bridge
        if bridge_candidates:
            bridge_candidates.sort(key=lambda x: x[2], reverse=True)
            if verbose:
                print(f"    Found {len(bridge_candidates)} valid compositions, best score={bridge_candidates[0][2]:.1f}")
            best_hop, best_bridge, _ = bridge_candidates[0]
            return best_hop, best_bridge

    elif bridge_type == "L→L":
        # Local → Local: search same task for docs about current_answer
        target_task = task_filter or current_hop.task_id
        same_task = [s for s in secrets
                    if s.task_id == target_task
                    and s.doc_id not in used_doc_ids
                    and s.doc_id != current_hop.doc_id]

        random.shuffle(same_task)

        for secret in same_task[:10]:
            bridge = discover_bridge(
                current_question=current_question,
                current_answer=current_answer,
                new_doc=secret.text,
                client=client,
                bridge_type="L→L",
            )
            if bridge:
                new_hop = Hop(
                    hop_type="L",
                    doc_id=secret.doc_id,
                    text=secret.text,
                    fact=bridge.answer,
                    company=secret.company,
                    task_id=secret.task_id,
                    secret_question=bridge.question,
                    secret_answer=bridge.answer,
                )
                if verbose:
                    print(f"    Found: Q={bridge.question[:50]}... A={bridge.answer}")
                return new_hop, bridge

    elif bridge_type == "W→W":
        # Web → Web: search web for docs about current_answer
        queries = [
            current_answer,
            f"{current_answer} information",
            f"what is {current_answer}",
        ]

        if verbose:
            print(f"    Searching web for: {queries[:3]}")

        all_hits = []
        for query in queries[:3]:
            hits = searcher.search(query, corpus="web", k=k_hits, web_backend=web_backend)
            all_hits.extend([h for h in hits
                           if h.doc_id not in used_doc_ids
                           and h.doc_id != current_hop.doc_id])

        seen = set()
        unique_hits = []
        for h in all_hits:
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                unique_hits.append(h)

        for hit in unique_hits[:10]:
            bridge = discover_bridge(
                current_question=current_question,
                current_answer=current_answer,
                new_doc=hit.text,
                client=client,
                bridge_type="W→W",
            )
            if bridge:
                new_hop = Hop(
                    hop_type="W",
                    doc_id=hit.doc_id,
                    text=hit.text,
                    fact=bridge.answer,
                )
                if verbose:
                    print(f"    Found: Q={bridge.question[:50]}... A={bridge.answer}")
                return new_hop, bridge

    if verbose:
        print(f"    No valid {bridge_type} bridge found")
    return None


def compose_question(hops: list[Hop], bridges: list[Bridge], client: VLLMClient) -> str:
    """Generate a multi-hop question from the chain."""

    # Build chain description with FULL bridge context
    chain_parts = []
    for i, hop in enumerate(hops):
        hop_desc = f"Hop {i+1} ({'Local' if hop.hop_type == 'L' else 'Web'}): "
        if hop.hop_type == "L":
            hop_desc += f"{hop.secret_question or 'N/A'} → {hop.fact}"
            if hop.company:
                hop_desc += f" [Company: {hop.company}]"
        else:
            hop_desc += hop.fact[:150]
        chain_parts.append(hop_desc)

        # Include edge triple so LLM understands the hop relationships
        if i < len(bridges):
            b = bridges[i]
            # Format: "SAP" → founded in → "Germany"
            edge_desc = f'  → Bridge: "{b.link}" → {b.relation} → "{b.answer}"'
            if b.question:
                edge_desc += f'\n       Q: {b.question}'
            chain_parts.append(edge_desc)

    chain_description = "\n".join(chain_parts)

    prompt = COMPOSE_QUESTION_PROMPT.format(
        n=len(hops),
        chain_description=chain_description,
        final_answer=hops[-1].fact,
    )

    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        stage="compose_question",
        max_tokens=200,
        temperature=0.5,
    )
    output = response.choices[0].message.content

    # Extract question
    m = re.search(r"QUESTION:\s*(.+)", output, re.DOTALL)
    if m:
        return m.group(1).strip()
    return output.strip()


def extend_question(
    current_question: str,
    current_answer: str,
    new_hop: Hop,
    bridge: Bridge,
    hop_num: int,
    company: str,
    client: VLLMClient,
) -> str:
    """Extend the running question to incorporate a new hop.

    Args:
        current_question: The question so far
        current_answer: The answer to the current question
        new_hop: The new hop being added
        bridge: The bridge connecting to the new hop
        hop_num: Which hop number this is (2, 3, etc.)
        company: Company name for context
        client: LLM client

    Returns:
        Extended question incorporating the new hop
    """
    prompt = EXTEND_QUESTION_PROMPT.format(
        company_name=company,
        current_question=current_question,
        current_answer=current_answer,
        new_doc=new_hop.text[:1500],
        hop_num=hop_num,
        bridge_link=bridge.link,
        bridge_relation=bridge.relation,
        bridge_answer=bridge.answer,
    )

    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        stage="extend_question",
        max_tokens=300,
        temperature=0.5,
    )
    output = response.choices[0].message.content

    # Extract extended question
    m = re.search(r"EXTENDED_QUESTION:\s*(.+)", output, re.DOTALL)
    if m:
        return m.group(1).strip()
    return output.strip()


def build_chain(
    pattern: str,
    start_hop: Hop,
    searcher: UnifiedSearcher,
    secrets: list[Secret],
    client: VLLMClient,
    web_backend: str = "bm25",
    k_hits: int = 10,
    verbose: bool = False,
    question_mode: str = "two-phase",
) -> Chain | None:
    """Build a chain following the given pattern.

    Args:
        question_mode: "two-phase" (compose question at end) or "iterative" (build incrementally)
    """

    if not pattern:
        raise ValueError("Pattern cannot be empty")
    if pattern[0].upper() != start_hop.hop_type:
        raise ValueError(f"Pattern '{pattern}' doesn't match start hop type '{start_hop.hop_type}'")

    pattern = pattern.upper()
    for c in pattern:
        if c not in ("L", "W"):
            raise ValueError(f"Invalid character in pattern: '{c}'. Use only L and W.")

    if verbose:
        print(f"\nBuilding chain: {pattern}")
        print(f"  Start: {start_hop.fact[:60]}...")
        if start_hop.task_id:
            print(f"  Task: {start_hop.task_id}")

    hops = [start_hop]
    bridges = []
    # Track task context - all local hops must be from this task
    task_context = start_hop.task_id
    used_doc_ids = {start_hop.doc_id}

    # Track current question and answer state (iterative composition)
    current_question = start_hop.secret_question or f"What is the key fact from {start_hop.company or 'this document'}?"
    current_answer = start_hop.secret_answer or start_hop.fact
    company_context = start_hop.company or ""

    if verbose:
        print(f"  Initial Q: {current_question}")
        print(f"  Initial A: {current_answer}")

    for i, next_type in enumerate(pattern[1:]):
        result = find_next_hop(
            current_hop=hops[-1],
            current_question=current_question,
            current_answer=current_answer,
            next_type=next_type,
            searcher=searcher,
            secrets=secrets,
            client=client,
            task_filter=task_context if next_type == "L" else None,
            used_doc_ids=used_doc_ids,
            web_backend=web_backend,
            k_hits=k_hits,
            verbose=verbose,
        )

        if result is None:
            if verbose:
                print(f"  Chain failed at hop {i+2}")
            return None

        next_hop, bridge = result
        bridge.from_idx = len(hops) - 1
        bridge.to_idx = len(hops)
        hops.append(next_hop)
        bridges.append(bridge)
        used_doc_ids.add(next_hop.doc_id)

        # Update task context if we found a local hop
        if next_hop.task_id and not task_context:
            task_context = next_hop.task_id
        if next_hop.company and not company_context:
            company_context = next_hop.company

        # Update current Q+A from bridge (iterative composition)
        current_question = bridge.question
        current_answer = bridge.answer
        if verbose:
            print(f"  Hop {i+2}: Q={current_question[:60]}...")
            print(f"          A={current_answer}")

    # Final question and answer come from iterative composition
    question = current_question
    answer = current_answer

    chain = Chain(
        chain_id="",  # Set by caller
        pattern=pattern,
        hops=hops,
        bridges=bridges,
        question=question,
        answer=answer,
    )

    if verbose:
        print(f"  Final Q: {question}")
        print(f"  Final A: {answer}")

    return chain


# =============================================================================
# Verification
# =============================================================================

def can_answer_question(
    question: str,
    context: str,
    client: VLLMClient,
    expected_answer: str = "",
) -> tuple[bool, str]:
    """Check if question can be answered with given context. Returns (answerable, reason)."""

    if not context.strip():
        context = "(No context provided)"

    prompt = CAN_ANSWER_PROMPT.format(
        context=context[:4000],
        question=question,
        expected_answer=expected_answer or "(not provided)",
    )

    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        stage="verify_answerable",
        max_tokens=200,
        temperature=0.0,
    )
    output = response.choices[0].message.content

    answerable = "ANSWERABLE: YES" in output.upper()

    # Extract reason/answer
    if answerable:
        m = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", output, re.DOTALL)
        reason = m.group(1).strip() if m else ""
    else:
        m = re.search(r"REASON:\s*(.+?)(?:\n|$)", output, re.DOTALL)
        reason = m.group(1).strip() if m else ""

    return answerable, reason


def verify_chain(chain: Chain, client: VLLMClient, verbose: bool = False) -> dict:
    """Verify that question requires all hops to answer."""

    if verbose:
        print("  Verifying chain...")

    results = {}
    expected = chain.answer

    # Test 1: Can answer with NO documents? (should fail - needs docs)
    no_docs_ok, no_docs_reason = can_answer_question(chain.question, "", client, expected)
    results["no_docs"] = {"answerable": no_docs_ok, "reason": no_docs_reason}

    # Test 2: Can answer with ONLY first hop? (should fail - needs more)
    first_ok, first_reason = can_answer_question(chain.question, chain.hops[0].text, client, expected)
    results["first_only"] = {"answerable": first_ok, "reason": first_reason}

    # Test 3: Can answer with ONLY last hop? (should fail - needs context)
    last_ok, last_reason = can_answer_question(chain.question, chain.hops[-1].text, client, expected)
    results["last_only"] = {"answerable": last_ok, "reason": last_reason}

    # Test 4: Can answer with ALL hops? (should succeed)
    all_context = "\n\n---\n\n".join(h.text for h in chain.hops)
    all_ok, all_reason = can_answer_question(chain.question, all_context, client, expected)
    results["all_hops"] = {"answerable": all_ok, "reason": all_reason}

    # Valid: can answer with all, cannot with subsets
    results["valid"] = (
        all_ok and
        not no_docs_ok and
        not first_ok and
        not last_ok
    )

    if verbose:
        status = "✓ VALID" if results["valid"] else "✗ INVALID"
        print(f"    {status}")
        if not results["valid"]:
            if no_docs_ok:
                print(f"      - Answerable without docs")
            if first_ok:
                print(f"      - Answerable with first hop only")
            if last_ok:
                print(f"      - Answerable with last hop only")
            if not all_ok:
                print(f"      - NOT answerable even with all hops")

    return results


def is_answer_rl_friendly(answer: str, max_words: int = 10) -> bool:
    """Check if answer is short enough for RL-friendly grading."""
    if not answer:
        return False
    words = answer.split()
    return len(words) <= max_words and len(answer) <= 100


# =============================================================================
# CLI
# =============================================================================

def chain_to_dict(chain: Chain) -> dict:
    """Convert chain to JSON-serializable dict."""
    # Get task_id from first local hop (all local hops share the same task)
    task_id = None
    for h in chain.hops:
        if h.task_id:
            task_id = h.task_id
            break

    # Determine answer source (L or W based on last hop)
    answer_source = chain.hops[-1].hop_type if chain.hops else None

    return {
        "chain_id": chain.chain_id,
        "pattern": chain.pattern,
        "task_id": task_id,  # Task context for this chain
        "answer_source": answer_source,  # "L" or "W"
        "hops": [
            {
                "type": h.hop_type,
                "doc_id": h.doc_id,
                "fact": h.fact,
                "company": h.company,
                "task_id": h.task_id,
                "secret_question": h.secret_question,
                "secret_answer": h.secret_answer,
                # Don't include full text in output - too large
            }
            for h in chain.hops
        ],
        "bridges": [
            {
                "link": b.link,
                "relation": b.relation,
                "answer": b.answer,
                "type": b.bridge_type,
                "from": b.from_idx,
                "to": b.to_idx,
            }
            for b in chain.bridges
        ],
        "question": chain.question,
        "answer": chain.answer,
        "verification": chain.verification,
    }


def main():
    parser = argparse.ArgumentParser(description="Build multi-hop question chains")
    parser.add_argument("--pattern", type=str, help="Single pattern (e.g., LW, LWLW)")
    parser.add_argument("--patterns", nargs="+", help="Multiple patterns to try")
    parser.add_argument("--random-patterns", action="store_true",
                       help="Generate random patterns")
    parser.add_argument("--min-length", type=int, default=2,
                       help="Min pattern length for random")
    parser.add_argument("--max-length", type=int, default=4,
                       help="Max pattern length for random")
    parser.add_argument("-n", type=int, default=5, help="Number of chains to generate")
    parser.add_argument("--company", type=str, help="Filter to specific company")
    parser.add_argument("--task", type=str, help="Filter to specific task (e.g., DR0001)")
    parser.add_argument("--output", "-o", type=str, help="Output JSONL path")
    parser.add_argument("--verify", action="store_true", default=True,
                       help="Verify chains (default: True)")
    parser.add_argument("--no-verify", action="store_false", dest="verify",
                       help="Skip verification")
    parser.add_argument("--require-local-answer", action="store_true", default=False,
                       help="Only generate patterns ending in L")
    parser.add_argument("--max-answer-words", type=int, default=10,
                       help="Max words in answer (default 10, for RL-friendly grading)")
    parser.add_argument("--question-mode", choices=["two-phase", "iterative"],
                       default="two-phase",
                       help="Question generation approach: two-phase (compose at end) or iterative (build incrementally)")
    parser.add_argument("--vllm-url", type=str, help="vLLM API URL")
    parser.add_argument("--vllm-model", type=str,
                       default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--web-backend", choices=["bm25", "dense", "bm25_rerank_dense"],
                       default="bm25")
    parser.add_argument("--k-hits", type=int, default=10, help="Hits per query")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--chunks-path", type=str)
    parser.add_argument("--secrets-path", type=str)
    parser.add_argument("--web-index-path", type=str)
    parser.add_argument("--entity-only", action="store_true",
                       help="Only use secrets with entity answers (not purely numeric)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Determine patterns
    if args.pattern:
        patterns = [args.pattern.upper()]
    elif args.patterns:
        patterns = [p.upper() for p in args.patterns]
    elif args.random_patterns:
        patterns = []
        for _ in range(args.n * 2):  # Generate extra since some may fail
            length = random.randint(args.min_length, args.max_length)
            pat = "".join(random.choice("LW") for _ in range(length))
            patterns.append(pat)
    else:
        patterns = ["LW"]  # Default

    # Filter patterns if requiring local answer (ending in L)
    if args.require_local_answer:
        original_count = len(patterns)
        patterns = [p for p in patterns if p.endswith("L")]
        if not patterns:
            # No patterns end in L - suggest alternatives
            print("Error: --require-local-answer is set but no patterns end in L")
            print("  Use patterns like: LWL, WL, LWLWL, LLL")
            print("  Or use --allow-web-answer to allow patterns ending in W")
            return 1
        if len(patterns) < original_count:
            print(f"  Filtered to {len(patterns)} patterns ending in L (verifiable answers)")

    chunks_path = Path(args.chunks_path) if args.chunks_path else DEFAULT_CHUNKS_LOCAL
    secrets_path = Path(args.secrets_path) if args.secrets_path else DEFAULT_SECRET_INVENTORY
    web_index = args.web_index_path or DEFAULT_WEB_BM25_INDEX

    # Initialize
    print("Loading resources...")
    searcher = UnifiedSearcher(local_chunks_path=str(chunks_path), web_bm25_index_path=web_index)
    print(f"  Web docs: {searcher.web_bm25.num_docs if searcher.web_bm25 else 0}")

    secrets = load_secrets(secrets_path, chunks_path=chunks_path)
    print(f"  Secrets: {len(secrets)}")

    if args.company:
        secrets = [s for s in secrets if args.company.lower() in s.company.lower()]
        print(f"  Filtered to company '{args.company}': {len(secrets)} secrets")

    if args.task:
        secrets = [s for s in secrets if s.task_id == args.task.upper()]
        print(f"  Filtered to task '{args.task}': {len(secrets)} secrets")

    if args.entity_only:
        # Filter to entity-valued answers (not purely numeric/date-like)
        import re
        def is_entity(ans: str) -> bool:
            # Remove common numeric patterns
            clean = re.sub(r'[\$%,\.\d]', '', ans).strip()
            if not clean:
                return False  # Pure number
            # Skip short date-like things
            if len(ans) < 5:
                return False
            # Skip if mostly digits
            digit_ratio = sum(c.isdigit() for c in ans) / max(len(ans), 1)
            if digit_ratio > 0.5:
                return False
            return True
        secrets = [s for s in secrets if is_entity(s.answer)]
        print(f"  Entity-only filter: {len(secrets)} secrets")

    if not secrets:
        print("No secrets found!")
        return 1

    vllm_url = args.vllm_url or os.getenv("VLLM_API_URL")
    if not vllm_url:
        print("Error: --vllm-url required (or set VLLM_API_URL)")
        return 1

    os.environ["VLLM_API_URL"] = vllm_url
    client = VLLMClient(model=args.vllm_model, api_url=vllm_url)
    print(f"  LLM: {args.vllm_model}")
    print(f"  Question mode: {args.question_mode}")

    # Build chains
    chains = []
    valid_count = 0
    pattern_idx = 0
    attempts = 0
    max_attempts = args.n * 5

    while len(chains) < args.n and attempts < max_attempts:
        attempts += 1
        pattern = patterns[pattern_idx % len(patterns)]
        pattern_idx += 1

        # Pick starting point based on first character
        if pattern[0] == "L":
            secret = random.choice(secrets)
            start_hop = Hop(
                hop_type="L",
                doc_id=secret.doc_id,
                text=secret.text,
                fact=secret.answer,
                company=secret.company,
                task_id=secret.task_id,
                secret_question=secret.question,
                secret_answer=secret.answer,
            )
        else:
            # W start: use initialize_web_start to get proper QA pair
            # Pick a random company context for bridging to local docs
            company = args.company or random.choice(["Lee's Market", "MediConn Solutions", "Elexion Automotive"])
            start_state = initialize_web_start(
                searcher=searcher,
                company=company,
                client=client,
                web_backend=args.web_backend,
                verbose=args.verbose,
            )
            if start_state is None:
                if args.verbose:
                    print(f"  Failed to initialize web start")
                continue
            start_hop = Hop(
                hop_type="W",
                doc_id=start_state.doc_id,
                text=start_state.doc,
                fact=start_state.answer,
                company=start_state.company,
                secret_question=start_state.question,
                secret_answer=start_state.answer,
            )

        chain = build_chain(
            pattern=pattern,
            start_hop=start_hop,
            searcher=searcher,
            secrets=secrets,
            client=client,
            web_backend=args.web_backend,
            k_hits=args.k_hits,
            verbose=args.verbose,
            question_mode=args.question_mode,
        )

        if chain is None:
            continue

        # Check answer is RL-friendly (short enough for grading)
        answer_ok = is_answer_rl_friendly(chain.answer, args.max_answer_words)
        if not answer_ok and args.verbose:
            print(f"  Skipping: answer too long ({len(chain.answer.split())} words)")
            continue

        # Verify (but keep chain regardless of result)
        if args.verify:
            verification = verify_chain(chain, client, verbose=args.verbose)
            chain.verification = verification
            if verification["valid"]:
                valid_count += 1

        chain.chain_id = f"chain_{len(chains)+1:04d}"
        chains.append(chain)

        # Show status
        valid_marker = ""
        if args.verify and chain.verification:
            valid_marker = " ✓" if chain.verification["valid"] else " ✗"
        answer_source = chain.hops[-1].hop_type
        print(f"\n[{len(chains)}/{args.n}] {chain.pattern}{valid_marker} (ans:{answer_source})")
        print(f"  Q: {chain.question}")
        print(f"  A: {chain.answer}")

    print(f"\n{'='*60}")
    print(f"Generated {len(chains)} chains ({attempts} attempts)")
    if args.verify:
        print(f"  Valid: {valid_count}/{len(chains)} ({100*valid_count/len(chains):.0f}%)" if chains else "")

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            for chain in chains:
                f.write(json.dumps(chain_to_dict(chain), ensure_ascii=False) + "\n")
        print(f"Wrote {output_path}")
    else:
        # Print summary
        for chain in chains:
            print(f"\n{chain.chain_id} ({chain.pattern}):")
            for i, hop in enumerate(chain.hops):
                prefix = "L" if hop.hop_type == "L" else "W"
                print(f"  {prefix}{i+1}: {hop.fact[:60]}...")
            print(f"  Q: {chain.question}")
            print(f"  A: {chain.answer}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
