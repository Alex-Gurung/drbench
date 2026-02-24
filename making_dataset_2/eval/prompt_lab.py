#!/usr/bin/env python3
"""Prompt lab: iterate on the two-prompt bridge composition approach.

Two-prompt approach:
  Prompt 1 (BRIDGE): Find a factual connection between the previous answer
    and a new document. Produce a simple, direct question.
  Prompt 2 (BLEND): Rewrite that question by replacing intermediate answers
    with indirect descriptions, creating a hierarchical multi-hop question.

Usage:
    /home/toolkit/.mamba/envs/vllm013/bin/python -m making_dataset_2.eval.prompt_lab \
        --model MiniMax-M2.1 \
        --base-url http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-m21-vllm:8000/v1 \
        --n 5 --seed 42

    # Quick iteration with Qwen3-30B:
    /home/toolkit/.mamba/envs/vllm013/bin/python -m making_dataset_2.eval.prompt_lab \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
        --base-url http://127.0.0.1:8000/v1 \
        --n 5 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from making_dataset_2.data_loading import (
    build_doc_lookup, filter_seed_secrets, load_chunks_local, load_secrets,
)
from making_dataset_2.llm import LLMClient
from making_dataset_2.retrieval.bm25 import BM25Searcher

# ============================================================================
# PROMPT 1: BRIDGE FINDING — find a factual connection, direct question is OK
# ============================================================================

PROMPT_1_BRIDGE = """\
You are finding factual connections between two documents.

Given a previous question-answer pair and a new document, find a new fact in the new document that connects to the previous answer. Write a simple, direct question about that new fact.

If the previous answer (or a close variant like singular/plural, abbreviation/full name) does not appear in or is not clearly discussed in the new document, output NO_BRIDGE. Most pairs will not connect.

PREVIOUS STATE:
- Question: {global_question}
- Answer: "{global_answer}"

NEW DOCUMENT:
<new_document>
{candidate_doc_text}
</new_document>

Rules:
- The previous answer must appear in or be discussed in the new document — if not, output NO_BRIDGE
- The new answer must be DIFFERENT from the previous answer (a new fact, not the same entity)
- The new answer must be 1-5 words, a specific fact (date, number, name, place) from the new document
- The question CAN use the previous answer directly (e.g., "When was Workday founded?")
- Do NOT reference the source — no "according to the ...", no "in the document/article/report/study/outlook/guide"
- BAD: "What role do AI chatbots serve according to the telehealth outlook?" (references a source)
- GOOD: "What role do AI chatbots serve in health systems?"

<answer>NO_BRIDGE: reason</answer>
or
<answer>
QUESTION: a direct question whose answer is a new fact from the new document
ANSWER: 1-5 words from the new document (must differ from the previous answer)
JUSTIFICATION: brief explanation of the connection
</answer>"""

# ============================================================================
# PROMPT 2: HIERARCHICAL BLENDING — rewrite to replace answers with descriptions
# ============================================================================

PROMPT_2_BLEND = """\
You are rewriting a question to make it require multi-hop reasoning.

You have a chain of question-answer pairs. Rewrite the LAST question so that instead of using intermediate answers directly, it describes them using context from earlier questions. The reader must solve each hop to answer the final question.

CHAIN:
{chain_description}

DIRECT QUESTION (to rewrite): "{direct_question}"
ANSWER: "{direct_answer}"

Example:
Chain: Q1: "What HR platform does Lee's Market use?" → A1: "Workday"
Direct question: "When was Workday founded?"
Rewritten: "When was the HR platform Lee's Market uses founded?"

Example:
Chain: Q1: "What system was discussed in Lee's Market's Q3 training budget?" → A1: "chatbots"
Direct question: "What CAGR is the chatbot market projected to grow at through 2026?"
BAD: "What CAGR is the chatbot market discussed in Lee's Market's Q3 training budget projected to grow at?" (still contains "chatbot")
GOOD: "What CAGR is the market for the system discussed in Lee's Market's Q3 training budget projected to grow at through 2026?"

Example:
Chain: Q1: "What internal system is being integrated at Elexion Automotive?" → A1: "CRM system"
Direct question: "What cloud platform extends CRM systems for automakers?"
BAD: "What cloud platform extends the CRM system being integrated at Elexion Automotive?" (contains "CRM system")
GOOD: "What cloud platform extends the system being integrated at Elexion Automotive for automakers?"

Rules:
- Replace each intermediate answer with an indirect description — the rewritten question must NOT contain any intermediate answer or close variant (singular/plural, abbreviation)
- The rewritten question must read naturally — like a real research question
- Do NOT reference any document, article, report, or study
- The answer stays the same

<answer>
REWRITTEN: the rewritten multi-hop question
</answer>"""

# ============================================================================
# Paths
# ============================================================================

CHUNKS_LOCAL = ROOT / "making_dataset" / "outputs" / "chunks_local.jsonl"
CHUNKS_WEB = ROOT / "making_dataset_2" / "outputs" / "chunks_web_drbench_urls.jsonl"
SECRETS = ROOT / "making_dataset_2" / "outputs" / "secret_inventory.jsonl"

DOC_LIMIT = 8000  # chars of doc text to include

# ============================================================================
# Parsers
# ============================================================================

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks that some models produce."""
    return _THINK_RE.sub("", text).strip()

def parse_bridge(text: str) -> dict | None:
    text = _strip_think(text)
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    block = matches[-1].strip()
    if block.upper().startswith("NO_BRIDGE"):
        return {"no_bridge": True, "reason": block}

    def kv(key):
        lines = block.splitlines()
        prefix = f"{key}:"
        collecting = False
        result = []
        for line in lines:
            s = line.strip()
            if s.upper().startswith(prefix.upper()):
                rest = s[len(prefix):].strip()
                if rest:
                    result.append(rest)
                collecting = True
                continue
            if collecting:
                if re.match(r"^[A-Z][A-Z_]{2,}:", s):
                    break
                if s:
                    result.append(s)
        return " ".join(result)

    q = kv("QUESTION")
    a = kv("ANSWER")
    j = kv("JUSTIFICATION")
    if not q or not a:
        return None
    return {"question": q, "answer": a, "justification": j}


def parse_blend(text: str) -> str | None:
    text = _strip_think(text)
    # Try <answer> tags first
    matches = _ANSWER_RE.findall(text)
    if matches:
        block = matches[-1].strip()
        # Try REWRITTEN: prefix
        for line in block.splitlines():
            s = line.strip()
            if s.upper().startswith("REWRITTEN:"):
                rest = s[len("REWRITTEN:"):].strip()
                if rest.startswith('"') and rest.endswith('"'):
                    rest = rest[1:-1]
                return rest
        # Whole block is the question
        if block.startswith('"') and block.endswith('"'):
            block = block[1:-1]
        if block and ("?" in block or block[0].isupper()):
            return block

    # Fallback: bare REWRITTEN: without <answer> tags (Qwen-style)
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("REWRITTEN:"):
            rest = s[len("REWRITTEN:"):].strip()
            if rest.startswith('"') and rest.endswith('"'):
                rest = rest[1:-1]
            if rest:
                return rest
    return None


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--n", type=int, default=5, help="Number of trials")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=8192)
    args = p.parse_args()

    # Load data
    print("Loading data...", flush=True)
    chunks = load_chunks_local(CHUNKS_LOCAL)
    doc_lookup = build_doc_lookup(chunks)
    secrets = load_secrets(SECRETS)
    eligible = filter_seed_secrets(secrets, doc_lookup)
    print(f"  {len(eligible)} seeds, {len(doc_lookup)} local docs")

    print("Building web BM25 index...", flush=True)
    web_searcher = BM25Searcher(CHUNKS_WEB)
    print(f"  {web_searcher.index.size} web chunks")

    llm = LLMClient(model=args.model, base_url=args.base_url)
    rng = random.Random(args.seed)

    # Pre-scan: find (seed, web_doc) pairs where the answer appears in the web doc
    print("\nPre-scanning for answer-in-doc matches...", flush=True)
    web_chunks = []
    with open(CHUNKS_WEB) as f:
        for line in f:
            web_chunks.append(json.loads(line))
    print(f"  {len(web_chunks)} web chunks loaded")

    # Classify answers as entity vs numeric
    def is_entity_answer(ans: str) -> bool:
        """True if answer is a named entity (not just a number/percentage)."""
        ans = ans.strip()
        if re.match(r'^[\d.,\s%$€£]+$', ans):
            return False
        if re.match(r'^Q[1-4]\s+\d{4}$', ans):
            return False
        if re.match(r'^\d{4}$', ans):
            return False
        return any(len(w) >= 3 and w.isalpha() for w in ans.split())

    # Find good pairs: answer appears in web doc text
    entity_pairs = []
    numeric_pairs = []
    for secret in eligible:
        ans = secret.answer.strip().lower()
        if len(ans) < 3:
            continue
        doc = doc_lookup.get(secret.doc_id)
        if not doc:
            continue
        is_ent = is_entity_answer(secret.answer)
        for wc in web_chunks:
            wtext = (wc.get("text") or "").lower()
            if ans in wtext:
                pair = (secret, doc, wc)
                if is_ent:
                    entity_pairs.append(pair)
                else:
                    numeric_pairs.append(pair)

    rng.shuffle(entity_pairs)
    rng.shuffle(numeric_pairs)

    # Deduplicate
    def dedup(pairs):
        seen = set()
        out = []
        for s, d, w in pairs:
            key = (s.question, w.get("doc_id", id(w)))
            if key not in seen:
                seen.add(key)
                out.append((s, d, w))
        return out

    entity_pairs = dedup(entity_pairs)
    numeric_pairs = dedup(numeric_pairs)
    print(f"  {len(entity_pairs)} entity pairs, {len(numeric_pairs)} numeric pairs")

    good_pairs = entity_pairs + numeric_pairs
    print(f"  {len(good_pairs)} total pairs\n")

    stats = {
        "total": 0, "parsed": 0, "no_bridge": 0, "bridge": 0,
        "good_bridge": 0, "bridge_issues": 0,
        "blend_ok": 0, "blend_fail": 0,
        "final_good": 0, "final_issues": 0,
    }

    n = min(args.n, len(good_pairs))
    for trial in range(n):
        print(f"\n{'='*80}")
        print(f"TRIAL {trial+1}/{n}")
        print(f"{'='*80}")

        secret, doc, wc = good_pairs[trial]

        print(f"  Seed Q: {secret.question}")
        print(f"  Seed A: {secret.answer}")
        print(f"  Company: {doc.meta.get('company_name', '?')}")
        print(f"  Doc: {secret.doc_id} ({len(doc.text)} chars)")

        candidate_doc_text = (wc.get("text") or "")[:DOC_LIMIT]
        hit_id = wc.get("doc_id", "?")
        print(f"  Web doc: {hit_id} ({len(candidate_doc_text)} chars)")
        print(f"  Web snippet: {candidate_doc_text[:150].replace(chr(10), ' ')}...")
        # Show where the answer appears in the web doc
        ans_lower = secret.answer.strip().lower()
        wtext_lower = candidate_doc_text.lower()
        idx = wtext_lower.find(ans_lower)
        if idx >= 0:
            context_start = max(0, idx - 40)
            context_end = min(len(candidate_doc_text), idx + len(ans_lower) + 40)
            print(f"  Answer in doc: ...{candidate_doc_text[context_start:context_end]}...")

        # ---- PROMPT 1: Bridge Finding ----
        prompt1 = PROMPT_1_BRIDGE.format(
            global_question=secret.question,
            global_answer=secret.answer,
            candidate_doc_text=candidate_doc_text,
        )

        t0 = time.time()
        raw1 = llm.chat(
            [{"role": "user", "content": prompt1}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        t1 = time.time()

        stats["total"] += 1

        # Show P1 output
        print(f"\n  P1 ({t1-t0:.1f}s):")
        m = _ANSWER_RE.search(raw1)
        if m:
            print(f"  <answer>{m.group(1).strip()[:300]}</answer>")
        else:
            print(f"  {raw1[:300]}")

        result = parse_bridge(raw1)
        if result is None:
            print("  P1 PARSE FAILED")
            continue

        stats["parsed"] += 1
        if result.get("no_bridge"):
            stats["no_bridge"] += 1
            print(f"  NO_BRIDGE: {result['reason'][:100]}")
            continue

        stats["bridge"] += 1
        direct_q = result["question"]
        direct_a = result["answer"]
        print(f"\n  P1 >> DIRECT Q: {direct_q}")
        print(f"  P1 >> ANSWER:   {direct_a}")
        print(f"  P1 >> JUSTIFICATION: {result['justification'][:150]}")

        # P1 quality checks
        p1_issues = []
        prev_a = secret.answer.strip().lower()
        if len(direct_a.split()) > 5:
            p1_issues.append(f"ANSWER_TOO_LONG ({len(direct_a.split())} words)")
        if direct_a.strip().lower() == prev_a or prev_a in direct_a.strip().lower() or direct_a.strip().lower() in prev_a:
            p1_issues.append(f"ANSWER_EQ_PREV ('{direct_a}' vs '{secret.answer}')")
        dq_lower = direct_q.lower()
        ref_patterns = ["according to the", "the document", "the article", "the report",
                        "the study", "the outlook", "the guide", "mentioned in the"]
        if any(p in dq_lower for p in ref_patterns):
            p1_issues.append("REFERENCES_DOC")

        if p1_issues:
            stats["bridge_issues"] += 1
            print(f"  P1 !! ISSUES: {', '.join(p1_issues)}")
            continue  # Don't bother blending a bad bridge
        else:
            stats["good_bridge"] += 1
            print(f"  P1 OK")

        # ---- PROMPT 2: Hierarchical Blending ----
        chain_desc = f'Q1: "{secret.question}" → A1: "{secret.answer}"'
        prompt2 = PROMPT_2_BLEND.format(
            chain_description=chain_desc,
            direct_question=direct_q,
            direct_answer=direct_a,
        )

        t2 = time.time()
        raw2 = llm.chat(
            [{"role": "user", "content": "/no_think\n" + prompt2}],
            temperature=0.3,  # Lower temp for rewriting
            max_tokens=2048,  # Short task but allow room for <think> blocks
        )
        t3 = time.time()

        print(f"\n  P2 ({t3-t2:.1f}s):")
        m2 = _ANSWER_RE.search(raw2)
        if m2:
            print(f"  <answer>{m2.group(1).strip()[:300]}</answer>")
        else:
            print(f"  {raw2[:300]}")

        blended = parse_blend(raw2)
        if blended is None:
            stats["blend_fail"] += 1
            print("  P2 PARSE FAILED")
            continue

        stats["blend_ok"] += 1
        print(f"\n  P2 >> BLENDED Q: {blended}")
        print(f"  P2 >> ANSWER:    {direct_a}")

        # Final quality checks on the blended question
        final_issues = []
        bq = blended.lower()

        if prev_a in bq:
            final_issues.append(f"PREV_ANSWER_IN_Q ('{prev_a}')")
        if "according to the" in bq or "the new document" in bq or "the document" in bq or "the article" in bq:
            final_issues.append("REFERENCES_DOC")
        # Check it's actually different from the direct question (did blending happen?)
        if blended.lower().strip().rstrip("?") == direct_q.lower().strip().rstrip("?"):
            final_issues.append("NO_CHANGE (identical to direct)")

        if final_issues:
            stats["final_issues"] += 1
            print(f"  !! FINAL ISSUES: {', '.join(final_issues)}")
        else:
            stats["final_good"] += 1
            print(f"  ** FINAL GOOD: {blended}")

    # Summary
    print(f"\n{'='*80}")
    print(f"PROMPT 1 (Bridge): {stats['total']} trials, {stats['parsed']} parsed, "
          f"{stats['bridge']} bridges ({stats['good_bridge']} good, {stats['bridge_issues']} issues), "
          f"{stats['no_bridge']} no_bridge")
    print(f"PROMPT 2 (Blend):  {stats['blend_ok']} blended, {stats['blend_fail']} parse fails")
    print(f"FINAL:  {stats['final_good']} good / {stats['blend_ok']} blended "
          f"({100*stats['final_good']/max(1,stats['blend_ok']):.0f}%), "
          f"{stats['final_issues']} with issues")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
