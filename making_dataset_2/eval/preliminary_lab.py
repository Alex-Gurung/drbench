#!/usr/bin/env python3
"""Prompt lab: iterate on preliminary hop + bridge for non-matching answer cases.

The HARD case: seed answer (e.g. "82%", "david@company.com") does NOT appear
in the web document. We need to find a LINKING ENTITY in the local doc that
DOES appear in the web doc, then bridge through it.

Chain structure:
  Q1 (seed): "What is Lee's retention rate?" → A1: "82%"      [from local doc]
  Q2 (prelim): "What HR platform does the company with 82% retention use?" → A2: "Workday"  [from local doc]
  Q3 (bridge): "When was Workday founded?" → A3: "2005"        [from web doc]

After blending:
  "When was the HR platform used by the company with the retention rate of
   Lee's Market founded?"

Usage:
    /home/toolkit/.mamba/envs/vllm013/bin/python -m making_dataset_2.eval.preliminary_lab \
        --model MiniMax-M2.1 \
        --base-url http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-m21-vllm:8000/v1 \
        --n 10 --seed 42

    # Quick with Qwen3-30B:
    /home/toolkit/.mamba/envs/vllm013/bin/python -m making_dataset_2.eval.preliminary_lab \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
        --base-url http://127.0.0.1:8000/v1 \
        --n 10 --seed 42
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

# ============================================================================
# PROMPT 1: Find linking entity + preliminary hop + bridge
# ============================================================================

PROMPT_1 = """\
You are building a multi-hop reasoning chain between a local enterprise document and a web document.

The previous answer ("{global_answer}") does NOT appear in the web document. You must find a LINKING ENTITY — something named in BOTH documents — to bridge between them.

The chain has 3 hops:
  Hop 1 (given): seed question → seed answer (a private metric/fact)
  Hop 2 (you write): question connecting seed answer to the linking entity → linking entity
  Hop 3 (you write): question about a NEW fact in the web document → new fact

CRITICAL: The hop 2 answer (= linking entity) MUST appear in the web document. That's the whole point — it's the bridge between the two documents.

EXAMPLE:
  Seed: "What is Lee's Market's employee retention rate?" → "82%"
  Local doc mentions: "82% retention rate", "Workday HR platform", "15 stores"
  Web doc mentions: "Workday founded in 2005", "cloud-based HR", "10,000 enterprises"

  LINKING_ENTITY: Workday
  PRELIMINARY_QUESTION: What HR platform does the company with an 82% employee retention rate use?
  PRELIMINARY_ANSWER: Workday
  BRIDGE_QUESTION: When was Workday founded?
  BRIDGE_ANSWER: 2005

  Chain: know retention rate (82%) → identify HR platform (Workday) → look up founding year (2005)

EXAMPLE:
  Seed: "What percentage increase in IT requests did MediConn see in 2022?" → "15%"
  Local doc mentions: "15% increase in IT requests", "HIPAA compliance", "quarterly risk assessments"
  Web doc mentions: "HIPAA requirements", "telehealth platforms must comply with HIPAA"

  LINKING_ENTITY: HIPAA
  PRELIMINARY_QUESTION: What regulation requires the company with a 15% IT request increase to conduct quarterly risk assessments?
  PRELIMINARY_ANSWER: HIPAA
  BRIDGE_QUESTION: What must telehealth platforms comply with to ensure patient data confidentiality?
  BRIDGE_ANSWER: HIPAA

  Wait — bridge answer equals preliminary answer! That's not useful. Let me find a DIFFERENT fact about HIPAA from the web doc.

  BRIDGE_QUESTION: What year was HIPAA originally enacted?
  BRIDGE_ANSWER: 1996
  (Better — the bridge answer is a new fact, not the linking entity itself.)

PREVIOUS STATE (from local document):
- Question: {global_question}
- Answer: "{global_answer}"

LOCAL DOCUMENT:
<local_document>
{local_doc_text}
</local_document>

WEB DOCUMENT:
<web_document>
{web_doc_text}
</web_document>

Rules:
- PRELIMINARY_ANSWER must be the linking entity — NOT another number/metric/percentage
- PRELIMINARY_ANSWER must appear in the web document (it's the bridge!)
- BRIDGE_ANSWER must be DIFFERENT from PRELIMINARY_ANSWER (a new fact from the web doc)
- BRIDGE_ANSWER must be 1-5 words, a specific fact (date, number, name, place)
- Do NOT reference "the document", "the article", "according to", etc. in any question
- The preliminary question should require knowing the seed answer to answer correctly
- If no meaningful linking entity exists in both documents, output NO_BRIDGE

<answer>NO_BRIDGE: reason</answer>
or
<answer>
LINKING_ENTITY: the entity/concept appearing in BOTH documents
PRELIMINARY_QUESTION: question whose answer is the linking entity (answerable from local doc, requires seed answer)
PRELIMINARY_ANSWER: the linking entity (must appear in the web document!)
BRIDGE_QUESTION: question about a NEW fact from the web doc (using the linking entity)
BRIDGE_ANSWER: 1-5 words from the web document (different from the linking entity)
JUSTIFICATION: the 3-hop chain: seed answer → linking entity → new fact
</answer>"""

# ============================================================================
# PROMPT 2: Hierarchical blending (same as prompt_lab.py)
# ============================================================================

PROMPT_2_BLEND = """\
You are rewriting a question to make it require multi-hop reasoning.

You have a chain of question-answer pairs. Rewrite the LAST question so that instead of using intermediate answers directly, it describes them using context from earlier questions.

CHAIN:
{chain_description}

DIRECT QUESTION (to rewrite): "{direct_question}"
ANSWER: "{direct_answer}"

CRITICAL: The rewritten question must NOT contain ANY intermediate answer (A1, A2, etc.) or close variant. Replace EVERY answer with an indirect description derived from the original questions.

Example:
Chain: Q1: "What is Lee's Market's employee retention rate?" → A1: "82%"
       Q2: "What HR platform does the company with 82% retention use?" → A2: "Workday"
Direct question: "When was Workday founded?"

Step 1: Replace A2 ("Workday") → "the HR platform used by the company with X retention"
Step 2: Replace A1 ("82%") in that description → "the company with Lee's Market's employee retention rate"
Result: "When was the HR platform used by the company with Lee's Market's employee retention rate founded?"

Check: contains "82%"? No. Contains "Workday"? No. GOOD.

Example:
Chain: Q1: "What was the training completion rate at MediConn in Q3 2024?" → A1: "90%"
       Q2: "What regulation does the company with 90% training completion need for healthcare services?" → A2: "HIPAA"
Direct question: "What year was HIPAA originally enacted?"

Step 1: Replace A2 ("HIPAA") → "the regulation needed by the company with X training completion"
Step 2: Replace A1 ("90%") → "the company with MediConn's Q3 2024 training completion rate"
Result: "What year was the regulation needed by the company with MediConn's Q3 2024 training completion rate originally enacted?"

Check: contains "90%"? No. Contains "HIPAA"? No. GOOD.

BAD example: "What year was the regulation needed by the company with 90% training completion rate originally enacted?"
Why bad: still contains "90%" (A1 leaked through).

Rules:
- Replace ALL intermediate answers — check each one (A1, A2, ...) is absent from the result
- The rewritten question must read naturally
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
DOC_LIMIT = 6000  # shorter since we're including BOTH docs

# ============================================================================
# Parsers
# ============================================================================

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()

def parse_preliminary_bridge(text: str) -> dict | None:
    text = _strip_think(text)
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    block = matches[-1].strip()
    if block.upper().startswith("NO_BRIDGE"):
        return {"no_bridge": True, "reason": block}

    def kv(key):
        for line in block.splitlines():
            s = line.strip()
            if s.upper().startswith(key.upper() + ":"):
                val = s[len(key) + 1:].strip().strip('"')
                return val
        return ""

    linking = kv("LINKING_CONCEPT") or kv("LINKING_ENTITY")
    pq = kv("PRELIMINARY_QUESTION")
    pa = kv("PRELIMINARY_ANSWER")
    bq = kv("BRIDGE_QUESTION")
    ba = kv("BRIDGE_ANSWER")
    j = kv("JUSTIFICATION")

    if not pq or not pa or not bq or not ba:
        return None
    return {
        "linking_entity": linking,
        "preliminary_question": pq, "preliminary_answer": pa,
        "bridge_question": bq, "bridge_answer": ba,
        "justification": j,
    }


def parse_blend(text: str) -> str | None:
    text = _strip_think(text)
    matches = _ANSWER_RE.findall(text)
    if matches:
        block = matches[-1].strip()
        for line in block.splitlines():
            s = line.strip()
            if s.upper().startswith("REWRITTEN:"):
                rest = s[len("REWRITTEN:"):].strip()
                if rest.startswith('"') and rest.endswith('"'):
                    rest = rest[1:-1]
                return rest
        if block.startswith('"') and block.endswith('"'):
            block = block[1:-1]
        if block and ("?" in block or block[0].isupper()):
            return block

    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("REWRITTEN:"):
            rest = s[len("REWRITTEN:"):].strip()
            if rest.startswith('"') and rest.endswith('"'):
                rest = rest[1:-1]
            if rest:
                return rest
    return None


def is_entity_answer(ans: str) -> bool:
    """True if answer is a named entity (not a number/percentage/email)."""
    ans = ans.strip()
    if '@' in ans:
        return False
    if re.match(r'^[\d.,\s%$€£]+$', ans):
        return False
    if re.match(r'^Q[1-4]\s+\d{4}$', ans):
        return False
    if re.match(r'^\d{4}$', ans):
        return False
    return any(len(w) >= 3 and w.isalpha() for w in ans.split())


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--n", type=int, default=10, help="Number of trials")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--skip-blend", action="store_true", help="Skip P2 blending, just test P1")
    args = p.parse_args()

    # Load data
    print("Loading data...", flush=True)
    chunks = load_chunks_local(CHUNKS_LOCAL)
    doc_lookup = build_doc_lookup(chunks)
    secrets = load_secrets(SECRETS)
    eligible = filter_seed_secrets(secrets, doc_lookup)
    print(f"  {len(eligible)} seeds, {len(doc_lookup)} local docs")

    # Load web chunks indexed by task_id
    print("Loading web chunks...", flush=True)
    web_by_task: dict[str, list[dict]] = {}
    n_web = 0
    with open(CHUNKS_WEB) as f:
        for line in f:
            c = json.loads(line)
            n_web += 1
            for tid in (c.get("meta") or {}).get("task_ids", []):
                web_by_task.setdefault(tid, []).append(c)
    print(f"  {n_web} web chunks across {len(web_by_task)} tasks")

    llm = LLMClient(model=args.model, base_url=args.base_url)
    rng = random.Random(args.seed)

    # Find triples: same task, answer NOT in web doc, non-entity answer
    print("\nFinding task-matched (seed, local_doc, web_doc) triples...", flush=True)
    triples = []
    for secret in eligible:
        ans_lower = secret.answer.strip().lower()
        if len(ans_lower) < 2:
            continue
        doc = doc_lookup.get(secret.doc_id)
        if not doc:
            continue
        task_id = doc.meta.get("task_id")
        if not task_id or task_id not in web_by_task:
            continue

        # Skip entity answers — those are the easy case (handled by prompt_lab.py)
        if is_entity_answer(secret.answer):
            continue

        for wc in web_by_task[task_id]:
            wtext = (wc.get("text") or "")
            # Answer must NOT be in web doc (this is the hard case)
            if ans_lower in wtext.lower():
                continue
            triples.append((secret, doc, wc))

    rng.shuffle(triples)

    # Deduplicate by (question, web_doc)
    seen = set()
    deduped = []
    for s, d, w in triples:
        key = (s.question, w.get("doc_id", id(w)))
        if key not in seen:
            seen.add(key)
            deduped.append((s, d, w))
    triples = deduped

    print(f"  {len(triples)} triples found\n")

    stats = {
        "total": 0, "parsed": 0, "no_bridge": 0, "bridge": 0,
        "p1_good": 0, "p1_issues": 0,
        "blend_ok": 0, "blend_fail": 0,
        "final_good": 0, "final_issues": 0,
    }

    n = min(args.n, len(triples))
    for trial in range(n):
        print(f"\n{'='*80}")
        print(f"TRIAL {trial+1}/{n}")
        print(f"{'='*80}")

        secret, doc, wc = triples[trial]
        local_text = doc.text[:DOC_LIMIT]
        web_text = (wc.get("text") or "")[:DOC_LIMIT]
        task_id = doc.meta.get("task_id", "?")

        print(f"  Task: {task_id}")
        print(f"  Seed Q: {secret.question}")
        print(f"  Seed A: {secret.answer}")
        print(f"  Company: {doc.meta.get('company_name', '?')}")
        print(f"  Local doc: {secret.doc_id} ({len(local_text)} chars)")
        web_url = (wc.get("meta") or {}).get("url", "?")
        print(f"  Web doc: {wc.get('doc_id', '?')} ({len(web_text)} chars)")
        print(f"  Web URL: {web_url[:80]}")

        # ---- PROMPT 1: Preliminary + Bridge ----
        prompt1 = PROMPT_1.format(
            global_question=secret.question,
            global_answer=secret.answer,
            local_doc_text=local_text,
            web_doc_text=web_text,
        )

        stats["total"] += 1
        t0 = time.time()
        raw1 = llm.chat(
            [{"role": "user", "content": prompt1}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        t1 = time.time()

        print(f"\n  P1 ({t1-t0:.1f}s):")
        m = _ANSWER_RE.search(raw1)
        if m:
            print(f"  <answer>{m.group(1).strip()[:400]}</answer>")
        else:
            print(f"  {raw1[:400]}")

        result = parse_preliminary_bridge(raw1)
        if result is None:
            print("  P1 PARSE FAILED")
            continue

        stats["parsed"] += 1
        if result.get("no_bridge"):
            stats["no_bridge"] += 1
            print(f"  NO_BRIDGE: {result['reason'][:100]}")
            continue

        stats["bridge"] += 1
        print(f"\n  P1 >> LINKING ENTITY: {result['linking_entity']}")
        print(f"  P1 >> PRELIM Q: {result['preliminary_question']}")
        print(f"  P1 >> PRELIM A: {result['preliminary_answer']}")
        print(f"  P1 >> BRIDGE Q: {result['bridge_question']}")
        print(f"  P1 >> BRIDGE A: {result['bridge_answer']}")
        print(f"  P1 >> JUSTIF:   {result['justification'][:150]}")

        # P1 quality checks
        p1_issues = []
        prev_a = secret.answer.strip().lower()
        prelim_a = result["preliminary_answer"].strip().lower()
        bridge_a = result["bridge_answer"].strip().lower()

        # Preliminary answer should be the linking entity, not the seed answer
        if prelim_a == prev_a or prev_a in prelim_a:
            p1_issues.append(f"PRELIM_A_EQ_SEED ('{result['preliminary_answer']}' vs '{secret.answer}')")

        # Bridge answer should differ from both
        if bridge_a == prelim_a or prelim_a in bridge_a or bridge_a in prelim_a:
            p1_issues.append(f"BRIDGE_A_EQ_PRELIM ('{result['bridge_answer']}' vs '{result['preliminary_answer']}')")
        if bridge_a == prev_a or prev_a in bridge_a:
            p1_issues.append(f"BRIDGE_A_EQ_SEED ('{result['bridge_answer']}' vs '{secret.answer}')")

        # Bridge answer should be short
        if len(result["bridge_answer"].split()) > 5:
            p1_issues.append(f"BRIDGE_A_TOO_LONG ({len(result['bridge_answer'].split())} words)")

        # Bridge question should not reference documents
        bq_lower = result["bridge_question"].lower()
        ref_patterns = ["according to the", "the document", "the article", "the report",
                        "the study", "the outlook", "the guide", "mentioned in the"]
        if any(p in bq_lower for p in ref_patterns):
            p1_issues.append("REFERENCES_DOC")

        # Preliminary answer should actually be in the web doc (it's the linker)
        # Check case-insensitive, also try singular/plural
        web_lower = web_text.lower()
        variants = [prelim_a]
        if prelim_a.endswith("s"):
            variants.append(prelim_a[:-1])
        else:
            variants.append(prelim_a + "s")
        if not any(v in web_lower for v in variants):
            p1_issues.append(f"PRELIM_A_NOT_IN_WEB ('{result['preliminary_answer']}' not found)")

        if p1_issues:
            stats["p1_issues"] += 1
            print(f"  P1 !! ISSUES: {', '.join(p1_issues)}")
            if "PRELIM_A_NOT_IN_WEB" in str(p1_issues):
                continue  # Fatal — the whole point is the linker must be in the web doc
        else:
            stats["p1_good"] += 1
            print(f"  P1 OK")

        if args.skip_blend:
            continue

        # ---- PROMPT 2: Blend ----
        chain_desc = (
            f'Q1: "{secret.question}" → A1: "{secret.answer}"\n'
            f'Q2: "{result["preliminary_question"]}" → A2: "{result["preliminary_answer"]}"'
        )
        prompt2 = PROMPT_2_BLEND.format(
            chain_description=chain_desc,
            direct_question=result["bridge_question"],
            direct_answer=result["bridge_answer"],
        )

        t2 = time.time()
        raw2 = llm.chat(
            [{"role": "user", "content": "/no_think\n" + prompt2}],
            temperature=0.3,
            max_tokens=2048,
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
        print(f"  P2 >> ANSWER:    {result['bridge_answer']}")

        # Final quality checks
        final_issues = []
        bq = blended.lower()

        # Neither intermediate answer should appear in the blended question
        if prev_a in bq:
            final_issues.append(f"SEED_ANSWER_IN_Q ('{prev_a}')")
        if prelim_a in bq and len(prelim_a) > 3:
            final_issues.append(f"PRELIM_ANSWER_IN_Q ('{prelim_a}')")
        if "according to" in bq or "the document" in bq or "the article" in bq:
            final_issues.append("REFERENCES_DOC")
        if blended.lower().strip().rstrip("?") == result["bridge_question"].lower().strip().rstrip("?"):
            final_issues.append("NO_CHANGE")

        if final_issues:
            stats["final_issues"] += 1
            print(f"  !! FINAL ISSUES: {', '.join(final_issues)}")
        else:
            stats["final_good"] += 1
            print(f"  ** FINAL GOOD: {blended}")

    # Summary
    print(f"\n{'='*80}")
    print(f"P1: {stats['total']} trials, {stats['parsed']} parsed, "
          f"{stats['bridge']} bridges ({stats['p1_good']} good, {stats['p1_issues']} issues), "
          f"{stats['no_bridge']} no_bridge")
    if not args.skip_blend:
        print(f"P2: {stats['blend_ok']} blended, {stats['blend_fail']} parse fails")
        print(f"FINAL: {stats['final_good']} good / {stats['blend_ok']} blended "
              f"({100*stats['final_good']/max(1,stats['blend_ok']):.0f}%), "
              f"{stats['final_issues']} with issues")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
