#!/usr/bin/env python3
"""Three-hop chain test: find valid chains through web documents.

Finds chains: Q1→A1 → Q_hop1→B → Q_hop2→E → Q3→A3
where each answer feeds into the next question.

Data prep:
  1. Pick privacy Q1 with entity answer A1
  2. Find web doc D2 (same task) containing A1
  3. Extract entities from D2 with spaCy
  4. Find entity E in D2 that appears in another privacy question Q3's text
  5. Model generates Q_hop1 (A1→B) and Q_hop2 (B→E)
  6. Q3 is verbatim

Usage:
    /home/toolkit/.mamba/envs/vllm013/bin/python -m making_dataset_2.eval.chain_test \
        --model MiniMax-M2.1 \
        --base-url http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-m21-vllm:8000/v1 \
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

import spacy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from making_dataset_2.data_loading import (
    build_doc_lookup, filter_seed_secrets, load_chunks_local, load_secrets,
)
from making_dataset_2.llm import LLMClient

# ============================================================================
# Prompts
# ============================================================================

PROMPT_HOP1 = """\
"{prev_a}" appears in this document. Write a question containing "{prev_a}" about a different fact.
Answer must be 1-5 words (entity, date, or number). No source references.

Example:
"Salesforce" appears in a document about healthcare platforms.
QUESTION: What healthcare platform does Salesforce offer?
ANSWER: Health Cloud

<document>
{doc_text}
</document>

<answer>
QUESTION: ...
ANSWER: ...
</answer>"""

PROMPT_HOP2 = """\
Write a question containing "{prev_a}" whose answer is "{target_entity}", based on this document.
Answer must be 1-5 words. No source references.

Example:
Write a question containing "Health Cloud" whose answer is "HIPAA".
QUESTION: What regulation must Health Cloud comply with?
ANSWER: HIPAA

<document>
{doc_text}
</document>

<answer>
QUESTION: ...
ANSWER: {target_entity}
</answer>"""

# ============================================================================
# Paths & constants
# ============================================================================

CHUNKS_LOCAL = ROOT / "making_dataset" / "outputs" / "chunks_local.jsonl"
CHUNKS_WEB = ROOT / "making_dataset_2" / "outputs" / "chunks_web_drbench_urls.jsonl"
SECRETS = ROOT / "making_dataset_2" / "outputs" / "secret_inventory.jsonl"
DOC_LIMIT = 8000

_ANS = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip(t):
    return _THINK.sub("", t).strip()


def parse_qa(text):
    text = _strip(text)

    def kv(block, k):
        for ln in block.splitlines():
            s = ln.strip()
            if s.upper().startswith(k.upper() + ":"):
                return s[len(k) + 1:].strip().strip('"')
        return ""

    # Try answer tags first
    ms = _ANS.findall(text)
    if ms:
        b = ms[-1].strip()
        q, a = kv(b, "QUESTION"), kv(b, "ANSWER")
        if q and a:
            return q, a

    # Fallback: raw text
    q, a = kv(text, "QUESTION"), kv(text, "ANSWER")
    if q and a:
        return q, a
    return None


def is_entity(ans: str) -> bool:
    ans = ans.strip()
    if "@" in ans:
        return False
    if re.match(r"^[\d.,\s%$]+$", ans):
        return False
    if re.match(r"^Q[1-4]\s+\d{4}$", ans):
        return False
    if re.match(r"^\d{4}$", ans):
        return False
    return any(len(w) >= 3 and w.isalpha() for w in ans.split())


def extract_entities(nlp, text: str) -> list[str]:
    """Extract unique entity texts from document using spaCy."""
    doc = nlp(text[:100000])  # spaCy limit
    seen = set()
    entities = []
    for ent in doc.ents:
        key = ent.text.strip().lower()
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        entities.append(ent.text.strip())
    return entities


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-only", action="store_true", help="Just show valid tuples, no model calls")
    args = p.parse_args()

    # Load spaCy
    print("Loading spaCy en_core_web_lg...", flush=True)
    nlp = spacy.load("en_core_web_lg")

    # Load data
    print("Loading data...", flush=True)
    chunks = load_chunks_local(CHUNKS_LOCAL)
    doc_lookup = build_doc_lookup(chunks)
    secrets = load_secrets(SECRETS)
    eligible = filter_seed_secrets(secrets, doc_lookup)
    print(f"  {len(eligible)} secrets, {len(doc_lookup)} local docs")

    # Load web chunks by task
    web_by_task: dict[str, list[dict]] = {}
    with open(CHUNKS_WEB) as f:
        for line in f:
            c = json.loads(line)
            for tid in (c.get("meta") or {}).get("task_ids", []):
                web_by_task.setdefault(tid, []).append(c)
    print(f"  {sum(len(v) for v in web_by_task.values())} web chunks across {len(web_by_task)} tasks")

    # Index secrets by task for Q3 lookup
    secrets_by_task: dict[str, list] = {}
    for s in eligible:
        doc = doc_lookup.get(s.doc_id)
        if doc:
            tid = doc.meta.get("task_id")
            if tid:
                secrets_by_task.setdefault(tid, []).append(s)

    rng = random.Random(args.seed)

    # Find valid tuples: (Q1, A1, D1, D2_text, E, Q3, A3)
    print("\nFinding valid chain tuples...", flush=True)
    tuples = []
    seen = set()

    for secret in eligible:
        if not is_entity(secret.answer):
            continue
        a1 = secret.answer.strip()
        a1_lower = a1.lower()
        if len(a1_lower) < 3:
            continue

        doc = doc_lookup.get(secret.doc_id)
        if not doc:
            continue
        task_id = doc.meta.get("task_id")
        if not task_id or task_id not in web_by_task:
            continue
        company = doc.meta.get("company_name", "?")

        # Find web docs D2 containing A1
        for wc in web_by_task[task_id]:
            wtext = wc.get("text") or ""
            if a1_lower not in wtext.lower():
                continue
            wid = wc.get("doc_id", "")
            if wid == secret.doc_id:
                continue

            # Extract entities from D2
            ents = extract_entities(nlp, wtext)

            # Find entity E that appears in some Q3's text (same task, Q3 != Q1)
            for e in ents:
                e_lower = e.lower()
                if e_lower == a1_lower:
                    continue  # Skip if E == A1
                if len(e_lower) < 3:
                    continue

                for q3_secret in secrets_by_task.get(task_id, []):
                    if q3_secret.question == secret.question:
                        continue  # Q3 != Q1
                    if e_lower in q3_secret.question.lower():
                        key = (secret.question, e, q3_secret.question)
                        if key in seen:
                            continue
                        seen.add(key)
                        tuples.append({
                            "q1": secret.question,
                            "a1": a1,
                            "d2_text": wtext[:DOC_LIMIT],
                            "d2_id": wid,
                            "entity_e": e,
                            "q3": q3_secret.question,
                            "a3": q3_secret.answer,
                            "task_id": task_id,
                            "company": company,
                        })

    rng.shuffle(tuples)
    print(f"  {len(tuples)} valid tuples found")

    if args.data_only:
        for i, t in enumerate(tuples[:args.n]):
            print(f"\n{'='*70}")
            print(f"TUPLE {i+1}")
            print(f"  Task: {t['task_id']} | {t['company']}")
            print(f"  Q1: {t['q1']}")
            print(f"  A1: {t['a1']}")
            print(f"  D2: {t['d2_id'][:60]} ({len(t['d2_text'])} chars)")
            print(f"  Entity E: {t['entity_e']}")
            print(f"  Q3: {t['q3']}")
            print(f"  A3: {t['a3']}")
        return

    # Model generation
    llm = LLMClient(model=args.model, base_url=args.base_url)
    n = min(args.n, len(tuples))
    good = 0

    for i in range(n):
        t = tuples[i]
        print(f"\n{'='*70}")
        print(f"CHAIN {i+1}/{n}  [{t['task_id']} | {t['company']}]")
        print(f"  Q1: {t['q1']} → A1: {t['a1']}")
        print(f"  D2: {t['d2_id'][:60]}")
        print(f"  Target E: {t['entity_e']}")
        print(f"  Q3: {t['q3']} → A3: {t['a3']}")

        # Hop 1: A1 → B
        prompt1 = PROMPT_HOP1.format(prev_a=t["a1"], doc_text=t["d2_text"])
        t0 = time.time()
        raw1 = llm.chat([{"role": "user", "content": "/no_think\n" + prompt1}],
                        temperature=0.7, max_tokens=4096)
        dt1 = time.time() - t0

        r1 = parse_qa(raw1)
        if not r1:
            print(f"  Hop1 PARSE FAIL ({dt1:.1f}s)")
            print(f"  RAW: {_strip(raw1)[:300]}")
            continue
        q_hop1, b = r1
        print(f"\n  Hop1 ({dt1:.1f}s): {q_hop1} → {b}")

        # Check hop1 quality
        issues = []
        if t["a1"].lower() not in q_hop1.lower():
            issues.append("A1_NOT_IN_Q")
        if len(b.split()) > 5:
            issues.append("B_TOO_LONG")
        if b.strip().lower() == t["a1"].strip().lower():
            issues.append("B_EQ_A1")
        if issues:
            print(f"  !! Hop1: {', '.join(issues)}")
            continue

        # Hop 2: B → E
        prompt2 = PROMPT_HOP2.format(prev_a=b, target_entity=t["entity_e"], doc_text=t["d2_text"])
        t0 = time.time()
        raw2 = llm.chat([{"role": "user", "content": "/no_think\n" + prompt2}],
                        temperature=0.7, max_tokens=4096)
        dt2 = time.time() - t0

        r2 = parse_qa(raw2)
        if not r2:
            print(f"  Hop2 PARSE FAIL ({dt2:.1f}s)")
            print(f"  RAW: {_strip(raw2)[:300]}")
            continue
        q_hop2, e_out = r2
        print(f"  Hop2 ({dt2:.1f}s): {q_hop2} → {e_out}")

        # Check hop2 quality
        issues2 = []
        if b.lower() not in q_hop2.lower():
            issues2.append("B_NOT_IN_Q")
        if t["entity_e"].lower() not in e_out.lower():
            issues2.append("E_WRONG")
        if issues2:
            print(f"  !! Hop2: {', '.join(issues2)}")
            continue

        # Full chain
        good += 1
        print(f"\n  ** CHAIN OK:")
        print(f"     Q1:     {t['q1']}")
        print(f"     → A1:   {t['a1']}")
        print(f"     Q_hop1: {q_hop1}")
        print(f"     → B:    {b}")
        print(f"     Q_hop2: {q_hop2}")
        print(f"     → E:    {e_out}")
        print(f"     Q3:     {t['q3']}")
        print(f"     → A3:   {t['a3']}")

    print(f"\n{'='*70}")
    print(f"RESULT: {good}/{n} valid chains ({100*good/max(1,n):.0f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
