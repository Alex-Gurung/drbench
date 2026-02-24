#!/usr/bin/env python3
"""Bridge lab: simple prompts, entity-based bridging, both directions.

Focuses on the EASY case: the previous answer (an entity) clearly appears
in the target document. Tests both local->web and web->local directions.

RESULTS (MiniMax-M2.1, 10 iterations):
  Iter | Seed | P1%  | E2E% | Key change
  -----+------+------+------+------------------------------------------
  1    | 42   | 20%  | 20%  | Baseline (too minimal prompt)
  2    | 42   | 40%  | 35%  | +/no_think, +lenient parser
  3    | 99   | 80%  | 50%  | +example in P1 (format compliance)
  4    | 123  | 90%  | 75%  | +explicit seed_a in P2 blend
  5    | 777  | 92%  | 76%  | +compound phrase fix in P2
  6    | 555  | 92%  | 76%  | +require seed in bridge Q, +REF_DOC check
  7    | 42   | 88%  | 88%  | +different domain in P2 examples (no copy)
  8    | 314  | 76%  | 76%  | verify consistency (different seed)
  9    | 42   | 100% | 100% | +BAD/GOOD no-reference examples in P1
  10   | 999  | 90%  | 87%  | verify consistency

  Average last 4 runs: ~88% end-to-end

KEY FINDINGS:
  - Simple prompts work: P1 ~250 words, P2 ~150 words
  - W>L outperforms L>W (local docs have richer structured data)
  - P2 very reliable (96-100%) after fixing example domain
  - Main failure mode: MiniMax think blocks consume tokens (~2% of P2)
  - Entity answers (chatbots, CRM, telehealth, ACC II) bridge reliably
  - Numeric/metric answers are the HARD case (see preliminary_lab.py)
  - REF_DOC eliminated by BAD/GOOD example pair in P1

Usage:
    PY=/home/toolkit/.mamba/envs/vllm013/bin/python

    # Both directions (default)
    $PY -m making_dataset_2.eval.bridge_lab \
        --model MiniMax-M2.1 \
        --base-url http://dns-4943305a-c17f-44b1-b767-9536529eb8bc-m21-vllm:8000/v1 \
        --n 15 --seed 42

    # One direction
    $PY -m making_dataset_2.eval.bridge_lab --direction local-to-web ...
    $PY -m making_dataset_2.eval.bridge_lab --direction web-to-local ...

    # Quick with Qwen3-30B
    $PY -m making_dataset_2.eval.bridge_lab \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
        --base-url http://127.0.0.1:8000/v1 --n 10
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
# PROMPT 1: Bridge finding (minimal)
# ============================================================================

PROMPT_1 = """\
The previous answer is "{prev_a}" (from: "{prev_q}").

Find it in the document below and write a question about a DIFFERENT fact from the same document. The question MUST mention "{prev_a}" by name.

If "{prev_a}" doesn't appear in the document, reply <answer>NO_BRIDGE</answer>.

IMPORTANT: Write the question as if no document exists. Do NOT say "according to the document/article/report/study" or "mentioned in the document" or reference any source.
BAD: "According to the report, what does chatbot technology enable?"
GOOD: "What does chatbot technology enable in retail?"

Example:
Previous: "Workday" (from: "What HR platform does Lee's use?")
<answer>
QUESTION: How many employees does the company that uses Workday have?
ANSWER: 3,200
</answer>

<document>
{doc_text}
</document>

<answer>
QUESTION: must mention "{prev_a}" — no source references
ANSWER: 1-5 words, different from "{prev_a}"
</answer>"""

# ============================================================================
# PROMPT 2: Blend (minimal)
# ============================================================================

PROMPT_2 = """\
Replace "{seed_a}" in this question with an indirect description from the chain.

Chain: {chain}
Question: "{question}"
Answer: "{answer}"

Examples:
Chain: Q1: "What database does Acme Corp use?" -> A1: "PostgreSQL"
Question: "When was PostgreSQL first released?"
REWRITTEN: "When was the database Acme Corp uses first released?"

Chain: Q1: "What framework does Nova Labs use?" -> A1: "Django"
Question: "What language are Django applications written in?"
REWRITTEN: "What language are applications using the framework Nova Labs uses written in?"
(Replace the whole phrase "Django applications" not just "Django")

The rewritten question must NOT contain "{seed_a}" or any variant. If "{seed_a}" is part of a compound phrase, replace the whole phrase.

<answer>REWRITTEN: ...</answer>"""

# ============================================================================
# Paths / config
# ============================================================================

CHUNKS_LOCAL = ROOT / "making_dataset" / "outputs" / "chunks_local.jsonl"
CHUNKS_WEB = ROOT / "making_dataset_2" / "outputs" / "chunks_web_drbench_urls.jsonl"
SECRETS = ROOT / "making_dataset_2" / "outputs" / "secret_inventory.jsonl"
DOC_LIMIT = 8000

# ============================================================================
# Parsers
# ============================================================================

_ANS = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip(t):
    return _THINK.sub("", t).strip()


def parse_bridge(text):
    text = _strip(text)

    # Check for NO_BRIDGE anywhere in text
    if "NO_BRIDGE" in text.upper():
        return {"no_bridge": True, "reason": "NO_BRIDGE"}

    def kv_from(block, k):
        for ln in block.splitlines():
            s = ln.strip()
            if s.upper().startswith(k.upper() + ":"):
                return s[len(k) + 1 :].strip().strip('"')
        return ""

    # Try <answer> tags first
    ms = _ANS.findall(text)
    if ms:
        b = ms[-1].strip()
        q, a = kv_from(b, "QUESTION"), kv_from(b, "ANSWER")
        if q and a:
            return {"question": q, "answer": a, "justification": kv_from(b, "JUSTIFICATION")}

    # Fallback: parse QUESTION/ANSWER lines from raw text
    q, a = kv_from(text, "QUESTION"), kv_from(text, "ANSWER")
    if q and a:
        return {"question": q, "answer": a, "justification": kv_from(text, "JUSTIFICATION")}
    return None


def parse_blend(text):
    text = _strip(text)
    ms = _ANS.findall(text)
    if ms:
        b = ms[-1].strip()
        for ln in b.splitlines():
            s = ln.strip()
            if s.upper().startswith("REWRITTEN:"):
                r = s[len("REWRITTEN:") :].strip().strip('"')
                if r:
                    return r
        if "?" in b:
            return b.strip().strip('"')
    for ln in text.splitlines():
        s = ln.strip()
        if s.upper().startswith("REWRITTEN:"):
            r = s[len("REWRITTEN:") :].strip().strip('"')
            if r:
                return r
    return None


def is_entity(ans):
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


# ============================================================================
# Main
# ============================================================================


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--direction",
        choices=["local-to-web", "web-to-local", "both"],
        default="both",
    )
    p.add_argument("--skip-blend", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--output", default=None, help="Save good results to JSONL file")
    args = p.parse_args()

    print("Loading data...", flush=True)
    chunks = load_chunks_local(CHUNKS_LOCAL)
    doc_lookup = build_doc_lookup(chunks)
    secrets = load_secrets(SECRETS)
    eligible = filter_seed_secrets(secrets, doc_lookup)

    # Load web chunks by task
    web_by_task: dict[str, list[dict]] = {}
    with open(CHUNKS_WEB) as f:
        for line in f:
            c = json.loads(line)
            for tid in (c.get("meta") or {}).get("task_ids", []):
                web_by_task.setdefault(tid, []).append(c)

    rng = random.Random(args.seed)
    llm = LLMClient(model=args.model, base_url=args.base_url)

    # Build pairs: entity answers appearing in same-task web docs
    l2w = []  # (seed_q, seed_a, target_text, label, company, task)
    w2l = []
    for secret in eligible:
        if not is_entity(secret.answer):
            continue
        ans_lower = secret.answer.strip().lower()
        if len(ans_lower) < 3:
            continue
        doc = doc_lookup.get(secret.doc_id)
        if not doc:
            continue
        task_id = doc.meta.get("task_id")
        if not task_id or task_id not in web_by_task:
            continue

        for wc in web_by_task[task_id]:
            wtext = wc.get("text") or ""
            if ans_lower not in wtext.lower():
                continue
            company = doc.meta.get("company_name", "?")
            # local->web: seed from local, target = web doc
            l2w.append((secret.question, secret.answer, wtext[:DOC_LIMIT], "L>W", company, task_id))
            # web->local: same entity, target = local doc
            w2l.append((secret.question, secret.answer, doc.text[:DOC_LIMIT], "W>L", company, task_id))

    # Deduplicate
    def dedup(pairs):
        seen = set()
        out = []
        for p in pairs:
            key = (p[0], p[2][:200])
            if key not in seen:
                seen.add(key)
                out.append(p)
        return out

    l2w = dedup(l2w)
    w2l = dedup(w2l)
    rng.shuffle(l2w)
    rng.shuffle(w2l)

    print(f"  {len(l2w)} L>W pairs, {len(w2l)} W>L pairs")

    # Select based on direction
    if args.direction == "local-to-web":
        pairs = l2w
    elif args.direction == "web-to-local":
        pairs = w2l
    else:
        pairs = []
        for i in range(max(len(l2w), len(w2l))):
            if i < len(l2w):
                pairs.append(l2w[i])
            if i < len(w2l):
                pairs.append(w2l[i])

    stats = {"total": 0, "bridge": 0, "good": 0, "blend_ok": 0, "final_good": 0}
    good_results: list[dict] = []
    n = min(args.n, len(pairs))

    for i in range(n):
        seed_q, seed_a, target, label, company, task = pairs[i]
        print(f"\n{'='*70}")
        print(f"TRIAL {i+1}/{n} [{label}] {company} ({task})")
        print(f"  Seed: {seed_q}")
        print(f"  Seed A: {seed_a}")
        print(f"  Target: {len(target)} chars")

        prompt = PROMPT_1.format(prev_q=seed_q, prev_a=seed_a, doc_text=target)
        stats["total"] += 1
        t0 = time.time()
        raw = llm.chat(
            [{"role": "user", "content": "/no_think\n" + prompt}],
            temperature=args.temperature,
            max_tokens=4096,
        )
        dt = time.time() - t0

        m = _ANS.search(raw)
        ans_block = m.group(1).strip()[:300] if m else raw[:200]
        print(f"  P1 ({dt:.1f}s): {ans_block}")

        r = parse_bridge(raw)
        if not r:
            print("  PARSE FAIL")
            continue
        if r.get("no_bridge"):
            print(f"  NO_BRIDGE")
            continue

        stats["bridge"] += 1
        q, a = r["question"], r["answer"]
        print(f"  >> Q: {q}")
        print(f"  >> A: {a}")

        # Quality checks
        issues = []
        al = a.strip().lower()
        sal = seed_a.strip().lower()
        ql = q.lower()
        if al == sal or sal in al or al in sal:
            issues.append("SAME_AS_SEED")
        if len(a.split()) > 5:
            issues.append("TOO_LONG")
        if any(p in ql for p in ["the document", "the article", "the report", "according to"]):
            issues.append("REF_DOC")
        # Seed answer must appear in bridge question (needed for P2 blending)
        if sal not in ql:
            variants = [sal]
            if sal.endswith("s"):
                variants.append(sal[:-1])
            else:
                variants.append(sal + "s")
            if not any(v in ql for v in variants):
                issues.append(f"SEED_NOT_IN_Q")

        if issues:
            print(f"  !! {', '.join(issues)}")
            continue
        stats["good"] += 1
        print(f"  P1 OK")

        if args.skip_blend:
            continue

        # P2: Blend
        chain = f'Q1: "{seed_q}" -> A1: "{seed_a}"'
        p2 = PROMPT_2.format(chain=chain, question=q, answer=a, seed_a=seed_a)
        t0 = time.time()
        raw2 = llm.chat(
            [{"role": "user", "content": "/no_think\n" + p2}],
            temperature=0.3,
            max_tokens=8192,
        )
        dt2 = time.time() - t0

        blended = parse_blend(raw2)
        if not blended:
            print(f"  P2 FAIL ({dt2:.1f}s)")
            continue
        stats["blend_ok"] += 1

        bl = blended.lower()
        issues2 = []
        if sal in bl and len(sal) > 2:
            issues2.append(f"SEED_IN_Q({sal})")
        if "the document" in bl or "according to" in bl:
            issues2.append("REF_DOC")
        if bl.rstrip("?") == ql.rstrip("?"):
            issues2.append("NO_CHANGE")
        # Check word overlap: blended Q should share content words with bridge Q
        stop = {"the", "a", "an", "is", "was", "are", "were", "of", "in", "for",
                "to", "and", "or", "that", "which", "what", "how", "when", "where",
                "does", "do", "did", "by", "with", "at", "on", "from", "its", "it"}
        bridge_words = {w for w in re.findall(r'\w+', ql) if w not in stop and len(w) > 2} - {sal}
        blend_words = {w for w in re.findall(r'\w+', bl) if w not in stop and len(w) > 2}
        if bridge_words:
            overlap = len(bridge_words & blend_words) / len(bridge_words)
            if overlap < 0.3:
                issues2.append(f"LOW_OVERLAP({overlap:.0%})")

        if issues2:
            print(f"  P2: {blended}")
            print(f"  !! {', '.join(issues2)}")
        else:
            stats["final_good"] += 1
            print(f"  ** GOOD: {blended} -> {a}")
            good_results.append({
                "seed_question": seed_q,
                "seed_answer": seed_a,
                "company": company,
                "task_id": task,
                "direction": label,
                "bridge_question": q,
                "bridge_answer": a,
                "blended_question": blended,
                "target_doc_snippet": target[:200],
            })

    print(f"\n{'='*70}")
    print(f"P1: {stats['total']} total, {stats['bridge']} bridges, {stats['good']} good")
    if not args.skip_blend:
        print(f"P2: {stats['blend_ok']} blended, {stats['final_good']} final good")
    print(
        f"RATE: {100*stats['final_good']/max(1,stats['total']):.0f}% end-to-end"
        f" ({100*stats['good']/max(1,stats['total']):.0f}% P1 pass)"
    )
    print(f"{'='*70}")

    if args.output and good_results:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r in good_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(good_results)} good results to {out}")


if __name__ == "__main__":
    main()
