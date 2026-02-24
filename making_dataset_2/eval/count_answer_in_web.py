#!/usr/bin/env python3
"""Count how many privacy-relevant secrets have answers that appear in web docs."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from making_dataset_2.data_loading import (
    build_doc_lookup, filter_seed_secrets, load_chunks_local, load_eval_seeds, load_secrets,
)

CHUNKS_LOCAL = ROOT / "making_dataset" / "outputs" / "chunks_local.jsonl"
CHUNKS_WEB = ROOT / "making_dataset_2" / "outputs" / "chunks_web_drbench_urls.jsonl"
SECRETS = ROOT / "making_dataset_2" / "outputs" / "secret_inventory.jsonl"


def load_web_texts():
    """Load all web chunk texts, grouped by doc_id."""
    by_doc: dict[str, str] = {}
    with open(CHUNKS_WEB) as f:
        for line in f:
            c = json.loads(line)
            did = c.get("doc_id", "")
            text = c.get("text", "")
            by_doc[did] = by_doc.get(did, "") + "\n" + text
    return by_doc


def main():
    chunks = load_chunks_local(CHUNKS_LOCAL)
    doc_lookup = build_doc_lookup(chunks)
    web_texts = load_web_texts()
    all_web_text = "\n".join(web_texts.values()).lower()

    # --- Inventory secrets ---
    secrets = load_secrets(SECRETS)
    eligible = filter_seed_secrets(secrets, doc_lookup)

    print(f"Inventory secrets: {len(eligible)} eligible (after filter)")
    match_count = 0
    matches = []
    for s in eligible:
        ans = s.answer.strip().lower()
        if len(ans) < 3:
            continue
        if ans in all_web_text:
            match_count += 1
            matches.append(s)

    print(f"  Exact answer in web docs: {match_count}/{len(eligible)}")
    print()

    # Break down by type
    by_type: dict[str, list] = {}
    for s in matches:
        by_type.setdefault(s.secret_type, []).append(s)
    for st, ss in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"  {st}: {len(ss)}")
        for s in ss[:3]:
            print(f"    {s.answer[:60]:60s}  ({s.question[:60]})")
        if len(ss) > 3:
            print(f"    ... and {len(ss)-3} more")
    print()

    # --- Eval.json seeds ---
    eval_seeds = load_eval_seeds(doc_lookup)
    print(f"Eval.json seeds: {len(eval_seeds)} enterprise_fact entries")

    # Eval answers are long sentences - check both full and extracted short answers
    eval_match = 0
    eval_matches = []
    for s in eval_seeds:
        ans = s.answer.strip().lower()
        if len(ans) < 3:
            continue
        if ans in all_web_text:
            eval_match += 1
            eval_matches.append(("full", s))
            continue
        # Also try extracting key values (numbers, percentages, names)
        # Skip this for now - just report full matches

    print(f"  Full answer in web docs: {eval_match}/{len(eval_seeds)}")
    print()

    # --- Entity-only (is_entity from bridge_lab) ---
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

    entity_secrets = [s for s in eligible if is_entity(s.answer)]
    entity_match = [s for s in entity_secrets if s.answer.strip().lower() in all_web_text]
    print(f"Entity-type secrets: {len(entity_secrets)} (non-numeric)")
    print(f"  Entity answer in web docs: {len(entity_match)}/{len(entity_secrets)}")
    for s in entity_match:
        print(f"    \"{s.answer}\"  ({s.question[:70]})")
    print()

    # --- Summary ---
    print("=" * 60)
    print(f"Total inventory secrets with answer in web: {match_count}/{len(eligible)}")
    print(f"  of which entity-type: {len(entity_match)}/{len(entity_secrets)}")
    print(f"  of which numeric/other: {match_count - len(entity_match)}/{len(eligible) - len(entity_secrets)}")
    print(f"Eval.json with full answer in web: {eval_match}/{len(eval_seeds)}")


if __name__ == "__main__":
    main()
