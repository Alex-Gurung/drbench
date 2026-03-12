#!/usr/bin/env python3
"""View extracted secrets from batch extraction.

Usage:
  python scripts/view_secrets.py                          # summary of all tasks
  python scripts/view_secrets.py --task DR0001             # secrets for one task
  python scripts/view_secrets.py --task DR0001 --full      # show source files too
  python scripts/view_secrets.py --search "revenue"        # search questions/answers
  python scripts/view_secrets.py --stats                   # per-task stats table
  python scripts/view_secrets.py --input path/to/secrets.jsonl  # custom input
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_PATH = Path("making_dataset_2/outputs/secrets_step35flash/extracted_secrets.jsonl")


def load(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def show_summary(rows: list[dict]):
    total_secrets = sum(len(r["secrets"]) for r in rows)
    tasks_with = sum(1 for r in rows if r["secrets"])
    print(f"{len(rows)} tasks, {tasks_with} with secrets, {total_secrets} total secrets\n")
    for r in sorted(rows, key=lambda x: x["task_id"]):
        n = len(r["secrets"])
        bar = "#" * min(n // 5, 40)
        print(f"  {r['task_id']}  {n:4d} secrets  {bar}")


def show_stats(rows: list[dict]):
    print(f"{'Task':<10} {'Files':>5} {'Secrets':>7} {'Per File':>8}")
    print("-" * 35)
    for r in sorted(rows, key=lambda x: x["task_id"]):
        n = len(r["secrets"])
        f = r["files_processed"]
        pf = f"{n/f:.1f}" if f else "–"
        print(f"{r['task_id']:<10} {f:>5} {n:>7} {pf:>8}")
    total_f = sum(r["files_processed"] for r in rows)
    total_s = sum(len(r["secrets"]) for r in rows)
    print("-" * 35)
    print(f"{'TOTAL':<10} {total_f:>5} {total_s:>7} {total_s/max(1,total_f):.1f}")


def show_task(rows: list[dict], task_id: str, full: bool):
    match = [r for r in rows if r["task_id"] == task_id]
    if not match:
        sys.exit(f"Task {task_id} not found")
    r = match[0]
    print(f"{r['task_id']}: {len(r['secrets'])} secrets from {r['files_processed']} files\n")
    for i, s in enumerate(r["secrets"], 1):
        print(f"  {i:3d}. Q: {s['question']}")
        print(f"       A: {s['answer']}")
        if full:
            src = s.get("source_file", s.get("source", ""))
            if src:
                print(f"       Source: {src}")
        print()


def search_secrets(rows: list[dict], query: str, full: bool):
    q = query.lower()
    hits = []
    for r in rows:
        for s in r["secrets"]:
            if q in s.get("question", "").lower() or q in str(s.get("answer", "")).lower():
                hits.append((r["task_id"], s))
    if not hits:
        print(f"No matches for '{query}'")
        return
    print(f"{len(hits)} matches for '{query}':\n")
    for tid, s in hits:
        print(f"  [{tid}] Q: {s['question']}")
        print(f"         A: {s['answer']}")
        if full:
            src = s.get("source_file", s.get("source", ""))
            if src:
                print(f"         Source: {src}")
        print()


def main():
    parser = argparse.ArgumentParser(description="View extracted secrets")
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--task", "-t", help="Show secrets for a specific task")
    parser.add_argument("--search", "-s", help="Search questions/answers")
    parser.add_argument("--stats", action="store_true", help="Per-task stats table")
    parser.add_argument("--full", "-f", action="store_true", help="Show source files")
    args = parser.parse_args()

    rows = load(args.input)
    if args.task:
        show_task(rows, args.task, args.full)
    elif args.search:
        search_secrets(rows, args.search, args.full)
    elif args.stats:
        show_stats(rows)
    else:
        show_summary(rows)


if __name__ == "__main__":
    main()
