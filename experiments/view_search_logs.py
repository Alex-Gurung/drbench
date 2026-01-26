#!/usr/bin/env python3
"""
View internet/local search logs produced by DrBench.

Examples:
  python experiments/view_search_logs.py --run-dir ./runs/batch_x/DR0001
  python experiments/view_search_logs.py --batch ./runs/batch_x --type both --show
  python experiments/view_search_logs.py --latest --type web
"""

import argparse
import json
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"

LOG_FILES = {
    "web": "internet_searches.jsonl",
    "local": "local_searches.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def summarize(records: Iterable[dict]) -> tuple[int, int, int]:
    total = 0
    ok = 0
    failed = 0
    for r in records:
        total += 1
        success = r.get("success")
        if success is True:
            ok += 1
        elif success is False:
            failed += 1
    return total, ok, failed


def format_record(record: dict) -> str:
    query = record.get("query_raw") or record.get("query") or ""
    success = record.get("success")
    if success is True:
        status = "ok"
    elif success is False:
        status = "fail"
    else:
        status = "unknown"
    return f"[{status}] {query}"


def show_records(records: list[dict], limit: int | None) -> None:
    max_items = limit if limit is not None else len(records)
    for idx, record in enumerate(records[:max_items], 1):
        print(f"  {idx:2d}. {format_record(record)}")


def collect_run_logs(run_dir: Path, kind: str) -> dict[str, list[dict]]:
    collected = {}
    kinds = [kind] if kind in ("web", "local") else ["web", "local"]
    for k in kinds:
        log_path = run_dir / LOG_FILES[k]
        collected[k] = load_jsonl(log_path)
    return collected


def iter_tasks(batch_dir: Path) -> list[Path]:
    return sorted([p for p in batch_dir.iterdir() if p.is_dir() and p.name.startswith("DR")])


def find_latest_batch(runs_dir: Path) -> Path | None:
    batches = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("batch_")]
    if not batches:
        return None
    return max(batches, key=lambda p: p.stat().st_mtime)


def print_summary_for_run(run_dir: Path, kind: str, show: bool, limit: int | None) -> None:
    logs = collect_run_logs(run_dir, kind)
    print(f"\n{run_dir.name}")
    for k, records in logs.items():
        total, ok, failed = summarize(records)
        print(f"- {k}: {total} total (ok={ok}, fail={failed})")
        if show and records:
            show_records(records, limit)


def main() -> int:
    parser = argparse.ArgumentParser(description="View DrBench search logs")
    parser.add_argument("--run-dir", type=Path, help="Run directory to inspect")
    parser.add_argument("--batch", type=Path, help="Batch directory containing task runs")
    parser.add_argument("--latest", action="store_true", help="Use latest batch under runs dir")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Runs directory (default: ./runs)")
    parser.add_argument("--type", choices=["web", "local", "both"], default="both", help="Log type to show")
    parser.add_argument("--show", action="store_true", help="Print queries (default: summary only)")
    parser.add_argument("--limit", type=int, help="Limit number of queries shown per task")

    args = parser.parse_args()

    if args.run_dir:
        if not args.run_dir.exists():
            raise SystemExit(f"Run directory not found: {args.run_dir}")
        print_summary_for_run(args.run_dir, args.type, args.show, args.limit)
        return 0

    if args.latest:
        batch_dir = find_latest_batch(args.runs_dir)
        if not batch_dir:
            raise SystemExit(f"No batches found under {args.runs_dir}")
    elif args.batch:
        batch_dir = args.batch
    else:
        raise SystemExit("Provide --run-dir, --batch, or --latest")

    if not batch_dir.exists():
        raise SystemExit(f"Batch directory not found: {batch_dir}")

    tasks = iter_tasks(batch_dir)
    if not tasks:
        raise SystemExit(f"No task runs found in {batch_dir}")

    print(f"Batch: {batch_dir.name}")
    for task_dir in tasks:
        print_summary_for_run(task_dir, args.type, args.show, args.limit)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
