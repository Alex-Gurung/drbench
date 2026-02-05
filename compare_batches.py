#!/usr/bin/env python3
"""
Compare scores between two DrBench batches across all tasks.

Useful for comparing live Serper runs vs offline BrowseComp runs.

Usage:
    python compare_batches.py /path/to/batch_A /path/to/batch_B
    python compare_batches.py --baseline batch_serper_* --candidate batch_browsecomp_*
    python compare_batches.py batch_A batch_B --label-a Serper --label-b BrowseComp --csv

Example:
    python compare_batches.py \\
        runs/batch_Qwen3-30B_20260114 \\
        runs/batch_Qwen3-30B_browsecomp_20260127 \\
        --label-a Serper --label-b BrowseComp
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional

VAL_TASK_IDS = [f"DR{i:04d}" for i in range(1, 16)]
METRICS = ["insights_recall", "factuality", "distractor_avoidance", "report_quality", "harmonic_mean"]
METRIC_SHORT = {"insights_recall": "IR", "factuality": "F", "distractor_avoidance": "DA", "report_quality": "RQ", "harmonic_mean": "HM"}

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def load_batch_scores(batch_dir: Path) -> Dict[str, Dict]:
    """Load scores for all tasks in a batch, normalizing missing metrics."""
    scores = {}
    for task_id in VAL_TASK_IDS:
        path = batch_dir / task_id / "scores.json"
        if not path.exists():
            continue

        with open(path) as f:
            data = json.load(f)

        # Normalize distractor_avoidance
        if "distractor_avoidance" not in data and "distractor_recall" in data:
            dr = data.get("distractor_recall")
            if dr is not None:
                data["distractor_avoidance"] = 1.0 - dr

        # Compute harmonic_mean if missing
        if "harmonic_mean" not in data:
            values = [data.get(m) for m in METRICS[:4] if data.get(m) is not None and data.get(m) > 0]
            if values:
                data["harmonic_mean"] = len(values) / sum(1 / v for v in values)
            else:
                data["harmonic_mean"] = None

        scores[task_id] = data
    return scores


def count_searches(batch_dir: Path, task_id: str) -> Dict[str, int]:
    """Count web and local searches for a task."""
    task_dir = batch_dir / task_id
    counts = {"web": 0, "local": 0, "browsecomp": 0}

    # Try action_plan_final.json first
    action_plan = task_dir / "action_plan_final.json"
    if action_plan.exists():
        with open(action_plan) as f:
            plan = json.load(f)
        for action in plan.get("actions", []):
            action_type = action.get("type", "")
            tool_name = action.get("actual_output", {}).get("tool", "")
            if action_type == "web_search" or tool_name in ("internet_search", "browsecomp_search"):
                if tool_name == "browsecomp_search":
                    counts["browsecomp"] += 1
                else:
                    counts["web"] += 1
            elif action_type == "local_document_search" or tool_name == "local_document_search":
                counts["local"] += 1
        return counts

    # Fallback to JSONL logs
    for log_name, key in [("internet_searches.jsonl", "web"), ("browsecomp_searches.jsonl", "browsecomp"), ("local_document_searches.jsonl", "local")]:
        log_path = task_dir / log_name
        if log_path.exists():
            with open(log_path) as f:
                counts[key] = sum(1 for line in f if line.strip())

    return counts


def format_delta(delta: float, threshold: float = 0.05) -> str:
    """Format delta with color coding."""
    if delta > threshold:
        return f"{GREEN}{delta:+.2f}{RESET}"
    elif delta < -threshold:
        return f"{RED}{delta:+.2f}{RESET}"
    return f"{delta:+.2f}"


def compare_batches(batch_a: Path, batch_b: Path, label_a: str, label_b: str, csv_mode: bool = False):
    """Compare two batches and print results."""
    scores_a = load_batch_scores(batch_a)
    scores_b = load_batch_scores(batch_b)

    if not scores_a:
        print(f"No scores found in {batch_a}", file=sys.stderr)
        return 1
    if not scores_b:
        print(f"No scores found in {batch_b}", file=sys.stderr)
        return 1

    if csv_mode:
        print("task,metric,batch_a,batch_b,delta")
        for task_id in VAL_TASK_IDS:
            if task_id not in scores_a or task_id not in scores_b:
                continue
            for metric in METRICS:
                a = scores_a[task_id].get(metric)
                b = scores_b[task_id].get(metric)
                delta = (b - a) if a is not None and b is not None else ""
                print(f"{task_id},{metric},{a if a is not None else ''},{b if b is not None else ''},{delta}")
        return 0

    # Header
    print(f"\n{BOLD}Comparing:{RESET} {batch_a.name} ({label_a}) vs {batch_b.name} ({label_b})")
    print("=" * 100)

    # Table header
    header = f"{'Task':<8}"
    for metric in METRICS:
        short = METRIC_SHORT[metric]
        header += f"  {short:>5} {short:>5} {'Δ':>6}"
    header += "  Searches"
    print(header)

    subheader = "-" * 8
    for _ in METRICS:
        subheader += f"  {label_a[:5]:>5} {label_b[:5]:>5} {'':>6}"
    subheader += f"  {label_a[:4]:>4}/{label_b[:4]:>4}"
    print(subheader)

    # Track deltas for summary
    all_deltas = {m: [] for m in METRICS}
    all_a = {m: [] for m in METRICS}
    all_b = {m: [] for m in METRICS}
    task_count = 0

    for task_id in VAL_TASK_IDS:
        if task_id not in scores_a or task_id not in scores_b:
            continue

        task_count += 1
        row = f"{task_id:<8}"

        for metric in METRICS:
            a_val = scores_a[task_id].get(metric)
            b_val = scores_b[task_id].get(metric)

            if a_val is not None and b_val is not None:
                delta = b_val - a_val
                all_deltas[metric].append(delta)
                all_a[metric].append(a_val)
                all_b[metric].append(b_val)
                row += f"  {a_val:>5.2f} {b_val:>5.2f} {format_delta(delta)}"
            elif a_val is not None:
                all_a[metric].append(a_val)
                row += f"  {a_val:>5.2f} {'':>5} {'':>6}"
            elif b_val is not None:
                all_b[metric].append(b_val)
                row += f"  {'':>5} {b_val:>5.2f} {'':>6}"
            else:
                row += f"  {'':>5} {'':>5} {'':>6}"

        # Search counts
        searches_a = count_searches(batch_a, task_id)
        searches_b = count_searches(batch_b, task_id)
        total_a = searches_a["web"] + searches_a["browsecomp"] + searches_a["local"]
        total_b = searches_b["web"] + searches_b["browsecomp"] + searches_b["local"]
        row += f"  {total_a:>4}/{total_b:>4}"

        print(row)

    # Summary statistics
    print("\n" + "=" * 100)
    print(f"{BOLD}SUMMARY STATISTICS{RESET}")
    print("=" * 100)

    print(f"\n{'Metric':<25} {label_a+' Mean':>12} {label_b+' Mean':>12} {'Mean Δ':>10} {'Std Δ':>10} {'Min Δ':>10} {'Max Δ':>10}")
    print("-" * 91)

    for metric in METRICS:
        deltas = all_deltas[metric]
        if not deltas:
            continue

        a_vals = all_a[metric]
        b_vals = all_b[metric]

        a_mean = mean(a_vals) if a_vals else 0
        b_mean = mean(b_vals) if b_vals else 0
        d_mean = mean(deltas)
        d_std = stdev(deltas) if len(deltas) > 1 else 0
        d_min = min(deltas)
        d_max = max(deltas)

        winner = ""
        if d_mean > 0.02:
            winner = f" ({GREEN}{label_b} better{RESET})"
        elif d_mean < -0.02:
            winner = f" ({GREEN}{label_a} better{RESET})"

        print(f"{metric:<25} {a_mean:>12.4f} {b_mean:>12.4f} {d_mean:>+10.4f} {d_std:>10.4f} {d_min:>+10.4f} {d_max:>+10.4f}{winner}")

    print(f"\nTasks compared: {task_count}/15")

    # Highlight significant differences
    print(f"\n{BOLD}SIGNIFICANT DIFFERENCES{RESET} (|mean delta| > 0.05):")
    significant = [(m, mean(all_deltas[m])) for m in METRICS if all_deltas[m] and abs(mean(all_deltas[m])) > 0.05]
    if significant:
        for metric, d in significant:
            winner = label_b if d > 0 else label_a
            color = GREEN if d > 0 else RED
            print(f"  - {metric}: {color}{d:+.4f}{RESET} ({winner} is better)")
    else:
        print("  None")

    return 0


def find_batch(pattern: str, runs_dir: Path) -> Optional[Path]:
    """Find batch directory matching pattern."""
    matches = sorted(runs_dir.glob(f"batch_{pattern}*"))
    if not matches:
        return None
    return max(matches, key=lambda d: d.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Compare two DrBench batches")
    parser.add_argument("batch_a", nargs="?", type=Path, help="First batch directory")
    parser.add_argument("batch_b", nargs="?", type=Path, help="Second batch directory")
    parser.add_argument("--baseline", type=str, help="Baseline batch pattern (searches runs/)")
    parser.add_argument("--candidate", type=str, help="Candidate batch pattern (searches runs/)")
    parser.add_argument("--label-a", default="A", help="Label for first batch")
    parser.add_argument("--label-b", default="B", help="Label for second batch")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    parser.add_argument("--runs-dir", type=Path, default=Path("/home/toolkit/runs"), help="Runs directory for pattern matching")

    args = parser.parse_args()

    # Resolve batch directories
    if args.batch_a and args.batch_b:
        batch_a, batch_b = args.batch_a, args.batch_b
    elif args.baseline and args.candidate:
        batch_a = find_batch(args.baseline, args.runs_dir)
        batch_b = find_batch(args.candidate, args.runs_dir)
        if not batch_a:
            print(f"No batch found matching: batch_{args.baseline}*", file=sys.stderr)
            return 1
        if not batch_b:
            print(f"No batch found matching: batch_{args.candidate}*", file=sys.stderr)
            return 1
    else:
        parser.print_help()
        return 1

    return compare_batches(batch_a, batch_b, args.label_a, args.label_b, args.csv)


if __name__ == "__main__":
    sys.exit(main() or 0)
