#!/usr/bin/env python3
"""
Summarize adversarial question runs: performance + privacy eval (if present).

Usage:
    python summarize_adversarial_batch.py --batch /path/to/batch
    python summarize_adversarial_batch.py --batch /path/to/batch --output summary.md
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


METRICS = [
    "insights_recall",
    "factuality",
    "distractor_avoidance",
    "report_quality",
    "harmonic_mean",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def task_dirs(batch_dir: Path) -> List[Path]:
    return sorted([p for p in batch_dir.iterdir() if p.is_dir() and p.name.startswith("DR")])


def load_question_set(task_dir: Path) -> Optional[str]:
    question_used = task_dir / "question_used.json"
    if not question_used.exists():
        return None
    data = load_json(question_used)
    return data.get("question_set")


def load_scores(task_dir: Path) -> Optional[Dict[str, float]]:
    path = task_dir / "scores.json"
    if not path.exists():
        return None
    data = load_json(path)
    return {m: float(data.get(m, 0.0)) for m in METRICS}


def summarize_scores(batch_dir: Path) -> Tuple[Dict[str, float], List[Tuple[str, Dict[str, float]]]]:
    rows: List[Tuple[str, Dict[str, float]]] = []
    totals = {m: 0.0 for m in METRICS}
    count = 0

    for task_dir in task_dirs(batch_dir):
        scores = load_scores(task_dir)
        if not scores:
            continue
        rows.append((task_dir.name, scores))
        for m in METRICS:
            totals[m] += scores.get(m, 0.0)
        count += 1

    averages = {m: (totals[m] / count) if count else 0.0 for m in METRICS}
    return averages, rows


def find_latest_privacy_eval(batch_dir: Path) -> Optional[Path]:
    privacy_root = batch_dir / "privacy"
    if not privacy_root.exists():
        return None
    candidates = list(privacy_root.glob("**/privacy_eval*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def summarize_privacy(path: Path) -> Dict[str, float]:
    data = load_json(path)
    single = data.get("single_task", []) or []
    cross = data.get("cross_task", []) or []

    total_tasks = len(single)
    total_insights = sum(r.get("insights_total", 0) for r in single)
    total_leaked = sum(r.get("insights_any_leaked", 0) for r in single)
    total_company_leaked = sum(1 for r in single if r.get("company_name_leaked"))

    leakage_rate = (total_leaked / total_insights) if total_insights else 0.0
    cross_rate = max((r.get("leakage_rate", 0.0) for r in cross), default=0.0)

    return {
        "tasks": total_tasks,
        "insights_leaked": total_leaked,
        "insights_total": total_insights,
        "leakage_rate": leakage_rate,
        "company_leaks": total_company_leaked,
        "cross_rate": cross_rate,
    }


def format_scores(rows: List[Tuple[str, Dict[str, float]]]) -> str:
    lines = []
    header = "Task  IR   F    DA   RQ   HM"
    lines.append(header)
    lines.append("-" * len(header))
    for task_id, scores in rows:
        lines.append(
            f"{task_id}  "
            f"{scores['insights_recall']:.2f} "
            f"{scores['factuality']:.2f} "
            f"{scores['distractor_avoidance']:.2f} "
            f"{scores['report_quality']:.2f} "
            f"{scores['harmonic_mean']:.2f}"
        )
    return "\n".join(lines)


def build_report(batch_dir: Path, output: Optional[Path]) -> None:
    question_sets = sorted({load_question_set(t) for t in task_dirs(batch_dir) if load_question_set(t)})
    averages, rows = summarize_scores(batch_dir)

    privacy_path = find_latest_privacy_eval(batch_dir)
    privacy_summary = summarize_privacy(privacy_path) if privacy_path else None

    lines = []
    lines.append(f"Batch: {batch_dir}")
    if question_sets:
        lines.append(f"Question set: {', '.join(question_sets)}")
    else:
        lines.append("Question set: (default or unknown)")
    lines.append("")
    lines.append("Performance (scores.json):")
    lines.append(format_scores(rows) if rows else "No scores found.")
    lines.append("")
    lines.append("Average metrics:")
    lines.append(
        "IR={:.2f} F={:.2f} DA={:.2f} RQ={:.2f} HM={:.2f}".format(
            averages["insights_recall"],
            averages["factuality"],
            averages["distractor_avoidance"],
            averages["report_quality"],
            averages["harmonic_mean"],
        )
    )

    lines.append("")
    lines.append("Privacy eval (latest):")
    if privacy_summary and privacy_path:
        lines.append(f"File: {privacy_path}")
        lines.append(
            "Tasks={tasks} Insights={insights_leaked}/{insights_total} LeakRate={leakage_rate:.0%} "
            "CompanyLeaks={company_leaks} CrossRate={cross_rate:.0%}".format(**privacy_summary)
        )
    else:
        lines.append("No privacy eval found under batch/privacy.")

    report = "\n".join(lines)
    print(report)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"\n[INFO] Wrote summary to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize adversarial runs (performance + privacy).")
    parser.add_argument("--batch", type=Path, required=True, help="Batch directory")
    parser.add_argument("--output", type=Path, help="Optional output markdown/text file")
    args = parser.parse_args()

    if not args.batch.exists():
        raise SystemExit(f"Batch directory not found: {args.batch}")

    build_report(args.batch, args.output)


if __name__ == "__main__":
    main()
