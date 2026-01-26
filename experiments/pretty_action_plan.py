#!/usr/bin/env python3
"""
Render an action_plan JSON file to a readable Markdown summary.

Usage:
  python pretty_action_plan.py /path/to/action_plan_final.json
  python pretty_action_plan.py /path/to/action_plan_final.json --out /path/to/action_plan.md
  python pretty_action_plan.py /path/to/action_plan_final.json --order completion
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
DEFAULT_WORKSPACE_DIR = REPO_ROOT / "outputs" / "research_workspace"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "N/A"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _status_label(status: str | None) -> str:
    if status == "completed":
        return "OK"
    if status == "failed":
        return "FAIL"
    if status == "in_progress":
        return "INPROG"
    if status == "pending":
        return "PEND"
    if status == "skipped":
        return "SKIP"
    return "UNK"


def _get_action_query(action: Dict[str, Any]) -> Optional[str]:
    params = action.get("parameters", {}) or {}
    return params.get("query") or params.get("url")


def _hash_file(path: Path) -> Optional[str]:
    try:
        content = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(content).hexdigest()


def _find_run_dir(plan_path: Path, plan_hash: Optional[str], runs_root: Path) -> Optional[Path]:
    if not runs_root.exists():
        return None
    candidates = list(runs_root.glob("batch_*/*/action_plan_final.json"))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if plan_hash and _hash_file(candidate) == plan_hash:
            return candidate.parent
    return None


def _find_report_in_workspace(plan_timestamp: Optional[datetime], workspace: Path) -> Optional[str]:
    if not workspace.exists():
        return None
    reports = sorted(workspace.glob("research_report_*.md"))
    if not reports:
        return None
    if not plan_timestamp:
        return reports[-1].read_text(encoding="utf-8", errors="ignore")

    best = None
    best_diff = None
    for report in reports:
        try:
            stamp = report.stem.replace("research_report_", "")
            report_time = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        diff = abs((report_time - plan_timestamp).total_seconds())
        if best_diff is None or diff < best_diff:
            best = report
            best_diff = diff
    if best:
        return best.read_text(encoding="utf-8", errors="ignore")
    return reports[-1].read_text(encoding="utf-8", errors="ignore")


def _load_report_text(run_dir: Optional[Path], plan_timestamp: Optional[datetime], workspace: Path) -> Optional[str]:
    if run_dir:
        report_path = run_dir / "results" / "research_report.md"
        if report_path.exists():
            return report_path.read_text(encoding="utf-8", errors="ignore")
        report_json = run_dir / "report.json"
        if report_json.exists():
            try:
                data = _load_json(report_json)
            except Exception:
                print(f"[WARN] Failed to parse {report_json}", file=sys.stderr)
                data = None
            if data and isinstance(data.get("report_text"), str):
                return data["report_text"]
    return _find_report_in_workspace(plan_timestamp, workspace)


def _load_scores(run_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not run_dir:
        return None
    scores_path = run_dir / "scores.json"
    if scores_path.exists():
        try:
            return _load_json(scores_path)
        except Exception:
            print(f"[WARN] Failed to parse {scores_path}", file=sys.stderr)
            return None
    eval_report = run_dir / "results" / "evaluation_report.json"
    if eval_report.exists():
        try:
            data = _load_json(eval_report)
        except Exception:
            print(f"[WARN] Failed to parse {eval_report}", file=sys.stderr)
            return None
        overall = data.get("overall_scores")
        if isinstance(overall, dict):
            return overall
    return None


def _sort_actions(actions: List[Dict[str, Any]], order: str) -> List[Dict[str, Any]]:
    if order == "completion":
        def key(a: Dict[str, Any]):
            parsed = _parse_time(a.get("completion_time"))
            return (parsed is None, parsed or datetime.max)
        return sorted(actions, key=key)
    if order == "status":
        return sorted(actions, key=lambda a: a.get("status", ""))
    return actions


def _summarize(plan: Dict[str, Any], scores: Optional[Dict[str, Any]]) -> str:
    actions = plan.get("actions", []) or []
    status_counts = Counter(a.get("status", "unknown") for a in actions)
    type_counts = Counter(a.get("type", "unknown") for a in actions)

    lines = [
        "# Action Plan Summary",
        f"- Research query: {plan.get('research_query', 'N/A')}",
        f"- Actions: {len(actions)} total (completed: {status_counts.get('completed', 0)}, "
        f"failed: {status_counts.get('failed', 0)}, pending: {status_counts.get('pending', 0)}, "
        f"in_progress: {status_counts.get('in_progress', 0)})",
        "- Action types: " + ", ".join(
            f"{t}:{c}" for t, c in sorted(type_counts.items(), key=lambda x: (-x[1], x[0]))
        ),
        "",
    ]
    if scores:
        grade = scores.get("harmonic_mean")
        if isinstance(grade, (int, float)):
            lines.insert(2, f"- Grade (harmonic_mean): {grade:.4f}")
        ir = scores.get("insights_recall")
        fa = scores.get("factuality")
        rq = scores.get("report_quality")
        da = scores.get("distractor_avoidance")
        metric_parts = []
        if isinstance(ir, (int, float)):
            metric_parts.append(f"IR {ir:.3f}")
        if isinstance(fa, (int, float)):
            metric_parts.append(f"F {fa:.3f}")
        if isinstance(da, (int, float)):
            metric_parts.append(f"DA {da:.3f}")
        if isinstance(rq, (int, float)):
            metric_parts.append(f"RQ {rq:.3f}")
        if metric_parts:
            lines.insert(3, f"- Scores: {', '.join(metric_parts)}")
    return "\n".join(lines)


def _format_actions(actions: List[Dict[str, Any]]) -> str:
    lines = ["# Actions", ""]
    for i, action in enumerate(actions, 1):
        status = _status_label(action.get("status"))
        action_type = action.get("type", "unknown")
        created_iter = action.get("created_in_iteration")
        completed_iter = action.get("iteration_completed")
        completed_iter = completed_iter + 1 if isinstance(completed_iter, int) else None
        exec_time = _format_duration(action.get("execution_time"))
        description = action.get("description", "").strip()
        query = _get_action_query(action)
        deps = action.get("dependencies") or []

        lines.append(
            f"{i}. [{status}] {action_type} | created_iter: {created_iter if created_iter is not None else '?'} "
            f"| completed_iter: {completed_iter if completed_iter is not None else '?'} | time: {exec_time}"
        )
        if description:
            lines.append(f"   - Description: {description}")
        if query:
            lines.append(f"   - Query: {query}")
        if deps:
            lines.append(f"   - Dependencies: {', '.join(deps)}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _format_scores(scores: Dict[str, Any]) -> str:
    lines = ["# Scores", ""]
    grade = scores.get("harmonic_mean")
    if isinstance(grade, (int, float)):
        lines.append(f"- grade (harmonic_mean): {grade:.4f}")
    for key, value in sorted(scores.items()):
        if isinstance(value, (int, float)):
            lines.append(f"- {key}: {value:.4f}")
    return "\n".join(lines).rstrip() + "\n"


def _format_report(report_text: str) -> str:
    if not report_text.strip():
        return "\n".join(
            [
                "# Final Report",
                "",
                "(report text not found)",
                "",
            ]
        )
    return "\n".join(
        [
            "# Final Report",
            "",
            "```",
            report_text.strip(),
            "```",
            "",
        ]
    )


def render_plan(
    plan: Dict[str, Any],
    order: str,
    report_text: Optional[str],
    scores: Optional[Dict[str, Any]],
) -> str:
    actions = plan.get("actions", []) or []
    actions = _sort_actions(actions, order)
    sections = [_summarize(plan, scores)]
    if scores:
        sections.append(_format_scores(scores))
    sections.append(_format_actions(actions))
    sections.append(_format_report(report_text or ""))
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render action_plan JSON to a readable Markdown file.")
    parser.add_argument("plan_path", type=Path, help="Path to action_plan_final.json")
    parser.add_argument("--out", type=Path, help="Write output to a Markdown file")
    parser.add_argument(
        "--order",
        choices=["creation", "completion", "status"],
        default="creation",
        help="Sort actions by creation order, completion time, or status",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Optional run directory (e.g., ./runs/<batch>/DR0001) to load report/scores",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Runs root used to locate the matching run if --run-dir is not provided",
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=DEFAULT_WORKSPACE_DIR,
        help="Workspace root used as a fallback for report text",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip embedding the final report in the output",
    )
    args = parser.parse_args()

    plan = _load_json(args.plan_path)
    plan_hash = _hash_file(args.plan_path)
    plan_time = _parse_time(plan.get("timestamp"))
    if args.run_dir and args.run_dir.exists():
        run_dir = args.run_dir
    else:
        run_dir = _find_run_dir(args.plan_path, plan_hash, args.runs_dir)
    report_text = None if args.no_report else _load_report_text(run_dir, plan_time, args.workspace_dir)
    scores = _load_scores(run_dir)

    output = render_plan(plan, args.order, report_text, scores)

    if args.out:
        args.out.write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
