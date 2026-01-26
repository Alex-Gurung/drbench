#!/usr/bin/env python3
"""
Summarize privacy eval runs stored under batch run directories.

Scans for: ./runs/batch_*/privacy/*/privacy_eval*.json
and prints a compact table per run for quick benchmarking.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich import box


def parse_model_and_timestamp(name: str) -> tuple[str, str | None, str | None]:
    """Parse 'ModelName_YYYYMMDD_HHMMSS' into (model, date, time)."""
    parts = name.rsplit("_", 2)
    if (
        len(parts) == 3
        and all(part.isdigit() for part in parts[1:])
        and len(parts[1]) == 8
        and len(parts[2]) == 6
    ):
        return parts[0], parts[1], parts[2]
    return name, None, None


def infer_batch_model(batch_dir: Path) -> str | None:
    """Extract model name from batch directory name."""
    if batch_dir.name.startswith("batch_"):
        model, _, _ = parse_model_and_timestamp(batch_dir.name[len("batch_"):])
        return model
    return None


from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"


def _latest_batch(runs_dir: Path) -> Optional[Path]:
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("batch_")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_model_from_name(run_id: str, eval_path: Path) -> tuple[str | None, str | None]:
    stem = eval_path.stem
    if stem.startswith("privacy_eval_"):
        suffix = stem[len("privacy_eval_"):]
        if suffix:
            return suffix, "filename"

    parts = run_id.split("_")
    if parts:
        last = parts[-1]
        if last and not last.isdigit():
            return last, "run_id"

    return None, None


def _resolve_model(config: dict, data: dict, run_id: str, eval_path: Path) -> tuple[str, str]:
    model = config.get("model")
    if model:
        return model, "config"
    model = data.get("model")
    if model:
        return model, "data"
    inferred, source = _infer_model_from_name(run_id, eval_path)
    if inferred:
        return inferred, source or "name"
    return "unknown", "unknown"


def _infer_task_field(single: list[dict], field: str) -> tuple[str | None, str | None]:
    values = [r.get(field) for r in single if r.get(field)]
    if not values:
        return None, None
    uniq = sorted({str(v) for v in values})
    if len(uniq) == 1:
        return uniq[0], "task_results"
    return "mixed", "task_results"


def _resolve_search_source(config: dict, data: dict, single: list[dict]) -> tuple[str, str]:
    value = config.get("search_source")
    if value:
        return value, "config"
    value = data.get("search_source")
    if value:
        return value, "data"
    inferred, source = _infer_task_field(single, "search_source")
    if inferred:
        return inferred, source or "task_results"
    return "-", "unknown"


def _resolve_question_source(config: dict, data: dict) -> tuple[str, str]:
    value = config.get("question_source")
    if value:
        return value, "config"
    value = data.get("question_source")
    if value:
        return value, "data"
    return "-", "unknown"


def _resolve_runs(config: dict, single: list[dict]) -> tuple[str | int, str]:
    runs = config.get("runs")
    if runs is not None:
        return runs, "config"
    values = [r.get("num_runs") for r in single if r.get("num_runs") is not None]
    if values:
        uniq = sorted({int(v) for v in values})
        if len(uniq) == 1:
            return uniq[0], "task_results"
        return "mixed", "task_results"
    return 1, "default"


def _resolve_batched(config: dict, runs: str | int) -> tuple[bool | None, str]:
    batched = config.get("batched")
    if batched is not None:
        return bool(batched), "config"
    if isinstance(runs, int) and runs > 1:
        return False, "runs>1"
    return None, "unknown"


def _resolve_intent_only(config: dict, single: list[dict]) -> tuple[bool | None, str]:
    intent_only = config.get("intent_only")
    if intent_only is not None:
        return bool(intent_only), "config"
    qa_types = set()
    for task in single:
        for row in task.get("per_question_results", []) or []:
            qa_type = row.get("qa_type")
            if qa_type:
                qa_types.add(qa_type)
    if not qa_types:
        return None, "unknown"
    if qa_types == {"intent"}:
        return True, "qa_types"
    return False, "qa_types"


def _format_bool(value: bool | None) -> str:
    if value is None:
        return "?"
    return "yes" if value else "no"


def _format_with_source(value: Any, source: str) -> str:
    if value is None or value == "":
        value_str = "-"
    elif isinstance(value, bool):
        value_str = _format_bool(value)
    else:
        value_str = str(value)
    if source and source not in ("config", "unknown", "default"):
        return f"{value_str} ({source})"
    return value_str


def _basename_or_dash(value: str | None) -> str:
    if not value:
        return "-"
    name = Path(value).name
    return name or value


def _count_task_field(single: list[dict], field: str) -> Counter:
    counter: Counter = Counter()
    for row in single:
        value = row.get(field)
        if value is None:
            continue
        counter[str(value)] += 1
    return counter


def _count_search_sources(single: list[dict]) -> Counter:
    counter: Counter = Counter()
    for row in single:
        source = row.get("searches_source")
        if not source:
            continue
        parts = [s.strip() for s in source.split(",") if s.strip()]
        for part in parts:
            key = part.split(":", 1)[0]
            counter[key] += 1
    return counter


def _count_qa_types(single: list[dict]) -> Counter:
    counter: Counter = Counter()
    for task in single:
        for row in task.get("per_question_results", []) or []:
            qa_type = row.get("qa_type")
            if qa_type:
                counter[str(qa_type)] += 1
    return counter


def _format_counter(counter: Counter, order: list[str] | None = None) -> str:
    if not counter:
        return "-"
    items: list[str] = []
    if order:
        for key in order:
            if key in counter:
                items.append(f"{key}={counter[key]}")
        for key in sorted(counter.keys()):
            if key not in order:
                items.append(f"{key}={counter[key]}")
    else:
        for key in sorted(counter.keys()):
            items.append(f"{key}={counter[key]}")
    return ", ".join(items)


def _summarize_eval(path: Path) -> Dict[str, Any]:
    data = _load_json(path)
    single = data.get("single_task", []) or []
    cross = data.get("cross_task", []) or []
    config = data.get("config", {}) or {}

    total_tasks = len(single)
    total_insights = sum(r.get("insights_total", 0) for r in single)
    total_leaked = sum(r.get("insights_any_leaked", 0) for r in single)
    total_company_leaked = sum(1 for r in single if r.get("company_name_leaked"))

    leakage_rate = (total_leaked / total_insights) if total_insights else 0.0
    cross_rate = max((r.get("leakage_rate", 0.0) for r in cross), default=0.0)

    run_dir = path.parent
    batch_dir = run_dir.parent.parent if run_dir.parent.name == "privacy" else run_dir.parent

    model, model_source = _resolve_model(config, data, run_dir.name, path)
    search_source, search_source_source = _resolve_search_source(config, data, single)
    question_source, question_source_source = _resolve_question_source(config, data)
    runs, runs_source = _resolve_runs(config, single)
    batched, batched_source = _resolve_batched(config, runs)
    batch_size = config.get("batch_size")
    intent_only, intent_source = _resolve_intent_only(config, single)
    secrets_file = config.get("secrets_file") or data.get("secrets_file")

    return {
        "path": str(path),
        "batch": batch_dir.name,
        "run_id": run_dir.name,
        "timestamp": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "tasks": total_tasks,
        "insights_leaked": total_leaked,
        "insights_total": total_insights,
        "leakage_rate": leakage_rate,
        "company_leaks": total_company_leaked,
        "cross_rate": cross_rate,
        "model": model,
        "model_source": model_source,
        "search_source": search_source,
        "search_source_source": search_source_source,
        "question_source": question_source,
        "question_source_source": question_source_source,
        "runs": runs,
        "runs_source": runs_source,
        "batched": batched,
        "batched_source": batched_source,
        "batch_size": batch_size,
        "intent_only": intent_only,
        "intent_source": intent_source,
        "secrets_file": secrets_file,
    }


def _find_eval_files(batch_dir: Path) -> List[Path]:
    privacy_root = batch_dir / "privacy"
    if not privacy_root.exists():
        return []
    return sorted(privacy_root.glob("**/privacy_eval*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def _summarize_batch_for_selector(batch_dir: Path) -> Dict[str, Any]:
    eval_files = _find_eval_files(batch_dir)
    privacy_count = len(eval_files)
    latest_eval = eval_files[0] if eval_files else None
    latest_eval_time = ""
    latest_run = ""
    latest_eval_mtime = 0.0
    if latest_eval:
        latest_eval_mtime = latest_eval.stat().st_mtime
        latest_eval_time = datetime.fromtimestamp(latest_eval_mtime).isoformat(timespec="seconds")
        latest_run = latest_eval.parent.name

    tasks = [d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith("DR")]
    batch_mtime = batch_dir.stat().st_mtime
    model = infer_batch_model(batch_dir) or "unknown"

    sort_mtime = latest_eval_mtime if latest_eval_mtime else batch_mtime

    return {
        "path": str(batch_dir),
        "batch": batch_dir.name,
        "model": model,
        "tasks": len(tasks),
        "privacy_count": privacy_count,
        "latest_run": latest_run,
        "latest_eval_time": latest_eval_time,
        "sort_mtime": sort_mtime,
    }


def _render_batch_selector_table(
    summaries: List[Dict[str, Any]],
    default_idx: int,
    console: Console,
) -> None:
    table = Table(title="Batches (Privacy Eval)", box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Idx", justify="right")
    table.add_column("Batch")
    table.add_column("Tasks", justify="right")
    table.add_column("Priv Runs", justify="right")
    table.add_column("Latest Run")
    table.add_column("Latest Eval")
    table.add_column("Model")

    for idx, summary in enumerate(summaries):
        idx_label = f"{idx}"
        row_style = "dim" if summary["privacy_count"] == 0 else ""
        if idx == default_idx:
            idx_label = f"{idx}*"
            row_style = "bold green"
        table.add_row(
            idx_label,
            summary["batch"],
            str(summary["tasks"]),
            str(summary["privacy_count"]),
            summary["latest_run"],
            summary["latest_eval_time"],
            summary["model"],
            style=row_style,
        )

    console.print(table)


def _select_batch_interactive(
    runs_dir: Path,
    default_batch: Optional[Path],
    console: Console,
) -> Optional[Path]:
    batch_dirs = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("batch_")]
    if not batch_dirs:
        console.print("No batch directories found.")
        return None

    summaries = [_summarize_batch_for_selector(b) for b in batch_dirs]
    summaries.sort(key=lambda s: (s["privacy_count"] == 0, -s["sort_mtime"]))

    default_idx = 0
    if default_batch:
        for idx, summary in enumerate(summaries):
            if summary["path"] == str(default_batch):
                default_idx = idx
                break

    _render_batch_selector_table(summaries, default_idx, console)

    if not sys.stdin.isatty():
        return Path(summaries[default_idx]["path"])

    try:
        choice = Prompt.ask("Select batch", default=str(default_idx)).strip().lower()
    except EOFError:
        choice = str(default_idx)

    if choice.isdigit():
        idx = int(choice)
        if 0 <= idx < len(summaries):
            return Path(summaries[idx]["path"])

    return Path(summaries[default_idx]["path"])


def _select_eval_runs_interactive(
    eval_files: List[Path],
    console: Console,
) -> List[Path]:
    if len(eval_files) <= 1:
        return eval_files

    rows = [_summarize_eval(p) for p in eval_files]
    table = Table(title="Privacy Eval Runs", box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Idx", justify="right")
    table.add_column("Run")
    table.add_column("Model")
    table.add_column("Search")
    table.add_column("Q Src")
    table.add_column("Runs", justify="right")
    table.add_column("Batched", justify="right")
    table.add_column("Batch Sz", justify="right")
    table.add_column("Intent", justify="right")
    table.add_column("Secrets")
    table.add_column("Tasks", justify="right")
    table.add_column("Insights", justify="right")
    table.add_column("Leak Rate", justify="right")
    table.add_column("Company Leaks", justify="right")
    table.add_column("Cross Rate", justify="right")
    table.add_column("Timestamp")

    for idx, row in enumerate(rows):
        insights = f"{row['insights_leaked']}/{row['insights_total']}"
        leak_rate = f"{row['leakage_rate']:.0%}"
        cross_rate = f"{row['cross_rate']:.0%}" if row["cross_rate"] else "-"
        batched = _format_bool(row.get("batched"))
        batch_size = row.get("batch_size")
        intent_only = _format_bool(row.get("intent_only"))
        secrets_file = _basename_or_dash(row.get("secrets_file"))
        table.add_row(
            str(idx),
            row["run_id"],
            row.get("model", ""),
            row.get("search_source", ""),
            row.get("question_source", ""),
            str(row.get("runs", "")),
            batched,
            str(batch_size) if batch_size else "-",
            intent_only,
            secrets_file,
            str(row["tasks"]),
            insights,
            leak_rate,
            str(row["company_leaks"]),
            cross_rate,
            row["timestamp"],
        )

    console.print(table)

    if not sys.stdin.isatty():
        return [eval_files[0]]

    try:
        choice = Prompt.ask("Select run index or 'a' for all", default="0").strip().lower()
    except EOFError:
        choice = "0"

    if choice in ("a", "all"):
        return eval_files
    if choice.isdigit():
        idx = int(choice)
        if 0 <= idx < len(eval_files):
            return [eval_files[idx]]
    return [eval_files[0]]


def _load_insight_recall(batch_dir: Path, task_ids: List[str]) -> Dict[str, Optional[float]]:
    scores = {}
    for task_id in task_ids:
        scores_path = batch_dir / task_id / "scores.json"
        if not scores_path.exists():
            scores[task_id] = None
            continue
        try:
            data = _load_json(scores_path)
        except Exception:
            scores[task_id] = None
            continue
        value = data.get("insights_recall")
        scores[task_id] = value if isinstance(value, (int, float)) else None
    return scores


def _aggregate_company(single: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    companies: Dict[str, Dict[str, Any]] = {}
    for r in single:
        company = r.get("company_name", "Unknown")
        entry = companies.setdefault(
            company,
            {
                "tasks": [],
                "insights_leaked": 0,
                "insights_total": 0,
                "distractors_leaked": 0,
                "distractors_total": 0,
                "company_leaks": 0,
                "direct_leaks": 0,
            },
        )
        entry["tasks"].append(r.get("task_id", ""))
        entry["insights_leaked"] += r.get("insights_any_leaked", 0)
        entry["insights_total"] += r.get("insights_total", 0)
        entry["distractors_leaked"] += r.get("distractors_any_leaked", 0)
        entry["distractors_total"] += r.get("distractors_total", 0)
        entry["company_leaks"] += 1 if r.get("company_name_leaked") else 0
        entry["direct_leaks"] += r.get("direct_company_leaks", 0)

    return sorted(companies.items(), key=lambda x: x[0].lower())


def _print_summary_table(rows: List[Dict[str, Any]], console: Console) -> None:
    table = Table(title="Privacy Eval Summary", box=box.ROUNDED)
    table.add_column("Batch", style="cyan")
    table.add_column("Run", style="dim")
    table.add_column("Model")
    table.add_column("Search")
    table.add_column("Q Src")
    table.add_column("Runs", justify="right")
    table.add_column("Batched", justify="right")
    table.add_column("Batch Sz", justify="right")
    table.add_column("Intent", justify="right")
    table.add_column("Secrets")
    table.add_column("Tasks", justify="right")
    table.add_column("Insights", justify="right")
    table.add_column("Leak Rate", justify="right")
    table.add_column("Company Leaks", justify="right")
    table.add_column("Cross Rate", justify="right")

    for r in rows:
        insights = f"{r['insights_leaked']}/{r['insights_total']}"
        leak_rate = f"{r['leakage_rate']:.0%}"
        cross_rate = f"{r['cross_rate']:.0%}" if r["cross_rate"] else "-"
        batched = _format_bool(r.get("batched"))
        batch_size = r.get("batch_size")
        intent_only = _format_bool(r.get("intent_only"))
        secrets_file = _basename_or_dash(r.get("secrets_file"))
        table.add_row(
            r["batch"],
            r["run_id"],
            r.get("model", ""),
            r.get("search_source", ""),
            r.get("question_source", ""),
            str(r.get("runs", "")),
            batched,
            str(batch_size) if batch_size else "-",
            intent_only,
            secrets_file,
            str(r["tasks"]),
            insights,
            leak_rate,
            str(r["company_leaks"]),
            cross_rate,
        )

    console.print(table)


def _print_legend(console: Console) -> None:
    console.print(
        Panel(
            "\n".join(
                [
                    "Summary Table:",
                    "  Batch: batch run folder name under runs/",
                    "  Run: privacy run folder name under batch/privacy/",
                    "  Model: adversary model used for the privacy eval",
                    "  Search: query scope (web, local, or both)",
                    "  Q Src: question source (eval, secrets, or both)",
                    "  Runs: adversary attempts per question",
                    "  Batched: whether batched evaluation was used",
                    "  Batch Sz: questions per batch (if batched)",
                    "  Intent: intent-only mode (only DR question evaluated)",
                    "  Secrets: secrets file basename (if used)",
                    "  Tasks: number of tasks evaluated (single-task results)",
                    "  Insights: leaked/total enterprise insights across tasks",
                    "  Leak Rate: total leaked insights / total insights",
                    "  Company Leaks: tasks where company name was inferred by adversary",
                    "  Cross Rate: max cross-task leakage rate (combined queries per company)",
                    "  '-' means no cross-task results were present",
                    "",
                    "Per-Task Table:",
                    "  Insights: leaked/total enterprise insights (relevant to research question)",
                    "  Distractors: leaked/total enterprise distractors (irrelevant facts)",
                    "  Leak Rate: leaked insights / total insights for the task",
                    "  Name Predicted: adversary inferred company name (any run)",
                    "  Name Mentioned: queries containing the company name",
                    "  IR (agent): insights_recall from scores.json (agent report performance)",
                    "",
                    "Per-Company Table:",
                    "  Insights: leaked/total enterprise insights across tasks",
                    "  Distractors: leaked/total enterprise distractors across tasks",
                    "  Leak Rate: leaked insights / total insights for the company",
                    "  Name Leaks (tasks): tasks with company name inferred / total tasks",
                    "  Name Mentioned (queries): total queries containing the company name",
                    "",
                    "Cross-Task Table:",
                    "  Combined queries across tasks for the same company",
                    "  Leak Rate: leaked insights / total insights in cross-task evaluation",
                ]
            ),
            title="Legend",
            box=box.ROUNDED,
        )
    )


def _format_metric(value: Optional[float]) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "-"


def _print_task_table(
    single: List[Dict[str, Any]],
    console: Console,
    insight_recall: Dict[str, Optional[float]],
) -> None:
    table = Table(title="Per-Task (Single-Task)", box=box.ROUNDED)
    table.add_column("Task", style="cyan")
    table.add_column("Company", style="dim")
    table.add_column("Insights", justify="right")
    table.add_column("Distractors", justify="right")
    table.add_column("Leak Rate", justify="right")
    table.add_column("Name Predicted", justify="right")
    table.add_column("Name Mentioned", justify="right")
    table.add_column("IR (agent)", justify="right")

    for r in sorted(single, key=lambda x: x.get("task_id", "")):
        insights = f"{r.get('insights_any_leaked', 0)}/{r.get('insights_total', 0)}"
        distractors = f"{r.get('distractors_any_leaked', 0)}/{r.get('distractors_total', 0)}"
        leak_rate = f"{r.get('leakage_rate', 0.0):.0%}"
        name_leaked = "yes" if r.get("company_name_leaked") else "no"
        ir_value = _format_metric(insight_recall.get(r.get("task_id", "")))
        table.add_row(
            r.get("task_id", ""),
            r.get("company_name", "")[:20],
            insights,
            distractors,
            leak_rate,
            name_leaked,
            str(r.get("direct_company_leaks", 0)),
            ir_value,
        )

    console.print(table)


def _print_company_table(single: List[Dict[str, Any]], console: Console) -> None:
    table = Table(title="Per-Company (Single-Task)", box=box.ROUNDED)
    table.add_column("Company", style="cyan")
    table.add_column("Tasks", justify="right")
    table.add_column("Insights", justify="right")
    table.add_column("Distractors", justify="right")
    table.add_column("Leak Rate", justify="right")
    table.add_column("Name Leaks (tasks)", justify="right")
    table.add_column("Name Mentioned (queries)", justify="right")

    for company, agg in _aggregate_company(single):
        insights = f"{agg['insights_leaked']}/{agg['insights_total']}"
        distractors = f"{agg['distractors_leaked']}/{agg['distractors_total']}"
        leak_rate = (
            f"{agg['insights_leaked'] / agg['insights_total']:.0%}"
            if agg["insights_total"]
            else "0%"
        )
        name_leaks = f"{agg['company_leaks']}/{len(agg['tasks'])}"
        table.add_row(
            company[:24],
            str(len(agg["tasks"])),
            insights,
            distractors,
            leak_rate,
            name_leaks,
            str(agg["direct_leaks"]),
        )

    console.print(table)


def _print_cross_table(cross: List[Dict[str, Any]], console: Console) -> None:
    if not cross:
        return
    table = Table(title="Cross-Task (Combined Queries)", box=box.ROUNDED)
    table.add_column("Company", style="cyan")
    table.add_column("Tasks", justify="right")
    table.add_column("Insights", justify="right")
    table.add_column("Leak Rate", justify="right")

    for r in sorted(cross, key=lambda x: x.get("company_name", "").lower()):
        insights = f"{r.get('insights_leaked', 0)}/{r.get('insights_total', 0)}"
        leak_rate = f"{r.get('leakage_rate', 0.0):.0%}"
        table.add_row(
            r.get("company_name", "")[:24],
            str(len(r.get("tasks", []))),
            insights,
            leak_rate,
        )

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize privacy eval runs by batch.")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Runs root directory")
    parser.add_argument("--batch", type=Path, help="Specific batch directory to scan")
    parser.add_argument("--latest", action="store_true", help="Only scan latest batch")
    parser.add_argument("--all", action="store_true", help="Scan all batch directories (default)")
    parser.add_argument("--interactive", action="store_true", help="Select batch/run interactively")
    parser.add_argument("--example-task", help="Show example details for a task id (e.g., DR0013)")
    parser.add_argument("--example-company", help="Show example details for a company name (substring match)")
    args = parser.parse_args()

    console = Console()

    if args.interactive:
        default_batch = args.batch
        if not default_batch and args.latest:
            default_batch = _latest_batch(args.runs_dir)
        selected = _select_batch_interactive(args.runs_dir, default_batch, console)
        if not selected:
            return
        batch_dirs = [selected]
    else:
        if args.batch:
            batch_dirs = [args.batch]
        elif args.latest:
            latest = _latest_batch(args.runs_dir)
            if not latest:
                print("No batch directories found.")
                return
            batch_dirs = [latest]
        else:
            batch_dirs = [p for p in args.runs_dir.iterdir() if p.is_dir() and p.name.startswith("batch_")]
            if not batch_dirs:
                print("No batch directories found.")
                return

    eval_files = []
    for batch_dir in sorted(batch_dirs, key=lambda p: p.stat().st_mtime, reverse=True):
        eval_files.extend(_find_eval_files(batch_dir))

    if not eval_files:
        print("No privacy eval files found under runs/batch_*/privacy")
        return

    if args.interactive:
        eval_files = _select_eval_runs_interactive(eval_files, console)

    rows = [_summarize_eval(p) for p in eval_files]
    _print_summary_table(rows, console)
    _print_legend(console)

    for eval_path in eval_files:
        data = _load_json(eval_path)
        single = data.get("single_task", []) or []
        cross = data.get("cross_task", []) or []

        run_dir = eval_path.parent
        batch_dir = run_dir.parent.parent if run_dir.parent.name == "privacy" else run_dir.parent

        config = data.get("config", {}) or {}
        adv_model, model_source = _resolve_model(config, data, run_dir.name, eval_path)
        search_source, search_source_source = _resolve_search_source(config, data, single)
        question_source, question_source_source = _resolve_question_source(config, data)
        runs, runs_source = _resolve_runs(config, single)
        batched, batched_source = _resolve_batched(config, runs)
        batch_size = config.get("batch_size")
        intent_only, intent_source = _resolve_intent_only(config, single)
        secrets_file = config.get("secrets_file") or data.get("secrets_file")
        secrets_source = "config" if config.get("secrets_file") else ("data" if data.get("secrets_file") else "unknown")
        secrets_label = _basename_or_dash(secrets_file)
        secrets_value = secrets_label if secrets_label != "-" else None

        config_lines = [
            f"Adversary model: {_format_with_source(adv_model, model_source)}",
            f"Search source: {_format_with_source(search_source, search_source_source)}",
            f"Question source: {_format_with_source(question_source, question_source_source)}",
            f"Runs/question: {_format_with_source(runs, runs_source)}",
            f"Batched: {_format_with_source(batched, batched_source)}",
        ]
        if batch_size:
            config_lines.append(f"Batch size: {batch_size}")
        config_lines.append(f"Intent only: {_format_with_source(intent_only, intent_source)}")
        if secrets_value:
            config_lines.append(f"Secrets file: {_format_with_source(secrets_value, secrets_source)}")

        search_source_counts = _count_task_field(single, "search_source")
        if search_source_counts:
            config_lines.append(
                "Search source (per-task): "
                + _format_counter(search_source_counts, order=["web", "local", "both", "mixed"])
            )

        searches_source_counts = _count_search_sources(single)
        if searches_source_counts:
            config_lines.append(
                "Search log source: "
                + _format_counter(searches_source_counts, order=["action_plan", "log", "missing"])
            )

        qa_type_counts = _count_qa_types(single)
        if qa_type_counts:
            config_lines.append(
                "QA types: "
                + _format_counter(
                    qa_type_counts,
                    order=["company_identity", "intent", "insight", "distractor"],
                )
            )

        extra_keys = [k for k in sorted(config.keys()) if k not in {
            "model",
            "search_source",
            "question_source",
            "runs",
            "batched",
            "batch_size",
            "intent_only",
            "secrets_file",
        }]
        for key in extra_keys:
            config_lines.append(f"{key}: {config.get(key)}")

        console.print(
            Panel(
                "\n".join(
                    [
                        f"Batch: {batch_dir.name}",
                        f"Run: {run_dir.name}",
                        *config_lines,
                        "All tables below are scoped to this batch/run.",
                    ]
                ),
                title="Run Details",
                box=box.ROUNDED,
            )
        )
        task_ids = [r.get("task_id", "") for r in single if r.get("task_id")]
        insight_recall = _load_insight_recall(batch_dir, task_ids)
        _print_task_table(single, console, insight_recall)
        _print_company_table(single, console)
        _print_cross_table(cross, console)

        if args.example_task:
            match = next((r for r in single if r.get("task_id") == args.example_task), None)
            if match:
                _print_example(console, match, label=f"Example Task: {args.example_task}")
        if args.example_company:
            company = args.example_company.lower()
            matches = [r for r in single if company in r.get("company_name", "").lower()]
            if matches:
                _print_example(console, matches[0], label=f"Example Company: {matches[0].get('company_name','')}")


def _print_example(console: Console, task_result: Dict[str, Any], label: str) -> None:
    direct_queries = task_result.get("direct_leak_queries", []) or []
    leaked = [r for r in task_result.get("per_question_results", []) if r.get("any_leaked")]

    lines = [
        f"Task: {task_result.get('task_id', '')}",
        f"Company: {task_result.get('company_name', '')}",
        f"Name Predicted: {'yes' if task_result.get('company_name_leaked') else 'no'}",
        f"Name Mentioned (queries): {len(direct_queries)}",
    ]
    if direct_queries:
        lines.append("")
        lines.append("Queries Mentioning Name:")
        for q in direct_queries[:5]:
            lines.append(f"- {q}")
        if len(direct_queries) > 5:
            lines.append(f"- ... and {len(direct_queries) - 5} more")
    if leaked:
        lines.append("")
        lines.append("Leaked Questions (first 3):")
        for r in leaked[:3]:
            lines.append(f"- [{r.get('id','')}] {r.get('question','')}")
        if len(leaked) > 3:
            lines.append(f"- ... and {len(leaked) - 3} more")

    console.print(
        Panel(
            "\n".join(lines),
            title=label,
            box=box.ROUNDED,
        )
    )


if __name__ == "__main__":
    main()
