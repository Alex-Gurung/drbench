#!/usr/bin/env python3
"""
Analyze a DrBench run and pretty-print action summaries.

Usage:
    python analyze_run.py <run_dir>
    python analyze_run.py ./runs/DR0001_Qwen3-30B_20260106_143052
    python analyze_run.py --latest
"""

import argparse
import json
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from drbench import task_loader

console = Console()

UNKNOWN_MODEL = "unknown (not recorded)"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
DEFAULT_VECTOR_STORES_DIR = REPO_ROOT / "outputs" / "vector_stores"


def find_latest_run(runs_dir: Path = DEFAULT_RUNS_DIR) -> Optional[Path]:
    """Find the most recently modified run directory."""
    runs_path = runs_dir
    if not runs_path.exists():
        return None

    run_dirs = [d for d in runs_path.iterdir() if d.is_dir()]
    if not run_dirs:
        return None

    return max(run_dirs, key=lambda d: d.stat().st_mtime)


def find_action_plan(run_dir: Path, vector_stores_dir: Optional[Path] = None) -> Optional[Path]:
    """Find the action_plan_final.json - checks standard outputs/vector_stores location."""
    # First check directly in run_dir (preferred for new runs)
    for name in ["action_plan_final.json", "action_plan_initial.json"]:
        direct = run_dir / name
        if direct.exists():
            return direct

    # Fall back to vector stores (older runs)
    vector_stores = vector_stores_dir or DEFAULT_VECTOR_STORES_DIR
    if vector_stores.exists():
        sessions = sorted(
            [d for d in vector_stores.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        for session_dir in sessions:
            for name in ["action_plan_final.json", "action_plan_initial.json"]:
                plan = session_dir / name
                if plan.exists():
                    return plan

    return None


def find_action_plan_by_time(
    target_time: float, tolerance_seconds: int = 300, vector_stores_dir: Optional[Path] = None
) -> Optional[Path]:
    """Find action plan created around the same time as a run."""
    vector_stores = vector_stores_dir or DEFAULT_VECTOR_STORES_DIR
    if not vector_stores.exists():
        return None

    best_match = None
    best_diff = float('inf')

    for session_dir in vector_stores.iterdir():
        if not session_dir.is_dir():
            continue
        plan = session_dir / "action_plan_final.json"
        if plan.exists():
            diff = abs(plan.stat().st_mtime - target_time)
            if diff < best_diff and diff < tolerance_seconds:
                best_diff = diff
                best_match = plan

    return best_match


def format_duration(seconds: Optional[float]) -> str:
    """Format duration in human-readable form."""
    if seconds is None:
        return "N/A"
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds/60:.1f}m"


def truncate(text: str, max_len: int = 80) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len-3] + "..."


def print_header(title: str, char: str = "="):
    """Print a formatted header."""
    console.print(f"\n[bold]{char * 60}[/bold]")
    console.print(f"[bold]  {title}[/bold]")
    console.print(f"[bold]{char * 60}[/bold]")

def parse_model_and_timestamp(name: str) -> tuple[str, Optional[str], Optional[str]]:
    """Parse <model>_<YYYYMMDD>_<HHMMSS> if present, else return name."""
    parts = name.rsplit("_", 2)
    if (
        len(parts) == 3
        and all(part.isdigit() for part in parts[1:])
        and len(parts[1]) == 8
        and len(parts[2]) == 6
    ):
        return parts[0], parts[1], parts[2]
    return name, None, None


def infer_run_model(run_dir: Path, batch_dir: Optional[Path] = None) -> Optional[str]:
    """Infer the model used to run a task or batch from directory names."""
    if batch_dir and batch_dir.name.startswith("batch_"):
        model, _, _ = parse_model_and_timestamp(batch_dir.name[len("batch_"):])
        return model

    name = run_dir.name
    if name.startswith("batch_"):
        model, _, _ = parse_model_and_timestamp(name[len("batch_"):])
        return model

    if name.startswith("DR"):
        parts = name.split("_")
        if len(parts) >= 4 and parts[-2].isdigit() and parts[-1].isdigit():
            return "_".join(parts[1:-2])

    return None


def infer_scoring_model(scores: Dict) -> Optional[str]:
    """Try to infer scoring model from scores metadata, else return None."""
    for key in ("scoring_model", "scorer_model", "grader_model", "eval_model", "model"):
        value = scores.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def print_action_summary(actions: List[Dict]):
    """Print summary statistics of actions."""
    print_header("ACTION SUMMARY")

    type_counts = Counter(a.get("type", "unknown") for a in actions)
    status_counts = Counter(a.get("status", "unknown") for a in actions)
    completed = [a for a in actions if a.get("status") == "completed"]
    total_time = sum(a.get("execution_time", 0) or 0 for a in completed)

    # Summary stats
    console.print(f"\n[bold]Total Actions:[/bold] {len(actions)}")
    console.print(f"  [green]Completed:[/green] {status_counts.get('completed', 0)}")
    console.print(f"  [red]Failed:[/red]    {status_counts.get('failed', 0)}")
    console.print(f"  [yellow]Pending:[/yellow]   {status_counts.get('pending', 0)}")
    console.print(f"\n[bold]Total Execution Time:[/bold] {format_duration(total_time)}")

    # Actions by type table
    table = Table(title="Actions by Type", show_header=True, header_style="bold")
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Time", justify="right")

    for action_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        type_actions = [a for a in actions if a.get("type") == action_type]
        completed_type = [a for a in type_actions if a.get("status") == "completed"]
        failed_type = [a for a in type_actions if a.get("status") == "failed"]
        time_sum = sum(a.get("execution_time", 0) or 0 for a in completed_type)

        status_parts = [f"[green]{len(completed_type)} ok[/green]"]
        if failed_type:
            status_parts.append(f"[red]{len(failed_type)} fail[/red]")

        table.add_row(
            action_type,
            str(count),
            " / ".join(status_parts),
            format_duration(time_sum) if time_sum > 0 else "[dim]-[/dim]"
        )

    console.print(table)


def print_external_calls(actions: List[Dict]):
    """Print details of external web calls."""
    external_types = {"web_search", "url_fetch", "file_download"}
    external_actions = [a for a in actions if a.get("type") in external_types]

    if not external_actions:
        console.print("\n[dim]No external calls made.[/dim]")
        return

    # Summary counts
    completed = sum(1 for a in external_actions if a.get("status") == "completed")
    failed = sum(1 for a in external_actions if a.get("status") == "failed")

    title = f"EXTERNAL CALLS  [green]{completed} completed[/green]  [red]{failed} failed[/red]"
    console.print(f"\n[bold]{title}[/bold]")
    console.print("-" * 80)

    for i, action in enumerate(external_actions, 1):
        status = action.get("status", "unknown")
        action_type = action.get("type", "unknown").upper()
        params = action.get("parameters", {})

        # Get the query/url from multiple possible locations
        query = params.get("query") or params.get("url") or action.get("description") or ""

        # Status styling
        if status == "completed":
            status_style = "[green]OK[/green]"
        elif status == "failed":
            status_style = "[red]X[/red]"
        else:
            status_style = "[yellow]o[/yellow]"

        # Type styling
        type_style = "[cyan]" if action_type == "WEB_SEARCH" else "[blue]"

        console.print(f"\n  {status_style} [{i:2d}] {type_style}{action_type}[/]")

        # Print full query with wrapping
        if query:
            wrapped = textwrap.fill(query, width=74, initial_indent="       ", subsequent_indent="       ")
            console.print(wrapped)
        else:
            console.print("       [dim](no query/url recorded)[/dim]")

        # Additional info line
        info_parts = []

        exec_time = action.get("execution_time")
        if exec_time is not None:
            info_parts.append(f"[dim]Time:[/dim] {format_duration(exec_time)}")

        output = action.get("actual_output", {})
        if output and action_type == "WEB_SEARCH":
            results_count = output.get("results_count", output.get("num_results"))
            if results_count is not None:
                info_parts.append(f"[dim]Results:[/dim] {results_count}")

        if info_parts:
            console.print(f"       {' | '.join(info_parts)}")

        # Error message
        error = action.get("error") or action.get("error_message")
        if error:
            error_wrapped = textwrap.fill(str(error)[:200], width=70, initial_indent="       ", subsequent_indent="       ")
            console.print(f"[red]{error_wrapped}[/red]")


def print_local_searches(actions: List[Dict]):
    """Print details of local document searches."""
    local_actions = [a for a in actions if a.get("type") == "local_document_search"]

    if not local_actions:
        console.print("\n[dim]No local document searches.[/dim]")
        return

    completed = sum(1 for a in local_actions if a.get("status") == "completed")
    failed = sum(1 for a in local_actions if a.get("status") == "failed")

    title = f"LOCAL DOCUMENT SEARCHES  [green]{completed} completed[/green]  [red]{failed} failed[/red]"
    console.print(f"\n[bold]{title}[/bold]")
    console.print("-" * 80)

    for i, action in enumerate(local_actions, 1):
        status = action.get("status", "unknown")
        params = action.get("parameters", {})
        query = params.get("query") or action.get("description") or ""

        if status == "completed":
            status_style = "[green]OK[/green]"
        elif status == "failed":
            status_style = "[red]X[/red]"
        else:
            status_style = "[yellow]o[/yellow]"

        console.print(f"\n  {status_style} [{i:2d}] [magenta]LOCAL_SEARCH[/magenta]")

        if query:
            wrapped = textwrap.fill(query, width=74, initial_indent="       ", subsequent_indent="       ")
            console.print(wrapped)

        info_parts = []
        exec_time = action.get("execution_time")
        if exec_time is not None:
            info_parts.append(f"[dim]Time:[/dim] {format_duration(exec_time)}")

        output = action.get("actual_output", {})
        if output:
            results_count = output.get("results_count", output.get("num_results"))
            if results_count is not None:
                info_parts.append(f"[dim]Results:[/dim] {results_count}")

        if info_parts:
            console.print(f"       {' | '.join(info_parts)}")


def print_all_actions(actions: List[Dict], verbose: bool = False):
    """Print chronological list of all actions."""
    print_header("ALL ACTIONS (by completion order)", "-")

    # Sort by completion time
    sorted_actions = sorted(
        actions,
        key=lambda a: a.get("completion_time") or "z",  # pending actions sort last
    )

    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("#", style="dim", width=3)
    table.add_column("", width=2)  # status icon
    table.add_column("Type", style="cyan", width=24)
    table.add_column("Time", justify="right", width=8)
    table.add_column("Description")

    for i, action in enumerate(sorted_actions, 1):
        status = action.get("status", "unknown")
        icon = {"completed": "[green]OK[/green]", "failed": "[red]X[/red]", "pending": "[yellow]o[/yellow]"}.get(status, "?")

        action_type = action.get("type", "unknown")
        desc = truncate(action.get("description", ""), 50)
        duration = format_duration(action.get("execution_time"))

        table.add_row(str(i), icon, action_type, duration, desc)

    console.print(table)


def print_scores(run_dir: Path):
    """Print evaluation scores."""
    print_header("EVALUATION SCORES")

    scores_path = run_dir / "scores.json"
    if not scores_path.exists():
        console.print("\n  [dim]No scores.json found.[/dim]")
        return

    try:
        scores = json.loads(scores_path.read_text())
    except Exception as e:
        console.print(f"\n  [red]Error reading scores: {e}[/red]")
        return

    for metric, value in scores.items():
        if isinstance(value, (int, float)):
            # Color score based on value
            if value >= 0.8:
                color = "green"
            elif value >= 0.2:
                color = "yellow"
            else:
                color = "red"
            console.print(f"  [bold]{metric}:[/bold] [{color}]{value:.4f}[/{color}]")
        elif isinstance(value, dict):
            console.print(f"  [bold]{metric}:[/bold]")
            for k, v in value.items():
                if isinstance(v, (int, float)):
                    console.print(f"    {k}: {v:.4f}")


def get_company_name(task_id: str, warn: bool = False) -> str:
    """Get company name from task config (if available)."""
    try:
        task = task_loader.get_task_from_id(task_id)
        config = task.get_task_config()
        return config.get("company_info", {}).get("name", "")
    except Exception as exc:
        if warn:
            console.print(f"[yellow]Warning: failed to load company info for {task_id}: {exc}[/yellow]")
        return ""


def analyze_run(
    run_dir: Path,
    verbose: bool = False,
    quiet: bool = False,
    vector_stores_dir: Optional[Path] = None,
) -> Optional[Dict]:
    """Main analysis function. Returns summary dict if quiet=True."""
    if not quiet:
        console.print(Panel(f"[bold]ANALYZING:[/bold] {run_dir.name}", style="blue"))

    # Find and load action plan
    action_plan_path = find_action_plan(run_dir, vector_stores_dir=vector_stores_dir)
    if not action_plan_path:
        if not quiet:
            console.print(f"\n[red]ERROR: No action_plan_final.json found in {run_dir}[/red]")
            console.print("\n[dim]Files in run directory:[/dim]")
            for p in sorted(run_dir.rglob("*.json"))[:10]:
                console.print(f"  {p.relative_to(run_dir)}")
        return None

    try:
        action_plan = json.loads(action_plan_path.read_text())
    except Exception as e:
        if not quiet:
            console.print(f"[red]ERROR: Failed to parse action plan: {e}[/red]")
        return None

    actions = action_plan.get("actions", [])
    query = action_plan.get("research_query", "N/A")

    # Build summary for batch mode
    status_counts = Counter(a.get("status", "unknown") for a in actions)
    type_counts = Counter(a.get("type", "unknown") for a in actions)
    web_searches = [a for a in actions if a.get("type") == "web_search"]

    # Get task ID and company
    task_id = run_dir.name.split("_")[0] if "_" in run_dir.name else run_dir.name
    company = get_company_name(task_id, warn=not quiet)

    # Load scores
    scores = {}
    scores_path = run_dir / "scores.json"
    if scores_path.exists():
        try:
            scores = json.loads(scores_path.read_text())
        except Exception as exc:
            if not quiet:
                console.print(f"[yellow]Warning: failed to read scores.json: {exc}[/yellow]")

    summary = {
        "task_id": task_id,
        "company": company,
        "run_dir": run_dir,
        "total_actions": len(actions),
        "completed": status_counts.get("completed", 0),
        "failed": status_counts.get("failed", 0),
        "pending": status_counts.get("pending", 0),
        "web_searches": len(web_searches),
        "web_completed": sum(1 for a in web_searches if a.get("status") == "completed"),
        "local_searches": type_counts.get("local_document_search", 0),
        "scores": scores,
    }

    if quiet:
        return summary

    # Show research query in full with wrapping
    console.print(f"\n[bold]Research Query:[/bold]")
    wrapped_query = textwrap.fill(query, width=76, initial_indent="  ", subsequent_indent="  ")
    console.print(f"[cyan]{wrapped_query}[/cyan]")

    iterations = f"{action_plan.get('current_iteration', '?')}/{action_plan.get('max_iterations', '?')}"
    console.print(f"\n[bold]Iterations:[/bold] {iterations}")

    # Print sections
    print_action_summary(actions)
    print_external_calls(actions)
    print_local_searches(actions)
    if verbose:
        print_all_actions(actions, verbose=verbose)
    print_scores(run_dir)

    console.print()
    return summary


def find_latest_batch(runs_dir: Path = DEFAULT_RUNS_DIR) -> Optional[Path]:
    """Find the most recent batch directory."""
    if not runs_dir.exists():
        return None

    batch_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")]
    if not batch_dirs:
        return None

    return max(batch_dirs, key=lambda d: d.stat().st_mtime)


def analyze_batch(batch_dir: Path, verbose: bool = False, vector_stores_dir: Optional[Path] = None):
    """Analyze all runs in a batch directory and print summary table."""
    console.print(Panel(f"[bold]BATCH ANALYSIS:[/bold] {batch_dir.name}", style="blue"))

    # Find all task directories in the batch
    task_dirs = sorted([d for d in batch_dir.iterdir() if d.is_dir()])

    if not task_dirs:
        console.print("[red]No task directories found in batch[/red]")
        return

    summaries = []
    for task_dir in task_dirs:
        if verbose:
            summary = analyze_run(task_dir, verbose=verbose, quiet=False, vector_stores_dir=vector_stores_dir)
        else:
            summary = analyze_run(task_dir, quiet=True, vector_stores_dir=vector_stores_dir)
        if summary:
            summaries.append(summary)

    # Print summary table
    print_batch_summary(summaries, batch_dir)


def print_batch_summary(summaries: List[Dict], batch_dir: Path):
    """Print a summary table of all tasks in a batch with all 5 paper metrics."""
    print_header("BATCH SUMMARY")

    # First table: Action statistics
    action_table = Table(title=f"Actions: {batch_dir.name}", show_header=True, header_style="bold")
    action_table.add_column("Task", style="cyan", width=8)
    action_table.add_column("Company", width=18)
    action_table.add_column("Actions", justify="right", width=8)
    action_table.add_column("Status", justify="center", width=16)
    action_table.add_column("Web", justify="right", width=7)
    action_table.add_column("Local", justify="right", width=6)

    total_actions = 0
    total_completed = 0
    total_failed = 0
    total_pending = 0
    total_web = 0
    total_web_ok = 0
    total_local = 0

    # Metrics aggregation
    metrics_agg = {
        "insights_recall": [],
        "factuality": [],
        "distractor_avoidance": [],
        "report_quality": [],
        "harmonic_mean": [],
    }

    current_company = None

    for s in summaries:
        total_actions += s["total_actions"]
        total_completed += s["completed"]
        total_failed += s["failed"]
        total_pending += s["pending"]
        total_web += s["web_searches"]
        total_web_ok += s["web_completed"]
        total_local += s["local_searches"]

        company = s.get("company", "")
        if current_company is not None and company != current_company:
            action_table.add_section()
        current_company = company

        status_parts = [f"[green]{s['completed']} OK[/green]"]
        if s["failed"]:
            status_parts.append(f"[red]{s['failed']} X[/red]")
        if s["pending"]:
            status_parts.append(f"[yellow]{s['pending']} o[/yellow]")
        status = " ".join(status_parts)

        web_str = f"{s['web_completed']}/{s['web_searches']}" if s["web_searches"] else "-"
        company_display = company[:16] + ".." if len(company) > 18 else company

        action_table.add_row(
            s["task_id"],
            company_display,
            str(s["total_actions"]),
            status,
            web_str,
            str(s["local_searches"]),
        )

        # Collect metrics
        scores = s.get("scores", {})
        for metric in metrics_agg:
            val = scores.get(metric)
            # Handle distractor_avoidance from distractor_recall
            if val is None and metric == "distractor_avoidance" and "distractor_recall" in scores:
                val = 1.0 - scores["distractor_recall"]
            if val is not None:
                metrics_agg[metric].append(val)

    # Add totals row to action table
    action_table.add_section()
    action_table.add_row(
        "[bold]TOTAL[/bold]",
        "",
        f"[bold]{total_actions}[/bold]",
        f"[green]{total_completed} OK[/green] [red]{total_failed} X[/red] [yellow]{total_pending} o[/yellow]",
        f"{total_web_ok}/{total_web}",
        str(total_local),
    )

    console.print(action_table)
    console.print()

    # Second table: Paper metrics (matching DrBench paper format)
    def format_score(val):
        if val is None:
            return "[dim]-[/dim]"
        if val >= 0.8:
            return f"[green]{val:.3f}[/green]"
        elif val >= 0.2:
            return f"[yellow]{val:.3f}[/yellow]"
        else:
            return f"[red]{val:.3f}[/red]"

    run_model = infer_run_model(batch_dir, batch_dir=batch_dir)
    scoring_model = None
    for s in summaries:
        scoring_model = infer_scoring_model(s.get("scores", {}))
        if scoring_model:
            break

    metrics_title = Text("Metrics (Paper Format)", style="bold")
    metrics_title.append("\n")
    display_run_model = run_model or UNKNOWN_MODEL
    display_scoring_model = scoring_model or run_model or UNKNOWN_MODEL
    metrics_title.append(
        f"Run model: {display_run_model}   Scoring model: {display_scoring_model}",
        style="dim",
    )

    metrics_table = Table(title=metrics_title, show_header=True, header_style="bold")
    metrics_table.add_column("Task", style="cyan", width=8)
    metrics_table.add_column("Insight\nRecall", justify="right", width=9)
    metrics_table.add_column("Factuality", justify="right", width=10)
    metrics_table.add_column("Distractor\nAvoid", justify="right", width=10)
    metrics_table.add_column("Report\nQuality", justify="right", width=9)
    metrics_table.add_column("Harmonic\nMean", justify="right", width=9)

    current_company = None
    for s in summaries:
        company = s.get("company", "")
        if current_company is not None and company != current_company:
            metrics_table.add_section()
        current_company = company

        scores = s.get("scores", {})
        ir = scores.get("insights_recall")
        fa = scores.get("factuality")
        da = scores.get("distractor_avoidance")
        if da is None and "distractor_recall" in scores:
            da = 1.0 - scores["distractor_recall"]
        rq = scores.get("report_quality")
        hm = scores.get("harmonic_mean")

        metrics_table.add_row(
            s["task_id"],
            format_score(ir),
            format_score(fa),
            format_score(da),
            format_score(rq),
            format_score(hm),
        )

    # Add averages row
    metrics_table.add_section()
    avg = lambda lst: sum(lst) / len(lst) if lst else None
    metrics_table.add_row(
        "[bold]AVG[/bold]",
        format_score(avg(metrics_agg["insights_recall"])),
        format_score(avg(metrics_agg["factuality"])),
        format_score(avg(metrics_agg["distractor_avoidance"])),
        format_score(avg(metrics_agg["report_quality"])),
        format_score(avg(metrics_agg["harmonic_mean"])),
    )

    console.print(metrics_table)

    # Print aggregate stats
    console.print(f"\n[bold]Tasks analyzed:[/bold] {len(summaries)}")
    if metrics_agg["harmonic_mean"]:
        hm_vals = metrics_agg["harmonic_mean"]
        console.print(f"[bold]Harmonic Mean:[/bold] {avg(hm_vals):.4f} (range: {min(hm_vals):.4f} - {max(hm_vals):.4f})")
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Analyze a DrBench run")
    parser.add_argument("run_dir", nargs="?", help="Path to run or batch directory")
    parser.add_argument("--latest", action="store_true", help="Analyze the most recent run")
    parser.add_argument("--batch", action="store_true", help="Analyze latest batch (summary table)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show all actions")
    parser.add_argument("-a", "--all", action="store_true", help="Analyze all runs")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Runs directory (default: ./runs)")
    parser.add_argument(
        "--vector-stores-dir",
        type=Path,
        default=DEFAULT_VECTOR_STORES_DIR,
        help="Vector stores directory for older runs",
    )
    parser.add_argument("--data-dir", type=Path, help="Override DRBENCH_DATA_DIR")

    args = parser.parse_args()

    if args.data_dir:
        import os
        os.environ["DRBENCH_DATA_DIR"] = str(args.data_dir)

    if args.batch:
        # Batch mode - analyze a batch directory
        if args.run_dir:
            batch_dir = Path(args.run_dir)
        else:
            batch_dir = find_latest_batch(args.runs_dir)
        if not batch_dir or not batch_dir.exists():
            console.print("[red]ERROR: No batch directory found[/red]")
            sys.exit(1)
        analyze_batch(batch_dir, verbose=args.verbose, vector_stores_dir=args.vector_stores_dir)
    elif args.all:
        runs_dir = args.runs_dir
        if not runs_dir.exists():
            console.print("[red]No runs directory found[/red]")
            sys.exit(1)
        for run_dir in sorted(runs_dir.iterdir()):
            if run_dir.is_dir():
                analyze_run(run_dir, verbose=args.verbose, vector_stores_dir=args.vector_stores_dir)
    elif args.latest or not args.run_dir:
        run_dir = find_latest_run(args.runs_dir)
        if not run_dir:
            console.print(f"[red]ERROR: No runs found in {args.runs_dir}[/red]")
            sys.exit(1)
        if run_dir.name.startswith("batch_"):
            task_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("DR")]
            if task_dirs:
                run_dir = max(task_dirs, key=lambda d: d.stat().st_mtime)
        analyze_run(run_dir, verbose=args.verbose, vector_stores_dir=args.vector_stores_dir)
    else:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            console.print(f"[red]ERROR: Directory not found: {run_dir}[/red]")
            sys.exit(1)
        # Check if it's a batch directory
        if run_dir.name.startswith("batch_"):
            analyze_batch(run_dir, verbose=args.verbose, vector_stores_dir=args.vector_stores_dir)
        else:
            analyze_run(run_dir, verbose=args.verbose, vector_stores_dir=args.vector_stores_dir)


if __name__ == "__main__":
    main()
