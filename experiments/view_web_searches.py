#!/usr/bin/env python3
"""
View web searches from batch runs.

Usage:
    python view_web_searches.py              # Interactive: pick batch and task
    python view_web_searches.py --latest     # Latest batch, all tasks
    python view_web_searches.py --task DR0001
    python view_web_searches.py --batch /path/to/batch --task DR0005
"""

import argparse
import json
import re
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import box

from drbench import task_loader

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
console = Console()


def get_batches(runs_dir: Path) -> list[Path]:
    return sorted(
        [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("batch_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def get_tasks(batch_dir: Path) -> list[Path]:
    return sorted([p for p in batch_dir.iterdir() if p.is_dir() and p.name.startswith("DR")])


def parse_model_and_timestamp(name: str) -> tuple[str, str | None, str | None]:
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
    if batch_dir.name.startswith("batch_"):
        model, _, _ = parse_model_and_timestamp(batch_dir.name[len("batch_"):])
        return model
    return None


def load_searches(task_dir: Path) -> dict | None:
    path = task_dir / "privacy" / "web_searches.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def load_question_info(task_id: str, task_dir: Path) -> dict | None:
    question_used = task_dir / "question_used.json"
    if question_used.exists():
        with question_used.open() as f:
            return json.load(f)
    try:
        task = task_loader.get_task_from_id(task_id)
        data = task.get_task_config()
    except Exception:
        return None
    return {"task_id": task_id, "question_set": "default", "dr_question": data.get("dr_question")}


def load_action_plan(task_dir: Path, source_path: str | None) -> dict | None:
    candidates = []
    if source_path:
        candidates.append(Path(source_path))
    candidates.extend([
        task_dir / "action_plan_final.json",
        task_dir / "action_plan_initial.json",
    ])
    for path in candidates:
        if path.exists():
            with path.open() as f:
                return json.load(f)
    return None


def build_query_metadata(action_plan: dict | None) -> dict[str, list[dict[str, int | None | str]]]:
    if not action_plan:
        return {}

    metadata_by_query: dict[str, list[dict[str, int | None | str]]] = {}
    for action in action_plan.get("actions", []):
        if action.get("type") != "web_search":
            continue
        query = action.get("parameters", {}).get("query")
        if not query:
            continue
        created_iter = action.get("created_in_iteration")
        if not isinstance(created_iter, int):
            action_id = action.get("id", "")
            match = re.match(r"adaptive_(\d+)_", action_id)
            if match:
                created_iter = int(match.group(1)) + 1

        completed_iter = action.get("iteration_completed")
        if isinstance(completed_iter, int):
            completed_iter = completed_iter + 1
        else:
            completed_iter = None

        metadata_by_query.setdefault(query, []).append(
            {
                "created_in_iteration": created_iter if isinstance(created_iter, int) else None,
                "iteration_completed": completed_iter,
                "status": action.get("status"),
            }
        )
    return metadata_by_query


def format_status(status: str | None) -> str:
    if status == "completed":
        return "OK"
    if status == "failed":
        return "X"
    if status in {"pending", "in_progress"}:
        return "o"
    return "?"


def display_searches(task_id: str, data: dict, task_dir: Path, batch_model: str | None) -> None:
    searches = data.get("searches", [])
    action_plan = load_action_plan(task_dir, data.get("source"))
    metadata_by_query = build_query_metadata(action_plan)

    # Get DR question
    question_info = load_question_info(task_id, task_dir) or {}
    dr_question = question_info.get("dr_question")

    lines = []
    if dr_question:
        lines.append(f"[bold cyan]Question:[/bold cyan] {dr_question}")
        question_set = question_info.get("question_set")
        if question_set:
            lines.append(f"[dim]Question set:[/dim] {question_set}")
        if batch_model:
            lines.append(f"[dim]Query model:[/dim] {batch_model}")
        lines.append("")
    elif batch_model:
        lines.append(f"[dim]Query model:[/dim] {batch_model}")
        lines.append("")

    if not searches:
        console.print(Panel(
            "\n".join(lines) if lines else "(no searches)",
            title=f"{task_id} - 0 web searches",
            box=box.ROUNDED,
        ))
        return

    table = Table(show_header=True, header_style="bold", box=box.SIMPLE, padding=(0, 1))
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Status", style="dim", width=6, justify="center")
    table.add_column("Query", overflow="fold")
    table.add_column("Created", style="dim", width=7, justify="right")
    table.add_column("Completed", style="dim", width=9, justify="right")

    for i, s in enumerate(searches, 1):
        query = s.get("query", "")
        meta_queue = metadata_by_query.get(query, [])
        meta = meta_queue.pop(0) if meta_queue else {}
        created_label = (
            str(meta.get("created_in_iteration"))
            if isinstance(meta.get("created_in_iteration"), int)
            else "?"
        )
        completed_label = (
            str(meta.get("iteration_completed"))
            if isinstance(meta.get("iteration_completed"), int)
            else "?"
        )
        status_label = format_status(meta.get("status") or s.get("status"))
        table.add_row(str(i), status_label, query, created_label, completed_label)

    group_items = []
    if lines:
        group_items.append("\n".join(lines))
    group_items.append(table)

    console.print(Panel(
        Group(*group_items),
        title=f"{task_id} - {len(searches)} web searches",
        box=box.ROUNDED,
    ))


def pick_batch(batches: list[Path]) -> Path | None:
    console.print("\n[bold]Available batches:[/bold]")
    for i, b in enumerate(batches, 1):
        console.print(f"  {i}. {b.name}")
    console.print()
    choice = Prompt.ask("Select batch", default="1")
    try:
        return batches[int(choice) - 1]
    except (ValueError, IndexError):
        return None


def pick_task(tasks: list[Path]) -> Path | None:
    console.print("\n[bold]Available tasks:[/bold]")
    for i, t in enumerate(tasks, 1):
        console.print(f"  {i}. {t.name}")
    console.print(f"  0. All tasks")
    console.print()
    choice = Prompt.ask("Select task", default="0")
    if choice == "0":
        return None  # all tasks
    try:
        return tasks[int(choice) - 1]
    except (ValueError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser(description="View web searches from batch runs")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--batch", type=Path, help="Batch directory")
    parser.add_argument("--latest", action="store_true", help="Use latest batch")
    parser.add_argument("--task", help="Specific task ID (e.g., DR0001)")
    parser.add_argument("--data-dir", type=Path, help="Override DRBENCH_DATA_DIR")
    args = parser.parse_args()

    if args.data_dir:
        import os
        os.environ["DRBENCH_DATA_DIR"] = str(args.data_dir)

    # Pick batch
    if args.batch:
        batch_dir = args.batch
    elif args.latest:
        batches = get_batches(args.runs_dir)
        if not batches:
            console.print("[red]No batches found[/red]")
            return
        batch_dir = batches[0]
    else:
        batches = get_batches(args.runs_dir)
        if not batches:
            console.print("[red]No batches found[/red]")
            return
        batch_dir = pick_batch(batches)
        if not batch_dir:
            return

    batch_model = infer_batch_model(batch_dir)
    console.print(f"\n[bold cyan]Batch:[/bold cyan] {batch_dir.name}\n")

    tasks = get_tasks(batch_dir)
    if not tasks:
        console.print("[red]No tasks found[/red]")
        return

    # Pick task(s)
    if args.task:
        selected = [t for t in tasks if t.name == args.task]
        if not selected:
            console.print(f"[red]Task {args.task} not found[/red]")
            return
    elif args.latest or args.batch:
        selected = tasks  # show all if batch specified
    else:
        single = pick_task(tasks)
        selected = [single] if single else tasks

    # Display
    for task_dir in selected:
        data = load_searches(task_dir)
        if data:
            display_searches(task_dir.name, data, task_dir, batch_model)
        else:
            console.print(f"[dim]{task_dir.name}: no web_searches.json (run backfill_web_searches.py)[/dim]")


if __name__ == "__main__":
    main()
