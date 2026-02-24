#!/usr/bin/env python3
"""Build deduplicated local document store (docs_local.jsonl).

Extracts full document text from all DrBench task files, deduplicating
format variants (.pdf vs .md, .jsonl vs .txt).

Usage:
    python -m making_dataset.data_prep.build_docs_local
    python -m making_dataset.data_prep.build_docs_local --include-tasks DR0001 DR0002

Output: making_dataset/outputs/docs_local.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from making_dataset.utils.progress import progress  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deduplicated local docs.")
    parser.add_argument(
        "--tasks-root",
        default=str(ROOT_DIR / "drbench" / "data" / "tasks"),
        help="Root tasks directory",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "docs_local.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--include-tasks",
        nargs="*",
        default=None,
        help="Optional list of task IDs to include (e.g., DR0001 DR0002)",
    )
    return parser.parse_args()


def _list_task_dirs(tasks_root: Path, include: Optional[List[str]]) -> List[Path]:
    """List task directories matching DR#### pattern."""
    tasks = []
    for entry in sorted(tasks_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == "SANITY0":
            continue
        if include and entry.name not in include:
            continue
        if re.fullmatch(r"DR\d{4}", entry.name):
            tasks.append(entry)
    return tasks


def _load_company_map(task_dirs: List[Path]) -> dict[str, dict[str, str]]:
    """Load company info from context.json for each task."""
    mapping: dict[str, dict[str, str]] = {}
    for task_dir in task_dirs:
        task_id = task_dir.name
        context_path = task_dir / "context.json"
        if not context_path.exists():
            continue
        try:
            context = json.loads(context_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        company_info = context.get("company_info") or {}
        name = company_info.get("name") or ""
        if name:
            mapping[task_id] = {"company_name": name}
    return mapping


def _should_skip(file_path: Path) -> bool:
    """Skip metadata files."""
    return file_path.name in {"qa_dict.json", "file_dict.json"}


def _extract_text(file_path: Path) -> str:
    """Extract text from .md, .txt, .jsonl, or .pdf files."""
    ext = file_path.suffix.lower()
    if ext in {".md", ".txt"}:
        return file_path.read_text(encoding="utf-8")
    if ext == ".jsonl":
        lines = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if "user" in obj and "message" in obj:
                lines.append(f"{obj['user']}: {obj['message']}")
            elif "from" in obj and "body" in obj:
                lines.append(f"From: {obj.get('from', '')}")
                lines.append(f"To: {obj.get('to', '')}")
                if obj.get("subject"):
                    lines.append(f"Subject: {obj['subject']}")
                lines.append(f"\n{obj['body']}\n")
            else:
                lines.append(json.dumps(obj))
        return "\n".join(lines)
    if ext == ".pdf":
        import pymupdf

        doc = pymupdf.open(str(file_path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)
    return ""


def _collect_files(task_dirs: List[Path]) -> List[tuple[Path, str]]:
    """Collect files with deduplication of format variants.

    Deduplication rules:
    - Skip .pdf if .md exists (prefer markdown)
    - Skip .jsonl if .txt exists (prefer human-readable)
    """
    items: List[tuple[Path, str]] = []
    for task_dir in task_dirs:
        task_id = task_dir.name
        files_dir = task_dir / "files"
        if not files_dir.exists():
            continue
        for folder in sorted(files_dir.iterdir()):
            if not folder.is_dir():
                continue
            for file_path in sorted(folder.iterdir()):
                if not file_path.is_file():
                    continue
                if _should_skip(file_path):
                    continue
                ext = file_path.suffix.lower()
                # Skip .pdf if .md exists
                if ext == ".pdf" and file_path.with_suffix(".md").exists():
                    continue
                # Skip .jsonl if .txt exists
                if ext == ".jsonl" and file_path.with_suffix(".txt").exists():
                    continue
                items.append((file_path, task_id))
    return items


def main() -> int:
    args = _parse_args()
    tasks_root = Path(args.tasks_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_dirs = _list_task_dirs(tasks_root, args.include_tasks)
    if not task_dirs:
        print("No tasks found to process.")
        return 1

    files_to_process = _collect_files(task_dirs)
    company_map = _load_company_map(task_dirs)

    doc_count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for file_path, task_id in progress(
            files_to_process, total=len(files_to_process), desc="Building docs"
        ):
            relative = file_path.relative_to(tasks_root)
            parts = list(relative.parts)
            if len(parts) < 4 or parts[1] != "files":
                continue
            subdir = parts[2]
            filename = parts[-1]
            doc_id = f"local/{task_id}/{subdir}/{filename}"

            try:
                text = _extract_text(file_path)
            except Exception as exc:
                print(f"Warning: Failed to extract {file_path}: {exc}")
                continue

            if not text.strip():
                continue

            record = {
                "doc_id": doc_id,
                "task_id": task_id,
                "company_name": company_map.get(task_id, {}).get("company_name", ""),
                "subdir": subdir,
                "filename": filename,
                "text": text,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            doc_count += 1

    print(f"Wrote {doc_count} documents to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
