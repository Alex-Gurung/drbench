#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from drbench.agents.drbench_agent.agent_tools.content_processor import ContentProcessor  # noqa: E402
from making_dataset.utils.progress import progress  # noqa: E402


@dataclass
class ChunkConfig:
    target_words: int = 450
    overlap_words: int = 50
    min_words: int = 300


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and chunk local DrBench files.")
    parser.add_argument(
        "--tasks-root",
        default=str(ROOT_DIR / "drbench" / "data" / "tasks"),
        help="Root tasks directory",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "chunks_local.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--workspace-dir",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "workspace"),
        help="Workspace dir for ContentProcessor",
    )
    parser.add_argument(
        "--target-words",
        type=int,
        default=450,
        help="Approx target words per chunk",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=50,
        help="Approx overlap words between chunks",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=300,
        help=(
            "Try to merge/pack chunks until they are at least this many words when possible "
            "(default: 300)."
        ),
    )
    parser.add_argument(
        "--prefer-md",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer .md over .pdf when both exist",
    )
    parser.add_argument(
        "--include-tasks",
        nargs="*",
        default=None,
        help="Optional list of task IDs to include (e.g., DR0001 DR0002)",
    )
    return parser.parse_args()


def _list_task_dirs(tasks_root: Path, include: Optional[List[str]]) -> List[Path]:
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


def _slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value


def _load_company_map(task_dirs: List[Path]) -> dict[str, dict[str, str]]:
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
            mapping[task_id] = {
                "company_name": name,
                "company": _slugify(name),
            }
    return mapping


def _load_file_dict(folder: Path) -> Optional[dict]:
    file_dict = folder / "file_dict.json"
    if not file_dict.exists():
        return None
    try:
        return json.loads(file_dict.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _should_skip(file_path: Path) -> bool:
    return file_path.name in {"qa_dict.json", "file_dict.json"}


def _prefer_md(file_path: Path) -> bool:
    if file_path.suffix.lower() != ".pdf":
        return False
    md_path = file_path.with_suffix(".md")
    return md_path.exists()


def _split_markdown_sections(text: str) -> List[tuple[str, str]]:
    title = None
    h2 = None
    h3 = None
    current_lines: List[str] = []
    sections: List[tuple[str, str]] = []

    def flush():
        nonlocal current_lines
        content = "\n".join(current_lines).strip()
        if content:
            parts = [p for p in [title, h2, h3] if p]
            prefix = " > ".join(parts) if parts else ""
            sections.append((prefix, content))
        current_lines = []

    for line in text.splitlines():
        if line.startswith("# "):
            flush()
            title = line[2:].strip()
            continue
        if line.startswith("## "):
            flush()
            h2 = line[3:].strip()
            h3 = None
            continue
        if line.startswith("### "):
            flush()
            h3 = line[4:].strip()
            continue
        current_lines.append(line)

    flush()
    return sections


def _paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _chunk_paragraphs(paragraphs: List[str], cfg: ChunkConfig) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    def count_words(s: str) -> int:
        return len(s.split())

    for para in paragraphs:
        words = count_words(para)
        # Only flush once we've hit a reasonable minimum size. This prevents lots of tiny chunks
        # when a document has many small Markdown sections.
        if current and current_words >= cfg.min_words and current_words + words > cfg.target_words:
            joined = "\n\n".join(current).strip()
            if joined:
                chunks.append(joined)
            # overlap: carry last overlap_words
            tail_words = " ".join(current).split()[-cfg.overlap_words :]
            current = [" ".join(tail_words)] if tail_words else []
            current_words = len(tail_words)
        current.append(para)
        current_words += words

    if current:
        joined = "\n\n".join(current).strip()
        if joined:
            chunks.append(joined)

    # If the final chunk is very short, try to merge it back into the previous one to avoid
    # leaving a low-context tail. Do this only when the merged chunk would remain reasonable.
    if len(chunks) >= 2:
        last = chunks[-1]
        last_words = count_words(last)
        if last_words < cfg.min_words:
            merged = chunks[-2].rstrip() + "\n\n" + last.lstrip()
            merged_words = count_words(merged)
            # Allow some growth over target_words, but don't create giant chunks.
            max_words = cfg.target_words + cfg.min_words
            if merged_words <= max_words:
                chunks[-2] = merged
                chunks.pop()

    return chunks


def _chunk_text(text: str, cfg: ChunkConfig) -> List[str]:
    paragraphs = _paragraphs(text)
    if not paragraphs:
        return []
    return _chunk_paragraphs(paragraphs, cfg)


def _emit_chunks(
    doc_id: str,
    text: str,
    source_type: str,
    cfg: ChunkConfig,
    meta: dict,
) -> Iterable[dict]:
    if meta.get("extension") == ".md":
        sections = _split_markdown_sections(text)
        if not sections:
            md_text = text
        else:
            # Represent each Markdown section as:
            #   <Title > H2 > H3>
            #   <section body>
            # and then chunk the concatenated doc. This allows merging multiple short sections
            # into a single chunk (min_words), while keeping headings informative.
            parts: List[str] = []
            for prefix, section_text in sections:
                if prefix:
                    parts.append(prefix)
                parts.append(section_text)
            md_text = "\n\n".join(parts)
        chunks = _chunk_text(md_text, cfg)
    else:
        chunks = _chunk_text(text, cfg)

    for idx, chunk in enumerate(chunks, start=1):
        chunk_id = f"{doc_id}#{idx:04d}"
        yield {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "source_type": source_type,
            "text": chunk,
            "offsets": None,
            "meta": meta,
        }


def _collect_files(task_dirs: List[Path], prefer_md: bool) -> List[tuple[Path, dict]]:
    items: List[tuple[Path, dict]] = []
    for task_dir in task_dirs:
        files_dir = task_dir / "files"
        if not files_dir.exists():
            continue
        for folder in sorted(files_dir.iterdir()):
            if not folder.is_dir():
                continue
            file_meta = _load_file_dict(folder) or {}
            for file_path in sorted(folder.iterdir()):
                if not file_path.is_file():
                    continue
                if _should_skip(file_path):
                    continue
                if prefer_md and _prefer_md(file_path):
                    continue
                items.append((file_path, file_meta))
    return items


def main() -> int:
    args = _parse_args()
    tasks_root = Path(args.tasks_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = ChunkConfig(
        target_words=args.target_words,
        overlap_words=args.overlap_words,
        min_words=args.min_words,
    )
    processor = ContentProcessor(workspace_dir=args.workspace_dir, model="none")

    task_dirs = _list_task_dirs(tasks_root, args.include_tasks)
    if not task_dirs:
        print("No tasks found to process.")
        return 1

    files_to_process = _collect_files(task_dirs, args.prefer_md)
    company_map = _load_company_map(task_dirs)

    with output_path.open("w", encoding="utf-8") as out:
        for file_path, file_meta in progress(
            files_to_process, total=len(files_to_process), desc="Local files"
        ):
            relative = file_path.relative_to(tasks_root)
            parts = list(relative.parts)
            if len(parts) < 4 or parts[1] != "files":
                continue
            task_id = parts[0]
            subdir = parts[2]
            filename = parts[-1]
            doc_id = f"local/{task_id}/{subdir}/{filename}"

            try:
                text = processor.extract_text_from_file(file_path)
            except Exception as exc:
                raise RuntimeError(f"Failed to extract {file_path}: {exc}")

            if not text.strip():
                continue

            meta = {
                "task_id": task_id,
                "subdir": subdir,
                "filename": filename,
                "relative_path": str(relative),
                "extension": file_path.suffix.lower(),
            }
            company_info = company_map.get(task_id)
            if company_info:
                meta.update(company_info)
            if file_meta:
                meta.update({
                    "file_title": file_meta.get("file_title"),
                    "file_name": file_meta.get("file_name"),
                    "introduction": file_meta.get("introduction"),
                })

            for record in _emit_chunks(doc_id, text, "local", cfg, meta):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
