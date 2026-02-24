#!/usr/bin/env python3
from __future__ import annotations

"""Chunk extracted DRBench seed-URL docs into passage-sized retrieval units.

Input:
- `making_dataset_2/outputs/docs_web_drbench_urls.jsonl` (from fetch_drbench_urls.py)

Output:
- `making_dataset_2/outputs/chunks_web_drbench_urls.jsonl`

Chunking is paragraph/heading aware and emits:
- `chunk_id = f"{doc_id}#{idx:04d}"`
- `offsets` character spans into the canonical doc text
- A heading prefix in `text` when available (`Title > H2 > H3`) to help retrieval.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from making_dataset.utils.progress import progress  # noqa: E402
from making_dataset_2.drbench_urls import WEB_POOL  # noqa: E402


DEFAULT_DOCS = ROOT_DIR / "making_dataset_2" / "outputs" / "docs_web_drbench_urls.jsonl"
DEFAULT_OUTPUT = ROOT_DIR / "making_dataset_2" / "outputs" / "chunks_web_drbench_urls.jsonl"


@dataclass(frozen=True)
class ParaBlock:
    text: str
    start: int
    end: int
    title: str = ""
    h2: str = ""
    h3: str = ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk drbench seed URL docs into passage chunks.")
    parser.add_argument("--docs", default=str(DEFAULT_DOCS), help="Docs JSONL produced by fetch_drbench_urls.py")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output chunks JSONL")
    parser.add_argument("--target-words", type=int, default=260, help="Approx target words per chunk")
    parser.add_argument("--overlap-words", type=int, default=50, help="Approx overlap words between chunks")
    parser.add_argument("--min-words", type=int, default=140, help="Try to keep chunks at least this many words")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on docs to process")
    return parser.parse_args()


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines).strip()
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def _load_latest_success_docs(path: Path) -> list[dict[str, Any]]:
    """Load docs JSONL and keep the latest successful record per doc_id."""
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = obj.get("doc_id")
            if not doc_id:
                continue
            if doc_id not in by_id:
                order.append(doc_id)
            text = (obj.get("text") or "").strip()
            err = obj.get("error")
            status = obj.get("http_status")
            if text and not err and isinstance(status, int) and 200 <= status < 300:
                by_id[doc_id] = obj
    docs: list[dict[str, Any]] = [by_id[doc_id] for doc_id in order if doc_id in by_id]
    return docs


def _parse_paragraph_blocks(text: str) -> list[ParaBlock]:
    """Parse a markdown-ish doc into paragraph blocks with heading context and offsets."""
    blocks: list[ParaBlock] = []

    title = ""
    h2 = ""
    h3 = ""

    pos = 0
    current_lines: list[str] = []
    para_start: Optional[int] = None

    def flush_para(end_pos: int) -> None:
        nonlocal current_lines, para_start
        if not current_lines or para_start is None:
            current_lines = []
            para_start = None
            return
        para_text = "\n".join(current_lines).strip()
        if para_text:
            blocks.append(
                ParaBlock(
                    text=para_text,
                    start=para_start,
                    end=end_pos,
                    title=title,
                    h2=h2,
                    h3=h3,
                )
            )
        current_lines = []
        para_start = None

    for line in text.splitlines(keepends=True):
        line_start = pos
        pos += len(line)
        s = line.strip()
        if not s:
            flush_para(pos)
            continue
        if s.startswith("#"):
            # heading line
            flush_para(line_start)
            level = len(s) - len(s.lstrip("#"))
            if level in {1, 2, 3} and len(s) > level and s[level] == " ":
                heading_text = s[level + 1 :].strip()
                if level == 1:
                    title = heading_text
                    h2 = ""
                    h3 = ""
                elif level == 2:
                    h2 = heading_text
                    h3 = ""
                else:
                    h3 = heading_text
            continue

        if para_start is None:
            para_start = line_start
        current_lines.append(s)

    flush_para(pos)
    return blocks


def _count_words(text: str) -> int:
    return len(text.split())


def _chunk_blocks(
    blocks: list[ParaBlock],
    *,
    target_words: int,
    overlap_words: int,
    min_words: int,
) -> list[list[ParaBlock]]:
    chunks: list[list[ParaBlock]] = []
    current: list[ParaBlock] = []
    current_words = 0

    def carry_overlap(prev: list[ParaBlock]) -> list[ParaBlock]:
        if overlap_words <= 0:
            return []
        tail: list[ParaBlock] = []
        tail_words = 0
        for b in reversed(prev):
            tail.insert(0, b)
            tail_words += _count_words(b.text)
            if tail_words >= overlap_words:
                break
        return tail

    for b in blocks:
        w = _count_words(b.text)
        if current and current_words >= min_words and current_words + w > target_words:
            chunks.append(current)
            current = carry_overlap(current)
            current_words = sum(_count_words(x.text) for x in current)
        current.append(b)
        current_words += w

    if current:
        chunks.append(current)

    # If final chunk is too small, try merging into previous.
    if len(chunks) >= 2:
        last = chunks[-1]
        last_words = sum(_count_words(x.text) for x in last)
        if last_words < min_words:
            merged = chunks[-2] + last
            merged_words = sum(_count_words(x.text) for x in merged)
            if merged_words <= target_words + min_words:
                chunks[-2] = merged
                chunks.pop()

    return chunks


def _heading_prefix(block: ParaBlock) -> str:
    parts = [p for p in [block.title, block.h2, block.h3] if p]
    return " > ".join(parts)


def _emit_chunk_records(
    *,
    doc: dict[str, Any],
    blocks: list[ParaBlock],
    target_words: int,
    overlap_words: int,
    min_words: int,
) -> list[dict[str, Any]]:
    doc_id = doc["doc_id"]
    meta_base = {
        "web_pool": WEB_POOL,
        "url": doc.get("url"),
        "industry": doc.get("industry"),
        "domain": doc.get("domain"),
        "seed_date": doc.get("seed_date"),
        "task_ids": doc.get("task_ids") or [],
        "title": doc.get("title") or "",
    }

    chunk_groups = _chunk_blocks(blocks, target_words=target_words, overlap_words=overlap_words, min_words=min_words)
    records: list[dict[str, Any]] = []
    for idx, group in enumerate(chunk_groups, start=1):
        if not group:
            continue
        prefix = _heading_prefix(group[0])
        body = "\n\n".join(b.text.strip() for b in group if b.text.strip())
        if prefix:
            chunk_text = prefix + "\n\n" + body
        else:
            chunk_text = body
        chunk_text = _normalize_text(chunk_text)
        if not chunk_text:
            continue
        records.append(
            {
                "chunk_id": f"{doc_id}#{idx:04d}",
                "doc_id": doc_id,
                "source_type": "web",
                "text": chunk_text,
                "offsets": {"start": int(group[0].start), "end": int(group[-1].end)},
                "meta": dict(meta_base),
            }
        )
    return records


def main() -> int:
    args = _parse_args()
    docs_path = Path(args.docs)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    docs = _load_latest_success_docs(docs_path)
    if args.limit is not None:
        docs = docs[: max(0, int(args.limit))]

    total_chunks = 0
    with output_path.open("w", encoding="utf-8") as out:
        for doc in progress(docs, total=len(docs), desc=f"Chunk {WEB_POOL}"):
            text = _normalize_text(doc.get("text") or "")
            if not text:
                continue
            blocks = _parse_paragraph_blocks(text)
            if not blocks:
                continue
            recs = _emit_chunk_records(
                doc=doc,
                blocks=blocks,
                target_words=int(args.target_words),
                overlap_words=int(args.overlap_words),
                min_words=int(args.min_words),
            )
            for rec in recs:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Wrote {total_chunks} chunks to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
