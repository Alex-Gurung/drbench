#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit local chunk sizes (words/chars) to catch overly short or overly large chunks."
    )
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "chunks_local.jsonl"),
        help="Local chunks JSONL",
    )
    parser.add_argument(
        "--out-json",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "local_chunk_audit.json"),
        help="Output JSON report",
    )
    parser.add_argument(
        "--out-md",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "local_chunk_audit.md"),
        help="Output Markdown report",
    )
    parser.add_argument(
        "--word-thresholds",
        type=int,
        nargs="*",
        default=[50, 100, 200, 400],
        help="Word-count thresholds to count under",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top K largest chunks to report (by char length)",
    )
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _quantiles(values: List[int], qs: List[float]) -> Dict[str, int]:
    if not values:
        return {}
    a = sorted(values)
    n = len(a)
    out: Dict[str, int] = {}
    for q in qs:
        idx = int(round((n - 1) * q))
        out[str(q)] = int(a[idx])
    return out


def _write_md(path: Path, report: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Local Chunk Size Audit")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"Total local chunks: {report['total_chunks']}")
    lines.append("")
    lines.append("### Words")
    lines.append("")
    lines.append(f"- avg_words: {report['avg_words']}")
    lines.append(f"- quantiles: {report['words_quantiles']}")
    lines.append("")
    lines.append("### Chars")
    lines.append("")
    lines.append(f"- avg_chars: {report['avg_chars']}")
    lines.append(f"- quantiles: {report['chars_quantiles']}")
    lines.append("")
    lines.append("### Under Word Thresholds")
    lines.append("")
    lines.append("| Threshold | Count | Percent |")
    lines.append("| --- | ---: | ---: |")
    for t, stats in report["under_word_thresholds"].items():
        lines.append(f"| < {t} | {stats['count']} | {stats['percent']:.2f}% |")
    lines.append("")
    lines.append("## Top Largest Chunks (by chars)")
    lines.append("")
    lines.append("| Rank | Chunk ID | Words | Chars |")
    lines.append("| --- | --- | ---: | ---: |")
    for i, item in enumerate(report["top_largest"], start=1):
        lines.append(f"| {i} | {item['chunk_id']} | {item['words']} | {item['chars']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    words: List[int] = []
    chars: List[int] = []
    largest: List[Dict[str, Any]] = []

    for rec in _iter_jsonl(input_path):
        if rec.get("source_type") != "local":
            continue
        text = rec.get("text") or ""
        if not text.strip():
            continue
        w = len(text.split())
        c = len(text)
        words.append(w)
        chars.append(c)
        largest.append({"chunk_id": rec.get("chunk_id"), "words": w, "chars": c})

    if not words:
        raise ValueError("No local chunks found in input.")

    largest_sorted = sorted(largest, key=lambda x: int(x["chars"]), reverse=True)[: int(args.top_k)]

    thresholds = sorted(set(int(x) for x in (args.word_thresholds or [])))
    under: Dict[str, Any] = {}
    n = len(words)
    for t in thresholds:
        ct = sum(1 for w in words if w < t)
        under[str(t)] = {"count": ct, "percent": (ct / n * 100.0) if n else 0.0}

    report: Dict[str, Any] = {
        "total_chunks": n,
        "avg_words": round(sum(words) / n, 2),
        "avg_chars": round(sum(chars) / n, 2),
        "words_quantiles": _quantiles(words, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]),
        "chars_quantiles": _quantiles(chars, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]),
        "under_word_thresholds": under,
        "top_largest": largest_sorted,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_md(out_md, report)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

