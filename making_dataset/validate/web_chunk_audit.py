#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import heapq
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit web chunk sizes to flag overly large docs."
    )
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "chunks_web.jsonl"),
        help="Web chunks JSONL",
    )
    parser.add_argument(
        "--out-json",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "web_chunk_audit.json"),
        help="Output JSON report",
    )
    parser.add_argument(
        "--out-md",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "web_chunk_audit.md"),
        help="Output Markdown report",
    )
    parser.add_argument(
        "--tokenizer-model",
        default=None,
        help="Optional HF tokenizer model for exact token counts",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top K largest docs to report",
    )
    parser.add_argument(
        "--token-thresholds",
        type=int,
        nargs="*",
        default=[8192, 16384, 32768],
        help="Token thresholds to count above",
    )
    parser.add_argument(
        "--char-per-token",
        type=float,
        default=4.0,
        help="Chars per token for estimation when tokenizer not provided",
    )
    return parser.parse_args()


def _iter_chunks(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_md(path: Path, report: Dict[str, Any]) -> None:
    lines = []
    lines.append("# Web Chunk Size Audit")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"Total docs: {report['total_docs']}")
    lines.append("")
    lines.append("### Token Thresholds")
    lines.append("")
    lines.append("| Threshold | Count | Percent |")
    lines.append("| --- | ---: | ---: |")
    for threshold, stats in report["thresholds"].items():
        lines.append(
            f"| {threshold} | {stats['count']} | {stats['percent']:.2f}% |"
        )
    lines.append("")
    lines.append("## Top Largest Docs")
    lines.append("")
    lines.append("| Rank | Doc ID | Est Tokens | Chars | Words |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for idx, item in enumerate(report["top_largest"], start=1):
        lines.append(
            f"| {idx} | {item['doc_id']} | {item['est_tokens']} | {item['chars']} | {item['words']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    tokenizer = None
    if args.tokenizer_model:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, trust_remote_code=True)

    thresholds = sorted(set(args.token_thresholds))
    threshold_counts = {str(t): 0 for t in thresholds}

    total_docs = 0
    total_chars = 0
    total_words = 0
    total_tokens = 0

    top_heap: list[tuple[int, Dict[str, Any]]] = []

    for record in _iter_chunks(input_path):
        if record.get("source_type") != "web":
            continue
        text = record.get("text") or ""
        if not text.strip():
            continue

        chars = len(text)
        words = len(text.split())
        if tokenizer:
            tokens = len(tokenizer.encode(text))
        else:
            tokens = int(math.ceil(chars / args.char_per_token))

        total_docs += 1
        total_chars += chars
        total_words += words
        total_tokens += tokens

        for threshold in thresholds:
            if tokens >= threshold:
                threshold_counts[str(threshold)] += 1

        item = {
            "doc_id": record.get("doc_id"),
            "est_tokens": tokens,
            "chars": chars,
            "words": words,
        }
        heapq.heappush(top_heap, (tokens, item))
        if len(top_heap) > args.top_k:
            heapq.heappop(top_heap)

    top_largest = [item for _, item in sorted(top_heap, key=lambda x: x[0], reverse=True)]

    report = {
        "total_docs": total_docs,
        "avg_chars": round(total_chars / total_docs, 2) if total_docs else 0,
        "avg_words": round(total_words / total_docs, 2) if total_docs else 0,
        "avg_tokens": round(total_tokens / total_docs, 2) if total_docs else 0,
        "thresholds": {},
        "top_largest": top_largest,
        "tokenizer_model": args.tokenizer_model,
        "char_per_token": args.char_per_token,
    }

    for threshold in thresholds:
        count = threshold_counts[str(threshold)]
        percent = (count / total_docs * 100) if total_docs else 0
        report["thresholds"][str(threshold)] = {
            "count": count,
            "percent": percent,
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
