#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a local_only dataset JSONL deterministically.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset JSONL (e.g., outputs/local_only.jsonl)",
    )
    parser.add_argument(
        "--chunks",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "chunks_local.jsonl"),
        help="Path to chunks_local.jsonl (for quote verification)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional max number of tasks to validate (for smoke tests)",
    )
    return parser.parse_args()


def _load_chunks_map(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("chunk_id")
            if not cid:
                raise ValueError(f"Missing chunk_id in chunks file: {path}")
            out[cid] = obj
    return out


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> int:
    args = _parse_args()
    dataset_path = Path(args.dataset)
    chunks_path = Path(args.chunks)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks not found: {chunks_path}")

    chunk_map = _load_chunks_map(chunks_path)

    errors: list[str] = []
    n = 0
    hop_counts: Counter[int] = Counter()
    required_types: Counter[str] = Counter()
    unnecessary_counts: Counter[int] = Counter()

    for task in _iter_jsonl(dataset_path):
        n += 1
        if args.max_items is not None and n > args.max_items:
            break

        try:
            mode = task.get("mode")
            if mode != "local_only":
                raise ValueError(f"mode must be local_only, got {mode}")

            question = (task.get("question") or "").strip()
            answer = (task.get("answer") or "").strip()
            if not question or not answer:
                raise ValueError("Missing question/answer")

            hops = task.get("tree", {}).get("hops")
            if not isinstance(hops, list) or len(hops) < 2:
                raise ValueError("tree.hops must be a list with len>=2")
            hop_counts[len(hops)] += 1

            target_chunk_id = task.get("gold", {}).get("target_chunk_id")
            if not target_chunk_id:
                raise ValueError("Missing gold.target_chunk_id")
            if hops[-1].get("chunk_id") != target_chunk_id:
                raise ValueError("Last hop chunk_id must equal gold.target_chunk_id")

            chunk = chunk_map.get(target_chunk_id)
            if not chunk:
                raise ValueError(f"Target chunk not found in chunks map: {target_chunk_id}")

            text = chunk.get("text") or ""
            gold = task.get("gold", {}) or {}

            # New format: anchor evidence by the answer substring.
            if "answer_in_text" in gold:
                ans_in_text = gold.get("answer_in_text")
                start = gold.get("answer_char_start")
                end = gold.get("answer_char_end")
                if not isinstance(ans_in_text, str) or not isinstance(start, int) or not isinstance(end, int):
                    raise ValueError("gold answer fields missing/invalid")
                if text[start:end] != ans_in_text:
                    raise ValueError("Answer offsets do not match the target chunk text")
            else:
                # Back-compat: older outputs stored a quote span.
                quote = gold.get("quote")
                start = gold.get("quote_char_start")
                end = gold.get("quote_char_end")
                if not isinstance(quote, str) or not isinstance(start, int) or not isinstance(end, int):
                    raise ValueError("gold quote fields missing/invalid")
                if text[start:end] != quote:
                    raise ValueError("Quote offsets do not match the target chunk text")

            required = task.get("privacy", {}).get("required_secrets") or []
            if not required:
                raise ValueError("privacy.required_secrets missing")
            st = (required[0].get("secret_type") or "unknown").lower()
            required_types[st] += 1

            unnecessary = task.get("privacy", {}).get("unnecessary_secrets") or []
            if not isinstance(unnecessary, list):
                raise ValueError("privacy.unnecessary_secrets must be a list")
            unnecessary_counts[len(unnecessary)] += 1
        except Exception as e:
            errors.append(f"task[{n}]: {e}")

    print(f"Validated tasks: {n}")
    print(f"Hop count distribution: {dict(sorted(hop_counts.items()))}")
    print(f"Required secret_type distribution: {dict(sorted(required_types.items()))}")
    print(f"Unnecessary secret counts (bucketed): {dict(sorted(unnecessary_counts.items()))}")

    if errors:
        print("\nErrors:")
        for err in errors[:50]:
            print(err)
        raise SystemExit(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
