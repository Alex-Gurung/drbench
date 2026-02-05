#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from making_dataset.utils.progress import progress  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a local-only multi-hop dataset by walking local neighbors and ending on a secret Q/A."
    )
    parser.add_argument(
        "--chunks",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "chunks_local.jsonl"),
        help="Path to chunks_local.jsonl",
    )
    parser.add_argument(
        "--neighbors",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "local_neighbors.jsonl"),
        help="Path to local_neighbors.jsonl",
    )
    parser.add_argument(
        "--secrets",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "secret_inventory.jsonl"),
        help="Path to secret_inventory.jsonl",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "local_only.jsonl"),
        help="Output dataset JSONL path",
    )
    parser.add_argument(
        "--workspace-id",
        default="drbench_merged_local_v1",
        help="Workspace identifier for the dataset",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=50,
        help="Number of tasks to generate (default: 50)",
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=4,
        help="Hop count (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed (default: 0)",
    )
    parser.add_argument(
        "--neighbor-sample-k",
        type=int,
        default=5,
        help="At each hop, sample from top-K neighbors for diversity (default: 5)",
    )
    parser.add_argument(
        "--max-unnecessary",
        type=int,
        default=20,
        help="Max number of unnecessary secrets to store per task (default: 20)",
    )
    parser.add_argument(
        "--answer-kind",
        choices=["any", "entity"],
        default="any",
        help=(
            "Filter which secrets can be used as the final answer. "
            "`entity` aims for short entity-like strings (default: any)."
        ),
    )
    parser.add_argument(
        "--max-answer-chars",
        type=int,
        default=120,
        help="Reject secrets whose answer exceeds this many chars (default: 120).",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_chunks_map(path: Path) -> dict[str, dict[str, Any]]:
    chunk_map: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("chunk_id")
            if not cid:
                raise ValueError(f"Missing chunk_id in chunks file: {path}")
            chunk_map[cid] = obj
    return chunk_map


def _load_neighbors_map(path: Path) -> dict[str, list[dict[str, Any]]]:
    neighbors: dict[str, list[dict[str, Any]]] = {}
    for rec in _load_jsonl(path):
        cid = rec.get("chunk_id")
        if not cid:
            raise ValueError(f"Missing chunk_id in neighbors file: {path}")
        neighbors[cid] = rec.get("neighbors") or []
    return neighbors


def _load_secrets_map(path: Path) -> dict[str, list[dict[str, Any]]]:
    secrets: dict[str, list[dict[str, Any]]] = {}
    for rec in _load_jsonl(path):
        cid = rec.get("chunk_id")
        if not cid:
            raise ValueError(f"Missing chunk_id in secrets file: {path}")
        items = rec.get("secrets") or []
        if items:
            secrets[cid] = items
    return secrets


def _answer_offsets(text: str, answer: str) -> tuple[str, int, int]:
    """
    Find the answer as a substring in the chunk text so we can store deterministic evidence offsets.
    We do NOT require the model to output a quote; we anchor on the answer itself.
    """
    needle = (answer or "").strip()
    if not needle:
        raise ValueError("Empty answer")

    start = text.find(needle)
    if start >= 0:
        return needle, start, start + len(needle)

    # Allow simple casing/whitespace differences (common for things like "Over 70%" vs "over 70%").
    pattern = re.escape(needle).replace(r"\ ", r"\s+")
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        raise ValueError("Answer not found in chunk text")
    return text[m.start() : m.end()], m.start(), m.end()


_PURE_NUMBER_RE = re.compile(r"^\s*\$?\s*[-+]?\d[\d,.\s]*\s*$")


def _is_entity_like_answer(answer: str) -> bool:
    a = (answer or "").strip()
    if not a:
        return False
    if "@" in a:
        return False
    if "%" in a:
        return False
    if _PURE_NUMBER_RE.match(a):
        return False
    # Require at least one letter (avoid timestamps-only, etc.).
    if not re.search(r"[A-Za-z]", a):
        return False
    return True


def _build_reverse_walk_path(
    *,
    target_chunk_id: str,
    neighbors: dict[str, list[dict[str, Any]]],
    hops: int,
    rng: random.Random,
    neighbor_sample_k: int,
) -> list[str] | None:
    path = [target_chunk_id]
    while len(path) < hops:
        current = path[-1]
        cand = neighbors.get(current) or []
        if not cand:
            return None
        top = cand[: max(1, neighbor_sample_k)]
        rng.shuffle(top)
        picked = None
        for item in top:
            nxt = item.get("chunk_id")
            if not nxt or nxt in path:
                continue
            picked = nxt
            break
        if not picked:
            return None
        path.append(picked)
    return list(reversed(path))


def main() -> int:
    args = _parse_args()
    chunks_path = Path(args.chunks)
    neighbors_path = Path(args.neighbors)
    secrets_path = Path(args.secrets)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.hops < 2:
        raise ValueError("--hops must be >= 2")
    if args.num_tasks <= 0:
        raise ValueError("--num-tasks must be > 0")

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks not found: {chunks_path}")
    if not neighbors_path.exists():
        raise FileNotFoundError(f"Neighbors not found: {neighbors_path}")
    if not secrets_path.exists():
        raise FileNotFoundError(f"Secrets not found: {secrets_path}")

    chunk_map = _load_chunks_map(chunks_path)
    neighbors = _load_neighbors_map(neighbors_path)
    secrets_by_chunk = _load_secrets_map(secrets_path)

    if args.answer_kind == "entity":
        target_chunk_ids = [
            cid
            for cid, items in secrets_by_chunk.items()
            if cid in chunk_map
            and any(
                (s.get("answer") and len(str(s.get("answer") or "")) <= args.max_answer_chars and _is_entity_like_answer(str(s.get("answer") or "")))
                for s in items
            )
        ]
    else:
        target_chunk_ids = [cid for cid in secrets_by_chunk.keys() if cid in chunk_map]
    if not target_chunk_ids:
        if args.answer_kind == "entity":
            raise ValueError(
                "No chunks with entity-like secrets found. Try re-running privacy_tagger.py or "
                "use --answer-kind any."
            )
        raise ValueError("No chunks with secrets found. Did you run privacy_tagger.py?")

    rng = random.Random(args.seed)
    tasks: list[dict[str, Any]] = []

    trials = 0
    max_trials = max(100, args.num_tasks * 50)
    while len(tasks) < args.num_tasks and trials < max_trials:
        trials += 1
        target_cid = rng.choice(target_chunk_ids)
        secret = rng.choice(secrets_by_chunk[target_cid])

        question = (secret.get("question") or "").strip()
        answer = (secret.get("answer") or "").strip()
        if not question or not answer:
            continue
        if len(answer) > args.max_answer_chars:
            continue
        if args.answer_kind == "entity" and not _is_entity_like_answer(answer):
            continue

        path = _build_reverse_walk_path(
            target_chunk_id=target_cid,
            neighbors=neighbors,
            hops=args.hops,
            rng=rng,
            neighbor_sample_k=args.neighbor_sample_k,
        )
        if not path:
            continue

        target_chunk = chunk_map[target_cid]
        target_text = target_chunk.get("text") or ""
        try:
            answer_in_text, a_start, a_end = _answer_offsets(target_text, answer)
        except Exception:
            # Skip items whose answer can't be anchored as a substring (e.g., list answers where
            # the parts are present but not contiguous). We'll revisit if needed.
            continue

        hops = []
        for hop_id, cid in enumerate(path, 1):
            ch = chunk_map.get(cid)
            if not ch:
                raise ValueError(f"Missing chunk in chunk_map referenced by neighbors: {cid}")
            hops.append(
                {
                    "hop_id": hop_id,
                    "chunk_id": cid,
                    "doc_id": ch.get("doc_id"),
                    "source_type": ch.get("source_type"),
                }
            )

        # Privacy: required vs unnecessary (based on evidence set).
        required = [
            {
                "chunk_id": target_cid,
                "question": question,
                "answer": answer,
                "secret_type": secret.get("secret_type"),
            }
        ]

        unnecessary: list[dict[str, Any]] = []
        seen = set()
        for cid in path[:-1]:
            for s in secrets_by_chunk.get(cid, []):
                key = (s.get("question"), s.get("answer"), s.get("secret_type"))
                if key in seen:
                    continue
                seen.add(key)
                unnecessary.append(
                    {
                        "chunk_id": cid,
                        "question": s.get("question"),
                        "answer": s.get("answer"),
                        "secret_type": s.get("secret_type"),
                    }
                )
                if len(unnecessary) >= args.max_unnecessary:
                    break
            if len(unnecessary) >= args.max_unnecessary:
                break

        tasks.append(
            {
                "workspace_id": args.workspace_id,
                "mode": "local_only",
                "question": question,
                "answer": answer,
                "answer_type": "extractive_or_short",
                "tree": {
                    "hops": hops,
                    "target_hop": len(hops),
                },
                "gold": {
                    "target_chunk_id": target_cid,
                    "answer_in_text": answer_in_text,
                    "answer_char_start": a_start,
                    "answer_char_end": a_end,
                },
                "privacy": {
                    "required_secrets": required,
                    "unnecessary_secrets": unnecessary,
                },
            }
        )

    if len(tasks) < args.num_tasks:
        raise ValueError(
            f"Only generated {len(tasks)}/{args.num_tasks} tasks after {trials} trials. "
            "Try reducing --hops or increasing neighbor density (--k when building neighbors)."
        )

    with out_path.open("w", encoding="utf-8") as out:
        for task in tasks:
            out.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path} ({len(tasks)} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
