#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a web-only dataset from BrowseComp-Plus decrypted tasks."
    )
    parser.add_argument(
        "--input",
        default="/home/toolkit/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl",
        help="BrowseComp-Plus decrypted JSONL (tasks).",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "web_only.jsonl"),
        help="Output dataset JSONL path.",
    )
    parser.add_argument(
        "--workspace-id",
        default="browsecomp_plus_web_v1",
        help="Workspace identifier.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=50,
        help="Number of tasks to emit (default: 50).",
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=4,
        help="Hop count (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for doc selection (default: 0).",
    )
    parser.add_argument(
        "--require-answer-span",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the answer to appear as an exact substring in a hop doc (default: enabled).",
    )
    parser.add_argument(
        "--answer-kind",
        choices=["any", "entity"],
        default="any",
        help="Filter output tasks by answer type (default: any).",
    )
    parser.add_argument(
        "--max-answer-chars",
        type=int,
        default=120,
        help="Reject tasks whose answer exceeds this many chars (default: 120).",
    )
    parser.add_argument(
        "--limit-input",
        type=int,
        default=None,
        help="Optional limit on number of input BrowseComp tasks scanned.",
    )
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _dedupe_docids(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for d in docs:
        docid = d.get("docid")
        if docid is None:
            continue
        docid = str(docid)
        if docid in seen:
            continue
        seen.add(docid)
        out.append(d)
    return out


def _find_answer_span(text: str, answer: str) -> Optional[Tuple[int, int]]:
    if not answer:
        return None
    start = text.find(answer)
    if start < 0:
        return None
    return start, start + len(answer)


def _is_entity_like(answer: str) -> bool:
    a = (answer or "").strip()
    if not a:
        return False
    if "@" in a:
        return False
    if "%" in a:
        return False
    # Entity-like: contains at least one letter.
    return bool(re.search(r"[A-Za-z]", a))


def main() -> int:
    args = _parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.num_tasks <= 0:
        raise ValueError("--num-tasks must be > 0")
    if args.hops < 1:
        raise ValueError("--hops must be >= 1")
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    rng = random.Random(args.seed)

    tasks: List[Dict[str, Any]] = []
    scanned = 0

    # Streaming scan until we have enough acceptable tasks.
    for rec in tqdm(_iter_jsonl(in_path), desc="Scanning BrowseComp tasks", unit="task"):
        scanned += 1
        if args.limit_input is not None and scanned > args.limit_input:
            break

        qid = str(rec.get("query_id") or "")
        question = (rec.get("query") or "").strip()
        answer = (rec.get("answer") or "").strip()
        if not question or not answer:
            continue
        if len(answer) > args.max_answer_chars:
            continue
        if args.answer_kind == "entity" and not _is_entity_like(answer):
            continue

        gold_docs = rec.get("gold_docs") or []
        evidence_docs = rec.get("evidence_docs") or []
        docs = _dedupe_docids(list(gold_docs) + list(evidence_docs))
        if len(docs) < args.hops:
            continue

        # Choose hop docs deterministically-ish: first fill from gold, then evidence.
        # Shuffle within gold/evidence to add variety without breaking provenance.
        gold = _dedupe_docids(list(gold_docs))
        ev = _dedupe_docids(list(evidence_docs))
        rng.shuffle(gold)
        rng.shuffle(ev)
        hop_docs = (gold + ev)[: args.hops]
        if len(hop_docs) < args.hops:
            continue

        # Evidence pointer: find exact answer span in any hop doc (prefer gold docs).
        answer_evidence = None
        if args.require_answer_span:
            for d in gold + ev:
                text = d.get("text") or ""
                span = _find_answer_span(text, answer)
                if span is None:
                    continue
                answer_evidence = {
                    "chunk_id": f"web/{d['docid']}#0001",
                    "doc_id": f"web/{d['docid']}",
                    "char_start": span[0],
                    "char_end": span[1],
                }
                break
            if answer_evidence is None:
                continue

        hops = []
        for hop_id, d in enumerate(hop_docs, 1):
            docid = str(d.get("docid"))
            if not docid:
                raise ValueError("Missing docid in hop doc")
            hops.append(
                {
                    "hop_id": hop_id,
                    "chunk_id": f"web/{docid}#0001",
                    "doc_id": f"web/{docid}",
                    "source_type": "web",
                }
            )

        tasks.append(
            {
                "workspace_id": args.workspace_id,
                "mode": "web_only",
                "question": question,
                "answer": answer,
                "answer_type": "exact_span" if args.require_answer_span else "string",
                "tree": {
                    "hops": hops,
                    "source": {"browsecomp_query_id": qid},
                },
                "gold": {
                    "answer_evidence": answer_evidence,
                    "gold_docids": [str(d.get("docid")) for d in gold],
                    "evidence_docids": [str(d.get("docid")) for d in ev],
                },
                "privacy": {"required_secrets": [], "unnecessary_secrets": []},
            }
        )

        if len(tasks) >= args.num_tasks:
            break

    if len(tasks) < args.num_tasks:
        raise ValueError(
            f"Only generated {len(tasks)}/{args.num_tasks} tasks after scanning {scanned} input tasks. "
            "Try increasing --limit-input or disabling --require-answer-span."
        )

    with out_path.open("w", encoding="utf-8") as out:
        for t in tasks:
            out.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path} ({len(tasks)} tasks; scanned {scanned})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
