"""Load secret inventory, eval.json seeds, local chunks, and build full-document lookups."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_SECRET_INVENTORY = ROOT_DIR / "making_dataset_2" / "outputs" / "secret_inventory.jsonl"
DEFAULT_CHUNKS_LOCAL = ROOT_DIR / "making_dataset" / "outputs" / "chunks_local.jsonl"


@dataclass
class Secret:
    """A single secret from the inventory."""
    chunk_id: str
    doc_id: str
    question: str
    answer: str
    secret_type: str
    justification: str
    quote: str
    doc_only_check: dict[str, Any]  # {with_doc: ..., without_doc: ...}


@dataclass
class LocalDoc:
    """Full text of a local document (concatenated from all its chunks)."""
    doc_id: str
    text: str
    chunk_ids: list[str]
    meta: dict[str, Any]


def load_secrets(path: Path | None = None) -> list[Secret]:
    """Load all secrets from secret_inventory.jsonl."""
    path = path or DEFAULT_SECRET_INVENTORY
    secrets: list[Secret] = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            chunk_id = row["chunk_id"]
            doc_id = row["doc_id"]
            for s in row.get("secrets", []):
                secrets.append(Secret(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    question=s["question"],
                    answer=s["answer"],
                    secret_type=s.get("secret_type", ""),
                    justification=s.get("justification", ""),
                    quote=s.get("quote", ""),
                    doc_only_check=s.get("doc_only_check", {}),
                ))
    return secrets


def load_chunks_local(path: Path | None = None) -> list[dict[str, Any]]:
    """Load all local chunks as raw dicts."""
    path = path or DEFAULT_CHUNKS_LOCAL
    chunks: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_doc_lookup(chunks: list[dict[str, Any]]) -> dict[str, LocalDoc]:
    """Build doc_id → LocalDoc mapping by concatenating chunks per document.

    Chunks are sorted by their offset within the document to preserve order.
    """
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        did = c.get("doc_id")
        if did:
            by_doc[did].append(c)

    docs: dict[str, LocalDoc] = {}
    for doc_id, doc_chunks in by_doc.items():
        # Sort by start offset if available, else by chunk_id
        doc_chunks.sort(key=lambda c: ((c.get("offsets") or {}).get("start", 0), c.get("chunk_id", "")))
        text = "\n\n".join(c.get("text", "") for c in doc_chunks)
        chunk_ids = [c["chunk_id"] for c in doc_chunks]
        # Use meta from first chunk
        meta = doc_chunks[0].get("meta", {}) if doc_chunks else {}
        docs[doc_id] = LocalDoc(doc_id=doc_id, text=text, chunk_ids=chunk_ids, meta=meta)
    return docs


def filter_seed_secrets(
    secrets: list[Secret],
    doc_lookup: dict[str, LocalDoc],
    *,
    task_id: str | None = None,
    company: str | None = None,
) -> list[Secret]:
    """Filter secrets suitable for chain seeds (Step 1 rules).

    Rules:
    - doc_only_check["with_doc"] must be truthy (answerable with doc)
    - Answer must be 1-5 words
    - Full document text must be >= 200 chars
    - Optional task_id / company filters
    """
    filtered: list[Secret] = []
    for s in secrets:
        # Exclude types that never produce good chain seeds
        if s.secret_type in ("names", "emails"):
            continue
        # Exclude broken answers
        if "[insert" in s.answer.lower() or "NOT_ANSWERABLE" in s.answer:
            continue

        # doc_only_check: with_doc should indicate answerable
        doc_check = s.doc_only_check
        if not doc_check.get("with_doc"):
            continue

        # Answer length: 1-2 words (short entity names)
        words = s.answer.strip().split()
        if not words or len(words) > 2:
            continue

        # Full document must exist and be >= 200 chars
        doc = doc_lookup.get(s.doc_id)
        if doc is None or len(doc.text) < 200:
            continue

        # Optional filters
        if task_id and doc.meta.get("task_id") != task_id:
            continue
        if company:
            doc_company = (doc.meta.get("company_name") or "").lower()
            if company.lower() not in doc_company:
                continue

        filtered.append(s)
    return filtered


# ---------------------------------------------------------------------------
# Eval.json seed loading
# ---------------------------------------------------------------------------

TASKS_DIR = ROOT_DIR / "drbench" / "data" / "tasks"

def _build_stem_to_docid(doc_lookup: dict[str, LocalDoc]) -> dict[tuple[str, str], str]:
    """Build (task_id, filename_stem) → doc_id lookup from doc_lookup."""
    mapping: dict[tuple[str, str], str] = {}
    for doc_id, doc in doc_lookup.items():
        tid = doc.meta.get("task_id", "")
        # Get filename stem from meta or doc_id
        fname = doc.meta.get("file_name", "")
        if not fname:
            fname = Path(doc_id).stem
        mapping[(tid, fname)] = doc_id
    return mapping


def load_eval_seeds(
    doc_lookup: dict[str, LocalDoc],
    *,
    task_id: str | None = None,
    company: str | None = None,
) -> list[Secret]:
    """Load seeds from eval.json files (hand-picked DrBench QA pairs).

    Returns Secret objects for compatibility with the existing pipeline.
    Only enterprise_fact entries with non-empty questions are included.
    """
    stem_to_docid = _build_stem_to_docid(doc_lookup)
    seeds: list[Secret] = []

    for eval_path in sorted(TASKS_DIR.glob("DR*/config/eval.json")):
        tid = eval_path.parent.parent.name  # e.g. DR0001
        if task_id and tid != task_id:
            continue

        # Read company name from context.json
        context_path = eval_path.parent / "context.json"
        company_name = ""
        if context_path.exists():
            ctx = json.loads(context_path.read_text())
            company_name = ctx.get("company_info", {}).get("name", "")

        if company and company.lower() not in company_name.lower():
            continue

        eval_data = json.loads(eval_path.read_text())
        for entry in eval_data.get("dr_report_evaluation_qa", []):
            if entry.get("type") != "enterprise_fact":
                continue
            q = entry.get("question", "").strip()
            a = entry.get("answer", "").strip()
            if not q or not a:
                continue

            # Map supporting_file_paths → doc_id
            doc_id = ""
            for fp in entry.get("supporting_file_paths", []):
                stem = Path(fp).stem  # e.g. "food-safety-compliance"
                candidate = stem_to_docid.get((tid, stem))
                if candidate:
                    doc_id = candidate
                    break

            if not doc_id:
                continue

            seeds.append(Secret(
                chunk_id=doc_id,
                doc_id=doc_id,
                question=q,
                answer=a,
                secret_type=entry.get("qa_type", ""),  # "insight" or "distractor"
                justification=entry.get("justification", ""),
                quote="",
                doc_only_check={"with_doc": True},
            ))

    return seeds
