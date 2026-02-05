#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, IO, Iterable, Optional


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Split pooled local artifacts (chunks_local.jsonl + secret_inventory.jsonl) into scoped "
            "sub-corpora, so we can run per-DRBench-task or per-company dataset generation."
        )
    )
    parser.add_argument(
        "--chunks",
        default=str(root / "outputs" / "chunks_local.jsonl"),
        help="Path to pooled chunks_local.jsonl",
    )
    parser.add_argument(
        "--secrets",
        default=str(root / "outputs" / "secret_inventory.jsonl"),
        help="Path to pooled secret_inventory.jsonl",
    )
    parser.add_argument(
        "--out-root",
        default=str(root / "outputs" / "scopes"),
        help="Output root directory (default: outputs/scopes)",
    )
    parser.add_argument(
        "--by",
        nargs="+",
        choices=["task", "company"],
        default=["task", "company"],
        help="Which scopes to write (default: task company).",
    )
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _open_writer(cache: Dict[Path, IO[str]], path: Path) -> IO[str]:
    handle = cache.get(path)
    if handle is not None:
        return handle
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    cache[path] = handle
    return handle


def main() -> int:
    args = _parse_args()
    chunks_path = Path(args.chunks)
    secrets_path = Path(args.secrets)
    out_root = Path(args.out_root)

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks not found: {chunks_path}")
    if not secrets_path.exists():
        raise FileNotFoundError(f"Secrets not found: {secrets_path}")

    by_task = "task" in args.by
    by_company = "company" in args.by
    if not (by_task or by_company):
        raise ValueError("--by must include at least one of: task, company")

    # Map chunk_id -> {task_id, company, company_name}
    chunk_scope: Dict[str, Dict[str, str]] = {}
    task_meta: Dict[str, Dict[str, Any]] = {}
    company_meta: Dict[str, Dict[str, Any]] = {}

    chunk_writers: Dict[Path, IO[str]] = {}
    secret_writers: Dict[Path, IO[str]] = {}

    try:
        # 1) Split chunks
        for rec in _iter_jsonl(chunks_path):
            if rec.get("source_type") != "local":
                continue
            chunk_id = rec.get("chunk_id")
            if not chunk_id:
                raise ValueError(f"Missing chunk_id in {chunks_path}")
            meta = rec.get("meta") or {}
            task_id = meta.get("task_id")
            company = meta.get("company")
            company_name = meta.get("company_name")
            if not task_id:
                raise ValueError(f"Missing meta.task_id for chunk {chunk_id}")
            if not company or not company_name:
                raise ValueError(f"Missing meta.company/company_name for chunk {chunk_id}")

            chunk_scope[str(chunk_id)] = {
                "task_id": str(task_id),
                "company": str(company),
                "company_name": str(company_name),
            }

            if by_task:
                tdir = out_root / "task" / str(task_id)
                out_chunks = tdir / "chunks_local.jsonl"
                _open_writer(chunk_writers, out_chunks).write(
                    json.dumps(rec, ensure_ascii=False) + "\n"
                )
                tm = task_meta.setdefault(
                    str(task_id),
                    {
                        "scope_type": "task",
                        "scope_value": str(task_id),
                        "company": str(company),
                        "company_name": str(company_name),
                        "num_chunks": 0,
                        "num_secret_chunks": 0,
                        "num_secret_items": 0,
                    },
                )
                tm["num_chunks"] += 1

            if by_company:
                cdir = out_root / "company" / str(company)
                out_chunks = cdir / "chunks_local.jsonl"
                _open_writer(chunk_writers, out_chunks).write(
                    json.dumps(rec, ensure_ascii=False) + "\n"
                )
                cm = company_meta.setdefault(
                    str(company),
                    {
                        "scope_type": "company",
                        "scope_value": str(company),
                        "company": str(company),
                        "company_name": str(company_name),
                        "num_chunks": 0,
                        "num_secret_chunks": 0,
                        "num_secret_items": 0,
                    },
                )
                cm["num_chunks"] += 1

        if not chunk_scope:
            raise ValueError("No local chunks found; did you pass the correct chunks file?")

        # 2) Split secrets by chunk_id membership
        for rec in _iter_jsonl(secrets_path):
            chunk_id = rec.get("chunk_id")
            if not chunk_id:
                raise ValueError(f"Missing chunk_id in {secrets_path}")
            scope = chunk_scope.get(str(chunk_id))
            if scope is None:
                # Secrets file should be derived from chunks_local.jsonl; if not, fail loud.
                raise ValueError(f"Secret record chunk_id not found in chunks map: {chunk_id}")

            task_id = scope["task_id"]
            company = scope["company"]

            secrets = rec.get("secrets") or []
            n_items = len(secrets)

            if by_task:
                tdir = out_root / "task" / task_id
                out_secrets = tdir / "secret_inventory.jsonl"
                _open_writer(secret_writers, out_secrets).write(
                    json.dumps(rec, ensure_ascii=False) + "\n"
                )
                if n_items:
                    task_meta[task_id]["num_secret_chunks"] += 1
                    task_meta[task_id]["num_secret_items"] += n_items

            if by_company:
                cdir = out_root / "company" / company
                out_secrets = cdir / "secret_inventory.jsonl"
                _open_writer(secret_writers, out_secrets).write(
                    json.dumps(rec, ensure_ascii=False) + "\n"
                )
                if n_items:
                    company_meta[company]["num_secret_chunks"] += 1
                    company_meta[company]["num_secret_items"] += n_items
    finally:
        for h in list(chunk_writers.values()) + list(secret_writers.values()):
            try:
                h.close()
            except Exception:
                pass

    # 3) Write scope summaries
    if by_task:
        for task_id, meta in sorted(task_meta.items()):
            (out_root / "task" / task_id / "scope.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
    if by_company:
        for company, meta in sorted(company_meta.items()):
            (out_root / "company" / company / "scope.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )

    print(f"Wrote scopes under {out_root}")
    if by_task:
        print(f"- task scopes: {len(task_meta)}")
    if by_company:
        print(f"- company scopes: {len(company_meta)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

