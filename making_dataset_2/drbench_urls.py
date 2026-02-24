from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict

WEB_POOL = "drbench_urls"
DOCID_PREFIX = f"{WEB_POOL}_"
DOC_ID_PREFIX = f"web/{WEB_POOL}/"


def docid_for_url(url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"{DOCID_PREFIX}{digest}"


def doc_id_for_docid(docid: str) -> str:
    return f"{DOC_ID_PREFIX}{docid}"


def doc_id_for_url(url: str) -> str:
    return doc_id_for_docid(docid_for_url(url))


def load_seed_urls(urls_json_path: Path) -> list[dict[str, Any]]:
    """Load drbench seed URLs from contexts/urls.json.

    Returns a list of dicts, in file order, deduplicated by URL.
    """
    data = json.loads(urls_json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {urls_json_path}")

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in data:
        if not isinstance(row, dict):
            continue
        url = (row.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(row)
    return out


def build_url_to_task_ids(tasks_root: Path) -> dict[str, list[str]]:
    """Map seed URL -> list of task IDs that reference it."""
    mapping: DefaultDict[str, list[str]] = defaultdict(list)
    for entry in sorted(tasks_root.iterdir()):
        if not entry.is_dir():
            continue
        context_path = entry / "context.json"
        if not context_path.exists():
            continue
        try:
            ctx = json.loads(context_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        url = ((ctx.get("url") or {}).get("url") or "").strip()
        if not url:
            continue
        mapping[url].append(entry.name)

    out: dict[str, list[str]] = {}
    for url, task_ids in mapping.items():
        out[url] = sorted(set(task_ids))
    return out

