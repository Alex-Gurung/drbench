from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from .run_context import get_log_path


def _get_attr(obj: Any, key: str) -> Optional[Any]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _extract_usage(usage: Any) -> Optional[Dict[str, int]]:
    if usage is None:
        return None
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


def log_generation(
    response: Any,
    request_kind: str,
    requested_model: Optional[str] = None,
    resolved_model: Optional[str] = None,
    stage: Optional[str] = None,
    source: Optional[str] = None,
    log_dir: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
    fail_on_missing_usage: bool = True,
) -> Dict[str, int]:
    response_id = _get_attr(response, "id") or f"local-{uuid.uuid4().hex}"
    usage = _extract_usage(_get_attr(response, "usage"))

    if usage is None:
        if fail_on_missing_usage:
            raise ValueError(
                "Missing token usage in LLM response; failing as configured."
            )
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    record: Dict[str, Any] = {
        "id": response_id,
        "request_kind": request_kind,
        "requested_model": requested_model,
        "resolved_model": resolved_model,
        "response_model": _get_attr(response, "model"),
        "source": source,
        "stage": stage,
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "usage": usage,
    }

    if extra:
        record.update(extra)

    log_path = get_log_path(log_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")

    return usage
