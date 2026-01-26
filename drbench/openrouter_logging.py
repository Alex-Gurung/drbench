"""
OpenRouter/LLM generation logging for DrBench.

Logs LLM generation metadata (tokens, models, timing) to JSONL.

Configuration:
    Logging is ON by default. Disable via CLI flag --no-log-generations.
    Log path is determined by RunConfig.run_dir (required).
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from drbench.config import get_run_config

logger = logging.getLogger(__name__)


def _get_attr(obj: Any, key: str) -> Optional[Any]:
    """Get attribute from object or dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _extract_usage(usage: Any) -> Optional[Dict[str, int]]:
    """Extract token usage from response."""
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


def _get_log_path() -> Path:
    """Get log file path from RunConfig.

    Raises:
        ValueError: If run_dir is not set in RunConfig.
    """
    run_dir = get_run_config().run_dir
    if not run_dir:
        raise ValueError(
            "--run-dir is required for logging. "
            "Either provide --run-dir or disable logging with --no-log-generations"
        )
    return Path(run_dir) / "llm_generations.jsonl"


def log_openrouter_generation(
    response: Any,
    request_kind: str,
    requested_model: Optional[str] = None,
    resolved_model: Optional[str] = None,
    source: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Log an LLM generation response.

    Args:
        response: LLM response object
        request_kind: Type of request (e.g., "chat", "embeddings")
        requested_model: Model that was requested
        resolved_model: Model that was actually used
        source: Source code location
        extra: Additional fields to include

    Raises on logging failure to avoid silent data loss.
    """
    cfg = get_run_config()

    # Check if logging is enabled via RunConfig (CLI flag)
    if not cfg.log_generations:
        return

    response_id = _get_attr(response, "id")
    if not response_id:
        return

    record: Dict[str, Any] = {
        "id": response_id,
        "request_kind": request_kind,
        "requested_model": requested_model,
        "resolved_model": resolved_model,
        "response_model": _get_attr(response, "model"),
        "source": source,
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    usage = _extract_usage(_get_attr(response, "usage"))
    if usage:
        record["usage"] = usage

    if extra:
        record.update(extra)

    log_path = _get_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
