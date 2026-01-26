"""
Internet search logging for DrBench.

Logs web search queries and results to JSONL for privacy analysis.

Configuration:
    Logging is ON by default. Disable via CLI flag --no-log-searches.
    Log path is determined by RunConfig.run_dir (required).
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from drbench.config import get_run_config

logger = logging.getLogger(__name__)


def _get_log_path() -> Path:
    """Get log file path from RunConfig.

    Raises:
        ValueError: If run_dir is not set in RunConfig.
    """
    run_dir = get_run_config().run_dir
    if not run_dir:
        raise ValueError(
            "--run-dir is required for logging. "
            "Either provide --run-dir or disable logging with --no-log-searches"
        )
    return Path(run_dir) / "internet_searches.jsonl"


def _safe_list(value: Any) -> Optional[List]:
    """Convert value to list or return None."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def log_internet_search(
    tool: str,
    query: str,
    params: Dict[str, Any],
    result: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Log an internet search query and result.

    Args:
        tool: Tool name (e.g., "web_search", "web_fetch")
        query: Search query string
        params: Tool parameters
        result: Tool result
        extra: Additional fields to include

    Raises on logging failure to avoid silent data loss.
    """
    cfg = get_run_config()

    # Check if logging is enabled via RunConfig (CLI flag)
    if not cfg.log_searches:
        return

    record: Dict[str, Any] = {
        "tool": tool,
        "query": query,
        "params": {
            k: v for k, v in params.items()
            if k not in ("query",) and v is not None
        },
        "success": result.get("success"),
        "data_retrieved": result.get("data_retrieved"),
        "results_count": result.get("results_count"),
        "error": result.get("error"),
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Add result URLs if present
    if result.get("results"):
        urls = []
        for r in result["results"][:10]:  # Limit to first 10
            if isinstance(r, dict) and r.get("url"):
                urls.append(r["url"])
        if urls:
            record["result_urls"] = urls

    if extra:
        record.update(extra)

    log_path = _get_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
