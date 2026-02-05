"""
BrowseComp search logging for DrBench.

Logs BrowseComp corpus search queries and results to JSONL for privacy analysis.

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
    return Path(run_dir) / "browsecomp_searches.jsonl"


def _truncate(text: Optional[str], max_len: int = 240) -> Optional[str]:
    """Truncate text to max length with ellipsis."""
    if not text:
        return text
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def log_browsecomp_search(
    tool: str,
    query: str,
    params: Dict[str, Any],
    result: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a BrowseComp search query and result.

    Args:
        tool: Tool name (e.g., "browsecomp_search")
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
        "query_raw": query,  # For privacy tool compatibility
        "params": {
            k: v for k, v in params.items()
            if k not in ("query",) and v is not None
        },
        "success": result.get("success"),
        "data_retrieved": result.get("data_retrieved"),
        "results_count": result.get("total_results") or len(result.get("results", []) or []),
        "error": result.get("error"),
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Add results in both formats for compatibility
    results = result.get("results") or []

    # result_docs format (BrowseComp native format)
    record["result_docs"] = [
        {
            "docid": r.get("docid"),
            "url": r.get("url"),
            "score": r.get("score"),
            "snippet": _truncate(r.get("text")),
        }
        for r in results[:10]
    ]

    # result_links format (internet_search compatible, for privacy tools)
    record["result_links"] = [
        {
            "title": f"BrowseComp Doc {r.get('docid', 'unknown')[:8] if r.get('docid') else 'unknown'}",
            "link": r.get("url", ""),
            "snippet": _truncate(r.get("text")),
        }
        for r in results[:10]
    ]

    if extra:
        record.update(extra)

    log_path = _get_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
