"""
Local document search logging for DrBench.

Logs local document search queries and results to JSONL.

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
    return Path(run_dir) / "local_searches.jsonl"


def _safe_list(value: Any) -> Optional[List]:
    """Convert value to list or return None."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def log_local_document_search(
    query: str,
    params: Dict[str, Any],
    result: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a local document search query and result.

    Args:
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
        "tool": "local_document_search",
        "query": query,
        "query_raw": params.get("raw_query"),
        "top_k": params.get("top_k"),
        "file_type_filter": _safe_list(params.get("file_type_filter")),
        "folder_filter": _safe_list(params.get("folder_filter")),
        "success": result.get("success"),
        "data_retrieved": result.get("data_retrieved"),
        "results_count": result.get("results_count"),
        "files_searched": result.get("files_searched"),
        "file_types_found": _safe_list(result.get("file_types_found")),
        "folders_searched": _safe_list(result.get("folders_searched")),
        "synthesis_length": len(str(result.get("synthesis", ""))) if result.get("synthesis") else 0,
        "error": result.get("error"),
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Add result file paths
    results = result.get("results") or {}
    local_docs = results.get("local_documents") or []
    record["result_files"] = [
        doc.get("file_path") for doc in local_docs if doc.get("file_path")
    ][:10]

    if extra:
        record.update(extra)

    log_path = _get_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
