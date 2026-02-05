from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Optional

DEFAULT_LOG_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "logs"
_RUN_ID: Optional[str] = None


def _generate_run_id() -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def get_run_id() -> str:
    global _RUN_ID
    if _RUN_ID is None:
        _RUN_ID = os.getenv("DATASET_RUN_ID") or _generate_run_id()
    return _RUN_ID


def get_log_dir(run_id: Optional[str] = None, root: Optional[Path] = None) -> Path:
    root = Path(root) if root else DEFAULT_LOG_ROOT
    run_id = run_id or get_run_id()
    path = root / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_log_path(log_dir: Optional[Path] = None) -> Path:
    log_dir = log_dir or get_log_dir()
    return log_dir / "llm_generations.jsonl"
