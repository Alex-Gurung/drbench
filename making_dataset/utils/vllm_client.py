from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .llm_logging import log_generation
from .run_context import get_log_dir

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_TIMEOUT = 180.0  # seconds


class VLLMClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        log_dir: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        api_url = api_url or os.getenv("VLLM_API_URL")
        if not api_url:
            raise ValueError("VLLM_API_URL is required for vLLM client")
        api_key = api_key or os.getenv("VLLM_API_KEY") or "not-needed"
        self.model = model
        self.client = OpenAI(base_url=f"{api_url.rstrip('/')}/v1", api_key=api_key, timeout=timeout)
        self.log_dir = get_log_dir() if log_dir is None else Path(log_dir)

    def chat(
        self,
        messages: List[Dict[str, str]],
        stage: str,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        log_generation(
            response,
            request_kind="chat.completions",
            requested_model=self.model,
            resolved_model=self.model,
            stage=stage,
            source="making_dataset.utils.vllm_client",
            log_dir=self.log_dir,
            extra={"provider": "vllm", **(extra or {})},
            fail_on_missing_usage=True,
        )
        return response


def chat_complete(
    messages: List[Dict[str, str]],
    stage: str,
    model: str = DEFAULT_MODEL,
    **kwargs: Any,
) -> str:
    client = VLLMClient(model=model)
    response = client.chat(messages=messages, stage=stage, **kwargs)
    return response.choices[0].message.content
