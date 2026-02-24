"""Thin LLM client wrapper over any OpenAI-compatible API."""

from __future__ import annotations

import os


class LLMClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        from openai import OpenAI

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY") or ""
        if not api_key:
            if base_url is None:
                raise RuntimeError("No API key and no base_url provided.")
            api_key = "DUMMY"

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
        self.model = model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 800,
    ) -> str:
        """Send chat completion, return assistant text."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
