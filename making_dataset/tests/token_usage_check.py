#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import List

from openai import OpenAI
from transformers import AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_PROMPT = "Summarize the purpose of FSMA 204 in one sentence."


def _build_prompt_tokens(tokenizer, messages: List[dict]) -> int:
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return len(tokenizer.encode(prompt_text))
    except Exception:
        # Fallback: naive encoding of concatenated content
        combined = "\n".join(m.get("content", "") for m in messages)
        return len(tokenizer.encode(combined))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check vLLM token usage vs tokenizer counts. Fails on mismatch."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument(
        "--api-url",
        default=os.getenv("VLLM_API_URL"),
        help="vLLM API URL (or set VLLM_API_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("VLLM_API_KEY", "not-needed"),
        help="vLLM API key (optional)",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max completion tokens",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Allowed relative mismatch (default 0.02 = 2%%)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.api_url:
        print("VLLM_API_URL is required", file=sys.stderr)
        return 2

    client = OpenAI(base_url=f"{args.api_url.rstrip('/')}/v1", api_key=args.api_key)
    messages = [{"role": "user", "content": args.prompt}]

    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=0,
    )

    usage = response.usage
    if usage is None:
        print("Missing usage in vLLM response", file=sys.stderr)
        return 2

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_tokens = _build_prompt_tokens(tokenizer, messages)
    completion_text = response.choices[0].message.content or ""
    completion_tokens = len(tokenizer.encode(completion_text))
    total_tokens = prompt_tokens + completion_tokens

    reported_total = int(getattr(usage, "total_tokens", 0) or 0)
    if reported_total <= 0:
        print("Invalid total_tokens in usage", file=sys.stderr)
        return 2

    mismatch = abs(total_tokens - reported_total) / max(reported_total, 1)

    print("Token usage check")
    print(f"  Reported: prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}")
    print(f"  Tokenizer: prompt={prompt_tokens} completion={completion_tokens} total={total_tokens}")
    print(f"  Relative mismatch: {mismatch:.4f}")

    if mismatch > args.tolerance:
        print(
            f"Mismatch {mismatch:.4f} exceeds tolerance {args.tolerance:.4f}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
