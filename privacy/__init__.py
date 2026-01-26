"""
Privacy evaluation package for DrBench.

Evaluates privacy leakage in deep research agents by analyzing
whether web search queries leak private enterprise information.
"""

from .eval import evaluate_task, evaluate_batch, load_searches
from .extract_secrets import extract_secrets
from .prompts import (
    build_adversary_prompt,
    build_batched_adversary_prompt,
    build_scorer_prompt,
    build_batched_scorer_prompt,
)

__all__ = [
    "evaluate_task",
    "evaluate_batch",
    "load_searches",
    "extract_secrets",
    "build_adversary_prompt",
    "build_batched_adversary_prompt",
    "build_scorer_prompt",
    "build_batched_scorer_prompt",
]
