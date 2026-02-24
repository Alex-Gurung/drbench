#!/usr/bin/env python3
"""Send a text file to vLLM and pretty-print the response.

Usage:
    python -m making_dataset_2.eval.test_prompt prompt.txt \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
        --base-url http://127.0.0.1:8000/v1
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.panel import Panel

from making_dataset_2.llm import LLMClient

console = Console(width=150)

p = argparse.ArgumentParser()
p.add_argument("prompt_file")
p.add_argument("--model", required=True)
p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
p.add_argument("--temperature", type=float, default=0.7)
p.add_argument("--max-tokens", type=int, default=16384)
args = p.parse_args()

prompt = Path(args.prompt_file).read_text()
console.print(Panel(prompt, title=f"[bold]Prompt[/bold] [dim]({len(prompt)} chars)[/dim]", border_style="blue"))

llm = LLMClient(model=args.model, base_url=args.base_url)
console.print("[dim]Calling LLM...[/dim]")
response = llm.chat([{"role": "user", "content": prompt}], temperature=args.temperature, max_tokens=args.max_tokens)

console.print(Panel(response, title=f"[bold]Response[/bold] [dim]({len(response)} chars)[/dim]", border_style="yellow"))
