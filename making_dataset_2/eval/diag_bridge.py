#!/usr/bin/env python3
"""Diagnostic: run one seed through Steps 2-4 and dump all raw LLM outputs.

Uses rich for pretty output. Shows documents, chain state, and full LLM responses.

Usage:
    source ~/initmamba.sh
    python -m making_dataset_2.eval.diag_bridge \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --base-url http://127.0.0.1:8000/v1 \
        --seed-source inventory \
        --seed 42 --n-candidates 3 --two-doc
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from making_dataset_2.data_loading import build_doc_lookup, filter_seed_secrets, load_chunks_local, load_eval_seeds, load_secrets
from making_dataset_2.llm import LLMClient
from making_dataset_2.parsing import parse_blend, parse_bridge, parse_validation
from making_dataset_2.pipeline.step2_query import generate_query
from making_dataset_2.pipeline.step3_retrieve import retrieve_candidates
from making_dataset_2.pipeline.step5_validate import precheck
from making_dataset_2.prompts import (
    build_bridge_blend_prompt,
    build_bridge_composition_prompt,
    build_bridge_two_documents_composition_prompt,
    build_bridge_validation_prompt,
)
from making_dataset_2.retrieval.dense import QWEN3_QUERY_PREFIX, LocalQwen3Backend
from making_dataset_2.retrieval.hybrid import HybridSearcher
from making_dataset_2.types import Bridge, ChainState

# Same defaults as chain_builder.py
DEFAULT_CHUNKS_LOCAL = ROOT_DIR / "making_dataset" / "outputs" / "chunks_local.jsonl"
DEFAULT_CHUNKS_WEB = ROOT_DIR / "making_dataset_2" / "outputs" / "chunks_web_drbench_urls.jsonl"
DEFAULT_DENSE_WEB = ROOT_DIR / "making_dataset_2" / "outputs" / "indexes" / "drbench_urls_dense.npz"
DEFAULT_BROWSECOMP_BM25 = "/home/toolkit/BrowseComp-Plus/indexes/bm25/"
DEFAULT_BROWSECOMP_DENSE = "/home/toolkit/BrowseComp-Plus/indexes/qwen3-embedding-4b/corpus.shard*.pkl"
DEFAULT_SECRETS = ROOT_DIR / "making_dataset_2" / "outputs" / "secret_inventory.jsonl"
DEFAULT_EMBEDDER_MODEL = "Qwen/Qwen3-Embedding-4B"

# DOC_TEXT_LIMIT = 4000
DOC_TEXT_LIMIT = 10000
# DOC_TEXT_LIMIT = 20000

console = Console(width=150)


def _iter_task_chunks(chunks_path: Path, task_id: str) -> list[dict]:
    """Load all chunks tagged with the given task_id."""
    import json
    results = []
    for line in chunks_path.open():
        rec = json.loads(line)
        tids = (rec.get("meta") or {}).get("task_ids", [])
        if task_id in tids:
            results.append(rec)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-candidates", type=int, default=3, help="How many candidates to bridge (default 3)")
    p.add_argument("--task", default=None)
    p.add_argument("--company", default=None)
    p.add_argument("--retrieval-mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    p.add_argument("--drbench-urls-only", action="store_true", help="Skip BrowseComp, use only DRBench URLs corpus")
    p.add_argument("--task-urls-only", action="store_true", help="Filter retrieval to chunks tagged with the seed's task_id")
    p.add_argument("--qa-type", choices=["insight", "distractor"], default=None, help="Filter seeds by qa_type")
    p.add_argument("--embedder-model", default=DEFAULT_EMBEDDER_MODEL)
    p.add_argument("--embedder-device", default="cpu")
    p.add_argument("--seed-source", choices=["eval", "inventory"], default="eval",
                    help="Seed source: eval (from eval.json) or inventory (LLM-generated)")
    p.add_argument("--two-doc", action="store_true", help="Use two-document bridge prompt (shows both current and candidate docs)")
    p.add_argument("--validate", action="store_true", help="Run 5a precheck + 5b LLM validation (off by default)")
    args = p.parse_args()

    chunks = load_chunks_local(DEFAULT_CHUNKS_LOCAL)
    doc_lookup = build_doc_lookup(chunks)
    if args.seed_source == "eval":
        eligible = load_eval_seeds(doc_lookup, task_id=args.task, company=args.company)
        if args.qa_type:
            eligible = [s for s in eligible if s.secret_type == args.qa_type]
        console.print(f"[dim]Loaded {len(eligible)} eval seeds, {len(doc_lookup)} docs[/dim]\n")
    else:
        secrets = load_secrets(DEFAULT_SECRETS)
        eligible = filter_seed_secrets(secrets, doc_lookup, task_id=args.task, company=args.company)
        if args.qa_type:
            eligible = [s for s in eligible if s.secret_type == args.qa_type]
        console.print(f"[dim]Loaded {len(eligible)} inventory seeds, {len(doc_lookup)} docs[/dim]\n")

    llm = LLMClient(model=args.model, base_url=args.base_url)

    # Shared embedder (same as chain_builder.py)
    qwen3_embedder = None
    embedder = None
    if not args.task_urls_only and args.retrieval_mode in ("dense", "hybrid"):
        from making_dataset.index.web_dense import Qwen3EosEmbedder
        query_prefix = QWEN3_QUERY_PREFIX if "qwen3" in args.embedder_model.lower() else ""
        console.print(f"[dim]Loading {args.embedder_model} on {args.embedder_device}...[/dim]")
        qwen3_embedder = Qwen3EosEmbedder(
            model_name_or_path=args.embedder_model,
            query_prefix=query_prefix,
            device=args.embedder_device,
        )
        embedder = LocalQwen3Backend(qwen3_embedder)
        console.print(f"[dim]  Embedder ready on {qwen3_embedder.device}[/dim]\n")

    # Searchers
    searchers = []
    cw = DEFAULT_CHUNKS_WEB
    dw = DEFAULT_DENSE_WEB
    if not args.task_urls_only and cw.exists():
        dp = dw if dw.exists() else None
        s = HybridSearcher(chunks_path=cw, dense_index_path=dp)
        searchers.append(s)
        console.print(f"[dim]DRBench-URLs searcher: {s.size} chunks[/dim]")

    bc = Path(DEFAULT_BROWSECOMP_BM25)
    if not args.task_urls_only and bc.exists() and not args.drbench_urls_only:
        from making_dataset_2.retrieval.browsecomp import BrowseCompSearcher
        dense_glob = DEFAULT_BROWSECOMP_DENSE if args.retrieval_mode != "bm25" else None
        bc_searcher = BrowseCompSearcher(
            bm25_index=str(bc),
            dense_shard_glob=dense_glob,
            model=args.embedder_model,
            embedder=qwen3_embedder,
        )
        searchers.append(bc_searcher)
        console.print(f"[dim]BrowseComp searcher: {bc_searcher.size} docs[/dim]")

    console.print()

    rng = random.Random(args.seed)
    secret = eligible[rng.randrange(len(eligible))]
    doc = doc_lookup[secret.doc_id]

    from making_dataset_2.types import HopRecord
    state = ChainState(
        pattern="LW",
        hop_history=[HopRecord(
            hop_number=1, hop_type="L",
            question=secret.question, answer=secret.answer,
            doc_id=secret.doc_id, doc_text=doc.text,
        )],
        global_question=secret.question,
        global_answer=secret.answer,
        used_doc_ids={secret.doc_id},
        task_id=doc.meta.get("task_id"),
        company=doc.meta.get("company_name"),
    )

    # ── Step 1: Seed ──
    seed_table = Table(show_header=False, box=None, padding=(0, 2))
    seed_table.add_column(style="bold cyan", width=10)
    seed_table.add_column()
    seed_table.add_row("Task", state.task_id or "")
    seed_table.add_row("Company", state.company or "")
    seed_table.add_row("QA Type", secret.secret_type)
    seed_table.add_row("Question", secret.question)
    seed_table.add_row("Answer", f"[bold yellow]{secret.answer}[/bold yellow]")
    seed_table.add_row("Doc", secret.doc_id)
    if doc.meta.get("relative_path"):
        seed_table.add_row("File", f"drbench/data/tasks/{doc.meta['relative_path']}")
    console.print(Panel(seed_table, title="[bold]Step 1: Seed[/bold]", border_style="blue"))

    # Show the current (seed) document
    current_doc_text = (state.hop_history[-1].doc_text if state.hop_history else "")[:DOC_TEXT_LIMIT]
    console.print(Panel(
        current_doc_text,
        title=f"[bold]Current Document[/bold] [dim]({len(current_doc_text)} chars)[/dim]",
        border_style="blue",
    ))

    if args.task_urls_only:
        console.print(Panel("[dim]--task-urls-only: loading task URL chunks directly[/dim]",
                            title="Steps 2-3: Skipped", border_style="dim"))
        from making_dataset_2.retrieval.types import RetrievalHit
        task_id = state.task_id
        hits = []
        for rec in _iter_task_chunks(DEFAULT_CHUNKS_WEB, task_id):
            hits.append(RetrievalHit(
                chunk_id=rec["chunk_id"], doc_id=rec["doc_id"],
                score=1.0, text=rec["text"], meta=rec.get("meta"),
            ))
        console.print(f"  [dim]{len(hits)} chunks for task {task_id}[/dim]\n")
    else:
        # Step 2
        console.rule("[bold]Step 2: Query Generation[/bold]")
        query, raw_query = generate_query(state, llm, target_corpus="web")
        console.print(f"  Query: [bold green]{query}[/bold green]\n")

        # Step 3
        console.rule("[bold]Step 3: Retrieval[/bold]")
        hits = retrieve_candidates(
            state, query,
            searchers=searchers,
            embedder=embedder,
            mode=args.retrieval_mode,
            k=10,
        )
        hit_table = Table(box=None, padding=(0, 1))
        hit_table.add_column("#", style="dim", width=3)
        hit_table.add_column("Score", width=7)
        hit_table.add_column("Doc ID", width=20)
        hit_table.add_column("Snippet")
        for i, h in enumerate(hits[:args.n_candidates + 2]):
            snippet = (h.text or "")[:120].replace("\n", " ")
            style = "bold" if i < args.n_candidates else "dim"
            hit_table.add_row(str(i), f"{h.score:.3f}", h.doc_id, snippet, style=style)
        console.print(hit_table)
        console.print()

    # ── Chain State ──
    state_table = Table(show_header=False, box=None, padding=(0, 2))
    state_table.add_column(style="bold cyan", width=14)
    state_table.add_column()
    state_table.add_row("Pattern", f"[bold]{state.pattern}[/bold]  (hop {state.current_hop}/{len(state.pattern)}, next: [bold]{state.next_hop_type or 'done'}[/bold])")
    state_table.add_row("Global Q", state.global_question)
    state_table.add_row("Global A", f"[bold yellow]{state.global_answer}[/bold yellow]")
    state_table.add_row("Used Docs", ", ".join(sorted(state.used_doc_ids)))
    console.print(Panel(state_table, title="[bold]Chain State[/bold]", border_style="cyan"))

    # Hop history
    if state.hop_history:
        hop_table = Table(title="Hop History", border_style="cyan", padding=(0, 1))
        hop_table.add_column("#", style="dim", width=3)
        hop_table.add_column("Type", width=4)
        hop_table.add_column("Question")
        hop_table.add_column("Answer", style="bold yellow")
        hop_table.add_column("Doc ID", style="dim")
        for h in state.hop_history:
            hop_table.add_row(str(h.hop_number), h.hop_type, h.question, h.answer, h.doc_id)
        console.print(hop_table)
        console.print()

    # ── Step 4: Bridge composition ──
    n = min(args.n_candidates, len(hits))
    for ci in range(n):
        hit = hits[ci]
        console.rule(f"[bold]Step 4: Bridge Composition  --  Candidate {ci}[/bold]")

        # Show candidate document
        doc_text = (hit.text or "")[:DOC_TEXT_LIMIT]
        hit_path = (hit.meta or {}).get("relative_path", "")
        hit_label = f"drbench/data/tasks/{hit_path}" if hit_path else hit.doc_id
        console.print(Panel(
            doc_text,
            title=f"[bold]Candidate Document[/bold] [dim]{hit_label} ({len(doc_text)} chars)[/dim]",
            border_style="green",
        ))

        # Build prompt
        if args.two_doc:
            prompt = build_bridge_two_documents_composition_prompt(state, doc_text)
        else:
            prompt = build_bridge_composition_prompt(state, doc_text)

        console.print(f"[dim]P1 Prompt: {len(prompt)} chars  |  Calling LLM...[/dim]")
        raw_bridge = llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=8192,
        )

        # Show full LLM output
        console.print(Panel(
            raw_bridge,
            title=f"[bold]P1 LLM Output[/bold] [dim]({len(raw_bridge)} chars)[/dim]",
            border_style="yellow",
        ))

        parsed = parse_bridge(raw_bridge)
        if parsed is None:
            console.print("[bold red]Parse result: NO_BRIDGE or parse failure[/bold red]\n")
            continue

        # Show parsed bridge
        result_table = Table(show_header=False, box=None, padding=(0, 2))
        result_table.add_column(style="bold magenta", width=16)
        result_table.add_column()
        result_table.add_row("QUESTION", parsed.question)
        result_table.add_row("ANSWER", f"[bold yellow]{parsed.answer}[/bold yellow]")
        if parsed.justification:
            result_table.add_row("JUSTIFICATION", parsed.justification)
        if parsed.bridge_phrase:
            result_table.add_row("BRIDGE_PHRASE", parsed.bridge_phrase)
        console.print(Panel(result_table, title="[bold]P1 Parsed Bridge[/bold]", border_style="magenta"))

        # P2: Blend (only for two-doc mode)
        blended_question = parsed.question
        if args.two_doc:
            p2_prompt = build_bridge_blend_prompt(state, parsed.question, parsed.answer)
            console.print(f"[dim]P2 Blend Prompt: {len(p2_prompt)} chars  |  Calling LLM...[/dim]")
            raw_blend = llm.chat(
                [{"role": "user", "content": "/no_think\n" + p2_prompt}],
                temperature=0.3,
                max_tokens=2048,
            )
            console.print(Panel(
                raw_blend,
                title=f"[bold]P2 Blend Output[/bold] [dim]({len(raw_blend)} chars)[/dim]",
                border_style="yellow",
            ))
            blended = parse_blend(raw_blend)
            if blended:
                blended_question = blended
                console.print(f"  [bold green]Blended:[/bold green] {blended_question}")
            else:
                console.print(f"  [bold red]P2 parse failed, using direct question[/bold red]")

        bridge = Bridge(
            bridge_phrase=parsed.bridge_phrase if not args.two_doc else "",
            question=blended_question,
            answer=parsed.answer,
            doc_id=hit.doc_id,
            doc_text=hit.text or "",
            candidate_index=ci,
            justification=parsed.justification,
        )

        # Optional validation
        if args.validate:
            reason = precheck(bridge, state)
            if reason:
                console.print(f"  [bold red]5a PRECHECK: FAIL[/bold red] -- {reason}\n")
                continue
            console.print(f"  [bold green]5a PRECHECK: PASS[/bold green]")

            val_prompt = build_bridge_validation_prompt(state, bridge)
            raw_val = llm.chat(
                [{"role": "user", "content": val_prompt}],
                temperature=0.1,
                max_tokens=8192,
            )
            console.print(Panel(raw_val, title="[bold]5b Validation LLM Output[/bold]", border_style="dim"))
            parsed_val = parse_validation(raw_val)
            if parsed_val is None:
                console.print("[bold red]  Validation parse failed[/bold red]")
            else:
                color = "green" if parsed_val.verdict == "ACCEPT" else "red"
                console.print(f"  Verdict: [bold {color}]{parsed_val.verdict}[/bold {color}]  Hops: {parsed_val.hops_required}")
                console.print(f"  Reason: {parsed_val.reason}")

        console.print()


if __name__ == "__main__":
    main()
