#!/usr/bin/env python3
"""Interactive chain builder: manually build multi-hop QA chains.

Enter question/answer pairs, pick a search pool (Local/Web/Both/Auto),
optionally write your own search query, browse retrieved documents,
and repeat. Prints the full chain at the end.

Auto mode ([A]) finds docs containing the answer via substring match.
When a web doc is selected, privacy questions whose answer appears in
the doc are suggested automatically.

Usage:
    python -m making_dataset_2.eval.interactive_chain
"""
from __future__ import annotations

import sys
from pathlib import Path

import spacy
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from making_dataset_2.data_loading import (
    build_doc_lookup, filter_seed_secrets, load_chunks_local, load_secrets,
)
from making_dataset_2.retrieval.hybrid import HybridSearcher, merge_results
from making_dataset_2.retrieval.types import RetrievalHit

CHUNKS_LOCAL = ROOT / "making_dataset" / "outputs" / "chunks_local.jsonl"
CHUNKS_WEB = ROOT / "making_dataset_2" / "outputs" / "chunks_web_drbench_urls.jsonl"
SECRETS = ROOT / "making_dataset_2" / "outputs" / "secret_inventory.jsonl"
DOC_LIMIT = 8000

console = Console(width=140)


def load_searchers():
    console.print("[dim]Loading local chunks...[/dim]", end=" ")
    local = HybridSearcher(chunks_path=CHUNKS_LOCAL)
    console.print(f"[dim]{local.size} chunks[/dim]")

    console.print("[dim]Loading web chunks...[/dim]", end=" ")
    web = HybridSearcher(chunks_path=CHUNKS_WEB)
    console.print(f"[dim]{web.size} chunks[/dim]\n")
    return local, web


def search(query: str, pool: str, local: HybridSearcher, web: HybridSearcher, k: int = 8):
    if pool == "L":
        return local.search(query, k=k, mode="bm25")
    elif pool == "W":
        return web.search(query, k=k, mode="bm25")
    else:
        local_hits = local.search(query, k=k, mode="bm25")
        web_hits = web.search(query, k=k, mode="bm25")
        return merge_results(local_hits, web_hits, k=k)


def auto_find(answer: str, local: HybridSearcher, web: HybridSearcher) -> list[RetrievalHit]:
    """Find all chunks whose text contains `answer` (case-insensitive substring)."""
    needle = answer.strip().lower()
    if len(needle) < 2:
        return []
    hits: list[RetrievalHit] = []
    seen_docs: set[str] = set()
    for searcher in (local, web):
        for i, text in enumerate(searcher._texts):
            if needle in text.lower():
                did = searcher._doc_ids[i]
                if did in seen_docs:
                    continue
                seen_docs.add(did)
                hits.append(RetrievalHit(
                    chunk_id=searcher._chunk_ids[i],
                    doc_id=did,
                    score=1.0,
                    text=text,
                    meta=searcher._metas[i],
                ))
    return hits


def precompute_question_entities(eligible_secrets, nlp) -> list[set[str]]:
    """Extract spaCy entities from each secret's question (run once at startup)."""
    result = []
    for s in eligible_secrets:
        doc = nlp(s.question)
        ents = {ent.text.strip().lower() for ent in doc.ents}
        result.append(ents)
    return result


def suggest_privacy_questions(
    doc_text: str, eligible_secrets, q_entities: list[set[str]], nlp,
) -> list[dict]:
    """Find secrets whose question mentions an entity from the web doc."""
    doc = nlp(doc_text)
    doc_entities = {ent.text.strip().lower() for ent in doc.ents}
    suggestions = []
    seen = set()
    for i, s in enumerate(eligible_secrets):
        shared = doc_entities & q_entities[i]
        if not shared:
            continue
        key = (s.question, s.answer)
        if key in seen:
            continue
        seen.add(key)
        suggestions.append({
            "question": s.question,
            "answer": s.answer,
            "doc_id": s.doc_id,
            "secret_type": s.secret_type,
            "shared_entities": sorted(shared),
        })
    return suggestions


def show_suggestions(suggestions: list[dict], limit: int = 10):
    if not suggestions:
        return
    show_n = min(limit, len(suggestions))
    t = Table(
        title=f"Suggested privacy questions ({show_n}/{len(suggestions)})",
        border_style="magenta",
        padding=(0, 1),
    )
    t.add_column("#", style="bold", width=3)
    t.add_column("Question")
    t.add_column("Answer", style="bold yellow")
    t.add_column("Shared", style="cyan")
    t.add_column("Type", style="dim", width=16)
    t.add_column("Source Doc", style="dim", max_width=30, no_wrap=True)

    for i in range(show_n):
        s = suggestions[i]
        shared = ", ".join(s.get("shared_entities", []))
        t.add_row(str(i), s["question"], s["answer"], shared, s["secret_type"], s["doc_id"][-30:])
    console.print(t)

    if len(suggestions) > show_n:
        if Prompt.ask(f"[dim]Show all {len(suggestions)}?[/dim]", choices=["y", "n"], default="n") == "y":
            show_suggestions(suggestions, limit=len(suggestions))


def show_answer_context(doc_text: str, answer: str, context_chars: int = 250):
    """Show snippets from the doc where the answer appears, for writing a bridge Q."""
    needle = answer.strip().lower()
    text_lower = doc_text.lower()
    pos = 0
    snippets = []
    while len(snippets) < 3:
        idx = text_lower.find(needle, pos)
        if idx == -1:
            break
        # Expand to sentence boundaries (find '. ' before and after)
        start = max(0, idx - context_chars)
        end = min(len(doc_text), idx + len(needle) + context_chars)
        # Snap to sentence start
        dot = doc_text.rfind(". ", start, idx)
        if dot != -1 and dot > start:
            start = dot + 2
        # Snap to sentence end
        dot = doc_text.find(". ", idx + len(needle), end)
        if dot != -1:
            end = dot + 1
        snippet = doc_text[start:end].replace("\n", " ").strip()
        # Bold the answer in the snippet
        ans_start = idx - start
        ans_end = ans_start + len(needle)
        snippet = snippet[:ans_start] + "[bold yellow]" + snippet[ans_start:ans_end] + "[/bold yellow]" + snippet[ans_end:]
        snippets.append(snippet)
        pos = idx + len(needle)

    if snippets:
        console.print(f"\n[dim][bold yellow]{answer}[/bold yellow] in context:[/dim]")
        for s in snippets:
            console.print(f"  ...{s}...")


def show_results(hits, pool_label: str, limit: int = 10):
    if not hits:
        console.print("[bold red]No results found.[/bold red]")
        return

    show_n = min(limit, len(hits))
    t = Table(title=f"Search Results ({pool_label}) — showing {show_n}/{len(hits)}", border_style="cyan", padding=(0, 1))
    t.add_column("#", style="bold", width=3)
    t.add_column("Score", width=8)
    t.add_column("Pool", width=5)
    t.add_column("Doc ID", max_width=45, no_wrap=True)
    t.add_column("Snippet")

    for i in range(show_n):
        h = hits[i]
        pool_tag = "[green]W[/green]" if h.doc_id.startswith("web/") else "[yellow]L[/yellow]"
        snippet = (h.text or "")[:120].replace("\n", " ")
        t.add_row(str(i), f"{h.score:.2f}", pool_tag, h.doc_id[-45:], snippet)
    console.print(t)

    if len(hits) > show_n:
        if Prompt.ask(f"[dim]Show all {len(hits)}?[/dim]", choices=["y", "n"], default="n") == "y":
            show_results(hits, pool_label, limit=len(hits))


def show_doc(hit):
    text = (hit.text or "")[:DOC_LIMIT]
    url = (hit.meta or {}).get("url", "")
    title = hit.doc_id
    if url:
        title += f"  [dim]{url}[/dim]"
    console.print(Panel(text, title=title, border_style="green", width=140))


def show_chain(hops):
    console.print()
    pattern = "".join(h["pool"] for h in hops)
    t = Table(title=f"Chain  [bold cyan]{pattern}[/bold cyan]", border_style="blue", padding=(0, 1))
    t.add_column("#", style="bold", width=3)
    t.add_column("Pool", width=5)
    t.add_column("Question")
    t.add_column("Answer", style="bold yellow")
    t.add_column("Doc ID", style="dim", max_width=40, no_wrap=True)

    for i, h in enumerate(hops):
        pool_tag = "[green]W[/green]" if h["pool"] == "W" else "[yellow]L[/yellow]"
        doc = h.get("doc_id", "-")
        t.add_row(str(i + 1), pool_tag, h["question"], h["answer"], doc[-40:] if doc else "-")
    console.print(t)
    console.print()


def main():
    local_s, web_s = load_searchers()

    # Load secrets for auto-suggest
    console.print("[dim]Loading secrets...[/dim]", end=" ")
    chunks = load_chunks_local(CHUNKS_LOCAL)
    doc_lookup = build_doc_lookup(chunks)
    secrets = load_secrets(SECRETS)
    eligible = filter_seed_secrets(secrets, doc_lookup)
    console.print(f"[dim]{len(eligible)} eligible secrets[/dim]")

    console.print("[dim]Loading spaCy...[/dim]", end=" ")
    nlp = spacy.load("en_core_web_lg")
    console.print(f"[dim]{nlp.meta['name']}[/dim]")

    console.print("[dim]Pre-computing question entities...[/dim]", end=" ")
    q_entities = precompute_question_entities(eligible, nlp)
    console.print(f"[dim]done[/dim]\n")

    console.print(Panel(
        "[bold]Interactive Chain Builder[/bold]\n\n"
        "Enter question/answer pairs to build a multi-hop chain.\n"
        "At each hop, pick a pool to search:\n"
        "  [yellow]L[/yellow]ocal / [green]W[/green]eb / [cyan]B[/cyan]oth (BM25)  |  [magenta]A[/magenta]uto-find (substring match)\n\n"
        "When a [green]web[/green] doc is selected, privacy questions are suggested automatically.\n\n"
        "Type [bold red]done[/bold red] at any prompt to finish and print the chain.",
        border_style="blue",
    ))

    hops: list[dict] = []
    hop_num = 0
    prefilled_q: str | None = None
    prefilled_a: str | None = None

    while True:
        hop_num += 1
        console.rule(f"[bold]Hop {hop_num}[/bold]")

        # Question (possibly pre-filled from suggestion)
        if prefilled_q:
            console.print(f"[dim]Pre-filled from suggestion:[/dim]")
            console.print(f"  Q: [bold]{prefilled_q}[/bold]")
            console.print(f"  A: [bold yellow]{prefilled_a}[/bold yellow]")
            confirm = Prompt.ask("[bold]Use this?[/bold]", choices=["y", "n"], default="y")
            if confirm == "y":
                q, a = prefilled_q, prefilled_a
            else:
                q = Prompt.ask("[bold]Question[/bold] [dim](or 'done')[/dim]")
                if q.strip().lower() == "done":
                    break
                a = Prompt.ask("[bold]Answer[/bold]")
                if a.strip().lower() == "done":
                    break
            prefilled_q = prefilled_a = None
        else:
            q = Prompt.ask("[bold]Question[/bold] [dim](or 'done')[/dim]")
            if q.strip().lower() == "done":
                break
            a = Prompt.ask("[bold]Answer[/bold]")
            if a.strip().lower() == "done":
                break

        # Pool
        pool = Prompt.ask(
            "[bold]Search pool[/bold]",
            choices=["L", "W", "B", "A", "l", "w", "b", "a"],
            default="B",
        ).upper()

        pool_label = {"L": "Local", "W": "Web", "B": "Both", "A": "Auto-find"}[pool]

        # Search loop (can re-search)
        selected_hit = None
        query = a.strip()
        while selected_hit is None:
            if pool == "A":
                hits = auto_find(query, local_s, web_s)
                show_results(hits, f"Auto-find: '{query}'")
            else:
                query = Prompt.ask(f"[bold]Search query[/bold] [dim](enter = '{query}')[/dim]", default=query)
                hits = search(query, pool, local_s, web_s)
                show_results(hits, pool_label)

            if not hits:
                query = Prompt.ask("[bold]Try another query[/bold]")
                pool = "B"  # fall back to BM25 on retry
                continue

            pick = Prompt.ask(
                "[bold]Pick doc[/bold] [dim](number, or 's' to re-search)[/dim]",
                default="0",
            )
            if pick.strip().lower() == "s":
                query = Prompt.ask("[bold]New query[/bold]")
                pool = Prompt.ask(
                    "[bold]Search pool[/bold]",
                    choices=["L", "W", "B", "A", "l", "w", "b", "a"],
                    default="B",
                ).upper()
                continue

            try:
                idx = int(pick)
                if 0 <= idx < len(hits):
                    selected_hit = hits[idx]
                else:
                    console.print("[red]Invalid index[/red]")
            except ValueError:
                console.print("[red]Enter a number or 's'[/red]")

        show_doc(selected_hit)

        doc_pool = "W" if selected_hit.doc_id.startswith("web/") else "L"
        hops.append({
            "question": q.strip(),
            "answer": a.strip(),
            "pool": doc_pool,
            "doc_id": selected_hit.doc_id,
        })

        # Auto-suggest privacy questions when a web doc is selected
        if doc_pool == "W":
            doc_text = (selected_hit.text or "")[:DOC_LIMIT]
            suggestions = suggest_privacy_questions(doc_text, eligible, q_entities, nlp)
            if suggestions:
                show_suggestions(suggestions)
                pick_sug = Prompt.ask(
                    "[bold]Pick suggestion[/bold] [dim](number, or 'skip')[/dim]",
                    default="skip",
                )
                if pick_sug.strip().lower() != "skip":
                    try:
                        si = int(pick_sug)
                        if 0 <= si < len(suggestions):
                            sug = suggestions[si]
                            shared = sug["shared_entities"]

                            # Pick which entity to bridge through
                            if len(shared) == 1:
                                bridge_ent = shared[0]
                            else:
                                console.print("[dim]Shared entities:[/dim]")
                                for ei, e in enumerate(shared):
                                    console.print(f"  [cyan][{ei}][/cyan] {e}")
                                ep = Prompt.ask("[bold]Pick entity to bridge[/bold]", default="0")
                                try:
                                    bridge_ent = shared[int(ep)]
                                except (ValueError, IndexError):
                                    bridge_ent = shared[0]

                            prev_answer = hops[-1]["answer"]

                            # Show context for both A1 and E in the web doc
                            console.print(f"\n[dim]── Previous answer (A1) in this doc ──[/dim]")
                            show_answer_context(doc_text, prev_answer)
                            console.print(f"\n[dim]── Bridge entity (E → connects to Q3) ──[/dim]")
                            show_answer_context(doc_text, bridge_ent)

                            console.print(f"\n[magenta]Connects to:[/magenta] \"{sug['question']}\" → {sug['answer']} [dim]({sug['doc_id'][-35:]})[/dim]")

                            # Bridge Q1: mentions A1, answer = B (intermediate)
                            console.print(f"\n[dim]Bridge Q1: mention [bold yellow]{prev_answer}[/bold yellow], answer = new intermediate entity[/dim]")
                            bq1 = Prompt.ask("[bold]Bridge Q1[/bold]")
                            if not bq1.strip():
                                continue
                            b_answer = Prompt.ask("[bold]Answer B[/bold]")
                            if not b_answer.strip():
                                continue

                            # Bridge Q2: mentions B, answer = E
                            console.print(f"\n[dim]Bridge Q2: mention [bold yellow]{b_answer}[/bold yellow], answer = [bold yellow]{bridge_ent}[/bold yellow][/dim]")
                            bq2 = Prompt.ask("[bold]Bridge Q2[/bold]")
                            if not bq2.strip():
                                continue

                            # Record both web hops
                            hop_num += 1
                            hops.append({
                                "question": bq1.strip(),
                                "answer": b_answer.strip(),
                                "pool": "W",
                                "doc_id": selected_hit.doc_id,
                            })
                            hop_num += 1
                            hops.append({
                                "question": bq2.strip(),
                                "answer": bridge_ent,
                                "pool": "W",
                                "doc_id": selected_hit.doc_id,
                            })
                            # Pre-fill next hop with the privacy secret Q/A
                            prefilled_q = sug["question"]
                            prefilled_a = sug["answer"]
                    except ValueError:
                        pass

    if hops:
        show_chain(hops)
    else:
        console.print("[dim]No hops recorded.[/dim]")


if __name__ == "__main__":
    main()
