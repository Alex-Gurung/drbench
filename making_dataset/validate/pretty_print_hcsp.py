#!/usr/bin/env python3
"""
Pretty-print HCSP (InfoSeek-style) tasks with full structure visualization.

Shows:
- Research tree structure (question root + evidence vertices)
- Linkage type and solving chain
- Constraints extracted from each evidence
- Validation results (shape, ablations)
- Diversity metadata

Usage:
    python pretty_print_hcsp.py --dataset outputs/scopes/task/DR0001/local_only.hcsp.jsonl
    python pretty_print_hcsp.py --dataset outputs/scopes/task/DR0001/mixed.hcsp.jsonl --format terminal
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ANSI colors for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretty-print InfoSeek-style HCSP tasks with full structure."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to HCSP dataset JSONL",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: alongside dataset with .report.md suffix)",
    )
    parser.add_argument(
        "--local-chunks",
        default=None,
        help="Path to chunks_local.jsonl for text previews",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=10,
        help="Max tasks to show (default: 10)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "terminal"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Show full chunk text (verbose)",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=200,
        help="Chars of context around evidence spans",
    )
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_chunks_map(path: Path) -> Dict[str, Dict[str, Any]]:
    chunk_map = {}
    if not path.exists():
        return chunk_map
    for rec in _iter_jsonl(path):
        chunk_id = rec.get("chunk_id", "")
        if chunk_id:
            chunk_map[chunk_id] = rec
    return chunk_map


def _snippet(text: str, start: int, end: int, ctx: int) -> str:
    """Extract snippet with highlighted span."""
    start = max(0, int(start))
    end = min(len(text), int(end))
    left = max(0, start - ctx)
    right = min(len(text), end + ctx)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    mid = text[left:start] + "**" + text[start:end] + "**" + text[end:right]
    return (prefix + mid + suffix).replace("\n", " ")


def _md_escape(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ")


def _linkage_emoji(linkage_type: str) -> str:
    """Get emoji for linkage type."""
    emojis = {
        "entity_chain": "🔗",
        "computational": "🧮",
        "selection": "🎯",
        "definitional": "📖",
        "creative": "✨",
    }
    return emojis.get(linkage_type, "❓")


def _linkage_description(linkage_type: str) -> str:
    """Get description for linkage type."""
    descriptions = {
        "entity_chain": "Web reveals entity needed for local lookup",
        "computational": "Answer requires combining local + web facts",
        "selection": "Web provides filter criterion for local options",
        "definitional": "Web defines term used in local",
        "creative": "LLM-discovered cross-corpus linkage",
    }
    return descriptions.get(linkage_type, "Unknown linkage type")


def _format_tree_ascii(task: Dict[str, Any]) -> List[str]:
    """Format research tree as ASCII art."""
    lines = []
    tree = task.get("tree", {})
    hcsp = tree.get("hcsp", {})
    hops = tree.get("hops", [])

    # Build vertex info from hops
    vertices = []
    for hop in hops:
        source = hop.get("source_type", "local")
        chunk_id = hop.get("chunk_id", "")
        vertex_id = hop.get("edge", {}).get("vertex_id", f"H{hop.get('hop_id', '?')}")
        vertices.append({
            "id": vertex_id,
            "source": source,
            "chunk_id": chunk_id,
        })

    # Get constraints from HCSP nodes
    nodes = hcsp.get("nodes", {})
    constraints_by_vertex = {}
    for node_id, node in nodes.items():
        if node.get("kind") == "constraint":
            constraint = node.get("constraint", {})
            # Parse vertex from node_id (e.g., "L1_C0" -> "L1")
            parts = node_id.split("_")
            if len(parts) >= 2:
                vertex = parts[0]
                if vertex not in constraints_by_vertex:
                    constraints_by_vertex[vertex] = []
                constraints_by_vertex[vertex].append(constraint.get("text", ""))

    # Root (question)
    question = task.get("question", "")[:60] + "..." if len(task.get("question", "")) > 60 else task.get("question", "")
    lines.append("```")
    lines.append("ROOT [question]")
    lines.append(f"  └─ \"{question}\"")
    lines.append("  │")

    # Evidence vertices
    for i, v in enumerate(vertices):
        is_last = (i == len(vertices) - 1)
        prefix = "  └─" if is_last else "  ├─"
        source_icon = "📁" if v["source"] == "local" else "🌐"

        lines.append(f"{prefix} [{v['id']}] {source_icon} {v['source']}")

        child_prefix = "     " if is_last else "  │  "
        lines.append(f"{child_prefix}chunk: {v['chunk_id'][:50]}...")

        # Show constraints for this vertex
        vertex_constraints = constraints_by_vertex.get(v["id"], [])
        for j, c in enumerate(vertex_constraints[:2]):  # Max 2 constraints shown
            c_text = c[:50] + "..." if len(c) > 50 else c
            lines.append(f"{child_prefix}  • {c_text}")

        if not is_last:
            lines.append("  │")

    lines.append("```")
    return lines


def _format_solving_chain(task: Dict[str, Any]) -> List[str]:
    """Format the expected solving chain."""
    lines = []
    gold = task.get("gold", {})
    linkage_type = gold.get("linkage_type", "")
    tree = task.get("tree", {})
    hops = tree.get("hops", [])

    lines.append("**Solving Chain**")
    lines.append("")

    steps = ["1. Read question"]

    if linkage_type == "entity_chain":
        for hop in hops:
            if hop.get("source_type") == "web":
                steps.append("2. Search web → identify entity from constraints")
            elif hop.get("source_type") == "local":
                steps.append("3. Search local with entity → find answer value")
        steps.append("4. Extract answer")
    elif linkage_type == "computational":
        step_num = 2
        for hop in hops:
            source = "web" if hop.get("source_type") == "web" else "local"
            val_type = "baseline" if source == "web" else "company"
            steps.append(f"{step_num}. Search {source} → get {val_type} value")
            step_num += 1
        steps.append(f"{step_num}. Compute difference/ratio")
        steps.append(f"{step_num + 1}. Return computed answer")
    elif linkage_type == "selection":
        steps.append("2. Search web → identify selection criterion/ranking")
        steps.append("3. Search local → find matching item's value")
        steps.append("4. Extract answer")
    elif linkage_type == "definitional":
        steps.append("2. Search local → find achievement/certification")
        steps.append("3. Search web → find definition/threshold")
        steps.append("4. Extract answer")
    else:
        step_num = 2
        for hop in hops:
            source = hop.get("source_type", "local")
            steps.append(f"{step_num}. Search {source} → gather evidence")
            step_num += 1
        steps.append(f"{step_num}. Extract answer")

    for step in steps:
        lines.append(f"   {step}")
    lines.append("")

    return lines


def _format_constraints_table(task: Dict[str, Any]) -> List[str]:
    """Format constraints as a table."""
    lines = []
    tree = task.get("tree", {})
    hcsp = tree.get("hcsp", {})
    nodes = hcsp.get("nodes", {})

    constraints = []
    for node_id, node in nodes.items():
        if node.get("kind") == "constraint":
            c = node.get("constraint", {})
            constraints.append({
                "id": node_id,
                "text": c.get("text", ""),
                "type": c.get("constraint_type", ""),
                "corpus": c.get("corpus", ""),
                "chunk_id": c.get("evidence", {}).get("chunk_id", ""),
            })

    if not constraints:
        lines.append("*No constraints extracted*")
        return lines

    lines.append("| ID | Corpus | Type | Constraint Text |")
    lines.append("|:---|:---:|:---:|:---|")

    for c in constraints:
        text = c["text"][:80] + "..." if len(c["text"]) > 80 else c["text"]
        corpus_icon = "📁" if c["corpus"] == "local" else "🌐"
        lines.append(f"| {c['id']} | {corpus_icon} {c['corpus']} | {c['type']} | {_md_escape(text)} |")

    lines.append("")
    return lines


def _format_diversity(task: Dict[str, Any]) -> List[str]:
    """Format diversity metadata."""
    lines = []
    diversity = task.get("diversity", {})

    if not diversity:
        return ["*No diversity metadata*", ""]

    hop_pattern = diversity.get("hop_pattern", "")
    local_constraints = diversity.get("local_constraints", 0)
    web_constraints = diversity.get("web_constraints", 0)
    total_hops = diversity.get("total_hops", 0)
    answer_corpus = diversity.get("answer_corpus", "")
    required_secrets = diversity.get("required_local_secrets", 0)

    # Visualize hop pattern
    pattern_visual = " → ".join(
        f"📁L" if c == "L" else f"🌐W" for c in hop_pattern
    )

    lines.append(f"- **Hop Pattern**: `{hop_pattern}` ({pattern_visual})")
    lines.append(f"- **Total Hops**: {total_hops}")
    lines.append(f"- **Constraints**: {local_constraints} local, {web_constraints} web")
    lines.append(f"- **Answer Corpus**: {answer_corpus}")
    lines.append(f"- **Required Local Secrets**: {required_secrets}")
    lines.append("")

    return lines


def _format_quality(task: Dict[str, Any]) -> List[str]:
    """Format quality/validation results."""
    lines = []
    quality = task.get("quality", {})

    if not quality:
        return ["*No quality validation run*", ""]

    det_pass = quality.get("deterministic_pass")
    full_info = quality.get("ablation_full_info")
    full_pass = quality.get("ablation_full_info_pass")
    no_info = quality.get("ablation_no_info")
    no_pass = quality.get("ablation_no_info_pass")

    det_icon = "✅" if det_pass else "❌"
    lines.append(f"- **Deterministic**: {det_icon} {'pass' if det_pass else 'fail'}")

    if full_pass is not None:
        full_icon = "✅" if full_pass else "❌"
        full_preview = (full_info or "")[:50]
        lines.append(f"- **Full Info Ablation**: {full_icon} `{_md_escape(full_preview)}`")

    if no_pass is not None:
        no_icon = "✅" if no_pass else "❌"
        lines.append(f"- **No Info Ablation**: {no_icon} (should be NOT_ANSWERABLE)")

    # Check for corpus-specific ablations (mixed mode)
    local_pass = quality.get("ablation_local_only_pass")
    web_pass = quality.get("ablation_web_only_pass")

    if local_pass is not None:
        local_icon = "✅" if local_pass else "❌"
        lines.append(f"- **Local-Only Ablation**: {local_icon} (should be NOT_ANSWERABLE)")

    if web_pass is not None:
        web_icon = "✅" if web_pass else "❌"
        lines.append(f"- **Web-Only Ablation**: {web_icon} (should be NOT_ANSWERABLE)")

    lines.append("")
    return lines


def _format_privacy(task: Dict[str, Any]) -> List[str]:
    """Format privacy information."""
    lines = []
    privacy = task.get("privacy", {})

    required = privacy.get("required_secrets", [])
    unnecessary = privacy.get("unnecessary_secrets", [])
    prompt_leaks = task.get("prompt_leaks", [])

    lines.append(f"- **Required Secrets**: {len(required)}")
    for s in required[:3]:  # Show first 3
        q = s.get("question", "")[:40]
        a = s.get("answer", "")[:20]
        is_int = "🔗" if s.get("is_intermediate") else "🎯"
        lines.append(f"  - {is_int} `{_md_escape(a)}` ({_md_escape(q)}...)")

    if len(required) > 3:
        lines.append(f"  - ... and {len(required) - 3} more")

    lines.append(f"- **Unnecessary Secrets**: {len(unnecessary)}")

    if prompt_leaks:
        lines.append(f"- **⚠️ Prompt Leaks**: {', '.join(prompt_leaks)}")

    lines.append("")
    return lines


def format_task_markdown(task: Dict[str, Any], idx: int, chunk_map: Dict[str, Dict], ctx_chars: int, show_text: bool) -> List[str]:
    """Format a single task as Markdown."""
    lines = []

    mode = task.get("mode", "unknown")
    question = task.get("question", "")
    answer = task.get("answer", "")
    answer_type = task.get("answer_type", "")
    gold = task.get("gold", {})
    linkage_type = gold.get("linkage_type", "")

    # Header
    mode_icon = {"local_only": "📁", "web_only": "🌐", "mixed": "🔀"}.get(mode, "❓")
    linkage_icon = _linkage_emoji(linkage_type) if linkage_type else ""

    lines.append(f"## Task {idx}: {mode_icon} {mode} {linkage_icon}")
    lines.append("")

    # Linkage type badge
    if linkage_type:
        lines.append(f"> **Linkage Type**: {linkage_icon} `{linkage_type}` — {_linkage_description(linkage_type)}")
        lines.append("")

    # Question & Answer
    lines.append("### Question")
    lines.append("")
    lines.append(f"> {textwrap.fill(question, width=90)}")
    lines.append("")
    lines.append(f"**Answer** (`{answer_type}`): `{_md_escape(answer)}`")
    lines.append("")

    # Research Tree Visualization
    lines.append("### Research Tree")
    lines.append("")
    lines.extend(_format_tree_ascii(task))
    lines.append("")

    # Solving Chain
    if linkage_type:
        lines.extend(_format_solving_chain(task))

    # Constraints Table
    lines.append("### Constraints")
    lines.append("")
    lines.extend(_format_constraints_table(task))

    # Hops Table
    lines.append("### Evidence Hops")
    lines.append("")
    tree = task.get("tree", {})
    hops = tree.get("hops", [])

    lines.append("| Hop | Source | Chunk ID | Bridge Query |")
    lines.append("|:---:|:---:|:---|:---|")

    for hop in hops:
        hop_id = hop.get("hop_id", "?")
        source = hop.get("source_type", "?")
        chunk_id = hop.get("chunk_id", "")[:40]
        edge = hop.get("edge", {})
        query = (edge.get("query") or "")[:50]
        source_icon = "📁" if source == "local" else "🌐"
        lines.append(f"| {hop_id} | {source_icon} {source} | `{_md_escape(chunk_id)}...` | {_md_escape(query)} |")

    lines.append("")

    # Show chunk text if requested
    if show_text and hops:
        lines.append("### Chunk Texts")
        lines.append("")
        for hop in hops:
            chunk_id = hop.get("chunk_id", "")
            if chunk_id in chunk_map:
                text = chunk_map[chunk_id].get("text", "")[:500]
                lines.append(f"**{hop.get('source_type', 'local')} - {chunk_id}**")
                lines.append("")
                lines.append(f"```")
                lines.append(text)
                lines.append("```")
                lines.append("")

    # Gold Evidence
    lines.append("### Gold Evidence")
    lines.append("")
    answer_evidence = gold.get("answer_evidence", {})
    if answer_evidence:
        chunk_id = answer_evidence.get("chunk_id", "")
        char_start = answer_evidence.get("char_start", 0)
        char_end = answer_evidence.get("char_end", 0)

        lines.append(f"- **Chunk**: `{chunk_id}`")
        lines.append(f"- **Span**: [{char_start}, {char_end}]")

        if chunk_id in chunk_map:
            text = chunk_map[chunk_id].get("text", "")
            snippet = _snippet(text, char_start, char_end, ctx_chars)
            lines.append(f"- **Snippet**: {snippet}")
        elif answer_evidence.get("text"):
            lines.append(f"- **Text**: `{_md_escape(answer_evidence.get('text'))}`")

    source_secret = gold.get("source_secret", {})
    if source_secret:
        lines.append(f"- **Source Question**: {_md_escape(source_secret.get('question', ''))}")
        lines.append(f"- **Secret Type**: `{source_secret.get('secret_type', '')}`")

    lines.append("")

    # Diversity Metadata
    lines.append("### Diversity Metadata")
    lines.append("")
    lines.extend(_format_diversity(task))

    # Quality / Validation
    lines.append("### Validation Quality")
    lines.append("")
    lines.extend(_format_quality(task))

    # Privacy
    lines.append("### Privacy")
    lines.append("")
    lines.extend(_format_privacy(task))

    lines.append("---")
    lines.append("")

    return lines


def format_task_terminal(task: Dict[str, Any], idx: int) -> str:
    """Format a single task for terminal output with colors."""
    C = Colors
    lines = []

    mode = task.get("mode", "unknown")
    question = task.get("question", "")
    answer = task.get("answer", "")
    gold = task.get("gold", {})
    linkage_type = gold.get("linkage_type", "")
    diversity = task.get("diversity", {})

    # Header
    lines.append(f"{C.BOLD}{C.HEADER}═══ Task {idx}: {mode.upper()} ═══{C.RESET}")

    if linkage_type:
        lines.append(f"{C.CYAN}Linkage: {linkage_type} — {_linkage_description(linkage_type)}{C.RESET}")

    lines.append("")
    lines.append(f"{C.BOLD}Question:{C.RESET}")
    lines.append(f"  {C.GREEN}{textwrap.fill(question, width=80, subsequent_indent='  ')}{C.RESET}")
    lines.append("")
    lines.append(f"{C.BOLD}Answer:{C.RESET} {C.YELLOW}{answer}{C.RESET}")
    lines.append("")

    # Hop pattern
    hop_pattern = diversity.get("hop_pattern", "")
    if hop_pattern:
        pattern_colored = ""
        for c in hop_pattern:
            if c == "L":
                pattern_colored += f"{C.BLUE}L{C.RESET}"
            else:
                pattern_colored += f"{C.CYAN}W{C.RESET}"
        lines.append(f"{C.BOLD}Hop Pattern:{C.RESET} {pattern_colored}")

    # Constraints summary
    tree = task.get("tree", {})
    hcsp = tree.get("hcsp", {})
    nodes = hcsp.get("nodes", {})
    constraint_count = sum(1 for n in nodes.values() if n.get("kind") == "constraint")
    lines.append(f"{C.BOLD}Constraints:{C.RESET} {constraint_count}")

    # Quality
    quality = task.get("quality", {})
    if quality:
        det_pass = quality.get("deterministic_pass")
        full_pass = quality.get("ablation_full_info_pass")
        status = f"{C.GREEN}✓{C.RESET}" if det_pass else f"{C.RED}✗{C.RESET}"
        ablation = f"{C.GREEN}✓{C.RESET}" if full_pass else f"{C.RED}✗{C.RESET}" if full_pass is not None else "-"
        lines.append(f"{C.BOLD}Validation:{C.RESET} det={status} ablation={ablation}")

    lines.append("")
    lines.append(f"{C.DIM}{'─' * 60}{C.RESET}")
    lines.append("")

    return "\n".join(lines)


def format_summary(tasks: List[Dict[str, Any]]) -> List[str]:
    """Format summary statistics."""
    lines = []
    lines.append("# Summary Statistics")
    lines.append("")

    total = len(tasks)
    if total == 0:
        lines.append("No tasks found.")
        return lines

    # Mode distribution
    modes = {}
    for t in tasks:
        mode = t.get("mode", "unknown")
        modes[mode] = modes.get(mode, 0) + 1

    lines.append("## Mode Distribution")
    lines.append("")
    for mode, count in sorted(modes.items()):
        pct = count / total * 100
        lines.append(f"- **{mode}**: {count} ({pct:.1f}%)")
    lines.append("")

    # Linkage type distribution
    linkages = {}
    for t in tasks:
        lt = t.get("gold", {}).get("linkage_type", "none")
        linkages[lt] = linkages.get(lt, 0) + 1

    lines.append("## Linkage Type Distribution")
    lines.append("")
    for lt, count in sorted(linkages.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        emoji = _linkage_emoji(lt)
        lines.append(f"- {emoji} **{lt}**: {count} ({pct:.1f}%)")
    lines.append("")

    # Hop pattern distribution
    patterns = {}
    for t in tasks:
        pattern = t.get("diversity", {}).get("hop_pattern", "?")
        patterns[pattern] = patterns.get(pattern, 0) + 1

    lines.append("## Hop Pattern Distribution")
    lines.append("")
    for pattern, count in sorted(patterns.items(), key=lambda x: -x[1])[:10]:
        pct = count / total * 100
        lines.append(f"- `{pattern}`: {count} ({pct:.1f}%)")
    lines.append("")

    # Quality metrics
    quality_tasks = [t for t in tasks if t.get("quality")]
    if quality_tasks:
        det_pass = sum(1 for t in quality_tasks if t.get("quality", {}).get("deterministic_pass"))
        full_pass = sum(1 for t in quality_tasks if t.get("quality", {}).get("ablation_full_info_pass"))
        no_pass = sum(1 for t in quality_tasks if t.get("quality", {}).get("ablation_no_info_pass"))

        lines.append("## Quality Metrics")
        lines.append("")
        lines.append(f"- **Deterministic Pass**: {det_pass}/{len(quality_tasks)} ({det_pass/len(quality_tasks)*100:.1f}%)")
        lines.append(f"- **Full Info Ablation Pass**: {full_pass}/{len(quality_tasks)} ({full_pass/len(quality_tasks)*100:.1f}%)")
        lines.append(f"- **No Info Ablation Pass**: {no_pass}/{len(quality_tasks)} ({no_pass/len(quality_tasks)*100:.1f}%)")
        lines.append("")

    # Constraint stats
    constraint_counts = []
    for t in tasks:
        nodes = t.get("tree", {}).get("hcsp", {}).get("nodes", {})
        count = sum(1 for n in nodes.values() if n.get("kind") == "constraint")
        constraint_counts.append(count)

    if constraint_counts:
        lines.append("## Constraint Statistics")
        lines.append("")
        lines.append(f"- **Min**: {min(constraint_counts)}")
        lines.append(f"- **Max**: {max(constraint_counts)}")
        lines.append(f"- **Avg**: {sum(constraint_counts)/len(constraint_counts):.1f}")
        lines.append("")

    return lines


def main() -> int:
    args = _parse_args()
    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        return 1

    # Load chunks if provided
    chunk_map = {}
    if args.local_chunks:
        chunks_path = Path(args.local_chunks)
        if chunks_path.exists():
            print(f"Loading chunks from {chunks_path}...")
            chunk_map = _load_chunks_map(chunks_path)
            print(f"  Loaded {len(chunk_map)} chunks")
    else:
        # Try to find chunks in same directory
        parent = dataset_path.parent
        for candidate in [parent / "chunks_local.jsonl", parent.parent / "chunks_local.jsonl"]:
            if candidate.exists():
                print(f"Auto-loading chunks from {candidate}...")
                chunk_map = _load_chunks_map(candidate)
                print(f"  Loaded {len(chunk_map)} chunks")
                break

    # Load tasks
    tasks = list(_iter_jsonl(dataset_path))
    print(f"Loaded {len(tasks)} tasks from {dataset_path}")

    if args.format == "terminal":
        # Terminal output
        for i, task in enumerate(tasks[:args.max_items], 1):
            print(format_task_terminal(task, i))

        if len(tasks) > args.max_items:
            print(f"... and {len(tasks) - args.max_items} more tasks")
    else:
        # Markdown output
        out_path = Path(args.out) if args.out else dataset_path.with_suffix(".report.md")

        lines = []
        lines.append(f"# HCSP Dataset Report: `{dataset_path.name}`")
        lines.append("")
        lines.append(f"*Generated from {len(tasks)} tasks*")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Summary first
        lines.extend(format_summary(tasks))
        lines.append("---")
        lines.append("")

        # Individual tasks
        lines.append("# Task Details")
        lines.append("")

        for i, task in enumerate(tasks[:args.max_items], 1):
            lines.extend(format_task_markdown(task, i, chunk_map, args.context_chars, args.show_text))

        if len(tasks) > args.max_items:
            lines.append(f"*... and {len(tasks) - args.max_items} more tasks not shown*")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
