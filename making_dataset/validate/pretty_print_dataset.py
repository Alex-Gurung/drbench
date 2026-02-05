#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretty-print dataset tasks (local_only/web_only/mixed) into a Markdown report."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset JSONL (e.g., outputs/mixed.jsonl)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output Markdown path (default: alongside dataset, with .md suffix)",
    )
    parser.add_argument(
        "--local-chunks",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "chunks_local.jsonl"),
        help="Path to chunks_local.jsonl (used for local hop previews)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=10,
        help="Max number of tasks to include (default: 10)",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=180,
        help="Chars of context to show around evidence offsets (default: 180)",
    )
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_local_chunks(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("chunk_id")
            if cid:
                out[str(cid)] = obj
    return out


def _snippet(text: str, start: int, end: int, *, ctx: int) -> str:
    start = max(0, int(start))
    end = min(len(text), int(end))
    left = max(0, start - ctx)
    right = min(len(text), end + ctx)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    mid = text[left:start] + "<<" + text[start:end] + ">>" + text[end:right]
    return (prefix + mid + suffix).replace("\n", "\\n")


def _md_escape(s: str) -> str:
    # Minimal; just avoid breaking tables.
    return (s or "").replace("|", "\\|")


def main() -> int:
    args = _parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    out_path = Path(args.out) if args.out else dataset_path.with_suffix(dataset_path.suffix + ".md")
    local_chunks_path = Path(args.local_chunks)
    if not local_chunks_path.exists():
        raise FileNotFoundError(f"Local chunks not found: {local_chunks_path}")

    local_chunks = _load_local_chunks(local_chunks_path)

    lines: list[str] = []
    lines.append(f"# Dataset Preview: `{dataset_path}`")
    lines.append("")

    n = 0
    for task in _iter_jsonl(dataset_path):
        n += 1
        if n > args.max_items:
            break

        mode = task.get("mode")
        q = (task.get("question") or "").strip()
        a = (task.get("answer") or "").strip()

        lines.append(f"## Task {n} ({_md_escape(str(mode))})")
        lines.append("")
        lines.append("**Question**")
        lines.append("")
        lines.append(textwrap.fill(q, width=100))
        lines.append("")
        lines.append(f"**Answer**: `{_md_escape(a)}`")
        lines.append("")

        # Tree
        hops = (task.get("tree") or {}).get("hops") or []
        lines.append("**Tree Hops**")
        lines.append("")
        lines.append("| Hop | Source | Doc ID | Chunk ID | Edge Query |")
        lines.append("| ---: | --- | --- | --- | --- |")
        for hop in hops:
            edge = hop.get("edge") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(hop.get("hop_id")),
                        _md_escape(str(hop.get("source_type") or "")),
                        f"`{_md_escape(str(hop.get('doc_id') or ''))}`",
                        f"`{_md_escape(str(hop.get('chunk_id') or ''))}`",
                        _md_escape(str(edge.get("query") or "")),
                    ]
                )
                + " |"
            )
        lines.append("")

        # Evidence / gold
        gold = task.get("gold") or {}
        if mode == "local_only":
            target = gold.get("target_chunk_id")
            if target and target in local_chunks:
                txt = local_chunks[target].get("text") or ""
                if "answer_char_start" in gold:
                    lines.append("**Gold Evidence (Local Answer Span)**")
                    lines.append("")
                    lines.append(f"- target_chunk_id: `{_md_escape(str(target))}`")
                    lines.append(
                        f"- snippet: `{_md_escape(_snippet(txt, gold['answer_char_start'], gold['answer_char_end'], ctx=args.context_chars))}`"
                    )
                    lines.append("")
        elif mode == "mixed":
            loc = (gold.get("local") or {}) if isinstance(gold.get("local"), dict) else {}
            web = (gold.get("web") or {}) if isinstance(gold.get("web"), dict) else {}
            lines.append("**Gold Evidence (Mixed)**")
            lines.append("")
            loc_cid = loc.get("chunk_id")
            if loc_cid and loc_cid in local_chunks:
                txt = local_chunks[loc_cid].get("text") or ""
                if "value_char_start" in loc:
                    lines.append(f"- local value: `{_md_escape(str(loc.get('value_str')))}`")
                    lines.append(f"  - chunk_id: `{_md_escape(str(loc_cid))}`")
                    lines.append(
                        f"  - snippet: `{_md_escape(_snippet(txt, loc['value_char_start'], loc['value_char_end'], ctx=args.context_chars))}`"
                    )
            if web:
                lines.append(f"- web value: `{_md_escape(str(web.get('value_str')))}`")
                lines.append(f"  - doc_id: `{_md_escape(str(web.get('doc_id')))}`")
                lines.append(
                    f"  - excerpt: `{_md_escape(str(web.get('excerpt') or '')[:300]).replace('\\n','\\\\n')}`"
                )
            comp = gold.get("compute") or {}
            if comp:
                lines.append(f"- compute: `{_md_escape(str(comp.get('formula') or ''))}`")
            ab = gold.get("ablation_check") or {}
            if ab:
                lines.append(
                    f"- ablation: with_both=`{_md_escape(str(ab.get('with_both')))}`, "
                    f"local_only=`{_md_escape(str(ab.get('local_only')))}`, "
                    f"web_only=`{_md_escape(str(ab.get('web_only')))}`"
                )
            lines.append("")
        elif mode == "web_only":
            ev = gold.get("answer_evidence") or {}
            if ev:
                lines.append("**Gold Evidence (Web Answer Span)**")
                lines.append("")
                lines.append(
                    f"- doc_id: `{_md_escape(str(ev.get('doc_id') or ''))}`, "
                    f"char_start={ev.get('char_start')}, char_end={ev.get('char_end')}"
                )
                lines.append("")

        # Privacy
        privacy = task.get("privacy") or {}
        req = privacy.get("required_secrets") or []
        un = privacy.get("unnecessary_secrets") or []
        lines.append(
            f"**Privacy**: required={len(req)} unnecessary={len(un)}"
        )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
