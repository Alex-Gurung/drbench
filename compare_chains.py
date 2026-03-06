"""Compare chain builder outputs from two runs (e.g. entity-only vs BM25).

Reads two JSONL files produced by chain_builder and generates a side-by-side
HTML comparison showing questions, answers, bridge entities, and verification.

Usage:
    python compare_chains.py /tmp/chains_entity_only.jsonl /tmp/chains_bm25.jsonl
    python compare_chains.py /tmp/chains_entity_only.jsonl /tmp/chains_bm25.jsonl --open
"""

import argparse
import html as html_mod
import json
import subprocess
import sys
from pathlib import Path


def _load(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _esc(s) -> str:
    return html_mod.escape(str(s))


def generate_html(a_chains: list[dict], b_chains: list[dict],
                  a_label: str, b_label: str, output: Path) -> None:
    n = max(len(a_chains), len(b_chains))

    def render_chain(c: dict | None) -> str:
        if c is None:
            return '<div class="empty">No chain</div>'

        meta = c.get("metadata", {})
        verif = c.get("verification", {})
        valid = verif.get("is_valid", False)
        complete = meta.get("complete", False)
        status_cls = "valid" if valid else ("complete" if complete else "incomplete")
        status_txt = "VALID" if valid else ("COMPLETE" if complete else "INCOMPLETE")

        hops = c.get("hops", [])
        trace = meta.get("trace", [])

        # Extract bridge info from trace
        bridges = {}
        for t in trace:
            if t.get("step") == "find_bridge":
                trans = t.get("transition", 0)
                bridges[trans] = {
                    "n_candidates": t.get("n_candidates", 0),
                    "top": t.get("top_candidates", [])[:3],
                }

        html = f'<div class="chain-status {status_cls}">{status_txt}</div>'
        html += f'<div class="chain-meta">{meta.get("n_hops", 0)} hops, '
        html += f'{meta.get("llm_calls", 0)} LLM calls, '
        html += f'{meta.get("elapsed_seconds", 0):.0f}s</div>'

        # Numbered question
        nq = c.get("numbered_questions", "")
        if nq:
            html += f'<div class="numbered-q">'
            for line in nq.splitlines():
                html += f'{_esc(line)}<br>'
            html += '</div>'

        html += f'<div class="final-answer">Final: <b>{_esc(c.get("global_answer", "?"))}</b></div>'

        # Hops detail
        for hop in hops:
            htype = hop.get("hop_type", "?")
            type_cls = "local" if htype == "L" else "web"
            doc_short = (hop.get("doc_id") or "").split("/")[-1][:50]
            quote = hop.get("quote", "")
            quote_disp = (quote[:180] + "...") if len(quote) > 180 else quote

            html += f'<div class="hop">'
            html += f'<div class="hop-header">'
            html += f'<span class="hop-type {type_cls}">{htype}</span> '
            html += f'<span class="hop-num">Q{hop.get("hop_number", "?")}</span>'
            html += f'</div>'
            html += f'<div class="hop-q">{_esc(hop.get("question", ""))}</div>'
            html += f'<div class="hop-a">A: <b>{_esc(hop.get("answer", ""))}</b></div>'
            html += f'<div class="hop-doc">{_esc(doc_short)}</div>'
            if quote_disp:
                html += f'<div class="hop-quote">"{_esc(quote_disp)}"</div>'
            html += '</div>'

        # Bridge info from trace
        for trans_num, binfo in sorted(bridges.items()):
            html += f'<div class="bridge-info">'
            html += f'Transition {trans_num}: <b>{binfo["n_candidates"]}</b> candidates'
            for tc in binfo["top"]:
                doc_s = (tc.get("doc", "") or "")[-40:]
                html += f'<div class="bridge-cand">'
                html += f'{_esc(tc.get("entity", "?"))} → {_esc(doc_s)} '
                html += f'(intra={tc.get("needs_intra", "?")}, score={tc.get("score", 0):.0f})'
                html += '</div>'
            html += '</div>'

        # Verification detail
        if verif:
            html += '<div class="verif">'
            for k in ["no_docs_pass", "first_only_pass", "last_only_pass", "all_docs_pass"]:
                v = verif.get(k)
                if v is not None:
                    icon = "pass" if v else "fail"
                    html += f'<span class="verif-item {icon}">{k.replace("_", " ")}</span> '
            html += '</div>'

        return html

    pairs_html = ""
    for i in range(n):
        a = a_chains[i] if i < len(a_chains) else None
        b = b_chains[i] if i < len(b_chains) else None
        seed_a = (a or {}).get("hops", [{}])[0].get("doc_id", "?") if a else "?"
        seed_b = (b or {}).get("hops", [{}])[0].get("doc_id", "?") if b else "?"
        same_seed = seed_a == seed_b
        seed_badge = ' <span class="same-seed">same seed</span>' if same_seed else ' <span class="diff-seed">different seed</span>'

        pair_pattern = (a or b or {}).get("pattern", "?")
        pairs_html += f"""
        <div class="pair" data-pattern="{pair_pattern}">
            <div class="pair-header">Chain {i+1} <span class="filter-btn" style="cursor:default">{pair_pattern}</span>{seed_badge}</div>
            <div class="pair-row">
                <div class="pair-col">{render_chain(a)}</div>
                <div class="pair-col">{render_chain(b)}</div>
            </div>
        </div>"""

    # Summary stats
    def stats(chains):
        valid = sum(1 for c in chains if c.get("verification", {}).get("is_valid"))
        complete = sum(1 for c in chains if c.get("metadata", {}).get("complete"))
        avg_hops = sum(c.get("metadata", {}).get("n_hops", 0) for c in chains) / max(1, len(chains))
        avg_time = sum(c.get("metadata", {}).get("elapsed_seconds", 0) for c in chains) / max(1, len(chains))
        avg_llm = sum(c.get("metadata", {}).get("llm_calls", 0) for c in chains) / max(1, len(chains))
        # Count trivial rejections from trace
        trivial_rejected = 0
        for c in chains:
            for t in c.get("metadata", {}).get("trace", []):
                if t.get("step", "").startswith("check_trivial") and t.get("trivial"):
                    trivial_rejected += 1
        return {"n": len(chains), "valid": valid, "complete": complete,
                "avg_hops": avg_hops, "avg_time": avg_time, "avg_llm": avg_llm,
                "trivial_rejected": trivial_rejected}

    sa, sb = stats(a_chains), stats(b_chains)

    # Per-pattern breakdown
    all_patterns = sorted(set(
        c.get("pattern", "?") for c in a_chains + b_chains
    ))

    def pattern_table(chains, label):
        rows = ""
        for pat in all_patterns:
            pc = [c for c in chains if c.get("pattern") == pat]
            if not pc:
                continue
            s = stats(pc)
            valid_cls = "green" if s['valid'] == s['n'] else ("" if s['valid'] > 0 else "red")
            rows += f"""<tr>
                <td><b>{_esc(pat)}</b></td>
                <td>{s['n']}</td>
                <td class="{valid_cls}"><b>{s['valid']}</b></td>
                <td>{s['complete']}</td>
                <td>{s['avg_hops']:.1f}</td>
                <td>{s['avg_llm']:.0f}</td>
                <td>{s['avg_time']:.0f}s</td>
                <td>{s['trivial_rejected']}</td>
            </tr>"""
        return f"""<table class="pattern-table">
            <tr><th>Pattern</th><th>N</th><th>Valid</th><th>Complete</th>
                <th>Avg Hops</th><th>Avg LLM</th><th>Avg Time</th><th>Trivial Rej</th></tr>
            {rows}</table>"""

    pattern_table_a = pattern_table(a_chains, a_label)
    pattern_table_b = pattern_table(b_chains, b_label)

    page = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Chain Comparison: {_esc(a_label)} vs {_esc(b_label)}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #f5f5f5; color: #222; line-height: 1.5; }}
.container {{ max-width: 1600px; margin: 0 auto; padding: 16px; }}
header {{ background: #fff; padding: 16px 24px; border-bottom: 1px solid #ddd;
          margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
header h1 {{ font-size: 16px; }}

.summary {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }}
.summary-col {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; }}
.summary-col h2 {{ font-size: 14px; margin-bottom: 8px; }}
.summary-col .stats {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.stat {{ text-align: center; }}
.stat .val {{ font-size: 1.6em; font-weight: 700; }}
.stat .lbl {{ font-size: 11px; color: #888; text-transform: uppercase; }}
.stat .val.green {{ color: #10b981; }}
.stat .val.red {{ color: #ef4444; }}

.pair {{ margin-bottom: 20px; }}
.pair-header {{ font-weight: 600; font-size: 14px; padding: 8px 12px;
                background: #e8e8e8; border-radius: 8px 8px 0 0; }}
.same-seed {{ background: #dcfce7; color: #166534; padding: 1px 6px; border-radius: 8px;
              font-size: 11px; font-weight: 500; }}
.diff-seed {{ background: #fee2e2; color: #991b1b; padding: 1px 6px; border-radius: 8px;
              font-size: 11px; font-weight: 500; }}
.pair-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0; }}
.pair-col {{ background: #fff; border: 1px solid #ddd; padding: 14px;
             min-height: 100px; }}
.pair-col:first-child {{ border-right: 2px solid #4361ee; border-radius: 0 0 0 8px; }}
.pair-col:last-child {{ border-radius: 0 0 8px 0; }}

.chain-status {{ font-weight: 700; font-size: 13px; margin-bottom: 6px; }}
.chain-status.valid {{ color: #10b981; }}
.chain-status.complete {{ color: #f59e0b; }}
.chain-status.incomplete {{ color: #ef4444; }}
.chain-meta {{ color: #888; font-size: 12px; margin-bottom: 8px; }}
.numbered-q {{ background: #f8f9fa; border: 1px solid #e5e7eb; border-radius: 6px;
               padding: 10px; font-size: 13px; margin: 8px 0; line-height: 1.6; }}
.final-answer {{ font-size: 13px; margin: 6px 0 10px; padding: 6px 10px;
                 background: #eef1ff; border-radius: 4px; }}

.hop {{ border-left: 3px solid #ddd; padding: 6px 10px; margin: 6px 0; font-size: 13px; }}
.hop-header {{ margin-bottom: 2px; }}
.hop-type {{ display: inline-block; width: 18px; height: 18px; border-radius: 3px;
             text-align: center; font-size: 11px; font-weight: 700; color: #fff; line-height: 18px; }}
.hop-type.local {{ background: #4361ee; }}
.hop-type.web {{ background: #7209b7; }}
.hop-num {{ font-weight: 600; }}
.hop-q {{ color: #333; }}
.hop-a {{ color: #333; }}
.hop-doc {{ color: #999; font-size: 11px; }}
.hop-quote {{ color: #666; font-size: 12px; font-style: italic; margin-top: 2px; }}

.bridge-info {{ background: #f0f4ff; border: 1px solid #c7d2fe; border-radius: 6px;
                padding: 8px 10px; margin: 6px 0; font-size: 12px; }}
.bridge-cand {{ color: #555; margin-left: 12px; }}

.verif {{ margin-top: 8px; }}
.verif-item {{ display: inline-block; padding: 1px 8px; border-radius: 10px;
               font-size: 11px; margin: 2px; }}
.verif-item.pass {{ background: #dcfce7; color: #166534; }}
.verif-item.fail {{ background: #fee2e2; color: #991b1b; }}

.empty {{ color: #aaa; font-style: italic; padding: 20px; }}

.pattern-table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }}
.pattern-table th {{ text-align: left; padding: 6px 8px; background: #f5f5f5; border: 1px solid #ddd;
                      font-size: 11px; text-transform: uppercase; color: #888; }}
.pattern-table td {{ padding: 6px 8px; border: 1px solid #ddd; }}
.pattern-table td.green {{ color: #10b981; }}
.pattern-table td.red {{ color: #ef4444; }}

.filters {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 16px; }}
.filter-btn {{ padding: 4px 12px; border-radius: 16px; border: 1px solid #ddd; background: #fff;
               color: #888; cursor: pointer; font-size: 12px; font-weight: 500; }}
.filter-btn:hover, .filter-btn.active {{ background: #eef1ff; color: #4361ee; border-color: #4361ee; }}
.pair.hidden {{ display: none; }}
</style>
<script>
function filterPattern(pat) {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    document.querySelectorAll('.pair').forEach(p => {{
        if (pat === 'all' || p.dataset.pattern === pat) {{
            p.classList.remove('hidden');
        }} else {{
            p.classList.add('hidden');
        }}
    }});
}}
</script>
</head><body>

<header><h1>Chain Comparison: {_esc(a_label)} vs {_esc(b_label)}</h1></header>
<div class="container">

<div class="summary">
  <div class="summary-col">
    <h2>{_esc(a_label)}</h2>
    <div class="stats">
      <div class="stat"><div class="val">{sa['n']}</div><div class="lbl">chains</div></div>
      <div class="stat"><div class="val green">{sa['valid']}</div><div class="lbl">valid</div></div>
      <div class="stat"><div class="val">{sa['complete']}</div><div class="lbl">complete</div></div>
      <div class="stat"><div class="val">{sa['avg_hops']:.1f}</div><div class="lbl">avg hops</div></div>
      <div class="stat"><div class="val">{sa['avg_llm']:.0f}</div><div class="lbl">avg LLM</div></div>
      <div class="stat"><div class="val">{sa['avg_time']:.0f}s</div><div class="lbl">avg time</div></div>
      <div class="stat"><div class="val">{sa['trivial_rejected']}</div><div class="lbl">trivial rej</div></div>
    </div>
    {pattern_table_a}
  </div>
  <div class="summary-col">
    <h2>{_esc(b_label)}</h2>
    <div class="stats">
      <div class="stat"><div class="val">{sb['n']}</div><div class="lbl">chains</div></div>
      <div class="stat"><div class="val green">{sb['valid']}</div><div class="lbl">valid</div></div>
      <div class="stat"><div class="val">{sb['complete']}</div><div class="lbl">complete</div></div>
      <div class="stat"><div class="val">{sb['avg_hops']:.1f}</div><div class="lbl">avg hops</div></div>
      <div class="stat"><div class="val">{sb['avg_llm']:.0f}</div><div class="lbl">avg LLM</div></div>
      <div class="stat"><div class="val">{sb['avg_time']:.0f}s</div><div class="lbl">avg time</div></div>
      <div class="stat"><div class="val">{sb['trivial_rejected']}</div><div class="lbl">trivial rej</div></div>
    </div>
    {pattern_table_b}
  </div>
</div>

<div class="filters">
  <span class="filter-btn active" onclick="filterPattern('all')">All ({n})</span>
  {"".join(f'<span class="filter-btn" onclick="filterPattern({chr(39)}{p}{chr(39)})">{p} ({sum(1 for c in a_chains if c.get("pattern")==p)})</span>' for p in all_patterns)}
</div>

{pairs_html}
</div></body></html>"""

    output.write_text(page, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Compare two chain builder runs side-by-side")
    p.add_argument("file_a", help="First JSONL (e.g. entity-only)")
    p.add_argument("file_b", help="Second JSONL (e.g. BM25)")
    p.add_argument("--label-a", default=None, help="Label for first run")
    p.add_argument("--label-b", default=None, help="Label for second run")
    p.add_argument("--output", default=None, help="Output HTML path")
    p.add_argument("--open", action="store_true", help="Open in browser")
    args = p.parse_args()

    path_a, path_b = Path(args.file_a), Path(args.file_b)
    a_chains = _load(path_a)
    b_chains = _load(path_b)

    label_a = args.label_a or path_a.stem
    label_b = args.label_b or path_b.stem

    out = Path(args.output) if args.output else Path(f"/tmp/chain_comparison.html")
    generate_html(a_chains, b_chains, label_a, label_b, out)
    print(f"Generated: {out} ({len(a_chains)} vs {len(b_chains)} chains)")

    if args.open:
        try:
            subprocess.run(["xdg-open", str(out)], check=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
