"""Pretty HTML viewer for chain builder output.

Reads a JSONL file from chain_builder and generates an interactive HTML page
with summary stats, per-pattern breakdown, filters, and detailed chain cards.

Usage:
    python view_generated_chains.py chains.jsonl
    python view_generated_chains.py chains.jsonl --open
    python view_generated_chains.py chains.jsonl -o custom_output.html --open
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


def _check_passed(t: dict) -> bool:
    """Determine if a check trace entry represents a pass."""
    step = t.get("step", "")
    if step in ("check_intra", "check_inter"):
        return t.get("passed", False)
    if step.startswith("check_trivial"):
        return not t.get("trivial", False)
    if step.startswith("check_backref"):
        return not t.get("independent", False)
    return True


def _stats(chains: list[dict]) -> dict:
    valid = sum(1 for c in chains if c.get("verification", {}).get("is_valid"))
    complete = sum(1 for c in chains if c.get("metadata", {}).get("complete"))
    avg_hops = sum(c.get("metadata", {}).get("n_hops", 0) for c in chains) / max(1, len(chains))
    avg_time = sum(c.get("metadata", {}).get("elapsed_seconds", 0) for c in chains) / max(1, len(chains))
    avg_llm = sum(c.get("metadata", {}).get("llm_calls", 0) for c in chains) / max(1, len(chains))
    trivial_rejected = 0
    backref_rejected = 0
    deterministic_rejected = 0
    for c in chains:
        for t in c.get("metadata", {}).get("trace", []):
            if t.get("step", "").startswith("check_trivial") and t.get("trivial"):
                trivial_rejected += 1
            if t.get("step", "").startswith("check_backref") and t.get("independent"):
                backref_rejected += 1
            if t.get("step") in ("check_intra", "check_inter") and not t.get("passed"):
                deterministic_rejected += 1
    return {"n": len(chains), "valid": valid, "complete": complete,
            "avg_hops": avg_hops, "avg_time": avg_time, "avg_llm": avg_llm,
            "trivial_rejected": trivial_rejected, "backref_rejected": backref_rejected,
            "deterministic_rejected": deterministic_rejected}


def _render_chain(c: dict, idx: int) -> str:
    meta = c.get("metadata", {})
    verif = c.get("verification", {})
    valid = verif.get("is_valid", False)
    complete = meta.get("complete", False)
    status_cls = "valid" if valid else ("complete" if complete else "incomplete")
    status_txt = "VALID" if valid else ("COMPLETE-INVALID" if complete else "INCOMPLETE")
    pattern = c.get("pattern", "?")
    chain_id = c.get("chain_id", "?")
    task_id = meta.get("task_id", "")

    hops = c.get("hops", [])
    trace = meta.get("trace", [])

    bridges = {}
    trivial_rejects = []
    check_trace = []  # All validation checks in order
    for t in trace:
        if t.get("step") == "find_bridge":
            trans = t.get("transition", 0)
            bridges[trans] = {
                "n_candidates": t.get("n_candidates", 0),
                "top": t.get("top_candidates", [])[:3],
            }
        if t.get("step", "").startswith("check_trivial") and t.get("trivial"):
            trivial_rejects.append({
                "q": t.get("question", "")[:80],
                "a": t.get("model_answer", ""),
                "why": t.get("justification", ""),
            })
        # Collect all validation-related trace entries
        step = t.get("step", "")
        if step in ("check_intra", "check_inter", "check_trivial_intra",
                     "check_trivial_inter", "check_backref_intra", "check_backref_inter"):
            check_trace.append(t)

    h = f'<div class="chain-card" data-pattern="{_esc(pattern)}" data-status="{status_cls}">'

    # Header row
    h += f'<div class="card-header {status_cls}-border">'
    h += f'<div class="header-left">'
    h += f'<span class="chain-idx">#{idx + 1}</span>'
    h += f'<span class="pattern-badge">{_esc(pattern)}</span>'
    h += f'<span class="status-badge {status_cls}">{status_txt}</span>'
    if task_id:
        h += f'<span class="task-badge">{_esc(task_id)}</span>'
    h += f'</div>'
    h += f'<div class="header-right">'
    h += f'<span class="meta-pill">{meta.get("n_hops", 0)} hops</span>'
    h += f'<span class="meta-pill">{meta.get("llm_calls", 0)} LLM calls</span>'
    h += f'<span class="meta-pill">{meta.get("elapsed_seconds", 0):.0f}s</span>'
    h += f'<span class="chain-id">{_esc(chain_id[:8])}</span>'
    h += f'</div>'
    h += f'</div>'

    h += '<div class="card-body">'

    # Numbered question
    nq = c.get("numbered_questions", "")
    if nq:
        h += f'<div class="numbered-q">'
        for line in nq.splitlines():
            h += f'{_esc(line)}<br>'
        h += '</div>'

    h += f'<div class="final-answer">Final answer: <strong>{_esc(c.get("global_answer", "?"))}</strong></div>'

    # Hops as a flow
    h += '<div class="hops-flow">'
    for i, hop in enumerate(hops):
        htype = hop.get("hop_type", "?")
        type_cls = "local" if htype == "L" else "web"
        doc_short = (hop.get("doc_id") or "").split("/")[-1][:45]
        doc_full = (hop.get("doc_id") or "")
        quote = hop.get("quote", "")
        quote_disp = (quote[:200] + "...") if len(quote) > 200 else quote

        h += f'<div class="hop {type_cls}-hop">'
        h += f'<div class="hop-head">'
        h += f'<span class="hop-type-tag {type_cls}">{htype}</span>'
        h += f'<span class="hop-label">Hop {hop.get("hop_number", "?")}</span>'
        h += f'</div>'
        h += f'<div class="hop-question">{_esc(hop.get("question", ""))}</div>'
        h += f'<div class="hop-answer">{_esc(hop.get("answer", ""))}</div>'
        h += f'<div class="hop-doc" title="{_esc(doc_full)}">{_esc(doc_short)}</div>'
        if quote_disp:
            h += f'<div class="hop-quote">{_esc(quote_disp)}</div>'
        h += '</div>'

        if i < len(hops) - 1:
            h += '<div class="hop-arrow">&#x2193;</div>'
    h += '</div>'

    # Bridge candidates
    if bridges:
        h += '<details class="trace-section"><summary>Bridge candidates</summary><div class="trace-body">'
        for trans_num, binfo in sorted(bridges.items()):
            h += f'<div class="bridge-info">'
            h += f'<span class="bridge-label">Transition {trans_num}:</span> <strong>{binfo["n_candidates"]}</strong> candidates'
            for tc in binfo["top"]:
                doc_s = (tc.get("doc", "") or "")[-40:]
                h += f'<div class="bridge-cand">'
                h += f'<code>{_esc(tc.get("entity", "?"))}</code> &rarr; {_esc(doc_s)} '
                h += f'<span class="bridge-meta">(intra={tc.get("needs_intra", "?")}, score={tc.get("score", 0):.0f})</span>'
                h += '</div>'
            h += '</div>'
        h += '</div></details>'

    # Trivial rejections
    if trivial_rejects:
        h += '<details class="trace-section"><summary>Trivial rejections ({n})</summary><div class="trace-body">'.format(
            n=len(trivial_rejects))
        for tr in trivial_rejects:
            h += f'<div class="trivial-rej">'
            h += f'<span class="trivial-q">{_esc(tr["q"])}</span> '
            h += f'&rarr; <strong>{_esc(tr["a"])}</strong>'
            if tr["why"]:
                h += f' <span class="trivial-why">{_esc(tr["why"][:100])}</span>'
            h += '</div>'
        h += '</div></details>'

    # Validation trace — every check with pass/fail detail
    if check_trace:
        n_pass = sum(1 for t in check_trace if _check_passed(t))
        n_fail = len(check_trace) - n_pass
        h += f'<details class="trace-section"><summary>Validation checks ({n_pass} passed, {n_fail} failed)</summary><div class="trace-body">'
        for t in check_trace:
            passed = _check_passed(t)
            step = t.get("step", "")
            trans = t.get("transition", "?")
            q_short = (t.get("question", "") or "")[:80]
            icon_cls = "pass" if passed else "fail"
            icon = "&#10003;" if passed else "&#10007;"

            h += f'<div class="check-entry {icon_cls}">'
            h += f'<span class="verif-item {icon_cls}">{icon}</span> '
            h += f'<code>{_esc(step)}</code> <span class="bridge-meta">(transition {trans})</span><br>'
            h += f'<span class="trivial-q">{_esc(q_short)}</span>'

            if step in ("check_intra", "check_inter"):
                err = t.get("error")
                ans = t.get("answer", "")
                quote = (t.get("quote", "") or "")[:100]
                if err:
                    h += f'<br><span style="color:var(--red)">FAIL: {_esc(err)}</span>'
                else:
                    h += f' &rarr; <strong>{_esc(ans)}</strong>'
                if quote:
                    h += f'<br><span class="trivial-why">Quote: {_esc(quote)}</span>'
            elif step.startswith("check_trivial"):
                trivial = t.get("trivial", False)
                model_ans = t.get("model_answer", "")
                just = t.get("justification", "")
                if trivial:
                    h += f'<br><span style="color:var(--red)">TRIVIAL: model answered "{_esc(model_ans)}"</span>'
                    if just:
                        h += f' <span class="trivial-why">{_esc(just[:120])}</span>'
                else:
                    h += f'<br><span style="color:var(--green)">Not trivial (requires document)</span>'
                    if just:
                        h += f' <span class="trivial-why">{_esc(just[:120])}</span>'
            elif step.startswith("check_backref"):
                independent = t.get("independent", False)
                model_ans = t.get("model_answer", "")
                just = t.get("justification", "")
                prev = t.get("prev_answer", "")
                if independent:
                    h += f'<br><span style="color:var(--red)">INDEPENDENT: answerable without "{_esc(prev)}" &rarr; "{_esc(model_ans)}"</span>'
                    if just:
                        h += f' <span class="trivial-why">{_esc(just[:120])}</span>'
                else:
                    h += f'<br><span style="color:var(--green)">Back-ref required (needs "{_esc(prev)}")</span>'
                    if just:
                        h += f' <span class="trivial-why">{_esc(just[:120])}</span>'

            h += '</div>'
        h += '</div></details>'

    # Verification
    if verif:
        h += '<div class="verif">'
        for k in ["no_docs_pass", "first_only_pass", "last_only_pass", "all_docs_pass"]:
            v = verif.get(k)
            if v is not None:
                icon = "pass" if v else "fail"
                label = k.replace("_pass", "").replace("_", " ")
                h += f'<span class="verif-item {icon}">{label} {"&#10003;" if v else "&#10007;"}</span>'
        h += '</div>'

    h += '</div></div>'
    return h


def generate_html(chains: list[dict], output: Path, title: str) -> None:
    all_patterns = sorted(set(c.get("pattern", "?") for c in chains))
    overall = _stats(chains)

    # Per-pattern stats
    pattern_rows = ""
    for pat in all_patterns:
        pc = [c for c in chains if c.get("pattern") == pat]
        s = _stats(pc)
        valid_pct = f"{100 * s['valid'] / s['n']:.0f}%" if s["n"] else "—"
        pattern_rows += f"""<tr>
            <td><strong>{_esc(pat)}</strong></td><td>{s['n']}</td>
            <td class="num-valid"><strong>{s['valid']}</strong> <span class="pct">{valid_pct}</span></td>
            <td>{s['complete']}</td>
            <td>{s['avg_hops']:.1f}</td><td>{s['avg_llm']:.0f}</td>
            <td>{s['avg_time']:.0f}s</td>
            <td>{s['deterministic_rejected']}</td>
            <td>{s['trivial_rejected']}</td>
            <td>{s['backref_rejected']}</td>
        </tr>"""

    # Filter buttons — two independent rows: pattern + status
    pattern_btns = '<button class="filter-btn active" data-group="pattern" data-filter="all">All patterns</button>'
    for pat in all_patterns:
        n_valid = sum(1 for c in chains if c.get("pattern") == pat and c.get("verification", {}).get("is_valid"))
        n_pat = sum(1 for c in chains if c.get("pattern") == pat)
        pattern_btns += f'<button class="filter-btn" data-group="pattern" data-filter="{_esc(pat)}">{_esc(pat)} <span class="btn-count">{n_valid}/{n_pat}</span></button>'

    status_btns = '<button class="filter-btn active" data-group="status" data-filter="all">Any status</button>'
    status_btns += '<button class="filter-btn filter-valid" data-group="status" data-filter="valid">Valid</button>'
    status_btns += '<button class="filter-btn filter-complete" data-group="status" data-filter="complete">Complete-invalid</button>'
    status_btns += '<button class="filter-btn filter-incomplete" data-group="status" data-filter="incomplete">Incomplete</button>'

    cards = ""
    for i, c in enumerate(chains):
        cards += _render_chain(c, i)

    valid_pct = f"{100 * overall['valid'] / overall['n']:.0f}" if overall["n"] else "0"

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
:root {{
    --bg: #f8f9fb;
    --card: #ffffff;
    --border: #e5e7eb;
    --border-light: #f0f1f3;
    --text: #1a1a2e;
    --text-secondary: #6b7280;
    --text-muted: #9ca3af;
    --accent: #6366f1;
    --accent-light: #eef2ff;
    --green: #059669;
    --green-bg: #ecfdf5;
    --green-border: #a7f3d0;
    --red: #dc2626;
    --red-bg: #fef2f2;
    --red-border: #fecaca;
    --amber: #d97706;
    --amber-bg: #fffbeb;
    --amber-border: #fde68a;
    --local-color: #6366f1;
    --web-color: #8b5cf6;
    --radius: 10px;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
    --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        background: var(--bg); color: var(--text); line-height: 1.6; font-size: 14px; }}

/* Top bar */
.topbar {{ background: var(--card); border-bottom: 1px solid var(--border);
           padding: 14px 24px; position: sticky; top: 0; z-index: 100;
           backdrop-filter: blur(8px); }}
.topbar-inner {{ max-width: 1000px; margin: 0 auto;
                 display: flex; align-items: center; justify-content: space-between; }}
.topbar h1 {{ font-size: 15px; font-weight: 600; letter-spacing: -0.01em; }}
.topbar .subtitle {{ font-size: 12px; color: var(--text-muted); }}

.container {{ max-width: 1000px; margin: 0 auto; padding: 20px 16px; }}

/* Stats row */
.stats-row {{ display: flex; gap: 12px; margin-bottom: 20px; }}
.stat-card {{ flex: 1; background: var(--card); border: 1px solid var(--border);
              border-radius: var(--radius); padding: 16px; text-align: center;
              box-shadow: var(--shadow-sm); }}
.stat-card .val {{ font-size: 28px; font-weight: 700; letter-spacing: -0.02em; display: block; }}
.stat-card .lbl {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase;
                   letter-spacing: 0.05em; margin-top: 2px; }}
.val-green {{ color: var(--green); }}
.val-red {{ color: var(--red); }}

/* Pattern table */
.table-wrap {{ background: var(--card); border: 1px solid var(--border);
               border-radius: var(--radius); overflow: hidden; margin-bottom: 20px;
               box-shadow: var(--shadow-sm); }}
.pattern-table {{ width: 100%; border-collapse: collapse; }}
.pattern-table th {{ background: var(--bg); font-size: 11px; text-transform: uppercase;
                     letter-spacing: 0.05em; color: var(--text-muted); padding: 10px 14px;
                     text-align: left; font-weight: 500; }}
.pattern-table td {{ padding: 10px 14px; border-top: 1px solid var(--border-light); font-size: 13px; }}
.pattern-table tr:hover td {{ background: #fafbfc; }}
.num-valid {{ color: var(--green); }}
.pct {{ color: var(--text-muted); font-size: 11px; }}

/* Filters */
.filters {{ margin-bottom: 20px; display: flex; gap: 6px; flex-wrap: wrap; }}
.filter-rows {{ display: flex; flex-direction: column; gap: 6px; }}
.filter-btn {{ border: 1px solid var(--border); background: var(--card); border-radius: 20px;
               padding: 6px 16px; font-size: 12px; cursor: pointer; transition: all 0.15s;
               font-weight: 500; color: var(--text-secondary); }}
.filter-btn:hover {{ background: var(--bg); border-color: #d1d5db; }}
.filter-btn.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
.btn-count {{ opacity: 0.7; font-weight: 400; }}
.filter-valid {{ color: var(--green) !important; border-color: var(--green-border) !important; }}
.filter-valid:hover {{ background: var(--green-bg) !important; }}
.filter-complete {{ color: var(--amber) !important; border-color: var(--amber-border) !important; }}
.filter-complete:hover {{ background: var(--amber-bg) !important; }}
.filter-incomplete {{ color: var(--red) !important; border-color: var(--red-border) !important; }}
.filter-incomplete:hover {{ background: var(--red-bg) !important; }}

/* Chain cards */
.chain-card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
               margin-bottom: 14px; box-shadow: var(--shadow-sm); overflow: hidden;
               transition: box-shadow 0.15s; }}
.chain-card:hover {{ box-shadow: var(--shadow); }}
.chain-card.hidden {{ display: none; }}

.card-header {{ padding: 12px 18px; display: flex; align-items: center;
                justify-content: space-between; border-bottom: 1px solid var(--border-light); }}
.valid-border {{ border-left: 3px solid var(--green); }}
.complete-border {{ border-left: 3px solid var(--amber); }}
.incomplete-border {{ border-left: 3px solid var(--red); }}
.header-left {{ display: flex; align-items: center; gap: 8px; }}
.header-right {{ display: flex; align-items: center; gap: 8px; }}

.chain-idx {{ font-weight: 700; color: var(--text-muted); font-size: 13px; min-width: 28px; }}
.pattern-badge {{ background: #f3f4f6; padding: 2px 10px; border-radius: 12px;
                  font-size: 11px; font-weight: 600; letter-spacing: 0.02em; color: var(--text); }}
.status-badge {{ padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;
                 letter-spacing: 0.01em; }}
.status-badge.valid {{ background: var(--green-bg); color: var(--green); }}
.status-badge.complete {{ background: var(--amber-bg); color: var(--amber); }}
.status-badge.incomplete {{ background: var(--red-bg); color: var(--red); }}
.task-badge {{ background: var(--accent-light); color: var(--accent); padding: 2px 8px;
              border-radius: 8px; font-size: 10px; font-weight: 500; }}
.meta-pill {{ font-size: 11px; color: var(--text-muted); background: var(--bg);
              padding: 2px 8px; border-radius: 8px; }}
.chain-id {{ color: var(--text-muted); font-size: 10px; font-family: 'SF Mono', 'Fira Code', monospace; }}

.card-body {{ padding: 16px 18px; }}

.numbered-q {{ background: var(--bg); border: 1px solid var(--border-light);
               border-radius: 8px; padding: 14px 18px; font-size: 13px; line-height: 1.8;
               margin-bottom: 12px; }}
.final-answer {{ font-size: 13px; padding: 8px 14px; background: var(--accent-light);
                 border-radius: 8px; margin-bottom: 16px; }}
.final-answer strong {{ color: var(--accent); }}

/* Hops */
.hops-flow {{ display: flex; flex-direction: column; gap: 0; }}
.hop {{ border: 1px solid var(--border-light); border-radius: 8px; padding: 12px 16px;
        position: relative; }}
.local-hop {{ border-left: 3px solid var(--local-color); }}
.web-hop {{ border-left: 3px solid var(--web-color); }}
.hop-head {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
.hop-type-tag {{ display: inline-flex; align-items: center; justify-content: center;
                 width: 22px; height: 22px; border-radius: 6px; font-size: 11px;
                 font-weight: 700; color: #fff; }}
.hop-type-tag.local {{ background: var(--local-color); }}
.hop-type-tag.web {{ background: var(--web-color); }}
.hop-label {{ font-weight: 600; font-size: 12px; color: var(--text-secondary); }}
.hop-question {{ color: var(--text); font-size: 13px; margin-bottom: 4px; }}
.hop-answer {{ color: var(--accent); font-weight: 600; font-size: 13px; margin-bottom: 4px; }}
.hop-doc {{ color: var(--text-muted); font-size: 11px; font-family: 'SF Mono', 'Fira Code', monospace; }}
.hop-quote {{ color: var(--text-secondary); font-size: 12px; font-style: italic; margin-top: 6px;
              padding: 8px 12px; background: var(--bg); border-radius: 6px; line-height: 1.5; }}
.hop-arrow {{ text-align: center; color: var(--text-muted); font-size: 16px; line-height: 1;
              padding: 2px 0; }}

/* Trace sections */
.trace-section {{ margin-top: 12px; font-size: 12px; }}
.trace-section summary {{ cursor: pointer; color: var(--text-muted); font-size: 12px;
                          padding: 4px 0; font-weight: 500; }}
.trace-section summary:hover {{ color: var(--text-secondary); }}
.trace-body {{ padding: 8px 0; }}
.bridge-info {{ background: var(--accent-light); border: 1px solid #c7d2fe; border-radius: 8px;
                padding: 8px 12px; margin: 4px 0; }}
.bridge-label {{ font-weight: 500; }}
.bridge-cand {{ color: var(--text-secondary); margin-left: 12px; margin-top: 2px; }}
.bridge-cand code {{ background: #f3f4f6; padding: 1px 5px; border-radius: 4px; font-size: 11px; }}
.bridge-meta {{ color: var(--text-muted); font-size: 11px; }}
.trivial-rej {{ padding: 4px 0; color: var(--text-secondary); }}
.trivial-q {{ font-style: italic; }}
.trivial-why {{ color: var(--text-muted); font-size: 11px; }}

/* Verification */
.verif {{ padding-top: 12px; display: flex; gap: 6px; flex-wrap: wrap;
          border-top: 1px solid var(--border-light); margin-top: 12px; }}
.verif-item {{ font-size: 11px; padding: 3px 10px; border-radius: 12px; font-weight: 500; }}
.verif-item.pass {{ background: var(--green-bg); color: var(--green); }}
.verif-item.fail {{ background: var(--red-bg); color: var(--red); }}

/* Check trace entries */
.check-entry {{ padding: 8px 12px; margin: 4px 0; border-radius: 6px; font-size: 12px; }}
.check-entry.pass {{ background: var(--green-bg); border: 1px solid var(--green-border); }}
.check-entry.fail {{ background: var(--red-bg); border: 1px solid var(--red-border); }}
.check-entry code {{ background: rgba(0,0,0,0.06); padding: 1px 5px; border-radius: 4px; font-size: 11px; }}

@media (max-width: 700px) {{
    .stats-row {{ flex-wrap: wrap; }}
    .stat-card {{ min-width: 80px; }}
    .card-header {{ flex-direction: column; align-items: flex-start; gap: 6px; }}
    .header-right {{ flex-wrap: wrap; }}
}}
</style>
</head>
<body>
<div class="topbar">
    <div class="topbar-inner">
        <div>
            <h1>{_esc(title)}</h1>
            <div class="subtitle">{valid_pct}% valid &middot; <span id="vis-count">{overall['n']} chains</span> &middot; {len(all_patterns)} patterns</div>
        </div>
        <div class="filter-rows">
            <div class="filters">{pattern_btns}</div>
            <div class="filters">{status_btns}</div>
        </div>
    </div>
</div>
<div class="container">
    <div class="stats-row">
        <div class="stat-card"><span class="val">{overall['n']}</span><span class="lbl">Total</span></div>
        <div class="stat-card"><span class="val val-green">{overall['valid']}</span><span class="lbl">Valid</span></div>
        <div class="stat-card"><span class="val">{overall['complete']}</span><span class="lbl">Complete</span></div>
        <div class="stat-card"><span class="val val-red">{overall['n'] - overall['complete']}</span><span class="lbl">Incomplete</span></div>
        <div class="stat-card"><span class="val">{overall['avg_hops']:.1f}</span><span class="lbl">Avg Hops</span></div>
        <div class="stat-card"><span class="val">{overall['avg_time']:.0f}s</span><span class="lbl">Avg Time</span></div>
    </div>
    <div class="table-wrap">
    <table class="pattern-table">
        <tr><th>Pattern</th><th>N</th><th>Valid</th><th>Complete</th>
            <th>Avg Hops</th><th>Avg LLM</th><th>Avg Time</th>
            <th>Det. Rej</th><th>Trivial Rej</th><th>Backref Rej</th></tr>
        {pattern_rows}
    </table>
    </div>
    <div id="chains">{cards}</div>
</div>
<script>
const state = {{ pattern: 'all', status: 'all' }};
function applyFilters() {{
    document.querySelectorAll('.chain-card').forEach(card => {{
        const pm = state.pattern === 'all' || card.dataset.pattern === state.pattern;
        const sm = state.status === 'all' || card.dataset.status === state.status;
        card.classList.toggle('hidden', !(pm && sm));
    }});
    // Update visible count
    const vis = document.querySelectorAll('.chain-card:not(.hidden)').length;
    const tot = document.querySelectorAll('.chain-card').length;
    const el = document.getElementById('vis-count');
    if (el) el.textContent = vis === tot ? tot + ' chains' : vis + ' / ' + tot + ' chains';
}}
document.querySelectorAll('.filter-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
        const group = btn.dataset.group;
        document.querySelectorAll('.filter-btn[data-group="' + group + '"]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state[group] = btn.dataset.filter;
        applyFilters();
    }});
}});
</script>
</body></html>"""

    output.write_text(page, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="View chain builder output as HTML.")
    p.add_argument("input", help="JSONL file from chain_builder")
    p.add_argument("-o", "--output", default=None, help="Output HTML path (default: /tmp/chains_viewer.html)")
    p.add_argument("--open", action="store_true", help="Open in browser")
    p.add_argument("--title", default=None, help="Page title")
    args = p.parse_args()

    input_path = Path(args.input)
    chains = _load(input_path)
    if not chains:
        print(f"No chains found in {input_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else Path("/tmp/chains_viewer.html")
    title = args.title or f"Chains: {input_path.name} ({len(chains)} chains)"

    generate_html(chains, output_path, title)
    print(f"Generated: {output_path} ({len(chains)} chains)")

    if args.open:
        subprocess.Popen(["xdg-open", str(output_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
