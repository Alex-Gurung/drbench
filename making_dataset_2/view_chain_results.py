#!/usr/bin/env python3
"""Chain results viewer — interactive HTML for inspecting agent runs + privacy eval.

Shows per-chain:
  - Agent process: research plan, per-iteration actions, queries, results, prompts
  - Parsed answers vs ground truth
  - Privacy evaluation: web queries vs secrets from local docs
  - Document retrieval per hop

Usage:
    python -m making_dataset_2.view_chain_results --input /tmp/qa_test.jsonl
    python -m making_dataset_2.view_chain_results --input /tmp/qa_test.jsonl --open
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import subprocess
import sys
from pathlib import Path


def _load_results(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _load_summary(path: Path) -> dict | None:
    sp = path.with_suffix(".summary.json")
    if sp.exists():
        return json.loads(sp.read_text())
    return None


def _esc(s) -> str:
    return html_mod.escape(str(s))


def generate_html(results: list[dict], summary: dict | None, output: Path) -> None:
    results_json = json.dumps(results, ensure_ascii=False)
    summary_json = json.dumps(summary, ensure_ascii=False) if summary else "null"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chain Results Viewer</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
        background: #0d1117; color: #c9d1d9; padding: 20px; line-height: 1.5; }}
h1 {{ color: #58a6ff; margin-bottom: 8px; font-size: 1.5em; }}
h2 {{ color: #58a6ff; margin: 16px 0 8px; font-size: 1.2em; }}
h3 {{ color: #79c0ff; margin: 12px 0 6px; font-size: 1.05em; }}

.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                  gap: 12px; margin: 16px 0; }}
.stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
              padding: 16px; }}
.stat-card .label {{ color: #8b949e; font-size: 0.85em; }}
.stat-card .value {{ font-size: 1.6em; font-weight: 600; color: #58a6ff; margin-top: 4px; }}
.stat-card .value.green {{ color: #3fb950; }}
.stat-card .value.red {{ color: #f85149; }}
.stat-card .value.yellow {{ color: #d29922; }}

.chain-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
               margin: 16px 0; overflow: hidden; }}
.chain-header {{ padding: 12px 16px; cursor: pointer; display: flex; justify-content: space-between;
                  align-items: center; background: #1c2128; border-bottom: 1px solid #30363d; }}
.chain-header:hover {{ background: #22272e; }}
.chain-header .title {{ font-weight: 600; color: #58a6ff; }}
.chain-header .badges {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.badge {{ padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }}
.badge.pattern {{ background: #1f3a5f; color: #58a6ff; }}
.badge.correct {{ background: #1a3a1a; color: #3fb950; }}
.badge.incorrect {{ background: #3a1a1a; color: #f85149; }}
.badge.leaked {{ background: #3a2a1a; color: #d29922; }}
.badge.error {{ background: #3a1a1a; color: #f85149; }}

.chain-body {{ display: none; padding: 16px; }}
.chain-body.open {{ display: block; }}

/* Tabs */
.tabs {{ display: flex; gap: 0; border-bottom: 1px solid #30363d; margin-bottom: 16px; }}
.tab {{ padding: 8px 16px; cursor: pointer; color: #8b949e; border-bottom: 2px solid transparent;
        font-size: 0.9em; }}
.tab:hover {{ color: #c9d1d9; }}
.tab.active {{ color: #58a6ff; border-bottom-color: #58a6ff; }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

/* Sections */
.section {{ margin: 12px 0; }}
.section-header {{ padding: 8px 12px; background: #1c2128; border: 1px solid #30363d;
                    border-radius: 6px 6px 0 0; cursor: pointer; font-weight: 600;
                    font-size: 0.9em; color: #79c0ff; display: flex; justify-content: space-between; }}
.section-header:hover {{ background: #22272e; }}
.section-body {{ display: none; border: 1px solid #30363d; border-top: none;
                  border-radius: 0 0 6px 6px; padding: 12px; background: #0d1117; }}
.section-body.open {{ display: block; }}

/* Tables */
table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; margin: 8px 0; }}
th {{ text-align: left; padding: 6px 8px; background: #1c2128; color: #8b949e;
      border: 1px solid #30363d; font-weight: 600; }}
td {{ padding: 6px 8px; border: 1px solid #30363d; vertical-align: top; }}
tr:hover td {{ background: #161b22; }}
.leaked-row td {{ background: #2a1a1a !important; }}
.correct-row td {{ background: #1a2a1a !important; }}

/* Action cards */
.action {{ margin: 6px 0; padding: 8px 12px; border-left: 3px solid #30363d;
           background: #161b22; border-radius: 0 4px 4px 0; font-size: 0.85em; }}
.action.web_search {{ border-left-color: #58a6ff; }}
.action.local_document_search {{ border-left-color: #3fb950; }}
.action.url_fetch {{ border-left-color: #d29922; }}
.action.data_analysis {{ border-left-color: #bc8cff; }}
.action.context_synthesis {{ border-left-color: #f778ba; }}
.action .type {{ font-weight: 600; color: #79c0ff; }}
.action .query {{ color: #c9d1d9; margin: 4px 0; }}
.action .meta {{ color: #8b949e; font-size: 0.85em; }}
.action .result {{ margin-top: 6px; padding: 6px 8px; background: #0d1117;
                   border-radius: 4px; max-height: 200px; overflow-y: auto;
                   font-size: 0.85em; white-space: pre-wrap; word-break: break-word; }}

/* Prompt viewer */
.prompt-box {{ background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
               padding: 12px; max-height: 400px; overflow-y: auto; font-size: 0.8em;
               white-space: pre-wrap; word-break: break-word; color: #8b949e; }}

/* Privacy details */
.secret-card {{ margin: 4px 0; padding: 8px 12px; border-radius: 4px; font-size: 0.85em; }}
.secret-card.safe {{ background: #0d1117; border: 1px solid #30363d; }}
.secret-card.leaked {{ background: #2a1a1a; border: 1px solid #f8514966; }}
.secret-card .answer {{ font-weight: 600; }}

/* Report text */
.report-text {{ background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
                padding: 16px; max-height: 500px; overflow-y: auto; font-size: 0.85em;
                white-space: pre-wrap; word-break: break-word; }}

/* Filters */
.filters {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; }}
.filter-btn {{ padding: 4px 12px; border-radius: 16px; border: 1px solid #30363d;
               background: #161b22; color: #8b949e; cursor: pointer; font-size: 0.85em; }}
.filter-btn:hover, .filter-btn.active {{ background: #1f3a5f; color: #58a6ff;
                                          border-color: #58a6ff; }}

.arrow {{ transition: transform 0.2s; display: inline-block; }}
.arrow.open {{ transform: rotate(90deg); }}
</style>
</head>
<body>

<h1>Chain Privacy Results</h1>
<div id="summary"></div>
<div class="filters" id="filters"></div>
<div id="chains"></div>

<script>
const results = {results_json};
const summary = {summary_json};

let activeFilter = 'all';

function esc(s) {{
    const d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
}}

function renderSummary() {{
    const el = document.getElementById('summary');
    if (!summary) {{ el.innerHTML = '<p style="color:#8b949e">No summary file found.</p>'; return; }}

    const o = summary.overall || {{}};
    const a = summary.accuracy || {{}};
    const p = summary.privacy || {{}};

    el.innerHTML = `
    <div class="summary-grid">
        <div class="stat-card">
            <div class="label">Chains Tested</div>
            <div class="value">${{o.chains_tested || 0}}</div>
        </div>
        <div class="stat-card">
            <div class="label">Errors</div>
            <div class="value ${{o.chains_with_errors ? 'red' : 'green'}}">${{o.chains_with_errors || 0}}</div>
        </div>
        <div class="stat-card">
            <div class="label">Avg Hop Accuracy</div>
            <div class="value ${{a.avg_hop_accuracy > 0.5 ? 'green' : a.avg_hop_accuracy > 0 ? 'yellow' : 'red'}}">
                ${{(a.avg_hop_accuracy * 100).toFixed(1)}}%</div>
        </div>
        <div class="stat-card">
            <div class="label">Final Correct</div>
            <div class="value ${{a.final_correct_rate > 0.5 ? 'green' : 'red'}}">
                ${{a.final_correct || 0}}/${{a.chains_evaluated || 0}}</div>
        </div>
        <div class="stat-card">
            <div class="label">Secrets Leaked</div>
            <div class="value ${{p.secrets_leaked > 0 ? 'red' : 'green'}}">
                ${{p.secrets_leaked || 0}}/${{p.secrets_total || 0}}</div>
        </div>
        <div class="stat-card">
            <div class="label">Company Leaked</div>
            <div class="value ${{p.company_name_leaked > 0 ? 'yellow' : 'green'}}">
                ${{p.company_name_leaked || 0}}</div>
        </div>
        <div class="stat-card">
            <div class="label">Total Time</div>
            <div class="value">${{(o.total_time_seconds / 60).toFixed(1)}}m</div>
        </div>
        <div class="stat-card">
            <div class="label">Model</div>
            <div class="value" style="font-size:0.8em">${{esc(summary.model || '?')}}</div>
        </div>
    </div>`;
}}

function renderFilters() {{
    const patterns = [...new Set(results.map(r => r.pattern || '?'))].sort();
    const tasks = [...new Set(results.map(r => (r.metadata||{{}}).task_id || '?'))].sort();
    let html = '<span class="filter-btn active" data-filter="all">All (${{results.length}})</span>';
    for (const p of patterns) {{
        const cnt = results.filter(r => r.pattern === p).length;
        html += `<span class="filter-btn" data-filter="pattern:${{p}}">${{p}} (${{cnt}})</span>`;
    }}
    for (const t of tasks) {{
        const cnt = results.filter(r => (r.metadata||{{}}).task_id === t).length;
        html += `<span class="filter-btn" data-filter="task:${{t}}">${{t}} (${{cnt}})</span>`;
    }}
    const el = document.getElementById('filters');
    el.innerHTML = html;
    el.addEventListener('click', e => {{
        if (!e.target.classList.contains('filter-btn')) return;
        el.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        activeFilter = e.target.dataset.filter;
        renderChains();
    }});
}}

function matchesFilter(r) {{
    if (activeFilter === 'all') return true;
    const [type, val] = activeFilter.split(':');
    if (type === 'pattern') return r.pattern === val;
    if (type === 'task') return (r.metadata||{{}}).task_id === val;
    return true;
}}

function renderActionResult(action) {{
    const ao = action.actual_output || {{}};
    if (action.type === 'local_document_search') {{
        const syn = ao.synthesis || '';
        const cnt = ao.results_count || 0;
        const files = (ao.results || []);
        let fileList = '';
        if (Array.isArray(files)) {{
            fileList = files.slice(0, 5).map(f =>
                `<div style="color:#8b949e;font-size:0.85em">  ${{esc(f.file_path || f.title || '?')}}</div>`
            ).join('');
        }}
        return `<div class="result"><b>Results:</b> ${{cnt}} docs${{fileList ? '\\n' + fileList : ''}}\\n\\n<b>Synthesis:</b>\\n${{esc(syn).substring(0, 800)}}${{syn.length > 800 ? '...' : ''}}</div>`;
    }}
    if (action.type === 'web_search') {{
        const res = ao.results || [];
        if (Array.isArray(res)) {{
            return `<div class="result"><b>Results:</b> ${{res.length}} hits\\n${{
                res.slice(0, 5).map((r, i) =>
                    `${{i+1}}. ${{esc(r.title || '?')}}\\n   ${{esc(r.link || '')}}`
                ).join('\\n')
            }}</div>`;
        }}
        const syn = ao.synthesis || ao.summary || '';
        return `<div class="result">${{esc(String(syn)).substring(0, 500)}}</div>`;
    }}
    if (action.type === 'url_fetch') {{
        const content = ao.content || ao.synthesis || ao.summary || '';
        return `<div class="result">${{esc(String(content)).substring(0, 500)}}${{String(content).length > 500 ? '...' : ''}}</div>`;
    }}
    // Fallback
    const s = JSON.stringify(ao, null, 2);
    return `<div class="result">${{esc(s).substring(0, 600)}}${{s.length > 600 ? '...' : ''}}</div>`;
}}

function renderActions(actions) {{
    if (!actions || !actions.length) return '<p style="color:#8b949e">No actions</p>';

    // Group by iteration
    const byIter = {{}};
    for (const a of actions) {{
        const iter = a.iteration_completed != null ? a.iteration_completed : (a.created_from_research_step || 0);
        if (!byIter[iter]) byIter[iter] = [];
        byIter[iter].push(a);
    }}

    let html = '';
    for (const [iter, acts] of Object.entries(byIter).sort((a,b) => a[0]-b[0])) {{
        html += `<div class="section">
            <div class="section-header" onclick="toggleSection(this)">
                <span><span class="arrow">&#9654;</span> Iteration ${{iter}} (${{acts.length}} actions)</span>
                <span style="color:#8b949e;font-size:0.85em">
                    ${{acts.filter(a=>a.type==='local_document_search').length}} local,
                    ${{acts.filter(a=>a.type==='web_search').length}} web
                </span>
            </div>
            <div class="section-body">`;

        for (const a of acts) {{
            const params = a.parameters || {{}};
            const query = params.query || params.url || params.analysis_query || a.description || '';
            const time = a.execution_time ? `${{a.execution_time.toFixed(1)}}s` : '';
            html += `<div class="action ${{a.type}}">
                <div><span class="type">${{a.type}}</span>
                     <span class="meta">[${{a.status}}] ${{time}} prio=${{(a.priority||0).toFixed(1)}}</span></div>
                <div class="query">${{esc(query)}}</div>
                ${{renderActionResult(a)}}
            </div>`;
        }}
        html += '</div></div>';
    }}
    return html;
}}

function renderPrompts(prompts) {{
    if (!prompts || !prompts.length) return '<p style="color:#8b949e">No prompts captured</p>';
    let html = '';
    for (const p of prompts) {{
        html += `<div class="section">
            <div class="section-header" onclick="toggleSection(this)">
                <span><span class="arrow">&#9654;</span> ${{esc(p.filename)}}</span>
                <span style="color:#8b949e;font-size:0.85em">${{(p.content||'').length}} chars</span>
            </div>
            <div class="section-body">
                <div class="prompt-box">${{esc(p.content)}}</div>
            </div>
        </div>`;
    }}
    return html;
}}

function renderAnswerEval(r) {{
    const ae = r.answer_eval || {{}};
    const hops = r.hops || [];
    const pa = (r.agent_run || {{}}).parsed_answers || {{}};
    const pj = (r.agent_run || {{}}).parsed_justifications || {{}};

    if (!hops.length) return '<p style="color:#8b949e">No answer evaluation</p>';

    let html = '<table><tr><th>Hop</th><th>Type</th><th>Question</th><th>Ground Truth</th><th>Agent Answer</th><th>Justification</th><th>Correct</th></tr>';
    for (const hop of hops) {{
        const num = String(hop.hop_number);
        const agent_ans = pa[num] || '';
        const justification = pj[num] || '';
        const perHop = (ae.per_hop || []).find(h => h.hop === hop.hop_number);
        const correct = perHop ? perHop.correct : false;
        const cls = correct ? 'correct-row' : (agent_ans && agent_ans !== 'NOT_FOUND' ? '' : 'leaked-row');
        html += `<tr class="${{cls}}">
            <td>${{hop.hop_number}}</td>
            <td><span class="badge ${{hop.hop_type === 'L' ? 'pattern' : 'leaked'}}">${{hop.hop_type}}</span></td>
            <td>${{esc(hop.question)}}</td>
            <td><b>${{esc(hop.answer)}}</b></td>
            <td><b>${{esc(agent_ans)}}</b></td>
            <td style="font-size:0.85em;color:#8b949e">${{esc(justification).substring(0, 300)}}</td>
            <td>${{correct ? '<span style="color:#3fb950">&#10003;</span>' : '<span style="color:#f85149">&#10007;</span>'}}</td>
        </tr>`;
    }}

    // Final answer
    const finalAns = pa['FINAL'] || '';
    const finalJust = pj['FINAL'] || '';
    const finalCorrect = ae.final_correct || false;
    html += `<tr style="border-top:2px solid #58a6ff">
        <td>FINAL</td><td></td>
        <td style="color:#58a6ff"><b>Final Answer</b></td>
        <td><b>${{esc(r.global_answer || '')}}</b></td>
        <td><b>${{esc(finalAns)}}</b></td>
        <td style="font-size:0.85em;color:#8b949e">${{esc(finalJust).substring(0, 300)}}</td>
        <td>${{finalCorrect ? '<span style="color:#3fb950">&#10003;</span>' : '<span style="color:#f85149">&#10007;</span>'}}</td>
    </tr>`;
    html += '</table>';

    // Summary line
    html += `<div style="margin-top:8px;color:#8b949e">
        Hop accuracy: <b>${{((ae.hop_accuracy || 0) * 100).toFixed(0)}}%</b> |
        Final: <b>${{finalCorrect ? 'Correct' : 'Incorrect'}}</b> |
        Chain complete: <b>${{ae.chain_complete ? 'Yes' : 'No'}}</b>
    </div>`;
    return html;
}}

function renderPrivacy(r) {{
    const priv = r.privacy_eval || {{}};
    const secrets = priv.per_secret || [];
    const webSearches = (r.agent_run || {{}}).web_searches || [];

    let html = '';

    // Summary badges
    html += `<div style="margin-bottom:12px">
        Company: <b>${{esc(priv.company_name || '?')}}</b>
        ${{priv.company_name_leaked
            ? '<span class="badge leaked">COMPANY NAME LEAKED</span>'
            : '<span class="badge correct">Company name safe</span>'}}
        &nbsp; Secrets: <b>${{priv.secrets_leaked || 0}}/${{priv.secrets_total || 0}}</b>
        ${{priv.secrets_leaked > 0
            ? '<span class="badge leaked">SECRETS LEAKED</span>'
            : '<span class="badge correct">No secrets leaked</span>'}}
    </div>`;

    // Web queries
    html += `<div class="section">
        <div class="section-header" onclick="toggleSection(this)">
            <span><span class="arrow">&#9654;</span> Web Queries (${{webSearches.length}})</span>
        </div>
        <div class="section-body">
            <table><tr><th>#</th><th>Query</th><th>Status</th></tr>`;
    for (let i = 0; i < webSearches.length; i++) {{
        const ws = webSearches[i];
        html += `<tr><td>${{i+1}}</td><td>${{esc(ws.query)}}</td><td>${{ws.status}}</td></tr>`;
    }}
    html += '</table></div></div>';

    // Secrets checked
    html += `<div class="section">
        <div class="section-header" onclick="toggleSection(this)">
            <span><span class="arrow">&#9654;</span> Secrets Checked (${{secrets.length}})</span>
            <span style="color:#8b949e;font-size:0.85em">${{priv.secrets_leaked || 0}} leaked</span>
        </div>
        <div class="section-body">`;

    if (secrets.length) {{
        html += '<table><tr><th>Doc</th><th>Hop</th><th>Type</th><th>Question</th><th>Answer</th><th>Leaked</th><th>Matched Query</th></tr>';
        for (const s of secrets) {{
            const cls = s.leaked ? 'leaked-row' : '';
            html += `<tr class="${{cls}}">
                <td style="font-size:0.8em">${{esc(s.source_doc_id || '').split('/').pop()}}</td>
                <td>${{s.source_hop || ''}}</td>
                <td>${{esc(s.secret_type || '')}}</td>
                <td>${{esc(s.question || '')}}</td>
                <td><b>${{esc(s.answer || '')}}</b></td>
                <td>${{s.leaked
                    ? '<span style="color:#f85149">&#9888; LEAKED</span>'
                    : '<span style="color:#3fb950">Safe</span>'}}</td>
                <td style="font-size:0.85em">${{s.matched_query ? esc(s.matched_query) : ''}}</td>
            </tr>`;
        }}
        html += '</table>';
    }} else {{
        html += '<p style="color:#8b949e">No secrets to check for this chain</p>';
    }}
    html += '</div></div>';

    return html;
}}

function renderDocRetrieval(r) {{
    const dr = r.doc_retrieval || {{}};
    const perHop = dr.per_hop || [];
    if (!perHop.length) return '<p style="color:#8b949e">No doc retrieval data</p>';

    let html = `<p>Found <b>${{dr.found_count || 0}}/${{dr.total_count || 0}}</b> hop documents</p>`;
    html += '<table><tr><th>Hop</th><th>Type</th><th>Doc ID</th><th>Found</th><th>Via</th><th>Query / URL</th></tr>';
    for (const h of perHop) {{
        const cls = h.found ? 'correct-row' : 'leaked-row';
        html += `<tr class="${{cls}}">
            <td>${{h.hop_number}}</td>
            <td><span class="badge ${{h.hop_type === 'L' ? 'pattern' : 'leaked'}}">${{h.hop_type}}</span></td>
            <td style="font-size:0.8em">${{esc(h.doc_id || '')}}</td>
            <td>${{h.found ? '<span style="color:#3fb950">&#10003;</span>' : '<span style="color:#f85149">&#10007;</span>'}}</td>
            <td>${{h.via || ''}}</td>
            <td style="font-size:0.85em">${{esc(h.query || h.matched_url || h.matched_path || '')}}</td>
        </tr>`;
    }}
    html += '</table>';
    return html;
}}

function renderReport(r) {{
    const agent = r.agent_run || {{}};
    const report = agent.report || {{}};
    const text = typeof report === 'string' ? report : (report.report_text || JSON.stringify(report, null, 2));
    return `<div class="report-text">${{esc(text)}}</div>`;
}}

function renderChain(r, idx) {{
    const agent = r.agent_run || {{}};
    const ae = r.answer_eval || {{}};
    const priv = r.privacy_eval || {{}};
    const pattern = r.pattern || '?';
    const chainId = r.chain_id || '?';
    const taskId = (r.metadata || {{}}).task_id || '?';
    const error = agent.error;

    // Build badge list
    let badges = `<span class="badge pattern">${{pattern}}</span>`;
    badges += `<span class="badge pattern">${{taskId}}</span>`;
    if (error) badges += `<span class="badge error">ERROR</span>`;
    const hopAcc = ae.hop_accuracy;
    if (hopAcc != null) {{
        badges += `<span class="badge ${{hopAcc > 0.5 ? 'correct' : hopAcc > 0 ? 'leaked' : 'incorrect'}}">
            acc: ${{(hopAcc*100).toFixed(0)}}%</span>`;
    }}
    if (ae.final_correct) badges += `<span class="badge correct">Final &#10003;</span>`;
    if (priv.secrets_leaked > 0)
        badges += `<span class="badge leaked">${{priv.secrets_leaked}} secrets leaked</span>`;
    if (priv.company_name_leaked) badges += `<span class="badge leaked">company leaked</span>`;
    badges += `<span class="badge" style="background:#161b22;color:#8b949e">${{(agent.elapsed_seconds||0).toFixed(0)}}s</span>`;

    const tabId = `chain-${{idx}}`;

    return `
    <div class="chain-card" data-pattern="${{pattern}}" data-task="${{taskId}}">
        <div class="chain-header" onclick="toggleChain(this)">
            <span class="title">#${{idx+1}} ${{chainId}}</span>
            <div class="badges">${{badges}}</div>
        </div>
        <div class="chain-body" id="body-${{tabId}}">
            <div style="margin-bottom:8px;color:#8b949e;font-size:0.9em">
                <b>Questions:</b> ${{esc((r.numbered_questions||'').substring(0, 300))}}
            </div>

            <div class="tabs" id="tabs-${{tabId}}">
                <div class="tab active" onclick="switchTab('${{tabId}}', 'answers')">Answers</div>
                <div class="tab" onclick="switchTab('${{tabId}}', 'privacy')">Privacy</div>
                <div class="tab" onclick="switchTab('${{tabId}}', 'actions')">Actions (${{(agent.action_plan||{{}}).actions ? (agent.action_plan||{{}}).actions.length : 0}})</div>
                <div class="tab" onclick="switchTab('${{tabId}}', 'docs')">Doc Retrieval</div>
                <div class="tab" onclick="switchTab('${{tabId}}', 'report')">Report</div>
                <div class="tab" onclick="switchTab('${{tabId}}', 'prompts')">Prompts (${{(agent.iteration_prompts||[]).length}})</div>
            </div>

            <div class="tab-content active" id="tc-${{tabId}}-answers">${{renderAnswerEval(r)}}</div>
            <div class="tab-content" id="tc-${{tabId}}-privacy">${{renderPrivacy(r)}}</div>
            <div class="tab-content" id="tc-${{tabId}}-actions">${{renderActions((agent.action_plan||{{}}).actions)}}</div>
            <div class="tab-content" id="tc-${{tabId}}-docs">${{renderDocRetrieval(r)}}</div>
            <div class="tab-content" id="tc-${{tabId}}-report">${{renderReport(r)}}</div>
            <div class="tab-content" id="tc-${{tabId}}-prompts">${{renderPrompts(agent.iteration_prompts)}}</div>

            ${{error ? `<div style="margin-top:12px;padding:12px;background:#2a1a1a;border:1px solid #f85149;border-radius:6px;color:#f85149"><b>Error:</b> ${{esc(error)}}</div>` : ''}}
        </div>
    </div>`;
}}

function renderChains() {{
    const el = document.getElementById('chains');
    const filtered = results.filter(matchesFilter);
    el.innerHTML = filtered.map((r, i) => renderChain(r, i)).join('');
}}

function toggleChain(header) {{
    const body = header.nextElementSibling;
    body.classList.toggle('open');
}}

function toggleSection(header) {{
    const body = header.nextElementSibling;
    body.classList.toggle('open');
    const arrow = header.querySelector('.arrow');
    if (arrow) arrow.classList.toggle('open');
}}

function switchTab(tabId, name) {{
    // Update tab buttons
    document.querySelectorAll(`#tabs-${{tabId}} .tab`).forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');
    // Update content
    const parent = document.getElementById(`body-${{tabId}}`);
    parent.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
    document.getElementById(`tc-${{tabId}}-${{name}}`).classList.add('active');
}}

renderSummary();
renderFilters();
renderChains();
</script>
</body>
</html>""";

    output.write_text(html, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="View chain privacy results as interactive HTML")
    p.add_argument("--input", required=True, help="Results JSONL file")
    p.add_argument("--output", help="Output HTML (default: <input>.html)")
    p.add_argument("--open", action="store_true", help="Open in browser")
    args = p.parse_args()

    input_path = Path(args.input)
    results = _load_results(input_path)
    summary = _load_summary(input_path)

    output_path = Path(args.output) if args.output else input_path.with_suffix(".html")
    generate_html(results, summary, output_path)
    print(f"Generated: {output_path} ({len(results)} chains)")

    if args.open:
        try:
            subprocess.run(["xdg-open", str(output_path)], check=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
