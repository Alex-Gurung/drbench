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


def _safe_json_for_html(obj) -> str:
    """Serialize to JSON safe for embedding in <script> tags."""
    s = json.dumps(obj, ensure_ascii=False)
    return s.replace("</", "<\\/").replace("<!--", "<\\!--")


def generate_html(results: list[dict], summary: dict | None, output: Path) -> None:
    results_json = _safe_json_for_html(results)
    summary_json = _safe_json_for_html(summary) if summary else "null"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chain Results Viewer</title>
<style>
:root {{
  --bg: #f8f9fb; --card: #ffffff; --border: #e5e7eb; --border-light: #f0f1f3;
  --primary: #6366f1; --primary-light: #eef2ff;
  --local: #6366f1; --web: #8b5cf6;
  --success: #059669; --success-bg: #ecfdf5; --success-border: #a7f3d0;
  --warning: #d97706; --warning-bg: #fffbeb; --warning-border: #fde68a;
  --danger: #dc2626; --danger-bg: #fef2f2; --danger-border: #fecaca;
  --text: #1a1a2e; --text-secondary: #6b7280; --text-muted: #9ca3af;
  --radius: 10px;
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
  --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        background: var(--bg); color: var(--text); line-height: 1.6; font-size: 14px; }}

.container {{ max-width: 1100px; margin: 0 auto; padding: 20px 16px; }}

/* Header */
header {{ background: var(--card); padding: 16px 24px; border-bottom: 1px solid var(--border);
          position: sticky; top: 0; z-index: 100; }}
header h1 {{ font-size: 15px; font-weight: 600; letter-spacing: -0.01em; color: var(--text); }}

/* Summary */
.summary-grid {{ display: flex; gap: 12px; margin: 16px 0; flex-wrap: wrap; }}
.stat-card {{ flex: 1; min-width: 100px; background: var(--card); border: 1px solid var(--border);
              border-radius: var(--radius); padding: 14px; text-align: center; box-shadow: var(--shadow-sm); }}
.stat-card .label {{ color: var(--text-muted); font-size: 10px; text-transform: uppercase;
                      letter-spacing: 0.05em; margin-top: 2px; }}
.stat-card .value {{ font-size: 24px; font-weight: 700; letter-spacing: -0.02em; color: var(--text); }}
.stat-card .value.green {{ color: var(--success); }}
.stat-card .value.red {{ color: var(--danger); }}
.stat-card .value.yellow {{ color: var(--warning); }}

/* Chain cards */
.chain-card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
               margin: 10px 0; overflow: hidden; box-shadow: var(--shadow-sm);
               transition: box-shadow 0.15s; }}
.chain-card:hover {{ box-shadow: var(--shadow); }}
.chain-header {{ padding: 10px 16px; cursor: pointer; display: flex; justify-content: space-between;
                  align-items: center; border-bottom: 1px solid var(--border-light);
                  transition: background 0.12s; }}
.chain-header:hover {{ background: #fafbfc; }}
.chain-header .title {{ font-weight: 600; font-size: 13px; color: var(--text-secondary); }}
.chain-header .badges {{ display: flex; gap: 5px; flex-wrap: wrap; }}
.badge {{ display: inline-block; padding: 2px 9px; border-radius: 12px; font-size: 11px;
          font-weight: 600; letter-spacing: 0.01em; }}
.badge.pattern {{ background: #f3f4f6; color: var(--text); }}
.badge.correct {{ background: var(--success-bg); color: var(--success); }}
.badge.incorrect {{ background: var(--danger-bg); color: var(--danger); }}
.badge.leaked {{ background: var(--warning-bg); color: var(--warning); }}
.badge.error {{ background: var(--danger-bg); color: var(--danger); }}

.chain-body {{ display: none; padding: 16px; }}
.chain-body.open {{ display: block; }}

/* Tabs */
.tabs {{ display: flex; gap: 0; border-bottom: 1px solid var(--border); margin-bottom: 16px; }}
.tab {{ padding: 8px 16px; cursor: pointer; color: var(--text-muted); border-bottom: 2px solid transparent;
        font-size: 12px; font-weight: 500; transition: all 0.15s; }}
.tab:hover {{ color: var(--text); background: var(--primary-light); }}
.tab.active {{ color: var(--primary); border-bottom-color: var(--primary); }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

/* Sections */
.section {{ margin: 10px 0; background: var(--card); border: 1px solid var(--border);
            border-radius: 8px; overflow: hidden; }}
.section-header {{ padding: 9px 14px; background: var(--bg); border-bottom: 1px solid var(--border-light);
                    cursor: pointer; font-weight: 500; font-size: 12px; color: var(--text-secondary);
                    display: flex; justify-content: space-between; align-items: center;
                    transition: background 0.12s; }}
.section-header:hover {{ background: #f0f1f3; }}
.section-body {{ display: none; padding: 12px; }}
.section-body.open {{ display: block; }}

/* Tables */
table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 8px 0; }}
th {{ text-align: left; padding: 8px 10px; background: var(--bg); color: var(--text-muted);
      border: 1px solid var(--border-light); font-weight: 500; font-size: 11px;
      text-transform: uppercase; letter-spacing: 0.04em; }}
td {{ padding: 8px 10px; border: 1px solid var(--border-light); vertical-align: top; }}
tr:hover td {{ background: #fafbfc; }}
.leaked-row td {{ background: var(--danger-bg) !important; }}
.correct-row td {{ background: var(--success-bg) !important; }}

/* Action cards */
.action {{ margin: 6px 0; padding: 10px 14px; border-left: 3px solid var(--border-light);
           background: var(--bg); border-radius: 0 6px 6px 0; font-size: 13px; }}
.action.web_search {{ border-left-color: var(--web); }}
.action.local_document_search {{ border-left-color: var(--local); }}
.action.url_fetch {{ border-left-color: var(--warning); }}
.action.data_analysis {{ border-left-color: #8b5cf6; }}
.action.context_synthesis {{ border-left-color: #ec4899; }}
.action .type {{ font-weight: 600; color: var(--primary); font-size: 12px; }}
.action .query {{ color: var(--text); margin: 4px 0; }}
.action .meta {{ color: var(--text-muted); font-size: 11px; }}
.action .result {{ margin-top: 6px; padding: 8px 10px; background: var(--card);
                   border: 1px solid var(--border-light); border-radius: 6px; max-height: 200px;
                   overflow-y: auto; font-size: 12px; white-space: pre-wrap;
                   word-break: break-word; font-family: 'SF Mono', 'Fira Code', ui-monospace, monospace; }}

/* Prompt viewer */
.prompt-box {{ background: var(--bg); border: 1px solid var(--border-light); border-radius: 6px;
               padding: 12px; max-height: 400px; overflow-y: auto; font-size: 12px;
               white-space: pre-wrap; word-break: break-word; color: var(--text-muted);
               font-family: 'SF Mono', 'Fira Code', ui-monospace, monospace; }}

/* Privacy details */
.secret-card {{ margin: 4px 0; padding: 8px 12px; border-radius: 6px; font-size: 13px; }}
.secret-card.safe {{ background: var(--bg); border: 1px solid var(--border-light); }}
.secret-card.leaked {{ background: var(--danger-bg); border: 1px solid var(--danger-border); }}
.secret-card .answer {{ font-weight: 600; }}

/* Report text */
.report-text {{ background: var(--bg); border: 1px solid var(--border-light); border-radius: 8px;
                padding: 16px; max-height: 500px; overflow-y: auto; font-size: 13px;
                white-space: pre-wrap; word-break: break-word; line-height: 1.7; }}

/* Filters */
.filters {{ display: flex; gap: 6px; flex-wrap: wrap; padding: 10px 24px;
            background: var(--card); border-bottom: 1px solid var(--border); }}
.filter-btn {{ padding: 5px 14px; border-radius: 20px; border: 1px solid var(--border);
               background: var(--card); color: var(--text-secondary); cursor: pointer;
               font-size: 12px; font-weight: 500; transition: all 0.15s; }}
.filter-btn:hover {{ background: var(--bg); border-color: #d1d5db; }}
.filter-btn.active {{ background: var(--primary); color: #fff; border-color: var(--primary); }}

.arrow {{ transition: transform 0.2s; display: inline-block; font-size: 10px; }}
.arrow.open {{ transform: rotate(90deg); }}
</style>
</head>
<body>

<header>
  <h1>Chain Privacy Results</h1>
</header>
<div style="background:var(--card);border-bottom:1px solid var(--border);padding:0 24px 10px;">
  <div class="filters" id="filters-pattern"></div>
  <div class="filters" id="filters-task"></div>
  <div class="filters" id="filters-status"></div>
</div>
<div class="container">
<div id="summary"></div>
<div id="chains"></div>
</div>

<script>
const results = {results_json};
const summary = {summary_json};

const filterState = {{ pattern: 'all', task: 'all', status: 'all' }};

function esc(s) {{
    const d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
}}

function renderSummary() {{
    const el = document.getElementById('summary');
    if (!summary) {{ el.innerHTML = '<p style="color:var(--text-muted)">No summary file found.</p>'; return; }}

    const o = summary.overall || {{}};
    const a = summary.accuracy || {{}};
    const p = summary.privacy || {{}};

    // Adversary eval aggregates
    let advTotalLeaked = 0, advTotalQ = 0;
    for (const r of results) {{
        const s = (r.adversary_eval||{{}}).summary || {{}};
        advTotalLeaked += s.total_leaked || 0;
        advTotalQ += s.total_questions || 0;
    }}

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
            <div class="label">Adv. Leaked</div>
            <div class="value ${{advTotalLeaked > 0 ? 'red' : 'green'}}">${{advTotalLeaked}}/${{advTotalQ}}</div>
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

function setupFilterRow(containerId, group, items) {{
    const el = document.getElementById(containerId);
    let html = `<span class="filter-btn active" data-group="${{group}}" data-val="all">All</span>`;
    for (const [label, val, cnt] of items) {{
        html += `<span class="filter-btn" data-group="${{group}}" data-val="${{val}}">${{label}} (${{cnt}})</span>`;
    }}
    el.innerHTML = html;
    el.addEventListener('click', e => {{
        if (!e.target.classList.contains('filter-btn')) return;
        el.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        filterState[group] = e.target.dataset.val;
        renderChains();
    }});
}}

function renderFilters() {{
    const patterns = [...new Set(results.map(r => r.pattern || '?'))].sort();
    const tasks = [...new Set(results.map(r => (r.metadata||{{}}).task_id || '?'))].sort();

    setupFilterRow('filters-pattern', 'pattern',
        patterns.map(p => [p, p, results.filter(r => r.pattern === p).length]));
    setupFilterRow('filters-task', 'task',
        tasks.map(t => [t, t, results.filter(r => (r.metadata||{{}}).task_id === t).length]));

    // Status filters: correct / incorrect / error / leaked
    const nCorrect = results.filter(r => (r.answer_eval||{{}}).final_correct).length;
    const nError = results.filter(r => (r.agent_run||{{}}).error).length;
    const nWrong = results.length - nCorrect - nError;
    const nLeaked = results.filter(r => (r.privacy_eval||{{}}).secrets_leaked > 0).length;
    setupFilterRow('filters-status', 'status', [
        ['Correct', 'correct', nCorrect],
        ['Wrong', 'wrong', nWrong],
        ['Error', 'error', nError],
        ['Leaked', 'leaked', nLeaked],
    ]);
}}

function matchesFilter(r) {{
    const {{ pattern, task, status }} = filterState;
    if (pattern !== 'all' && r.pattern !== pattern) return false;
    if (task !== 'all' && (r.metadata||{{}}).task_id !== task) return false;
    if (status === 'correct' && !(r.answer_eval||{{}}).final_correct) return false;
    if (status === 'wrong' && ((r.answer_eval||{{}}).final_correct || (r.agent_run||{{}}).error)) return false;
    if (status === 'error' && !(r.agent_run||{{}}).error) return false;
    if (status === 'leaked' && !((r.privacy_eval||{{}}).secrets_leaked > 0)) return false;
    return true;
}}

function renderActionResult(action) {{
    const ao = action.actual_output || {{}};
    if (action.type === 'local_document_search') {{
        const syn = ao.synthesis || ao.summary || '';
        const cnt = ao.results_count || 0;
        const folders = ao.folders_searched || [];
        const results = ao.results || [];
        let detailHtml = '';
        if (Array.isArray(results) && results.length && results[0].docid != null) {{
            // Vector store results (BrowseComp-style: docid, score, url, text)
            detailHtml = results.slice(0, 5).map((r, i) => {{
                const title = (r.text || '').split('\\n').find(l => l.startsWith('title:'));
                const label = title ? title.replace('title:', '').trim() : r.url || `doc#${{r.docid}}`;
                return `<div style="color:#6c757d;font-size:0.85em">  ${{i+1}}. ${{esc(label)}} (score=${{(r.score||0).toFixed(2)}})</div>`;
            }}).join('');
        }} else if (Array.isArray(folders) && folders.length) {{
            // Standard local search: show searched folders and file types
            const types = (ao.file_types_found || []).join(', ');
            detailHtml = `<div style="color:#6c757d;font-size:0.85em">  ${{ao.files_searched || 0}} files searched (${{types || '?'}})</div>`
                + folders.slice(0, 3).map(f => `<div style="color:#6c757d;font-size:0.85em">  ${{esc(f.split('/').slice(-2).join('/'))}}</div>`).join('')
                + (folders.length > 3 ? `<div style="color:#6c757d;font-size:0.85em">  ...+${{folders.length - 3}} more</div>` : '');
        }}
        return `<div class="result"><b>Results:</b> ${{cnt}} docs${{detailHtml ? '\\n' + detailHtml : ''}}\\n\\n<b>Synthesis:</b>\\n${{esc(syn).substring(0, 800)}}${{syn.length > 800 ? '...' : ''}}</div>`;
    }}
    if (action.type === 'web_search') {{
        const res = ao.results || [];
        if (Array.isArray(res)) {{
            // BrowseComp results have: docid, score, url, text
            // Serper results have: title, link, snippet
            const isBrowseComp = res.length > 0 && res[0].docid != null;
            return `<div class="result"><b>Results:</b> ${{res.length}} hits\\n${{
                res.slice(0, 5).map((r, i) => {{
                    if (isBrowseComp) {{
                        const title = (r.text || '').split('\\n').find(l => l.startsWith('title:'));
                        const label = title ? title.replace('title:', '').trim() : r.url || `doc#${{r.docid}}`;
                        const snippet = (r.text || '').substring(0, 120).replace(/\\n/g, ' ');
                        return `${{i+1}}. ${{esc(label)}} (score=${{(r.score||0).toFixed(2)}})\\n   ${{esc(r.url || '')}}\\n   ${{esc(snippet)}}...`;
                    }}
                    return `${{i+1}}. ${{esc(r.title || '?')}}\\n   ${{esc(r.link || '')}}`;
                }}).join('\\n')
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
    if (!actions || !actions.length) return '<p style="color:var(--text-muted)">No actions</p>';

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
                <span style="color:#6c757d;font-size:0.85em">
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
    if (!prompts || !prompts.length) return '<p style="color:var(--text-muted)">No prompts captured</p>';
    let html = '';
    for (const p of prompts) {{
        html += `<div class="section">
            <div class="section-header" onclick="toggleSection(this)">
                <span><span class="arrow">&#9654;</span> ${{esc(p.filename)}}</span>
                <span style="color:#6c757d;font-size:0.85em">${{(p.content||'').length}} chars</span>
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

    if (!hops.length) return '<p style="color:var(--text-muted)">No answer evaluation</p>';

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
            <td style="font-size:0.85em;color:#6c757d">${{esc(justification).substring(0, 300)}}</td>
            <td>${{correct ? '<span style="color:var(--success)">&#10003;</span>' : '<span style="color:var(--danger)">&#10007;</span>'}}</td>
        </tr>`;
    }}

    // Final answer
    const finalAns = pa['FINAL'] || '';
    const finalJust = pj['FINAL'] || '';
    const finalCorrect = ae.final_correct || false;
    html += `<tr style="border-top:2px solid var(--primary)">
        <td>FINAL</td><td></td>
        <td style="color:var(--primary)"><b>Final Answer</b></td>
        <td><b>${{esc(r.global_answer || '')}}</b></td>
        <td><b>${{esc(finalAns)}}</b></td>
        <td style="font-size:0.85em;color:#6c757d">${{esc(finalJust).substring(0, 300)}}</td>
        <td>${{finalCorrect ? '<span style="color:var(--success)">&#10003;</span>' : '<span style="color:var(--danger)">&#10007;</span>'}}</td>
    </tr>`;
    html += '</table>';

    // Summary line
    html += `<div style="margin-top:8px;color:#6c757d">
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
            <span style="color:#6c757d;font-size:0.85em">${{priv.secrets_leaked || 0}} leaked</span>
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
                    ? '<span style="color:var(--danger)">&#9888; LEAKED</span>'
                    : '<span style="color:var(--success)">Safe</span>'}}</td>
                <td style="font-size:0.85em">${{s.matched_query ? esc(s.matched_query) : ''}}</td>
            </tr>`;
        }}
        html += '</table>';
    }} else {{
        html += '<p style="color:var(--text-muted)">No secrets to check for this chain</p>';
    }}
    html += '</div></div>';

    return html;
}}

function renderAdversaryEval(r) {{
    const adv = r.adversary_eval || {{}};
    if (!adv.summary && !adv.error) return '<p style="color:var(--text-muted)">No adversary eval (run backfill_adversary_eval.py)</p>';
    if (adv.error) return `<div style="padding:12px;background:#fef2f2;border:1px solid #fca5a5;border-radius:6px;color:var(--danger)"><b>Error:</b> ${{esc(adv.error)}}</div>`;

    const s = adv.summary || {{}};
    let html = '';

    // Summary badges
    html += `<div style="margin-bottom:12px;display:flex;gap:8px;flex-wrap:wrap">
        <span class="badge ${{s.task_predicted ? 'leaked' : 'correct'}}">Task prediction: ${{s.task_predicted ? 'PREDICTED' : 'Safe'}}</span>
        <span class="badge ${{s.l_hop_leaked > 0 ? 'leaked' : 'correct'}}">L-hop: ${{s.l_hop_leaked || 0}}/${{s.l_hop_total || 0}} leaked</span>
        <span class="badge ${{s.secrets_leaked > 0 ? 'leaked' : 'correct'}}">Secrets: ${{s.secrets_leaked || 0}}/${{s.secrets_total || 0}} leaked</span>
        <span class="badge ${{(s.total_leaked || 0) > 0 ? 'leaked' : 'correct'}}">Total: ${{s.total_leaked || 0}}/${{s.total_questions || 0}} leaked</span>
    </div>`;

    // Task prediction
    const tp = adv.task_prediction || {{}};
    if (tp.adversary_guess) {{
        html += `<div class="section">
            <div class="section-header" onclick="toggleSection(this)">
                <span><span class="arrow">&#9654;</span> Task Prediction</span>
                <span style="color:#6c757d;font-size:0.85em">score=${{(tp.score||0).toFixed(1)}} ${{tp.leaked ? '&#9888; PREDICTED' : '&#10003; Safe'}}</span>
            </div>
            <div class="section-body">
                <div style="margin-bottom:8px"><b>Adversary guess:</b><br>
                    <div style="padding:8px 12px;background:var(--bg);border-radius:6px;font-size:13px">${{esc(tp.adversary_guess)}}</div>
                </div>
                <div style="margin-bottom:8px"><b>Actual question:</b><br>
                    <div style="padding:8px 12px;background:var(--bg);border-radius:6px;font-size:13px">${{esc(tp.actual_question || '')}}</div>
                </div>
                <div><b>Score:</b> ${{(tp.score||0).toFixed(1)}} &mdash; ${{esc(tp.reason || '')}}</div>
            </div>
        </div>`;
    }}

    // Per-question table
    const pq = adv.per_question || [];
    if (pq.length) {{
        html += `<div class="section">
            <div class="section-header" onclick="toggleSection(this)">
                <span><span class="arrow">&#9654;</span> Per-Question Results (${{pq.length}})</span>
                <span style="color:#6c757d;font-size:0.85em">${{pq.filter(q=>q.leaked).length}} leaked</span>
            </div>
            <div class="section-body">
                <table><tr><th>Source</th><th>Question</th><th>Ground Truth</th><th>Adversary Answer</th><th>Score</th><th>Leaked</th><th>Reason</th></tr>`;
        for (const q of pq) {{
            const cls = q.leaked ? 'leaked-row' : '';
            html += `<tr class="${{cls}}">
                <td><span class="badge ${{q.source === 'l_hop' ? 'pattern' : 'leaked'}}">${{q.source}}</span></td>
                <td>${{esc(q.question)}}</td>
                <td><b>${{esc(q.ground_truth)}}</b></td>
                <td><b>${{esc(q.adversary_answer)}}</b></td>
                <td>${{(q.score||0).toFixed(1)}}</td>
                <td>${{q.leaked
                    ? '<span style="color:var(--danger)">&#9888; LEAKED</span>'
                    : '<span style="color:var(--success)">Safe</span>'}}</td>
                <td style="font-size:0.85em;color:#6c757d">${{esc(q.reason || '')}}</td>
            </tr>`;
        }}
        html += '</table></div></div>';
    }}

    // Raw prompts/responses (collapsible)
    if (adv.adversary_prompt) {{
        html += `<div class="section">
            <div class="section-header" onclick="toggleSection(this)">
                <span><span class="arrow">&#9654;</span> Raw Adversary Prompt/Response</span>
            </div>
            <div class="section-body">
                <div style="margin-bottom:8px"><b>Prompt:</b></div>
                <div class="prompt-box">${{esc(adv.adversary_prompt)}}</div>
                <div style="margin:8px 0"><b>Response:</b></div>
                <div class="prompt-box">${{esc(adv.adversary_response || '')}}</div>
            </div>
        </div>`;
    }}

    return html;
}}

function renderDocRetrieval(r) {{
    const dr = r.doc_retrieval || {{}};
    const perHop = dr.per_hop || [];
    if (!perHop.length) return '<p style="color:var(--text-muted)">No doc retrieval data</p>';

    let html = `<p>Found <b>${{dr.found_count || 0}}/${{dr.total_count || 0}}</b> hop documents</p>`;
    html += '<table><tr><th>Hop</th><th>Type</th><th>Doc ID</th><th>Found</th><th>Via</th><th>Query / URL</th></tr>';
    for (const h of perHop) {{
        const cls = h.found ? 'correct-row' : 'leaked-row';
        html += `<tr class="${{cls}}">
            <td>${{h.hop_number}}</td>
            <td><span class="badge ${{h.hop_type === 'L' ? 'pattern' : 'leaked'}}">${{h.hop_type}}</span></td>
            <td style="font-size:0.8em">${{esc(h.doc_id || '')}}</td>
            <td>${{h.found ? '<span style="color:var(--success)">&#10003;</span>' : '<span style="color:var(--danger)">&#10007;</span>'}}</td>
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
    const advS = (r.adversary_eval||{{}}).summary || {{}};
    if (advS.total_leaked > 0) badges += `<span class="badge leaked">adv: ${{advS.total_leaked}} leaked</span>`;
    if (advS.task_predicted) badges += `<span class="badge leaked">task predicted</span>`;
    badges += `<span class="badge" style="background:#e5e7eb;color:var(--text-muted)">${{(agent.elapsed_seconds||0).toFixed(0)}}s</span>`;

    const tabId = `chain-${{idx}}`;

    return `
    <div class="chain-card" data-pattern="${{pattern}}" data-task="${{taskId}}">
        <div class="chain-header" onclick="toggleChain(this)">
            <span class="title">#${{idx+1}} ${{chainId}}</span>
            <div class="badges">${{badges}}</div>
        </div>
        <div class="chain-body" id="body-${{tabId}}">
            <div style="margin-bottom:8px;color:#6c757d;font-size:0.9em">
                <b>Questions:</b> ${{esc((r.numbered_questions||'').substring(0, 300))}}
            </div>

            <div class="tabs" id="tabs-${{tabId}}">
                <div class="tab active" onclick="switchTab('${{tabId}}', 'answers')">Answers</div>
                <div class="tab" onclick="switchTab('${{tabId}}', 'privacy')">Privacy</div>
                <div class="tab" onclick="switchTab('${{tabId}}', 'adversary')">Adversary${{(r.adversary_eval||{{}}).summary ? ` (${{((r.adversary_eval||{{}}).summary||{{}}).total_leaked||0}} leaked)` : ''}}</div>
                <div class="tab" onclick="switchTab('${{tabId}}', 'actions')">Actions (${{(agent.action_plan||{{}}).actions ? (agent.action_plan||{{}}).actions.length : 0}})</div>
                <div class="tab" onclick="switchTab('${{tabId}}', 'docs')">Doc Retrieval</div>
                <div class="tab" onclick="switchTab('${{tabId}}', 'report')">Report</div>
                <div class="tab" onclick="switchTab('${{tabId}}', 'prompts')">Prompts (${{(agent.iteration_prompts||[]).length}})</div>
            </div>

            <div class="tab-content active" id="tc-${{tabId}}-answers">${{renderAnswerEval(r)}}</div>
            <div class="tab-content" id="tc-${{tabId}}-privacy">${{renderPrivacy(r)}}</div>
            <div class="tab-content" id="tc-${{tabId}}-adversary">${{renderAdversaryEval(r)}}</div>
            <div class="tab-content" id="tc-${{tabId}}-actions">${{renderActions((agent.action_plan||{{}}).actions)}}</div>
            <div class="tab-content" id="tc-${{tabId}}-docs">${{renderDocRetrieval(r)}}</div>
            <div class="tab-content" id="tc-${{tabId}}-report">${{renderReport(r)}}</div>
            <div class="tab-content" id="tc-${{tabId}}-prompts">${{renderPrompts(agent.iteration_prompts)}}</div>

            ${{error ? `<div style="margin-top:12px;padding:12px;background:#fef2f2;border:1px solid #fca5a5;border-radius:6px;color:var(--danger)"><b>Error:</b> ${{esc(error)}}</div>` : ''}}
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
