#!/usr/bin/env python3
"""Chain viewer — interactive HTML for inspecting generated chains.

Shows each chain's numbered questions, hop-by-hop Q/A with quotes,
bridge candidates tried, check results, and the 4-condition verification.

Usage:
    python -m making_dataset_2.view_chains --input chains.jsonl
    python -m making_dataset_2.view_chains --input chains.jsonl --output viewer.html --open
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path


def _load_chains(path: Path) -> list[dict]:
    chains = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                chains.append(json.loads(line))
    return chains


def _build_stats(chains: list[dict]) -> dict:
    patterns = Counter(c.get("pattern", "?") for c in chains)
    companies = Counter(c.get("metadata", {}).get("company", "?") for c in chains)
    valid = sum(1 for c in chains if c.get("verification", {}).get("is_valid"))
    complete = sum(1 for c in chains if c.get("metadata", {}).get("complete"))
    return {
        "total": len(chains),
        "valid": valid,
        "complete": complete,
        "by_pattern": dict(patterns.most_common()),
        "by_company": dict(companies.most_common()),
    }


def generate_html(chains: list[dict], output: Path) -> None:
    stats = _build_stats(chains)
    chains_json = json.dumps(chains, ensure_ascii=False)
    stats_json = json.dumps(stats, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chain Viewer</title>
<style>
:root {{
  --bg: #f8f9fa; --card: #ffffff; --primary: #4361ee; --primary-light: #eef1ff;
  --local: #4361ee; --web: #7209b7; --success: #10b981; --warning: #f59e0b;
  --danger: #ef4444; --text: #212529; --text-muted: #6c757d; --border: #dee2e6;
  --shadow: 0 2px 8px rgba(0,0,0,0.08);
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; }}

/* Header */
header {{ background: var(--card); padding: 14px 24px; border-bottom: 1px solid var(--border); box-shadow: var(--shadow); }}
header h1 {{ font-size: 18px; font-weight: 600; color: var(--primary); margin-bottom: 6px; }}
.stats {{ display: flex; gap: 20px; flex-wrap: wrap; font-size: 13px; color: var(--text-muted); }}
.stat {{ display: flex; align-items: center; gap: 5px; }}
.stat-value {{ color: var(--text); font-weight: 600; }}

/* Filters */
.filters {{ display: flex; gap: 10px; padding: 10px 24px; background: var(--card); border-bottom: 1px solid var(--border); flex-wrap: wrap; align-items: center; }}
select, input[type="text"] {{ padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px; font-size: 13px; background: white; }}
input[type="text"] {{ min-width: 200px; }}
select:focus, input:focus {{ outline: none; border-color: var(--primary); }}
.filter-label {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; }}

/* Layout */
.layout {{ display: grid; grid-template-columns: 320px 1fr; height: calc(100vh - 110px); }}
.sidebar {{ background: var(--card); border-right: 1px solid var(--border); overflow-y: auto; }}
.detail {{ overflow-y: auto; padding: 24px; }}

/* Sidebar items */
.chain-item {{ padding: 10px 14px; border-bottom: 1px solid var(--bg); cursor: pointer; transition: background 0.12s; }}
.chain-item:hover {{ background: var(--primary-light); }}
.chain-item.selected {{ background: var(--primary-light); border-left: 3px solid var(--primary); }}
.chain-q {{ font-size: 12px; color: var(--text); line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; margin-bottom: 6px; }}
.chain-meta {{ display: flex; gap: 6px; flex-wrap: wrap; align-items: center; font-size: 11px; color: var(--text-muted); }}

/* Badges */
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 500; }}
.badge-L {{ background: var(--primary-light); color: var(--primary); }}
.badge-W {{ background: #f3e8ff; color: var(--web); }}
.badge-valid {{ background: #dcfce7; color: #166534; }}
.badge-invalid {{ background: #fee2e2; color: #991b1b; }}
.badge-incomplete {{ background: #fef3c7; color: #92400e; }}
.badge-private {{ background: #f3e8ff; color: #6b21a8; }}
.badge-public {{ background: #e0f2fe; color: #075985; }}

/* Numbered question box */
.numbered-box {{ background: var(--card); border: 2px solid var(--primary); border-radius: 8px; padding: 16px 20px; margin-bottom: 20px; line-height: 1.7; font-size: 14px; box-shadow: var(--shadow); }}
.nq-row {{ display: flex; gap: 8px; align-items: baseline; margin-bottom: 6px; }}
.nq-num {{ font-weight: 700; color: var(--primary); white-space: nowrap; min-width: 24px; }}
.nq-q {{ flex: 1; }}
.nq-arrow {{ color: var(--text-muted); margin: 0 4px; }}
.nq-a {{ color: var(--success); font-weight: 700; white-space: nowrap; }}
.nq-final {{ background: var(--primary-light); border-radius: 4px; padding: 2px 6px; }}

/* Privacy assessment */
.privacy-box {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 16px; box-shadow: var(--shadow); }}
.privacy-box h3 {{ font-size: 13px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 10px; }}
.privacy-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; font-size: 13px; }}
.privacy-icon {{ width: 20px; text-align: center; }}
.privacy-secret {{ background: #fef3c7; border: 1px solid #fde68a; border-radius: 6px; padding: 8px 12px; font-size: 13px; margin-bottom: 8px; }}
.privacy-secret .secret-label {{ font-size: 11px; font-weight: 600; color: #92400e; text-transform: uppercase; }}
.privacy-verdict {{ display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600; }}
.privacy-verdict.secure {{ background: #dcfce7; color: #166534; }}
.privacy-verdict.exposed {{ background: #fee2e2; color: #991b1b; }}
.privacy-verdict.untested {{ background: #f3f4f6; color: #6b7280; }}

/* Hop sections */
.hop-section {{ margin-bottom: 12px; background: var(--card); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; box-shadow: var(--shadow); }}
.hop-header {{ padding: 10px 16px; background: var(--bg); border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 10px; font-weight: 600; font-size: 13px; }}
.hop-body {{ padding: 16px; }}

.qa-line {{ margin-bottom: 6px; font-size: 14px; }}
.qa-line .label {{ font-weight: 600; color: var(--text-muted); margin-right: 6px; }}
.qa-line .q-text {{ color: var(--text); }}
.qa-line .a-text {{ color: var(--success); font-weight: 600; }}

.quote-box {{ background: #fffbeb; border-left: 3px solid var(--warning); padding: 10px 14px; border-radius: 0 6px 6px 0; margin: 8px 0; font-size: 13px; font-style: italic; color: #92400e; line-height: 1.5; }}
.quote-box .quote-src {{ font-style: normal; font-size: 11px; color: var(--text-muted); margin-top: 4px; }}

.doc-id {{ font-family: ui-monospace, SFMono-Regular, monospace; font-size: 11px; color: var(--text-muted); }}

/* Trace steps */
.trace-step {{ background: var(--bg); border-radius: 6px; padding: 8px 12px; margin-top: 8px; font-size: 12px; border-left: 3px solid var(--border); }}
.trace-step.pass {{ border-left-color: var(--success); }}
.trace-step.fail {{ border-left-color: var(--danger); }}
.trace-label {{ font-weight: 600; font-size: 11px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 2px; }}

/* Verification grid */
.verify-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 8px; }}
.verify-cell {{ background: var(--bg); padding: 12px; border-radius: 6px; text-align: center; }}
.verify-cell.pass {{ border-bottom: 3px solid var(--success); }}
.verify-cell.fail {{ border-bottom: 3px solid var(--danger); }}
.verify-label {{ font-size: 11px; color: var(--text-muted); margin-bottom: 4px; }}
.verify-result {{ font-weight: 600; font-size: 14px; }}
.verify-pass {{ color: var(--success); }}
.verify-fail {{ color: var(--danger); }}

/* Collapsible */
details {{ margin-top: 6px; }}
details summary {{ cursor: pointer; font-size: 11px; color: var(--text-muted); }}
details summary:hover {{ color: var(--text); }}
details pre {{ background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px; margin-top: 6px; max-height: 400px; overflow: auto; font-size: 12px; font-family: ui-monospace, SFMono-Regular, monospace; white-space: pre-wrap; word-break: break-word; line-height: 1.5; }}

.metadata {{ color: var(--text-muted); font-size: 12px; margin-top: 16px; padding-top: 12px; border-top: 1px solid var(--border); }}

/* Tabs */
.tab-bar {{ display: flex; gap: 0; border-bottom: 2px solid var(--border); margin-bottom: 16px; }}
.tab {{ padding: 8px 20px; cursor: pointer; font-size: 13px; font-weight: 600; color: var(--text-muted); border-bottom: 2px solid transparent; margin-bottom: -2px; transition: all 0.15s; }}
.tab:hover {{ color: var(--text); }}
.tab.active {{ color: var(--primary); border-bottom-color: var(--primary); }}
.tab-panel {{ display: none; }}
.tab-panel.active {{ display: block; }}

/* Agent run card */
.agent-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 16px; box-shadow: var(--shadow); }}
.agent-card h3 {{ font-size: 13px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 10px; }}
.agent-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }}
.agent-stat {{ background: var(--bg); padding: 10px 14px; border-radius: 6px; }}
.agent-stat .stat-num {{ font-size: 22px; font-weight: 700; color: var(--primary); }}
.agent-stat .stat-label {{ font-size: 11px; color: var(--text-muted); }}

/* Doc retrieval grid */
.doc-grid {{ display: grid; gap: 8px; margin-top: 8px; }}
.doc-row {{ display: flex; align-items: center; gap: 10px; padding: 8px 12px; background: var(--bg); border-radius: 6px; font-size: 13px; }}
.doc-row.found {{ border-left: 3px solid var(--success); }}
.doc-row.not-found {{ border-left: 3px solid var(--danger); opacity: 0.7; }}
.doc-query {{ font-size: 11px; color: var(--text-muted); font-style: italic; margin-top: 2px; }}

/* Privacy results table */
.priv-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.priv-table th {{ text-align: left; padding: 8px 10px; background: var(--bg); border-bottom: 2px solid var(--border); font-size: 11px; text-transform: uppercase; color: var(--text-muted); }}
.priv-table td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }}
.priv-table tr:hover {{ background: var(--primary-light); }}
.badge-leaked {{ background: #fee2e2; color: #991b1b; padding: 1px 6px; border-radius: 8px; font-size: 9px; font-weight: 600; }}
.badge-safe {{ background: #dcfce7; color: #166534; padding: 1px 6px; border-radius: 8px; font-size: 9px; font-weight: 600; }}
.badge-rules-leaked {{ background: #dbeafe; color: #1e40af; padding: 1px 5px; border-radius: 8px; font-size: 9px; font-weight: 600; }}
.badge-rules-safe {{ background: #eff6ff; color: #93c5fd; padding: 1px 5px; border-radius: 8px; font-size: 9px; font-weight: 600; }}
.badge-adv-leaked {{ background: #fee2e2; color: #991b1b; padding: 1px 5px; border-radius: 8px; font-size: 9px; font-weight: 600; }}
.badge-adv-safe {{ background: #fef2f2; color: #fca5a5; padding: 1px 5px; border-radius: 8px; font-size: 9px; font-weight: 600; }}
.badge-insight {{ background: #dbeafe; color: #1e40af; }}
.badge-distractor {{ background: #f3e8ff; color: #6b21a8; }}

/* Queries list */
.query-list {{ list-style: none; padding: 0; margin: 0; }}
.query-list li {{ padding: 6px 10px; border-bottom: 1px solid var(--border); font-size: 13px; font-family: ui-monospace, SFMono-Regular, monospace; }}
.query-list li:nth-child(odd) {{ background: var(--bg); }}
.query-num {{ color: var(--text-muted); margin-right: 6px; }}

/* Privacy summary cards at top of privacy tab */
.priv-summary {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }}
.priv-summary-card {{ border-radius: 8px; padding: 16px; }}
.priv-summary-card h4 {{ font-size: 12px; font-weight: 700; text-transform: uppercase; margin-bottom: 10px; letter-spacing: 0.5px; }}
.priv-summary-card.rules {{ background: #eff6ff; border: 2px solid #bfdbfe; }}
.priv-summary-card.rules h4 {{ color: #1e40af; }}
.priv-summary-card.adversary {{ background: #fef2f2; border: 2px solid #fecaca; }}
.priv-summary-card.adversary h4 {{ color: #991b1b; }}
.priv-stat-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 13px; }}
.priv-stat-big {{ font-size: 28px; font-weight: 800; line-height: 1; }}
.priv-link {{ display: inline-block; margin-top: 8px; font-size: 11px; color: var(--primary); cursor: pointer; text-decoration: underline; }}
.priv-link:hover {{ color: var(--text); }}

/* Summary view */
.summary-view {{ padding: 24px; max-width: 1400px; margin: 0 auto; }}
.summary-view h2 {{ font-size: 18px; font-weight: 700; margin-bottom: 16px; color: var(--primary); }}
.summary-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.summary-table th {{ text-align: left; padding: 8px 12px; background: var(--primary-light); border-bottom: 2px solid var(--primary); font-size: 11px; text-transform: uppercase; color: var(--primary); font-weight: 700; position: sticky; top: 0; z-index: 1; }}
.summary-table td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }}
.summary-table tr:hover {{ background: var(--primary-light); }}
.summary-table .q-cell {{ line-height: 1.6; }}
.summary-table .q-cell .hop-line {{ display: flex; gap: 6px; align-items: baseline; margin-bottom: 2px; }}
.summary-table .q-cell .hop-num {{ font-weight: 700; color: var(--primary); min-width: 18px; font-size: 11px; }}
.summary-table .q-cell .hop-badge {{ font-size: 9px; padding: 1px 5px; border-radius: 6px; font-weight: 600; }}
.summary-table .q-cell .hop-q {{ flex: 1; }}
.summary-table .q-cell .hop-arrow {{ color: var(--text-muted); }}
.summary-table .q-cell .hop-a {{ color: var(--success); font-weight: 700; white-space: nowrap; }}
.summary-table .final-answer {{ font-weight: 700; color: var(--success); font-size: 14px; }}
.summary-toggle {{ background: var(--primary); color: white; border: none; border-radius: 6px; padding: 6px 14px; font-size: 12px; font-weight: 600; cursor: pointer; margin-left: auto; }}
.summary-toggle:hover {{ opacity: 0.9; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 7px; height: 7px; }}
::-webkit-scrollbar-track {{ background: #f1f1f1; }}
::-webkit-scrollbar-thumb {{ background: #c1c1c1; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: #a1a1a1; }}

.no-results {{ padding: 40px; text-align: center; color: var(--text-muted); }}
</style>
</head>
<body>

<header>
  <div style="display:flex;align-items:center;gap:16px">
    <h1 style="flex:1">Chain Viewer</h1>
    <button class="summary-toggle" id="btn-summary" onclick="toggleSummary()">Summary View</button>
  </div>
  <div class="stats" id="stats-bar"></div>
</header>

<div class="filters">
  <div><span class="filter-label">Pattern</span>
    <select id="f-pattern"><option value="">All</option></select></div>
  <div><span class="filter-label">Company</span>
    <select id="f-company"><option value="">All</option></select></div>
  <div><span class="filter-label">Status</span>
    <select id="f-status"><option value="">All</option><option value="valid">Valid</option><option value="invalid">Invalid</option><option value="incomplete">Incomplete</option></select></div>
  <div><span class="filter-label">Rules</span>
    <select id="f-rules"><option value="">All</option><option value="leaked">Leaked</option><option value="safe">Safe</option><option value="untested">Not Tested</option></select></div>
  <div><span class="filter-label">Adversary</span>
    <select id="f-adversary"><option value="">All</option><option value="leaked">Leaked</option><option value="safe">Safe</option><option value="untested">Not Tested</option></select></div>
  <div><span class="filter-label">Search</span>
    <input type="text" id="f-search" placeholder="Question, answer, entity..."></div>
</div>

<div id="summary-view" class="summary-view" style="display:none;height:calc(100vh - 110px);overflow-y:auto"></div>

<div class="layout" id="detail-layout">
  <div class="sidebar" id="chain-list"></div>
  <div class="detail" id="detail">
    <div class="no-results">Select a chain from the sidebar</div>
  </div>
</div>

<script>
const CHAINS = {chains_json};
const STATS = {stats_json};

function esc(s) {{ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }}
function trunc(s, n) {{ return s && s.length > n ? s.slice(0, n) + '...' : (s || ''); }}

let selectedIdx = null;
let summaryMode = false;

function toggleSummary() {{
  summaryMode = !summaryMode;
  document.getElementById('summary-view').style.display = summaryMode ? 'block' : 'none';
  document.getElementById('detail-layout').style.display = summaryMode ? 'none' : 'grid';
  document.querySelector('.filters').style.display = summaryMode ? 'none' : 'flex';
  document.getElementById('btn-summary').textContent = summaryMode ? 'Detail View' : 'Summary View';
  if (summaryMode) renderSummary();
}}

function renderSummary() {{
  const el = document.getElementById('summary-view');
  // Group by company
  const byCompany = {{}};
  CHAINS.forEach(c => {{
    const co = c.metadata?.company || 'Unknown';
    if (!byCompany[co]) byCompany[co] = [];
    byCompany[co].push(c);
  }});

  let h = '';
  for (const [company, chains] of Object.entries(byCompany)) {{
    h += `<h2>${{esc(company)}} (${{chains.length}} chains)</h2>`;
    h += `<table class="summary-table"><thead><tr>`;
    h += `<th style="width:60px">Pattern</th>`;
    h += `<th style="width:70px">Task</th>`;
    h += `<th>Questions &amp; Answers</th>`;
    h += `<th style="width:90px">Final Answer</th>`;
    h += `<th style="width:60px">Status</th>`;
    h += `</tr></thead><tbody>`;

    for (const c of chains) {{
      const hops = c.hops || [];
      const s = getStatus(c);
      const nqLines = (c.numbered_questions || '').split('\\n').filter(l => l.trim());

      h += `<tr>`;
      h += `<td><span class="badge badge-L">${{esc(c.pattern)}}</span></td>`;
      h += `<td style="font-family:monospace;font-size:11px">${{esc(c.metadata?.task_id || '')}}</td>`;

      // Questions cell with all hops
      h += `<td class="q-cell">`;
      for (let i = 0; i < hops.length; i++) {{
        const hop = hops[i];
        const isFinal = i === hops.length - 1;
        const qText = (i < nqLines.length) ? nqLines[i].replace(/^[0-9]+[^a-zA-Z]*/, '') : hop.question;
        const typeBadge = hop.hop_type === 'L'
          ? '<span class="hop-badge" style="background:var(--primary-light);color:var(--primary)">L</span>'
          : '<span class="hop-badge" style="background:#f3e8ff;color:var(--web)">W</span>';
        h += `<div class="hop-line${{isFinal ? ' nq-final' : ''}}">`;
        h += `<span class="hop-num">${{i+1}}.</span>`;
        h += typeBadge;
        h += `<span class="hop-q">${{esc(qText)}}</span>`;
        h += `<span class="hop-arrow">&rarr;</span>`;
        h += `<span class="hop-a">${{esc(hop.answer)}}</span>`;
        h += `</div>`;
      }}
      h += `</td>`;

      h += `<td class="final-answer">${{esc(c.global_answer || hops[hops.length-1]?.answer || '')}}</td>`;
      h += `<td><span class="badge badge-${{s}}">${{s}}</span></td>`;
      h += `</tr>`;
    }}
    h += `</tbody></table>`;
  }}
  el.innerHTML = h;
}}

function getStatus(c) {{
  if (!c.metadata?.complete) return 'incomplete';
  if (c.verification?.is_valid) return 'valid';
  return 'invalid';
}}

function getPrivacySummary(c) {{
  if (!c.agent_run) return null;
  const pe = c.privacy_eval || {{}};
  const qc = pe.quick_check || {{}};
  const ae = pe.adversary_eval || {{}};
  const s = ae.summary || {{}};
  const companyLeak = qc.company_name_leaked || false;
  const metricsLeak = (qc.metrics_leaked || 0) > 0;
  const insightsLeak = (s.insights_leaked || 0) > 0;
  const distractorsLeak = (s.distractors_leaked || 0) > 0;
  const anyLeak = companyLeak || metricsLeak || insightsLeak;
  return {{
    companyLeak, metricsLeak, insightsLeak, distractorsLeak, anyLeak,
    metricsLeaked: qc.metrics_leaked||0, metricsTotal: qc.metrics_total||0,
    insightsLeaked: s.insights_leaked||0, insightsTotal: s.insights_total||0,
    distractorsLeaked: s.distractors_leaked||0, distractorsTotal: s.distractors_total||0,
  }};
}}

function getRulesFilter(c) {{
  if (!c.agent_run) return 'untested';
  const qc = (c.privacy_eval || {{}}).quick_check || {{}};
  return (qc.company_name_leaked || (qc.metrics_leaked || 0) > 0) ? 'leaked' : 'safe';
}}

function getAdvFilter(c) {{
  if (!c.agent_run) return 'untested';
  const s = ((c.privacy_eval || {{}}).adversary_eval || {{}}).summary || {{}};
  return ((s.insights_leaked || 0) > 0 || (s.distractors_leaked || 0) > 0) ? 'leaked' : 'safe';
}}

// Init
(function() {{
  // Stats bar
  const tested = CHAINS.filter(c => c.agent_run);
  const sb = document.getElementById('stats-bar');
  let statsHtml = `
    <div class="stat"><span class="stat-value">${{STATS.total}}</span> chains</div>
    <div class="stat"><span class="stat-value" style="color:var(--success)">${{STATS.valid}}</span> valid</div>
    <div class="stat"><span class="stat-value">${{STATS.complete}}</span> complete</div>
    ${{Object.entries(STATS.by_pattern).map(([p,n]) => `<div class="stat"><span class="stat-value">${{p}}</span>: ${{n}}</div>`).join('')}}
  `;
  if (tested.length) {{
    const aggRulesLeak = tested.filter(c => getRulesFilter(c) === 'leaked').length;
    const aggAdvLeak = tested.filter(c => getAdvFilter(c) === 'leaked').length;
    const aggInsights = tested.reduce((a,c) => a + (c.privacy_eval?.adversary_eval?.summary?.insights_leaked||0), 0);
    const aggInsightsTotal = tested.reduce((a,c) => a + (c.privacy_eval?.adversary_eval?.summary?.insights_total||0), 0);
    statsHtml += `
      <div class="stat" style="border-left:1px solid var(--border);padding-left:12px">
        <span class="stat-value">${{tested.length}}</span> tested</div>
      <div class="stat"><span class="stat-value" style="color:${{aggRulesLeak > 0 ? 'var(--danger)' : 'var(--success)'}}">${{aggRulesLeak}}</span> rules leaked</div>
      <div class="stat"><span class="stat-value" style="color:${{aggAdvLeak > 0 ? 'var(--danger)' : 'var(--success)'}}">${{aggAdvLeak}}</span> adv leaked</div>
      <div class="stat"><span class="stat-value" style="color:${{aggInsights > 0 ? 'var(--danger)' : 'var(--success)'}}">${{aggInsights}}/${{aggInsightsTotal}}</span> insights</div>
    `;
  }}
  sb.innerHTML = statsHtml;

  // Populate filters
  for (const p of Object.keys(STATS.by_pattern)) {{
    const o = document.createElement('option'); o.value = p; o.textContent = p;
    document.getElementById('f-pattern').appendChild(o);
  }}
  for (const c of Object.keys(STATS.by_company)) {{
    const o = document.createElement('option'); o.value = c; o.textContent = c;
    document.getElementById('f-company').appendChild(o);
  }}

  ['f-pattern','f-company','f-status','f-rules','f-adversary'].forEach(id =>
    document.getElementById(id).addEventListener('change', renderList));
  document.getElementById('f-search').addEventListener('input', renderList);

  renderList();
  if (CHAINS.length) showChain(0);
}})();

function renderList() {{
  const pat = document.getElementById('f-pattern').value;
  const comp = document.getElementById('f-company').value;
  const stat = document.getElementById('f-status').value;
  const rulesF = document.getElementById('f-rules').value;
  const advF = document.getElementById('f-adversary').value;
  const search = document.getElementById('f-search').value.toLowerCase();

  const el = document.getElementById('chain-list');
  let h = '';
  CHAINS.forEach((c, i) => {{
    const s = getStatus(c);
    const rf = getRulesFilter(c);
    const af = getAdvFilter(c);
    if (pat && c.pattern !== pat) return;
    if (comp && (c.metadata?.company||'') !== comp) return;
    if (stat && s !== stat) return;
    if (rulesF && rf !== rulesF) return;
    if (advF && af !== advF) return;
    if (search) {{
      const text = JSON.stringify(c).toLowerCase();
      if (!text.includes(search)) return;
    }}

    const badgeCls = 'badge-' + s;
    const nHops = (c.hops||[]).length;
    const firstQ = c.hops?.[0]?.question || '';
    const ps = getPrivacySummary(c);

    h += `<div class="chain-item ${{i === selectedIdx ? 'selected' : ''}}" onclick="showChain(${{i}})">`;
    h += `<div class="chain-q">${{esc(firstQ)}}</div>`;
    h += `<div class="chain-meta">`;
    h += `<span class="badge badge-L">${{esc(c.pattern)}}</span> `;
    h += `<span class="badge ${{badgeCls}}">${{s}}</span> `;
    h += `<span>${{nHops}} hops</span> `;
    if (ps) {{
      // Rules pill
      const rulesLeak = ps.companyLeak || ps.metricsLeaked > 0;
      const rTip = rulesLeak
        ? `Rules: Co=${{ps.companyLeak ? 'YES' : 'no'}}, ${{ps.metricsLeaked}}/${{ps.metricsTotal}} metrics`
        : 'Rules: no regex matches';
      h += `<span class="badge ${{rulesLeak ? 'badge-rules-leaked' : 'badge-rules-safe'}}" title="${{esc(rTip)}}">R:${{rulesLeak ? 'LEAK' : 'OK'}}</span> `;
      // Adversary pill
      const advLeak = ps.insightsLeaked > 0 || ps.distractorsLeaked > 0;
      const aTip = advLeak
        ? `Adversary: ${{ps.insightsLeaked}}/${{ps.insightsTotal}} insights, ${{ps.distractorsLeaked}} distractors`
        : 'Adversary: could not infer private info';
      h += `<span class="badge ${{advLeak ? 'badge-adv-leaked' : 'badge-adv-safe'}}" title="${{esc(aTip)}}">A:${{advLeak ? 'LEAK' : 'OK'}}</span> `;
    }}
    h += `<span>${{esc(c.metadata?.company||'')}}</span>`;
    h += `</div></div>`;
  }});
  el.innerHTML = h || '<div class="no-results">No chains match filters</div>';
}}

function switchTab(tabId) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tabId));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === 'panel-' + tabId));
}}

function scrollToSection(sectionId) {{
  const el = document.getElementById(sectionId);
  if (el) el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
}}

function showChain(idx) {{
  selectedIdx = idx;
  renderList();

  const c = CHAINS[idx];
  const el = document.getElementById('detail');
  const s = getStatus(c);
  const badgeCls = 'badge-' + s;
  const hops = c.hops || [];
  const trace = c.metadata?.trace || [];
  const hasAgent = !!c.agent_run;

  let h = '';

  // Title
  h += `<div style="margin-bottom:16px;display:flex;align-items:center;gap:10px">`;
  h += `<span style="font-size:18px;font-weight:700">Chain ${{esc(c.chain_id)}}</span>`;
  h += `<span class="badge badge-L">${{esc(c.pattern)}}</span>`;
  h += `<span class="badge ${{badgeCls}}">${{s.toUpperCase()}}</span>`;
  h += `<span style="color:var(--text-muted);font-size:13px">${{esc(c.metadata?.company||'')}} / ${{esc(c.metadata?.task_id||'')}}</span>`;
  h += `</div>`;

  // Tabs
  h += `<div class="tab-bar">`;
  h += `<div class="tab active" data-tab="chain" onclick="switchTab('chain')">Chain Details</div>`;
  if (hasAgent) h += `<div class="tab" data-tab="agent" onclick="switchTab('agent')">Agent Process</div>`;
  if (hasAgent) h += `<div class="tab" data-tab="privacy" onclick="switchTab('privacy')">Privacy Eval</div>`;
  h += `</div>`;

  // =====================================================
  // TAB 1: Chain Details
  // =====================================================
  h += `<div class="tab-panel active" id="panel-chain">`;

  // Numbered questions with answers
  if (hops.length) {{
    h += `<div class="numbered-box">`;
    const nqLines = (c.numbered_questions || '').split('\\n').filter(l => l.trim());
    for (let i = 0; i < hops.length; i++) {{
      const hop = hops[i];
      const isFinal = i === hops.length - 1;
      const qText = (i < nqLines.length) ? nqLines[i] : `${{i+1}}. ${{hop.question}}`;
      h += `<div class="nq-row${{isFinal ? ' nq-final' : ''}}">`;
      h += `<div class="nq-q">${{esc(qText)}}</div>`;
      h += `<span class="nq-arrow">&rarr;</span>`;
      h += `<span class="nq-a">${{esc(hop.answer)}}</span>`;
      h += `</div>`;
    }}
    h += `</div>`;
  }}

  // Privacy assessment (chain-level)
  if (hops.length) {{
    const privateHops = hops.filter(h => h.hop_type === 'L');
    const publicHops = hops.filter(h => h.hop_type === 'W');
    const seed = hops[0];
    const v = c.verification || {{}};

    let verdict = 'untested', verdictText = 'Not verified';
    if (v.is_valid !== undefined) {{
      if (v.is_valid) {{ verdict = 'secure'; verdictText = 'Private info required — chain needs enterprise docs'; }}
      else {{ verdict = 'exposed'; verdictText = 'Weak — answerable without all private docs'; }}
    }}

    h += `<div class="privacy-box"><h3>Chain Privacy</h3>`;
    h += `<div class="privacy-secret">`;
    h += `<div class="secret-label">Enterprise Secret (Seed)</div>`;
    h += `<div><strong>Q:</strong> ${{esc(seed.question)}}</div>`;
    h += `<div><strong>A:</strong> <span style="color:var(--danger);font-weight:600">${{esc(seed.answer)}}</span></div>`;
    h += `</div>`;
    h += `<div class="privacy-row"><span class="privacy-icon">&#128274;</span> <strong>${{privateHops.length}}</strong> private doc${{privateHops.length !== 1 ? 's' : ''}} &nbsp; <span class="privacy-icon">&#127760;</span> <strong>${{publicHops.length}}</strong> public doc${{publicHops.length !== 1 ? 's' : ''}}</div>`;
    h += `<div style="margin-top:10px"><span class="privacy-verdict ${{verdict}}">${{verdictText}}</span></div>`;
    if (v.is_valid !== undefined) {{
      const icon = (ok) => ok ? '<span style="color:var(--success)">&#10003;</span>' : '<span style="color:var(--danger)">&#10007;</span>';
      h += `<div style="margin-top:10px;font-size:12px;color:var(--text-muted)">`;
      h += `${{icon(!v.no_docs_pass)}} No docs &nbsp; ${{icon(!v.first_only_pass)}} First only &nbsp; ${{icon(!v.last_only_pass)}} Last only &nbsp; ${{icon(v.all_docs_pass)}} All docs`;
      h += `</div>`;
    }}
    h += `</div>`;
  }}

  // Hops
  for (const hop of hops) {{
    const isLocal = hop.hop_type === 'L';
    const typeLabel = isLocal ? 'LOCAL' : 'WEB';
    const typeBadge = isLocal ? 'badge-L' : 'badge-W';
    const privBadge = isLocal
      ? '<span class="badge badge-private">PRIVATE</span>'
      : '<span class="badge badge-public">PUBLIC</span>';
    const isFinal = hop.hop_number === hops[hops.length - 1].hop_number;
    const docShort = (hop.doc_id || '').split('/').pop() || hop.doc_id;

    h += `<div class="hop-section">`;
    h += `<div class="hop-header"><span class="badge ${{typeBadge}}">${{typeLabel}}</span> Hop ${{hop.hop_number}}${{isFinal ? ' (final)' : ''}} ${{privBadge}}</div>`;
    h += `<div class="hop-body">`;
    h += `<div class="qa-line"><span class="label">Q:</span><span class="q-text">${{esc(hop.question)}}</span></div>`;
    h += `<div class="qa-line"><span class="label">A:</span><span class="a-text">${{esc(hop.answer)}}</span></div>`;
    if (hop.quote) {{
      h += `<div class="quote-box">"${{esc(hop.quote)}}"<div class="quote-src">${{esc(docShort)}}</div></div>`;
    }}
    h += `<div class="doc-id">${{esc(hop.doc_id)}}</div>`;

    const hopTraces = trace.filter(t => t.transition === hop.hop_number || (t.step === 'find_bridge' && t.transition === hop.hop_number));
    const fb = trace.find(t => t.step === 'find_bridge' && hopTraces.includes(t));
    if (fb) {{
      h += `<details><summary>Bridge candidates (${{fb.n_candidates}} total)</summary><pre>`;
      for (const tc of (fb.top_candidates || [])) {{
        h += `${{esc(tc.entity||tc.bridge_entity||'')}} -> ${{esc(trunc(tc.doc||'',50))}} (intra=${{tc.needs_intra}}, score=${{tc.score}})\\n`;
      }}
      h += `</pre></details>`;
    }}
    const checks = hopTraces.filter(t => t.step === 'check_intra' || t.step === 'check_inter');
    for (const chk of checks) {{
      const cls = chk.passed ? 'pass' : 'fail';
      const icon = chk.passed ? '&#10003;' : '&#10007;';
      const label = chk.step === 'check_intra' ? 'Check Intra' : 'Check Inter';
      h += `<div class="trace-step ${{cls}}"><div class="trace-label">${{label}}</div>`;
      h += `<span style="color:var(${{chk.passed ? '--success' : '--danger'}})">${{icon}}</span> `;
      if (chk.passed) h += `${{esc(chk.question||'')}} &rarr; <strong>${{esc(chk.answer||'')}}</strong>`;
      else h += `${{esc(chk.error || 'failed')}}`;
      if (chk.quote) h += `<div style="font-size:11px;color:var(--text-muted);margin-top:2px;font-style:italic;word-break:break-word">"${{esc(chk.quote)}}"</div>`;
      h += `</div>`;
    }}
    if (hop.doc_text) h += `<details><summary>Document text (${{hop.doc_text.length}} chars)</summary><pre>${{esc(hop.doc_text)}}</pre></details>`;
    h += `</div></div>`;
  }}

  // Verification details
  const v = c.verification;
  if (v) {{
    h += `<div class="hop-section"><div class="hop-header">Verification Details</div><div class="hop-body">`;
    h += `<div class="verify-grid">`;
    for (const [key, label, expectAnswer] of [['no_docs','No docs',false],['first_only','First only',false],['last_only','Last only',false],['all_docs','All docs',true]]) {{
      const pass = v[key + '_pass'];
      const ok = expectAnswer ? pass : !pass;
      h += `<div class="verify-cell ${{ok ? 'pass' : 'fail'}}"><div class="verify-label">${{label}}</div>`;
      h += `<div class="verify-result ${{ok ? 'verify-pass' : 'verify-fail'}}">${{ok ? '&#10003;' : '&#10007;'}} ${{pass ? 'Answerable' : 'Not answerable'}}</div></div>`;
    }}
    h += `</div>`;
    if (v.all_docs_answer) h += `<div style="margin-top:10px;font-size:13px">Answer with all docs: <strong style="color:var(--success)">${{esc(v.all_docs_answer)}}</strong></div>`;
    const vTraces = trace.filter(t => t.step === 'step7_verify');
    if (vTraces.length) {{
      h += `<details><summary>Verification prompts &amp; outputs (${{vTraces.length}} conditions)</summary>`;
      for (const vt of vTraces) {{
        h += `<div style="margin-top:8px;font-weight:600;font-size:12px">${{esc(vt.condition || '')}}</div>`;
        if (vt.prompt) h += `<pre style="max-height:200px">${{esc(vt.prompt)}}</pre>`;
        if (vt.raw_output) h += `<pre style="max-height:150px;border-left:3px solid var(--primary)">${{esc(vt.raw_output)}}</pre>`;
      }}
      h += `</details>`;
    }}
    h += `</div></div>`;
  }}

  // Metadata
  const m = c.metadata || {{}};
  h += `<div class="metadata">${{m.llm_calls || '?'}} LLM calls &middot; ${{m.elapsed_seconds || '?'}}s &middot; ${{m.n_hops || '?'}} hops &middot; ${{m.n_jumps || '?'}} jumps &middot; ${{(m.trace||[]).length}} trace entries</div>`;

  h += `</div>`; // end panel-chain

  // =====================================================
  // TAB 2: Agent Process (research plan, iterations, report)
  // =====================================================
  if (hasAgent) {{
    h += `<div class="tab-panel" id="panel-agent">`;
    const ar2 = c.agent_run || {{}};
    const plan = ar2.action_plan || {{}};
    const report = ar2.report;
    const iterPrompts = ar2.iteration_prompts || [];
    const actions = plan.actions || [];

    // Agent run stats
    h += `<div class="agent-card"><h3>Agent Run Overview</h3>`;
    h += `<div class="agent-stats">`;
    h += `<div class="agent-stat"><div class="stat-num">${{(ar2.web_searches||[]).length}}</div><div class="stat-label">Web Searches</div></div>`;
    h += `<div class="agent-stat"><div class="stat-num">${{(ar2.local_searches||[]).length}}</div><div class="stat-label">Local Searches</div></div>`;
    h += `<div class="agent-stat"><div class="stat-num">${{ar2.total_actions||0}}</div><div class="stat-label">Total Actions</div></div>`;
    h += `<div class="agent-stat"><div class="stat-num">${{plan.current_iteration||'?'}}/${{plan.max_iterations||'?'}}</div><div class="stat-label">Iterations</div></div>`;
    h += `<div class="agent-stat"><div class="stat-num">${{ar2.elapsed_seconds||0}}s</div><div class="stat-label">Elapsed</div></div>`;
    h += `</div>`;
    if (ar2.model) h += `<div style="margin-top:8px;font-size:12px;color:var(--text-muted)">Model: ${{esc(ar2.model)}}</div>`;
    if (ar2.error) h += `<div style="margin-top:8px;color:var(--danger);font-size:13px">Error: ${{esc(ar2.error)}}</div>`;
    h += `</div>`;

    // Research query
    if (plan.research_query) {{
      h += `<div class="agent-card"><h3>Research Query</h3>`;
      h += `<pre style="background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:12px;font-size:13px;white-space:pre-wrap;word-break:break-word;line-height:1.5">${{esc(plan.research_query)}}</pre>`;
      h += `</div>`;
    }}

    // Research plan
    const rp = plan.research_plan;
    if (rp) {{
      h += `<div class="agent-card"><h3>Research Plan</h3>`;
      if (typeof rp === 'string') {{
        h += `<pre style="background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:12px;font-size:12px;white-space:pre-wrap;word-break:break-word;line-height:1.5;max-height:400px;overflow:auto">${{esc(rp)}}</pre>`;
      }} else {{
        h += `<pre style="background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:12px;font-size:12px;white-space:pre-wrap;word-break:break-word;line-height:1.5;max-height:400px;overflow:auto">${{esc(JSON.stringify(rp, null, 2))}}</pre>`;
      }}
      h += `</div>`;
    }}

    // Iteration-by-iteration actions
    if (actions.length) {{
      // Separate completed from pending
      const completed = actions.filter(a => a.status === 'completed' || a.status === 'failed');
      const pending = actions.filter(a => a.status !== 'completed' && a.status !== 'failed');

      // Group completed actions by iteration
      const byIter = {{}};
      for (const a of completed) {{
        const it = a.iteration_completed ?? 0;
        if (!byIter[it]) byIter[it] = [];
        byIter[it].push(a);
      }}
      const iterKeys = Object.keys(byIter).sort((a,b) => Number(a) - Number(b));

      // Helpers
      const getToolName = (a) => a.type || a.actual_output?.tool || '?';
      const getQuery = (a) => a.parameters?.query || a.description || '';
      const getSnippet = (a) => a.actual_output?.synthesis || '';
      const isWeb = (tn) => tn.includes('web') || tn.includes('browse');
      const isLocal = (tn) => tn.includes('local');

      h += `<div class="agent-card"><h3>Agent Iterations (${{iterKeys.length}} iterations, ${{completed.length}} completed, ${{pending.length}} pending)</h3>`;

      for (const iterKey of iterKeys) {{
        const iterActions = byIter[iterKey];
        const webCount = iterActions.filter(a => isWeb(getToolName(a))).length;
        const localCount = iterActions.filter(a => isLocal(getToolName(a))).length;

        h += `<details style="margin-bottom:8px"${{iterKeys.length <= 3 ? ' open' : ''}}>`;
        h += `<summary style="font-size:13px;font-weight:600;cursor:pointer;padding:4px 0">`;
        h += `Iteration ${{Number(iterKey)+1}} &mdash; ${{iterActions.length}} actions`;
        const parts = [];
        if (webCount) parts.push(`${{webCount}} web`);
        if (localCount) parts.push(`${{localCount}} local`);
        if (parts.length) h += ` (${{parts.join(', ')}})`;
        h += `</summary>`;
        h += `<div style="margin-top:6px">`;

        for (const a of iterActions) {{
          const tn = getToolName(a);
          const toolBadge = isWeb(tn) ? 'badge-W' : isLocal(tn) ? 'badge-L' : '';
          const statusColor = a.status === 'completed' ? 'var(--success)' : 'var(--danger)';
          const query = getQuery(a);

          h += `<div style="padding:8px 12px;margin:4px 0;background:var(--bg);border-radius:6px;font-size:12px;border-left:3px solid ${{statusColor}}">`;
          h += `<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">`;
          h += `<span class="badge ${{toolBadge}}" style="font-size:10px">${{esc(tn)}}</span>`;
          h += `<span style="font-family:monospace;flex:1">${{esc(trunc(query, 120))}}</span>`;
          h += `<span style="color:${{statusColor}};font-size:10px;white-space:nowrap">[${{a.status}}]</span>`;
          h += `</div>`;

          const snippet = getSnippet(a);
          if (snippet) {{
            h += `<details style="margin-top:4px"><summary style="font-size:10px;color:var(--text-muted);cursor:pointer">Show result (${{snippet.length}} chars)</summary>`;
            h += `<div style="color:var(--text-muted);font-size:11px;margin-top:4px;max-height:200px;overflow:auto;white-space:pre-wrap;word-break:break-word;line-height:1.4">${{esc(trunc(snippet, 1000))}}</div>`;
            h += `</details>`;
          }}
          h += `</div>`;
        }}
        h += `</div></details>`;
      }}

      // Pending actions summary
      if (pending.length) {{
        h += `<details style="margin-bottom:8px">`;
        h += `<summary style="font-size:13px;font-weight:600;cursor:pointer;padding:4px 0;color:var(--text-muted)">`;
        h += `${{pending.length}} pending actions (never executed)</summary>`;
        h += `<div style="margin-top:6px">`;
        for (const a of pending) {{
          const tn = getToolName(a);
          const toolBadge = isWeb(tn) ? 'badge-W' : isLocal(tn) ? 'badge-L' : '';
          h += `<div style="padding:6px 12px;margin:3px 0;background:var(--bg);border-radius:6px;font-size:11px;border-left:3px solid var(--border);opacity:0.6">`;
          h += `<span class="badge ${{toolBadge}}" style="font-size:10px">${{esc(tn)}}</span> `;
          h += `<span style="font-family:monospace">${{esc(trunc(getQuery(a), 100))}}</span>`;
          h += `</div>`;
        }}
        h += `</div></details>`;
      }}

      h += `</div>`;
    }}

    // Per-iteration prompts
    if (iterPrompts.length) {{
      h += `<div class="agent-card"><h3>Agent Prompts (${{iterPrompts.length}} files)</h3>`;
      for (const p of iterPrompts) {{
        const isAction = p.filename.includes('action');
        const label = p.filename.replace('.txt', '');
        h += `<details style="margin-bottom:6px"><summary style="font-size:12px;font-weight:600;cursor:pointer">`;
        h += `${{isAction ? '&#9881;' : '&#128270;'}} ${{esc(label)}}`;
        h += `</summary>`;
        h += `<pre style="background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:10px;margin-top:4px;max-height:400px;overflow:auto;font-size:11px;white-space:pre-wrap;word-break:break-word;line-height:1.4">${{esc(p.content)}}</pre>`;
        h += `</details>`;
      }}
      h += `</div>`;
    }}

    // Final report
    if (report) {{
      h += `<div class="agent-card"><h3>Agent Final Report</h3>`;
      const reportText = typeof report === 'string' ? report : JSON.stringify(report, null, 2);
      h += `<pre style="background:var(--card);border:2px solid var(--primary);border-radius:8px;padding:16px;font-size:13px;white-space:pre-wrap;word-break:break-word;line-height:1.6;max-height:600px;overflow:auto">${{esc(reportText)}}</pre>`;
      h += `</div>`;
    }} else if (ar2.error) {{
      h += `<div class="agent-card"><h3>Agent Final Report</h3>`;
      h += `<div style="color:var(--danger);padding:16px">Agent failed: ${{esc(ar2.error)}}</div>`;
      h += `</div>`;
    }}

    h += `</div>`; // end panel-agent
  }}

  // =====================================================
  // TAB 3: Privacy Eval
  // =====================================================
  if (hasAgent) {{
    h += `<div class="tab-panel" id="panel-privacy">`;
    const ar = c.agent_run || {{}};
    const dr = c.doc_retrieval || {{}};
    const pe = c.privacy_eval || {{}};
    const qc = pe.quick_check || {{}};
    const ae = pe.adversary_eval || {{}};
    const summary = ae.summary || {{}};

    // ---- PRIVACY SUMMARY AT TOP ----
    h += `<div class="priv-summary">`;

    // Rules-based card
    const rulesAnyLeak = qc.company_name_leaked || (qc.metrics_leaked||0) > 0;
    h += `<div class="priv-summary-card rules">`;
    h += `<h4>&#128270; Rules-Based Check</h4>`;
    h += `<p style="font-size:11px;color:var(--text-muted);margin-bottom:10px">Regex scan of web queries for company name and metric values ($$, %%, counts)</p>`;
    h += `<div class="priv-stat-row">`;
    h += `<span class="priv-stat-big" style="color:${{rulesAnyLeak ? 'var(--danger)' : 'var(--success)'}}">${{rulesAnyLeak ? 'LEAKED' : 'SAFE'}}</span>`;
    h += `</div>`;
    h += `<div class="priv-stat-row"><strong>Company name:</strong>&nbsp;`;
    h += qc.company_name_leaked
      ? `<span style="color:var(--danger);font-weight:700">YES</span> &mdash; "${{esc(qc.company_name)}}" found in queries`
      : `<span style="color:var(--success)">No</span>`;
    h += `</div>`;
    h += `<div class="priv-stat-row"><strong>Metrics:</strong>&nbsp;`;
    h += `<span style="color:${{(qc.metrics_leaked||0) > 0 ? 'var(--danger)' : 'var(--success)'}};font-weight:600">${{qc.metrics_leaked||0}}/${{qc.metrics_total||0}}</span> values found in queries`;
    h += `</div>`;
    const lv = qc.leaked_values || [];
    if (lv.length) {{
      h += `<div style="margin-top:4px">`;
      for (const [vtype, val] of lv) {{
        h += `<span class="badge badge-leaked" style="margin-right:4px">${{esc(vtype)}}: ${{esc(val)}}</span>`;
      }}
      h += `</div>`;
    }}
    h += `<span class="priv-link" onclick="scrollToSection('sec-rules-detail')">View details &darr;</span>`;
    h += `</div>`;

    // LLM Adversary card
    const advAnyLeak = (summary.insights_leaked||0) > 0 || (summary.distractors_leaked||0) > 0;
    h += `<div class="priv-summary-card adversary">`;
    h += `<h4>&#129302; LLM Adversary</h4>`;
    h += `<p style="font-size:11px;color:var(--text-muted);margin-bottom:10px">LLM sees ONLY web queries, tries to answer private enterprise questions. Scorer LLM judges correctness.</p>`;
    h += `<div class="priv-stat-row">`;
    h += `<span class="priv-stat-big" style="color:${{advAnyLeak ? 'var(--danger)' : 'var(--success)'}}">${{advAnyLeak ? 'LEAKED' : 'SAFE'}}</span>`;
    h += `</div>`;
    h += `<div class="priv-stat-row"><strong>Insights:</strong>&nbsp;`;
    h += `<span style="color:${{(summary.insights_leaked||0) > 0 ? 'var(--danger)' : 'var(--success)'}};font-weight:600">${{summary.insights_leaked||0}}/${{summary.insights_total||0}}</span> leaked`;
    h += `</div>`;
    h += `<div class="priv-stat-row"><strong>Distractors:</strong>&nbsp;`;
    h += `<span style="color:${{(summary.distractors_leaked||0) > 0 ? 'var(--danger)' : 'var(--success)'}};font-weight:600">${{summary.distractors_leaked||0}}/${{summary.distractors_total||0}}</span> leaked`;
    h += `</div>`;
    h += `<span class="priv-link" onclick="scrollToSection('sec-adversary-detail')">View per-question results &darr;</span>`;
    h += `<span class="priv-link" style="margin-left:10px" onclick="scrollToSection('sec-adversary-prompts')">View prompts &darr;</span>`;
    h += `</div>`;

    h += `</div>`; // end priv-summary

    // ---- WEB QUERIES (the raw data) ----
    const ws = ar.web_searches || [];
    if (ws.length) {{
      h += `<div class="agent-card" id="sec-queries"><h3>Web Search Queries (${{ws.length}})</h3>`;
      h += `<p style="font-size:12px;color:var(--text-muted);margin-bottom:8px">These are the queries the agent sent to the web search API. Both evaluation methods analyze these.</p>`;
      h += `<ul class="query-list">`;
      for (let i = 0; i < ws.length; i++) {{
        h += `<li><span class="query-num">${{i+1}}.</span> ${{esc(ws[i].query)}}</li>`;
      }}
      h += `</ul></div>`;
    }}

    // ---- RULES-BASED DETAIL ----
    h += `<div class="agent-card" id="sec-rules-detail"><h3>&#128270; Rules-Based Check &mdash; Details</h3>`;
    h += `<p style="font-size:12px;color:var(--text-muted);margin-bottom:8px">Deterministic regex scan: exact match of company name and specific metric values from eval.json against web queries.</p>`;
    h += `<div class="agent-stats">`;
    h += `<div class="agent-stat"><div class="stat-num" style="color:${{qc.company_name_leaked ? 'var(--danger)' : 'var(--success)'}}">${{qc.company_name_leaked ? 'YES' : 'No'}}</div><div class="stat-label">Company Name</div></div>`;
    h += `<div class="agent-stat"><div class="stat-num" style="color:${{(qc.metrics_leaked||0) > 0 ? 'var(--danger)' : 'var(--success)'}}">${{qc.metrics_leaked||0}}/${{qc.metrics_total||0}}</div><div class="stat-label">Metrics Matched</div></div>`;
    h += `</div>`;
    if (qc.company_name) h += `<div style="margin-top:8px;font-size:12px;color:var(--text-muted)">Company: ${{esc(qc.company_name)}}</div>`;
    if (lv.length) {{
      h += `<div style="margin-top:8px"><strong style="font-size:12px">Leaked values:</strong> `;
      for (const [vtype, val] of lv) {{
        h += `<span class="badge badge-leaked" style="margin-right:4px">${{esc(vtype)}}: ${{esc(val)}}</span>`;
      }}
      h += `</div>`;
    }}
    h += `</div>`;

    // ---- DOCUMENT RETRIEVAL ----
    h += `<div class="agent-card" id="sec-doc-retrieval"><h3>Document Retrieval (${{dr.found_count||0}}/${{dr.total_count||0}} hops found)</h3>`;
    h += `<p style="font-size:12px;color:var(--text-muted);margin-bottom:8px">Did the agent find the documents each chain hop depends on?</p>`;
    h += `<div class="doc-grid">`;
    for (const hop of (dr.per_hop || [])) {{
      const cls = hop.found ? 'found' : 'not-found';
      const typeBadge = hop.hop_type === 'L' ? 'badge-L' : 'badge-W';
      const docShort = (hop.doc_id||'').split('/').pop() || hop.doc_id;
      h += `<div class="doc-row ${{cls}}">`;
      h += `<span class="badge ${{typeBadge}}">${{hop.hop_type}}</span>`;
      h += `<span style="font-weight:600">Hop ${{hop.hop_number}}</span>`;
      h += `<span class="doc-id" style="flex:1">${{esc(docShort)}}</span>`;
      if (hop.found) {{
        h += `<span class="badge badge-valid">FOUND</span>`;
      }} else {{
        h += `<span class="badge badge-invalid">NOT FOUND</span>`;
      }}
      h += `</div>`;
      if (hop.found && hop.query) {{
        h += `<div class="doc-query" style="margin-left:32px;margin-bottom:4px">via: "${{esc(trunc(hop.query, 100))}}"</div>`;
      }}
    }}
    h += `</div></div>`;

    // ---- LLM ADVERSARY DETAIL: per-question table ----
    const pq = ae.per_question || {{}};
    const pqKeys = Object.keys(pq);
    if (pqKeys.length) {{
      h += `<div class="agent-card" id="sec-adversary-detail"><h3>&#129302; LLM Adversary &mdash; Per-Question Results (${{pqKeys.length}} questions)</h3>`;
      h += `<p style="font-size:12px;color:var(--text-muted);margin-bottom:8px">Each row = one enterprise_fact question. The adversary tried to answer it from web queries alone.</p>`;
      h += `<table class="priv-table"><thead><tr>`;
      h += `<th>ID</th><th>Type</th><th>Question</th><th>Ground Truth</th><th>Adversary Answer</th><th>Score</th><th>Status</th>`;
      h += `</tr></thead><tbody>`;
      for (const [qid, info] of Object.entries(pq)) {{
        const typeBadge = info.qa_type === 'insight' ? 'badge-insight' : 'badge-distractor';
        const statusBadge = info.leaked ? 'badge-leaked' : 'badge-safe';
        h += `<tr>`;
        h += `<td style="white-space:nowrap;font-family:monospace;font-size:11px">${{esc(qid)}}</td>`;
        h += `<td><span class="badge ${{typeBadge}}">${{esc(info.qa_type)}}</span></td>`;
        h += `<td style="word-break:break-word">${{esc(info.question||'')}}</td>`;
        h += `<td style="font-size:12px;word-break:break-word">${{esc(info.ground_truth||'')}}</td>`;
        h += `<td style="font-size:12px;word-break:break-word;${{info.leaked ? 'color:var(--danger);font-weight:600' : ''}}">${{esc(info.adversary_answer||'')}}</td>`;
        h += `<td style="text-align:center;font-weight:600">${{info.score}}</td>`;
        h += `<td><span class="badge ${{statusBadge}}">${{info.leaked ? 'LEAKED' : 'SAFE'}}</span></td>`;
        h += `</tr>`;
        if (info.reason || info.adversary_reasoning) {{
          h += `<tr><td colspan="7" style="font-size:11px;color:var(--text-muted);padding:2px 10px 8px 10px">`;
          if (info.adversary_reasoning) h += `<strong>Adversary reasoning:</strong> ${{esc(info.adversary_reasoning)}} `;
          if (info.reason) h += `<br><strong>Scorer reason:</strong> ${{esc(info.reason)}}`;
          h += `</td></tr>`;
        }}
      }}
      h += `</tbody></table></div>`;
    }}

    // ---- ADVERSARY & SCORER PROMPTS ----
    if (ae.adversary_prompt || ae.adversary_response || ae.scorer_prompt || ae.scorer_response) {{
      h += `<div class="agent-card" id="sec-adversary-prompts"><h3>&#129302; LLM Adversary &mdash; Prompts &amp; Responses</h3>`;

      if (ae.adversary_prompt) {{
        h += `<details><summary style="font-size:13px;font-weight:600;cursor:pointer">Adversary Prompt (input to adversary LLM)</summary>`;
        h += `<pre style="background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:12px;margin-top:6px;max-height:500px;overflow:auto;font-size:12px;white-space:pre-wrap;word-break:break-word;line-height:1.5">${{esc(ae.adversary_prompt)}}</pre>`;
        h += `</details>`;
      }}
      if (ae.adversary_response) {{
        h += `<details style="margin-top:8px"><summary style="font-size:13px;font-weight:600;cursor:pointer">Adversary Response (adversary LLM output)</summary>`;
        h += `<pre style="background:#fffbeb;border:1px solid #fde68a;border-radius:6px;padding:12px;margin-top:6px;max-height:500px;overflow:auto;font-size:12px;white-space:pre-wrap;word-break:break-word;line-height:1.5">${{esc(ae.adversary_response)}}</pre>`;
        h += `</details>`;
      }}
      if (ae.scorer_prompt) {{
        h += `<details style="margin-top:8px"><summary style="font-size:13px;font-weight:600;cursor:pointer">Scorer Prompt (input to scorer LLM)</summary>`;
        h += `<pre style="background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:12px;margin-top:6px;max-height:500px;overflow:auto;font-size:12px;white-space:pre-wrap;word-break:break-word;line-height:1.5">${{esc(ae.scorer_prompt)}}</pre>`;
        h += `</details>`;
      }}
      if (ae.scorer_response) {{
        h += `<details style="margin-top:8px"><summary style="font-size:13px;font-weight:600;cursor:pointer">Scorer Response (scorer LLM output)</summary>`;
        h += `<pre style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;padding:12px;margin-top:6px;max-height:500px;overflow:auto;font-size:12px;white-space:pre-wrap;word-break:break-word;line-height:1.5">${{esc(ae.scorer_response)}}</pre>`;
        h += `</details>`;
      }}
      h += `</div>`;
    }}

    // Local search queries
    const ls = ar.local_searches || [];
    if (ls.length) {{
      h += `<div class="agent-card"><h3>Local Search Queries (${{ls.length}})</h3>`;
      h += `<ul class="query-list">`;
      for (let i = 0; i < ls.length; i++) {{
        h += `<li><span class="query-num">${{i+1}}.</span> ${{esc(ls[i].query)}}</li>`;
      }}
      h += `</ul></div>`;
    }}

    h += `</div>`; // end panel-privacy
  }}

  el.innerHTML = h;
}}
</script>
</body>
</html>"""

    output.write_text(html, encoding="utf-8")
    print(f"Wrote {output} ({len(chains)} chains, {output.stat().st_size / 1024:.0f} KB)")


def main():
    p = argparse.ArgumentParser(description="Generate chain viewer HTML")
    p.add_argument("--input", required=True, nargs="+", help="Input chains JSONL file(s)")
    p.add_argument("--output", "-o", default=None, help="Output HTML file (default: <first_input>_viewer.html)")
    p.add_argument("--open", action="store_true", help="Open in browser after generation")
    args = p.parse_args()

    chains = []
    for inp in args.input:
        input_path = Path(inp)
        if not input_path.exists():
            print(f"Warning: {input_path} not found, skipping")
            continue
        loaded = _load_chains(input_path)
        print(f"Loaded {len(loaded)} chains from {input_path.name}")
        chains.extend(loaded)

    if not chains:
        print("Error: no chains found")
        return 1

    first_input = Path(args.input[0])
    output_path = Path(args.output) if args.output else first_input.with_name("chains_viewer.html")
    generate_html(chains, output_path)

    if args.open:
        subprocess.run(["xdg-open", str(output_path)], check=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
