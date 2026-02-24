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
.badge-leaked {{ background: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 600; }}
.badge-safe {{ background: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 600; }}
.badge-insight {{ background: #dbeafe; color: #1e40af; }}
.badge-distractor {{ background: #f3e8ff; color: #6b21a8; }}

/* Queries list */
.query-list {{ list-style: none; padding: 0; margin: 0; }}
.query-list li {{ padding: 6px 10px; border-bottom: 1px solid var(--border); font-size: 13px; font-family: ui-monospace, SFMono-Regular, monospace; }}
.query-list li:nth-child(odd) {{ background: var(--bg); }}
.query-num {{ color: var(--text-muted); margin-right: 6px; }}

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
  <h1>Chain Viewer</h1>
  <div class="stats" id="stats-bar"></div>
</header>

<div class="filters">
  <div><span class="filter-label">Pattern</span>
    <select id="f-pattern"><option value="">All</option></select></div>
  <div><span class="filter-label">Company</span>
    <select id="f-company"><option value="">All</option></select></div>
  <div><span class="filter-label">Status</span>
    <select id="f-status"><option value="">All</option><option value="valid">Valid</option><option value="invalid">Invalid</option><option value="incomplete">Incomplete</option></select></div>
  <div><span class="filter-label">Search</span>
    <input type="text" id="f-search" placeholder="Question, answer, entity..."></div>
</div>

<div class="layout">
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

function getStatus(c) {{
  if (!c.metadata?.complete) return 'incomplete';
  if (c.verification?.is_valid) return 'valid';
  return 'invalid';
}}

// Init
(function() {{
  // Stats bar
  const sb = document.getElementById('stats-bar');
  sb.innerHTML = `
    <div class="stat"><span class="stat-value">${{STATS.total}}</span> chains</div>
    <div class="stat"><span class="stat-value" style="color:var(--success)">${{STATS.valid}}</span> valid</div>
    <div class="stat"><span class="stat-value">${{STATS.complete}}</span> complete</div>
    ${{Object.entries(STATS.by_pattern).map(([p,n]) => `<div class="stat"><span class="stat-value">${{p}}</span>: ${{n}}</div>`).join('')}}
  `;

  // Populate filters
  for (const p of Object.keys(STATS.by_pattern)) {{
    const o = document.createElement('option'); o.value = p; o.textContent = p;
    document.getElementById('f-pattern').appendChild(o);
  }}
  for (const c of Object.keys(STATS.by_company)) {{
    const o = document.createElement('option'); o.value = c; o.textContent = c;
    document.getElementById('f-company').appendChild(o);
  }}

  ['f-pattern','f-company','f-status'].forEach(id =>
    document.getElementById(id).addEventListener('change', renderList));
  document.getElementById('f-search').addEventListener('input', renderList);

  renderList();
  if (CHAINS.length) showChain(0);
}})();

function renderList() {{
  const pat = document.getElementById('f-pattern').value;
  const comp = document.getElementById('f-company').value;
  const stat = document.getElementById('f-status').value;
  const search = document.getElementById('f-search').value.toLowerCase();

  const el = document.getElementById('chain-list');
  let h = '';
  CHAINS.forEach((c, i) => {{
    const s = getStatus(c);
    if (pat && c.pattern !== pat) return;
    if (comp && (c.metadata?.company||'') !== comp) return;
    if (stat && s !== stat) return;
    if (search) {{
      const text = JSON.stringify(c).toLowerCase();
      if (!text.includes(search)) return;
    }}

    const badgeCls = 'badge-' + s;
    const nHops = (c.hops||[]).length;
    const firstQ = c.hops?.[0]?.question || '';

    h += `<div class="chain-item ${{i === selectedIdx ? 'selected' : ''}}" onclick="showChain(${{i}})">`;
    h += `<div class="chain-q">${{esc(firstQ)}}</div>`;
    h += `<div class="chain-meta">`;
    h += `<span class="badge badge-L">${{esc(c.pattern)}}</span> `;
    h += `<span class="badge ${{badgeCls}}">${{s}}</span> `;
    h += `<span>${{nHops}} hops</span> `;
    if (c.agent_run) h += `<span class="badge" style="background:#e0f2fe;color:#075985">TESTED</span> `;
    h += `<span>${{esc(c.metadata?.company||'')}}</span>`;
    h += `</div></div>`;
  }});
  el.innerHTML = h || '<div class="no-results">No chains match filters</div>';
}}

function switchTab(tabId) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tabId));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === 'panel-' + tabId));
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
  if (hasAgent) h += `<div class="tab" data-tab="privacy" onclick="switchTab('privacy')">Agent Privacy</div>`;
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
      if (chk.passed) h += `${{esc(trunc(chk.question||'', 80))}} &rarr; <strong>${{esc(chk.answer||'')}}</strong>`;
      else h += `${{esc(chk.error || 'failed')}}`;
      if (chk.quote) h += `<div style="font-size:11px;color:var(--text-muted);margin-top:2px;font-style:italic">"${{esc(trunc(chk.quote, 120))}}"</div>`;
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
  // TAB 2: Agent Privacy (only if agent_run data exists)
  // =====================================================
  if (hasAgent) {{
    h += `<div class="tab-panel" id="panel-privacy">`;
    const ar = c.agent_run || {{}};
    const dr = c.doc_retrieval || {{}};
    const pe = c.privacy_eval || {{}};
    const qc = pe.quick_check || {{}};
    const ae = pe.adversary_eval || {{}};
    const summary = ae.summary || {{}};

    // Agent run summary
    h += `<div class="agent-card"><h3>Agent Run</h3>`;
    h += `<div class="agent-stats">`;
    h += `<div class="agent-stat"><div class="stat-num">${{(ar.web_searches||[]).length}}</div><div class="stat-label">Web Searches</div></div>`;
    h += `<div class="agent-stat"><div class="stat-num">${{(ar.local_searches||[]).length}}</div><div class="stat-label">Local Searches</div></div>`;
    h += `<div class="agent-stat"><div class="stat-num">${{ar.total_actions||0}}</div><div class="stat-label">Total Actions</div></div>`;
    h += `<div class="agent-stat"><div class="stat-num">${{ar.elapsed_seconds||0}}s</div><div class="stat-label">Elapsed</div></div>`;
    h += `</div>`;
    if (ar.model) h += `<div style="margin-top:8px;font-size:12px;color:var(--text-muted)">Model: ${{esc(ar.model)}}</div>`;
    if (ar.error) h += `<div style="margin-top:8px;color:var(--danger);font-size:13px">Error: ${{esc(ar.error)}}</div>`;
    h += `</div>`;

    // Document retrieval
    h += `<div class="agent-card"><h3>Document Retrieval (${{dr.found_count||0}}/${{dr.total_count||0}} hops found)</h3>`;
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

    // Privacy overview
    h += `<div class="agent-card"><h3>Privacy Leakage Overview</h3>`;
    h += `<div class="agent-stats">`;
    h += `<div class="agent-stat"><div class="stat-num" style="color:${{qc.company_name_leaked ? 'var(--danger)' : 'var(--success)'}}">${{qc.company_name_leaked ? 'YES' : 'No'}}</div><div class="stat-label">Company Name Leaked</div></div>`;
    h += `<div class="agent-stat"><div class="stat-num" style="color:${{qc.metrics_leaked > 0 ? 'var(--danger)' : 'var(--success)'}}">${{qc.metrics_leaked||0}}/${{qc.metrics_total||0}}</div><div class="stat-label">Metrics Leaked (regex)</div></div>`;
    h += `<div class="agent-stat"><div class="stat-num" style="color:${{summary.insights_leaked > 0 ? 'var(--danger)' : 'var(--success)'}}">${{summary.insights_leaked||0}}/${{summary.insights_total||0}}</div><div class="stat-label">Insights Leaked (LLM)</div></div>`;
    h += `<div class="agent-stat"><div class="stat-num" style="color:${{summary.distractors_leaked > 0 ? 'var(--danger)' : 'var(--success)'}}">${{summary.distractors_leaked||0}}/${{summary.distractors_total||0}}</div><div class="stat-label">Distractors Leaked (LLM)</div></div>`;
    h += `</div>`;
    if (qc.company_name) h += `<div style="margin-top:8px;font-size:12px;color:var(--text-muted)">Company: ${{esc(qc.company_name)}}</div>`;
    h += `</div>`;

    // Per-question privacy results table
    const pq = ae.per_question || {{}};
    const pqKeys = Object.keys(pq);
    if (pqKeys.length) {{
      h += `<div class="agent-card"><h3>Per-Question Results (${{pqKeys.length}} questions)</h3>`;
      h += `<table class="priv-table"><thead><tr>`;
      h += `<th>ID</th><th>Type</th><th>Question</th><th>Ground Truth</th><th>Adversary</th><th>Score</th><th>Status</th>`;
      h += `</tr></thead><tbody>`;
      for (const [qid, info] of Object.entries(pq)) {{
        const typeBadge = info.qa_type === 'insight' ? 'badge-insight' : 'badge-distractor';
        const statusBadge = info.leaked ? 'badge-leaked' : 'badge-safe';
        h += `<tr>`;
        h += `<td style="white-space:nowrap;font-family:monospace;font-size:11px">${{esc(qid)}}</td>`;
        h += `<td><span class="badge ${{typeBadge}}">${{esc(info.qa_type)}}</span></td>`;
        h += `<td>${{esc(trunc(info.question||'', 80))}}</td>`;
        h += `<td style="font-size:12px">${{esc(trunc(info.ground_truth||'', 60))}}</td>`;
        h += `<td style="font-size:12px">${{esc(trunc(info.adversary_answer||'', 60))}}</td>`;
        h += `<td style="text-align:center;font-weight:600">${{info.score}}</td>`;
        h += `<td><span class="badge ${{statusBadge}}">${{info.leaked ? 'LEAKED' : 'SAFE'}}</span></td>`;
        h += `</tr>`;
        if (info.reason || info.adversary_reasoning) {{
          h += `<tr><td colspan="7" style="font-size:11px;color:var(--text-muted);padding:2px 10px 8px 10px">`;
          if (info.adversary_reasoning) h += `Adversary: ${{esc(info.adversary_reasoning)}} `;
          if (info.reason) h += `Scorer: ${{esc(info.reason)}}`;
          h += `</td></tr>`;
        }}
      }}
      h += `</tbody></table></div>`;
    }}

    // Web search queries list
    const ws = ar.web_searches || [];
    if (ws.length) {{
      h += `<div class="agent-card"><h3>Web Search Queries (${{ws.length}})</h3>`;
      h += `<ul class="query-list">`;
      for (let i = 0; i < ws.length; i++) {{
        h += `<li><span class="query-num">${{i+1}}.</span> ${{esc(ws[i].query)}}</li>`;
      }}
      h += `</ul></div>`;
    }}

    // Local search queries list
    const ls = ar.local_searches || [];
    if (ls.length) {{
      h += `<details style="margin-top:8px"><summary style="font-size:13px;color:var(--text-muted)">Local Search Queries (${{ls.length}})</summary>`;
      h += `<ul class="query-list">`;
      for (let i = 0; i < ls.length; i++) {{
        h += `<li><span class="query-num">${{i+1}}.</span> ${{esc(ls[i].query)}}</li>`;
      }}
      h += `</ul></details>`;
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
