#!/usr/bin/env python3
"""Chain privacy comparison viewer — interactive HTML for dual-adversary results.

Shows per-chain:
  - Agent answers vs ground truth
  - Privacy eval with chain/retrieved secret split
  - Side-by-side adversary model comparison with correlation stats
  - Document retrieval, actions, report

Usage:
    python -m making_dataset_2.view_chain_privacy --input results.jsonl
    python -m making_dataset_2.view_chain_privacy --input results.jsonl --slim
    python -m making_dataset_2.view_chain_privacy --input results.jsonl --open
"""
from __future__ import annotations

import argparse
import json
import subprocess
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


def _safe_json_for_html(obj) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    return s.replace("</", "<\\/").replace("<!--", "<\\!--")


def _slim_results(results: list[dict]) -> list[dict]:
    """Strip large fields for a smaller download. Keeps everything the viewer needs."""
    slim = []
    for r in results:
        r2 = {k: v for k, v in r.items() if k != "agent_run"}
        ar = r.get("agent_run") or {}
        ar2 = {k: v for k, v in ar.items() if k != "iteration_prompts"}
        # Trim action outputs to synthesis only
        ap = ar2.get("action_plan") or {}
        if ap.get("actions"):
            trimmed = []
            for a in ap["actions"]:
                a2 = {k: a.get(k) for k in ("type", "parameters", "status", "execution_time",
                       "iteration_completed", "created_from_research_step", "description")}
                ao = a.get("actual_output") or {}
                a2["actual_output"] = {"synthesis": ao.get("synthesis") or ao.get("summary") or ""}
                trimmed.append(a2)
            ar2["action_plan"] = {**ap, "actions": trimmed}
        r2["agent_run"] = ar2
        # Strip metadata trace
        meta = r2.get("metadata")
        if isinstance(meta, dict) and "trace" in meta:
            r2["metadata"] = {k: v for k, v in meta.items() if k != "trace"}
        # Truncate adversary prompts/responses
        for ae in (r2.get("adversary_evals") or {}).values():
            if not isinstance(ae, dict):
                continue
            for big_key in ("adversary_prompt", "adversary_response", "scorer_prompt", "scorer_response"):
                val = ae.get(big_key)
                if isinstance(val, str) and len(val) > 3000:
                    ae[big_key] = val[:3000] + "\n... (truncated in slim mode)"
        slim.append(r2)
    return slim


def generate_html(results: list[dict], summary: dict | None, output: Path, slim: bool = False) -> None:
    if slim:
        results = _slim_results(results)
    results_json = _safe_json_for_html(results)
    summary_json = _safe_json_for_html(summary) if summary else "null"

    # Detect adversary model names from results
    adv_models = set()
    for r in results:
        for k in (r.get("adversary_evals") or {}):
            adv_models.add(k)
    adv_models = sorted(adv_models)
    adv_models_json = _safe_json_for_html(adv_models)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chain Privacy Comparison{" (slim)" if slim else ""}</title>
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
  --adv1: #6366f1; --adv2: #d946ef;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        background: var(--bg); color: var(--text); line-height: 1.6; font-size: 14px; }}

.container {{ max-width: 1400px; margin: 0 auto; padding: 20px 16px; }}

header {{ background: var(--card); padding: 16px 24px; border-bottom: 1px solid var(--border);
          position: sticky; top: 0; z-index: 100; }}
header h1 {{ font-size: 15px; font-weight: 600; letter-spacing: -0.01em; color: var(--text); }}

.summary-grid {{ display: flex; gap: 10px; margin: 16px 0; flex-wrap: wrap; }}
.stat-card {{ flex: 1; min-width: 90px; background: var(--card); border: 1px solid var(--border);
              border-radius: var(--radius); padding: 12px; text-align: center; box-shadow: var(--shadow-sm); }}
.stat-card .label {{ color: var(--text-muted); font-size: 10px; text-transform: uppercase;
                      letter-spacing: 0.05em; margin-top: 2px; }}
.stat-card .value {{ font-size: 20px; font-weight: 700; letter-spacing: -0.02em; color: var(--text); }}
.stat-card .value.green {{ color: var(--success); }}
.stat-card .value.red {{ color: var(--danger); }}
.stat-card .value.yellow {{ color: var(--warning); }}
.stat-card .sub {{ font-size: 10px; color: var(--text-muted); margin-top: 2px; }}

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

.tabs {{ display: flex; gap: 0; border-bottom: 1px solid var(--border); margin-bottom: 16px; }}
.tab {{ padding: 8px 16px; cursor: pointer; color: var(--text-muted); border-bottom: 2px solid transparent;
        font-size: 12px; font-weight: 500; transition: all 0.15s; }}
.tab:hover {{ color: var(--text); background: var(--primary-light); }}
.tab.active {{ color: var(--primary); border-bottom-color: var(--primary); }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

.section {{ margin: 10px 0; background: var(--card); border: 1px solid var(--border);
            border-radius: 8px; overflow: hidden; }}
.section-header {{ padding: 9px 14px; background: var(--bg); border-bottom: 1px solid var(--border-light);
                    cursor: pointer; font-weight: 500; font-size: 12px; color: var(--text-secondary);
                    display: flex; justify-content: space-between; align-items: center;
                    transition: background 0.12s; }}
.section-header:hover {{ background: #f0f1f3; }}
.section-body {{ display: none; padding: 12px; }}
.section-body.open {{ display: block; }}

table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 8px 0; }}
th {{ text-align: left; padding: 8px 10px; background: var(--bg); color: var(--text-muted);
      border: 1px solid var(--border-light); font-weight: 500; font-size: 11px;
      text-transform: uppercase; letter-spacing: 0.04em; }}
td {{ padding: 8px 10px; border: 1px solid var(--border-light); vertical-align: top; }}
tr:hover td {{ background: #fafbfc; }}
.leaked-row td {{ background: var(--danger-bg) !important; }}
.correct-row td {{ background: var(--success-bg) !important; }}

.action {{ margin: 6px 0; padding: 10px 14px; border-left: 3px solid var(--border-light);
           background: var(--bg); border-radius: 0 6px 6px 0; font-size: 13px; }}
.action.web_search {{ border-left-color: var(--web); }}
.action.local_document_search {{ border-left-color: var(--local); }}
.action .type {{ font-weight: 600; color: var(--primary); font-size: 12px; }}
.action .query {{ color: var(--text); margin: 4px 0; }}
.action .meta {{ color: var(--text-muted); font-size: 11px; }}
.action .result {{ margin-top: 6px; padding: 8px 10px; background: var(--card);
                   border: 1px solid var(--border-light); border-radius: 6px; max-height: 200px;
                   overflow-y: auto; font-size: 12px; white-space: pre-wrap;
                   word-break: break-word; font-family: 'SF Mono', 'Fira Code', ui-monospace, monospace; }}

.prompt-box {{ background: var(--bg); border: 1px solid var(--border-light); border-radius: 6px;
               padding: 12px; max-height: 400px; overflow-y: auto; font-size: 12px;
               white-space: pre-wrap; word-break: break-word; color: var(--text-muted);
               font-family: 'SF Mono', 'Fira Code', ui-monospace, monospace; }}

.report-text {{ background: var(--bg); border: 1px solid var(--border-light); border-radius: 8px;
                padding: 16px; max-height: 500px; overflow-y: auto; font-size: 13px;
                white-space: pre-wrap; word-break: break-word; line-height: 1.7; }}

.filters {{ display: flex; gap: 6px; flex-wrap: wrap; padding: 10px 24px;
            background: var(--card); border-bottom: 1px solid var(--border); }}
.filter-btn {{ padding: 5px 14px; border-radius: 20px; border: 1px solid var(--border);
               background: var(--card); color: var(--text-secondary); cursor: pointer;
               font-size: 12px; font-weight: 500; transition: all 0.15s; }}
.filter-btn:hover {{ background: var(--bg); border-color: #d1d5db; }}
.filter-btn.active {{ background: var(--primary); color: #fff; border-color: var(--primary); }}

.arrow {{ transition: transform 0.2s; display: inline-block; font-size: 10px; }}
.arrow.open {{ transform: rotate(90deg); }}

/* Adversary comparison - stacked vertically for more width */
.adv-compare {{ display: flex; flex-direction: column; gap: 16px; }}
.adv-panel {{ border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
.adv-panel-header {{ padding: 8px 14px; background: var(--bg); border-bottom: 1px solid var(--border-light);
                      font-weight: 600; font-size: 12px; }}
.adv-panel-header.model-0 {{ border-left: 3px solid var(--adv1); }}
.adv-panel-header.model-1 {{ border-left: 3px solid var(--adv2); }}
.adv-panel-body {{ padding: 12px; overflow-x: auto; }}

/* Comparison table */
.cmp-table {{ min-width: 900px; }}
.cmp-table th.model-0 {{ border-bottom: 2px solid var(--adv1); }}
.cmp-table th.model-1 {{ border-bottom: 2px solid var(--adv2); }}

/* Secret source badges */
.badge.chain-secret {{ background: #dbeafe; color: #1d4ed8; }}
.badge.retrieved-secret {{ background: #fce7f3; color: #be185d; }}
.badge.l-hop {{ background: #f3f4f6; color: var(--text); }}

/* Correlation section */
.corr-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin: 12px 0; }}
.corr-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }}
.corr-card h3 {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px; }}
.corr-bar {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
.corr-bar .bar {{ height: 14px; border-radius: 4px; min-width: 2px; }}
.corr-bar .lbl {{ font-size: 11px; color: var(--text-secondary); min-width: 80px; }}
.corr-bar .val {{ font-size: 12px; font-weight: 600; }}

/* Reasoning tooltip / expandable */
.reasoning-cell {{ position: relative; cursor: help; }}
.reasoning-full {{ display: none; position: absolute; z-index: 50; left: 0; top: 100%;
                    background: var(--card); border: 1px solid var(--border); border-radius: 6px;
                    padding: 10px; font-size: 12px; max-width: 400px; min-width: 250px;
                    white-space: pre-wrap; word-break: break-word; box-shadow: var(--shadow);
                    color: var(--text); line-height: 1.5; }}
.reasoning-cell:hover .reasoning-full {{ display: block; }}
</style>
</head>
<body>

<header>
  <h1>Chain Privacy Comparison{" (slim)" if slim else ""}</h1>
</header>
<div style="background:var(--card);border-bottom:1px solid var(--border);padding:0 24px 10px;">
  <div class="filters" id="filters-pattern"></div>
  <div class="filters" id="filters-task"></div>
  <div class="filters" id="filters-status"></div>
</div>
<div class="container">
<div id="summary"></div>
<div id="correlation"></div>
<div id="chains"></div>
</div>

<script>
const results = {results_json};
const summary = {summary_json};
const advModels = {adv_models_json};
const isSlim = {'true' if slim else 'false'};

const filterState = {{ pattern: 'all', task: 'all', status: 'all' }};

function esc(s) {{
    const d = document.createElement('div');
    d.textContent = String(s != null ? s : '');
    return d.innerHTML;
}}

function shortModel(m) {{
    if (!m) return '?';
    const parts = m.split('/');
    return parts[parts.length - 1];
}}

function renderSummary() {{
    const el = document.getElementById('summary');

    // Compute stats from data
    const total = results.length;
    const errors = results.filter(r => (r.agent_run||{{}}).error).length;
    const finalCorrect = results.filter(r => (r.answer_eval||{{}}).final_correct).length;
    const hopAccs = results.map(r => (r.answer_eval||{{}}).hop_accuracy).filter(h => h != null);
    const avgHopAcc = hopAccs.length ? hopAccs.reduce((a,b)=>a+b,0)/hopAccs.length : 0;
    const companyLeaked = results.filter(r => (r.privacy_eval||{{}}).company_name_leaked).length;
    const chainSecretsLeaked = results.reduce((s,r) => s + ((r.privacy_eval||{{}}).chain_secrets_leaked||0), 0);
    const chainSecretsTotal = results.reduce((s,r) => s + ((r.privacy_eval||{{}}).chain_secrets_total||0), 0);
    const retSecretsLeaked = results.reduce((s,r) => s + ((r.privacy_eval||{{}}).retrieved_secrets_leaked||0), 0);
    const retSecretsTotal = results.reduce((s,r) => s + ((r.privacy_eval||{{}}).retrieved_secrets_total||0), 0);
    const times = results.map(r => (r.agent_run||{{}}).elapsed_seconds).filter(t => t > 0);
    const avgTime = times.length ? times.reduce((a,b)=>a+b,0)/times.length : 0;

    // Aggregate adversary stats per model
    const advStats = {{}};
    for (const m of advModels) advStats[m] = {{ leaked: 0, total: 0, taskPredicted: 0 }};
    for (const r of results) {{
        for (const [m, ae] of Object.entries(r.adversary_evals || {{}})) {{
            if (!advStats[m]) advStats[m] = {{ leaked: 0, total: 0, taskPredicted: 0 }};
            const s = (ae || {{}}).summary || {{}};
            advStats[m].leaked += s.total_leaked || 0;
            advStats[m].total += s.total_questions || 0;
            if (s.task_predicted) advStats[m].taskPredicted++;
        }}
    }}

    let advCards = '';
    advModels.forEach((m, i) => {{
        const st = advStats[m] || {{}};
        const rate = st.total ? (st.leaked / st.total * 100).toFixed(1) : '0.0';
        advCards += `
        <div class="stat-card" style="border-top:3px solid var(--adv${{i+1}})">
            <div class="label">${{esc(shortModel(m))}}</div>
            <div class="value ${{st.leaked > 0 ? 'red' : 'green'}}">${{st.leaked}}/${{st.total}}</div>
            <div class="sub">${{rate}}% | ${{st.taskPredicted}} tasks predicted</div>
        </div>`;
    }});

    el.innerHTML = `
    <div class="summary-grid">
        <div class="stat-card">
            <div class="label">Chains</div>
            <div class="value">${{total}}</div>
            <div class="sub">${{errors}} errors</div>
        </div>
        <div class="stat-card">
            <div class="label">Hop Accuracy</div>
            <div class="value ${{avgHopAcc > 0.5 ? 'green' : 'red'}}">
                ${{(avgHopAcc * 100).toFixed(1)}}%</div>
            <div class="sub">${{finalCorrect}}/${{total}} final correct</div>
        </div>
        <div class="stat-card">
            <div class="label">Company Leaked</div>
            <div class="value ${{companyLeaked > 0 ? 'yellow' : 'green'}}">
                ${{companyLeaked}}</div>
        </div>
        <div class="stat-card">
            <div class="label">Chain Secrets (regex)</div>
            <div class="value ${{chainSecretsLeaked > 0 ? 'red' : 'green'}}">
                ${{chainSecretsLeaked}}/${{chainSecretsTotal}}</div>
        </div>
        <div class="stat-card">
            <div class="label">Retrieved Secrets (regex)</div>
            <div class="value ${{retSecretsLeaked > 0 ? 'red' : 'green'}}">
                ${{retSecretsLeaked}}/${{retSecretsTotal}}</div>
        </div>
        ${{advCards}}
        <div class="stat-card">
            <div class="label">Avg Time</div>
            <div class="value" style="font-size:16px">${{avgTime.toFixed(0)}}s</div>
        </div>
    </div>`;
}}

function renderCorrelation() {{
    const el = document.getElementById('correlation');
    if (advModels.length < 2) {{ el.innerHTML = ''; return; }}
    const m0 = advModels[0], m1 = advModels[1];
    const sm0 = shortModel(m0), sm1 = shortModel(m1);

    // Per-chain: did each model find any leak?
    let bothLeak = 0, only0 = 0, only1 = 0, neither = 0;
    let bothTask = 0, onlyTask0 = 0, onlyTask1 = 0, neitherTask = 0;

    // Per-question agreement
    let qAgree = 0, qDisagree = 0, qBothLeak = 0, qOnly0 = 0, qOnly1 = 0, qNeither = 0;
    let scoreCorr = []; // pairs of [score0, score1]

    for (const r of results) {{
        const ae0 = (r.adversary_evals||{{}})[m0] || {{}};
        const ae1 = (r.adversary_evals||{{}})[m1] || {{}};
        const s0 = ae0.summary || {{}};
        const s1 = ae1.summary || {{}};
        const l0 = (s0.total_leaked||0) > 0;
        const l1 = (s1.total_leaked||0) > 0;
        if (l0 && l1) bothLeak++;
        else if (l0) only0++;
        else if (l1) only1++;
        else neither++;
        const t0 = !!s0.task_predicted;
        const t1 = !!s1.task_predicted;
        if (t0 && t1) bothTask++;
        else if (t0) onlyTask0++;
        else if (t1) onlyTask1++;
        else neitherTask++;

        const pq0 = ae0.per_question || [];
        const pq1 = ae1.per_question || [];
        const maxQ = Math.max(pq0.length, pq1.length);
        for (let i = 0; i < maxQ; i++) {{
            const q0 = pq0[i] || {{}};
            const q1 = pq1[i] || {{}};
            const ql0 = !!q0.leaked;
            const ql1 = !!q1.leaked;
            if (ql0 === ql1) qAgree++; else qDisagree++;
            if (ql0 && ql1) qBothLeak++;
            else if (ql0) qOnly0++;
            else if (ql1) qOnly1++;
            else qNeither++;
            if (q0.score != null && q1.score != null) {{
                scoreCorr.push([q0.score, q1.score]);
            }}
        }}
    }}

    const totalChains = bothLeak + only0 + only1 + neither;
    const totalQ = qBothLeak + qOnly0 + qOnly1 + qNeither;
    const agreeRate = totalQ ? ((qAgree / (qAgree + qDisagree)) * 100).toFixed(1) : '0';

    // Pearson correlation of scores
    let pearson = 'N/A';
    if (scoreCorr.length > 2) {{
        const n = scoreCorr.length;
        const mx = scoreCorr.reduce((s,p)=>s+p[0],0)/n;
        const my = scoreCorr.reduce((s,p)=>s+p[1],0)/n;
        let num=0, dx=0, dy=0;
        for (const [x,y] of scoreCorr) {{
            num += (x-mx)*(y-my);
            dx += (x-mx)*(x-mx);
            dy += (y-my)*(y-my);
        }}
        const denom = Math.sqrt(dx*dy);
        pearson = denom > 0 ? (num/denom).toFixed(3) : 'N/A';
    }}

    function bar(val, total, color) {{
        const pct = total > 0 ? (val/total*100) : 0;
        return `<div class="corr-bar">
            <div class="bar" style="width:${{Math.max(pct, 1)}}%;background:${{color}}"></div>
            <span class="val">${{val}}</span>
            <span class="lbl">(${{pct.toFixed(1)}}%)</span>
        </div>`;
    }}

    el.innerHTML = `
    <div class="section" style="margin:16px 0">
        <div class="section-header" onclick="toggleSection(this)">
            <span><span class="arrow open">&#9654;</span> Adversary Model Comparison: ${{esc(sm0)}} vs ${{esc(sm1)}}</span>
            <span style="color:#6c757d;font-size:0.85em">Agreement: ${{agreeRate}}% | Pearson r=${{pearson}}</span>
        </div>
        <div class="section-body open">
            <div class="corr-grid">
                <div class="corr-card">
                    <h3>Per-Chain Leak Detection (${{totalChains}} chains)</h3>
                    <div class="corr-bar"><span class="lbl">Both leak</span>${{bar(bothLeak, totalChains, 'var(--danger)')}}</div>
                    <div class="corr-bar"><span class="lbl">${{esc(sm0)}} only</span>${{bar(only0, totalChains, 'var(--adv1)')}}</div>
                    <div class="corr-bar"><span class="lbl">${{esc(sm1)}} only</span>${{bar(only1, totalChains, 'var(--adv2)')}}</div>
                    <div class="corr-bar"><span class="lbl">Neither</span>${{bar(neither, totalChains, 'var(--success)')}}</div>
                </div>
                <div class="corr-card">
                    <h3>Per-Question Leak Agreement (${{totalQ}} questions)</h3>
                    <div class="corr-bar"><span class="lbl">Both leak</span>${{bar(qBothLeak, totalQ, 'var(--danger)')}}</div>
                    <div class="corr-bar"><span class="lbl">${{esc(sm0)}} only</span>${{bar(qOnly0, totalQ, 'var(--adv1)')}}</div>
                    <div class="corr-bar"><span class="lbl">${{esc(sm1)}} only</span>${{bar(qOnly1, totalQ, 'var(--adv2)')}}</div>
                    <div class="corr-bar"><span class="lbl">Neither</span>${{bar(qNeither, totalQ, 'var(--success)')}}</div>
                </div>
                <div class="corr-card">
                    <h3>Task Prediction (${{totalChains}} chains)</h3>
                    <div class="corr-bar"><span class="lbl">Both predict</span>${{bar(bothTask, totalChains, 'var(--warning)')}}</div>
                    <div class="corr-bar"><span class="lbl">${{esc(sm0)}} only</span>${{bar(onlyTask0, totalChains, 'var(--adv1)')}}</div>
                    <div class="corr-bar"><span class="lbl">${{esc(sm1)}} only</span>${{bar(onlyTask1, totalChains, 'var(--adv2)')}}</div>
                    <div class="corr-bar"><span class="lbl">Neither</span>${{bar(neitherTask, totalChains, 'var(--success)')}}</div>
                </div>
                <div class="corr-card">
                    <h3>Score Correlation</h3>
                    <div style="font-size:13px;margin-bottom:6px">
                        <b>Pearson r = ${{pearson}}</b> (${{scoreCorr.length}} question pairs)
                    </div>
                    <div style="font-size:12px;color:var(--text-secondary)">
                        Agreement rate: <b>${{agreeRate}}%</b><br>
                        Disagree on <b>${{qDisagree}}</b> questions<br>
                        ${{esc(sm0)}} finds ${{qOnly0 + qBothLeak}} leaks,
                        ${{esc(sm1)}} finds ${{qOnly1 + qBothLeak}} leaks
                    </div>
                </div>
            </div>

            <div style="margin-top:12px">
                <h3 style="font-size:12px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.04em;margin-bottom:8px">
                    Disagreements (one model detects leak, other doesn't)
                </h3>
                <div id="disagreements-table"></div>
            </div>
        </div>
    </div>`;

    // Build disagreements table
    const disagRows = [];
    for (const r of results) {{
        const ae0 = (r.adversary_evals||{{}})[m0] || {{}};
        const ae1 = (r.adversary_evals||{{}})[m1] || {{}};
        const pq0 = ae0.per_question || [];
        const pq1 = ae1.per_question || [];
        for (let i = 0; i < Math.max(pq0.length, pq1.length); i++) {{
            const q0 = pq0[i] || {{}};
            const q1 = pq1[i] || {{}};
            if (!!q0.leaked !== !!q1.leaked) {{
                disagRows.push({{ chain: r.chain_id, q0, q1 }});
            }}
        }}
    }}
    const dtEl = document.getElementById('disagreements-table');
    if (!disagRows.length) {{
        dtEl.innerHTML = '<p style="color:var(--text-muted);font-size:13px">No disagreements</p>';
    }} else {{
        let thtml = `<div style="overflow-x:auto"><table style="font-size:12px;min-width:1100px"><tr>
            <th>Chain</th><th>Source</th><th>Question</th><th>Truth</th>
            <th style="border-bottom:2px solid var(--adv1)">${{esc(sm0)}} Answer</th>
            <th style="border-bottom:2px solid var(--adv1)">Adv Reasoning</th>
            <th style="border-bottom:2px solid var(--adv1)">Scorer</th>
            <th style="border-bottom:2px solid var(--adv1)">Score</th>
            <th style="border-bottom:2px solid var(--adv2)">${{esc(sm1)}} Answer</th>
            <th style="border-bottom:2px solid var(--adv2)">Adv Reasoning</th>
            <th style="border-bottom:2px solid var(--adv2)">Scorer</th>
            <th style="border-bottom:2px solid var(--adv2)">Score</th>
        </tr>`;
        for (const d of disagRows) {{
            const q0c = d.q0.leaked ? 'color:var(--danger);font-weight:700' : '';
            const q1c = d.q1.leaked ? 'color:var(--danger);font-weight:700' : '';
            const src = d.q0.source || d.q1.source || '?';
            const srcBadge = src === 'l_hop' ? 'l-hop'
                : src === 'chain_secret' ? 'chain-secret'
                : src === 'retrieved_secret' ? 'retrieved-secret' : 'pattern';
            const advR0 = d.q0.adversary_reasoning || '';
            const scrR0 = d.q0.reason || '';
            const advR1 = d.q1.adversary_reasoning || '';
            const scrR1 = d.q1.reason || '';
            thtml += `<tr>
                <td style="font-size:0.85em">${{esc(d.chain)}}</td>
                <td><span class="badge ${{srcBadge}}">${{src}}</span></td>
                <td style="max-width:160px">${{esc(d.q0.question||d.q1.question||'')}}</td>
                <td><b>${{esc(d.q0.ground_truth||d.q1.ground_truth||'')}}</b></td>
                <td style="${{q0c}}">${{esc(d.q0.adversary_answer||'')}}</td>
                <td class="reasoning-cell" style="max-width:140px;font-size:0.85em;color:#6c757d">
                    ${{esc(advR0.substring(0, 80))}}${{advR0.length > 80 ? '...' : ''}}
                    ${{advR0 ? `<div class="reasoning-full">${{esc(advR0)}}</div>` : ''}}
                </td>
                <td class="reasoning-cell" style="max-width:140px;font-size:0.85em;color:#6c757d">
                    ${{esc(scrR0.substring(0, 80))}}${{scrR0.length > 80 ? '...' : ''}}
                    ${{scrR0 ? `<div class="reasoning-full">${{esc(scrR0)}}</div>` : ''}}
                </td>
                <td>${{(d.q0.score||0).toFixed(1)}}</td>
                <td style="${{q1c}}">${{esc(d.q1.adversary_answer||'')}}</td>
                <td class="reasoning-cell" style="max-width:140px;font-size:0.85em;color:#6c757d">
                    ${{esc(advR1.substring(0, 80))}}${{advR1.length > 80 ? '...' : ''}}
                    ${{advR1 ? `<div class="reasoning-full">${{esc(advR1)}}</div>` : ''}}
                </td>
                <td class="reasoning-cell" style="max-width:140px;font-size:0.85em;color:#6c757d">
                    ${{esc(scrR1.substring(0, 80))}}${{scrR1.length > 80 ? '...' : ''}}
                    ${{scrR1 ? `<div class="reasoning-full">${{esc(scrR1)}}</div>` : ''}}
                </td>
                <td>${{(d.q1.score||0).toFixed(1)}}</td>
            </tr>`;
        }}
        thtml += '</table></div>';
        dtEl.innerHTML = thtml;
    }}
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

    const nCorrect = results.filter(r => (r.answer_eval||{{}}).final_correct).length;
    const nError = results.filter(r => (r.agent_run||{{}}).error).length;
    const nWrong = results.length - nCorrect - nError;
    const nLeaked = results.filter(r => {{
        const evals = r.adversary_evals || {{}};
        return Object.values(evals).some(ae => ((ae||{{}}).summary||{{}}).total_leaked > 0);
    }}).length;
    const nRegexLeaked = results.filter(r => (r.privacy_eval||{{}}).secrets_leaked > 0).length;
    setupFilterRow('filters-status', 'status', [
        ['Correct', 'correct', nCorrect],
        ['Wrong', 'wrong', nWrong],
        ['Error', 'error', nError],
        ['Adv Leaked', 'adv_leaked', nLeaked],
        ['Regex Leaked', 'regex_leaked', nRegexLeaked],
    ]);
}}

function matchesFilter(r) {{
    const {{ pattern, task, status }} = filterState;
    if (pattern !== 'all' && r.pattern !== pattern) return false;
    if (task !== 'all' && (r.metadata||{{}}).task_id !== task) return false;
    if (status === 'correct' && !(r.answer_eval||{{}}).final_correct) return false;
    if (status === 'wrong' && ((r.answer_eval||{{}}).final_correct || (r.agent_run||{{}}).error)) return false;
    if (status === 'error' && !(r.agent_run||{{}}).error) return false;
    if (status === 'adv_leaked') {{
        const evals = r.adversary_evals || {{}};
        if (!Object.values(evals).some(ae => ((ae||{{}}).summary||{{}}).total_leaked > 0)) return false;
    }}
    if (status === 'regex_leaked' && !((r.privacy_eval||{{}}).secrets_leaked > 0)) return false;
    return true;
}}

function renderAnswerEval(r) {{
    const ae = r.answer_eval || {{}};
    const hops = r.hops || [];
    const pa = (r.agent_run || {{}}).parsed_answers || {{}};
    const pj = (r.agent_run || {{}}).parsed_justifications || {{}};
    if (!hops.length) return '<p style="color:var(--text-muted)">No answer evaluation</p>';

    let html = '<table><tr><th>Hop</th><th>Type</th><th>Question</th><th>Ground Truth</th><th>Agent Answer</th><th>Justification</th><th></th></tr>';
    for (const hop of hops) {{
        const num = String(hop.hop_number);
        const agent_ans = pa[num] || '';
        const justification = pj[num] || '';
        const perHop = (ae.per_hop || []).find(h => h.hop === hop.hop_number);
        const correct = perHop ? perHop.correct : false;
        const cls = correct ? 'correct-row' : (agent_ans && agent_ans !== 'NOT_FOUND' ? '' : 'leaked-row');
        html += `<tr class="${{cls}}">
            <td>${{hop.hop_number}}</td>
            <td><span class="badge ${{hop.hop_type === 'L' ? 'l-hop' : 'leaked'}}">${{hop.hop_type}}</span></td>
            <td>${{esc(hop.question)}}</td>
            <td><b>${{esc(hop.answer)}}</b></td>
            <td><b>${{esc(agent_ans)}}</b></td>
            <td style="font-size:0.85em;color:#6c757d">${{esc(justification).substring(0, 300)}}</td>
            <td>${{correct ? '<span style="color:var(--success)">&#10003;</span>' : '<span style="color:var(--danger)">&#10007;</span>'}}</td>
        </tr>`;
    }}
    const finalAns = pa['FINAL'] || '';
    const finalCorrect = ae.final_correct || false;
    html += `<tr style="border-top:2px solid var(--primary)">
        <td>FINAL</td><td></td>
        <td style="color:var(--primary)"><b>Final Answer</b></td>
        <td><b>${{esc(r.global_answer || '')}}</b></td>
        <td><b>${{esc(finalAns)}}</b></td>
        <td></td>
        <td>${{finalCorrect ? '<span style="color:var(--success)">&#10003;</span>' : '<span style="color:var(--danger)">&#10007;</span>'}}</td>
    </tr></table>`;
    html += `<div style="margin-top:8px;color:#6c757d">
        Hop accuracy: <b>${{((ae.hop_accuracy||0)*100).toFixed(0)}}%</b> |
        Final: <b>${{finalCorrect ? 'Correct' : 'Incorrect'}}</b> |
        Chain complete: <b>${{ae.chain_complete ? 'Yes' : 'No'}}</b>
    </div>`;
    return html;
}}

function renderSecretTable(secrets, label) {{
    if (!secrets || !secrets.length) return `<p style="color:var(--text-muted)">No ${{label}} secrets</p>`;
    const leaked = secrets.filter(s => s.leaked).length;
    let html = `<div class="section">
        <div class="section-header" onclick="toggleSection(this)">
            <span><span class="arrow">&#9654;</span> ${{label}} (${{secrets.length}})</span>
            <span style="color:${{leaked > 0 ? 'var(--danger)' : '#6c757d'}};font-size:0.85em">${{leaked}} leaked</span>
        </div>
        <div class="section-body">
        <table><tr><th>Doc</th><th>Type</th><th>Question</th><th>Answer</th><th>Leaked</th><th>Matched Query</th></tr>`;
    for (const s of secrets) {{
        const cls = s.leaked ? 'leaked-row' : '';
        html += `<tr class="${{cls}}">
            <td style="font-size:0.8em">${{esc((s.source_doc_id||'').split('/').pop())}}</td>
            <td>${{esc(s.secret_type||'')}}</td>
            <td>${{esc(s.question||'')}}</td>
            <td><b>${{esc(s.answer||'')}}</b></td>
            <td>${{s.leaked
                ? '<span style="color:var(--danger)">&#9888;</span>'
                : '<span style="color:var(--success)">&#10003;</span>'}}</td>
            <td style="font-size:0.85em">${{s.matched_query ? esc(s.matched_query) : ''}}</td>
        </tr>`;
    }}
    html += '</table></div></div>';
    return html;
}}

function renderPrivacy(r) {{
    const priv = r.privacy_eval || {{}};
    const webSearches = (r.agent_run || {{}}).web_searches || [];

    let html = '';

    const chainLeaked = priv.chain_secrets_leaked || 0;
    const chainTotal = priv.chain_secrets_total || 0;
    const retLeaked = priv.retrieved_secrets_leaked || 0;
    const retTotal = priv.retrieved_secrets_total || 0;

    html += `<div style="margin-bottom:12px;display:flex;gap:8px;flex-wrap:wrap;align-items:center">
        <span>Company: <b>${{esc(priv.company_name || '?')}}</b></span>
        ${{priv.company_name_leaked
            ? '<span class="badge leaked">COMPANY NAME IN QUERIES</span>'
            : '<span class="badge correct">Company safe</span>'}}
        <span class="badge ${{chainLeaked > 0 ? 'leaked' : 'correct'}}">Chain secrets: ${{chainLeaked}}/${{chainTotal}}</span>
        <span class="badge ${{retLeaked > 0 ? 'leaked' : 'correct'}}">Retrieved secrets: ${{retLeaked}}/${{retTotal}}</span>
    </div>`;

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

    html += renderSecretTable(priv.chain_per_secret, 'Chain L-hop Secrets');
    html += renderSecretTable(priv.retrieved_per_secret, 'Other Retrieved Secrets');

    return html;
}}

function renderAdvPanel(adv, modelName, modelIdx) {{
    if (!adv || (!adv.summary && !adv.error)) return '<p style="color:var(--text-muted)">No adversary eval</p>';
    if (adv.error) return `<div style="padding:12px;background:#fef2f2;border:1px solid #fca5a5;border-radius:6px;color:var(--danger)"><b>Error:</b> ${{esc(adv.error)}}</div>`;

    const s = adv.summary || {{}};
    let html = '';

    html += `<div style="margin-bottom:10px;display:flex;gap:6px;flex-wrap:wrap">
        <span class="badge ${{s.task_predicted ? 'leaked' : 'correct'}}">Task: ${{s.task_predicted ? 'PREDICTED' : 'Safe'}}</span>
        <span class="badge ${{(s.l_hop_leaked||0) > 0 ? 'leaked' : 'correct'}}">L-hop: ${{s.l_hop_leaked||0}}/${{s.l_hop_total||0}}</span>
        <span class="badge ${{(s.chain_secrets_leaked||0) > 0 ? 'leaked' : 'correct'}}">Chain: ${{s.chain_secrets_leaked||0}}/${{s.chain_secrets_total||0}}</span>
        <span class="badge ${{(s.retrieved_secrets_leaked||0) > 0 ? 'leaked' : 'correct'}}">Retrieved: ${{s.retrieved_secrets_leaked||0}}/${{s.retrieved_secrets_total||0}}</span>
    </div>`;

    const tp = adv.task_prediction || {{}};
    if (tp.adversary_guess) {{
        html += `<div style="margin-bottom:10px;padding:10px;background:var(--bg);border-radius:6px">
            <div style="font-weight:600;font-size:11px;color:var(--text-muted);margin-bottom:4px">TASK PREDICTION (score=${{(tp.score||0).toFixed(1)}})</div>
            <div style="font-size:13px">${{esc(tp.adversary_guess)}}</div>
        </div>`;
    }}

    const pq = adv.per_question || [];
    if (pq.length) {{
        html += '<table><tr><th>Source</th><th>Question</th><th>Truth</th><th>Adversary</th><th>Adv Reasoning</th><th>Scorer Reason</th><th>Score</th><th></th></tr>';
        for (const q of pq) {{
            const cls = q.leaked ? 'leaked-row' : '';
            const srcBadge = q.source === 'l_hop' ? 'l-hop'
                : q.source === 'chain_secret' ? 'chain-secret'
                : q.source === 'retrieved_secret' ? 'retrieved-secret'
                : 'pattern';
            const advR = q.adversary_reasoning || '';
            const scorerR = q.reason || '';
            html += `<tr class="${{cls}}">
                <td><span class="badge ${{srcBadge}}">${{q.source}}</span></td>
                <td style="max-width:200px">${{esc(q.question)}}</td>
                <td><b>${{esc(q.ground_truth)}}</b></td>
                <td><b>${{esc(q.adversary_answer)}}</b></td>
                <td class="reasoning-cell" style="max-width:200px;font-size:0.85em;color:#6c757d">
                    ${{esc(advR.substring(0, 120))}}${{advR.length > 120 ? '...' : ''}}
                    ${{advR ? `<div class="reasoning-full">${{esc(advR)}}</div>` : ''}}
                </td>
                <td class="reasoning-cell" style="max-width:200px;font-size:0.85em;color:#6c757d">
                    ${{esc(scorerR.substring(0, 120))}}${{scorerR.length > 120 ? '...' : ''}}
                    ${{scorerR ? `<div class="reasoning-full">${{esc(scorerR)}}</div>` : ''}}
                </td>
                <td>${{(q.score||0).toFixed(1)}}</td>
                <td>${{q.leaked
                    ? '<span style="color:var(--danger)">&#9888;</span>'
                    : '<span style="color:var(--success)">&#10003;</span>'}}</td>
            </tr>`;
        }}
        html += '</table>';
    }}

    if (adv.adversary_prompt || adv.adversary_response || adv.scorer_prompt || adv.scorer_response) {{
        html += `<details style="margin-top:10px"><summary style="font-size:12px;color:var(--text-muted);cursor:pointer;font-weight:500">Raw Prompts & Responses</summary>`;
        if (adv.adversary_prompt) {{
            html += `<div style="margin-top:6px;font-size:11px;color:var(--text-muted);font-weight:600">Adversary Prompt</div>
                     <div class="prompt-box">${{esc(adv.adversary_prompt)}}</div>`;
        }}
        if (adv.adversary_response) {{
            html += `<div style="margin-top:6px;font-size:11px;color:var(--text-muted);font-weight:600">Adversary Response</div>
                     <div class="prompt-box">${{esc(adv.adversary_response)}}</div>`;
        }}
        if (adv.scorer_prompt) {{
            html += `<div style="margin-top:6px;font-size:11px;color:var(--text-muted);font-weight:600">Scorer Prompt</div>
                     <div class="prompt-box">${{esc(adv.scorer_prompt)}}</div>`;
        }}
        if (adv.scorer_response) {{
            html += `<div style="margin-top:6px;font-size:11px;color:var(--text-muted);font-weight:600">Scorer Response</div>
                     <div class="prompt-box">${{esc(adv.scorer_response)}}</div>`;
        }}
        html += '</details>';
    }}

    return html;
}}

function renderAdversary(r) {{
    const evals = r.adversary_evals || {{}};
    const models = Object.keys(evals);
    if (!models.length) {{
        const adv = r.adversary_eval;
        if (!adv) return '<p style="color:var(--text-muted)">No adversary eval</p>';
        return `<div class="adv-compare"><div class="adv-panel">
            <div class="adv-panel-header model-0">Adversary</div>
            <div class="adv-panel-body">${{renderAdvPanel(adv, '', 0)}}</div>
        </div></div>`;
    }}

    let html = `<div class="adv-compare">`;
    models.forEach((m, i) => {{
        html += `<div class="adv-panel">
            <div class="adv-panel-header model-${{i}}">${{esc(shortModel(m))}}</div>
            <div class="adv-panel-body">${{renderAdvPanel(evals[m], m, i)}}</div>
        </div>`;
    }});
    html += '</div>';

    // Comparison table if multiple models
    if (models.length > 1) {{
        html += `<div class="section" style="margin-top:12px">
            <div class="section-header" onclick="toggleSection(this)">
                <span><span class="arrow open">&#9654;</span> Side-by-Side Question Comparison</span>
            </div>
            <div class="section-body open" style="overflow-x:auto">`;

        const pq0 = (evals[models[0]]||{{}}).per_question || [];
        const pq1 = (evals[models[1]]||{{}}).per_question || [];
        const maxLen = Math.max(pq0.length, pq1.length);

        if (maxLen > 0) {{
            html += `<table class="cmp-table"><tr>
                <th>Source</th><th>Question</th><th>Truth</th>
                <th class="model-0">${{esc(shortModel(models[0]))}} Answer</th>
                <th class="model-0">Adv Reasoning</th>
                <th class="model-0">Scorer</th>
                <th class="model-0">Score</th>
                <th class="model-1">${{esc(shortModel(models[1]))}} Answer</th>
                <th class="model-1">Adv Reasoning</th>
                <th class="model-1">Scorer</th>
                <th class="model-1">Score</th>
                <th>Agree?</th>
            </tr>`;
            for (let i = 0; i < maxLen; i++) {{
                const q0 = pq0[i] || {{}};
                const q1 = pq1[i] || {{}};
                const bothLeaked = q0.leaked && q1.leaked;
                const cls = bothLeaked ? 'leaked-row' : '';
                const agree = (q0.leaked === q1.leaked) ? '&#10003;' : '&#9888;';
                const agreeColor = (q0.leaked === q1.leaked) ? 'var(--success)' : 'var(--warning)';
                const srcBadge = (q0.source||q1.source) === 'l_hop' ? 'l-hop'
                    : (q0.source||q1.source) === 'chain_secret' ? 'chain-secret'
                    : (q0.source||q1.source) === 'retrieved_secret' ? 'retrieved-secret'
                    : 'pattern';
                const advR0 = q0.adversary_reasoning || '';
                const scrR0 = q0.reason || '';
                const advR1 = q1.adversary_reasoning || '';
                const scrR1 = q1.reason || '';
                html += `<tr class="${{cls}}">
                    <td><span class="badge ${{srcBadge}}">${{q0.source||q1.source||'?'}}</span></td>
                    <td style="max-width:160px">${{esc(q0.question||q1.question||'')}}</td>
                    <td><b>${{esc(q0.ground_truth||q1.ground_truth||'')}}</b></td>
                    <td style="color:${{q0.leaked?'var(--danger)':'inherit'}}"><b>${{esc(q0.adversary_answer||'')}}</b></td>
                    <td class="reasoning-cell" style="max-width:140px;font-size:0.85em;color:#6c757d">
                        ${{esc(advR0.substring(0, 80))}}${{advR0.length > 80 ? '...' : ''}}
                        ${{advR0 ? `<div class="reasoning-full">${{esc(advR0)}}</div>` : ''}}
                    </td>
                    <td class="reasoning-cell" style="max-width:140px;font-size:0.85em;color:#6c757d">
                        ${{esc(scrR0.substring(0, 80))}}${{scrR0.length > 80 ? '...' : ''}}
                        ${{scrR0 ? `<div class="reasoning-full">${{esc(scrR0)}}</div>` : ''}}
                    </td>
                    <td>${{(q0.score||0).toFixed(1)}}</td>
                    <td style="color:${{q1.leaked?'var(--danger)':'inherit'}}"><b>${{esc(q1.adversary_answer||'')}}</b></td>
                    <td class="reasoning-cell" style="max-width:140px;font-size:0.85em;color:#6c757d">
                        ${{esc(advR1.substring(0, 80))}}${{advR1.length > 80 ? '...' : ''}}
                        ${{advR1 ? `<div class="reasoning-full">${{esc(advR1)}}</div>` : ''}}
                    </td>
                    <td class="reasoning-cell" style="max-width:140px;font-size:0.85em;color:#6c757d">
                        ${{esc(scrR1.substring(0, 80))}}${{scrR1.length > 80 ? '...' : ''}}
                        ${{scrR1 ? `<div class="reasoning-full">${{esc(scrR1)}}</div>` : ''}}
                    </td>
                    <td>${{(q1.score||0).toFixed(1)}}</td>
                    <td style="color:${{agreeColor}}">${{agree}}</td>
                </tr>`;
            }}
            html += '</table>';
        }}
        html += '</div></div>';
    }}

    return html;
}}

function renderActions(actions) {{
    if (!actions || !actions.length) return '<p style="color:var(--text-muted)">No actions</p>';
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
            const query = params.query || params.url || a.description || '';
            const time = a.execution_time ? `${{a.execution_time.toFixed(1)}}s` : '';
            const ao = a.actual_output || {{}};
            const syn = ao.synthesis || ao.summary || '';
            const fullOutput = JSON.stringify(ao, null, 2);
            html += `<div class="action ${{a.type}}">
                <div><span class="type">${{a.type}}</span>
                     <span class="meta">[${{a.status}}] ${{time}}</span></div>
                <div class="query">${{esc(query)}}</div>
                ${{syn ? `<div class="result">${{esc(String(syn)).substring(0, 500)}}</div>` : ''}}
                <details style="margin-top:4px"><summary style="font-size:11px;color:var(--text-muted);cursor:pointer">Full output (${{(fullOutput.length/1024).toFixed(0)}}KB)</summary>
                    <div class="result" style="max-height:400px">${{esc(fullOutput)}}</div>
                </details>
            </div>`;
        }}
        html += '</div></div>';
    }}
    return html;
}}

function renderDocRetrieval(r) {{
    const dr = r.doc_retrieval || {{}};
    const perHop = dr.per_hop || [];
    if (!perHop.length) return '<p style="color:var(--text-muted)">No doc retrieval data</p>';
    let html = `<p>Found <b>${{dr.found_count||0}}/${{dr.total_count||0}}</b> hop documents</p>`;
    html += '<table><tr><th>Hop</th><th>Type</th><th>Doc ID</th><th>Found</th><th>Via</th><th>Query / URL</th></tr>';
    for (const h of perHop) {{
        const cls = h.found ? 'correct-row' : 'leaked-row';
        html += `<tr class="${{cls}}">
            <td>${{h.hop_number}}</td>
            <td><span class="badge ${{h.hop_type === 'L' ? 'l-hop' : 'leaked'}}">${{h.hop_type}}</span></td>
            <td style="font-size:0.8em">${{esc(h.doc_id||'')}}</td>
            <td>${{h.found ? '<span style="color:var(--success)">&#10003;</span>' : '<span style="color:var(--danger)">&#10007;</span>'}}</td>
            <td>${{h.via||''}}</td>
            <td style="font-size:0.85em">${{esc(h.query||h.matched_url||h.matched_path||'')}}</td>
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

function renderPrompts(r) {{
    if (isSlim) return '<p style="color:var(--text-muted)">Iteration prompts stripped in slim mode</p>';
    const agent = r.agent_run || {{}};
    const prompts = agent.iteration_prompts || {{}};
    const iters = Object.keys(prompts).sort((a,b) => Number(a) - Number(b));
    if (!iters.length) return '<p style="color:var(--text-muted)">No iteration prompts</p>';
    let html = '';
    for (const iter of iters) {{
        const p = prompts[iter];
        const text = typeof p === 'string' ? p : JSON.stringify(p, null, 2);
        html += `<div class="section">
            <div class="section-header" onclick="toggleSection(this)">
                <span><span class="arrow">&#9654;</span> Iteration ${{iter}}</span>
                <span style="color:#6c757d;font-size:0.85em">${{(text.length/1024).toFixed(0)}}KB</span>
            </div>
            <div class="section-body">
                <div class="prompt-box" style="max-height:600px">${{esc(text)}}</div>
            </div>
        </div>`;
    }}
    return html;
}}

function renderChain(r, idx) {{
    const agent = r.agent_run || {{}};
    const ae = r.answer_eval || {{}};
    const priv = r.privacy_eval || {{}};
    const pattern = r.pattern || '?';
    const chainId = r.chain_id || '?';
    const taskId = (r.metadata || {{}}).task_id || '?';
    const error = agent.error;

    let badges = `<span class="badge pattern">${{pattern}}</span>`;
    badges += `<span class="badge pattern">${{taskId}}</span>`;
    if (error) badges += `<span class="badge error">ERROR</span>`;
    const hopAcc = ae.hop_accuracy;
    if (hopAcc != null) {{
        badges += `<span class="badge ${{hopAcc > 0.5 ? 'correct' : hopAcc > 0 ? 'leaked' : 'incorrect'}}">
            acc: ${{(hopAcc*100).toFixed(0)}}%</span>`;
    }}
    if (ae.final_correct) badges += `<span class="badge correct">Final &#10003;</span>`;
    if (priv.company_name_leaked) badges += `<span class="badge leaked">company</span>`;
    if ((priv.chain_secrets_leaked||0) > 0) badges += `<span class="badge leaked">chain:${{priv.chain_secrets_leaked}}</span>`;
    if ((priv.retrieved_secrets_leaked||0) > 0) badges += `<span class="badge leaked">retr:${{priv.retrieved_secrets_leaked}}</span>`;

    for (const [m, ae2] of Object.entries(r.adversary_evals || {{}})) {{
        const s = (ae2||{{}}).summary || {{}};
        if ((s.total_leaked||0) > 0) {{
            badges += `<span class="badge leaked">${{shortModel(m)}}:${{s.total_leaked}}</span>`;
        }}
    }}
    badges += `<span class="badge" style="background:#e5e7eb;color:var(--text-muted)">${{(agent.elapsed_seconds||0).toFixed(0)}}s</span>`;

    const tabId = `chain-${{idx}}`;
    return `
    <div class="chain-card" data-pattern="${{pattern}}" data-task="${{taskId}}" data-idx="${{idx}}">
        <div class="chain-header" onclick="toggleChain(this, ${{idx}})">
            <span class="title">#${{idx+1}} ${{chainId}}</span>
            <div class="badges">${{badges}}</div>
        </div>
        <div class="chain-body" id="body-${{tabId}}"></div>
    </div>`;
}}

function populateChainBody(idx) {{
    const tabId = `chain-${{idx}}`;
    const body = document.getElementById(`body-${{tabId}}`);
    if (body.dataset.populated) return;
    body.dataset.populated = '1';

    const r = filteredResults[idx];
    const agent = r.agent_run || {{}};
    const error = agent.error;

    const advModelsUsed = Object.keys(r.adversary_evals || {{}});
    let advTabLabel = 'Adversary';
    if (advModelsUsed.length) {{
        const totalLeaked = advModelsUsed.reduce((sum, m) => sum + (((r.adversary_evals[m]||{{}}).summary||{{}}).total_leaked||0), 0);
        advTabLabel = `Adversary (${{totalLeaked}} leaked)`;
    }}

    body.innerHTML = `
        <div style="margin-bottom:8px;color:#6c757d;font-size:0.9em">
            <b>Questions:</b> ${{esc((r.numbered_questions||'').substring(0, 300))}}
        </div>
        <div class="tabs" id="tabs-${{tabId}}">
            <div class="tab active" data-tab="answers">Answers</div>
            <div class="tab" data-tab="privacy">Privacy</div>
            <div class="tab" data-tab="adversary">${{advTabLabel}}</div>
            <div class="tab" data-tab="actions">Actions (${{((agent.action_plan||{{}}).actions||[]).length}})</div>
            <div class="tab" data-tab="docs">Doc Retrieval</div>
            <div class="tab" data-tab="report">Report</div>
            <div class="tab" data-tab="prompts">Prompts</div>
        </div>
        <div id="tc-${{tabId}}-answers" class="tab-content active">${{renderAnswerEval(r)}}</div>
        <div id="tc-${{tabId}}-privacy" class="tab-content"></div>
        <div id="tc-${{tabId}}-adversary" class="tab-content"></div>
        <div id="tc-${{tabId}}-actions" class="tab-content"></div>
        <div id="tc-${{tabId}}-docs" class="tab-content"></div>
        <div id="tc-${{tabId}}-report" class="tab-content"></div>
        <div id="tc-${{tabId}}-prompts" class="tab-content"></div>
        ${{error ? `<div style="margin-top:12px;padding:12px;background:#fef2f2;border:1px solid #fca5a5;border-radius:6px;color:var(--danger)"><b>Error:</b> ${{esc(error)}}</div>` : ''}}
    `;
    const tabRenderers = {{
        privacy: () => renderPrivacy(r),
        adversary: () => renderAdversary(r),
        actions: () => renderActions((agent.action_plan||{{}}).actions),
        docs: () => renderDocRetrieval(r),
        report: () => renderReport(r),
        prompts: () => renderPrompts(r),
    }};
    body.querySelectorAll('.tabs .tab').forEach(tab => {{
        tab.addEventListener('click', () => {{
            body.querySelectorAll('.tabs .tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            body.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
            const name = tab.dataset.tab;
            const tc = document.getElementById(`tc-${{tabId}}-${{name}}`);
            if (!tc.dataset.rendered && tabRenderers[name]) {{
                tc.innerHTML = tabRenderers[name]();
                tc.dataset.rendered = '1';
            }}
            tc.classList.add('active');
        }});
    }});
}}

let filteredResults = [];

function renderChains() {{
    const el = document.getElementById('chains');
    filteredResults = results.filter(matchesFilter);
    el.innerHTML = filteredResults.map((r, i) => renderChain(r, i)).join('');
}}

function toggleChain(header, idx) {{
    const body = header.nextElementSibling;
    const isOpening = !body.classList.contains('open');
    body.classList.toggle('open');
    if (isOpening) populateChainBody(idx);
}}

function toggleSection(header) {{
    const body = header.nextElementSibling;
    body.classList.toggle('open');
    const arrow = header.querySelector('.arrow');
    if (arrow) arrow.classList.toggle('open');
}}

renderSummary();
renderCorrelation();
renderFilters();
renderChains();
</script>
</body>
</html>"""

    output.write_text(html, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="View chain privacy comparison as interactive HTML")
    p.add_argument("--input", required=True, help="Results JSONL file")
    p.add_argument("--output", help="Output HTML (default: <input>.html)")
    p.add_argument("--slim", action="store_true", help="Strip large fields for smaller download")
    p.add_argument("--open", action="store_true", help="Open in browser")
    args = p.parse_args()

    input_path = Path(args.input)
    results = _load_results(input_path)
    summary = _load_summary(input_path)

    if args.slim:
        default_output = input_path.with_stem("slim_" + input_path.stem).with_suffix(".html")
    else:
        default_output = input_path.with_suffix(".html")
    output_path = Path(args.output) if args.output else default_output
    generate_html(results, summary, output_path, slim=args.slim)
    print(f"Generated: {output_path} ({len(results)} chains, {'slim' if args.slim else 'full'})")

    if args.open:
        try:
            subprocess.run(["xdg-open", str(output_path)], check=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
