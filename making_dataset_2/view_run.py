#!/usr/bin/env python3
"""Self-contained HTML viewer for a DrBench agent run.

Reads action_plan_final.json, research_plan.json, report.json, scores.json,
and question_used.json from a run directory and generates a single HTML file
with embedded data showing the full action timeline, search results with
content, and final report.

Usage:
    python -m making_dataset_2.view_run ./runs/batch_X/DR0001
    python -m making_dataset_2.view_run ./runs/batch_X/DR0001 --open
    python -m making_dataset_2.view_run ./runs/batch_X/DR0001 -o custom_output.html
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_data(run_dir: Path) -> dict:
    """Collect all data from a run directory into a single dict for the viewer."""
    action_plan = _load_json(run_dir / "action_plan_final.json")
    if action_plan is None:
        action_plan = _load_json(run_dir / "action_plan_initial.json")

    research_plan = _load_json(run_dir / "research_plan.json")
    report = _load_json(run_dir / "report.json")
    scores = _load_json(run_dir / "scores.json")
    question_info = _load_json(run_dir / "question_used.json")

    # Config might be one level up
    config = _load_json(run_dir / "config.json") or _load_json(run_dir.parent / "config.json")

    # Privacy data
    privacy_eval = _load_json(run_dir / "privacy" / "privacy_eval.json")
    web_searches = _load_json(run_dir / "privacy" / "web_searches.json")

    return {
        "action_plan": action_plan,
        "research_plan": research_plan,
        "report": report,
        "scores": scores,
        "question_info": question_info,
        "config": config,
        "privacy_eval": privacy_eval,
        "web_searches": web_searches,
        "task_id": run_dir.name if run_dir.name.startswith("DR") else question_info.get("task_id", "") if question_info else "",
        "run_dir": str(run_dir),
    }


def generate_html(run_dir: Path, output: Path | None = None) -> Path:
    """Generate self-contained HTML viewer and return the output path."""
    data = _collect_data(run_dir)
    data_json = json.dumps(data, ensure_ascii=False, default=str)

    html = _HTML_TEMPLATE.replace("/*DATA_PLACEHOLDER*/null", data_json)

    if output is None:
        output = run_dir / "viewer.html"
    output.write_text(html, encoding="utf-8")
    return output


_HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DrBench Run Viewer</title>
<style>
:root {
  --bg: #f5f6fa; --card: #ffffff; --primary: #4361ee; --primary-light: #eef1ff;
  --success: #10b981; --success-bg: #ecfdf5; --warning: #f59e0b; --warning-bg: #fffbeb;
  --danger: #ef4444; --danger-bg: #fef2f2; --info: #6366f1; --info-bg: #eef2ff;
  --muted: #6c757d; --text: #1a1a2e; --border: #e2e8f0;
  --web: #7c3aed; --local: #0891b2; --enterprise: #ea580c; --analysis: #65a30d;
  --shadow: 0 1px 3px rgba(0,0,0,0.08); --shadow-lg: 0 4px 12px rgba(0,0,0,0.12);
  --mono: "SF Mono", "Fira Code", "Fira Mono", Menlo, Consolas, monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); line-height: 1.55; font-size: 14px; }

/* Header */
.header { background: var(--card); padding: 20px 28px; border-bottom: 1px solid var(--border); box-shadow: var(--shadow); }
.header h1 { font-size: 18px; font-weight: 700; color: var(--primary); margin-bottom: 8px; }
.question-box { background: var(--primary-light); border: 1px solid #c7d2fe; border-radius: 8px; padding: 12px 16px; margin: 10px 0; white-space: pre-wrap; font-size: 14px; line-height: 1.6; }
.meta-row { display: flex; gap: 24px; flex-wrap: wrap; font-size: 13px; color: var(--muted); margin-top: 8px; }
.meta-item { display: flex; align-items: center; gap: 5px; }
.meta-value { color: var(--text); font-weight: 600; }

/* Scores bar */
.scores-bar { display: flex; gap: 16px; flex-wrap: wrap; padding: 12px 28px; background: var(--card); border-bottom: 1px solid var(--border); }
.score-chip { display: flex; align-items: center; gap: 6px; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600; }
.score-chip.hm { background: var(--primary-light); color: var(--primary); font-size: 15px; }
.score-chip.good { background: var(--success-bg); color: var(--success); }
.score-chip.mid { background: var(--warning-bg); color: var(--warning); }
.score-chip.bad { background: var(--danger-bg); color: var(--danger); }

/* Tabs */
.tabs { display: flex; background: var(--card); border-bottom: 1px solid var(--border); padding: 0 28px; }
.tab { padding: 10px 20px; font-size: 13px; font-weight: 500; cursor: pointer; border-bottom: 2px solid transparent; color: var(--muted); transition: all 0.15s; }
.tab:hover { color: var(--text); }
.tab.active { color: var(--primary); border-bottom-color: var(--primary); }
.tab-content { display: none; padding: 20px 28px; }
.tab-content.active { display: block; }

/* Timeline */
.iter-group { margin-bottom: 20px; }
.iter-header { font-size: 13px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; padding: 4px 0; border-bottom: 1px solid var(--border); }
.action-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 8px; box-shadow: var(--shadow); overflow: hidden; }
.action-header { display: flex; align-items: center; gap: 10px; padding: 10px 14px; cursor: pointer; user-select: none; }
.action-header:hover { background: #fafbff; }
.action-type { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.3px; padding: 2px 8px; border-radius: 4px; white-space: nowrap; }
.type-web_search { background: #ede9fe; color: var(--web); }
.type-local_document_search, .type-local_file_analysis { background: #ecfeff; color: var(--local); }
.type-enterprise_api, .type-mcp_query { background: #fff7ed; color: var(--enterprise); }
.type-data_analysis, .type-context_synthesis { background: #f7fee7; color: var(--analysis); }
.type-url_fetch { background: #fdf4ff; color: #a855f7; }
.status-badge { font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; }
.status-completed { background: var(--success-bg); color: var(--success); }
.status-failed { background: var(--danger-bg); color: var(--danger); }
.status-pending { background: #f1f5f9; color: var(--muted); }
.status-in_progress { background: var(--warning-bg); color: var(--warning); }
.status-skipped { background: #f1f5f9; color: var(--muted); }
.action-query { flex: 1; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.action-time { font-size: 12px; color: var(--muted); white-space: nowrap; }
.expand-icon { font-size: 12px; color: var(--muted); transition: transform 0.15s; width: 16px; text-align: center; }
.action-card.open .expand-icon { transform: rotate(90deg); }
.action-detail { display: none; padding: 0 14px 14px; border-top: 1px solid var(--border); }
.action-card.open .action-detail { display: block; }
.detail-section { margin-top: 10px; }
.detail-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.3px; color: var(--muted); margin-bottom: 4px; }
.detail-text { font-size: 13px; color: var(--text); }

/* Search results */
.result-card { background: #fafbff; border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; margin-top: 6px; }
.result-rank { font-size: 11px; font-weight: 700; color: var(--muted); }
.result-score { font-size: 11px; font-weight: 600; color: var(--primary); margin-left: 8px; }
.result-url { font-size: 12px; color: var(--web); word-break: break-all; margin-top: 2px; }
.result-text { font-size: 12px; color: var(--text); margin-top: 4px; line-height: 1.5; max-height: 120px; overflow: hidden; position: relative; }
.result-text.expanded { max-height: none; }
.result-text.truncated::after { content: ""; position: absolute; bottom: 0; left: 0; right: 0; height: 30px; background: linear-gradient(transparent, #fafbff); }
.show-more { font-size: 11px; color: var(--primary); cursor: pointer; margin-top: 4px; font-weight: 600; }

/* Research plan */
.plan-area { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; margin-bottom: 10px; }
.plan-area h3 { font-size: 14px; font-weight: 600; margin-bottom: 6px; }
.plan-area p, .plan-area li { font-size: 13px; line-height: 1.6; }
.plan-area ul { padding-left: 20px; }

/* Report */
.report-box { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 20px 24px; white-space: pre-wrap; font-size: 13px; line-height: 1.7; max-height: 800px; overflow-y: auto; }

/* Stats summary */
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 20px; }
.stat-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; text-align: center; }
.stat-card .label { font-size: 11px; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.stat-card .value { font-size: 22px; font-weight: 700; color: var(--primary); margin-top: 2px; }

/* Privacy tab */
.privacy-badge { display: inline-block; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; }
.badge-pass { background: var(--success-bg); color: var(--success); }
.badge-fail { background: var(--danger-bg); color: var(--danger); }
.badge-partial { background: var(--warning-bg); color: var(--warning); }
.privacy-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; margin-bottom: 10px; }
.privacy-card h4 { font-size: 13px; font-weight: 600; margin-bottom: 6px; }
.privacy-card .ground-truth { color: var(--text); font-size: 12px; margin-top: 6px; }
.query-list { list-style: decimal; padding-left: 24px; font-size: 13px; line-height: 1.8; }
.query-list li { padding: 2px 0; }
.supporting-docs { margin-top: 10px; font-size: 13px; }
.supporting-docs summary { cursor: pointer; font-weight: 600; font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.3px; }
.supporting-docs summary:hover { color: var(--text); }
.email-card { background: #fafbff; border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; margin-top: 6px; }
.email-header { font-size: 12px; color: var(--muted); margin-bottom: 2px; }
.email-subject { font-size: 12px; font-weight: 600; color: var(--text); margin-bottom: 4px; }
.email-body { font-size: 12px; color: var(--text); line-height: 1.5; white-space: pre-wrap; max-height: 200px; overflow-y: auto; }
.doc-card { background: #fafbff; border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; margin-top: 6px; }
.doc-header { font-size: 12px; font-weight: 600; color: var(--primary); margin-bottom: 4px; }
.doc-text { font-size: 11px; font-family: var(--mono); color: var(--text); line-height: 1.4; white-space: pre-wrap; max-height: 300px; overflow-y: auto; background: #f8fafc; border: 1px solid var(--border); border-radius: 4px; padding: 8px; margin: 0; }
</style>
</head>
<body>

<div id="app"></div>

<script>
const DATA = /*DATA_PLACEHOLDER*/null;

function esc(s) { const d = document.createElement("div"); d.textContent = s || ""; return d.innerHTML; }
function fmt_time(s) { if (s == null) return "—"; if (s < 1) return (s*1000).toFixed(0)+"ms"; if (s < 60) return s.toFixed(1)+"s"; return (s/60).toFixed(1)+"m"; }
function score_class(v) { if (v >= 0.7) return "good"; if (v >= 0.4) return "mid"; return "bad"; }

function render() {
  const app = document.getElementById("app");
  const ap = DATA.action_plan || {};
  const rp = DATA.research_plan || {};
  const report = DATA.report || {};
  const scores = DATA.scores || {};
  const qi = DATA.question_info || {};
  const cfg = DATA.config || {};
  const actions = ap.actions || [];

  // Group actions by iteration
  const byIter = {};
  let maxIter = -1;
  actions.forEach((a, i) => {
    a._idx = i;
    const iter = a.created_in_iteration != null ? a.created_in_iteration : (a.iteration_completed != null ? a.iteration_completed : -1);
    if (!byIter[iter]) byIter[iter] = [];
    byIter[iter].push(a);
    if (iter > maxIter) maxIter = iter;
  });

  // Count stats
  const total = actions.length;
  const completed = actions.filter(a => a.status === "completed").length;
  const failed = actions.filter(a => a.status === "failed").length;
  const webSearchCount = actions.filter(a => a.type === "web_search").length;
  const localSearchCount = actions.filter(a => a.type === "local_document_search" || a.type === "local_file_analysis").length;
  const totalTime = actions.reduce((s, a) => s + (a.execution_time || 0), 0);

  let html = "";

  // Header
  html += `<div class="header">`;
  html += `<h1>DrBench Run Viewer</h1>`;
  if (qi.dr_question) html += `<div class="question-box">${esc(qi.dr_question)}</div>`;
  html += `<div class="meta-row">`;
  if (DATA.task_id) html += `<div class="meta-item">Task: <span class="meta-value">${esc(DATA.task_id)}</span></div>`;
  if (cfg.model) html += `<div class="meta-item">Model: <span class="meta-value">${esc(cfg.model)}</span></div>`;
  if (cfg.llm_provider) html += `<div class="meta-item">Provider: <span class="meta-value">${esc(cfg.llm_provider)}</span></div>`;
  if (cfg.browsecomp_enabled) html += `<div class="meta-item">BrowseComp: <span class="meta-value">ON</span></div>`;
  html += `</div></div>`;

  // Scores bar
  if (scores && Object.keys(scores).length > 0) {
    html += `<div class="scores-bar">`;
    if (scores.harmonic_mean != null) html += `<div class="score-chip hm">HM ${scores.harmonic_mean.toFixed(3)}</div>`;
    for (const [k, v] of Object.entries(scores)) {
      if (k === "harmonic_mean" || typeof v !== "number") continue;
      html += `<div class="score-chip ${score_class(v)}">${esc(k.replace(/_/g, " "))} ${v.toFixed(3)}</div>`;
    }
    html += `</div>`;
  }

  // Privacy data
  const priv = DATA.privacy_eval || {};
  const webSearches = DATA.web_searches || {};
  const hasPrivacy = priv && (priv.adversary_eval || priv.quick_check);

  // Tabs
  html += `<div class="tabs">`;
  html += `<div class="tab active" data-tab="timeline">Timeline (${total})</div>`;
  html += `<div class="tab" data-tab="plan">Research Plan</div>`;
  html += `<div class="tab" data-tab="report">Report</div>`;
  if (hasPrivacy) html += `<div class="tab" data-tab="privacy">Privacy</div>`;
  html += `</div>`;

  // Tab: Timeline
  html += `<div class="tab-content active" id="tab-timeline">`;

  // Stats grid
  html += `<div class="stats-grid">`;
  html += `<div class="stat-card"><div class="label">Actions</div><div class="value">${total}</div></div>`;
  html += `<div class="stat-card"><div class="label">Completed</div><div class="value" style="color:var(--success)">${completed}</div></div>`;
  html += `<div class="stat-card"><div class="label">Failed</div><div class="value" style="color:var(--danger)">${failed}</div></div>`;
  html += `<div class="stat-card"><div class="label">Web Searches</div><div class="value" style="color:var(--web)">${webSearchCount}</div></div>`;
  html += `<div class="stat-card"><div class="label">Local Searches</div><div class="value" style="color:var(--local)">${localSearchCount}</div></div>`;
  html += `<div class="stat-card"><div class="label">Total Time</div><div class="value">${fmt_time(totalTime)}</div></div>`;
  html += `</div>`;

  // Action groups by iteration
  const sortedIters = Object.keys(byIter).map(Number).sort((a,b) => a - b);
  for (const iter of sortedIters) {
    const group = byIter[iter];
    const label = iter < 0 ? "Unknown Iteration" : `Iteration ${iter + 1}`;
    html += `<div class="iter-group">`;
    html += `<div class="iter-header">${label} — ${group.length} action${group.length > 1 ? "s" : ""}</div>`;
    for (const a of group) {
      html += renderAction(a);
    }
    html += `</div>`;
  }
  html += `</div>`;

  // Tab: Research Plan
  html += `<div class="tab-content" id="tab-plan">`;
  const planAreas = (rp && rp.plan && rp.plan.research_investigation_areas) || (rp && rp.research_investigation_areas) || [];
  if (planAreas.length) {
    if (rp.query) html += `<div class="plan-area"><h3>Query</h3><p>${esc(rp.query)}</p></div>`;
    for (const area of planAreas) {
      html += `<div class="plan-area">`;
      html += `<h3>${esc(area.research_focus || area.area || area.name || "Research Area " + (area.area_id || ""))}</h3>`;
      if (area.business_rationale) html += `<p style="margin-bottom:6px">${esc(area.business_rationale)}</p>`;
      if (area.information_needs && area.information_needs.length) {
        html += `<div class="detail-label" style="margin-top:8px">Information Needs</div><ul>`;
        for (const n of area.information_needs) html += `<li>${esc(n)}</li>`;
        html += `</ul>`;
      }
      if (area.key_concepts && area.key_concepts.length) {
        html += `<div class="detail-label" style="margin-top:8px">Key Concepts</div><ul>`;
        for (const c of area.key_concepts) html += `<li>${esc(c)}</li>`;
        html += `</ul>`;
      }
      if (area.expected_insights && area.expected_insights.length) {
        html += `<div class="detail-label" style="margin-top:8px">Expected Insights</div><ul>`;
        for (const ei of area.expected_insights) html += `<li>${esc(ei)}</li>`;
        html += `</ul>`;
      }
      if (area.questions && area.questions.length) {
        html += `<ul>`;
        for (const q of area.questions) html += `<li>${esc(typeof q === "string" ? q : q.question || JSON.stringify(q))}</li>`;
        html += `</ul>`;
      }
      html += `</div>`;
    }
  } else {
    html += `<p style="color:var(--muted)">No research plan data available.</p>`;
  }
  html += `</div>`;

  // Tab: Report
  html += `<div class="tab-content" id="tab-report">`;
  const reportText = report ? (report.report_text || JSON.stringify(report, null, 2)) : "No report available.";
  html += `<div class="report-box">${esc(reportText)}</div>`;
  html += `</div>`;

  // Tab: Privacy
  if (hasPrivacy) {
    html += `<div class="tab-content" id="tab-privacy">`;
    const qc = priv.quick_check || {};
    const ae = priv.adversary_eval || {};
    const aeSummary = ae.summary || {};

    // Summary bar
    html += `<div class="stats-grid">`;
    html += `<div class="stat-card"><div class="label">Web Queries</div><div class="value">${priv.total_queries || 0}</div></div>`;
    html += `<div class="stat-card"><div class="label">Company Name</div><div class="value"><span class="privacy-badge ${qc.company_name_leaked ? 'badge-fail' : 'badge-pass'}">${qc.company_name_leaked ? 'LEAKED' : 'OK'}</span></div></div>`;
    html += `<div class="stat-card"><div class="label">Insights Leaked</div><div class="value" style="color:${aeSummary.insights_leaked ? 'var(--danger)' : 'var(--success)'}">${aeSummary.insights_leaked || 0}/${aeSummary.insights_total || 0}</div></div>`;
    html += `<div class="stat-card"><div class="label">Distractors Leaked</div><div class="value" style="color:${aeSummary.distractors_leaked ? 'var(--danger)' : 'var(--success)'}">${aeSummary.distractors_leaked || 0}/${aeSummary.distractors_total || 0}</div></div>`;
    html += `<div class="stat-card"><div class="label">Regex Metrics</div><div class="value" style="color:${qc.metrics_leaked ? 'var(--danger)' : 'var(--success)'}">${qc.metrics_leaked || 0}/${qc.metrics_total || 0}</div></div>`;
    html += `</div>`;

    // Web queries list
    const searches_list = (webSearches && webSearches.searches) || [];
    if (searches_list.length) {
      html += `<div class="privacy-card"><h4>Web Search Queries</h4><ol class="query-list">`;
      for (const s of searches_list) {
        const statusBadge = s.status === "completed" ? "badge-pass" : s.status === "failed" ? "badge-fail" : "badge-partial";
        html += `<li><span class="privacy-badge ${statusBadge}" style="margin-right:6px">${esc(s.status)}</span>${esc(s.query)}</li>`;
      }
      html += `</ol></div>`;
    }

    // Per-question adversary results
    const perQ = ae.per_question || {};
    const qids = Object.keys(perQ);
    if (qids.length) {
      html += `<div class="privacy-card"><h4>Adversary Evaluation (per question)</h4>`;
      for (const qid of qids) {
        const q = perQ[qid];
        const leaked = q.leaked;
        const badgeClass = leaked ? (q.score >= 1.0 ? "badge-fail" : "badge-partial") : "badge-pass";
        const badgeText = leaked ? (q.score >= 1.0 ? "LEAKED" : "PARTIAL") : "OK";
        html += `<div style="margin:10px 0;padding:10px;background:#fafbff;border:1px solid var(--border);border-radius:6px">`;
        html += `<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">`;
        html += `<span class="privacy-badge ${badgeClass}">${badgeText}</span>`;
        html += `<span style="font-size:11px;color:var(--muted)">[${esc(q.qa_type)}] ${esc(qid)}</span>`;
        html += `<span style="font-size:11px;color:var(--muted);margin-left:auto">score: ${q.score}</span>`;
        html += `</div>`;
        html += `<div style="font-size:13px;font-weight:500;margin-bottom:4px">${esc(q.question)}</div>`;
        html += `<div style="font-size:12px"><b>Adversary:</b> ${esc(q.adversary_answer || "—")}</div>`;
        if (q.adversary_reasoning) html += `<div style="font-size:12px;color:var(--muted)"><b>Why:</b> ${esc(q.adversary_reasoning)}</div>`;
        if (q.reason) html += `<div style="font-size:12px;color:var(--muted)"><b>Scorer:</b> ${esc(q.reason)}</div>`;
        html += `<div style="font-size:12px;margin-top:4px"><b>Ground truth:</b> <span class="ground-truth">${esc(q.ground_truth)}</span></div>`;

        // Supporting documents
        const docs = q.supporting_docs || [];
        if (docs.length) {
          html += `<details class="supporting-docs"><summary>Supporting Documents (${docs.length})</summary>`;
          for (const doc of docs) {
            if (doc.type === "email" && doc.emails) {
              html += `<div class="doc-header" style="margin-top:8px">${esc(doc.filename)}</div>`;
              for (const em of doc.emails) {
                html += `<div class="email-card">`;
                html += `<div class="email-header">From: ${esc(em.from)} | Date: ${esc(em.date)}</div>`;
                html += `<div class="email-subject">Subject: ${esc(em.subject)}</div>`;
                html += `<div class="email-body">${esc(em.body)}</div>`;
                html += `</div>`;
              }
            } else if (doc.text) {
              html += `<div class="doc-card">`;
              html += `<div class="doc-header">${esc(doc.filename)}</div>`;
              html += `<pre class="doc-text">${esc(doc.text)}</pre>`;
              html += `</div>`;
            } else {
              html += `<div class="doc-card"><div class="doc-header">${esc(doc.filename)} (${esc(doc.type)})</div></div>`;
            }
          }
          html += `</details>`;
        }

        html += `</div>`;
      }
      html += `</div>`;
    }

    html += `</div>`;
  }

  app.innerHTML = html;

  // Tab switching
  document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById("tab-" + tab.dataset.tab).classList.add("active");
    });
  });

  // Action expand/collapse
  document.querySelectorAll(".action-header").forEach(h => {
    h.addEventListener("click", () => h.parentElement.classList.toggle("open"));
  });

  // Show-more buttons
  document.querySelectorAll(".show-more").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const textEl = btn.previousElementSibling;
      textEl.classList.toggle("expanded");
      textEl.classList.toggle("truncated");
      btn.textContent = textEl.classList.contains("expanded") ? "Show less" : "Show more";
    });
  });
}

function renderAction(a) {
  const query = (a.parameters || {}).query || (a.parameters || {}).url || a.description || "";
  const typeClass = "type-" + (a.type || "").replace(/\\s/g, "_");
  const statusClass = "status-" + (a.status || "pending");

  let h = `<div class="action-card" data-idx="${a._idx}">`;
  h += `<div class="action-header">`;
  h += `<span class="expand-icon">&#9654;</span>`;
  h += `<span class="action-type ${typeClass}">${esc(a.type || "unknown")}</span>`;
  h += `<span class="status-badge ${statusClass}">${esc(a.status || "?")}</span>`;
  h += `<span class="action-query">${esc(query)}</span>`;
  h += `<span class="action-time">${fmt_time(a.execution_time)}</span>`;
  h += `</div>`;

  // Detail panel
  h += `<div class="action-detail">`;

  if (a.description) {
    h += `<div class="detail-section"><div class="detail-label">Description</div><div class="detail-text">${esc(a.description)}</div></div>`;
  }
  if (a.dependencies && a.dependencies.length) {
    h += `<div class="detail-section"><div class="detail-label">Dependencies</div><div class="detail-text">${esc(a.dependencies.join(", "))}</div></div>`;
  }
  if (a.preferred_tool) {
    h += `<div class="detail-section"><div class="detail-label">Preferred Tool</div><div class="detail-text">${esc(a.preferred_tool)}</div></div>`;
  }
  if (a.priority != null) {
    h += `<div class="detail-section"><div class="detail-label">Priority</div><div class="detail-text">${a.priority.toFixed(2)}</div></div>`;
  }

  // Actual output / results
  const out = a.actual_output;
  if (out) {
    if (out.error && !out.success) {
      h += `<div class="detail-section"><div class="detail-label">Error</div><div class="detail-text" style="color:var(--danger)">${esc(out.error)}</div></div>`;
    }
    if (out.summary) {
      h += `<div class="detail-section"><div class="detail-label">Summary</div><div class="detail-text">${esc(out.summary)}</div></div>`;
    }

    // Search results with full text
    const results = out.results || out.search_results || [];
    if (results.length) {
      h += `<div class="detail-section"><div class="detail-label">Results (${results.length})</div>`;
      for (let i = 0; i < results.length; i++) {
        const r = results[i];
        h += `<div class="result-card">`;
        h += `<span class="result-rank">#${i+1}</span>`;
        if (r.score != null) h += `<span class="result-score">score: ${r.score.toFixed(4)}</span>`;
        if (r.docid) h += `<span class="result-score" style="margin-left:8px">${esc(r.docid)}</span>`;
        if (r.url) h += `<div class="result-url">${esc(r.url)}</div>`;
        if (r.title) h += `<div style="font-size:13px;font-weight:600;margin-top:2px">${esc(r.title)}</div>`;
        const text = r.text || r.snippet || r.content || "";
        if (text) {
          const needsTrunc = text.length > 500;
          h += `<div class="result-text ${needsTrunc ? "truncated" : ""}">${esc(text)}</div>`;
          if (needsTrunc) h += `<div class="show-more">Show more</div>`;
        }
        h += `</div>`;
      }
      h += `</div>`;
    }

    // Processed URLs
    if (out.urls_processed != null) {
      h += `<div class="detail-section"><div class="detail-label">URLs Processed</div><div class="detail-text">${out.urls_processed}</div></div>`;
    }
    if (out.content_stored_in_vector != null) {
      h += `<div class="detail-section"><div class="detail-label">Stored in Vector DB</div><div class="detail-text">${out.content_stored_in_vector} docs</div></div>`;
    }

    // Catch-all for other fields
    const shownKeys = new Set(["tool", "tool_name", "query", "success", "data_retrieved", "error", "summary", "results", "search_results", "results_count", "source", "content_stored_in_vector", "stored_in_vector", "urls_processed"]);
    const extra = Object.entries(out).filter(([k]) => !shownKeys.has(k));
    if (extra.length) {
      h += `<div class="detail-section"><div class="detail-label">Raw Output</div><pre style="font-size:11px;background:#f8f9fa;padding:8px;border-radius:4px;overflow-x:auto;max-height:200px">${esc(JSON.stringify(Object.fromEntries(extra), null, 2))}</pre></div>`;
    }
  } else if (a.error) {
    h += `<div class="detail-section"><div class="detail-label">Error</div><div class="detail-text" style="color:var(--danger)">${esc(a.error)}</div></div>`;
  }

  h += `</div></div>`;
  return h;
}

render();
</script>
</body>
</html>'''


def main():
    p = argparse.ArgumentParser(description="Generate HTML viewer for a DrBench agent run.")
    p.add_argument("run_dir", type=Path, help="Path to run directory (e.g. runs/batch_X/DR0001)")
    p.add_argument("-o", "--output", type=Path, help="Output HTML file path")
    p.add_argument("--open", action="store_true", help="Open in browser after generating")
    args = p.parse_args()

    if not args.run_dir.exists():
        print(f"[ERROR] Run directory not found: {args.run_dir}", file=sys.stderr)
        return 1

    html_path = generate_html(args.run_dir, args.output)
    print(f"Viewer: {html_path}")

    if args.open:
        subprocess.Popen(["xdg-open", str(html_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return 0


if __name__ == "__main__":
    sys.exit(main())
