#!/usr/bin/env python3
"""Pipeline Viewer — interactive HTML viewer for the multi-hop chain dataset.

Shows every stage of the pipeline in detail:
  Step 1: Seeds (eval.json + inventory)
  Step 2: Query generation
  Step 3: Retrieval candidates
  Step 4-5: Bridge composition & selection
  Step 7: Chain verification

Re-run at any stage to see whatever data is available so far.

Usage:
    python -m making_dataset_2.view_pipeline
    python -m making_dataset_2.view_pipeline --output viewer.html --open
    python -m making_dataset_2.view_pipeline --chains chains.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from making_dataset_2.data_loading import (
    build_doc_lookup,
    load_chunks_local,
    load_eval_seeds,
    load_secrets,
    filter_seed_secrets,
    LocalDoc,
    Secret,
)

COMPANIES = {
    f"DR{i:04d}": name
    for i, name in [
        *[(j, "Lee's Market") for j in range(1, 6)],
        *[(j, "MediConn Solutions") for j in range(6, 11)],
        *[(j, "Elexion Automotive") for j in range(11, 16)],
    ]
}

DEFAULT_INVENTORY = ROOT_DIR / "making_dataset_2" / "outputs" / "secret_inventory.jsonl"
DEFAULT_CHUNKS_LOCAL = ROOT_DIR / "making_dataset" / "outputs" / "chunks_local.jsonl"
DEFAULT_STEP2 = ROOT_DIR / "making_dataset_2" / "outputs" / "step2_all_eval.jsonl"
DEFAULT_OUTPUT = ROOT_DIR / "making_dataset_2" / "outputs" / "pipeline_viewer.html"


def _load_inventory_raw(path: Path) -> list[dict]:
    """Load raw inventory records preserving quality_scores."""
    if not path.exists():
        # Try partial file
        partials = sorted(path.parent.glob(path.name + ".*.partial"))
        if partials:
            path = partials[-1]
        else:
            return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_seed_items(
    eval_seeds: list[Secret],
    inventory_raw: list[dict],
    doc_lookup: dict[str, LocalDoc],
) -> list[dict[str, Any]]:
    """Build unified seed items for both sources."""
    items = []
    idx = 0

    # Eval seeds
    for s in eval_seeds:
        doc = doc_lookup.get(s.doc_id)
        items.append({
            "id": f"eval_{idx:04d}",
            "source": "eval",
            "task_id": doc.meta.get("task_id", "") if doc else "",
            "company": doc.meta.get("company_name", "") if doc else "",
            "question": s.question,
            "answer": s.answer,
            "secret_type": s.secret_type,
            "justification": s.justification,
            "doc_id": s.doc_id,
            "doc_only_check": s.doc_only_check,
            "quality_scores": None,
        })
        idx += 1

    # Inventory secrets
    for record in inventory_raw:
        doc_id = record.get("doc_id", "")
        doc = doc_lookup.get(doc_id)
        for s in record.get("secrets", []):
            items.append({
                "id": f"inv_{idx:04d}",
                "source": "inventory",
                "task_id": doc.meta.get("task_id", "") if doc else "",
                "company": doc.meta.get("company_name", "") if doc else "",
                "question": s.get("question", ""),
                "answer": s.get("answer", ""),
                "secret_type": s.get("secret_type", ""),
                "justification": s.get("justification", ""),
                "doc_id": doc_id,
                "doc_only_check": s.get("doc_only_check", {}),
                "quality_scores": s.get("quality_scores"),
            })
            idx += 1

    return items


def _load_jsonl(path: Path | None) -> list[dict]:
    """Load a JSONL file. Returns [] if missing."""
    if path is None or not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _attach_step2(items: list[dict], step2_path: Path | None) -> None:
    """Attach step2+3 results to matching seed items in-place."""
    records = _load_jsonl(step2_path)
    if not records:
        return

    # Build lookup: (task_id, seed_q) → step2 record
    lookup: dict[tuple[str, str], dict] = {}
    for r in records:
        key = (r.get("task_id", ""), r.get("seed_q", ""))
        lookup[key] = r

    matched = 0
    for it in items:
        key = (it["task_id"], it["question"])
        r = lookup.get(key)
        if r:
            it["step2"] = {
                "query": r.get("query", ""),
                "reasoning": r.get("reasoning", ""),
                "n_hits": r.get("n_hits", 0),
                "top_doc_ids": r.get("top_doc_ids", []),
                "top_scores": r.get("top_scores", []),
                "top_snippets": r.get("top_snippets", []),
            }
            matched += 1
    print(f"  Step 2+3 matched: {matched}/{len(items)} seeds")


def _compute_stats(items: list[dict], chains: list[dict]) -> dict:
    by_source = defaultdict(int)
    by_company = defaultdict(int)
    by_task = defaultdict(int)
    by_type = defaultdict(int)
    for it in items:
        by_source[it["source"]] += 1
        if it["company"]:
            by_company[it["company"]] += 1
        if it["task_id"]:
            by_task[it["task_id"]] += 1
        if it["secret_type"]:
            by_type[it["secret_type"]] += 1

    chain_valid = sum(1 for c in chains if c.get("verification", {}).get("is_valid"))
    chain_complete = sum(1 for c in chains if c.get("metadata", {}).get("complete"))

    # Step 2+3 stats
    step2_count = sum(1 for it in items if it.get("step2"))
    step2_with_hits = sum(1 for it in items if it.get("step2", {}).get("n_hits", 0) > 0)
    all_doc_ids: set[str] = set()
    for it in items:
        s2 = it.get("step2")
        if s2:
            all_doc_ids.update(s2.get("top_doc_ids", []))

    return {
        "total_seeds": len(items),
        "by_source": dict(by_source),
        "by_company": dict(by_company),
        "by_task": dict(by_task),
        "by_type": dict(by_type),
        "total_chains": len(chains),
        "chains_valid": chain_valid,
        "chains_complete": chain_complete,
        "step2_count": step2_count,
        "step2_with_hits": step2_with_hits,
        "step2_unique_docs": len(all_doc_ids),
    }


def _build_doc_texts(items: list[dict], doc_lookup: dict[str, LocalDoc]) -> dict[str, str]:
    """Extract doc texts for items that need them (truncated to keep HTML small)."""
    docs = {}
    seen = set()
    for it in items:
        did = it["doc_id"]
        if did and did not in seen:
            seen.add(did)
            doc = doc_lookup.get(did)
            if doc:
                # Truncate to 8K chars for HTML size
                docs[did] = doc.text[:8000]
    return docs


def generate_html(
    items: list[dict],
    chains: list[dict],
    stats: dict,
    doc_texts: dict[str, str],
) -> str:
    items_json = json.dumps(items, ensure_ascii=False)
    chains_json = json.dumps(chains, ensure_ascii=False)
    stats_json = json.dumps(stats, ensure_ascii=False)
    docs_json = json.dumps(doc_texts, ensure_ascii=False)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pipeline Viewer — Multi-Hop Chain Dataset</title>
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
.stat-eval {{ color: var(--primary); }}
.stat-inv {{ color: var(--web); }}
.stat-valid {{ color: var(--success); }}

/* Tabs */
.tabs {{ display: flex; background: var(--card); border-bottom: 1px solid var(--border); padding: 0 24px; }}
.tab {{ padding: 10px 20px; font-size: 13px; font-weight: 500; cursor: pointer; border-bottom: 2px solid transparent; color: var(--text-muted); transition: all 0.15s; }}
.tab:hover {{ color: var(--text); background: var(--primary-light); }}
.tab.active {{ color: var(--primary); border-bottom-color: var(--primary); }}
.tab .tab-count {{ background: #e9ecef; padding: 1px 7px; border-radius: 10px; font-size: 11px; margin-left: 6px; }}
.tab.active .tab-count {{ background: var(--primary-light); color: var(--primary); }}

/* Filters */
.filters {{ display: flex; gap: 10px; padding: 10px 24px; background: var(--card); border-bottom: 1px solid var(--border); flex-wrap: wrap; align-items: center; }}
select, input[type="text"] {{ padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px; font-size: 13px; background: white; }}
input[type="text"] {{ min-width: 200px; }}
select:focus, input:focus {{ outline: none; border-color: var(--primary); }}
.filter-label {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; }}

/* Layout */
.main {{ display: grid; grid-template-columns: 380px 1fr; height: calc(100vh - 160px); }}
.sidebar {{ background: var(--card); border-right: 1px solid var(--border); overflow-y: auto; }}
.detail {{ overflow-y: auto; padding: 24px; }}

/* Sidebar items */
.seed-item {{ padding: 10px 14px; border-bottom: 1px solid var(--bg); cursor: pointer; transition: background 0.12s; }}
.seed-item:hover {{ background: var(--primary-light); }}
.seed-item.selected {{ background: var(--primary-light); border-left: 3px solid var(--primary); }}
.seed-q {{ font-size: 13px; color: var(--text); line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; margin-bottom: 6px; }}
.seed-meta {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 500; }}
.badge-eval {{ background: var(--primary-light); color: var(--primary); }}
.badge-inv {{ background: #f3e8ff; color: var(--web); }}
.badge-insight {{ background: #dcfce7; color: #166534; }}
.badge-distractor {{ background: #fef3c7; color: #92400e; }}
.badge-kpi {{ background: #dcfce7; color: #166534; }}
.badge-money {{ background: #fef9c3; color: #854d0e; }}
.badge-names {{ background: #fce7f3; color: #9d174d; }}
.badge-emails {{ background: #e0e7ff; color: #3730a3; }}
.badge-dates {{ background: #f3e8ff; color: #6b21a8; }}
.badge-other {{ background: #e5e7eb; color: #374151; }}
.seed-task {{ font-size: 11px; color: var(--text-muted); }}
.seed-answer {{ font-size: 12px; color: var(--success); font-weight: 600; }}

/* Detail panel */
.detail-empty {{ display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-muted); }}

.stage {{ margin-bottom: 20px; background: var(--card); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; box-shadow: var(--shadow); }}
.stage-header {{ padding: 12px 16px; background: var(--bg); border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 10px; cursor: pointer; }}
.stage-header:hover {{ background: #eef0f2; }}
.stage-num {{ width: 26px; height: 26px; background: var(--primary); color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; flex-shrink: 0; }}
.stage-num.pending {{ background: #d1d5db; }}
.stage-title {{ font-weight: 600; font-size: 14px; flex: 1; }}
.stage-status {{ font-size: 11px; padding: 2px 8px; border-radius: 10px; }}
.stage-status.done {{ background: #dcfce7; color: #166534; }}
.stage-status.pending {{ background: #e5e7eb; color: #6b757d; }}
.stage-arrow {{ font-size: 10px; color: var(--text-muted); transition: transform 0.2s; }}
.stage-header.open .stage-arrow {{ transform: rotate(180deg); }}
.stage-body {{ padding: 16px; display: none; }}
.stage-body.open {{ display: block; }}

.field {{ margin-bottom: 12px; }}
.field:last-child {{ margin-bottom: 0; }}
.field-label {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 3px; }}
.field-value {{ font-size: 14px; }}
.field-value.answer {{ color: var(--success); font-weight: 600; }}
.field-value.mono {{ font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px; color: var(--text-muted); }}

.quality-bar {{ display: flex; gap: 12px; flex-wrap: wrap; }}
.quality-item {{ display: flex; align-items: center; gap: 4px; font-size: 12px; }}
.quality-score {{ width: 22px; height: 22px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 600; color: white; }}
.qs-5 {{ background: var(--success); }}
.qs-4 {{ background: #34d399; }}
.qs-3 {{ background: var(--warning); }}
.qs-2 {{ background: #f97316; }}
.qs-1 {{ background: var(--danger); }}

.doc-preview {{ background: #f8f9fa; border: 1px solid var(--border); border-radius: 6px; padding: 12px; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px; line-height: 1.5; max-height: 200px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; }}
.doc-toggle {{ background: var(--primary); color: white; border: none; padding: 4px 10px; border-radius: 4px; font-size: 11px; cursor: pointer; margin-top: 8px; }}
.doc-toggle:hover {{ opacity: 0.9; }}

/* Chain detail */
.hop-flow {{ display: flex; align-items: center; gap: 0; overflow-x: auto; padding: 12px 0; }}
.hop-box {{ width: 140px; padding: 10px; border-radius: 8px; text-align: center; border: 2px solid; background: white; flex-shrink: 0; }}
.hop-box.local {{ border-color: var(--local); background: #eef1ff; }}
.hop-box.web {{ border-color: var(--web); background: #f3e8ff; }}
.hop-label {{ font-size: 10px; font-weight: 700; text-transform: uppercase; margin-bottom: 4px; }}
.hop-label.local {{ color: var(--local); }}
.hop-label.web {{ color: var(--web); }}
.hop-content {{ font-size: 12px; line-height: 1.3; max-height: 3em; overflow: hidden; }}
.hop-arrow {{ padding: 0 8px; font-size: 20px; color: var(--text-muted); flex-shrink: 0; }}

.verif-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }}
.verif-item {{ background: var(--bg); padding: 10px; border-radius: 6px; }}
.verif-label {{ font-size: 11px; color: var(--text-muted); margin-bottom: 3px; }}
.verif-result {{ font-weight: 600; display: flex; align-items: center; gap: 4px; font-size: 13px; }}
.verif-pass {{ color: var(--success); }}
.verif-fail {{ color: var(--danger); }}

.no-results {{ padding: 40px; text-align: center; color: var(--text-muted); }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: grid; grid-template-columns: 380px 1fr; height: calc(100vh - 160px); }}
.tab-content.active.full-width {{ display: block; height: calc(100vh - 160px); overflow-y: auto; padding: 24px; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 7px; height: 7px; }}
::-webkit-scrollbar-track {{ background: #f1f1f1; }}
::-webkit-scrollbar-thumb {{ background: #c1c1c1; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: #a1a1a1; }}

/* Modal */
.modal-overlay {{ position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); z-index: 1000; display: none; }}
.modal-overlay.open {{ display: flex; justify-content: center; align-items: center; }}
.modal {{ background: var(--card); border-radius: 12px; width: 90%; max-width: 900px; max-height: 85vh; display: flex; flex-direction: column; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }}
.modal-header {{ display: flex; justify-content: space-between; align-items: center; padding: 14px 20px; border-bottom: 1px solid var(--border); }}
.modal-title {{ font-size: 15px; font-weight: 600; }}
.modal-close {{ background: none; border: none; font-size: 22px; cursor: pointer; color: var(--text-muted); padding: 4px 8px; border-radius: 4px; }}
.modal-close:hover {{ background: var(--bg); color: var(--text); }}
.modal-body {{ padding: 20px; overflow-y: auto; flex: 1; }}
.modal-body pre {{ white-space: pre-wrap; word-wrap: break-word; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 13px; line-height: 1.6; margin: 0; }}
</style>
</head>
<body>

<header>
  <h1>Pipeline Viewer — Multi-Hop Chain Dataset</h1>
  <div class="stats" id="stats-bar"></div>
</header>

<div class="tabs" id="tabs">
  <div class="tab active" data-tab="seeds">Seeds <span class="tab-count" id="tc-seeds">0</span></div>
  <div class="tab" data-tab="chains">Chains <span class="tab-count" id="tc-chains">0</span></div>
</div>

<div class="filters" id="filters-seeds">
  <div><span class="filter-label">Source</span>
    <select id="f-source"><option value="">All</option><option value="eval">Eval</option><option value="inventory">Inventory</option></select></div>
  <div><span class="filter-label">Company</span>
    <select id="f-company"><option value="">All</option></select></div>
  <div><span class="filter-label">Task</span>
    <select id="f-task"><option value="">All</option></select></div>
  <div><span class="filter-label">Type</span>
    <select id="f-type"><option value="">All</option></select></div>
  <div><span class="filter-label">Step 2</span>
    <select id="f-step2"><option value="">All</option><option value="yes">Has query</option><option value="no">No query</option></select></div>
  <div><span class="filter-label">Search</span>
    <input type="text" id="f-search" placeholder="Question, answer, query..."></div>
</div>

<!-- Seeds tab -->
<div class="tab-content active" id="tab-seeds">
  <div class="sidebar" id="seed-list"></div>
  <div class="detail" id="seed-detail">
    <div class="detail-empty">Select a seed to view pipeline details</div>
  </div>
</div>

<!-- Chains tab -->
<div class="tab-content full-width" id="tab-chains">
  <div id="chains-content"></div>
</div>

<!-- Document modal -->
<div class="modal-overlay" id="doc-modal">
  <div class="modal">
    <div class="modal-header">
      <span class="modal-title" id="modal-title">Document</span>
      <button class="modal-close" onclick="closeModal()">&times;</button>
    </div>
    <div class="modal-body"><pre id="modal-content"></pre></div>
  </div>
</div>

<script>
const ITEMS = {items_json};
const CHAINS = {chains_json};
const STATS = {stats_json};
const DOCS = {docs_json};

let selectedId = null;
let activeTab = "seeds";

// ── Helpers ──────────────────────────────────────────────────────────────
function esc(s) {{ return s ? String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;") : ""; }}
function trunc(s, n) {{ return s && s.length > n ? s.slice(0, n) + "..." : (s || ""); }}

function badgeClass(type) {{
  const m = {{insight:"badge-insight", distractor:"badge-distractor", kpi_numeric:"badge-kpi",
    money:"badge-money", names:"badge-names", emails:"badge-emails", dates:"badge-dates",
    other_sensitive:"badge-other", ids:"badge-other"}};
  return m[type] || "badge-other";
}}

// ── Init ─────────────────────────────────────────────────────────────────
function init() {{
  // Stats
  const sb = document.getElementById("stats-bar");
  sb.innerHTML = `
    <div class="stat"><span class="stat-value">${{STATS.total_seeds}}</span> seeds</div>
    <div class="stat"><span class="stat-value stat-eval">${{STATS.by_source.eval || 0}}</span> eval</div>
    <div class="stat"><span class="stat-value stat-inv">${{STATS.by_source.inventory || 0}}</span> inventory</div>
    <div class="stat"><span class="stat-value">${{STATS.step2_count || 0}}</span> queries <span style="color:var(--success)">${{STATS.step2_with_hits||0}} w/hits</span></div>
    <div class="stat"><span class="stat-value">${{STATS.step2_unique_docs || 0}}</span> unique docs</div>
    <div class="stat"><span class="stat-value stat-valid">${{STATS.total_chains}}</span> chains (${{STATS.chains_valid}} valid)</div>
    <div class="stat">Tasks: ${{Object.keys(STATS.by_task).length}}</div>
  `;
  document.getElementById("tc-seeds").textContent = STATS.total_seeds;
  document.getElementById("tc-chains").textContent = STATS.total_chains;

  // Populate filters
  const companies = [...new Set(ITEMS.map(i => i.company).filter(Boolean))].sort();
  const tasks = [...new Set(ITEMS.map(i => i.task_id).filter(Boolean))].sort();
  const types = [...new Set(ITEMS.map(i => i.secret_type).filter(Boolean))].sort();
  populateSelect("f-company", companies);
  populateSelect("f-task", tasks);
  populateSelect("f-type", types);

  // Events
  ["f-source","f-company","f-task","f-type","f-step2"].forEach(id =>
    document.getElementById(id).addEventListener("change", renderSeedList));
  document.getElementById("f-search").addEventListener("input", renderSeedList);

  document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => switchTab(t.dataset.tab)));

  renderSeedList();
  renderChains();
  if (ITEMS.length > 0) selectSeed(ITEMS[0].id);
}}

function populateSelect(id, values) {{
  const sel = document.getElementById(id);
  values.forEach(v => {{ const o = document.createElement("option"); o.value = v; o.textContent = v; sel.appendChild(o); }});
}}

function switchTab(tab) {{
  activeTab = tab;
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === tab));
  document.querySelectorAll(".tab-content").forEach(c => c.classList.toggle("active", c.id === "tab-" + tab));
  // Show/hide seed filters
  document.getElementById("filters-seeds").style.display = tab === "seeds" ? "flex" : "none";
}}

// ── Seed List ────────────────────────────────────────────────────────────
function getFiltered() {{
  const src = document.getElementById("f-source").value;
  const co = document.getElementById("f-company").value;
  const task = document.getElementById("f-task").value;
  const type = document.getElementById("f-type").value;
  const step2 = document.getElementById("f-step2").value;
  const search = document.getElementById("f-search").value.toLowerCase();
  return ITEMS.filter(it => {{
    if (src && it.source !== src) return false;
    if (co && it.company !== co) return false;
    if (task && it.task_id !== task) return false;
    if (type && it.secret_type !== type) return false;
    if (step2 === "yes" && !it.step2) return false;
    if (step2 === "no" && it.step2) return false;
    if (search) {{
      const hay = (it.question + " " + it.answer + " " + it.doc_id + " " + (it.step2 ? it.step2.query : "")).toLowerCase();
      if (!hay.includes(search)) return false;
    }}
    return true;
  }});
}}

function renderSeedList() {{
  const filtered = getFiltered();
  const container = document.getElementById("seed-list");
  if (!filtered.length) {{ container.innerHTML = '<div class="no-results">No seeds match filters</div>'; return; }}

  container.innerHTML = filtered.map(it => `
    <div class="seed-item ${{it.id === selectedId ? 'selected' : ''}}" data-id="${{it.id}}">
      <div class="seed-q">${{esc(it.question)}}</div>
      <div class="seed-meta">
        <span class="badge ${{it.source === 'eval' ? 'badge-eval' : 'badge-inv'}}">${{it.source}}</span>
        <span class="badge ${{badgeClass(it.secret_type)}}">${{esc(it.secret_type)}}</span>
        <span class="seed-answer">${{esc(trunc(it.answer, 20))}}</span>
        <span class="seed-task">${{esc(it.task_id)}}</span>
      </div>
      ${{it.step2 ? `<div style="font-size:11px;color:var(--primary);margin-top:4px;opacity:0.7">&#x1F50D; ${{esc(trunc(it.step2.query, 60))}}</div>` : ""}}
    </div>
  `).join("");

  container.querySelectorAll(".seed-item").forEach(el =>
    el.addEventListener("click", () => selectSeed(el.dataset.id)));
}}

// ── Seed Detail ──────────────────────────────────────────────────────────
function selectSeed(id) {{
  selectedId = id;
  renderSeedList();
  const it = ITEMS.find(i => i.id === id);
  if (!it) return;
  renderSeedDetail(it);
}}

function renderSeedDetail(it) {{
  const panel = document.getElementById("seed-detail");
  const docText = DOCS[it.doc_id] || "";
  const docShort = docText ? trunc(docText, 500) : "(not loaded)";
  const hasDoc = !!docText;

  // Quality scores
  let qsHtml = "";
  if (it.quality_scores) {{
    qsHtml = `<div class="field"><div class="field-label">Quality Scores</div><div class="quality-bar">` +
      Object.entries(it.quality_scores).map(([k,v]) =>
        `<div class="quality-item"><span class="quality-score qs-${{v}}">${{v}}</span>${{k}}</div>`
      ).join("") + `</div></div>`;
  }}

  // Doc-only check
  let docCheckHtml = "";
  const dc = it.doc_only_check || {{}};
  if (dc.with_doc || dc.without_doc) {{
    docCheckHtml = `<div class="field"><div class="field-label">Doc-Only Check</div>
      <div class="field-value" style="font-size:12px">
        <span style="color:var(--success)">With doc:</span> ${{esc(String(dc.with_doc || ""))}}
        &nbsp;&nbsp;
        <span style="color:var(--danger)">Without doc:</span> ${{esc(String(dc.without_doc || ""))}}
      </div></div>`;
  }}

  panel.innerHTML = `
    <!-- Step 1: Seed -->
    <div class="stage">
      <div class="stage-header open" onclick="toggleStage(this)">
        <span class="stage-num">1</span>
        <span class="stage-title">Seed Selection</span>
        <span class="stage-status done">done</span>
        <span class="stage-arrow">&#9660;</span>
      </div>
      <div class="stage-body open">
        <div class="field"><div class="field-label">Question</div><div class="field-value">${{esc(it.question)}}</div></div>
        <div class="field"><div class="field-label">Answer</div><div class="field-value answer">${{esc(it.answer)}}</div></div>
        <div class="field">
          <div class="field-label">Source / Type / Task</div>
          <div class="field-value">
            <span class="badge ${{it.source === 'eval' ? 'badge-eval' : 'badge-inv'}}">${{it.source}}</span>
            <span class="badge ${{badgeClass(it.secret_type)}}">${{esc(it.secret_type)}}</span>
            &nbsp; ${{esc(it.task_id)}} &mdash; ${{esc(it.company)}}
          </div>
        </div>
        ${{it.justification ? `<div class="field"><div class="field-label">Justification</div><div class="field-value" style="font-size:12px;font-style:italic;color:var(--text-muted)">${{esc(it.justification)}}</div></div>` : ""}}
        ${{qsHtml}}
        ${{docCheckHtml}}
        <div class="field">
          <div class="field-label">Document</div>
          <div class="field-value mono">${{esc(it.doc_id)}}</div>
          ${{hasDoc ? `<div class="doc-preview">${{esc(docShort)}}</div>
            <button class="doc-toggle" onclick="openModal('${{esc(it.doc_id).replace(/'/g, "\\\\'")}}')">View Full Document</button>` : ""}}
        </div>
      </div>
    </div>

    <!-- Step 2: Query -->
    ${{renderStep2(it)}}

    <!-- Step 3: Retrieval -->
    ${{renderStep3(it)}}

    <!-- Step 4-5: Bridge -->
    <div class="stage">
      <div class="stage-header" onclick="toggleStage(this)">
        <span class="stage-num pending">4</span>
        <span class="stage-title">Bridge Composition &amp; Validation</span>
        <span class="stage-status pending">pending</span>
        <span class="stage-arrow">&#9660;</span>
      </div>
      <div class="stage-body">
        <div class="field"><div class="field-label">Description</div>
          <div class="field-value" style="font-size:13px;color:var(--text-muted)">
            For each candidate, the LLM proposes a bridge: a new question extending the chain
            through a shared entity. Bridges are validated for multi-hop dependency, grounding,
            and answer quality. The best bridge is selected.
          </div>
        </div>
      </div>
    </div>

    <!-- Step 7: Verification -->
    <div class="stage">
      <div class="stage-header" onclick="toggleStage(this)">
        <span class="stage-num pending">7</span>
        <span class="stage-title">Verification / Ablation</span>
        <span class="stage-status pending">pending</span>
        <span class="stage-arrow">&#9660;</span>
      </div>
      <div class="stage-body">
        <div class="field"><div class="field-label">Description</div>
          <div class="field-value" style="font-size:13px;color:var(--text-muted)">
            Four ablation tests: (1) no docs &rarr; not answerable, (2) first hop only &rarr; not answerable,
            (3) last hop only &rarr; not answerable, (4) all hops &rarr; answerable. Valid chains pass all four.
          </div>
        </div>
      </div>
    </div>
  `;
}}

function renderStep2(it) {{
  const s2 = it.step2;
  if (!s2) {{
    return `<div class="stage">
      <div class="stage-header" onclick="toggleStage(this)">
        <span class="stage-num pending">2</span>
        <span class="stage-title">Query Generation</span>
        <span class="stage-status pending">no data</span>
        <span class="stage-arrow">&#9660;</span>
      </div>
      <div class="stage-body">
        <div class="field"><div class="field-value" style="font-size:13px;color:var(--text-muted)">
          No step 2 data for this seed. Only eval seeds have query generation results.
        </div></div>
      </div>
    </div>`;
  }}
  return `<div class="stage">
    <div class="stage-header open" onclick="toggleStage(this)">
      <span class="stage-num">2</span>
      <span class="stage-title">Query Generation</span>
      <span class="stage-status done">done</span>
      <span class="stage-arrow">&#9660;</span>
    </div>
    <div class="stage-body open">
      <div class="field"><div class="field-label">Generated Query</div>
        <div class="field-value" style="font-size:15px;font-weight:600;color:var(--primary)">${{esc(s2.query)}}</div>
      </div>
      ${{s2.reasoning && s2.reasoning !== s2.query ? `<div class="field"><div class="field-label">LLM Reasoning</div>
        <div class="doc-preview" style="max-height:150px">${{esc(s2.reasoning)}}</div>
      </div>` : ""}}
    </div>
  </div>`;
}}

function renderStep3(it) {{
  const s2 = it.step2;
  if (!s2) {{
    return `<div class="stage">
      <div class="stage-header" onclick="toggleStage(this)">
        <span class="stage-num pending">3</span>
        <span class="stage-title">Retrieval</span>
        <span class="stage-status pending">no data</span>
        <span class="stage-arrow">&#9660;</span>
      </div>
      <div class="stage-body">
        <div class="field"><div class="field-value" style="font-size:13px;color:var(--text-muted)">
          No retrieval data for this seed.
        </div></div>
      </div>
    </div>`;
  }}
  const hitCount = s2.n_hits || 0;
  const ids = s2.top_doc_ids || [];
  const scores = s2.top_scores || [];
  const snippets = s2.top_snippets || [];

  let hitsHtml = "";
  for (let i = 0; i < ids.length; i++) {{
    const docId = ids[i] || "";
    const score = scores[i] !== undefined ? scores[i].toFixed(3) : "?";
    const snippet = snippets[i] || "";
    const shortId = docId.length > 50 ? "..." + docId.slice(-47) : docId;
    hitsHtml += `<div style="background:var(--bg);padding:8px 10px;border-radius:6px;margin-bottom:6px;border-left:3px solid var(--web)">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:12px;font-weight:600;color:var(--web)">#${{i+1}}</span>
        <span style="font-size:11px;color:var(--text-muted);font-family:monospace">${{esc(shortId)}}</span>
        <span style="font-size:11px;background:#e9ecef;padding:1px 6px;border-radius:4px">${{score}}</span>
      </div>
      ${{snippet ? `<div style="font-size:11px;color:var(--text-muted);margin-top:4px;line-height:1.4">${{esc(snippet)}}</div>` : ""}}
    </div>`;
  }}

  return `<div class="stage">
    <div class="stage-header open" onclick="toggleStage(this)">
      <span class="stage-num">3</span>
      <span class="stage-title">Retrieval</span>
      <span class="stage-status done">${{hitCount}} hits</span>
      <span class="stage-arrow">&#9660;</span>
    </div>
    <div class="stage-body open">
      <div class="field"><div class="field-label">Top ${{ids.length}} Results (of ${{hitCount}} total)</div>
        ${{hitsHtml}}
      </div>
    </div>
  </div>`;
}}

function toggleStage(header) {{
  header.classList.toggle("open");
  const body = header.nextElementSibling;
  body.classList.toggle("open");
}}

// ── Chains Tab ───────────────────────────────────────────────────────────
function renderChains() {{
  const container = document.getElementById("chains-content");
  if (!CHAINS.length) {{
    container.innerHTML = '<div class="no-results">No chains yet. Run chain_builder.py to generate chains.</div>';
    return;
  }}

  container.innerHTML = CHAINS.map((chain, ci) => {{
    const meta = chain.metadata || {{}};
    const hops = chain.hops || [];
    const verif = chain.verification || {{}};
    const isValid = verif.is_valid;
    const statusBadge = isValid === true ? '<span class="badge badge-insight">VALID</span>'
      : isValid === false ? '<span class="badge" style="background:#fee2e2;color:#991b1b">INVALID</span>'
      : '<span class="badge badge-other">UNVERIFIED</span>';

    // Hop flow
    const hopFlow = hops.map((h, i) => {{
      const cls = h.hop_type === "L" ? "local" : "web";
      const label = h.hop_type === "L" ? "LOCAL" : "WEB";
      return `<div class="hop-box ${{cls}}">
        <div class="hop-label ${{cls}}">${{label}} ${{h.hop_number}}</div>
        <div class="hop-content">${{esc(trunc(h.question, 60))}}</div>
        <div style="font-size:11px;color:var(--success);font-weight:600;margin-top:4px">${{esc(trunc(h.answer, 30))}}</div>
      </div>${{i < hops.length - 1 ? '<div class="hop-arrow">&rarr;</div>' : ''}}`;
    }}).join("");

    // Verification
    const verifItems = [
      ["no_docs", "No docs", false], ["first_only", "First only", false],
      ["last_only", "Last only", false], ["all_docs", "All docs", true]
    ];
    const verifHtml = verifItems.map(([key, label, expect]) => {{
      const pass = verif[key + "_pass"];
      if (pass === undefined) return "";
      const ok = expect ? pass : !pass;
      return `<div class="verif-item">
        <div class="verif-label">${{label}}</div>
        <div class="verif-result ${{ok ? 'verif-pass' : 'verif-fail'}}">${{ok ? '&#10003;' : '&#10007;'}} ${{pass ? 'Answerable' : 'Not answerable'}}</div>
      </div>`;
    }}).join("");

    return `
    <div class="stage" style="margin-bottom:16px">
      <div class="stage-header open" onclick="toggleStage(this)">
        <span class="stage-num">${{ci+1}}</span>
        <span class="stage-title">${{esc(chain.chain_id)}} &mdash; ${{esc(chain.pattern)}} ${{statusBadge}}</span>
        <span style="font-size:12px;color:var(--text-muted)">${{esc(meta.task_id || '')}} / ${{esc(meta.company || '')}}</span>
        <span class="stage-arrow">&#9660;</span>
      </div>
      <div class="stage-body open">
        <div class="field"><div class="field-label">Question</div><div class="field-value">${{esc(chain.global_question)}}</div></div>
        <div class="field"><div class="field-label">Answer</div><div class="field-value answer">${{esc(chain.global_answer)}}</div></div>
        <div class="field"><div class="field-label">Hop Flow</div><div class="hop-flow">${{hopFlow}}</div></div>
        ${{verifHtml ? `<div class="field"><div class="field-label">Verification</div><div class="verif-grid">${{verifHtml}}</div></div>` : ''}}
        <div class="field" style="font-size:12px;color:var(--text-muted)">
          ${{meta.llm_calls ? meta.llm_calls + ' LLM calls' : ''}}
          ${{meta.elapsed_seconds ? ' &middot; ' + meta.elapsed_seconds + 's' : ''}}
        </div>

        <div class="field"><div class="field-label">Hops Detail</div></div>
        ${{hops.map(h => `
          <div style="background:var(--bg);padding:10px;border-radius:6px;margin-bottom:8px;border-left:3px solid var(${{h.hop_type==='L'?'--local':'--web'}})">
            <div style="font-weight:600;font-size:13px;color:var(${{h.hop_type==='L'?'--local':'--web'}})">${{h.hop_type==='L'?'Local':'Web'}} Hop ${{h.hop_number}}</div>
            <div style="font-size:12px;margin-top:4px"><strong>Q:</strong> ${{esc(h.question)}}</div>
            <div style="font-size:12px;color:var(--success)"><strong>A:</strong> ${{esc(h.answer)}}</div>
            <div style="font-size:11px;color:var(--text-muted);font-family:monospace;margin-top:2px">${{esc(h.doc_id)}}</div>
          </div>
        `).join("")}}
      </div>
    </div>`;
  }}).join("");
}}

// ── Modal ────────────────────────────────────────────────────────────────
function openModal(docId) {{
  const text = DOCS[docId];
  if (!text) return;
  document.getElementById("modal-title").textContent = docId.split("/").pop();
  document.getElementById("modal-content").textContent = text;
  document.getElementById("doc-modal").classList.add("open");
  document.body.style.overflow = "hidden";
}}
function closeModal() {{
  document.getElementById("doc-modal").classList.remove("open");
  document.body.style.overflow = "";
}}
document.getElementById("doc-modal").addEventListener("click", function(e) {{ if (e.target === this) closeModal(); }});
document.addEventListener("keydown", function(e) {{ if (e.key === "Escape") closeModal(); }});

init();
</script>
</body>
</html>
'''


def main() -> int:
    p = argparse.ArgumentParser(description="Generate pipeline viewer HTML")
    p.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT))
    p.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    p.add_argument("--chunks-local", default=str(DEFAULT_CHUNKS_LOCAL))
    p.add_argument("--step2", default=str(DEFAULT_STEP2), help="Step 2+3 results JSONL")
    p.add_argument("--chains", default=None, help="Chain output JSONL (optional)")
    p.add_argument("--open", action="store_true", help="Open in browser")
    args = p.parse_args()

    print("Loading chunks...")
    chunks = load_chunks_local(Path(args.chunks_local))
    doc_lookup = build_doc_lookup(chunks)
    print(f"  {len(doc_lookup)} documents")

    print("Loading eval seeds...")
    eval_seeds = load_eval_seeds(doc_lookup)
    print(f"  {len(eval_seeds)} eval seeds")

    print("Loading inventory secrets...")
    inventory_raw = _load_inventory_raw(Path(args.inventory))
    inv_count = sum(len(r.get("secrets", [])) for r in inventory_raw)
    print(f"  {len(inventory_raw)} docs, {inv_count} secrets")

    items = _build_seed_items(eval_seeds, inventory_raw, doc_lookup)
    print(f"  {len(items)} total seed items")

    print("Loading step 2+3 results...")
    _attach_step2(items, Path(args.step2))

    chains = _load_jsonl(Path(args.chains) if args.chains else None)
    if chains:
        print(f"  {len(chains)} chains loaded")

    stats = _compute_stats(items, chains)
    doc_texts = _build_doc_texts(items, doc_lookup)
    print(f"  {len(doc_texts)} doc texts for viewer")

    print("Generating HTML...")
    html = generate_html(items, chains, stats, doc_texts)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Written to: {out}")
    print(f"File size: {out.stat().st_size / 1024:.1f} KB")

    if args.open:
        try:
            subprocess.run(["xdg-open", str(out)], check=False)
        except Exception:
            print(f"Open manually: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
