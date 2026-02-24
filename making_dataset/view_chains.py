#!/usr/bin/env python3
"""
Generate interactive HTML viewer for multi-hop reasoning chains.

Usage:
    python making_dataset/view_chains.py [--input FILE] [--output FILE] [--open]

Input format (JSONL from chain_builder.py):
{
  "chain_id": "chain_001",
  "pattern": "LW",
  "task_id": "DR0001",
  "answer_source": "W",
  "hops": [
    {"type": "L", "doc_id": "local/DR0001/...", "fact": "SAP", "company": "Lee's Market",
     "task_id": "DR0001", "secret_question": "What vendor...?", "secret_answer": "SAP"},
    {"type": "W", "doc_id": "web/12345", "fact": "Germany"}
  ],
  "bridges": [
    {"link": "SAP", "relation": "founded in", "answer": "Germany", "type": "L→W", "from": 0, "to": 1}
  ],
  "question": "Where was the company's vendor founded?",
  "answer": "Germany",
  "verification": {
    "valid": true,
    "no_docs": {"answerable": false, "reason": "Missing company info"},
    "first_only": {"answerable": false, "reason": "Missing location"},
    "last_only": {"answerable": false, "reason": "Missing which company"},
    "all_hops": {"answerable": true, "reason": "Germany"}
  }
}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any

COMPANIES = {
    "DR0001": "Lee's Market", "DR0002": "Lee's Market", "DR0003": "Lee's Market",
    "DR0004": "Lee's Market", "DR0005": "Lee's Market",
    "DR0006": "MediConn Solutions", "DR0007": "MediConn Solutions", "DR0008": "MediConn Solutions",
    "DR0009": "MediConn Solutions", "DR0010": "MediConn Solutions",
    "DR0011": "Elexion Automotive", "DR0012": "Elexion Automotive", "DR0013": "Elexion Automotive",
    "DR0014": "Elexion Automotive", "DR0015": "Elexion Automotive",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file."""
    chains = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chains.append(json.loads(line))
    return chains


def load_documents(path: Path) -> dict[str, str]:
    """Load documents from docs_local.jsonl into a dict keyed by doc_id."""
    docs = {}
    if not path.exists():
        return docs
    for record in load_jsonl(path):
        doc_id = record.get("doc_id", "")
        text = record.get("text", "")
        if doc_id:
            docs[doc_id] = text
    return docs


def get_chain_company(chain: dict) -> str | None:
    """Extract company from chain's local hops."""
    for hop in chain.get("hops", []):
        if hop.get("company"):
            return hop["company"]
        doc_id = hop.get("doc_id", "")
        if doc_id.startswith("local/"):
            parts = doc_id.split("/")
            if len(parts) >= 2:
                task_id = parts[1]
                if task_id in COMPANIES:
                    return COMPANIES[task_id]
    return None


def compute_stats(chains: list[dict]) -> dict[str, Any]:
    """Compute aggregate statistics."""
    stats = {
        "total": len(chains),
        "valid": 0,
        "invalid": 0,
        "unverified": 0,
        "by_pattern": defaultdict(int),
        "by_company": defaultdict(int),
        "by_length": defaultdict(int),
        "patterns": set(),
        "companies": set(),
    }

    for chain in chains:
        pattern = chain.get("pattern", "?")
        stats["by_pattern"][pattern] += 1
        stats["patterns"].add(pattern)
        stats["by_length"][len(pattern)] += 1

        company = get_chain_company(chain)
        if company:
            stats["by_company"][company] += 1
            stats["companies"].add(company)

        verification = chain.get("verification", {})
        if not verification:
            stats["unverified"] += 1
        elif verification.get("valid"):
            stats["valid"] += 1
        else:
            stats["invalid"] += 1

    stats["patterns"] = sorted(stats["patterns"])
    stats["companies"] = sorted(stats["companies"])
    return stats


def generate_html(chains: list[dict], stats: dict, documents: dict[str, str]) -> str:
    """Generate self-contained HTML viewer."""

    chains_json = json.dumps(chains, ensure_ascii=False)
    stats_json = json.dumps({
        "total": stats["total"],
        "valid": stats["valid"],
        "invalid": stats["invalid"],
        "unverified": stats["unverified"],
        "by_pattern": dict(stats["by_pattern"]),
        "by_company": dict(stats["by_company"]),
        "by_length": dict(stats["by_length"]),
        "patterns": stats["patterns"],
        "companies": stats["companies"],
    }, ensure_ascii=False)
    docs_json = json.dumps(documents, ensure_ascii=False)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Multi-Hop Chain Viewer</title>
<style>
:root {{
  --bg: #f8f9fa;
  --card: #ffffff;
  --primary: #4361ee;
  --primary-light: #eef1ff;
  --local: #4361ee;
  --web: #7209b7;
  --success: #10b981;
  --warning: #f59e0b;
  --danger: #ef4444;
  --text: #212529;
  --text-muted: #6c757d;
  --border: #dee2e6;
  --shadow: 0 2px 8px rgba(0,0,0,0.08);
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; }}

header {{ background: var(--card); padding: 16px 24px; border-bottom: 1px solid var(--border); box-shadow: var(--shadow); }}
header h1 {{ font-size: 20px; font-weight: 600; margin-bottom: 8px; color: var(--primary); }}
.stats {{ display: flex; gap: 24px; flex-wrap: wrap; font-size: 14px; color: var(--text-muted); }}
.stat {{ display: flex; align-items: center; gap: 6px; }}
.stat-value {{ color: var(--text); font-weight: 600; }}
.stat-valid {{ color: var(--success); }}
.stat-invalid {{ color: var(--danger); }}
.stat-unverified {{ color: var(--warning); }}

.filters {{ background: var(--card); padding: 12px 24px; border-bottom: 1px solid var(--border); display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }}
.filter-group {{ display: flex; align-items: center; gap: 8px; }}
.filter-group label {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; }}
select, input[type="text"] {{ background: white; border: 1px solid var(--border); color: var(--text); padding: 8px 12px; border-radius: 6px; font-size: 14px; }}
select:focus, input:focus {{ outline: none; border-color: var(--primary); }}

.main {{ display: grid; grid-template-columns: 360px 1fr; height: calc(100vh - 140px); max-width: 1800px; margin: 0 auto; }}
.sidebar {{ background: var(--card); border-right: 1px solid var(--border); overflow-y: auto; }}
.detail {{ overflow-y: auto; padding: 24px; }}

.chain-list {{ }}
.chain-item {{ padding: 12px 16px; border-bottom: 1px solid var(--bg); cursor: pointer; transition: background 0.15s; }}
.chain-item:hover {{ background: var(--primary-light); }}
.chain-item.selected {{ background: var(--primary-light); border-left: 3px solid var(--primary); }}
.chain-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }}
.chain-id {{ font-weight: 600; font-size: 14px; }}
.chain-pattern {{ font-family: monospace; background: #e9ecef; padding: 2px 8px; border-radius: 4px; font-size: 12px; color: var(--text); }}
.chain-question {{ font-size: 13px; color: var(--text-muted); line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }}
.chain-meta {{ display: flex; gap: 12px; margin-top: 8px; font-size: 12px; }}
.chain-company {{ color: var(--primary); }}
.validity-badge {{ padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 500; }}
.validity-valid {{ background: #dcfce7; color: #166534; }}
.validity-invalid {{ background: #fee2e2; color: #991b1b; }}
.validity-unverified {{ background: #fef3c7; color: #92400e; }}

.detail-empty {{ display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-muted); }}

.chain-detail {{ }}
.detail-header {{ margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }}
.detail-title {{ font-size: 18px; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center; gap: 12px; }}
.detail-question {{ background: var(--card); border: 1px solid var(--border); padding: 16px; border-radius: 8px; margin-bottom: 12px; box-shadow: var(--shadow); }}
.detail-question-label {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px; }}
.detail-question-text {{ font-size: 15px; line-height: 1.5; }}
.detail-answer {{ display: flex; align-items: center; gap: 8px; }}
.detail-answer-label {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; }}
.detail-answer-text {{ font-weight: 600; color: var(--success); }}
.answer-source {{ font-size: 12px; color: var(--text-muted); font-weight: normal; }}
.detail-meta {{ display: flex; gap: 16px; margin-top: 8px; font-size: 13px; color: var(--text-muted); }}
.detail-meta span {{ display: flex; gap: 4px; }}

.chain-viz {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 24px; margin-bottom: 24px; overflow-x: auto; box-shadow: var(--shadow); }}
.viz-container {{ display: flex; align-items: center; gap: 0; min-width: max-content; justify-content: center; }}
.hop-node {{ display: flex; flex-direction: column; align-items: center; position: relative; }}
.hop-box {{ width: 130px; padding: 12px; border-radius: 8px; text-align: center; border: 2px solid; background: white; }}
.hop-local {{ border-color: var(--local); background: #eef1ff; }}
.hop-web {{ border-color: var(--web); background: #f3e8ff; }}
.hop-type {{ font-size: 11px; font-weight: 700; margin-bottom: 4px; text-transform: uppercase; }}
.hop-type-L {{ color: var(--local); }}
.hop-type-W {{ color: var(--web); }}
.hop-fact {{ font-size: 12px; color: var(--text); line-height: 1.3; max-height: 3.9em; overflow: hidden; }}
.hop-index {{ position: absolute; top: -8px; left: -8px; width: 22px; height: 22px; background: var(--text); color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 600; }}

.bridge {{ display: flex; flex-direction: column; align-items: center; width: 100px; position: relative; }}
.bridge-arrow {{ color: var(--text-muted); font-size: 24px; }}
.bridge-label {{ font-size: 11px; color: var(--text-muted); text-align: center; max-width: 90px; line-height: 1.2; background: #e9ecef; padding: 3px 8px; border-radius: 4px; }}
.bridge-type {{ font-size: 10px; color: var(--text-muted); margin-top: 2px; }}

.section {{ margin-bottom: 24px; }}
.section-title {{ font-size: 13px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }}

.verification {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }}
.verif-item {{ background: var(--card); border: 1px solid var(--border); padding: 12px; border-radius: 8px; }}
.verif-label {{ font-size: 12px; color: var(--text-muted); margin-bottom: 4px; }}
.verif-value {{ font-weight: 600; display: flex; align-items: center; gap: 6px; }}
.verif-pass {{ color: var(--success); }}
.verif-fail {{ color: var(--danger); }}
.verif-icon {{ font-size: 1rem; }}
.verif-reason {{ font-size: 11px; color: var(--text-muted); margin-top: 4px; font-style: italic; }}

.hops-list {{ display: flex; flex-direction: column; gap: 12px; }}
.hop-detail {{ background: var(--card); border: 1px solid var(--border); padding: 16px; border-radius: 8px; border-left: 3px solid; box-shadow: var(--shadow); }}
.hop-detail.local {{ border-left-color: var(--local); }}
.hop-detail.web {{ border-left-color: var(--web); }}
.hop-detail-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
.hop-detail-type {{ font-weight: 600; font-size: 14px; }}
.hop-detail-type.local {{ color: var(--local); }}
.hop-detail-type.web {{ color: var(--web); }}
.hop-detail-idx {{ font-size: 12px; color: var(--text-muted); }}
.hop-detail-row {{ display: flex; margin-bottom: 6px; }}
.hop-detail-label {{ width: 80px; font-size: 12px; color: var(--text-muted); flex-shrink: 0; }}
.hop-detail-value {{ font-size: 13px; word-break: break-word; }}
.hop-detail-doc {{ font-family: monospace; font-size: 11px; color: var(--text-muted); }}

.bridges-list {{ display: flex; flex-direction: column; gap: 8px; }}
.bridge-detail {{ background: var(--card); border: 1px solid var(--border); padding: 12px; border-radius: 8px; display: flex; align-items: center; gap: 16px; }}
.bridge-indices {{ font-family: monospace; color: var(--text-muted); font-size: 14px; min-width: 60px; }}
.bridge-link {{ flex: 1; }}
.bridge-link-label {{ font-size: 12px; color: var(--text-muted); }}
.bridge-link-value {{ font-size: 14px; }}

.no-results {{ padding: 40px; text-align: center; color: var(--text-muted); }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: #f1f1f1; }}
::-webkit-scrollbar-thumb {{ background: #c1c1c1; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: #a1a1a1; }}

/* Document Modal */
.modal-overlay {{
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0,0,0,0.5);
  z-index: 1000;
  display: none;
}}
.modal-overlay.open {{ display: flex; justify-content: center; align-items: center; }}
.modal {{
  background: var(--card);
  border-radius: 12px;
  width: 90%;
  max-width: 900px;
  max-height: 85vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}}
.modal-header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
}}
.modal-title {{
  font-size: 16px;
  font-weight: 600;
  color: var(--text);
}}
.modal-close {{
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: var(--text-muted);
  padding: 4px 8px;
  border-radius: 4px;
}}
.modal-close:hover {{ background: var(--bg); color: var(--text); }}
.modal-body {{
  padding: 20px;
  overflow-y: auto;
  flex: 1;
}}
.modal-body pre {{
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
  font-size: 13px;
  line-height: 1.6;
  color: var(--text);
  margin: 0;
}}
.view-doc-btn {{
  background: var(--local);
  color: white;
  border: none;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 10px;
  cursor: pointer;
  margin-left: 8px;
}}
.view-doc-btn:hover {{ opacity: 0.9; }}

/* Pipeline Reference */
.pipeline-toggle {{
  background: var(--primary-light);
  border: 1px solid var(--border);
  padding: 12px 16px;
  border-radius: 8px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}}
.pipeline-toggle:hover {{ background: #e0e7ff; }}
.pipeline-toggle-text {{ font-weight: 600; font-size: 14px; color: var(--primary); }}
.pipeline-toggle-arrow {{ font-size: 12px; color: var(--text-muted); transition: transform 0.2s; }}
.pipeline-toggle.open .pipeline-toggle-arrow {{ transform: rotate(180deg); }}
.pipeline-content {{ display: none; }}
.pipeline-content.open {{ display: block; }}

.pipeline-stages {{ display: flex; flex-direction: column; gap: 16px; }}
.pipeline-stage {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
.pipeline-stage-header {{
  padding: 12px 16px;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 12px;
}}
.pipeline-stage-num {{
  width: 28px;
  height: 28px;
  background: var(--primary);
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 600;
  flex-shrink: 0;
}}
.pipeline-stage-title {{ font-weight: 600; font-size: 14px; }}
.pipeline-stage-desc {{ font-size: 12px; color: var(--text-muted); margin-left: auto; }}
.pipeline-stage-body {{ padding: 16px; }}
.pipeline-stage-row {{ margin-bottom: 12px; }}
.pipeline-stage-row:last-child {{ margin-bottom: 0; }}
.pipeline-label {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px; }}
.pipeline-value {{ font-size: 13px; }}
.pipeline-prompt {{
  background: #1e293b;
  color: #e2e8f0;
  padding: 12px;
  border-radius: 6px;
  font-family: ui-monospace, SFMono-Regular, monospace;
  font-size: 11px;
  line-height: 1.5;
  white-space: pre-wrap;
  overflow-x: auto;
  max-height: 200px;
  overflow-y: auto;
}}
.pipeline-arrow {{
  display: flex;
  justify-content: center;
  padding: 8px 0;
  color: var(--text-muted);
  font-size: 20px;
}}
.pipeline-io {{ display: flex; gap: 24px; flex-wrap: wrap; }}
.pipeline-io-item {{ flex: 1; min-width: 200px; }}
.pipeline-io-label {{ font-size: 10px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px; }}
.pipeline-io-value {{ font-size: 12px; background: var(--bg); padding: 8px; border-radius: 4px; }}
</style>
</head>
<body>

<header>
    <h1>Multi-Hop Chain Viewer</h1>
    <div class="stats">
        <div class="stat"><span class="stat-value" id="stat-total">0</span> chains</div>
        <div class="stat"><span class="stat-value stat-valid" id="stat-valid">0</span> valid</div>
        <div class="stat"><span class="stat-value stat-invalid" id="stat-invalid">0</span> invalid</div>
        <div class="stat"><span class="stat-value stat-unverified" id="stat-unverified">0</span> unverified</div>
        <div class="stat">Patterns: <span class="stat-value" id="stat-patterns">-</span></div>
    </div>
</header>

<div class="filters">
    <div class="filter-group">
        <label>Pattern</label>
        <select id="filter-pattern">
            <option value="">All</option>
        </select>
    </div>
    <div class="filter-group">
        <label>Length</label>
        <select id="filter-length">
            <option value="">All</option>
        </select>
    </div>
    <div class="filter-group">
        <label>Company</label>
        <select id="filter-company">
            <option value="">All</option>
        </select>
    </div>
    <div class="filter-group">
        <label>Status</label>
        <select id="filter-status">
            <option value="">All</option>
            <option value="valid">Valid</option>
            <option value="invalid">Invalid</option>
            <option value="unverified">Unverified</option>
        </select>
    </div>
    <div class="filter-group">
        <label>Search</label>
        <input type="text" id="filter-search" placeholder="Question, fact, link...">
    </div>
</div>

<div class="main">
    <div class="sidebar">
        <div class="chain-list" id="chain-list"></div>
    </div>
    <div class="detail" id="detail">
        <div class="detail-empty">Select a chain to view details</div>
    </div>
</div>

<div class="modal-overlay" id="doc-modal">
  <div class="modal">
    <div class="modal-header">
      <span class="modal-title" id="modal-title">Document</span>
      <button class="modal-close" onclick="closeDocModal()">&times;</button>
    </div>
    <div class="modal-body">
      <pre id="modal-content"></pre>
    </div>
  </div>
</div>

<script>
const CHAINS = {chains_json};
const STATS = {stats_json};
const DOCS = {docs_json};

let selectedChainId = null;
let filteredChains = CHAINS;

function getChainCompany(chain) {{
    for (const hop of (chain.hops || [])) {{
        if (hop.company) return hop.company;
    }}
    return null;
}}

function getVerificationStatus(chain) {{
    const v = chain.verification;
    if (!v || Object.keys(v).length === 0) return "unverified";
    return v.valid ? "valid" : "invalid";
}}

// Helper to get answerable value from verification item (handles both nested and flat format)
function getVerifAnswerable(verification, key) {{
    const val = verification[key];
    if (val === undefined) return undefined;
    if (typeof val === 'object' && val !== null) return val.answerable;
    return val;
}}

function getVerifReason(verification, key) {{
    const val = verification[key];
    if (typeof val === 'object' && val !== null) return val.reason || '';
    return '';
}}

function escapeHtml(str) {{
    if (!str) return "";
    return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}}

function truncate(str, len) {{
    if (!str) return "";
    return str.length > len ? str.slice(0, len) + "..." : str;
}}

function initFilters() {{
    // Patterns
    const patternSelect = document.getElementById("filter-pattern");
    STATS.patterns.forEach(p => {{
        const opt = document.createElement("option");
        opt.value = p;
        opt.textContent = p;
        patternSelect.appendChild(opt);
    }});

    // Lengths
    const lengths = Object.keys(STATS.by_length).sort((a, b) => a - b);
    const lengthSelect = document.getElementById("filter-length");
    lengths.forEach(l => {{
        const opt = document.createElement("option");
        opt.value = l;
        opt.textContent = l + "-hop";
        lengthSelect.appendChild(opt);
    }});

    // Companies
    const companySelect = document.getElementById("filter-company");
    STATS.companies.forEach(c => {{
        const opt = document.createElement("option");
        opt.value = c;
        opt.textContent = c;
        companySelect.appendChild(opt);
    }});

    // Stats
    document.getElementById("stat-total").textContent = STATS.total;
    document.getElementById("stat-valid").textContent = STATS.valid;
    document.getElementById("stat-invalid").textContent = STATS.invalid;
    document.getElementById("stat-unverified").textContent = STATS.unverified;
    document.getElementById("stat-patterns").textContent = STATS.patterns.length;
}}

function applyFilters() {{
    const pattern = document.getElementById("filter-pattern").value;
    const length = document.getElementById("filter-length").value;
    const company = document.getElementById("filter-company").value;
    const status = document.getElementById("filter-status").value;
    const search = document.getElementById("filter-search").value.toLowerCase();

    filteredChains = CHAINS.filter(chain => {{
        if (pattern && chain.pattern !== pattern) return false;
        if (length && chain.pattern.length !== parseInt(length)) return false;
        if (company && getChainCompany(chain) !== company) return false;
        if (status && getVerificationStatus(chain) !== status) return false;
        if (search) {{
            const text = [
                chain.question || "",
                chain.answer || "",
                ...(chain.hops || []).map(h => h.fact || ""),
                ...(chain.bridges || []).map(b => b.link || ""),
            ].join(" ").toLowerCase();
            if (!text.includes(search)) return false;
        }}
        return true;
    }});

    renderChainList();
}}

function renderChainList() {{
    const container = document.getElementById("chain-list");

    if (filteredChains.length === 0) {{
        container.innerHTML = '<div class="no-results">No chains match filters</div>';
        return;
    }}

    container.innerHTML = filteredChains.map((chain, idx) => {{
        const status = getVerificationStatus(chain);
        const company = getChainCompany(chain);
        const isSelected = chain.chain_id === selectedChainId;

        return `
            <div class="chain-item ${{isSelected ? 'selected' : ''}}" data-idx="${{idx}}">
                <div class="chain-header">
                    <span class="chain-id">${{escapeHtml(chain.chain_id || `Chain ${{idx + 1}}`)}}</span>
                    <span class="chain-pattern">${{escapeHtml(chain.pattern || '?')}}</span>
                </div>
                <div class="chain-question">${{escapeHtml(chain.question || 'No question')}}</div>
                <div class="chain-meta">
                    ${{company ? `<span class="chain-company">${{escapeHtml(company)}}</span>` : ''}}
                    <span class="validity-badge validity-${{status}}">${{status.toUpperCase()}}</span>
                </div>
            </div>
        `;
    }}).join("");

    container.querySelectorAll(".chain-item").forEach(item => {{
        item.addEventListener("click", () => {{
            const idx = parseInt(item.dataset.idx);
            selectChain(filteredChains[idx]);
        }});
    }});
}}

function selectChain(chain) {{
    selectedChainId = chain.chain_id;
    renderChainList();
    renderDetail(chain);
}}

function renderDetail(chain) {{
    const container = document.getElementById("detail");
    const status = getVerificationStatus(chain);
    const company = getChainCompany(chain);
    const hops = chain.hops || [];
    const bridges = chain.bridges || [];
    const verification = chain.verification || {{}};

    // Build visualization
    let vizHtml = '';
    hops.forEach((hop, i) => {{
        const typeClass = hop.type === 'L' ? 'local' : 'web';
        vizHtml += `
            <div class="hop-node">
                <div class="hop-box hop-${{typeClass}}">
                    <span class="hop-index">${{i}}</span>
                    <div class="hop-type hop-type-${{hop.type}}">${{hop.type === 'L' ? 'LOCAL' : 'WEB'}}</div>
                    <div class="hop-fact">${{escapeHtml(truncate(hop.fact || hop.text || '', 60))}}</div>
                </div>
            </div>
        `;

        // Add bridge arrow if not last hop
        if (i < hops.length - 1) {{
            const bridge = bridges[i] || {{}};
            vizHtml += `
                <div class="bridge">
                    <span class="bridge-arrow">\u2192</span>
                    <span class="bridge-label">${{escapeHtml(truncate(bridge.link || '?', 30))}}</span>
                    <span class="bridge-type">${{escapeHtml(bridge.type || bridge.bridge_type || '')}}</span>
                </div>
            `;
        }}
    }});

    // Build verification section
    const verifItems = [
        {{ key: 'no_docs', label: 'Without any docs', expect: false, desc: 'Cannot answer without documents' }},
        {{ key: 'first_only', label: 'First hop only', expect: false, desc: 'Cannot answer with only first hop' }},
        {{ key: 'last_only', label: 'Last hop only', expect: false, desc: 'Cannot answer with only last hop' }},
        {{ key: 'all_hops', label: 'All hops', expect: true, desc: 'Can answer with all hops' }},
    ];

    const verifHtml = verifItems.map(item => {{
        const value = getVerifAnswerable(verification, item.key);
        const reason = getVerifReason(verification, item.key);
        const hasValue = value !== undefined;
        const passed = hasValue && (item.expect ? value : !value);
        return `
            <div class="verif-item">
                <div class="verif-label">${{item.label}}</div>
                <div class="verif-value ${{hasValue ? (passed ? 'verif-pass' : 'verif-fail') : ''}}">
                    ${{hasValue ? (passed ? '\u2713' : '\u2717') : '-'}}
                    ${{hasValue ? (value ? 'Answerable' : 'Not answerable') : 'Not tested'}}
                </div>
                ${{reason ? `<div class="verif-reason">${{escapeHtml(truncate(reason, 80))}}</div>` : ''}}
            </div>
        `;
    }}).join("");

    // Build hops detail list
    const hopsDetailHtml = hops.map((hop, i) => {{
        const typeClass = hop.type === 'L' ? 'local' : 'web';
        const typeLabel = hop.type === 'L' ? 'Local' : 'Web';
        const hasDocContent = hop.type === 'L' && hop.doc_id && DOCS[hop.doc_id];
        const viewBtn = hasDocContent ? `<button class="view-doc-btn" onclick="openDocModal('${{hop.doc_id.replace(/'/g, "\\\\'")}}')">View</button>` : '';
        return `
            <div class="hop-detail ${{typeClass}}">
                <div class="hop-detail-header">
                    <span class="hop-detail-type ${{typeClass}}">${{typeLabel}} Hop</span>
                    <span class="hop-detail-idx">Hop ${{i}}</span>
                </div>
                ${{hop.doc_id ? `<div class="hop-detail-row"><span class="hop-detail-label">Document</span><span class="hop-detail-value hop-detail-doc">${{escapeHtml(hop.doc_id)}}${{viewBtn}}</span></div>` : ''}}
                ${{hop.fact ? `<div class="hop-detail-row"><span class="hop-detail-label">Fact</span><span class="hop-detail-value">${{escapeHtml(hop.fact)}}</span></div>` : ''}}
                ${{hop.text ? `<div class="hop-detail-row"><span class="hop-detail-label">Text</span><span class="hop-detail-value">${{escapeHtml(truncate(hop.text, 300))}}</span></div>` : ''}}
                ${{hop.task_id ? `<div class="hop-detail-row"><span class="hop-detail-label">Task</span><span class="hop-detail-value">${{escapeHtml(hop.task_id)}}</span></div>` : ''}}
                ${{hop.company ? `<div class="hop-detail-row"><span class="hop-detail-label">Company</span><span class="hop-detail-value">${{escapeHtml(hop.company)}}</span></div>` : ''}}
                ${{hop.secret_question ? `<div class="hop-detail-row"><span class="hop-detail-label">Secret Q</span><span class="hop-detail-value">${{escapeHtml(hop.secret_question)}}</span></div>` : ''}}
                ${{hop.secret_answer ? `<div class="hop-detail-row"><span class="hop-detail-label">Secret A</span><span class="hop-detail-value">${{escapeHtml(hop.secret_answer)}}</span></div>` : ''}}
            </div>
        `;
    }}).join("");

    // Build bridges detail list
    const bridgesDetailHtml = bridges.map((bridge, i) => {{
        return `
            <div class="bridge-detail">
                <span class="bridge-indices">${{bridge.from ?? i}} \u2192 ${{bridge.to ?? i + 1}}</span>
                <div class="bridge-link">
                    <div class="bridge-link-label">${{escapeHtml(bridge.type || bridge.bridge_type || '')}}</div>
                    <div class="bridge-link-value">
                        <strong>${{escapeHtml(bridge.link || '?')}}</strong>
                        ${{bridge.relation ? ` \u2192 ${{escapeHtml(bridge.relation)}}` : ''}}
                        ${{bridge.answer ? ` \u2192 <em>${{escapeHtml(bridge.answer)}}</em>` : ''}}
                    </div>
                </div>
            </div>
        `;
    }}).join("");

    container.innerHTML = `
        <div class="chain-detail">
            <div class="detail-header">
                <div class="detail-title">
                    <span>${{escapeHtml(chain.chain_id || 'Chain')}}</span>
                    <span class="chain-pattern">${{escapeHtml(chain.pattern || '?')}}</span>
                    <span class="validity-badge validity-${{status}}">${{status.toUpperCase()}}</span>
                </div>
                <div class="detail-question">
                    <div class="detail-question-label">Question</div>
                    <div class="detail-question-text">${{escapeHtml(chain.question || 'No question')}}</div>
                </div>
                <div class="detail-answer">
                    <span class="detail-answer-label">Answer:</span>
                    <span class="detail-answer-text">${{escapeHtml(chain.answer || '?')}}</span>
                    ${{chain.answer_source ? `<span class="answer-source">(from ${{chain.answer_source === 'L' ? 'Local' : 'Web'}})</span>` : ''}}
                </div>
                ${{chain.task_id || company ? `
                <div class="detail-meta">
                    ${{chain.task_id ? `<span><strong>Task:</strong> ${{escapeHtml(chain.task_id)}}</span>` : ''}}
                    ${{company ? `<span><strong>Company:</strong> ${{escapeHtml(company)}}</span>` : ''}}
                </div>
                ` : ''}}
            </div>

            <div class="section">
                <div class="section-title">Chain Visualization</div>
                <div class="chain-viz">
                    <div class="viz-container">
                        ${{vizHtml}}
                    </div>
                </div>
            </div>

            ${{Object.keys(verification).length > 0 ? `
            <div class="section">
                <div class="section-title">Verification / Ablation</div>
                <div class="verification">
                    ${{verifHtml}}
                </div>
            </div>
            ` : ''}}

            <div class="section">
                <div class="section-title">Hops (${{hops.length}})</div>
                <div class="hops-list">
                    ${{hopsDetailHtml}}
                </div>
            </div>

            ${{bridges.length > 0 ? `
            <div class="section">
                <div class="section-title">Bridges (${{bridges.length}})</div>
                <div class="bridges-list">
                    ${{bridgesDetailHtml}}
                </div>
            </div>
            ` : ''}}

            <div class="section">
                <div class="pipeline-toggle" onclick="togglePipeline()">
                    <span class="pipeline-toggle-text">Pipeline Reference (How This Chain Was Built)</span>
                    <span class="pipeline-toggle-arrow">\u25BC</span>
                </div>
                <div class="pipeline-content" id="pipeline-content">
                    <div class="pipeline-stages">

                        <!-- HOP DATA STRUCTURE -->
                        <div class="pipeline-stage">
                            <div class="pipeline-stage-header">
                                <span class="pipeline-stage-num">\u2139</span>
                                <span class="pipeline-stage-title">Hop Data Structure</span>
                                <span class="pipeline-stage-desc">What each hop contains</span>
                            </div>
                            <div class="pipeline-stage-body">
                                <div class="pipeline-stage-row">
                                    <div class="pipeline-label">Local Hop (L) Fields</div>
                                    <div class="pipeline-prompt">{{
  "type": "L",
  "doc_id": "local/DR0001/DI001_pdf/report.md",   // Path to enterprise document
  "fact": "Lee's Market uses SAP",                 // Extracted fact (pointer to entity)
  "text": "...",                                   // Full document text (optional)
  "company": "Lee's Market",                       // Company name
  "secret_question": "What vendor does Lee's use?", // From secret_inventory
  "secret_answer": "SAP"                           // Ground truth answer
}}</div>
                                </div>
                                <div class="pipeline-stage-row">
                                    <div class="pipeline-label">Web Hop (W) Fields</div>
                                    <div class="pipeline-prompt">{{
  "type": "W",
  "doc_id": "web/browsecomp/12345",               // Web document ID
  "fact": "SAP was founded in Germany in 1972",    // NEW info about the entity
  "text": "..."                                    // Document excerpt
}}</div>
                                </div>
                                <div class="pipeline-stage-row">
                                    <div class="pipeline-label">Key Insight</div>
                                    <div class="pipeline-value">
                                        <strong>L hop</strong> provides a <em>pointer</em> (entity name like "SAP")<br>
                                        <strong>W hop</strong> provides <em>new information</em> about that entity (where SAP was founded)
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="pipeline-arrow">\u2193</div>

                        <!-- STAGE 1: START HOP -->
                        <div class="pipeline-stage">
                            <div class="pipeline-stage-header">
                                <span class="pipeline-stage-num">1</span>
                                <span class="pipeline-stage-title">Start Hop Selection</span>
                                <span class="pipeline-stage-desc">Select initial local secret</span>
                            </div>
                            <div class="pipeline-stage-body">
                                <div class="pipeline-stage-row">
                                    <div class="pipeline-label">Source</div>
                                    <div class="pipeline-value"><code>secret_inventory.jsonl</code> \u2192 221 documents, 943 secrets</div>
                                </div>
                                <div class="pipeline-stage-row">
                                    <div class="pipeline-label">Selection Criteria</div>
                                    <div class="pipeline-value">
                                        1. Filter by target company (Lee's Market, MediConn, Elexion)<br>
                                        2. Pick secret with entity mention (names, vendors, locations)<br>
                                        3. Avoid pure numeric metrics (8%, $1.2M) as start hops
                                    </div>
                                </div>
                                <div class="pipeline-io">
                                    <div class="pipeline-io-item">
                                        <div class="pipeline-io-label">Input</div>
                                        <div class="pipeline-io-value">Company filter + pattern[0] must be "L"</div>
                                    </div>
                                    <div class="pipeline-io-item">
                                        <div class="pipeline-io-label">Output</div>
                                        <div class="pipeline-io-value">Hop 0: {{type: "L", doc_id, fact, company, secret_question, secret_answer}}</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="pipeline-arrow">\u2193</div>

                        <!-- STAGE 2: QUERY GENERATION -->
                        <div class="pipeline-stage">
                            <div class="pipeline-stage-header">
                                <span class="pipeline-stage-num">2</span>
                                <span class="pipeline-stage-title">Query Generation</span>
                                <span class="pipeline-stage-desc">Create search queries from current hop</span>
                            </div>
                            <div class="pipeline-stage-body">
                                <div class="pipeline-stage-row">
                                    <div class="pipeline-label">Query Sources (from current hop)</div>
                                    <div class="pipeline-value">
                                        1. <strong>Entity names</strong>: "SAP", "Kubernetes", "Portland"<br>
                                        2. <strong>Fact text</strong>: "Lee's Market uses SAP for inventory"<br>
                                        3. <strong>Secret answer</strong>: Skip if pure metric (8%, $1.2M)
                                    </div>
                                </div>
                                <div class="pipeline-stage-row">
                                    <div class="pipeline-label">Query Filtering</div>
                                    <div class="pipeline-prompt"># Skip raw metrics - they find same-fact docs
if re.match(r'^[\\d.,%$]+$', answer.strip()):
    pass  # Don't query for "8%"
else:
    queries.append(answer)</div>
                                </div>
                                <div class="pipeline-io">
                                    <div class="pipeline-io-item">
                                        <div class="pipeline-io-label">Input</div>
                                        <div class="pipeline-io-value">Current hop fact + secret_answer</div>
                                    </div>
                                    <div class="pipeline-io-item">
                                        <div class="pipeline-io-label">Output</div>
                                        <div class="pipeline-io-value">queries: ["SAP", "SAP inventory system", ...]</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="pipeline-arrow">\u2193</div>

                        <!-- STAGE 3: RETRIEVAL -->
                        <div class="pipeline-stage">
                            <div class="pipeline-stage-header">
                                <span class="pipeline-stage-num">3</span>
                                <span class="pipeline-stage-title">Retrieval</span>
                                <span class="pipeline-stage-desc">Search for candidate documents</span>
                            </div>
                            <div class="pipeline-stage-body">
                                <div class="pipeline-stage-row">
                                    <div class="pipeline-label">Retrieval Backend</div>
                                    <div class="pipeline-value">
                                        <strong>BM25</strong>: Fast keyword matching (default for L\u2192W)<br>
                                        <strong>Dense</strong>: Semantic embedding search (Qwen3-Embedding)<br>
                                        <strong>Hybrid</strong>: BM25 candidates, reranked by dense
                                    </div>
                                </div>
                                <div class="pipeline-stage-row">
                                    <div class="pipeline-label">Corpus Selection</div>
                                    <div class="pipeline-value">
                                        <strong>L\u2192W</strong>: Search web corpus (BrowseComp-Plus)<br>
                                        <strong>W\u2192L</strong>: Search local docs (same company filter)<br>
                                        <strong>L\u2192L</strong>: Search local docs (same company, different doc)<br>
                                        <strong>W\u2192W</strong>: Search web corpus (different doc)
                                    </div>
                                </div>
                                <div class="pipeline-io">
                                    <div class="pipeline-io-item">
                                        <div class="pipeline-io-label">Input</div>
                                        <div class="pipeline-io-value">queries + next_hop_type (L or W) + corpus</div>
                                    </div>
                                    <div class="pipeline-io-item">
                                        <div class="pipeline-io-label">Output</div>
                                        <div class="pipeline-io-value">candidates: top-K documents (K=20 typical)</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="pipeline-arrow">\u2193</div>

                        <!-- STAGE 4: BRIDGE DISCOVERY -->
                        <div class="pipeline-stage">
                            <div class="pipeline-stage-header">
                                <span class="pipeline-stage-num">4</span>
                                <span class="pipeline-stage-title">Bridge Discovery</span>
                                <span class="pipeline-stage-desc">LLM finds linking entity + NEW fact</span>
                            </div>
                            <div class="pipeline-stage-body">
                                <div class="pipeline-stage-row">
                                    <div class="pipeline-label">Bridge Types</div>
                                    <div class="pipeline-value">
                                        <strong>L\u2192W</strong>: Local mentions entity \u2192 web doc provides NEW info about it<br>
                                        <strong>W\u2192L</strong>: Web mentions entity \u2192 find local secret mentioning same entity<br>
                                        <strong>L\u2192L</strong>: Same company context links two different facts<br>
                                        <strong>W\u2192W</strong>: Web doc chains through shared entity
                                    </div>
                                </div>
                                <div class="pipeline-stage-row">
                                    <div class="pipeline-label">Prompt (L\u2192W Bridge)</div>
                                    <div class="pipeline-prompt">Given a PRIVATE enterprise fact and a PUBLIC web document:

PRIVATE FACT (from local doc):
"{{current_fact}}"

PUBLIC WEB DOCUMENT:
"{{candidate_text}}"

Find if the web doc provides NEW INFORMATION about an entity mentioned in the private fact.

IMPORTANT: The web answer must be NEW information NOT already in the private fact.
If the web doc just restates the same metric/fact, output NO_BRIDGE.

INVALID (same fact):
- Private: "8% reduction" + Web: "8%" \u2192 NO_BRIDGE
- Private: "uses SAP" + Web: "uses SAP" \u2192 NO_BRIDGE

VALID (new info):
- Private: "uses SAP" + Web: "SAP founded in Germany" \u2192 BRIDGE: Germany

Output:
BRIDGE_FOUND: YES/NO
LINK: &lt;shared entity/concept&gt;
ANSWER: &lt;NEW fact from web, 1-5 words, extractive&gt;</div>
                                </div>
                                <div class="pipeline-stage-row">
                                    <div class="pipeline-label">Validation</div>
                                    <div class="pipeline-prompt"># REJECT bridges where answer == current fact
if bridge_answer.lower().strip() == current_fact.lower().strip():
    return None  # Same fact, not a valid bridge
if bridge_answer.lower() in current_fact.lower():
    return None  # Answer is substring of fact</div>
                                </div>
                                <div class="pipeline-io">
                                    <div class="pipeline-io-item">
                                        <div class="pipeline-io-label">Input</div>
                                        <div class="pipeline-io-value">current_hop + candidate_doc (iterate through candidates)</div>
                                    </div>
                                    <div class="pipeline-io-item">
                                        <div class="pipeline-io-label">Output</div>
                                        <div class="pipeline-io-value">Bridge: {{link: "SAP", type: "L\u2192W"}} + next_hop doc</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="pipeline-arrow">\u2193 <em style="font-size:11px;color:var(--text-muted)">Repeat stages 2-4 for each hop in pattern</em></div>

                        <!-- STAGE 5: QUESTION COMPOSITION -->
                        <div class="pipeline-stage">
                            <div class="pipeline-stage-header">
                                <span class="pipeline-stage-num">5</span>
                                <span class="pipeline-stage-title">Question Composition</span>
                                <span class="pipeline-stage-desc">LLM generates multi-hop question</span>
                            </div>
                            <div class="pipeline-stage-body">
                                <div class="pipeline-stage-row">
                                    <div class="pipeline-label">2-Hop Shortcut</div>
                                    <div class="pipeline-value">
                                        For simple LW chains, the bridge question can be used directly:<br>
                                        <code>"Where was [entity from L] founded?"</code> \u2192 answer from W
                                    </div>
                                </div>
                                <div class="pipeline-stage-row">
                                    <div class="pipeline-label">Prompt (Compose Question)</div>
                                    <div class="pipeline-prompt">You have a {{n}}-hop reasoning chain. Generate a single question requiring ALL hops.

CHAIN:
Hop 0 (L): "{{hop0_fact}}" [company: {{company}}]
  \u2192 Bridge: {{bridge0_link}} ({{bridge0_type}})
Hop 1 (W): "{{hop1_fact}}"
  \u2192 Bridge: {{bridge1_link}} ({{bridge1_type}})
...

FINAL ANSWER: {{final_answer}}

Requirements:
- Question references info from FIRST hop
- Answer comes from LAST hop
- Must traverse ALL intermediate hops
- Do NOT mention company names directly

Output:
QUESTION: &lt;your question&gt;</div>
                                </div>
                                <div class="pipeline-io">
                                    <div class="pipeline-io-item">
                                        <div class="pipeline-io-label">Input</div>
                                        <div class="pipeline-io-value">hops[] + bridges[] + answer (from last hop)</div>
                                    </div>
                                    <div class="pipeline-io-item">
                                        <div class="pipeline-io-label">Output</div>
                                        <div class="pipeline-io-value">question: "What is the population of the city where..."</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="pipeline-arrow">\u2193</div>

                        <!-- STAGE 6: VERIFICATION -->
                        <div class="pipeline-stage">
                            <div class="pipeline-stage-header">
                                <span class="pipeline-stage-num">6</span>
                                <span class="pipeline-stage-title">Verification / Ablation</span>
                                <span class="pipeline-stage-desc">Test if question requires all hops</span>
                            </div>
                            <div class="pipeline-stage-body">
                                <div class="pipeline-stage-row">
                                    <div class="pipeline-label">Ablation Tests</div>
                                    <div class="pipeline-value">
                                        <strong>no_docs</strong>: Q + "" \u2192 must be NOT answerable<br>
                                        <strong>first_only</strong>: Q + hop[0].text \u2192 must be NOT answerable<br>
                                        <strong>last_only</strong>: Q + hop[n-1].text \u2192 must be NOT answerable<br>
                                        <strong>all_hops</strong>: Q + all_hops.text \u2192 must be answerable<br><br>
                                        <strong>Valid chain</strong>: all_hops=True AND others=False
                                    </div>
                                </div>
                                <div class="pipeline-stage-row">
                                    <div class="pipeline-label">Prompt (Can Answer)</div>
                                    <div class="pipeline-prompt">Given ONLY the context below, can you answer this question?

CONTEXT:
{{context}}

QUESTION: {{question}}

EXPECTED ANSWER: {{expected_answer}}

If the context contains enough information to arrive at the expected answer:
ANSWERABLE: YES
ANSWER: &lt;your derived answer&gt;

If the context is missing key information needed to answer:
ANSWERABLE: NO
REASON: &lt;what specific information is missing&gt;</div>
                                </div>
                                <div class="pipeline-io">
                                    <div class="pipeline-io-item">
                                        <div class="pipeline-io-label">Input</div>
                                        <div class="pipeline-io-value">question + context variants (4 tests)</div>
                                    </div>
                                    <div class="pipeline-io-item">
                                        <div class="pipeline-io-label">Output</div>
                                        <div class="pipeline-io-value">verification: {{valid, no_docs: {{answerable, reason}}, ...}}</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="pipeline-arrow">\u2193</div>

                        <!-- FINAL OUTPUT -->
                        <div class="pipeline-stage">
                            <div class="pipeline-stage-header">
                                <span class="pipeline-stage-num">\u2713</span>
                                <span class="pipeline-stage-title">Final Chain Output</span>
                                <span class="pipeline-stage-desc">chains.jsonl format</span>
                            </div>
                            <div class="pipeline-stage-body">
                                <div class="pipeline-stage-row">
                                    <div class="pipeline-label">Output Schema</div>
                                    <div class="pipeline-prompt">{{
  "chain_id": "chain_001",
  "pattern": "LW",
  "task_id": "DR0001",
  "answer_source": "W",
  "hops": [
    {{"type": "L", "doc_id": "local/...", "fact": "SAP", "company": "Lee's Market", "task_id": "DR0001", "secret_question": "What vendor...?", "secret_answer": "SAP"}},
    {{"type": "W", "doc_id": "web/...", "fact": "Germany"}}
  ],
  "bridges": [
    {{"link": "SAP", "relation": "founded in", "answer": "Germany", "type": "L\u2192W", "from": 0, "to": 1}}
  ],
  "question": "Where was the company's vendor founded?",
  "answer": "Germany",
  "verification": {{
    "valid": true,
    "no_docs": {{"answerable": false, "reason": "Missing company info"}},
    "first_only": {{"answerable": false, "reason": "Missing location"}},
    "last_only": {{"answerable": false, "reason": "Missing which company"}},
    "all_hops": {{"answerable": true, "reason": "Germany"}}
  }}
}}</div>
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            </div>
        </div>
    `;
}}

// Event listeners
document.getElementById("filter-pattern").addEventListener("change", applyFilters);
document.getElementById("filter-length").addEventListener("change", applyFilters);
document.getElementById("filter-company").addEventListener("change", applyFilters);
document.getElementById("filter-status").addEventListener("change", applyFilters);
document.getElementById("filter-search").addEventListener("input", applyFilters);

// Initialize
initFilters();
renderChainList();

// Select first chain if available
if (CHAINS.length > 0) {{
    selectChain(CHAINS[0]);
}}

// Pipeline toggle
function togglePipeline() {{
  const toggle = document.querySelector('.pipeline-toggle');
  const content = document.getElementById('pipeline-content');
  toggle.classList.toggle('open');
  content.classList.toggle('open');
}}

// Document modal functions
function openDocModal(docId) {{
  const text = DOCS[docId];
  if (!text) return;

  document.getElementById('modal-title').textContent = docId.split('/').pop();
  document.getElementById('modal-content').textContent = text;
  document.getElementById('doc-modal').classList.add('open');
  document.body.style.overflow = 'hidden';
}}

function closeDocModal() {{
  document.getElementById('doc-modal').classList.remove('open');
  document.body.style.overflow = '';
}}

// Close modal on overlay click
document.getElementById('doc-modal').addEventListener('click', function(e) {{
  if (e.target === this) closeDocModal();
}});

// Close modal on Escape key
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') closeDocModal();
}});
</script>
</body>
</html>
'''


OUTPUTS_DIR = Path(__file__).parent / "outputs"


def main():
    parser = argparse.ArgumentParser(description="Generate HTML viewer for multi-hop chains")
    parser.add_argument("--input", "-i", default="making_dataset/outputs/chains.jsonl",
                        help="Input JSONL file (default: making_dataset/outputs/chains.jsonl)")
    parser.add_argument("--output", "-o", default="making_dataset/outputs/chains_viewer.html",
                        help="Output HTML file (default: making_dataset/outputs/chains_viewer.html)")
    parser.add_argument("--docs", default=str(OUTPUTS_DIR / "docs_local.jsonl"),
                        help="Path to docs_local.jsonl for document viewing")
    parser.add_argument("--open", action="store_true", help="Open in browser after generating")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    docs_path = Path(args.docs)

    # Check if input exists, if not generate demo data
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Generating demo data for preview...")
        chains = generate_demo_data()
    else:
        print(f"Loading {input_path}...")
        chains = load_jsonl(input_path)
        print(f"  Loaded {len(chains)} chains")

    print(f"Loading documents from {docs_path}...")
    documents = load_documents(docs_path)
    print(f"  Loaded {len(documents)} document texts")

    print("Computing stats...")
    stats = compute_stats(chains)
    print(f"  {stats['valid']} valid, {stats['invalid']} invalid, {stats['unverified']} unverified")
    print(f"  Patterns: {', '.join(stats['patterns'])}")

    print("Generating HTML...")
    html = generate_html(chains, stats, documents)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Written to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    if args.open:
        try:
            subprocess.run(["xdg-open", str(output_path)], check=False)
        except Exception:
            print(f"Could not open browser. Open manually: {output_path}")


def generate_demo_data() -> list[dict]:
    """Generate demo chains for preview when no input file exists."""
    return [
        {
            "chain_id": "demo_001",
            "pattern": "LW",
            "task_id": "DR0001",
            "answer_source": "W",
            "hops": [
                {"type": "L", "doc_id": "local/DR0001/DI001/inventory.txt", "fact": "SAP", "company": "Lee's Market", "task_id": "DR0001", "secret_question": "What inventory system does Lee's Market use?", "secret_answer": "SAP"},
                {"type": "W", "doc_id": "web/sap_wikipedia", "fact": "Germany"},
            ],
            "bridges": [
                {"link": "SAP", "relation": "founded in", "answer": "Germany", "type": "L→W", "from": 0, "to": 1}
            ],
            "question": "Where was the inventory system vendor used by Lee's Market founded?",
            "answer": "Germany",
            "verification": {
                "valid": True,
                "no_docs": {"answerable": False, "reason": "No information about inventory systems"},
                "first_only": {"answerable": False, "reason": "Missing location information"},
                "last_only": {"answerable": False, "reason": "Missing which company uses SAP"},
                "all_hops": {"answerable": True, "reason": "Germany"}
            }
        },
        {
            "chain_id": "demo_002",
            "pattern": "LWLW",
            "task_id": "DR0001",
            "answer_source": "W",
            "hops": [
                {"type": "L", "doc_id": "local/DR0001/DI001/tech.txt", "fact": "Kubernetes", "company": "Lee's Market", "task_id": "DR0001", "secret_question": "What orchestration platform does Lee's Market use?", "secret_answer": "Kubernetes"},
                {"type": "W", "doc_id": "web/kubernetes_history", "fact": "Google"},
                {"type": "L", "doc_id": "local/DR0001/DI002/locations.txt", "fact": "Portland, Oregon", "company": "Lee's Market", "task_id": "DR0001", "secret_question": "Where is Lee's Market HQ?", "secret_answer": "Portland, Oregon"},
                {"type": "W", "doc_id": "web/portland_stats", "fact": "652,000"},
            ],
            "bridges": [
                {"link": "Kubernetes", "relation": "developed by", "answer": "Google", "type": "L→W", "from": 0, "to": 1},
                {"link": "Lee's Market", "relation": "located in", "answer": "Portland", "type": "W→L", "from": 1, "to": 2},
                {"link": "Portland", "relation": "population", "answer": "652,000", "type": "L→W", "from": 2, "to": 3}
            ],
            "question": "What is the population of the city where the company using Kubernetes is headquartered?",
            "answer": "652,000",
            "verification": {
                "valid": True,
                "no_docs": {"answerable": False, "reason": "No company or location information"},
                "first_only": {"answerable": False, "reason": "Missing HQ location"},
                "last_only": {"answerable": False, "reason": "Missing which company uses Kubernetes"},
                "all_hops": {"answerable": True, "reason": "652,000"}
            }
        },
        {
            "chain_id": "demo_003",
            "pattern": "LLW",
            "task_id": "DR0006",
            "answer_source": "W",
            "hops": [
                {"type": "L", "doc_id": "local/DR0006/DI001/energy.txt", "fact": "450,000 kWh", "company": "MediConn Solutions", "task_id": "DR0006", "secret_question": "What is MediConn's annual energy usage?", "secret_answer": "450,000 kWh"},
                {"type": "L", "doc_id": "local/DR0006/DI002/systems.txt", "fact": "FreshTrack", "company": "MediConn Solutions", "task_id": "DR0006", "secret_question": "What inventory system does MediConn use?", "secret_answer": "FreshTrack"},
                {"type": "W", "doc_id": "web/freshtrack_news", "fact": "Oracle"},
            ],
            "bridges": [
                {"link": "MediConn", "relation": "uses", "answer": "FreshTrack", "type": "L→L", "from": 0, "to": 1},
                {"link": "FreshTrack", "relation": "acquired by", "answer": "Oracle", "type": "L→W", "from": 1, "to": 2}
            ],
            "question": "What company acquired the inventory system used by the healthcare company consuming 450,000 kWh?",
            "answer": "Oracle",
            "verification": {
                "valid": False,
                "no_docs": {"answerable": False, "reason": "No company information"},
                "first_only": {"answerable": False, "reason": "Missing inventory system"},
                "last_only": {"answerable": True, "reason": "Oracle is mentioned directly"},
                "all_hops": {"answerable": True, "reason": "Oracle"}
            }
        },
        {
            "chain_id": "demo_004",
            "pattern": "WWL",
            "task_id": "DR0011",
            "answer_source": "L",
            "hops": [
                {"type": "W", "doc_id": "web/ev_market_2024", "fact": "Tesla"},
                {"type": "W", "doc_id": "web/tesla_suppliers", "fact": "Panasonic"},
                {"type": "L", "doc_id": "local/DR0011/DI001/partners.txt", "fact": "Elexion Automotive", "company": "Elexion Automotive", "task_id": "DR0011", "secret_question": "Who partnered with Panasonic in 2023?", "secret_answer": "Elexion Automotive"},
            ],
            "bridges": [
                {"link": "Tesla", "relation": "sources from", "answer": "Panasonic", "type": "W→W", "from": 0, "to": 1},
                {"link": "Panasonic", "relation": "partnered with", "answer": "Elexion", "type": "W→L", "from": 1, "to": 2}
            ],
            "question": "Which automotive company partnered with the battery supplier of the EV market leader?",
            "answer": "Elexion Automotive",
            "verification": {}
        },
    ]


if __name__ == "__main__":
    main()
