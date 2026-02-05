#!/usr/bin/env python3
"""
Dataset Visualizer Generator

Generates a self-contained HTML file for exploring the making_dataset outputs:
- Chunks (local + sampled web)
- Trees (multi-hop reasoning chains)
- Secrets (privacy questions)
- Q&A tasks
- LLM generation logs
- Architecture & prompts

Usage:
    /home/toolkit/.mamba/envs/vllm013/bin/python visualize.py
    # Output: outputs/visualizer.html
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

OUTPUTS_DIR = Path(__file__).parent / "outputs"
GENERATE_DIR = Path(__file__).parent / "generate"


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load JSONL file, optionally limiting records."""
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def load_llm_logs(logs_dir: Path) -> list[dict[str, Any]]:
    """Load all LLM generation logs from all run directories."""
    all_logs = []
    if not logs_dir.exists():
        return all_logs
    for run_dir in sorted(logs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        log_file = run_dir / "llm_generations.jsonl"
        if log_file.exists():
            run_id = run_dir.name
            for record in load_jsonl(log_file):
                record["run_id"] = run_id
                all_logs.append(record)
    return all_logs


def extract_prompts() -> dict[str, str]:
    """Extract prompt templates from Python source files."""
    prompts = {}

    # privacy_tagger.py prompts
    privacy_tagger = GENERATE_DIR / "privacy_tagger.py"
    if privacy_tagger.exists():
        content = privacy_tagger.read_text()

        # PROMPT_TEMPLATE
        match = re.search(r'PROMPT_TEMPLATE\s*=\s*"""(.+?)"""', content, re.DOTALL)
        if match:
            prompts["privacy_inventory"] = match.group(1).strip()

        # ANSWER_WITH_DOC_TEMPLATE
        match = re.search(r'ANSWER_WITH_DOC_TEMPLATE\s*=\s*"""(.+?)"""', content, re.DOTALL)
        if match:
            prompts["privacy_doc_answer"] = match.group(1).strip()

        # ANSWER_WITHOUT_DOC_TEMPLATE
        match = re.search(r'ANSWER_WITHOUT_DOC_TEMPLATE\s*=\s*"""(.+?)"""', content, re.DOTALL)
        if match:
            prompts["privacy_nodoc_answer"] = match.group(1).strip()

    # mixed_dataset_poc.py prompts
    mixed_poc = GENERATE_DIR / "mixed_dataset_poc.py"
    if mixed_poc.exists():
        content = mixed_poc.read_text()
        match = re.search(r'ANSWER_WITH_EVIDENCE_TEMPLATE\s*=\s*"""(.+?)"""', content, re.DOTALL)
        if match:
            prompts["mixed_answer_with_both"] = match.group(1).strip()

    return prompts


def build_pipeline() -> list[dict[str, Any]]:
    """Build pipeline definition with source paths."""
    return [
        {
            "name": "chunk_local.py",
            "source": "making_dataset/data_prep/chunk_local.py",
            "inputs": ["drbench/data/tasks/DR*/files/*"],
            "outputs": ["chunks_local.jsonl"],
            "description": "Extract local DR0001-DR0015 files into semantic chunks (~450 words each)"
        },
        {
            "name": "chunk_web.py",
            "source": "making_dataset/data_prep/chunk_web.py",
            "inputs": ["BrowseComp-Plus corpus"],
            "outputs": ["chunks_web.jsonl"],
            "description": "Extract web corpus from BrowseComp-Plus (whole-doc alignment)"
        },
        {
            "name": "merge_chunks.py",
            "source": "making_dataset/data_prep/merge_chunks.py",
            "inputs": ["chunks_local.jsonl", "chunks_web.jsonl"],
            "outputs": ["chunks.jsonl"],
            "description": "Merge local and web chunks into unified corpus"
        },
        {
            "name": "build_local_neighbors.py",
            "source": "making_dataset/edges/build_local_neighbors.py",
            "inputs": ["chunks_local.jsonl"],
            "outputs": ["local_neighbors.jsonl"],
            "description": "Build BM25-based neighbor graph (top-20 per chunk)"
        },
        {
            "name": "privacy_tagger.py",
            "source": "making_dataset/generate/privacy_tagger.py",
            "inputs": ["chunks_local.jsonl"],
            "outputs": ["secret_inventory.jsonl"],
            "description": "LLM-based secret extraction with doc-only verification",
            "has_prompts": True,
            "prompt_keys": ["privacy_inventory", "privacy_doc_answer", "privacy_nodoc_answer"]
        },
        {
            "name": "local_multihop_dataset.py",
            "source": "making_dataset/generate/local_multihop_dataset.py",
            "inputs": ["chunks_local.jsonl", "local_neighbors.jsonl", "secret_inventory.jsonl"],
            "outputs": ["local_only.jsonl"],
            "description": "Generate local-only multi-hop tasks via random walk on neighbor graph"
        },
        {
            "name": "web_only_dataset.py",
            "source": "making_dataset/generate/web_only_dataset.py",
            "inputs": ["BrowseComp-Plus tasks"],
            "outputs": ["web_only.jsonl"],
            "description": "Adapt BrowseComp-Plus tasks into dataset schema"
        },
        {
            "name": "mixed_dataset_poc.py",
            "source": "making_dataset/generate/mixed_dataset_poc.py",
            "inputs": ["chunks_local.jsonl", "secret_inventory.jsonl", "web corpus"],
            "outputs": ["mixed.jsonl"],
            "description": "POC: Cross-corpus tasks combining local secrets + web facts",
            "has_prompts": True,
            "prompt_keys": ["mixed_answer_with_both"]
        }
    ]


def generate_html(data: dict[str, Any]) -> str:
    """Generate the complete HTML visualizer."""
    data_json = json.dumps(data, ensure_ascii=False, indent=None)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dataset Visualizer</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
:root {{
  --bg: #f8f9fa;
  --card: #ffffff;
  --primary: #4361ee;
  --primary-light: #eef1ff;
  --success: #2ec4b6;
  --local: #4361ee;
  --web: #7209b7;
  --text: #212529;
  --text-muted: #6c757d;
  --border: #dee2e6;
  --shadow: 0 2px 8px rgba(0,0,0,0.08);
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
}}
header {{
  background: var(--card);
  border-bottom: 1px solid var(--border);
  padding: 0 24px;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: var(--shadow);
}}
.header-content {{
  max-width: 1600px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  gap: 32px;
}}
.logo {{
  font-size: 18px;
  font-weight: 600;
  color: var(--primary);
  padding: 16px 0;
}}
nav {{
  display: flex;
  gap: 4px;
}}
.tab {{
  padding: 16px 20px;
  border: none;
  background: none;
  cursor: pointer;
  font-size: 14px;
  color: var(--text-muted);
  border-bottom: 2px solid transparent;
  transition: all 0.2s;
}}
.tab:hover {{ color: var(--text); }}
.tab.active {{
  color: var(--primary);
  border-bottom-color: var(--primary);
}}
main {{
  max-width: 1600px;
  margin: 0 auto;
  padding: 24px;
}}
.view {{ display: none; }}
.view.active {{ display: block; }}

/* Cards */
.card {{
  background: var(--card);
  border-radius: 12px;
  padding: 24px;
  box-shadow: var(--shadow);
  margin-bottom: 16px;
}}
.card-title {{
  font-size: 14px;
  color: var(--text-muted);
  margin-bottom: 8px;
}}
.card-value {{
  font-size: 32px;
  font-weight: 600;
}}

/* Stats grid */
.stats-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}}

/* Badges */
.badge {{
  display: inline-block;
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}}
.badge-local {{ background: #eef1ff; color: var(--local); }}
.badge-web {{ background: #f3e8ff; color: var(--web); }}
.badge-mixed {{ background: #fef3c7; color: #92400e; }}
.badge-kpi {{ background: #dcfce7; color: #166534; }}
.badge-money {{ background: #fef9c3; color: #854d0e; }}
.badge-names {{ background: #fce7f3; color: #9d174d; }}
.badge-emails {{ background: #e0e7ff; color: #3730a3; }}
.badge-dates {{ background: #f3e8ff; color: #6b21a8; }}
.badge-ids {{ background: #fed7aa; color: #9a3412; }}
.badge-other {{ background: #e5e7eb; color: #374151; }}

/* Two-column layout */
.two-col {{
  display: grid;
  grid-template-columns: 350px 1fr;
  gap: 24px;
  height: calc(100vh - 140px);
}}
.sidebar {{
  background: var(--card);
  border-radius: 12px;
  box-shadow: var(--shadow);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}}
.sidebar-header {{
  padding: 16px;
  border-bottom: 1px solid var(--border);
}}
.sidebar-header input {{
  width: 100%;
  padding: 10px 14px;
  border: 1px solid var(--border);
  border-radius: 8px;
  font-size: 14px;
}}
.sidebar-filters {{
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}}
.sidebar-filters select {{
  padding: 6px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  font-size: 13px;
  background: white;
}}
.sidebar-list {{
  flex: 1;
  overflow-y: auto;
}}
.list-item {{
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  transition: background 0.15s;
}}
.list-item:hover {{ background: var(--primary-light); }}
.list-item.active {{ background: var(--primary-light); border-left: 3px solid var(--primary); }}
.list-item-id {{
  font-family: monospace;
  font-size: 12px;
  color: var(--text-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.list-item-preview {{
  font-size: 13px;
  color: var(--text);
  margin-top: 4px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}}

/* Detail panel */
.detail-panel {{
  background: var(--card);
  border-radius: 12px;
  box-shadow: var(--shadow);
  overflow-y: auto;
  padding: 24px;
}}
.detail-section {{
  margin-bottom: 24px;
}}
.detail-section h3 {{
  font-size: 14px;
  color: var(--text-muted);
  margin-bottom: 12px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}
.detail-text {{
  font-family: "SF Mono", Monaco, monospace;
  font-size: 13px;
  line-height: 1.6;
  background: #f8f9fa;
  padding: 16px;
  border-radius: 8px;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 400px;
  overflow-y: auto;
}}
.meta-table {{
  width: 100%;
  font-size: 13px;
}}
.meta-table td {{
  padding: 8px 0;
  border-bottom: 1px solid var(--border);
}}
.meta-table td:first-child {{
  color: var(--text-muted);
  width: 120px;
}}

/* Neighbors */
.neighbor-list {{
  display: flex;
  flex-direction: column;
  gap: 8px;
}}
.neighbor-item {{
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px;
  background: #f8f9fa;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.15s;
}}
.neighbor-item:hover {{ background: var(--primary-light); }}
.neighbor-score {{
  font-size: 12px;
  color: var(--text-muted);
}}

/* Task cards */
.task-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
  gap: 16px;
}}
.task-card {{
  background: var(--card);
  border-radius: 12px;
  padding: 20px;
  box-shadow: var(--shadow);
}}
.task-header {{
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}}
.task-question {{
  font-size: 16px;
  font-weight: 500;
  margin-bottom: 12px;
}}
.task-answer {{
  background: #dcfce7;
  color: #166534;
  padding: 12px;
  border-radius: 8px;
  font-weight: 500;
  margin-bottom: 16px;
}}
.task-section {{
  margin-top: 12px;
}}
.task-section-title {{
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 6px;
}}
.secret-item {{
  font-size: 13px;
  padding: 8px;
  background: #f8f9fa;
  border-radius: 6px;
  margin-bottom: 6px;
}}

/* Tree visualization */
.tree-container {{
  display: grid;
  grid-template-columns: 1fr 400px;
  gap: 24px;
  height: calc(100vh - 200px);
}}
.tree-svg-container {{
  background: var(--card);
  border-radius: 12px;
  box-shadow: var(--shadow);
  overflow: hidden;
}}
.tree-svg {{
  width: 100%;
  height: 100%;
}}
.tree-node {{
  cursor: pointer;
}}
.tree-node circle {{
  stroke-width: 3;
  transition: all 0.2s;
}}
.tree-node:hover circle {{
  stroke-width: 4;
}}
.tree-node.local circle {{ fill: #eef1ff; stroke: var(--local); }}
.tree-node.web circle {{ fill: #f3e8ff; stroke: var(--web); }}
.tree-node.target circle {{ fill: #dcfce7; stroke: var(--success); stroke-width: 4; }}
.tree-label {{
  font-size: 11px;
  fill: var(--text);
}}
.tree-link {{
  fill: none;
  stroke: var(--border);
  stroke-width: 2;
}}
.tree-controls {{
  padding: 16px;
  border-bottom: 1px solid var(--border);
}}
.tree-controls select {{
  width: 100%;
  padding: 10px;
  border: 1px solid var(--border);
  border-radius: 8px;
  font-size: 14px;
}}

/* Secrets table */
.secrets-table {{
  width: 100%;
  border-collapse: collapse;
}}
.secrets-table th {{
  text-align: left;
  padding: 12px;
  background: #f8f9fa;
  font-size: 12px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}
.secrets-table td {{
  padding: 12px;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
}}
.secrets-table tr {{
  cursor: pointer;
  transition: background 0.15s;
}}
.secrets-table tbody tr:hover {{
  background: var(--primary-light);
}}
.secret-expanded {{
  background: #f8f9fa;
  padding: 16px;
  margin-top: 8px;
  border-radius: 8px;
}}

/* LLM Logs */
.logs-summary {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}}
.logs-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}}
.logs-table th {{
  text-align: left;
  padding: 10px;
  background: #f8f9fa;
  font-size: 12px;
  color: var(--text-muted);
}}
.logs-table td {{
  padding: 10px;
  border-bottom: 1px solid var(--border);
}}
.logs-table tbody tr {{
  cursor: pointer;
}}
.logs-table tbody tr:hover {{
  background: var(--primary-light);
}}

/* Pipeline */
.pipeline-container {{
  display: flex;
  flex-direction: column;
  gap: 16px;
}}
.pipeline-node {{
  background: var(--card);
  border-radius: 12px;
  box-shadow: var(--shadow);
  overflow: hidden;
}}
.pipeline-header {{
  padding: 16px 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  cursor: pointer;
  transition: background 0.15s;
}}
.pipeline-header:hover {{
  background: var(--primary-light);
}}
.pipeline-number {{
  width: 32px;
  height: 32px;
  background: var(--primary);
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 14px;
}}
.pipeline-info {{
  flex: 1;
}}
.pipeline-name {{
  font-weight: 600;
  font-size: 15px;
}}
.pipeline-desc {{
  font-size: 13px;
  color: var(--text-muted);
}}
.pipeline-arrow {{
  color: var(--text-muted);
  transition: transform 0.2s;
}}
.pipeline-node.expanded .pipeline-arrow {{
  transform: rotate(180deg);
}}
.pipeline-body {{
  display: none;
  padding: 20px;
  border-top: 1px solid var(--border);
}}
.pipeline-node.expanded .pipeline-body {{
  display: block;
}}
.pipeline-files {{
  display: flex;
  gap: 24px;
  margin-bottom: 16px;
}}
.pipeline-files div {{
  flex: 1;
}}
.pipeline-files h4 {{
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 8px;
}}
.pipeline-files code {{
  display: block;
  font-size: 12px;
  padding: 4px 0;
}}
.prompt-box {{
  margin-top: 16px;
}}
.prompt-box h4 {{
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 8px;
}}
.prompt-box pre {{
  background: #1e1e1e;
  color: #d4d4d4;
  padding: 16px;
  border-radius: 8px;
  font-size: 12px;
  overflow-x: auto;
  white-space: pre-wrap;
}}

/* Modal */
.modal-overlay {{
  display: none;
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0,0,0,0.5);
  z-index: 200;
  align-items: center;
  justify-content: center;
}}
.modal-overlay.active {{
  display: flex;
}}
.modal {{
  background: var(--card);
  border-radius: 12px;
  width: 90%;
  max-width: 800px;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}}
.modal-header {{
  padding: 20px;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}}
.modal-header h2 {{
  font-size: 18px;
}}
.modal-close {{
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: var(--text-muted);
}}
.modal-body {{
  padding: 20px;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: #f1f1f1; }}
::-webkit-scrollbar-thumb {{ background: #c1c1c1; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: #a1a1a1; }}

/* Empty state */
.empty-state {{
  text-align: center;
  padding: 48px;
  color: var(--text-muted);
}}

/* Filters row */
.filters-row {{
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
  flex-wrap: wrap;
  align-items: center;
}}
.filters-row input {{
  flex: 1;
  min-width: 200px;
  padding: 10px 14px;
  border: 1px solid var(--border);
  border-radius: 8px;
}}
.filters-row select {{
  padding: 10px 14px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: white;
}}
</style>
</head>
<body>
<header>
  <div class="header-content">
    <div class="logo">Dataset Visualizer</div>
    <nav>
      <button class="tab active" data-view="overview">Overview</button>
      <button class="tab" data-view="chunks">Chunks</button>
      <button class="tab" data-view="trees">Trees</button>
      <button class="tab" data-view="secrets">Secrets</button>
      <button class="tab" data-view="tasks">Q&A Tasks</button>
      <button class="tab" data-view="logs">LLM Logs</button>
      <button class="tab" data-view="architecture">Architecture</button>
    </nav>
  </div>
</header>

<main>
  <!-- Overview -->
  <div id="overview" class="view active"></div>

  <!-- Chunks -->
  <div id="chunks" class="view"></div>

  <!-- Trees -->
  <div id="trees" class="view"></div>

  <!-- Secrets -->
  <div id="secrets" class="view"></div>

  <!-- Tasks -->
  <div id="tasks" class="view"></div>

  <!-- Logs -->
  <div id="logs" class="view"></div>

  <!-- Architecture -->
  <div id="architecture" class="view"></div>
</main>

<!-- Modal -->
<div class="modal-overlay" id="modal">
  <div class="modal">
    <div class="modal-header">
      <h2 id="modal-title">Details</h2>
      <button class="modal-close" onclick="closeModal()">&times;</button>
    </div>
    <div class="modal-body" id="modal-body"></div>
  </div>
</div>

<script>
const DATA = {data_json};

// Tab navigation
document.querySelectorAll('.tab').forEach(tab => {{
  tab.addEventListener('click', () => {{
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.view).classList.add('active');
  }});
}});

// Modal
function openModal(title, content) {{
  document.getElementById('modal-title').textContent = title;
  document.getElementById('modal-body').innerHTML = content;
  document.getElementById('modal').classList.add('active');
}}
function closeModal() {{
  document.getElementById('modal').classList.remove('active');
}}
document.getElementById('modal').addEventListener('click', (e) => {{
  if (e.target.id === 'modal') closeModal();
}});

// Helper functions
function escapeHtml(text) {{
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}}

function getBadgeClass(type) {{
  const map = {{
    'local': 'badge-local',
    'web': 'badge-web',
    'mixed': 'badge-mixed',
    'local_only': 'badge-local',
    'web_only': 'badge-web',
    'kpi_numeric': 'badge-kpi',
    'money': 'badge-money',
    'names': 'badge-names',
    'emails': 'badge-emails',
    'dates': 'badge-dates',
    'ids': 'badge-ids',
  }};
  return map[type] || 'badge-other';
}}

function truncate(text, len = 100) {{
  if (!text) return '';
  return text.length > len ? text.substring(0, len) + '...' : text;
}}

// ===== OVERVIEW =====
function renderOverview() {{
  const localCount = DATA.chunks_local.length;
  const webCount = DATA.chunks_web.length;
  const secretCount = DATA.secrets.reduce((sum, s) => sum + (s.secrets?.length || 0), 0);
  const localTasks = DATA.local_only.length;
  const webTasks = DATA.web_only.length;
  const mixedTasks = DATA.mixed.length;
  const logCount = DATA.llm_logs.length;

  // Company breakdown
  const companies = {{}};
  DATA.chunks_local.forEach(c => {{
    const company = c.meta?.company_name || 'Unknown';
    companies[company] = (companies[company] || 0) + 1;
  }});

  document.getElementById('overview').innerHTML = `
    <h2 style="margin-bottom: 20px;">Dataset Overview</h2>
    <div class="stats-grid">
      <div class="card">
        <div class="card-title">Local Chunks</div>
        <div class="card-value">${{localCount.toLocaleString()}}</div>
      </div>
      <div class="card">
        <div class="card-title">Web Chunks (sampled)</div>
        <div class="card-value">${{webCount.toLocaleString()}}</div>
      </div>
      <div class="card">
        <div class="card-title">Extracted Secrets</div>
        <div class="card-value">${{secretCount.toLocaleString()}}</div>
      </div>
      <div class="card">
        <div class="card-title">Total Tasks</div>
        <div class="card-value">${{localTasks + webTasks + mixedTasks}}</div>
      </div>
      <div class="card">
        <div class="card-title">LLM Log Entries</div>
        <div class="card-value">${{logCount.toLocaleString()}}</div>
      </div>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
      <div class="card">
        <h3 style="margin-bottom: 16px;">Tasks by Mode</h3>
        <div style="display: flex; gap: 16px;">
          <div><span class="badge badge-local">local_only</span> ${{localTasks}}</div>
          <div><span class="badge badge-web">web_only</span> ${{webTasks}}</div>
          <div><span class="badge badge-mixed">mixed</span> ${{mixedTasks}}</div>
        </div>
      </div>
      <div class="card">
        <h3 style="margin-bottom: 16px;">Chunks by Company</h3>
        <div style="display: flex; gap: 16px; flex-wrap: wrap;">
          ${{Object.entries(companies).map(([name, count]) =>
            `<div><strong>${{name}}</strong>: ${{count}}</div>`
          ).join('')}}
        </div>
      </div>
    </div>
  `;
}}

// ===== CHUNKS =====
let currentChunk = null;
let filteredChunks = [];

function renderChunks() {{
  const allChunks = [...DATA.chunks_local, ...DATA.chunks_web];
  filteredChunks = allChunks;

  const tasks = [...new Set(DATA.chunks_local.map(c => c.meta?.task_id).filter(Boolean))].sort();
  const companies = [...new Set(DATA.chunks_local.map(c => c.meta?.company_name).filter(Boolean))];

  document.getElementById('chunks').innerHTML = `
    <div class="two-col">
      <div class="sidebar">
        <div class="sidebar-header">
          <input type="text" id="chunk-search" placeholder="Search chunks...">
        </div>
        <div class="sidebar-filters">
          <select id="chunk-source-filter">
            <option value="">All Sources</option>
            <option value="local">Local</option>
            <option value="web">Web</option>
          </select>
          <select id="chunk-task-filter">
            <option value="">All Tasks</option>
            ${{tasks.map(t => `<option value="${{t}}">${{t}}</option>`).join('')}}
          </select>
          <select id="chunk-company-filter">
            <option value="">All Companies</option>
            ${{companies.map(c => `<option value="${{c}}">${{c}}</option>`).join('')}}
          </select>
        </div>
        <div class="sidebar-list" id="chunk-list"></div>
      </div>
      <div class="detail-panel" id="chunk-detail">
        <div class="empty-state">Select a chunk to view details</div>
      </div>
    </div>
  `;

  document.getElementById('chunk-search').addEventListener('input', filterChunks);
  document.getElementById('chunk-source-filter').addEventListener('change', filterChunks);
  document.getElementById('chunk-task-filter').addEventListener('change', filterChunks);
  document.getElementById('chunk-company-filter').addEventListener('change', filterChunks);

  filterChunks();
}}

function filterChunks() {{
  const search = document.getElementById('chunk-search').value.toLowerCase();
  const source = document.getElementById('chunk-source-filter').value;
  const task = document.getElementById('chunk-task-filter').value;
  const company = document.getElementById('chunk-company-filter').value;

  const allChunks = [...DATA.chunks_local, ...DATA.chunks_web];
  filteredChunks = allChunks.filter(c => {{
    if (source && c.source_type !== source) return false;
    if (task && c.meta?.task_id !== task) return false;
    if (company && c.meta?.company_name !== company) return false;
    if (search && !c.text?.toLowerCase().includes(search) && !c.chunk_id?.toLowerCase().includes(search)) return false;
    return true;
  }});

  renderChunkList();
}}

function renderChunkList() {{
  const list = document.getElementById('chunk-list');
  const toShow = filteredChunks.slice(0, 200); // Virtual scroll - show first 200

  list.innerHTML = toShow.map((c, i) => `
    <div class="list-item" data-index="${{i}}" onclick="selectChunk(${{i}})">
      <div class="list-item-id">
        <span class="badge ${{getBadgeClass(c.source_type)}}">${{c.source_type}}</span>
        ${{escapeHtml(c.chunk_id || '')}}
      </div>
      <div class="list-item-preview">${{escapeHtml(truncate(c.text, 80))}}</div>
    </div>
  `).join('') + (filteredChunks.length > 200 ? `<div class="list-item" style="color: var(--text-muted);">... and ${{filteredChunks.length - 200}} more</div>` : '');
}}

function selectChunk(index) {{
  currentChunk = filteredChunks[index];
  document.querySelectorAll('#chunk-list .list-item').forEach((el, i) => {{
    el.classList.toggle('active', i === index);
  }});
  renderChunkDetail();
}}

function renderChunkDetail() {{
  if (!currentChunk) return;
  const c = currentChunk;

  // Find neighbors
  const neighborData = DATA.neighbors.find(n => n.chunk_id === c.chunk_id);
  const neighbors = neighborData?.neighbors?.slice(0, 5) || [];

  // Find secrets
  const secretData = DATA.secrets.find(s => s.chunk_id === c.chunk_id);
  const secrets = secretData?.secrets || [];

  document.getElementById('chunk-detail').innerHTML = `
    <div class="detail-section">
      <h3>Metadata</h3>
      <table class="meta-table">
        <tr><td>Chunk ID</td><td><code>${{escapeHtml(c.chunk_id || '')}}</code></td></tr>
        <tr><td>Doc ID</td><td><code>${{escapeHtml(c.doc_id || '')}}</code></td></tr>
        <tr><td>Source</td><td><span class="badge ${{getBadgeClass(c.source_type)}}">${{c.source_type}}</span></td></tr>
        ${{c.meta?.task_id ? `<tr><td>Task</td><td>${{c.meta.task_id}}</td></tr>` : ''}}
        ${{c.meta?.company_name ? `<tr><td>Company</td><td>${{c.meta.company_name}}</td></tr>` : ''}}
      </table>
    </div>

    <div class="detail-section">
      <h3>Text Content</h3>
      <div class="detail-text">${{escapeHtml(c.text || '')}}</div>
    </div>

    ${{neighbors.length > 0 ? `
    <div class="detail-section">
      <h3>Neighbors (Top 5)</h3>
      <div class="neighbor-list">
        ${{neighbors.map(n => `
          <div class="neighbor-item" onclick="jumpToChunk('${{n.chunk_id}}')">
            <code style="flex:1; font-size: 11px;">${{escapeHtml(n.chunk_id)}}</code>
            <span class="neighbor-score">Score: ${{n.score?.toFixed(2) || 'N/A'}}</span>
          </div>
        `).join('')}}
      </div>
    </div>
    ` : ''}}

    ${{secrets.length > 0 ? `
    <div class="detail-section">
      <h3>Extracted Secrets (${{secrets.length}})</h3>
      ${{secrets.map(s => `
        <div class="secret-item">
          <div><strong>Q:</strong> ${{escapeHtml(s.question)}}</div>
          <div><strong>A:</strong> ${{escapeHtml(s.answer)}}</div>
          <div><span class="badge ${{getBadgeClass(s.secret_type)}}">${{s.secret_type}}</span></div>
        </div>
      `).join('')}}
    </div>
    ` : ''}}
  `;
}}

function jumpToChunk(chunkId) {{
  const allChunks = [...DATA.chunks_local, ...DATA.chunks_web];
  const index = allChunks.findIndex(c => c.chunk_id === chunkId);
  if (index >= 0) {{
    document.getElementById('chunk-search').value = '';
    document.getElementById('chunk-source-filter').value = '';
    document.getElementById('chunk-task-filter').value = '';
    document.getElementById('chunk-company-filter').value = '';
    filterChunks();
    selectChunk(index);
  }}
}}

// ===== TREES =====
function renderTrees() {{
  const allTasks = [...DATA.local_only, ...DATA.web_only, ...DATA.mixed];

  document.getElementById('trees').innerHTML = `
    <div style="display: flex; flex-direction: column; gap: 16px; height: calc(100vh - 140px);">
      <div class="card" style="flex-shrink: 0;">
        <div style="display: flex; gap: 16px; align-items: flex-start;">
          <select id="tree-task-select" style="min-width: 300px; padding: 10px; border: 1px solid var(--border); border-radius: 8px;">
            <option value="">Select a task to visualize...</option>
            ${{allTasks.map((t, i) => `<option value="${{i}}">[#${{i+1}}] ${{t.mode}}</option>`).join('')}}
          </select>
          <div id="tree-question-display" style="flex: 1; font-size: 14px; color: var(--text-muted);">
            Select a task to see details...
          </div>
        </div>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 450px; gap: 16px; flex: 1; min-height: 0;">
        <div style="display: flex; flex-direction: column; gap: 16px; overflow-y: auto;">
          <div class="card" id="tree-task-info" style="display: none;"></div>
          <div class="card tree-svg-container" style="flex: 1; min-height: 200px;">
            <svg id="tree-svg" class="tree-svg"></svg>
          </div>
        </div>
        <div class="detail-panel" id="tree-detail" style="overflow-y: auto;">
          <div class="empty-state">Click a node to view chunk details</div>
        </div>
      </div>
    </div>
  `;

  document.getElementById('tree-task-select').addEventListener('change', (e) => {{
    if (e.target.value) {{
      const task = allTasks[parseInt(e.target.value)];
      renderTreeTaskInfo(task);
      renderTree(task);
    }}
  }});
}}

function renderTreeTaskInfo(task) {{
  // Show full question
  document.getElementById('tree-question-display').innerHTML = `
    <strong>Question:</strong> ${{escapeHtml(task.question)}}
  `;

  // Build task info panel
  let infoHtml = `
    <div style="display: flex; gap: 8px; margin-bottom: 16px;">
      <span class="badge ${{getBadgeClass(task.mode)}}">${{task.mode}}</span>
      <span class="badge badge-other">${{task.answer_type || 'unknown'}}</span>
    </div>
    <div style="background: #dcfce7; padding: 12px; border-radius: 8px; margin-bottom: 16px;">
      <strong>Answer:</strong> ${{escapeHtml(task.answer)}}
    </div>
  `;

  // Show gold evidence based on task type
  if (task.mode === 'mixed' && task.gold) {{
    const g = task.gold;
    infoHtml += `<h4 style="margin-bottom: 8px; color: var(--text-muted);">Gold Evidence</h4>`;

    if (g.local) {{
      infoHtml += `
        <div style="background: #eef1ff; padding: 12px; border-radius: 8px; margin-bottom: 8px;">
          <div style="font-size: 12px; color: var(--text-muted);">Local Value</div>
          <div><strong>${{escapeHtml(g.local.value_str)}}</strong> from <code style="font-size: 11px;">${{escapeHtml(g.local.chunk_id)}}</code></div>
        </div>
      `;
    }}

    if (g.web) {{
      infoHtml += `
        <div style="background: #f3e8ff; padding: 12px; border-radius: 8px; margin-bottom: 8px;">
          <div style="font-size: 12px; color: var(--text-muted);">Web Value</div>
          <div><strong>${{escapeHtml(g.web.value_str)}}</strong></div>
          ${{g.web.title ? `<div style="font-size: 12px; margin-top: 4px;">${{escapeHtml(g.web.title)}}</div>` : ''}}
          ${{g.web.excerpt ? `<div style="font-size: 12px; color: var(--text-muted); margin-top: 8px; font-style: italic;">"...${{escapeHtml(truncate(g.web.excerpt, 200))}}..."</div>` : ''}}
        </div>
      `;
    }}

    if (g.compute) {{
      infoHtml += `
        <div style="background: #fef3c7; padding: 12px; border-radius: 8px; margin-bottom: 8px;">
          <div style="font-size: 12px; color: var(--text-muted);">Computation</div>
          <div><code>${{escapeHtml(g.compute.formula || g.compute.kind)}}</code></div>
          <div style="font-size: 12px;">Inputs: local=${{g.compute.inputs?.local_percent}}, web=${{g.compute.inputs?.web_percent}}</div>
        </div>
      `;
    }}

    if (g.ablation_check) {{
      infoHtml += `
        <div style="background: #f8f9fa; padding: 12px; border-radius: 8px;">
          <div style="font-size: 12px; color: var(--text-muted);">Ablation Check</div>
          <table style="font-size: 12px; width: 100%;">
            <tr><td>With both:</td><td><strong>${{escapeHtml(g.ablation_check.with_both)}}</strong></td></tr>
            <tr><td>Local only:</td><td>${{escapeHtml(g.ablation_check.local_only)}}</td></tr>
            <tr><td>Web only:</td><td>${{escapeHtml(g.ablation_check.web_only)}}</td></tr>
          </table>
        </div>
      `;
    }}
  }} else if (task.mode === 'web_only' && task.gold) {{
    const g = task.gold;
    infoHtml += `<h4 style="margin-bottom: 8px; color: var(--text-muted);">Gold Evidence</h4>`;

    if (g.gold_docids) {{
      infoHtml += `
        <div style="background: #f3e8ff; padding: 12px; border-radius: 8px; margin-bottom: 8px;">
          <div style="font-size: 12px; color: var(--text-muted);">Gold Doc IDs</div>
          <div style="font-size: 12px;">${{g.gold_docids.map(d => `<code>${{d}}</code>`).join(', ')}}</div>
        </div>
      `;
    }}

    if (g.evidence_docids) {{
      infoHtml += `
        <div style="background: #f8f9fa; padding: 12px; border-radius: 8px;">
          <div style="font-size: 12px; color: var(--text-muted);">Evidence Doc IDs (${{g.evidence_docids.length}})</div>
          <div style="font-size: 12px;">${{g.evidence_docids.map(d => `<code>${{d}}</code>`).join(', ')}}</div>
        </div>
      `;
    }}

    if (task.tree?.source?.browsecomp_query_id) {{
      infoHtml += `
        <div style="margin-top: 8px; font-size: 12px; color: var(--text-muted);">
          BrowseComp Query ID: <code>${{task.tree.source.browsecomp_query_id}}</code>
        </div>
      `;
    }}
  }} else if (task.mode === 'local_only' && task.gold) {{
    const g = task.gold;
    infoHtml += `<h4 style="margin-bottom: 8px; color: var(--text-muted);">Gold Evidence</h4>`;
    infoHtml += `
      <div style="background: #eef1ff; padding: 12px; border-radius: 8px;">
        <div style="font-size: 12px; color: var(--text-muted);">Answer in text</div>
        <div><strong>"${{escapeHtml(g.answer_in_text || g.quote || '')}}"</strong></div>
        <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">
          chars ${{g.answer_char_start || g.quote_char_start}}-${{g.answer_char_end || g.quote_char_end}} in <code>${{escapeHtml(g.target_chunk_id)}}</code>
        </div>
      </div>
    `;
  }}

  // Show privacy info
  if (task.privacy?.required_secrets?.length > 0) {{
    infoHtml += `<h4 style="margin: 16px 0 8px; color: var(--text-muted);">Required Secrets</h4>`;
    task.privacy.required_secrets.forEach(s => {{
      infoHtml += `
        <div style="background: #fef2f2; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; font-size: 13px;">
          <span class="badge ${{getBadgeClass(s.secret_type)}}">${{s.secret_type}}</span>
          <strong>${{escapeHtml(s.answer)}}</strong> - ${{escapeHtml(truncate(s.question, 60))}}
        </div>
      `;
    }});
  }}

  document.getElementById('tree-task-info').style.display = 'block';
  document.getElementById('tree-task-info').innerHTML = infoHtml;
}}

function renderTree(task) {{
  const svg = d3.select('#tree-svg');
  svg.selectAll('*').remove();

  if (!task?.tree?.hops) return;

  const width = svg.node().getBoundingClientRect().width;
  const height = svg.node().getBoundingClientRect().height;
  const margin = {{ top: 40, right: 40, bottom: 40, left: 40 }};

  const hops = task.tree.hops;
  const targetHop = task.tree.target_hop;

  // Create linear layout (left to right)
  const nodeWidth = (width - margin.left - margin.right) / hops.length;
  const nodes = hops.map((hop, i) => ({{
    ...hop,
    x: margin.left + nodeWidth * i + nodeWidth / 2,
    y: height / 2,
    isTarget: hop.hop_id === targetHop
  }}));

  const g = svg.append('g');

  // Links
  g.selectAll('.tree-link')
    .data(nodes.slice(1))
    .enter()
    .append('line')
    .attr('class', 'tree-link')
    .attr('x1', (d, i) => nodes[i].x)
    .attr('y1', (d, i) => nodes[i].y)
    .attr('x2', d => d.x)
    .attr('y2', d => d.y);

  // Nodes
  const node = g.selectAll('.tree-node')
    .data(nodes)
    .enter()
    .append('g')
    .attr('class', d => `tree-node ${{d.source_type}} ${{d.isTarget ? 'target' : ''}}`)
    .attr('transform', d => `translate(${{d.x}},${{d.y}})`)
    .on('click', (e, d) => showNodeDetail(d, task));

  node.append('circle')
    .attr('r', 25);

  node.append('text')
    .attr('class', 'tree-label')
    .attr('text-anchor', 'middle')
    .attr('dy', 5)
    .text(d => `Hop ${{d.hop_id}}`);

  // Labels below
  node.append('text')
    .attr('class', 'tree-label')
    .attr('text-anchor', 'middle')
    .attr('dy', 45)
    .text(d => d.source_type);
}}

function showNodeDetail(hop, task) {{
  const allChunks = [...DATA.chunks_local, ...DATA.chunks_web];
  const chunk = allChunks.find(c => c.chunk_id === hop.chunk_id);

  let html = `
    <div class="detail-section">
      <h3>Hop ${{hop.hop_id}} ${{hop.isTarget ? '(Target)' : ''}}</h3>
      <table class="meta-table">
        <tr><td>Chunk ID</td><td><code style="font-size: 11px;">${{escapeHtml(hop.chunk_id)}}</code></td></tr>
        <tr><td>Doc ID</td><td><code style="font-size: 11px;">${{escapeHtml(hop.doc_id || '')}}</code></td></tr>
        <tr><td>Source</td><td><span class="badge ${{getBadgeClass(hop.source_type)}}">${{hop.source_type}}</span></td></tr>
      </table>
    </div>
  `;

  // Show edge query if available (mixed tasks)
  if (hop.edge) {{
    html += `
      <div class="detail-section">
        <h3>Retrieval Edge</h3>
        <div style="background: #fef3c7; padding: 12px; border-radius: 8px;">
          <div style="font-size: 12px; color: var(--text-muted);">Query used to reach this hop:</div>
          <div style="font-style: italic; margin-top: 4px;">"${{escapeHtml(hop.edge.query)}}"</div>
          <div style="font-size: 12px; margin-top: 8px;">
            Corpus: <code>${{hop.edge.corpus}}</code>
            ${{hop.edge.web_backend ? ` | Backend: <code>${{hop.edge.web_backend}}</code>` : ''}}
          </div>
        </div>
      </div>
    `;
  }}

  // Show chunk text
  if (chunk) {{
    html += `
      <div class="detail-section">
        <h3>Chunk Text</h3>
        <div class="detail-text">${{escapeHtml(chunk.text || '')}}</div>
      </div>
    `;
  }} else {{
    html += `
      <div class="detail-section">
        <h3>Chunk Text</h3>
        <div class="empty-state" style="padding: 16px;">Chunk not in sampled data (web chunks are sampled)</div>
      </div>
    `;
  }}

  // Show gold evidence for target hop
  if (hop.isTarget && task.gold) {{
    const g = task.gold;

    if (task.mode === 'mixed') {{
      // For mixed, show both local and web evidence
      if (g.local && hop.source_type === 'local') {{
        html += `
          <div class="detail-section">
            <h3>Local Evidence</h3>
            <div class="detail-text" style="background: #dcfce7;">"${{escapeHtml(g.local.value_str)}}" at chars ${{g.local.value_char_start}}-${{g.local.value_char_end}}</div>
          </div>
        `;
      }}
      if (g.web && hop.source_type === 'web') {{
        html += `
          <div class="detail-section">
            <h3>Web Evidence</h3>
            <div class="detail-text" style="background: #dcfce7;">
              <div>"${{escapeHtml(g.web.value_str)}}"</div>
              ${{g.web.excerpt ? `<div style="margin-top: 8px; font-size: 12px; color: var(--text-muted);">Excerpt: "${{escapeHtml(g.web.excerpt)}}"</div>` : ''}}
            </div>
          </div>
        `;
      }}
    }} else {{
      // For local_only and web_only
      const evidenceText = g.answer_in_text || g.quote || g.quote_raw || '';
      if (evidenceText) {{
        html += `
          <div class="detail-section">
            <h3>Gold Evidence</h3>
            <div class="detail-text" style="background: #dcfce7;">"${{escapeHtml(evidenceText)}}"</div>
          </div>
        `;
      }}
    }}
  }}

  document.getElementById('tree-detail').innerHTML = html;
}}

// ===== SECRETS =====
function renderSecrets() {{
  const allSecrets = [];
  DATA.secrets.forEach(s => {{
    (s.secrets || []).forEach(secret => {{
      allSecrets.push({{ ...secret, chunk_id: s.chunk_id, doc_id: s.doc_id }});
    }});
  }});

  const types = [...new Set(allSecrets.map(s => s.secret_type))].sort();

  document.getElementById('secrets').innerHTML = `
    <div class="filters-row">
      <input type="text" id="secret-search" placeholder="Search secrets...">
      <select id="secret-type-filter">
        <option value="">All Types</option>
        ${{types.map(t => `<option value="${{t}}">${{t}}</option>`).join('')}}
      </select>
    </div>
    <div class="card">
      <table class="secrets-table">
        <thead>
          <tr>
            <th>Chunk ID</th>
            <th>Question</th>
            <th>Answer</th>
            <th>Type</th>
          </tr>
        </thead>
        <tbody id="secrets-tbody"></tbody>
      </table>
    </div>
  `;

  window.allSecrets = allSecrets;
  document.getElementById('secret-search').addEventListener('input', filterSecrets);
  document.getElementById('secret-type-filter').addEventListener('change', filterSecrets);
  filterSecrets();
}}

function filterSecrets() {{
  const search = document.getElementById('secret-search').value.toLowerCase();
  const type = document.getElementById('secret-type-filter').value;

  const filtered = window.allSecrets.filter(s => {{
    if (type && s.secret_type !== type) return false;
    if (search && !s.question?.toLowerCase().includes(search) && !s.answer?.toLowerCase().includes(search)) return false;
    return true;
  }});

  document.getElementById('secrets-tbody').innerHTML = filtered.slice(0, 200).map(s => `
    <tr onclick="showSecretDetail(this, '${{escapeHtml(JSON.stringify(s).replace(/'/g, "\\\\'"))}}')">
      <td><code style="font-size: 11px;">${{escapeHtml(truncate(s.chunk_id, 40))}}</code></td>
      <td>${{escapeHtml(truncate(s.question, 60))}}</td>
      <td><strong>${{escapeHtml(truncate(s.answer, 30))}}</strong></td>
      <td><span class="badge ${{getBadgeClass(s.secret_type)}}">${{s.secret_type}}</span></td>
    </tr>
  `).join('');
}}

function showSecretDetail(row, secretJson) {{
  const s = JSON.parse(secretJson.replace(/\\\\'/g, "'"));
  openModal('Secret Details', `
    <table class="meta-table">
      <tr><td>Chunk ID</td><td><code>${{escapeHtml(s.chunk_id)}}</code></td></tr>
      <tr><td>Question</td><td>${{escapeHtml(s.question)}}</td></tr>
      <tr><td>Answer</td><td><strong>${{escapeHtml(s.answer)}}</strong></td></tr>
      <tr><td>Type</td><td><span class="badge ${{getBadgeClass(s.secret_type)}}">${{s.secret_type}}</span></td></tr>
      <tr><td>Justification</td><td>${{escapeHtml(s.justification || 'N/A')}}</td></tr>
    </table>
    ${{s.doc_only_check ? `
    <h3 style="margin-top: 20px; margin-bottom: 12px;">Doc-Only Verification</h3>
    <table class="meta-table">
      <tr><td>With Doc</td><td>${{escapeHtml(s.doc_only_check.with_doc)}}</td></tr>
      <tr><td>Without Doc</td><td>${{escapeHtml(s.doc_only_check.without_doc)}}</td></tr>
    </table>
    ` : ''}}
  `);
}}

// ===== TASKS =====
function renderTasks() {{
  const allTasks = [...DATA.local_only, ...DATA.web_only, ...DATA.mixed];

  document.getElementById('tasks').innerHTML = `
    <div class="filters-row">
      <input type="text" id="task-search" placeholder="Search questions...">
      <select id="task-mode-filter">
        <option value="">All Modes</option>
        <option value="local_only">local_only</option>
        <option value="web_only">web_only</option>
        <option value="mixed">mixed</option>
      </select>
    </div>
    <div class="task-grid" id="task-grid"></div>
  `;

  window.allTasks = allTasks;
  document.getElementById('task-search').addEventListener('input', filterTasks);
  document.getElementById('task-mode-filter').addEventListener('change', filterTasks);
  filterTasks();
}}

function filterTasks() {{
  const search = document.getElementById('task-search').value.toLowerCase();
  const mode = document.getElementById('task-mode-filter').value;

  const filtered = window.allTasks.filter(t => {{
    if (mode && t.mode !== mode) return false;
    if (search && !t.question?.toLowerCase().includes(search)) return false;
    return true;
  }});

  document.getElementById('task-grid').innerHTML = filtered.map(t => `
    <div class="task-card">
      <div class="task-header">
        <span class="badge ${{getBadgeClass(t.mode)}}">${{t.mode}}</span>
        <span class="badge badge-other">${{t.answer_type || 'unknown'}}</span>
      </div>
      <div class="task-question">${{escapeHtml(t.question)}}</div>
      <div class="task-answer">${{escapeHtml(t.answer)}}</div>

      <div class="task-section">
        <div class="task-section-title">Tree (${{t.tree?.hops?.length || 0}} hops)</div>
        <div style="font-size: 12px; color: var(--text-muted);">
          ${{(t.tree?.hops || []).map(h => `<span class="badge ${{getBadgeClass(h.source_type)}}">${{h.hop_id}}</span>`).join(' → ')}}
        </div>
      </div>

      ${{t.privacy?.required_secrets?.length ? `
      <div class="task-section">
        <div class="task-section-title">Required Secrets</div>
        ${{t.privacy.required_secrets.slice(0, 2).map(s => `
          <div class="secret-item">
            <span class="badge ${{getBadgeClass(s.secret_type)}}">${{s.secret_type}}</span>
            ${{escapeHtml(truncate(s.question, 50))}}
          </div>
        `).join('')}}
      </div>
      ` : ''}}
    </div>
  `).join('');
}}

// ===== LOGS =====
function renderLogs() {{
  const logs = DATA.llm_logs;
  const runs = [...new Set(logs.map(l => l.run_id))].sort().reverse();

  // Aggregate stats
  let totalPrompt = 0, totalCompletion = 0;
  const byStage = {{}};
  logs.forEach(l => {{
    totalPrompt += l.usage?.prompt_tokens || 0;
    totalCompletion += l.usage?.completion_tokens || 0;
    const stage = l.stage || 'unknown';
    byStage[stage] = (byStage[stage] || 0) + (l.usage?.total_tokens || 0);
  }});

  document.getElementById('logs').innerHTML = `
    <div class="logs-summary">
      <div class="card">
        <div class="card-title">Total Entries</div>
        <div class="card-value">${{logs.length.toLocaleString()}}</div>
      </div>
      <div class="card">
        <div class="card-title">Prompt Tokens</div>
        <div class="card-value">${{totalPrompt.toLocaleString()}}</div>
      </div>
      <div class="card">
        <div class="card-title">Completion Tokens</div>
        <div class="card-value">${{totalCompletion.toLocaleString()}}</div>
      </div>
    </div>

    <div class="card" style="margin-bottom: 24px;">
      <h3 style="margin-bottom: 12px;">Tokens by Stage</h3>
      <div style="display: flex; gap: 16px; flex-wrap: wrap;">
        ${{Object.entries(byStage).map(([stage, tokens]) =>
          `<div><strong>${{stage}}</strong>: ${{tokens.toLocaleString()}}</div>`
        ).join('')}}
      </div>
    </div>

    <div class="filters-row">
      <select id="log-run-filter">
        <option value="">All Runs</option>
        ${{runs.map(r => `<option value="${{r}}">${{r}}</option>`).join('')}}
      </select>
      <select id="log-stage-filter">
        <option value="">All Stages</option>
        ${{Object.keys(byStage).map(s => `<option value="${{s}}">${{s}}</option>`).join('')}}
      </select>
    </div>

    <div class="card">
      <table class="logs-table">
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Run</th>
            <th>Stage</th>
            <th>Chunk ID</th>
            <th>Prompt</th>
            <th>Completion</th>
          </tr>
        </thead>
        <tbody id="logs-tbody"></tbody>
      </table>
    </div>
  `;

  window.allLogs = logs;
  document.getElementById('log-run-filter').addEventListener('change', filterLogs);
  document.getElementById('log-stage-filter').addEventListener('change', filterLogs);
  filterLogs();
}}

function filterLogs() {{
  const run = document.getElementById('log-run-filter').value;
  const stage = document.getElementById('log-stage-filter').value;

  const filtered = window.allLogs.filter(l => {{
    if (run && l.run_id !== run) return false;
    if (stage && l.stage !== stage) return false;
    return true;
  }});

  document.getElementById('logs-tbody').innerHTML = filtered.slice(0, 200).map((l, i) => `
    <tr onclick="showLogDetail(${{i}})">
      <td>${{l.timestamp_iso || new Date(l.timestamp * 1000).toISOString()}}</td>
      <td><code style="font-size: 11px;">${{escapeHtml(truncate(l.run_id || '', 20))}}</code></td>
      <td>${{escapeHtml(l.stage || '')}}</td>
      <td><code style="font-size: 11px;">${{escapeHtml(truncate(l.chunk_id || l.task_id || '', 30))}}</code></td>
      <td>${{l.usage?.prompt_tokens?.toLocaleString() || 0}}</td>
      <td>${{l.usage?.completion_tokens?.toLocaleString() || 0}}</td>
    </tr>
  `).join('');

  window.filteredLogs = filtered;
}}

function showLogDetail(index) {{
  const l = window.filteredLogs[index];
  openModal('LLM Log Entry', `
    <table class="meta-table">
      <tr><td>Timestamp</td><td>${{l.timestamp_iso || new Date(l.timestamp * 1000).toISOString()}}</td></tr>
      <tr><td>Run ID</td><td><code>${{escapeHtml(l.run_id || '')}}</code></td></tr>
      <tr><td>Stage</td><td>${{escapeHtml(l.stage || '')}}</td></tr>
      <tr><td>Model</td><td>${{escapeHtml(l.resolved_model || l.requested_model || '')}}</td></tr>
      <tr><td>Chunk/Task ID</td><td><code>${{escapeHtml(l.chunk_id || l.task_id || '')}}</code></td></tr>
      <tr><td>Prompt Tokens</td><td>${{l.usage?.prompt_tokens?.toLocaleString() || 0}}</td></tr>
      <tr><td>Completion Tokens</td><td>${{l.usage?.completion_tokens?.toLocaleString() || 0}}</td></tr>
    </table>
  `);
}}

// ===== ARCHITECTURE =====
function renderArchitecture() {{
  document.getElementById('architecture').innerHTML = `
    <h2 style="margin-bottom: 20px;">Pipeline Architecture</h2>
    <div class="pipeline-container">
      ${{DATA.pipeline.map((step, i) => `
        <div class="pipeline-node" id="pipeline-${{i}}">
          <div class="pipeline-header" onclick="togglePipeline(${{i}})">
            <div class="pipeline-number">${{i + 1}}</div>
            <div class="pipeline-info">
              <div class="pipeline-name">${{step.name}}</div>
              <div class="pipeline-desc">${{step.description}}</div>
            </div>
            <div class="pipeline-arrow">&#9660;</div>
          </div>
          <div class="pipeline-body">
            <div class="pipeline-files">
              <div>
                <h4>Source</h4>
                <code>${{step.source}}</code>
              </div>
              <div>
                <h4>Inputs</h4>
                ${{step.inputs.map(i => `<code>${{i}}</code>`).join('')}}
              </div>
              <div>
                <h4>Outputs</h4>
                ${{step.outputs.map(o => `<code>${{o}}</code>`).join('')}}
              </div>
            </div>
            ${{step.has_prompts ? `
              ${{step.prompt_keys.map(key => DATA.prompts[key] ? `
                <div class="prompt-box">
                  <h4>${{key}}</h4>
                  <pre>${{escapeHtml(DATA.prompts[key])}}</pre>
                </div>
              ` : '').join('')}}
            ` : ''}}
          </div>
        </div>
      `).join('')}}
    </div>
  `;
}}

function togglePipeline(index) {{
  document.getElementById(`pipeline-${{index}}`).classList.toggle('expanded');
}}

// Initialize all views
renderOverview();
renderChunks();
renderTrees();
renderSecrets();
renderTasks();
renderLogs();
renderArchitecture();
</script>
</body>
</html>
'''


def main():
    parser = argparse.ArgumentParser(description="Generate dataset visualizer HTML")
    parser.add_argument(
        "--output",
        default=str(OUTPUTS_DIR / "visualizer.html"),
        help="Output HTML path",
    )
    parser.add_argument(
        "--web-sample",
        type=int,
        default=500,
        help="Number of web chunks to sample (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    args = parser.parse_args()

    print("Loading data...")

    # Load chunks
    chunks_local = load_jsonl(OUTPUTS_DIR / "chunks_local.jsonl")
    print(f"  Local chunks: {len(chunks_local)}")

    chunks_web_all = load_jsonl(OUTPUTS_DIR / "chunks_web.jsonl")
    random.seed(args.seed)
    chunks_web = random.sample(chunks_web_all, min(args.web_sample, len(chunks_web_all))) if chunks_web_all else []
    print(f"  Web chunks: {len(chunks_web)} (sampled from {len(chunks_web_all)})")

    # Load neighbors
    neighbors = load_jsonl(OUTPUTS_DIR / "local_neighbors.jsonl")
    print(f"  Neighbors: {len(neighbors)}")

    # Load secrets
    secrets = load_jsonl(OUTPUTS_DIR / "secret_inventory.jsonl")
    print(f"  Secret docs: {len(secrets)}")

    # Load tasks
    local_only = load_jsonl(OUTPUTS_DIR / "local_only.jsonl")
    web_only = load_jsonl(OUTPUTS_DIR / "web_only.jsonl")
    mixed = load_jsonl(OUTPUTS_DIR / "mixed.jsonl")
    print(f"  Tasks: {len(local_only)} local, {len(web_only)} web, {len(mixed)} mixed")

    # Load LLM logs
    llm_logs = load_llm_logs(OUTPUTS_DIR / "logs")
    print(f"  LLM log entries: {len(llm_logs)}")

    # Extract prompts
    prompts = extract_prompts()
    print(f"  Prompts: {list(prompts.keys())}")

    # Build pipeline
    pipeline = build_pipeline()

    # Prepare data
    data = {
        "chunks_local": chunks_local,
        "chunks_web": chunks_web,
        "neighbors": neighbors,
        "secrets": secrets,
        "local_only": local_only,
        "web_only": web_only,
        "mixed": mixed,
        "llm_logs": llm_logs,
        "prompts": prompts,
        "pipeline": pipeline,
    }

    print("Generating HTML...")
    html = generate_html(data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    print(f"Written to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
