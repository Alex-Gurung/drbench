#!/usr/bin/env python3
"""
Secrets Inventory Viewer

Generates a self-contained HTML viewer for the document-based secret inventory:
- Task/Company hierarchy navigation (sidebar tree)
- Document grouping with metadata
- Secret type filtering
- Doc-only verification status display

Usage:
    python view_secrets.py
    python view_secrets.py --input outputs/secret_inventory.jsonl
    python view_secrets.py --output /tmp/secrets.html
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path
from typing import Any

OUTPUTS_DIR = Path(__file__).parent / "outputs"

COMPANIES = {
    "DR0001": "Lee's Market",
    "DR0002": "Lee's Market",
    "DR0003": "Lee's Market",
    "DR0004": "Lee's Market",
    "DR0005": "Lee's Market",
    "DR0006": "MediConn Solutions",
    "DR0007": "MediConn Solutions",
    "DR0008": "MediConn Solutions",
    "DR0009": "MediConn Solutions",
    "DR0010": "MediConn Solutions",
    "DR0011": "Elexion Automotive",
    "DR0012": "Elexion Automotive",
    "DR0013": "Elexion Automotive",
    "DR0014": "Elexion Automotive",
    "DR0015": "Elexion Automotive",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_doc_id(doc_id: str) -> dict[str, str]:
    """Parse doc_id like 'local/DR0001/DI001_pdf/filename.md' into components."""
    parts = doc_id.split("/")
    if len(parts) >= 4 and parts[0] == "local":
        return {
            "task_id": parts[1],
            "subdir": parts[2],
            "filename": "/".join(parts[3:]),
        }
    return {"task_id": "unknown", "subdir": "unknown", "filename": doc_id}


def build_hierarchy(records: list[dict]) -> dict[str, Any]:
    """Build task -> company -> subdir -> documents hierarchy."""
    hierarchy: dict[str, Any] = {}

    for record in records:
        doc_id = record.get("doc_id") or record.get("chunk_id") or ""
        secrets = record.get("secrets") or []

        parsed = parse_doc_id(doc_id)
        task_id = parsed["task_id"]
        subdir = parsed["subdir"]
        filename = parsed["filename"]
        company = COMPANIES.get(task_id, "Unknown")

        if task_id not in hierarchy:
            hierarchy[task_id] = {
                "company": company,
                "subdirs": {},
                "doc_count": 0,
                "secret_count": 0,
            }

        if subdir not in hierarchy[task_id]["subdirs"]:
            hierarchy[task_id]["subdirs"][subdir] = {
                "documents": [],
                "doc_count": 0,
                "secret_count": 0,
            }

        hierarchy[task_id]["subdirs"][subdir]["documents"].append({
            "doc_id": doc_id,
            "filename": filename,
            "secrets": secrets,
        })
        hierarchy[task_id]["subdirs"][subdir]["doc_count"] += 1
        hierarchy[task_id]["subdirs"][subdir]["secret_count"] += len(secrets)
        hierarchy[task_id]["doc_count"] += 1
        hierarchy[task_id]["secret_count"] += len(secrets)

    return hierarchy


def compute_stats(records: list[dict]) -> dict[str, Any]:
    """Compute aggregate statistics."""
    total_docs = len(records)
    total_secrets = 0
    by_type: dict[str, int] = {}
    verified_count = 0

    for record in records:
        secrets = record.get("secrets") or []
        total_secrets += len(secrets)
        for secret in secrets:
            stype = secret.get("secret_type") or "unknown"
            by_type[stype] = by_type.get(stype, 0) + 1
            if secret.get("doc_only_check", {}).get("without_doc") == "NOT_ANSWERABLE":
                verified_count += 1

    return {
        "total_docs": total_docs,
        "total_secrets": total_secrets,
        "by_type": by_type,
        "verified_count": verified_count,
    }


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


def generate_html(hierarchy: dict, stats: dict, records: list[dict], documents: dict[str, str]) -> str:
    """Generate the complete HTML viewer."""
    data = {
        "hierarchy": hierarchy,
        "stats": stats,
        "records": records,
        "documents": documents,
    }
    data_json = json.dumps(data, ensure_ascii=False, indent=None)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Secrets Inventory Viewer</title>
<style>
:root {{
  --bg: #f8f9fa;
  --card: #ffffff;
  --primary: #4361ee;
  --primary-light: #eef1ff;
  --success: #10b981;
  --warning: #f59e0b;
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
  padding: 16px 24px;
  box-shadow: var(--shadow);
}}
.header-content {{
  max-width: 1800px;
  margin: 0 auto;
}}
.logo {{
  font-size: 20px;
  font-weight: 600;
  color: var(--primary);
  margin-bottom: 12px;
}}
.stats-bar {{
  display: flex;
  gap: 24px;
  font-size: 14px;
  color: var(--text-muted);
  flex-wrap: wrap;
}}
.stat {{ display: flex; gap: 6px; align-items: center; }}
.stat-value {{ font-weight: 600; color: var(--text); }}

.filters-bar {{
  display: flex;
  gap: 12px;
  padding: 12px 24px;
  background: var(--card);
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
  align-items: center;
}}
.filters-bar select, .filters-bar input {{
  padding: 8px 12px;
  border: 1px solid var(--border);
  border-radius: 6px;
  font-size: 14px;
  background: white;
}}
.filters-bar input {{ min-width: 200px; }}

main {{
  display: grid;
  grid-template-columns: 320px 1fr;
  max-width: 1800px;
  margin: 0 auto;
  height: calc(100vh - 140px);
}}

/* Sidebar tree */
.sidebar {{
  background: var(--card);
  border-right: 1px solid var(--border);
  overflow-y: auto;
  padding: 12px 0;
}}
.tree-item {{
  padding: 8px 16px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  transition: background 0.15s;
}}
.tree-item:hover {{ background: var(--primary-light); }}
.tree-item.active {{ background: var(--primary-light); border-left: 3px solid var(--primary); }}
.tree-item.task {{ font-weight: 600; }}
.tree-item.subdir {{ padding-left: 32px; color: var(--text-muted); }}
.tree-item.doc {{ padding-left: 48px; font-size: 13px; }}
.tree-arrow {{
  width: 16px;
  text-align: center;
  color: var(--text-muted);
  font-size: 10px;
}}
.tree-count {{
  margin-left: auto;
  background: #e9ecef;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 11px;
  color: var(--text-muted);
}}
.tree-children {{ display: none; }}
.tree-children.expanded {{ display: block; }}
.company-label {{
  font-size: 11px;
  color: var(--text-muted);
  font-weight: normal;
  margin-left: 4px;
}}

/* Detail panel */
.detail-panel {{
  padding: 24px;
  overflow-y: auto;
}}
.doc-header {{
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}}
.doc-title {{
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 8px;
  word-break: break-all;
}}
.doc-meta {{
  display: flex;
  gap: 16px;
  font-size: 13px;
  color: var(--text-muted);
}}
.doc-meta span {{ display: flex; gap: 4px; }}

.secrets-list {{
  display: flex;
  flex-direction: column;
  gap: 16px;
}}
.secret-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  box-shadow: var(--shadow);
}}
.secret-header {{
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 12px;
}}
.badge {{
  display: inline-block;
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 500;
}}
.badge-kpi {{ background: #dcfce7; color: #166534; }}
.badge-money {{ background: #fef9c3; color: #854d0e; }}
.badge-names {{ background: #fce7f3; color: #9d174d; }}
.badge-emails {{ background: #e0e7ff; color: #3730a3; }}
.badge-dates {{ background: #f3e8ff; color: #6b21a8; }}
.badge-ids {{ background: #fed7aa; color: #9a3412; }}
.badge-other {{ background: #e5e7eb; color: #374151; }}
.verified {{
  color: var(--success);
  font-size: 12px;
  display: flex;
  align-items: center;
  gap: 4px;
}}
.unverified {{
  color: var(--warning);
  font-size: 12px;
  display: flex;
  align-items: center;
  gap: 4px;
}}

.secret-question {{
  font-size: 14px;
  margin-bottom: 8px;
}}
.secret-question strong {{ color: var(--text-muted); }}
.secret-answer {{
  background: #f0fdf4;
  border-left: 3px solid var(--success);
  padding: 10px 14px;
  font-weight: 500;
  margin-bottom: 8px;
}}
.secret-justification {{
  font-size: 12px;
  color: var(--text-muted);
  font-style: italic;
}}
.verification-details {{
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed var(--border);
  font-size: 12px;
  color: var(--text-muted);
}}
.verification-details code {{
  background: #f1f5f9;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 11px;
}}

.empty-state {{
  text-align: center;
  padding: 48px;
  color: var(--text-muted);
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: #f1f1f1; }}
::-webkit-scrollbar-thumb {{ background: #c1c1c1; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: #a1a1a1; }}

.hidden {{ display: none !important; }}

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
.doc-title {{
  display: flex;
  align-items: center;
  gap: 12px;
}}
.doc-title-text {{
  font-size: 18px;
  font-weight: 600;
  word-break: break-all;
}}
.view-doc-btn {{
  background: var(--primary);
  color: white;
  border: none;
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 12px;
  cursor: pointer;
  white-space: nowrap;
}}
.view-doc-btn:hover {{ opacity: 0.9; }}
</style>
</head>
<body>
<header>
  <div class="header-content">
    <div class="logo">Secrets Inventory Viewer</div>
    <div class="stats-bar" id="stats-bar"></div>
  </div>
</header>

<div class="filters-bar">
  <select id="filter-task">
    <option value="">All Tasks</option>
  </select>
  <select id="filter-company">
    <option value="">All Companies</option>
  </select>
  <select id="filter-type">
    <option value="">All Types</option>
  </select>
  <input type="text" id="filter-search" placeholder="Search questions/answers...">
</div>

<main>
  <div class="sidebar" id="sidebar"></div>
  <div class="detail-panel" id="detail">
    <div class="empty-state">Select a document from the tree to view its secrets</div>
  </div>
</main>

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
const DATA = {data_json};

let selectedDocId = null;
let expandedTasks = new Set();
let expandedSubdirs = new Set();

// Initialize
function init() {{
  renderStats();
  populateFilters();
  renderTree();

  document.getElementById('filter-task').addEventListener('change', renderTree);
  document.getElementById('filter-company').addEventListener('change', renderTree);
  document.getElementById('filter-type').addEventListener('change', renderTree);
  document.getElementById('filter-search').addEventListener('input', renderTree);
}}

function renderStats() {{
  const s = DATA.stats;
  const typeStr = Object.entries(s.by_type)
    .sort((a, b) => b[1] - a[1])
    .map(([t, c]) => `${{t}}(${{c}})`)
    .join(' ');

  document.getElementById('stats-bar').innerHTML = `
    <div class="stat"><span class="stat-value">${{s.total_docs}}</span> documents</div>
    <div class="stat"><span class="stat-value">${{s.total_secrets}}</span> secrets</div>
    <div class="stat"><span class="stat-value">${{s.verified_count}}</span> verified</div>
    <div class="stat">By type: ${{typeStr}}</div>
  `;
}}

function populateFilters() {{
  const taskSelect = document.getElementById('filter-task');
  const companySelect = document.getElementById('filter-company');
  const typeSelect = document.getElementById('filter-type');

  // Tasks
  Object.keys(DATA.hierarchy).sort().forEach(taskId => {{
    const opt = document.createElement('option');
    opt.value = taskId;
    opt.textContent = taskId;
    taskSelect.appendChild(opt);
  }});

  // Companies
  const companies = [...new Set(Object.values(DATA.hierarchy).map(t => t.company))];
  companies.sort().forEach(company => {{
    const opt = document.createElement('option');
    opt.value = company;
    opt.textContent = company;
    companySelect.appendChild(opt);
  }});

  // Types
  Object.keys(DATA.stats.by_type).sort().forEach(type => {{
    const opt = document.createElement('option');
    opt.value = type;
    opt.textContent = type;
    typeSelect.appendChild(opt);
  }});
}}

function matchesFilters(doc, taskId, company) {{
  const filterTask = document.getElementById('filter-task').value;
  const filterCompany = document.getElementById('filter-company').value;
  const filterType = document.getElementById('filter-type').value;
  const filterSearch = document.getElementById('filter-search').value.toLowerCase();

  if (filterTask && taskId !== filterTask) return false;
  if (filterCompany && company !== filterCompany) return false;

  if (filterType) {{
    const hasType = doc.secrets.some(s => s.secret_type === filterType);
    if (!hasType) return false;
  }}

  if (filterSearch) {{
    const searchStr = JSON.stringify(doc).toLowerCase();
    if (!searchStr.includes(filterSearch)) return false;
  }}

  return true;
}}

function renderTree() {{
  const sidebar = document.getElementById('sidebar');
  let html = '';

  const sortedTasks = Object.keys(DATA.hierarchy).sort();

  for (const taskId of sortedTasks) {{
    const task = DATA.hierarchy[taskId];
    const isTaskExpanded = expandedTasks.has(taskId);

    // Count visible docs/secrets for this task
    let visibleDocs = 0;
    let visibleSecrets = 0;

    for (const [subdirName, subdir] of Object.entries(task.subdirs)) {{
      for (const doc of subdir.documents) {{
        if (matchesFilters(doc, taskId, task.company)) {{
          visibleDocs++;
          visibleSecrets += doc.secrets.length;
        }}
      }}
    }}

    if (visibleDocs === 0) continue;  // Skip empty tasks after filtering

    html += `
      <div class="tree-item task" onclick="toggleTask('${{taskId}}')" data-task="${{taskId}}">
        <span class="tree-arrow">${{isTaskExpanded ? '&#9660;' : '&#9654;'}}</span>
        ${{taskId}}
        <span class="company-label">(${{task.company}})</span>
        <span class="tree-count">${{visibleDocs}} docs / ${{visibleSecrets}} secrets</span>
      </div>
      <div class="tree-children ${{isTaskExpanded ? 'expanded' : ''}}" id="task-${{taskId}}">
    `;

    const sortedSubdirs = Object.keys(task.subdirs).sort();
    for (const subdirName of sortedSubdirs) {{
      const subdir = task.subdirs[subdirName];
      const subdirKey = `${{taskId}}/${{subdirName}}`;
      const isSubdirExpanded = expandedSubdirs.has(subdirKey);

      // Count visible docs for this subdir
      let subdirVisibleDocs = 0;
      let subdirVisibleSecrets = 0;
      for (const doc of subdir.documents) {{
        if (matchesFilters(doc, taskId, task.company)) {{
          subdirVisibleDocs++;
          subdirVisibleSecrets += doc.secrets.length;
        }}
      }}

      if (subdirVisibleDocs === 0) continue;

      html += `
        <div class="tree-item subdir" onclick="toggleSubdir('${{subdirKey}}')" data-subdir="${{subdirKey}}">
          <span class="tree-arrow">${{isSubdirExpanded ? '&#9660;' : '&#9654;'}}</span>
          ${{subdirName}}
          <span class="tree-count">${{subdirVisibleDocs}} / ${{subdirVisibleSecrets}}</span>
        </div>
        <div class="tree-children ${{isSubdirExpanded ? 'expanded' : ''}}" id="subdir-${{subdirKey.replace('/', '-')}}">
      `;

      for (const doc of subdir.documents) {{
        if (!matchesFilters(doc, taskId, task.company)) continue;

        const isActive = selectedDocId === doc.doc_id;
        html += `
          <div class="tree-item doc ${{isActive ? 'active' : ''}}" onclick="selectDoc('${{escapeAttr(doc.doc_id)}}')" data-doc="${{escapeAttr(doc.doc_id)}}">
            ${{escapeHtml(doc.filename)}}
            <span class="tree-count">${{doc.secrets.length}}</span>
          </div>
        `;
      }}

      html += '</div>';
    }}

    html += '</div>';
  }}

  sidebar.innerHTML = html || '<div class="empty-state">No documents match filters</div>';
}}

function toggleTask(taskId) {{
  if (expandedTasks.has(taskId)) {{
    expandedTasks.delete(taskId);
  }} else {{
    expandedTasks.add(taskId);
  }}
  renderTree();
}}

function toggleSubdir(subdirKey) {{
  if (expandedSubdirs.has(subdirKey)) {{
    expandedSubdirs.delete(subdirKey);
  }} else {{
    expandedSubdirs.add(subdirKey);
  }}
  renderTree();
}}

function selectDoc(docId) {{
  selectedDocId = docId;
  renderTree();
  renderDetail(docId);
}}

function renderDetail(docId) {{
  const detail = document.getElementById('detail');

  // Find the document
  let doc = null;
  let taskId = null;
  let company = null;
  let subdirName = null;

  for (const [tid, task] of Object.entries(DATA.hierarchy)) {{
    for (const [sname, subdir] of Object.entries(task.subdirs)) {{
      for (const d of subdir.documents) {{
        if (d.doc_id === docId) {{
          doc = d;
          taskId = tid;
          company = task.company;
          subdirName = sname;
          break;
        }}
      }}
      if (doc) break;
    }}
    if (doc) break;
  }}

  if (!doc) {{
    detail.innerHTML = '<div class="empty-state">Document not found</div>';
    return;
  }}

  const hasDocContent = DATA.documents && DATA.documents[doc.doc_id];

  let html = `
    <div class="doc-header">
      <div class="doc-title">
        <span class="doc-title-text">${{escapeHtml(doc.filename)}}</span>
        ${{hasDocContent ? `<button class="view-doc-btn" onclick="openDocModal('${{escapeAttr(doc.doc_id)}}')">View Document</button>` : ''}}
      </div>
      <div class="doc-meta">
        <span><strong>Task:</strong> ${{taskId}}</span>
        <span><strong>Company:</strong> ${{company}}</span>
        <span><strong>Subdir:</strong> ${{subdirName}}</span>
        <span><strong>Secrets:</strong> ${{doc.secrets.length}}</span>
      </div>
    </div>
    <div class="secrets-list">
  `;

  if (doc.secrets.length === 0) {{
    html += '<div class="empty-state">No secrets extracted from this document</div>';
  }} else {{
    for (const secret of doc.secrets) {{
      const stype = secret.secret_type || 'other';
      const badgeClass = getBadgeClass(stype);
      const docCheck = secret.doc_only_check || {{}};
      const isVerified = docCheck.without_doc === 'NOT_ANSWERABLE';

      html += `
        <div class="secret-card">
          <div class="secret-header">
            <span class="badge ${{badgeClass}}">${{escapeHtml(stype)}}</span>
            ${{isVerified
              ? '<span class="verified">&#10003; Verified doc-only</span>'
              : '<span class="unverified">&#9888; Not verified</span>'
            }}
          </div>
          <div class="secret-question">
            <strong>Q:</strong> ${{escapeHtml(secret.question || '')}}
          </div>
          <div class="secret-answer">
            ${{escapeHtml(secret.answer || '')}}
          </div>
          ${{secret.justification ? `<div class="secret-justification">${{escapeHtml(secret.justification)}}</div>` : ''}}
          ${{docCheck.with_doc ? `
            <div class="verification-details">
              <div><strong>With doc:</strong> <code>${{escapeHtml(docCheck.with_doc)}}</code></div>
              <div><strong>Without doc:</strong> <code>${{escapeHtml(docCheck.without_doc || 'N/A')}}</code></div>
            </div>
          ` : ''}}
        </div>
      `;
    }}
  }}

  html += '</div>';
  detail.innerHTML = html;
}}

function getBadgeClass(type) {{
  const map = {{
    'kpi_numeric': 'badge-kpi',
    'money': 'badge-money',
    'names': 'badge-names',
    'emails': 'badge-emails',
    'dates': 'badge-dates',
    'ids': 'badge-ids',
  }};
  return map[type] || 'badge-other';
}}

function escapeHtml(text) {{
  const div = document.createElement('div');
  div.textContent = text || '';
  return div.innerHTML;
}}

function escapeAttr(text) {{
  return (text || '').replace(/'/g, "\\\\'").replace(/"/g, '\\\\"');
}}

function openDocModal(docId) {{
  const text = DATA.documents && DATA.documents[docId];
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

init();
</script>
</body>
</html>
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate secrets inventory HTML viewer")
    parser.add_argument(
        "--input",
        default=str(OUTPUTS_DIR / "secret_inventory.jsonl"),
        help="Input secret inventory JSONL",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUTS_DIR / "secrets_viewer.html"),
        help="Output HTML path",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the viewer in browser after generating",
    )
    parser.add_argument(
        "--docs",
        default=str(OUTPUTS_DIR / "docs_local.jsonl"),
        help="Path to docs_local.jsonl for document viewing",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    docs_path = Path(args.docs)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    print(f"Loading {input_path}...")
    records = load_jsonl(input_path)
    print(f"  Loaded {len(records)} documents")

    print(f"Loading documents from {docs_path}...")
    documents = load_documents(docs_path)
    print(f"  Loaded {len(documents)} document texts")

    print("Building hierarchy...")
    hierarchy = build_hierarchy(records)
    print(f"  {len(hierarchy)} tasks")

    print("Computing stats...")
    stats = compute_stats(records)
    print(f"  {stats['total_secrets']} secrets, {stats['verified_count']} verified")

    print("Generating HTML...")
    html = generate_html(hierarchy, stats, records, documents)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Written to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    if args.open:
        webbrowser.open(f"file://{output_path.absolute()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
