#!/usr/bin/env python3
"""Generate an interactive HTML viewer for extracted secrets.

Usage:
  python scripts/view_secrets_html.py                        # default input, opens browser
  python scripts/view_secrets_html.py --input path/to.jsonl  # custom input
  python scripts/view_secrets_html.py --no-open              # don't open browser
"""

import argparse
import html as html_mod
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_PATH = Path("making_dataset_2/outputs/secrets_step35flash/extracted_secrets.jsonl")


def _safe_json(obj) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    return s.replace("</", "<\\/").replace("<!--", "<\\!--")


def generate(rows: list[dict], output: Path):
    total = sum(len(r["secrets"]) for r in rows)
    tasks_with = sum(1 for r in rows if r["secrets"])
    data_json = _safe_json(rows)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Secret Inventory Viewer — {len(rows)} tasks, {total} secrets</title>
<style>
:root {{
  --bg: #f8f9fb; --card: #fff; --border: #e5e7eb; --border-light: #f0f1f3;
  --primary: #6366f1; --primary-light: #eef2ff;
  --success: #059669; --success-bg: #ecfdf5;
  --warning: #d97706; --warning-bg: #fffbeb;
  --danger: #dc2626; --danger-bg: #fef2f2;
  --text: #1a1a2e; --text2: #6b7280; --text-muted: #9ca3af;
  --radius: 10px;
  --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        background: var(--bg); color: var(--text); line-height: 1.6; font-size: 14px; }}

/* Header */
header {{ background: var(--card); padding: 14px 24px; border-bottom: 1px solid var(--border);
          position: sticky; top: 0; z-index: 100; box-shadow: var(--shadow); }}
header h1 {{ font-size: 16px; font-weight: 600; color: var(--primary); margin-bottom: 4px; }}
.header-stats {{ display: flex; gap: 20px; font-size: 13px; color: var(--text-muted); flex-wrap: wrap; }}
.header-stats b {{ color: var(--text); }}

/* Filters */
.filters {{ display: flex; gap: 10px; padding: 10px 24px; background: var(--card);
            border-bottom: 1px solid var(--border); flex-wrap: wrap; align-items: center; }}
.filters input, .filters select {{ padding: 6px 10px; border: 1px solid var(--border);
    border-radius: 6px; font-size: 13px; background: white; }}
.filters input {{ min-width: 260px; flex: 1; max-width: 500px; }}
.filters input:focus, .filters select:focus {{ outline: none; border-color: var(--primary); }}
.filter-label {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.03em; }}
.match-count {{ font-size: 12px; color: var(--text-muted); margin-left: auto; }}

/* Layout */
.main {{ display: grid; grid-template-columns: 340px 1fr; height: calc(100vh - 120px); }}
.sidebar {{ background: var(--card); border-right: 1px solid var(--border); overflow-y: auto; }}
.detail {{ overflow-y: auto; padding: 20px 24px; }}

/* Sidebar */
.task-item {{ padding: 10px 14px; border-bottom: 1px solid var(--bg); cursor: pointer;
              transition: background 0.12s; display: flex; justify-content: space-between; align-items: center; }}
.task-item:hover {{ background: var(--primary-light); }}
.task-item.selected {{ background: var(--primary-light); border-left: 3px solid var(--primary); }}
.task-item .tid {{ font-weight: 600; font-size: 13px; }}
.task-item .count {{ font-size: 12px; color: var(--text-muted); }}
.task-item .bar {{ height: 4px; background: var(--primary); border-radius: 2px; opacity: 0.3; margin-top: 4px; }}
.task-item.selected .bar {{ opacity: 0.7; }}
.task-meta {{ font-size: 11px; color: var(--text-muted); }}

/* Detail */
.detail h2 {{ font-size: 15px; font-weight: 600; margin-bottom: 4px; }}
.detail .task-summary {{ font-size: 13px; color: var(--text2); margin-bottom: 16px; }}

/* Secret cards */
.secret {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
           margin: 8px 0; padding: 14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
           transition: box-shadow 0.15s; }}
.secret:hover {{ box-shadow: var(--shadow); }}
.secret .num {{ font-size: 11px; font-weight: 600; color: var(--text-muted); margin-bottom: 4px; }}
.secret .question {{ font-size: 13px; color: var(--text); margin-bottom: 6px; line-height: 1.5; }}
.secret .answer {{ font-size: 14px; font-weight: 600; color: var(--primary); padding: 6px 10px;
                   background: var(--primary-light); border-radius: 6px; display: inline-block; margin-bottom: 4px; }}
.secret .source {{ font-size: 11px; color: var(--text-muted); margin-top: 4px; }}
.secret .source code {{ background: #f3f4f6; padding: 1px 5px; border-radius: 3px; font-size: 11px; }}

/* Credential secrets */
.secret.credential {{ border-left: 3px solid var(--danger); }}
.secret.credential .answer {{ background: var(--danger-bg); color: var(--danger); }}

/* Highlight */
mark {{ background: #fef08a; padding: 0 2px; border-radius: 2px; }}

/* Empty */
.empty {{ text-align: center; padding: 60px 20px; color: var(--text-muted); }}
.empty h3 {{ font-size: 16px; margin-bottom: 6px; color: var(--text2); }}

/* Source groups */
.source-group {{ margin-top: 16px; }}
.source-group h3 {{ font-size: 12px; font-weight: 600; color: var(--text2); padding: 6px 0;
                     border-bottom: 1px solid var(--border-light); margin-bottom: 6px; }}
</style>
</head>
<body>

<header>
  <h1>Secret Inventory Viewer</h1>
  <div class="header-stats">
    <span><b>{len(rows)}</b> tasks</span>
    <span><b>{tasks_with}</b> with secrets</span>
    <span><b>{total}</b> total secrets</span>
    <span id="filtered-count"></span>
  </div>
</header>

<div class="filters">
  <div>
    <div class="filter-label">Search</div>
    <input type="text" id="search" placeholder="Search questions, answers, sources...">
  </div>
  <div>
    <div class="filter-label">Sort by</div>
    <select id="sort">
      <option value="id">Task ID</option>
      <option value="count-desc">Most secrets</option>
      <option value="count-asc">Fewest secrets</option>
      <option value="density">Density (per file)</option>
    </select>
  </div>
  <div class="match-count" id="match-count"></div>
</div>

<div class="main">
  <div class="sidebar" id="sidebar"></div>
  <div class="detail" id="detail">
    <div class="empty">
      <h3>Select a task</h3>
      <p>Click a task in the sidebar to view its secrets</p>
    </div>
  </div>
</div>

<script>
const DATA = {data_json};
let selectedTask = null;
let searchQuery = '';

const credentialPatterns = /password|api.key|credential|secret.key|token|auth/i;

function isCredential(s) {{
  return credentialPatterns.test(s.question || '') || credentialPatterns.test(s.answer || '');
}}

function highlight(text, query) {{
  if (!query) return escHtml(text);
  const esc = escHtml(text);
  const re = new RegExp('(' + query.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&') + ')', 'gi');
  return esc.replace(re, '<mark>$1</mark>');
}}

function escHtml(s) {{
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}}

function matchesSearch(task, q) {{
  if (!q) return true;
  const ql = q.toLowerCase();
  if (task.task_id.toLowerCase().includes(ql)) return true;
  return task.secrets.some(s =>
    (s.question || '').toLowerCase().includes(ql) ||
    String(s.answer || '').toLowerCase().includes(ql) ||
    (s.source_file || '').toLowerCase().includes(ql) ||
    (s.source || '').toLowerCase().includes(ql)
  );
}}

function filteredSecrets(task, q) {{
  if (!q) return task.secrets;
  const ql = q.toLowerCase();
  return task.secrets.filter(s =>
    (s.question || '').toLowerCase().includes(ql) ||
    String(s.answer || '').toLowerCase().includes(ql) ||
    (s.source_file || '').toLowerCase().includes(ql) ||
    (s.source || '').toLowerCase().includes(ql)
  );
}}

function getMaxSecrets() {{
  return Math.max(1, ...DATA.map(t => t.secrets.length));
}}

function sortedData() {{
  const sort = document.getElementById('sort').value;
  const filtered = DATA.filter(t => matchesSearch(t, searchQuery));
  if (sort === 'count-desc') filtered.sort((a, b) => b.secrets.length - a.secrets.length);
  else if (sort === 'count-asc') filtered.sort((a, b) => a.secrets.length - b.secrets.length);
  else if (sort === 'density') filtered.sort((a, b) =>
    (b.files_processed ? b.secrets.length / b.files_processed : 0) -
    (a.files_processed ? a.secrets.length / a.files_processed : 0));
  else filtered.sort((a, b) => a.task_id.localeCompare(b.task_id));
  return filtered;
}}

function renderSidebar() {{
  const tasks = sortedData();
  const maxS = getMaxSecrets();
  const totalFiltered = tasks.reduce((s, t) => s + t.secrets.length, 0);
  document.getElementById('match-count').textContent =
    searchQuery ? tasks.length + ' tasks, ' + totalFiltered + ' secrets matching' : '';

  const sb = document.getElementById('sidebar');
  sb.innerHTML = tasks.map(t => {{
    const pct = Math.round(t.secrets.length / maxS * 100);
    const density = t.files_processed ? (t.secrets.length / t.files_processed).toFixed(1) : '–';
    const sel = selectedTask === t.task_id ? ' selected' : '';
    const matchCount = searchQuery ? filteredSecrets(t, searchQuery).length : t.secrets.length;
    return `<div class="task-item${{sel}}" data-id="${{t.task_id}}" onclick="selectTask('${{t.task_id}}')">
      <div>
        <div class="tid">${{t.task_id}}</div>
        <div class="task-meta">${{t.files_processed}} files &middot; ${{density}}/file</div>
        <div class="bar" style="width:${{pct}}%"></div>
      </div>
      <div class="count">${{matchCount}}</div>
    </div>`;
  }}).join('');
}}

function renderDetail() {{
  const det = document.getElementById('detail');
  if (!selectedTask) {{
    det.innerHTML = '<div class="empty"><h3>Select a task</h3><p>Click a task in the sidebar</p></div>';
    return;
  }}
  const task = DATA.find(t => t.task_id === selectedTask);
  if (!task) return;

  const secrets = filteredSecrets(task, searchQuery);

  // Group by source file
  const groups = {{}};
  secrets.forEach((s, i) => {{
    const src = s.source_file || s.source || 'Unknown';
    if (!groups[src]) groups[src] = [];
    groups[src].push({{ ...s, _idx: i + 1 }});
  }});

  const nCred = secrets.filter(isCredential).length;
  let html = `<h2>${{task.task_id}}</h2>
    <div class="task-summary">
      ${{secrets.length}} secrets from ${{task.files_processed}} files
      ${{nCred ? ' &middot; <span style="color:var(--danger)">' + nCred + ' credentials</span>' : ''}}
    </div>`;

  const srcKeys = Object.keys(groups).sort();
  for (const src of srcKeys) {{
    html += `<div class="source-group"><h3>${{escHtml(src)}}</h3>`;
    for (const s of groups[src]) {{
      const cred = isCredential(s) ? ' credential' : '';
      html += `<div class="secret${{cred}}">
        <div class="num">#${{s._idx}}</div>
        <div class="question">${{highlight(s.question || '', searchQuery)}}</div>
        <div class="answer">${{highlight(String(s.answer || ''), searchQuery)}}</div>
        ${{s.source ? '<div class="source">Source: ' + highlight(s.source, searchQuery) + '</div>' : ''}}
      </div>`;
    }}
    html += '</div>';
  }}

  if (!secrets.length) {{
    html += '<div class="empty"><h3>No matching secrets</h3></div>';
  }}
  det.innerHTML = html;
}}

function selectTask(id) {{
  selectedTask = id;
  renderSidebar();
  renderDetail();
}}

document.getElementById('search').addEventListener('input', e => {{
  searchQuery = e.target.value.trim();
  renderSidebar();
  renderDetail();
}});
document.getElementById('sort').addEventListener('change', () => {{
  renderSidebar();
}});

// Init
renderSidebar();
if (DATA.length && DATA[0].secrets.length) selectTask(DATA.find(t => t.secrets.length)?.task_id || DATA[0].task_id);
</script>
</body>
</html>"""
    output.write_text(html)
    print(f"Written to {output} ({len(rows)} tasks, {total} secrets)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    rows = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    out = args.output or args.input.parent / "secrets_viewer.html"
    generate(rows, out)

    if not args.no_open:
        try:
            subprocess.run(["xdg-open", str(out)], check=False, timeout=5)
        except Exception:
            pass


if __name__ == "__main__":
    main()
