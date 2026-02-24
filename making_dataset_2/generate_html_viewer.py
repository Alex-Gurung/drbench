#!/usr/bin/env python3
"""Generate a self-contained HTML viewer for secret_inventory.jsonl results."""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from making_dataset.generate.privacy_tagger import (
    PROMPT_TEMPLATE,
    QUALITY_CHECK_TEMPLATE,
    ANSWER_WITH_DOC_TEMPLATE,
    ANSWER_WITHOUT_DOC_TEMPLATE,
    _extract_text,
)


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return html.escape(text)


def load_original_qa_pairs(tasks_root: Path) -> list[dict[str, Any]]:
    """Load all original qa_dict.json files from drbench tasks."""
    qa_pairs = []
    
    for task_dir in sorted(tasks_root.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("DR"):
            continue
        
        files_dir = task_dir / "files"
        if not files_dir.exists():
            continue
        
        for subdir in sorted(files_dir.iterdir()):
            if not subdir.is_dir():
                continue
            
            qa_dict_file = subdir / "qa_dict.json"
            if not qa_dict_file.exists():
                continue
            
            try:
                with open(qa_dict_file) as f:
                    qa_data = json.load(f)
                
                # Extract document path
                doc_id = f"local/{task_dir.name}/{subdir.name}"
                doc_name = subdir.name
                
                qa_pairs.append({
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "question": qa_data.get("specific_question", ""),
                    "answer": qa_data.get("answer", ""),
                    "qa_type": qa_data.get("qa_type", ""),
                    "justification": qa_data.get("justification", ""),
                    "dr_question": qa_data.get("dr_question", ""),
                    "insight_id": qa_data.get("insight_id", ""),
                })
            except Exception as e:
                print(f"Warning: Failed to load {qa_dict_file}: {e}")
                continue
    
    return qa_pairs


def generate_html(
    secret_inventory_path: Path,
    tasks_root: Path,
    output_path: Path,
) -> None:
    """Generate self-contained HTML viewer."""
    
    # Load new secrets
    documents = []
    with open(secret_inventory_path) as f:
        for line in f:
            if not line.strip():
                continue
            documents.append(json.loads(line))
    
    # Collect all new secrets with document info
    all_new_secrets = []
    for doc_idx, doc in enumerate(documents):
        doc_id = doc.get("doc_id", f"document_{doc_idx}")
        doc_name = Path(doc_id).name if doc_id else f"Document {doc_idx+1}"
        secrets = doc.get("secrets", [])
        
        # Load document text
        doc_text = ""
        doc_path = None
        if doc_id.startswith("local/"):
            parts = doc_id.split("/")
            if len(parts) >= 3:
                task_id = parts[1]
                file_path = tasks_root / task_id / "files" / "/".join(parts[2:])
                if file_path.exists():
                    doc_path = file_path
                    doc_text = _extract_text(file_path)
        
        for secret_idx, secret in enumerate(secrets):
            all_new_secrets.append({
                "doc_idx": doc_idx,
                "secret_idx": secret_idx,
                "doc_id": doc_id,
                "doc_name": doc_name,
                "doc_path": str(doc_path) if doc_path else None,
                "doc_text": doc_text,
                "secret": secret,
                "source": "new",
            })
    
    # Load original QA pairs
    original_qa_pairs = load_original_qa_pairs(tasks_root)
    all_original_qa = []
    for qa_idx, qa in enumerate(original_qa_pairs):
        # Try to load document text
        doc_text = ""
        doc_path = None
        if qa["doc_id"].startswith("local/"):
            parts = qa["doc_id"].split("/")
            if len(parts) >= 3:
                task_id = parts[1]
                file_path = tasks_root / task_id / "files" / "/".join(parts[2:])
                if file_path.exists():
                    doc_path = file_path
                    doc_text = _extract_text(file_path)
        
        all_original_qa.append({
            "qa_idx": qa_idx,
            "doc_id": qa["doc_id"],
            "doc_name": qa["doc_name"],
            "doc_path": str(doc_path) if doc_path else None,
            "doc_text": doc_text,
            "question": qa["question"],
            "answer": qa["answer"],
            "qa_type": qa["qa_type"],
            "justification": qa["justification"],
            "dr_question": qa["dr_question"],
            "insight_id": qa["insight_id"],
            "source": "original",
        })
    
    # Calculate statistics
    total_new_docs = len(documents)
    total_new_secrets = len(all_new_secrets)
    accepted_new_secrets = sum(
        1 for s in all_new_secrets
        if s["secret"].get("doc_only_check", {}).get("with_doc")
        and "NOT_ANSWERABLE" not in str(s["secret"].get("doc_only_check", {}).get("with_doc", "")).upper()
    )
    
    total_original_qa = len(all_original_qa)
    
    # Prepare base64-encoded data for embedding
    new_secrets_b64_json = json.dumps(base64.b64encode(json.dumps(all_new_secrets, default=str, ensure_ascii=True).encode('utf-8')).decode('ascii'))
    original_qa_b64_json = json.dumps(base64.b64encode(json.dumps(all_original_qa, default=str, ensure_ascii=True).encode('utf-8')).decode('ascii'))
    rejected_new_secrets = total_new_secrets - accepted_new_secrets
    
    # Build HTML - use .format() instead of f-string to avoid brace escaping issues
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Privacy Tagger Results Viewer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f5;
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 2em;
            color: #333;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        
        .header p {{
            color: #666;
            font-size: 1em;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-top: 25px;
        }}
        
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            text-align: center;
            border: 1px solid #e0e0e0;
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
            color: #333;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .toggle-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .toggle-label {{
            font-weight: 500;
            color: #333;
        }}
        
        .toggle-switch {{
            position: relative;
            display: inline-block;
            width: 60px;
            height: 30px;
        }}
        
        .toggle-switch input {{
            opacity: 0;
            width: 0;
            height: 0;
        }}
        
        .slider {{
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: 0.3s;
            border-radius: 30px;
        }}
        
        .slider:before {{
            position: absolute;
            content: "";
            height: 22px;
            width: 22px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: 0.3s;
            border-radius: 50%;
        }}
        
        input:checked + .slider {{
            background-color: #4CAF50;
        }}
        
        input:checked + .slider:before {{
            transform: translateX(30px);
        }}
        
        .controls {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .search-box {{
            width: 100%;
            padding: 12px 15px;
            font-size: 1em;
            border: 1px solid #ddd;
            border-radius: 6px;
            transition: border-color 0.2s;
        }}
        
        .search-box:focus {{
            outline: none;
            border-color: #4CAF50;
        }}
        
        .table-container {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        thead {{
            background: #f8f9fa;
            border-bottom: 2px solid #e0e0e0;
        }}
        
        th {{
            padding: 15px 20px;
            text-align: left;
            font-weight: 600;
            color: #333;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        tbody tr {{
            border-bottom: 1px solid #f0f0f0;
            cursor: pointer;
            transition: background-color 0.2s;
        }}
        
        tbody tr:hover {{
            background-color: #f8f9fa;
        }}
        
        tbody tr:last-child {{
            border-bottom: none;
        }}
        
        td {{
            padding: 15px 20px;
            color: #333;
        }}
        
        .question-cell {{
            font-weight: 500;
            max-width: 500px;
        }}
        
        .answer-cell {{
            color: #2e7d32;
            font-weight: 500;
            max-width: 300px;
        }}
        
        .type-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: 500;
            text-transform: uppercase;
        }}
        
        .type-kpi_numeric {{ background: #e3f2fd; color: #1565c0; }}
        .type-money {{ background: #fff3e0; color: #e65100; }}
        .type-dates {{ background: #f3e5f5; color: #6a1b9a; }}
        .type-names {{ background: #e8f5e9; color: #2e7d32; }}
        .type-emails {{ background: #fce4ec; color: #c2185b; }}
        .type-other_sensitive {{ background: #fff9c4; color: #f57f17; }}
        .type-insight {{ background: #e8f5e9; color: #2e7d32; }}
        .type-distractor {{ background: #ffebee; color: #c62828; }}
        
        .status-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: 500;
        }}
        
        .status-accepted {{
            background: #e8f5e9;
            color: #2e7d32;
        }}
        
        .status-rejected {{
            background: #ffebee;
            color: #c62828;
        }}
        
        .doc-name {{
            color: #666;
            font-size: 0.9em;
        }}
        
        /* Modal */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
            animation: fadeIn 0.2s;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        .modal-content {{
            background-color: white;
            margin: 2% auto;
            padding: 0;
            border-radius: 8px;
            width: 90%;
            max-width: 1200px;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            animation: slideDown 0.2s;
        }}
        
        @keyframes slideDown {{
            from {{
                transform: translateY(-20px);
                opacity: 0;
            }}
            to {{
                transform: translateY(0);
                opacity: 1;
            }}
        }}
        
        .modal-header {{
            background: #f8f9fa;
            padding: 25px 30px;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .modal-header h2 {{
            font-size: 1.5em;
            font-weight: 600;
            color: #333;
        }}
        
        .close {{
            color: #666;
            font-size: 2em;
            font-weight: bold;
            cursor: pointer;
            line-height: 1;
            transition: color 0.2s;
        }}
        
        .close:hover {{
            color: #333;
        }}
        
        .modal-body {{
            padding: 30px;
        }}
        
        .detail-section {{
            margin-bottom: 25px;
        }}
        
        .detail-section h3 {{
            color: #333;
            font-size: 1.2em;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #e0e0e0;
        }}
        
        .detail-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin: 12px 0;
            border-left: 3px solid #4CAF50;
        }}
        
        .detail-box strong {{
            color: #333;
            display: block;
            margin-bottom: 8px;
        }}
        
        .quality-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
            margin: 15px 0;
        }}
        
        .quality-item {{
            background: white;
            padding: 12px;
            border-radius: 6px;
            text-align: center;
            border: 1px solid #e0e0e0;
        }}
        
        .quality-label {{
            font-size: 0.8em;
            color: #666;
            margin-bottom: 6px;
            text-transform: capitalize;
        }}
        
        .quality-value {{
            font-size: 1.8em;
            font-weight: bold;
        }}
        
        .quality-pass {{
            color: #2e7d32;
        }}
        
        .quality-fail {{
            color: #c62828;
        }}
        
        .prompt-display {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            white-space: pre-wrap;
            overflow-x: auto;
            margin: 12px 0;
        }}
        
        .doc-text-display {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
        }}
        
        .doc-check-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin: 15px 0;
        }}
        
        .doc-check-box {{
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }}
        
        .doc-check-with {{
            background: #e8f5e9;
            border-left: 3px solid #2e7d32;
        }}
        
        .doc-check-without {{
            background: #ffebee;
            border-left: 3px solid #c62828;
        }}
        
        .doc-check-result {{
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            margin-top: 10px;
            padding: 10px;
            background: white;
            border-radius: 4px;
        }}
        
        details {{
            margin: 12px 0;
        }}
        
        summary {{
            cursor: pointer;
            padding: 10px;
            background: #f0f0f0;
            border-radius: 4px;
            font-weight: 500;
            user-select: none;
        }}
        
        summary:hover {{
            background: #e0e0e0;
        }}
        
        .empty-state {{
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }}
        
        .empty-state h3 {{
            font-size: 1.3em;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Privacy Tagger Results Viewer</h1>
            <p>Compare original drbench QA pairs with LLM-generated secrets</p>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">""" + str(total_new_secrets) + """</div>
                    <div class="stat-label">New Secrets (Total)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">""" + str(accepted_new_secrets) + """</div>
                    <div class="stat-label">New Secrets (Accepted)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">""" + str(total_original_qa) + """</div>
                    <div class="stat-label">Original QA Pairs (Total)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">""" + str(len([q for q in all_original_qa if q["qa_type"] == "insight"])) + """</div>
                    <div class="stat-label">Original (Insights)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">""" + str(len([q for q in all_original_qa if q["qa_type"] == "distractor"])) + """</div>
                    <div class="stat-label">Original (Distractors)</div>
                </div>
            </div>
            <div class="stats" id="statsContainer" style="margin-top: 15px;">
                <!-- Current view stats will be updated by JavaScript -->
            </div>
        </div>
        
        <div class="toggle-container">
            <span class="toggle-label">Show:</span>
            <label class="toggle-switch">
                <input type="checkbox" id="toggleSwitch" checked>
                <span class="slider"></span>
            </label>
            <span class="toggle-label" id="toggleLabel">New LLM-Generated Secrets</span>
        </div>
        
        <div class="controls">
            <input type="text" class="search-box" id="searchBox" placeholder="🔍 Search questions, answers, or document names...">
        </div>
        
        <div class="table-container">
            <table id="secretsTable">
                <thead>
                    <tr>
                        <th>Document</th>
                        <th>Question</th>
                        <th>Answer</th>
                        <th>Type</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="tableBody">
                    <!-- Rows will be inserted by JavaScript -->
                </tbody>
            </table>
        </div>
    </div>
    
    <!-- Modal -->
    <div id="modal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2 id="modalTitle">Details</h2>
                <span class="close" onclick="closeModal()">&times;</span>
            </div>
            <div class="modal-body" id="modalBody">
                <!-- Content will be inserted here -->
            </div>
        </div>
    </div>
    
    <script>
        // Load JSON data from base64-encoded strings
        let newSecretsData = [];
        let originalQAData = [];
        
        try {{
            const newSecretsB64 = {new_secrets_b64_json};
            const newSecretsJson = atob(newSecretsB64);
            newSecretsData = JSON.parse(newSecretsJson);
            console.log('Loaded', newSecretsData.length, 'new secrets');
        }} catch(e) {{
            console.error('Error parsing newSecretsData:', e);
        }}
        
        try {{
            const originalQAB64 = {original_qa_b64_json};
            const originalQAJson = atob(originalQAB64);
            originalQAData = JSON.parse(originalQAJson);
            console.log('Loaded', originalQAData.length, 'original QA pairs');
        }} catch(e) {{
            console.error('Error parsing originalQAData:', e);
        }}
        
        let currentData = newSecretsData;
        let currentSource = 'new';
        
        function updateStats() {{
            const stats = currentSource === 'new' 
                ? {{
                    total: newSecretsData.length,
                    accepted: {accepted_new_secrets},
                    rejected: {rejected_new_secrets},
                    label: 'New Secrets'
                }}
                : {{
                    total: originalQAData.length,
                    accepted: originalQAData.filter(q => q.qa_type === 'insight').length,
                    rejected: originalQAData.filter(q => q.qa_type === 'distractor').length,
                    label: 'Original QA Pairs'
                }};
            
            document.getElementById('statsContainer').innerHTML = `
                <div style="grid-column: 1 / -1; font-weight: 600; color: #666; margin-bottom: 10px;">Current View: ${{stats.label}}</div>
                <div class="stat-card">
                    <div class="stat-value">${{stats.total}}</div>
                    <div class="stat-label">Total</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${{stats.accepted}}</div>
                    <div class="stat-label">Accepted/Insight</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${{stats.rejected}}</div>
                    <div class="stat-label">Distractors (Non-Insights)</div>
                </div>
            `;
        }}
        
        function renderTable() {{
            const tbody = document.getElementById('tableBody');
            if (!tbody) {{
                console.error('tableBody not found');
                return;
            }}
            tbody.innerHTML = '';
            
            if (!currentData || currentData.length === 0) {{
                tbody.innerHTML = '<tr><td colspan="5" class="empty-state"><h3>No data available</h3></td></tr>';
                return;
            }}
            
            currentData.forEach((item, idx) => {{
                let question, answer, secretType, status, statusText, docName;
                
                if (currentSource === 'new') {{
                    const secret = item.secret;
                    question = secret.question || '';
                    answer = secret.answer || '';
                    secretType = secret.secret_type || 'unknown';
                    const docOnlyCheck = secret.doc_only_check || {{}};
                    const qualityScores = secret.quality_scores || {{}};
                    const withDoc = docOnlyCheck.with_doc || '';
                    const withoutDoc = docOnlyCheck.without_doc || '';
                    
                    if (!withDoc || withDoc.toUpperCase().includes('NOT_ANSWERABLE')) {{
                        status = 'rejected';
                        statusText = 'Rejected';
                    }} else if (withoutDoc && !withoutDoc.toUpperCase().includes('NOT_ANSWERABLE')) {{
                        status = 'rejected';
                        statusText = 'Rejected';
                    }} else if (Object.keys(qualityScores).length > 0 && Math.min(...Object.values(qualityScores)) < 3) {{
                        status = 'rejected';
                        statusText = 'Rejected';
                    }} else {{
                        status = 'accepted';
                        statusText = 'Accepted';
                    }}
                    docName = item.doc_name;
                }} else {{
                    question = item.question || '';
                    answer = item.answer || '';
                    secretType = item.qa_type || 'unknown';
                    status = item.qa_type === 'insight' ? 'accepted' : 'rejected';
                    statusText = item.qa_type === 'insight' ? 'Insight' : 'Distractor';
                    docName = item.doc_name;
                }}
                
                const questionDisplay = question.length > 100 ? question.substring(0, 100) + '...' : question;
                const answerDisplay = answer.length > 80 ? answer.substring(0, 80) + '...' : answer;
                
                const row = document.createElement('tr');
                row.onclick = function() {{ openModal(idx); }};
                row.innerHTML = `
                    <td><div class="doc-name">${{escapeHtml(docName)}}</div></td>
                    <td class="question-cell">${{escapeHtml(questionDisplay)}}</td>
                    <td class="answer-cell">${{escapeHtml(answerDisplay)}}</td>
                    <td><span class="type-badge type-${{secretType}}">${{secretType}}</span></td>
                    <td><span class="status-badge status-${{status}}">${{statusText}}</span></td>
                `;
                tbody.appendChild(row);
            }});
        }}
        
        function openModal(idx) {{
            const item = currentData[idx];
            let content = '';
            
            if (currentSource === 'new') {{
                const secret = item.secret;
                const question = secret.question || '';
                const answer = secret.answer || '';
                const secretType = secret.secret_type || 'unknown';
                const justification = secret.justification || '';
                const qualityScores = secret.quality_scores || {{}};
                const docOnlyCheck = secret.doc_only_check || {{}};
                const docText = item.doc_text || '';
                
                const withDoc = docOnlyCheck.with_doc || '';
                const withoutDoc = docOnlyCheck.without_doc || '';
                let status = 'accepted';
                let statusText = 'Accepted';
                if (!withDoc || withDoc.toUpperCase().includes('NOT_ANSWERABLE')) {{
                    status = 'rejected';
                    statusText = 'Rejected (not answerable with doc)';
                }} else if (withoutDoc && !withoutDoc.toUpperCase().includes('NOT_ANSWERABLE')) {{
                    status = 'rejected';
                    statusText = 'Rejected (answerable without doc)';
                }} else if (Object.keys(qualityScores).length > 0 && Math.min(...Object.values(qualityScores)) < 3) {{
                    status = 'rejected';
                    statusText = 'Rejected (low quality)';
                }}
                
                content = `
                    <div class="detail-section">
                        <h3>Question & Answer</h3>
                        <div class="detail-box">
                            <strong>Question:</strong>
                            ${{escapeHtml(question)}}
                        </div>
                        <div class="detail-box">
                            <strong>Answer:</strong>
                            <span style="color: #2e7d32; font-weight: 600;">${{escapeHtml(answer)}}</span>
                        </div>
                        <div class="detail-box">
                            <strong>Type:</strong>
                            <span class="type-badge type-${{secretType}}">${{secretType}}</span>
                            <span class="status-badge status-${{status}}" style="margin-left: 10px;">${{statusText}}</span>
                        </div>
                        ${{justification ? `<div class="detail-box"><strong>Justification:</strong>${{escapeHtml(justification)}}</div>` : ''}}
                    </div>
                `;
                
                if (Object.keys(qualityScores).length > 0) {{
                    let scoresHtml = '<div class="detail-section"><h3>Quality Scores</h3><div class="quality-grid">';
                    for (const [dim, score] of Object.entries(qualityScores)) {{
                        const pass = score >= 3;
                        scoresHtml += `
                            <div class="quality-item">
                                <div class="quality-label">${{dim.replace(/_/g, ' ')}}</div>
                                <div class="quality-value ${{pass ? 'quality-pass' : 'quality-fail'}}">${{score}}/5</div>
                            </div>
                        `;
                    }}
                    scoresHtml += '</div></div>';
                    content += scoresHtml;
                    
                    if (docText) {{
                        const qualityPrompt = `""" + escape_html(QUALITY_CHECK_TEMPLATE).replace('{question}', '${escapeHtml(question)}').replace('{answer}', '${escapeHtml(answer)}').replace('{chunk_text}', '${escapeHtml(docText)}') + """`;
                        content += `
                            <div class="detail-section">
                                <h3>Quality Check Prompt</h3>
                                <details>
                                    <summary>Show Prompt</summary>
                                    <div class="prompt-display">${{qualityPrompt}}</div>
                                </details>
                            </div>
                        `;
                    }}
                }}
                
                if (Object.keys(docOnlyCheck).length > 0) {{
                    let withDocPrompt = '';
                    let withoutDocPrompt = '';
                    if (docText) {{
                        withDocPrompt = `""" + escape_html(ANSWER_WITH_DOC_TEMPLATE).replace('{chunk_text}', '${escapeHtml(docText)}').replace('{question}', '${escapeHtml(question)}') + """`;
                    }}
                    withoutDocPrompt = `""" + escape_html(ANSWER_WITHOUT_DOC_TEMPLATE).replace('{question}', '${escapeHtml(question)}') + """`;
                    
                    content += `
                        <div class="detail-section">
                            <h3>Document-Only Verification</h3>
                            <div class="doc-check-grid">
                                <div class="doc-check-box doc-check-with">
                                    <strong>✓ Answer WITH Document</strong>
                                    ${{withDocPrompt ? `<details><summary>Show Prompt</summary><div class="prompt-display">${{withDocPrompt}}</div></details>` : ''}}
                                    <div class="doc-check-result">${{escapeHtml(withDoc)}}</div>
                                </div>
                                <div class="doc-check-box doc-check-without">
                                    <strong>✗ Answer WITHOUT Document</strong>
                                    <details><summary>Show Prompt</summary><div class="prompt-display">${{withoutDocPrompt}}</div></details>
                                    <div class="doc-check-result">${{escapeHtml(withoutDoc)}}</div>
                                </div>
                            </div>
                        </div>
                    `;
                }}
                
                if (docText) {{
                    const generationPrompt = `""" + escape_html(PROMPT_TEMPLATE) + """` + escapeHtml(docText);
                    content += `
                        <div class="detail-section">
                            <h3>Full Document Text</h3>
                            <div class="doc-text-display">${{escapeHtml(docText)}}</div>
                        </div>
                        <div class="detail-section">
                            <h3>Generation Prompt</h3>
                            <details>
                                <summary>Show Prompt</summary>
                                <div class="prompt-display">${{generationPrompt}}</div>
                            </details>
                        </div>
                    `;
                }}
            }} else {{
                const question = item.question || '';
                const answer = item.answer || '';
                const qaType = item.qa_type || '';
                const justification = item.justification || '';
                const drQuestion = item.dr_question || '';
                const docText = item.doc_text || '';
                
                content = `
                    <div class="detail-section">
                        <h3>Question & Answer</h3>
                        <div class="detail-box">
                            <strong>Question:</strong>
                            ${{escapeHtml(question)}}
                        </div>
                        <div class="detail-box">
                            <strong>Answer:</strong>
                            <span style="color: #2e7d32; font-weight: 600;">${{escapeHtml(answer)}}</span>
                        </div>
                        <div class="detail-box">
                            <strong>Type:</strong>
                            <span class="type-badge type-${{qaType}}">${{qaType}}</span>
                        </div>
                        ${{justification ? `<div class="detail-box"><strong>Justification:</strong>${{escapeHtml(justification)}}</div>` : ''}}
                        ${{drQuestion ? `<div class="detail-box"><strong>DR Question:</strong>${{escapeHtml(drQuestion)}}</div>` : ''}}
                    </div>
                `;
                
                if (docText) {{
                    content += `
                        <div class="detail-section">
                            <h3>Full Document Text</h3>
                            <div class="doc-text-display">${{escapeHtml(docText)}}</div>
                        </div>
                    `;
                }}
            }}
            
            document.getElementById('modalTitle').textContent = `${{currentSource === 'new' ? 'New Secret' : 'Original QA Pair'}}: ${{escapeHtml(question.substring(0, 50))}}...`;
            document.getElementById('modalBody').innerHTML = content;
            document.getElementById('modal').style.display = 'block';
        }}
        
        function closeModal() {{
            document.getElementById('modal').style.display = 'none';
        }}
        
        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
        
        // Toggle functionality
        document.getElementById('toggleSwitch').addEventListener('change', function(e) {{
            if (e.target.checked) {{
                currentData = newSecretsData;
                currentSource = 'new';
                document.getElementById('toggleLabel').textContent = 'New LLM-Generated Secrets';
            }} else {{
                currentData = originalQAData;
                currentSource = 'original';
                document.getElementById('toggleLabel').textContent = 'Original DrBench QA Pairs';
            }}
            updateStats();
            renderTable();
            document.getElementById('searchBox').value = '';
        }});
        
        // Search functionality
        document.getElementById('searchBox').addEventListener('input', function(e) {{
            const searchTerm = e.target.value.toLowerCase();
            const rows = document.querySelectorAll('#secretsTable tbody tr');
            
            rows.forEach(row => {{
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(searchTerm) ? '' : 'none';
            }});
        }});
        
        // Close modal when clicking outside
        window.onclick = function(event) {{
            const modal = document.getElementById('modal');
            if (event.target === modal) {{
                closeModal();
            }}
        }}
        
        // Initialize
        console.log('Initializing viewer...');
        console.log('New secrets:', newSecretsData.length);
        console.log('Original QA:', originalQAData.length);
        
        try {{
            updateStats();
            renderTable();
            console.log('Viewer initialized successfully');
        }} catch(e) {{
            console.error('Error initializing viewer:', e);
            const tbody = document.getElementById('tableBody');
            if (tbody) {{
                tbody.innerHTML = '<tr><td colspan="5" class="empty-state"><h3>Error loading data. Check console for details.</h3></td></tr>';
            }}
        }}
    </script>
</body>
</html>
"""
    
    # Write HTML file
    # Format the HTML template with actual values
    # Replace our Python variable placeholders first
    html_content = html_content.replace('{new_secrets_b64_json}', new_secrets_b64_json)
    html_content = html_content.replace('{original_qa_b64_json}', original_qa_b64_json)
    html_content = html_content.replace('{accepted_new_secrets}', str(accepted_new_secrets))
    html_content = html_content.replace('{rejected_new_secrets}', str(rejected_new_secrets))
    html_content = html_content.replace('{total_new_secrets}', str(total_new_secrets))
    html_content = html_content.replace('{total_original_qa}', str(total_original_qa))
    html_content = html_content.replace('{insight_count}', str(len([q for q in all_original_qa if q["qa_type"] == "insight"])))
    html_content = html_content.replace('{distractor_count}', str(len([q for q in all_original_qa if q["qa_type"] == "distractor"])))
    
    # Now convert {{ to { and }} to } for JavaScript code
    # Template literals use ${...} so ${{ should become ${, and }} should become }
    # First, replace ${{ with ${ (template literals)
    html_content = html_content.replace('${{', '${')
    # Then replace remaining {{ with {
    html_content = html_content.replace('{{', '{')
    # Replace }} with }
    html_content = html_content.replace('}}', '}')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ Generated HTML viewer: {output_path}")
    print(f"   New secrets: {total_new_secrets} (accepted: {accepted_new_secrets})")
    print(f"   Original QA pairs: {total_original_qa}")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML viewer for secret_inventory.jsonl")
    parser.add_argument(
        "--input",
        default=str(ROOT_DIR / "making_dataset_2" / "outputs" / "secret_inventory.jsonl"),
        help="Path to secret_inventory.jsonl",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT_DIR / "making_dataset_2" / "outputs" / "privacy_tagger_viewer.html"),
        help="Output HTML file path",
    )
    parser.add_argument(
        "--tasks-root",
        default=str(ROOT_DIR / "drbench" / "data" / "tasks"),
        help="Root tasks directory for loading document text",
    )
    args = parser.parse_args()
    
    generate_html(
        secret_inventory_path=Path(args.input),
        tasks_root=Path(args.tasks_root),
        output_path=Path(args.output),
    )
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
