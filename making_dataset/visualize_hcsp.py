#!/usr/bin/env python3
"""
HCSP Dataset Visualizer Generator

Extends the base dataset visualizer with InfoSeek-style HCSP task visualization:
- HCSP Tasks tab: Full task browser with research tree viz, linkage badges
- Constraints tab: Browse all constraints across tasks
- Validation tab: Summary stats, pass rates, ablation results
- Enhanced Trees tab: HCSP Research Trees sub-view

Usage:
    python visualize_hcsp.py
    # Output: outputs/visualizer_hcsp.html
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
SCOPES_DIR = OUTPUTS_DIR / "scopes" / "task"


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


def load_hcsp_tasks(scopes_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load HCSP tasks from outputs/scopes/task/<TASK_ID>/*.hcsp.jsonl"""
    tasks_by_id: dict[str, list[dict[str, Any]]] = {}

    if not scopes_dir.exists():
        return tasks_by_id

    for task_dir in sorted(scopes_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name
        tasks_by_id[task_id] = []

        # Load all .hcsp.jsonl files
        for hcsp_file in task_dir.glob("*.hcsp.jsonl"):
            for task in load_jsonl(hcsp_file):
                task["_task_id"] = task_id
                task["_source_file"] = hcsp_file.name
                tasks_by_id[task_id].append(task)

    return tasks_by_id


def extract_all_constraints(hcsp_tasks_by_id: dict[str, list[dict]]) -> list[dict]:
    """Extract constraints across all tasks for Constraints tab."""
    all_constraints = []

    for task_id, tasks in hcsp_tasks_by_id.items():
        for task_idx, task in enumerate(tasks):
            hcsp = task.get("tree", {}).get("hcsp", {})
            nodes = hcsp.get("nodes", {})

            for node_id, node in nodes.items():
                if node.get("kind") == "constraint":
                    constraint = node.get("constraint", {})
                    all_constraints.append({
                        "task_id": task_id,
                        "task_idx": task_idx,
                        "node_id": node_id,
                        "text": constraint.get("text", ""),
                        "constraint_type": constraint.get("constraint_type", "other"),
                        "corpus": constraint.get("corpus", "unknown"),
                        "evidence": constraint.get("evidence", {}),
                        "question": task.get("question", ""),
                        "linkage_type": task.get("gold", {}).get("linkage_type", ""),
                    })

    return all_constraints


def compute_validation_summary(hcsp_tasks_by_id: dict[str, list[dict]]) -> dict:
    """Compute pass rates and failure list."""
    total_tasks = 0
    passed_tasks = 0
    ablation_full_pass = 0
    ablation_no_info_pass = 0
    failures = []
    linkage_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}

    for task_id, tasks in hcsp_tasks_by_id.items():
        for task in tasks:
            total_tasks += 1
            quality = task.get("quality", {})

            # Track pass/fail
            if quality.get("deterministic_pass"):
                passed_tasks += 1
            else:
                failures.append({
                    "task_id": task_id,
                    "question": task.get("question", "")[:80],
                    "reason": "deterministic_check_failed",
                })

            # Ablation checks
            if quality.get("ablation_full_info_pass"):
                ablation_full_pass += 1
            if quality.get("ablation_no_info_pass"):
                ablation_no_info_pass += 1

            # Linkage type distribution
            linkage = task.get("gold", {}).get("linkage_type", "unknown")
            linkage_counts[linkage] = linkage_counts.get(linkage, 0) + 1

            # Mode distribution
            mode = task.get("mode", "unknown")
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

    return {
        "total_tasks": total_tasks,
        "passed_tasks": passed_tasks,
        "pass_rate": passed_tasks / total_tasks if total_tasks > 0 else 0,
        "ablation_full_pass": ablation_full_pass,
        "ablation_no_info_pass": ablation_no_info_pass,
        "failures": failures[:50],  # Limit to 50
        "linkage_counts": linkage_counts,
        "mode_counts": mode_counts,
    }


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

    # HCSP prompts
    synthesize_file = GENERATE_DIR / "hcsp" / "synthesize.py"
    if synthesize_file.exists():
        content = synthesize_file.read_text()
        match = re.search(r'SYNTHESIZE_QUESTION_TEMPLATE\s*=\s*"""(.+?)"""', content, re.DOTALL)
        if match:
            prompts["hcsp_question_synthesis"] = match.group(1).strip()

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
        },
        {
            "name": "hcsp_dataset.py",
            "source": "making_dataset/generate/hcsp_dataset.py",
            "inputs": ["chunks_local.jsonl", "secret_inventory.jsonl", "web corpus"],
            "outputs": ["outputs/scopes/task/DR*/*.hcsp.jsonl"],
            "description": "InfoSeek-style HCSP task generation with research trees",
            "has_prompts": True,
            "prompt_keys": ["hcsp_question_synthesis"]
        },
    ]


def generate_html(data: dict[str, Any]) -> str:
    """Generate the complete HTML visualizer with HCSP support."""
    data_json = json.dumps(data, ensure_ascii=False, indent=None)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HCSP Dataset Visualizer</title>
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
  max-width: 1800px;
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
  flex-wrap: wrap;
}}
.tab {{
  padding: 16px 16px;
  border: none;
  background: none;
  cursor: pointer;
  font-size: 13px;
  color: var(--text-muted);
  border-bottom: 2px solid transparent;
  transition: all 0.2s;
  white-space: nowrap;
}}
.tab:hover {{ color: var(--text); }}
.tab.active {{
  color: var(--primary);
  border-bottom-color: var(--primary);
}}
.tab.hcsp-tab {{
  background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
  border-radius: 6px 6px 0 0;
  font-weight: 500;
}}
main {{
  max-width: 1800px;
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
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
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

/* Linkage type badges */
.badge-entity_chain {{ background: #e0f2fe; color: #0369a1; }}
.badge-computational {{ background: #fce7f3; color: #be185d; }}
.badge-selection {{ background: #dcfce7; color: #166534; }}
.badge-definitional {{ background: #f3e8ff; color: #7e22ce; }}
.badge-creative {{ background: #fef9c3; color: #854d0e; }}

/* Constraint type badges */
.badge-attribute {{ background: #dbeafe; color: #1e40af; }}
.badge-relation {{ background: #fef3c7; color: #92400e; }}
.badge-temporal {{ background: #d1fae5; color: #065f46; }}

/* Hop pattern visualization */
.hop-pattern {{
  display: flex;
  gap: 4px;
  align-items: center;
}}
.hop-badge {{
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  font-weight: 600;
}}
.hop-badge.L {{ background: #eef1ff; border: 2px solid var(--local); color: var(--local); }}
.hop-badge.W {{ background: #f3e8ff; border: 2px solid var(--web); color: var(--web); }}

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
  cursor: pointer;
  transition: transform 0.15s, box-shadow 0.15s;
}}
.task-card:hover {{
  transform: translateY(-2px);
  box-shadow: 0 4px 16px rgba(0,0,0,0.12);
}}
.task-header {{
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
  flex-wrap: wrap;
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

/* HCSP Task detail */
.hcsp-task-detail {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}}
.hcsp-constraints-list {{
  display: flex;
  flex-direction: column;
  gap: 8px;
}}
.hcsp-constraint-item {{
  padding: 12px;
  background: #f8f9fa;
  border-radius: 8px;
  border-left: 3px solid var(--border);
}}
.hcsp-constraint-item.local {{ border-left-color: var(--local); }}
.hcsp-constraint-item.web {{ border-left-color: var(--web); }}

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
.tree-node.question rect {{ fill: #fef3c7; stroke: #f59e0b; stroke-width: 2; }}
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
  max-width: 900px;
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

/* Validation status */
.status-pass {{ color: #166534; }}
.status-fail {{ color: #dc2626; }}

/* Sub-tabs */
.sub-tabs {{
  display: flex;
  gap: 4px;
  margin-bottom: 16px;
  border-bottom: 1px solid var(--border);
}}
.sub-tab {{
  padding: 10px 16px;
  border: none;
  background: none;
  cursor: pointer;
  font-size: 13px;
  color: var(--text-muted);
  border-bottom: 2px solid transparent;
}}
.sub-tab:hover {{ color: var(--text); }}
.sub-tab.active {{
  color: var(--primary);
  border-bottom-color: var(--primary);
}}
.sub-view {{ display: none; }}
.sub-view.active {{ display: block; }}
</style>
</head>
<body>
<header>
  <div class="header-content">
    <div class="logo">HCSP Dataset Visualizer</div>
    <nav>
      <button class="tab active" data-view="overview">Overview</button>
      <button class="tab" data-view="chunks">Chunks</button>
      <button class="tab" data-view="trees">Trees</button>
      <button class="tab" data-view="secrets">Secrets</button>
      <button class="tab" data-view="tasks">Q&A Tasks</button>
      <button class="tab hcsp-tab" data-view="hcsp-tasks">HCSP Tasks</button>
      <button class="tab hcsp-tab" data-view="constraints">Constraints</button>
      <button class="tab hcsp-tab" data-view="validation">Validation</button>
      <button class="tab" data-view="logs">LLM Logs</button>
      <button class="tab" data-view="architecture">Architecture</button>
    </nav>
  </div>
</header>

<main>
  <div id="overview" class="view active"></div>
  <div id="chunks" class="view"></div>
  <div id="trees" class="view"></div>
  <div id="secrets" class="view"></div>
  <div id="tasks" class="view"></div>
  <div id="hcsp-tasks" class="view"></div>
  <div id="constraints" class="view"></div>
  <div id="validation" class="view"></div>
  <div id="logs" class="view"></div>
  <div id="architecture" class="view"></div>
</main>

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

const LINKAGE_INFO = {{
  entity_chain: {{ emoji: '🔗', desc: 'Web reveals entity needed for local lookup' }},
  computational: {{ emoji: '🧮', desc: 'Answer requires combining local + web facts' }},
  selection: {{ emoji: '🎯', desc: 'Web provides filter criterion for local options' }},
  definitional: {{ emoji: '📖', desc: 'Web defines term used in local' }},
  creative: {{ emoji: '✨', desc: 'LLM-discovered cross-corpus linkage' }},
}};

document.querySelectorAll('.tab').forEach(tab => {{
  tab.addEventListener('click', () => {{
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.view).classList.add('active');
  }});
}});

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

function escapeHtml(text) {{
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}}

function getBadgeClass(type) {{
  const map = {{
    'local': 'badge-local', 'web': 'badge-web', 'mixed': 'badge-mixed',
    'local_only': 'badge-local', 'web_only': 'badge-web',
    'kpi_numeric': 'badge-kpi', 'money': 'badge-money', 'names': 'badge-names',
    'emails': 'badge-emails', 'dates': 'badge-dates', 'ids': 'badge-ids',
    'entity_chain': 'badge-entity_chain', 'computational': 'badge-computational',
    'selection': 'badge-selection', 'definitional': 'badge-definitional',
    'creative': 'badge-creative', 'attribute': 'badge-attribute',
    'relation': 'badge-relation', 'temporal': 'badge-temporal',
  }};
  return map[type] || 'badge-other';
}}

function truncate(text, len = 100) {{
  if (!text) return '';
  return text.length > len ? text.substring(0, len) + '...' : text;
}}

function renderHopPattern(pattern) {{
  if (!pattern) return '';
  return pattern.split('').map(c => `<span class="hop-badge ${{c}}">${{c}}</span>`).join('');
}}

function getLinkageBadge(linkageType) {{
  if (!linkageType) return '';
  const info = LINKAGE_INFO[linkageType] || {{ emoji: '❓', desc: 'Unknown' }};
  return `<span class="badge ${{getBadgeClass(linkageType)}}" title="${{info.desc}}">${{info.emoji}} ${{linkageType}}</span>`;
}}

function renderOverview() {{
  const localCount = DATA.chunks_local.length;
  const webCount = DATA.chunks_web.length;
  const secretCount = DATA.secrets.reduce((sum, s) => sum + (s.secrets?.length || 0), 0);
  const localTasks = DATA.local_only.length;
  const webTasks = DATA.web_only.length;
  const mixedTasks = DATA.mixed.length;
  const logCount = DATA.llm_logs.length;
  const hcspCount = DATA.hcsp_tasks.length;
  const companies = {{}};
  DATA.chunks_local.forEach(c => {{
    const company = c.meta?.company_name || 'Unknown';
    companies[company] = (companies[company] || 0) + 1;
  }});
  const validation = DATA.validation_summary;

  document.getElementById('overview').innerHTML = `
    <h2 style="margin-bottom: 8px;">Dataset Overview</h2>
    <p style="color: var(--text-muted); margin-bottom: 24px; max-width: 1000px; line-height: 1.6;">This visualizer displays the complete <strong>HCSP (Hierarchical Compositional Search Problem)</strong> dataset generation pipeline. The dataset combines <strong>local enterprise documents</strong> (internal company files from DrBench tasks DR0001-DR0015) with <strong>web corpus</strong> (BrowseComp-Plus) to create multi-hop reasoning tasks that require both private and public information sources. Each HCSP task has a <em>research tree</em> structure showing how evidence from different sources must be combined to answer the question.</p>
    <div class="stats-grid">
      <div class="card"><div class="card-title">Local Chunks</div><div class="card-value">${{localCount.toLocaleString()}}</div></div>
      <div class="card"><div class="card-title">Web Chunks (sampled)</div><div class="card-value">${{webCount.toLocaleString()}}</div></div>
      <div class="card"><div class="card-title">Extracted Secrets</div><div class="card-value">${{secretCount.toLocaleString()}}</div></div>
      <div class="card"><div class="card-title">Legacy Tasks</div><div class="card-value">${{localTasks + webTasks + mixedTasks}}</div></div>
      <div class="card" style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);"><div class="card-title">HCSP Tasks</div><div class="card-value">${{hcspCount}}</div></div>
      <div class="card"><div class="card-title">LLM Log Entries</div><div class="card-value">${{logCount.toLocaleString()}}</div></div>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 24px;">
      <div class="card"><h3 style="margin-bottom: 16px;">Legacy Tasks by Mode</h3><div style="display: flex; gap: 16px; flex-wrap: wrap;"><div><span class="badge badge-local">local_only</span> ${{localTasks}}</div><div><span class="badge badge-web">web_only</span> ${{webTasks}}</div><div><span class="badge badge-mixed">mixed</span> ${{mixedTasks}}</div></div></div>
      <div class="card"><h3 style="margin-bottom: 16px;">HCSP by Linkage Type</h3><div style="display: flex; gap: 12px; flex-wrap: wrap;">${{Object.entries(validation.linkage_counts || {{}}).map(([type, count]) => `<div>${{getLinkageBadge(type)}} ${{count}}</div>`).join('')}}</div></div>
      <div class="card"><h3 style="margin-bottom: 16px;">Chunks by Company</h3><div style="display: flex; gap: 16px; flex-wrap: wrap;">${{Object.entries(companies).map(([name, count]) => `<div><strong>${{name}}</strong>: ${{count}}</div>`).join('')}}</div></div>
    </div>
    ${{hcspCount > 0 ? `<div class="card" style="margin-top: 24px;"><h3 style="margin-bottom: 16px;">HCSP Validation Summary</h3><div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;"><div><div style="font-size: 12px; color: var(--text-muted);">Pass Rate</div><div style="font-size: 24px; font-weight: 600; color: ${{validation.pass_rate > 0.8 ? '#166534' : '#dc2626'}};">${{(validation.pass_rate * 100).toFixed(1)}}%</div></div><div><div style="font-size: 12px; color: var(--text-muted);">Total Tasks</div><div style="font-size: 24px; font-weight: 600;">${{validation.total_tasks}}</div></div><div><div style="font-size: 12px; color: var(--text-muted);">Ablation (Full Info)</div><div style="font-size: 24px; font-weight: 600;">${{validation.ablation_full_pass}}</div></div><div><div style="font-size: 12px; color: var(--text-muted);">Ablation (No Info)</div><div style="font-size: 24px; font-weight: 600;">${{validation.ablation_no_info_pass}}</div></div></div></div>` : ''}}
  `;
}}

let currentChunk = null;
let filteredChunks = [];

function renderChunks() {{
  const allChunks = [...DATA.chunks_local, ...DATA.chunks_web];
  filteredChunks = allChunks;
  const tasks = [...new Set(DATA.chunks_local.map(c => c.meta?.task_id).filter(Boolean))].sort();
  const companies = [...new Set(DATA.chunks_local.map(c => c.meta?.company_name).filter(Boolean))];

  document.getElementById('chunks').innerHTML = `
    <div style="margin-bottom: 16px;">
      <h2 style="margin-bottom: 8px;">Document Chunks</h2>
      <p style="color: var(--text-muted); max-width: 800px;"><strong>Pipeline Stage:</strong> Data Preparation (chunk_local.py, chunk_web.py) → Chunks are the atomic units of text used throughout the pipeline. Local chunks come from enterprise documents (~450 words each), while web chunks are from BrowseComp-Plus. Each chunk has neighbors (similar chunks) used for multi-hop traversal.</p>
    </div>
    <div class="two-col">
      <div class="sidebar">
        <div class="sidebar-header"><input type="text" id="chunk-search" placeholder="Search chunks..."></div>
        <div class="sidebar-filters">
          <select id="chunk-source-filter"><option value="">All Sources</option><option value="local">Local</option><option value="web">Web</option></select>
          <select id="chunk-task-filter"><option value="">All Tasks</option>${{tasks.map(t => `<option value="${{t}}">${{t}}</option>`).join('')}}</select>
          <select id="chunk-company-filter"><option value="">All Companies</option>${{companies.map(c => `<option value="${{c}}">${{c}}</option>`).join('')}}</select>
        </div>
        <div class="sidebar-list" id="chunk-list"></div>
      </div>
      <div class="detail-panel" id="chunk-detail"><div class="empty-state">Select a chunk to view details</div></div>
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
  const toShow = filteredChunks.slice(0, 200);
  list.innerHTML = toShow.map((c, i) => `<div class="list-item" data-index="${{i}}" onclick="selectChunk(${{i}})"><div class="list-item-id"><span class="badge ${{getBadgeClass(c.source_type)}}">${{c.source_type}}</span> ${{escapeHtml(c.chunk_id || '')}}</div><div class="list-item-preview">${{escapeHtml(truncate(c.text, 80))}}</div></div>`).join('') + (filteredChunks.length > 200 ? `<div class="list-item" style="color: var(--text-muted);">... and ${{filteredChunks.length - 200}} more</div>` : '');
}}

function selectChunk(index) {{
  currentChunk = filteredChunks[index];
  document.querySelectorAll('#chunk-list .list-item').forEach((el, i) => {{ el.classList.toggle('active', i === index); }});
  renderChunkDetail();
}}

function renderChunkDetail() {{
  if (!currentChunk) return;
  const c = currentChunk;
  const neighborData = DATA.neighbors.find(n => n.chunk_id === c.chunk_id);
  const neighbors = neighborData?.neighbors?.slice(0, 5) || [];
  const secretData = DATA.secrets.find(s => s.chunk_id === c.chunk_id);
  const secrets = secretData?.secrets || [];
  const wordCount = (c.text || '').split(/\s+/).length;
  document.getElementById('chunk-detail').innerHTML = `
    <div class="detail-section"><h3>Metadata</h3><table class="meta-table"><tr><td>Chunk ID</td><td><code>${{escapeHtml(c.chunk_id || '')}}</code></td></tr><tr><td>Doc ID</td><td><code>${{escapeHtml(c.doc_id || '')}}</code></td></tr><tr><td>Source</td><td><span class="badge ${{getBadgeClass(c.source_type)}}">${{c.source_type}}</span></td></tr><tr><td>Word Count</td><td>${{wordCount}} words</td></tr>${{c.meta?.task_id ? `<tr><td>Task</td><td>${{c.meta.task_id}}</td></tr>` : ''}}${{c.meta?.company_name ? `<tr><td>Company</td><td>${{c.meta.company_name}}</td></tr>` : ''}}${{c.meta?.file_title ? `<tr><td>File Title</td><td>${{c.meta.file_title}}</td></tr>` : ''}}</table></div>
    <div class="detail-section"><h3>Full Text Content</h3><div class="detail-text" style="max-height: none;">${{escapeHtml(c.text || '')}}</div></div>
    ${{neighbors.length > 0 ? `<div class="detail-section"><h3>BM25 Neighbors (Top 5)</h3><p style="font-size: 12px; color: var(--text-muted); margin-bottom: 8px;">Similar chunks found via BM25 retrieval - used for multi-hop traversal in task generation.</p><div class="neighbor-list">${{neighbors.map(n => `<div class="neighbor-item" onclick="jumpToChunk('${{n.chunk_id}}')"><code style="flex:1; font-size: 11px;">${{escapeHtml(n.chunk_id)}}</code><span class="neighbor-score">Score: ${{n.score?.toFixed(2) || 'N/A'}}</span></div>`).join('')}}</div></div>` : ''}}
    ${{secrets.length > 0 ? `<div class="detail-section"><h3>Extracted Secrets (${{secrets.length}})</h3><p style="font-size: 12px; color: var(--text-muted); margin-bottom: 8px;">Privacy-sensitive facts extracted by LLM (privacy_tagger.py stage).</p>${{secrets.map(s => `<div class="secret-item"><div><strong>Q:</strong> ${{escapeHtml(s.question)}}</div><div><strong>A:</strong> ${{escapeHtml(s.answer)}}</div><div style="margin-top: 4px;"><span class="badge ${{getBadgeClass(s.secret_type)}}">${{s.secret_type}}</span></div>${{s.justification ? `<div style="margin-top: 4px; font-size: 12px; color: var(--text-muted);"><em>Justification: ${{escapeHtml(s.justification)}}</em></div>` : ''}}</div>`).join('')}}</div>` : ''}}
  `;
}}

function jumpToChunk(chunkId) {{
  const allChunks = [...DATA.chunks_local, ...DATA.chunks_web];
  const index = allChunks.findIndex(c => c.chunk_id === chunkId);
  if (index >= 0) {{ document.getElementById('chunk-search').value = ''; document.getElementById('chunk-source-filter').value = ''; document.getElementById('chunk-task-filter').value = ''; document.getElementById('chunk-company-filter').value = ''; filterChunks(); selectChunk(index); }}
}}

function renderTrees() {{
  const allTasks = [...DATA.local_only, ...DATA.web_only, ...DATA.mixed];
  const hcspTaskIds = [...new Set(DATA.hcsp_tasks.map(t => t._task_id))].sort();
  document.getElementById('trees').innerHTML = `
    <div style="margin-bottom: 16px;">
      <h2 style="margin-bottom: 8px;">Research Trees</h2>
      <p style="color: var(--text-muted); max-width: 900px;"><strong>Pipeline Stage:</strong> Tree Building → Each task has a research tree showing how evidence chunks connect. <strong>Legacy Trees</strong> show simple hop sequences, while <strong>HCSP Research Trees</strong> show hierarchical structures with a question root node and evidence vertices. Click nodes in the visualization to see chunk details.</p>
    </div>
    <div class="sub-tabs"><button class="sub-tab active" data-subtab="legacy-trees">Legacy Trees</button><button class="sub-tab" data-subtab="hcsp-trees">HCSP Research Trees</button></div>
    <div id="legacy-trees" class="sub-view active">
      <div style="display: flex; flex-direction: column; gap: 16px; height: calc(100vh - 200px);">
        <div class="card" style="flex-shrink: 0;"><div style="display: flex; gap: 16px; align-items: flex-start;"><select id="tree-task-select" style="min-width: 300px; padding: 10px; border: 1px solid var(--border); border-radius: 8px;"><option value="">Select a legacy task to visualize...</option>${{allTasks.map((t, i) => `<option value="${{i}}">[#${{i+1}}] ${{t.mode}}</option>`).join('')}}</select><div id="tree-question-display" style="flex: 1; font-size: 14px; color: var(--text-muted);">Select a task to see details...</div></div></div>
        <div style="display: grid; grid-template-columns: 1fr 450px; gap: 16px; flex: 1; min-height: 0;"><div style="display: flex; flex-direction: column; gap: 16px; overflow-y: auto;"><div class="card" id="tree-task-info" style="display: none;"></div><div class="card tree-svg-container" style="flex: 1; min-height: 200px;"><svg id="tree-svg" class="tree-svg"></svg></div></div><div class="detail-panel" id="tree-detail" style="overflow-y: auto;"><div class="empty-state">Click a node to view chunk details</div></div></div>
      </div>
    </div>
    <div id="hcsp-trees" class="sub-view">
      <div style="display: flex; flex-direction: column; gap: 16px; height: calc(100vh - 200px);">
        <div class="card" style="flex-shrink: 0;"><div style="display: flex; gap: 16px; align-items: center;"><select id="hcsp-tree-task-select" style="min-width: 150px; padding: 10px; border: 1px solid var(--border); border-radius: 8px;"><option value="">Select Task ID...</option>${{hcspTaskIds.map(id => `<option value="${{id}}">${{id}}</option>`).join('')}}</select><select id="hcsp-tree-idx-select" style="min-width: 150px; padding: 10px; border: 1px solid var(--border); border-radius: 8px;"><option value="">Select task...</option></select><div id="hcsp-tree-question" style="flex: 1; font-size: 14px; color: var(--text-muted);">Select a task to visualize the research tree...</div></div></div>
        <div style="display: grid; grid-template-columns: 1fr 450px; gap: 16px; flex: 1; min-height: 0;"><div style="display: flex; flex-direction: column; gap: 16px; overflow-y: auto;"><div class="card" id="hcsp-tree-info" style="display: none;"></div><div class="card tree-svg-container" style="flex: 1; min-height: 300px;"><svg id="hcsp-tree-svg" class="tree-svg"></svg></div></div><div class="detail-panel" id="hcsp-tree-detail" style="overflow-y: auto;"><div class="empty-state">Click a node to view details</div></div></div>
      </div>
    </div>
  `;
  document.querySelectorAll('#trees .sub-tab').forEach(tab => {{ tab.addEventListener('click', () => {{ document.querySelectorAll('#trees .sub-tab').forEach(t => t.classList.remove('active')); document.querySelectorAll('#trees .sub-view').forEach(v => v.classList.remove('active')); tab.classList.add('active'); document.getElementById(tab.dataset.subtab).classList.add('active'); }}); }});
  document.getElementById('tree-task-select').addEventListener('change', (e) => {{ if (e.target.value) {{ const task = allTasks[parseInt(e.target.value)]; renderTreeTaskInfo(task); renderTree(task); }} }});
  document.getElementById('hcsp-tree-task-select').addEventListener('change', (e) => {{ const taskId = e.target.value; const idxSelect = document.getElementById('hcsp-tree-idx-select'); idxSelect.innerHTML = '<option value="">Select task...</option>'; if (taskId) {{ const tasks = DATA.hcsp_tasks.filter(t => t._task_id === taskId); tasks.forEach((t, i) => {{ const linkage = t.gold?.linkage_type || 'unknown'; idxSelect.innerHTML += `<option value="${{i}}">[#${{i+1}}] ${{linkage}} - ${{truncate(t.question, 40)}}</option>`; }}); }} }});
  document.getElementById('hcsp-tree-idx-select').addEventListener('change', (e) => {{ const taskId = document.getElementById('hcsp-tree-task-select').value; const idx = parseInt(e.target.value); if (taskId && !isNaN(idx)) {{ const tasks = DATA.hcsp_tasks.filter(t => t._task_id === taskId); const task = tasks[idx]; if (task) {{ renderHCSPTreeInfo(task); renderHCSPTree(task); }} }} }});
}}

function renderTreeTaskInfo(task) {{
  document.getElementById('tree-question-display').innerHTML = `<strong>Question:</strong> ${{escapeHtml(task.question)}}`;
  let infoHtml = `<div style="display: flex; gap: 8px; margin-bottom: 16px;"><span class="badge ${{getBadgeClass(task.mode)}}">${{task.mode}}</span><span class="badge badge-other">${{task.answer_type || 'unknown'}}</span></div><div style="background: #dcfce7; padding: 12px; border-radius: 8px; margin-bottom: 16px;"><strong>Answer:</strong> ${{escapeHtml(task.answer)}}</div>`;
  if (task.privacy?.required_secrets?.length > 0) {{ infoHtml += `<h4 style="margin: 16px 0 8px; color: var(--text-muted);">Required Secrets</h4>`; task.privacy.required_secrets.forEach(s => {{ infoHtml += `<div style="background: #fef2f2; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; font-size: 13px;"><span class="badge ${{getBadgeClass(s.secret_type)}}">${{s.secret_type}}</span> <strong>${{escapeHtml(s.answer)}}</strong> - ${{escapeHtml(truncate(s.question, 60))}}</div>`; }}); }}
  document.getElementById('tree-task-info').style.display = 'block';
  document.getElementById('tree-task-info').innerHTML = infoHtml;
}}

function renderTree(task) {{
  const svg = d3.select('#tree-svg'); svg.selectAll('*').remove();
  if (!task?.tree?.hops) return;
  const width = svg.node().getBoundingClientRect().width; const height = svg.node().getBoundingClientRect().height; const margin = {{ top: 40, right: 40, bottom: 40, left: 40 }};
  const hops = task.tree.hops; const targetHop = task.tree.target_hop;
  const nodeWidth = (width - margin.left - margin.right) / hops.length;
  const nodes = hops.map((hop, i) => ({{ ...hop, x: margin.left + nodeWidth * i + nodeWidth / 2, y: height / 2, isTarget: hop.hop_id === targetHop }}));
  const g = svg.append('g');
  g.selectAll('.tree-link').data(nodes.slice(1)).enter().append('line').attr('class', 'tree-link').attr('x1', (d, i) => nodes[i].x).attr('y1', (d, i) => nodes[i].y).attr('x2', d => d.x).attr('y2', d => d.y);
  const node = g.selectAll('.tree-node').data(nodes).enter().append('g').attr('class', d => `tree-node ${{d.source_type}} ${{d.isTarget ? 'target' : ''}}`).attr('transform', d => `translate(${{d.x}},${{d.y}})`).on('click', (e, d) => showNodeDetail(d, task));
  node.append('circle').attr('r', 25);
  node.append('text').attr('class', 'tree-label').attr('text-anchor', 'middle').attr('dy', 5).text(d => `Hop ${{d.hop_id}}`);
  node.append('text').attr('class', 'tree-label').attr('text-anchor', 'middle').attr('dy', 45).text(d => d.source_type);
}}

function showNodeDetail(hop, task) {{
  const allChunks = [...DATA.chunks_local, ...DATA.chunks_web]; const chunk = allChunks.find(c => c.chunk_id === hop.chunk_id);
  let html = `<div class="detail-section"><h3>Hop ${{hop.hop_id}} ${{hop.isTarget ? '(Target)' : ''}}</h3><table class="meta-table"><tr><td>Chunk ID</td><td><code style="font-size: 11px;">${{escapeHtml(hop.chunk_id)}}</code></td></tr><tr><td>Doc ID</td><td><code style="font-size: 11px;">${{escapeHtml(hop.doc_id || '')}}</code></td></tr><tr><td>Source</td><td><span class="badge ${{getBadgeClass(hop.source_type)}}">${{hop.source_type}}</span></td></tr></table></div>`;
  if (hop.edge) {{ html += `<div class="detail-section"><h3>Retrieval Edge</h3><div style="background: #fef3c7; padding: 12px; border-radius: 8px;"><div style="font-size: 12px; color: var(--text-muted);">Query used:</div><div style="font-style: italic; margin-top: 4px;">"${{escapeHtml(hop.edge.query || '')}}"</div></div></div>`; }}
  if (chunk) {{ html += `<div class="detail-section"><h3>Chunk Text</h3><div class="detail-text">${{escapeHtml(chunk.text || '')}}</div></div>`; }}
  document.getElementById('tree-detail').innerHTML = html;
}}

function renderHCSPTreeInfo(task) {{
  document.getElementById('hcsp-tree-question').innerHTML = `<strong>Q:</strong> ${{escapeHtml(truncate(task.question, 100))}}`;
  const diversity = task.diversity || {{}}; const quality = task.quality || {{}}; const linkage = task.gold?.linkage_type;
  let html = `<div style="display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap;"><span class="badge ${{getBadgeClass(task.mode)}}">${{task.mode}}</span>${{getLinkageBadge(linkage)}}<span class="badge badge-other">${{task.answer_type || 'unknown'}}</span>${{quality.deterministic_pass ? '<span class="badge" style="background: #dcfce7; color: #166534;">PASS</span>' : '<span class="badge" style="background: #fef2f2; color: #dc2626;">FAIL</span>'}}</div><div style="background: #dcfce7; padding: 12px; border-radius: 8px; margin-bottom: 16px;"><strong>Answer:</strong> ${{escapeHtml(task.answer)}}</div><div style="margin-bottom: 16px;"><div style="font-size: 12px; color: var(--text-muted); margin-bottom: 4px;">Hop Pattern</div><div class="hop-pattern">${{renderHopPattern(diversity.hop_pattern)}}</div></div>`;
  if (task.privacy?.required_secrets?.length > 0) {{ html += `<h4 style="margin: 16px 0 8px; color: var(--text-muted);">Required Secrets (${{task.privacy.required_secrets.length}})</h4>`; task.privacy.required_secrets.slice(0, 3).forEach(s => {{ html += `<div style="background: #fef2f2; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; font-size: 13px;"><span class="badge ${{getBadgeClass(s.secret_type)}}">${{s.secret_type}}</span> ${{s.is_intermediate ? '<span style="font-size: 10px; color: #6b7280;">(intermediate)</span>' : ''}} <strong>${{escapeHtml(truncate(s.answer, 30))}}</strong></div>`; }}); }}
  document.getElementById('hcsp-tree-info').style.display = 'block'; document.getElementById('hcsp-tree-info').innerHTML = html;
}}

function renderHCSPTree(task) {{
  const svg = d3.select('#hcsp-tree-svg'); svg.selectAll('*').remove();
  const hops = task.tree?.hops || []; if (hops.length === 0) return;
  const width = svg.node().getBoundingClientRect().width || 600; const height = svg.node().getBoundingClientRect().height || 400; const margin = {{ top: 60, right: 40, bottom: 40, left: 40 }};
  const treeData = {{ id: 'root', kind: 'question', label: 'Question', children: hops.map((hop, i) => {{ const hcspNodes = task.tree?.hcsp?.nodes || {{}}; const constraintCount = Object.values(hcspNodes).filter(n => n.kind === 'constraint' && n.id.startsWith(`${{hop.source_type === 'local' ? 'L' : 'W'}}${{i+1}}_`)).length; return {{ ...hop, id: `hop_${{i}}`, kind: 'evidence', label: `${{hop.source_type === 'local' ? 'L' : 'W'}}${{i+1}}`, constraintCount }}; }}) }};
  const nodeWidth = (width - margin.left - margin.right) / (hops.length + 1); const g = svg.append('g');
  const rootX = width / 2; const rootY = margin.top;
  g.append('rect').attr('class', 'tree-node question').attr('x', rootX - 50).attr('y', rootY - 20).attr('width', 100).attr('height', 40).attr('rx', 6).style('fill', '#fef3c7').style('stroke', '#f59e0b').style('stroke-width', 2).style('cursor', 'pointer').on('click', () => showHCSPNodeDetail(treeData, task));
  g.append('text').attr('x', rootX).attr('y', rootY + 5).attr('text-anchor', 'middle').style('font-size', '12px').style('font-weight', '500').text('Question');
  const evidenceY = height / 2;
  treeData.children.forEach((node, i) => {{
    const x = margin.left + nodeWidth * i + nodeWidth / 2;
    g.append('line').attr('class', 'tree-link').attr('x1', rootX).attr('y1', rootY + 20).attr('x2', x).attr('y2', evidenceY - 25);
    const nodeG = g.append('g').attr('class', `tree-node ${{node.source_type}}`).attr('transform', `translate(${{x}},${{evidenceY}})`).style('cursor', 'pointer').on('click', () => showHCSPNodeDetail(node, task));
    nodeG.append('circle').attr('r', 25);
    nodeG.append('text').attr('class', 'tree-label').attr('text-anchor', 'middle').attr('dy', 5).text(node.label);
    if (node.constraintCount > 0) {{ nodeG.append('circle').attr('cx', 18).attr('cy', -18).attr('r', 10).style('fill', '#f59e0b'); nodeG.append('text').attr('x', 18).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '10px').style('fill', 'white').text(node.constraintCount); }}
    nodeG.append('text').attr('class', 'tree-label').attr('text-anchor', 'middle').attr('dy', 45).text(node.source_type);
  }});
}}

function showHCSPNodeDetail(node, task) {{
  let html = '';
  if (node.kind === 'question') {{ html = `<div class="detail-section"><h3>Question (Root)</h3><div class="detail-text">${{escapeHtml(task.question)}}</div></div><div class="detail-section"><h3>Answer</h3><div style="background: #dcfce7; padding: 12px; border-radius: 8px;"><strong>${{escapeHtml(task.answer)}}</strong></div></div>`; }}
  else {{
    const allChunks = [...DATA.chunks_local, ...DATA.chunks_web]; const chunk = allChunks.find(c => c.chunk_id === node.chunk_id);
    html = `<div class="detail-section"><h3>Evidence: ${{node.label}}</h3><table class="meta-table"><tr><td>Chunk ID</td><td><code style="font-size: 11px;">${{escapeHtml(node.chunk_id)}}</code></td></tr><tr><td>Source</td><td><span class="badge ${{getBadgeClass(node.source_type)}}">${{node.source_type}}</span></td></tr></table></div>`;
    const hcspNodes = task.tree?.hcsp?.nodes || {{}}; const constraints = Object.values(hcspNodes).filter(n => n.kind === 'constraint' && n.id.startsWith(node.label + '_'));
    if (constraints.length > 0) {{ html += `<div class="detail-section"><h3>Constraints (${{constraints.length}})</h3><div class="hcsp-constraints-list">`; constraints.forEach(c => {{ const constraint = c.constraint || {{}}; html += `<div class="hcsp-constraint-item ${{constraint.corpus}}"><div style="margin-bottom: 4px;"><span class="badge ${{getBadgeClass(constraint.constraint_type)}}">${{constraint.constraint_type}}</span> <span class="badge ${{getBadgeClass(constraint.corpus)}}">${{constraint.corpus}}</span></div><div style="font-size: 13px;">${{escapeHtml(constraint.text)}}</div></div>`; }}); html += `</div></div>`; }}
    if (chunk) {{ html += `<div class="detail-section"><h3>Chunk Text</h3><div class="detail-text" style="max-height: 200px;">${{escapeHtml(chunk.text || '')}}</div></div>`; }}
    if (node.edge?.query) {{ html += `<div class="detail-section"><h3>Bridge Query</h3><div style="background: #fef3c7; padding: 12px; border-radius: 8px; font-style: italic;">"${{escapeHtml(node.edge.query)}}"</div></div>`; }}
  }}
  document.getElementById('hcsp-tree-detail').innerHTML = html;
}}

function renderSecrets() {{
  const allSecrets = []; DATA.secrets.forEach(s => {{ (s.secrets || []).forEach(secret => {{ allSecrets.push({{ ...secret, chunk_id: s.chunk_id, doc_id: s.doc_id }}); }}); }});
  const types = [...new Set(allSecrets.map(s => s.secret_type))].sort();
  document.getElementById('secrets').innerHTML = `
    <div style="margin-bottom: 16px;">
      <h2 style="margin-bottom: 8px;">Extracted Secrets</h2>
      <p style="color: var(--text-muted); max-width: 900px;"><strong>Pipeline Stage:</strong> privacy_tagger.py → Secrets are privacy-sensitive facts extracted from local enterprise documents using an LLM. Each secret is a Q&A pair representing information that should NOT be leaked to external services. Types include: <span class="badge badge-kpi">kpi_numeric</span> (metrics), <span class="badge badge-money">money</span> (financial), <span class="badge badge-names">names</span> (people), <span class="badge badge-emails">emails</span>, <span class="badge badge-dates">dates</span>, <span class="badge badge-ids">ids</span> (identifiers).</p>
    </div>
    <div class="filters-row"><input type="text" id="secret-search" placeholder="Search secrets..."><select id="secret-type-filter"><option value="">All Types</option>${{types.map(t => `<option value="${{t}}">${{t}}</option>`).join('')}}</select></div><div class="card"><table class="secrets-table"><thead><tr><th style="width: 200px;">Chunk ID</th><th>Question</th><th>Answer</th><th>Type</th></tr></thead><tbody id="secrets-tbody"></tbody></table></div>`;
  window.allSecrets = allSecrets; document.getElementById('secret-search').addEventListener('input', filterSecrets); document.getElementById('secret-type-filter').addEventListener('change', filterSecrets); filterSecrets();
}}

function filterSecrets() {{
  const search = document.getElementById('secret-search').value.toLowerCase(); const type = document.getElementById('secret-type-filter').value;
  const filtered = window.allSecrets.filter(s => {{ if (type && s.secret_type !== type) return false; if (search && !s.question?.toLowerCase().includes(search) && !s.answer?.toLowerCase().includes(search)) return false; return true; }});
  document.getElementById('secrets-tbody').innerHTML = filtered.slice(0, 200).map(s => `<tr onclick="showSecretDetail(this, '${{escapeHtml(JSON.stringify(s).replace(/'/g, "\\\\'"))}}')"><td><code style="font-size: 11px; word-break: break-all;">${{escapeHtml(s.chunk_id)}}</code></td><td style="max-width: 400px;">${{escapeHtml(s.question)}}</td><td><strong>${{escapeHtml(s.answer)}}</strong></td><td><span class="badge ${{getBadgeClass(s.secret_type)}}">${{s.secret_type}}</span></td></tr>`).join('');
}}

function showSecretDetail(row, secretJson) {{
  const s = JSON.parse(secretJson.replace(/\\\\'/g, "'")); openModal('Secret Details', `<table class="meta-table"><tr><td>Chunk ID</td><td><code>${{escapeHtml(s.chunk_id)}}</code></td></tr><tr><td>Question</td><td>${{escapeHtml(s.question)}}</td></tr><tr><td>Answer</td><td><strong>${{escapeHtml(s.answer)}}</strong></td></tr><tr><td>Type</td><td><span class="badge ${{getBadgeClass(s.secret_type)}}">${{s.secret_type}}</span></td></tr><tr><td>Justification</td><td>${{escapeHtml(s.justification || 'N/A')}}</td></tr></table>`);
}}

function renderTasks() {{
  const allTasks = [...DATA.local_only, ...DATA.web_only, ...DATA.mixed];
  document.getElementById('tasks').innerHTML = `
    <div style="margin-bottom: 16px;">
      <h2 style="margin-bottom: 8px;">Q&A Tasks (Legacy)</h2>
      <p style="color: var(--text-muted); max-width: 900px;"><strong>Pipeline Stage:</strong> Legacy task generation (local_multihop_dataset.py, web_only_dataset.py, mixed_dataset_poc.py) → These are older-format tasks created before the HCSP architecture. They have simpler structures with direct hop sequences. <span class="badge badge-local">local_only</span> uses only enterprise docs, <span class="badge badge-web">web_only</span> uses BrowseComp-Plus, and <span class="badge badge-mixed">mixed</span> combines both.</p>
    </div>
    <div class="filters-row"><input type="text" id="task-search" placeholder="Search questions..."><select id="task-mode-filter"><option value="">All Modes</option><option value="local_only">local_only</option><option value="web_only">web_only</option><option value="mixed">mixed</option></select></div><div class="task-grid" id="task-grid"></div>`;
  window.allTasks = allTasks; document.getElementById('task-search').addEventListener('input', filterTasks); document.getElementById('task-mode-filter').addEventListener('change', filterTasks); filterTasks();
}}

function filterTasks() {{
  const search = document.getElementById('task-search').value.toLowerCase(); const mode = document.getElementById('task-mode-filter').value;
  const filtered = window.allTasks.filter(t => {{ if (mode && t.mode !== mode) return false; if (search && !t.question?.toLowerCase().includes(search)) return false; return true; }});
  document.getElementById('task-grid').innerHTML = filtered.slice(0, 50).map(t => `<div class="task-card"><div class="task-header"><span class="badge ${{getBadgeClass(t.mode)}}">${{t.mode}}</span><span class="badge badge-other">${{t.answer_type || 'unknown'}}</span></div><div class="task-question">${{escapeHtml(t.question)}}</div><div class="task-answer">${{escapeHtml(t.answer)}}</div><div class="task-section"><div class="task-section-title">Tree (${{t.tree?.hops?.length || 0}} hops)</div><div style="font-size: 12px; color: var(--text-muted);">${{(t.tree?.hops || []).map(h => `<span class="badge ${{getBadgeClass(h.source_type)}}">${{h.hop_id}}</span>`).join(' → ')}}</div></div></div>`).join('');
}}

function renderHCSPTasks() {{
  const tasks = DATA.hcsp_tasks; const taskIds = [...new Set(tasks.map(t => t._task_id))].sort(); const linkageTypes = [...new Set(tasks.map(t => t.gold?.linkage_type).filter(Boolean))];
  document.getElementById('hcsp-tasks').innerHTML = `
    <div style="margin-bottom: 16px;">
      <h2 style="margin-bottom: 8px;">HCSP Tasks</h2>
      <p style="color: var(--text-muted); max-width: 1000px; line-height: 1.6;"><strong>Pipeline Stage:</strong> HCSP Task Generation (hcsp_dataset.py) → The main output of the pipeline. Each HCSP task has a hierarchical research tree structure with constraints extracted from evidence. <strong>Click any task card</strong> to see the step-by-step creation process, including document selection, bridge queries, constraint extraction, and validation results.</p>
      <div style="margin-top: 12px; display: flex; gap: 16px; flex-wrap: wrap;">
        <div style="font-size: 12px;"><strong>Linkage Types:</strong> ${{Object.keys(LINKAGE_INFO).map(type => `${{LINKAGE_INFO[type].emoji}} ${{type}}`).join(' | ')}}</div>
      </div>
    </div>
    <div class="filters-row"><input type="text" id="hcsp-task-search" placeholder="Search questions..."><select id="hcsp-task-id-filter"><option value="">All Task IDs</option>${{taskIds.map(id => `<option value="${{id}}">${{id}}</option>`).join('')}}</select><select id="hcsp-linkage-filter"><option value="">All Linkage Types</option>${{linkageTypes.map(t => `<option value="${{t}}">${{t}}</option>`).join('')}}</select><select id="hcsp-mode-filter"><option value="">All Modes</option><option value="local_only">local_only</option><option value="mixed">mixed</option></select></div><div class="task-grid" id="hcsp-task-grid"></div>`;
  document.getElementById('hcsp-task-search').addEventListener('input', filterHCSPTasks); document.getElementById('hcsp-task-id-filter').addEventListener('change', filterHCSPTasks); document.getElementById('hcsp-linkage-filter').addEventListener('change', filterHCSPTasks); document.getElementById('hcsp-mode-filter').addEventListener('change', filterHCSPTasks); filterHCSPTasks();
}}

function filterHCSPTasks() {{
  const search = document.getElementById('hcsp-task-search').value.toLowerCase(); const taskId = document.getElementById('hcsp-task-id-filter').value; const linkage = document.getElementById('hcsp-linkage-filter').value; const mode = document.getElementById('hcsp-mode-filter').value;
  const filtered = DATA.hcsp_tasks.filter(t => {{ if (taskId && t._task_id !== taskId) return false; if (linkage && t.gold?.linkage_type !== linkage) return false; if (mode && t.mode !== mode) return false; if (search && !t.question?.toLowerCase().includes(search)) return false; return true; }});
  document.getElementById('hcsp-task-grid').innerHTML = filtered.slice(0, 50).map((t, i) => {{ const diversity = t.diversity || {{}}; const quality = t.quality || {{}}; const linkageType = t.gold?.linkage_type; return `<div class="task-card" onclick="showHCSPTaskDetail(${{DATA.hcsp_tasks.indexOf(t)}})"><div class="task-header"><span class="badge badge-other">${{t._task_id}}</span><span class="badge ${{getBadgeClass(t.mode)}}">${{t.mode}}</span>${{getLinkageBadge(linkageType)}}${{quality.deterministic_pass ? '<span class="badge" style="background: #dcfce7; color: #166534;">PASS</span>' : '<span class="badge" style="background: #fef2f2; color: #dc2626;">FAIL</span>'}}</div><div class="task-question">${{escapeHtml(truncate(t.question, 150))}}</div><div class="task-answer">${{escapeHtml(truncate(t.answer, 50))}}</div><div class="task-section"><div class="task-section-title">Hop Pattern</div><div class="hop-pattern">${{renderHopPattern(diversity.hop_pattern)}}</div></div><div class="task-section"><div class="task-section-title">Constraints: ${{diversity.local_constraints || 0}} local + ${{diversity.web_constraints || 0}} web</div></div>${{t.privacy?.required_secrets?.length ? `<div class="task-section"><div class="task-section-title">Required Secrets (${{t.privacy.required_secrets.length}})</div>${{t.privacy.required_secrets.slice(0, 2).map(s => `<div class="secret-item"><span class="badge ${{getBadgeClass(s.secret_type)}}">${{s.secret_type}}</span> ${{s.is_intermediate ? '<span style="font-size: 10px; color: #6b7280;">(intermediate)</span>' : ''}} ${{escapeHtml(truncate(s.answer, 30))}}</div>`).join('')}}</div>` : ''}}</div>`; }}).join('');
}}

function showHCSPTaskDetail(index) {{
  const t = DATA.hcsp_tasks[index]; const diversity = t.diversity || {{}}; const quality = t.quality || {{}}; const linkageType = t.gold?.linkage_type;
  const hops = t.tree?.hops || [];
  const allChunks = [...DATA.chunks_local, ...DATA.chunks_web];

  // Build step-by-step creation process
  let creationSteps = `<div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 20px; border-radius: 12px; margin-bottom: 24px;">
    <h3 style="margin-bottom: 16px; color: #0369a1;">📋 Task Creation Process (Step-by-Step)</h3>
    <div style="display: flex; flex-direction: column; gap: 16px;">`;

  // Step 1: Select starting local document with secret
  const startChunk = allChunks.find(c => c.chunk_id === hops[0]?.chunk_id);
  const sourceSecret = t.gold?.source_secret;
  creationSteps += `<div style="display: flex; gap: 12px; align-items: flex-start;">
    <div style="width: 28px; height: 28px; background: #0369a1; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; flex-shrink: 0;">1</div>
    <div style="flex: 1;">
      <div style="font-weight: 600; margin-bottom: 4px;">Select Starting Document with Secret</div>
      <div style="font-size: 13px; color: var(--text-muted);">Found a local document containing a privacy-sensitive fact.</div>
      <div style="background: white; padding: 10px; border-radius: 6px; margin-top: 8px; font-size: 12px;">
        <div><strong>Chunk:</strong> <code>${{hops[0]?.chunk_id || 'N/A'}}</code></div>
        ${{sourceSecret ? `<div style="margin-top: 4px;"><strong>Secret Q:</strong> ${{escapeHtml(sourceSecret.question)}}</div><div><strong>Secret A:</strong> <span style="color: #dc2626;">${{escapeHtml(sourceSecret.answer)}}</span></div>` : ''}}
      </div>
    </div>
  </div>`;

  // Step 2-N: Build multi-hop research path
  hops.slice(1).forEach((hop, i) => {{
    const hopChunk = allChunks.find(c => c.chunk_id === hop.chunk_id);
    const stepNum = i + 2;
    const isWeb = hop.source_type === 'web';
    creationSteps += `<div style="display: flex; gap: 12px; align-items: flex-start;">
      <div style="width: 28px; height: 28px; background: ${{isWeb ? '#7c3aed' : '#0369a1'}}; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; flex-shrink: 0;">${{stepNum}}</div>
      <div style="flex: 1;">
        <div style="font-weight: 600; margin-bottom: 4px;">Retrieve ${{isWeb ? 'Web' : 'Local'}} Evidence (Hop ${{hop.hop_id}})</div>
        <div style="font-size: 13px; color: var(--text-muted);">${{isWeb ? 'Searched web corpus to find related public information.' : 'Followed BM25 neighbor link to related local document.'}}</div>
        <div style="background: white; padding: 10px; border-radius: 6px; margin-top: 8px; font-size: 12px;">
          ${{hop.edge?.query ? `<div><strong>Bridge Query:</strong> <em>"${{escapeHtml(hop.edge.query)}}"</em></div>` : ''}}
          <div><strong>Retrieved:</strong> <code>${{hop.chunk_id}}</code></div>
          ${{hopChunk ? `<div style="margin-top: 4px; color: var(--text-muted);">${{escapeHtml(truncate(hopChunk.text, 150))}}</div>` : ''}}
        </div>
      </div>
    </div>`;
  }});

  // Step: Extract constraints from evidence
  const constraints = Object.values(t.tree?.hcsp?.nodes || {{}}).filter(n => n.kind === 'constraint');
  creationSteps += `<div style="display: flex; gap: 12px; align-items: flex-start;">
    <div style="width: 28px; height: 28px; background: #f59e0b; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; flex-shrink: 0;">${{hops.length + 1}}</div>
    <div style="flex: 1;">
      <div style="font-weight: 600; margin-bottom: 4px;">Extract Constraints from Evidence</div>
      <div style="font-size: 13px; color: var(--text-muted);">LLM extracts key facts as constraints that will be woven into the question.</div>
      <div style="background: white; padding: 10px; border-radius: 6px; margin-top: 8px; font-size: 12px;">
        <div><strong>Extracted:</strong> ${{constraints.length}} constraints (${{diversity.local_constraints || 0}} local + ${{diversity.web_constraints || 0}} web)</div>
      </div>
    </div>
  </div>`;

  // Step: Synthesize question
  creationSteps += `<div style="display: flex; gap: 12px; align-items: flex-start;">
    <div style="width: 28px; height: 28px; background: #10b981; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; flex-shrink: 0;">${{hops.length + 2}}</div>
    <div style="flex: 1;">
      <div style="font-weight: 600; margin-bottom: 4px;">Synthesize Question</div>
      <div style="font-size: 13px; color: var(--text-muted);">LLM weaves all constraints into a natural question that requires multi-hop reasoning.</div>
      <div style="background: white; padding: 10px; border-radius: 6px; margin-top: 8px; font-size: 12px;">
        <div><strong>Linkage Type:</strong> ${{getLinkageBadge(linkageType)}} - ${{LINKAGE_INFO[linkageType]?.desc || 'Unknown'}}</div>
      </div>
    </div>
  </div>`;

  // Step: Validate quality
  creationSteps += `<div style="display: flex; gap: 12px; align-items: flex-start;">
    <div style="width: 28px; height: 28px; background: ${{quality.deterministic_pass ? '#10b981' : '#dc2626'}}; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; flex-shrink: 0;">✓</div>
    <div style="flex: 1;">
      <div style="font-weight: 600; margin-bottom: 4px;">Validate Task Quality</div>
      <div style="font-size: 13px; color: var(--text-muted);">Run ablation tests to ensure the question requires the evidence.</div>
      <div style="background: white; padding: 10px; border-radius: 6px; margin-top: 8px; font-size: 12px;">
        <div><strong>Deterministic:</strong> ${{quality.deterministic_pass ? '✅ PASS' : '❌ FAIL'}}</div>
        <div><strong>With Full Info:</strong> ${{escapeHtml(String(quality.ablation_full_info || 'N/A'))}}</div>
        <div><strong>Without Info:</strong> ${{escapeHtml(String(quality.ablation_no_info || 'N/A'))}}</div>
      </div>
    </div>
  </div>`;

  creationSteps += `</div></div>`;

  let html = `<div style="display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap;"><span class="badge badge-other">${{t._task_id}}</span><span class="badge ${{getBadgeClass(t.mode)}}">${{t.mode}}</span>${{getLinkageBadge(linkageType)}}<span class="badge badge-other">${{t.answer_type}}</span>${{quality.deterministic_pass ? '<span class="badge" style="background: #dcfce7; color: #166534;">PASS</span>' : '<span class="badge" style="background: #fef2f2; color: #dc2626;">FAIL</span>'}}</div>`;

  html += creationSteps;

  html += `<h3 style="margin-bottom: 8px;">Final Question</h3><div class="detail-text" style="margin-bottom: 16px; max-height: none;">${{escapeHtml(t.question)}}</div><h3 style="margin-bottom: 8px;">Answer</h3><div style="background: #dcfce7; padding: 12px; border-radius: 8px; margin-bottom: 16px;"><strong>${{escapeHtml(t.answer)}}</strong></div>`;

  html += `<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"><div><h3 style="margin-bottom: 8px;">Diversity Metadata</h3><table class="meta-table"><tr><td>Hop Pattern</td><td><div class="hop-pattern">${{renderHopPattern(diversity.hop_pattern)}}</div></td></tr><tr><td>Total Hops</td><td>${{diversity.total_hops || 0}}</td></tr><tr><td>Local Constraints</td><td>${{diversity.local_constraints || 0}}</td></tr><tr><td>Web Constraints</td><td>${{diversity.web_constraints || 0}}</td></tr><tr><td>Answer Corpus</td><td><span class="badge ${{getBadgeClass(diversity.answer_corpus)}}">${{diversity.answer_corpus}}</span></td></tr><tr><td>Required Secrets</td><td>${{diversity.required_local_secrets || 0}}</td></tr></table></div><div><h3 style="margin-bottom: 8px;">Quality Validation</h3><table class="meta-table"><tr><td>Deterministic Pass</td><td class="${{quality.deterministic_pass ? 'status-pass' : 'status-fail'}}">${{quality.deterministic_pass ? 'Yes' : 'No'}}</td></tr><tr><td>Ablation (Full Info)</td><td>${{escapeHtml(String(quality.ablation_full_info || 'N/A'))}}</td></tr><tr><td>Ablation (No Info)</td><td>${{escapeHtml(String(quality.ablation_no_info || 'N/A'))}}</td></tr></table></div></div>`;
  if (t.privacy?.required_secrets?.length) {{ html += `<h3 style="margin: 24px 0 8px;">Required Secrets (${{t.privacy.required_secrets.length}})</h3><div class="hcsp-constraints-list">${{t.privacy.required_secrets.map(s => `<div class="hcsp-constraint-item local"><div style="display: flex; gap: 8px; margin-bottom: 4px;"><span class="badge ${{getBadgeClass(s.secret_type)}}">${{s.secret_type}}</span> ${{s.is_intermediate ? '<span class="badge badge-other">intermediate</span>' : '<span class="badge" style="background: #dcfce7; color: #166534;">final</span>'}}</div><div><strong>Q:</strong> ${{escapeHtml(s.question)}}</div><div><strong>A:</strong> ${{escapeHtml(s.answer)}}</div></div>`).join('')}}</div>`; }}
  html += `<h3 style="margin: 24px 0 8px;">HCSP Constraints (${{constraints.length}})</h3><div class="hcsp-constraints-list">${{constraints.map(n => {{ const c = n.constraint || {{}}; return `<div class="hcsp-constraint-item ${{c.corpus}}"><div style="display: flex; gap: 8px; margin-bottom: 4px;"><span style="font-family: monospace; font-size: 11px; color: var(--text-muted);">${{n.id}}</span> <span class="badge ${{getBadgeClass(c.constraint_type)}}">${{c.constraint_type}}</span> <span class="badge ${{getBadgeClass(c.corpus)}}">${{c.corpus}}</span></div><div style="font-size: 13px;">${{escapeHtml(c.text)}}</div>${{c.evidence?.text ? `<div style="font-size: 11px; color: var(--text-muted); margin-top: 4px; font-style: italic;">Evidence: "${{escapeHtml(truncate(c.evidence.text, 100))}}"</div>` : ''}}</div>`; }}).join('')}}</div>`;
  openModal('HCSP Task Details', html);
}}

function renderConstraints() {{
  const constraints = DATA.all_constraints; const types = [...new Set(constraints.map(c => c.constraint_type))].sort(); const corpora = [...new Set(constraints.map(c => c.corpus))]; const taskIds = [...new Set(constraints.map(c => c.task_id))].sort();
  document.getElementById('constraints').innerHTML = `
    <div style="margin-bottom: 16px;">
      <h2 style="margin-bottom: 8px;">HCSP Constraints</h2>
      <p style="color: var(--text-muted); max-width: 1000px; line-height: 1.6;"><strong>Pipeline Stage:</strong> Constraint Extraction → Constraints are atomic facts extracted from evidence chunks that get woven into the final question. Each constraint has a type: <span class="badge badge-attribute">attribute</span> (properties), <span class="badge badge-relation">relation</span> (connections), <span class="badge badge-temporal">temporal</span> (time-based), or <span class="badge badge-other">other</span>. Constraints from <span class="badge badge-local">local</span> docs represent private enterprise data, while <span class="badge badge-web">web</span> constraints are public information.</p>
    </div>
    <div class="filters-row"><input type="text" id="constraint-search" placeholder="Search constraints..."><select id="constraint-type-filter"><option value="">All Types</option>${{types.map(t => `<option value="${{t}}">${{t}}</option>`).join('')}}</select><select id="constraint-corpus-filter"><option value="">All Corpora</option>${{corpora.map(c => `<option value="${{c}}">${{c}}</option>`).join('')}}</select><select id="constraint-task-filter"><option value="">All Tasks</option>${{taskIds.map(id => `<option value="${{id}}">${{id}}</option>`).join('')}}</select></div><div class="card"><table class="secrets-table"><thead><tr><th>Task</th><th>Node ID</th><th>Constraint Text</th><th>Type</th><th>Corpus</th></tr></thead><tbody id="constraints-tbody"></tbody></table></div>`;
  document.getElementById('constraint-search').addEventListener('input', filterConstraints); document.getElementById('constraint-type-filter').addEventListener('change', filterConstraints); document.getElementById('constraint-corpus-filter').addEventListener('change', filterConstraints); document.getElementById('constraint-task-filter').addEventListener('change', filterConstraints); filterConstraints();
}}

function filterConstraints() {{
  const search = document.getElementById('constraint-search').value.toLowerCase(); const type = document.getElementById('constraint-type-filter').value; const corpus = document.getElementById('constraint-corpus-filter').value; const taskId = document.getElementById('constraint-task-filter').value;
  const filtered = DATA.all_constraints.filter(c => {{ if (type && c.constraint_type !== type) return false; if (corpus && c.corpus !== corpus) return false; if (taskId && c.task_id !== taskId) return false; if (search && !c.text?.toLowerCase().includes(search)) return false; return true; }});
  document.getElementById('constraints-tbody').innerHTML = filtered.slice(0, 200).map(c => `<tr onclick="showConstraintDetail('${{escapeHtml(JSON.stringify(c).replace(/'/g, "\\\\'"))}}')"><td><span class="badge badge-other">${{c.task_id}}</span></td><td><code style="font-size: 11px;">${{escapeHtml(c.node_id)}}</code></td><td>${{escapeHtml(truncate(c.text, 80))}}</td><td><span class="badge ${{getBadgeClass(c.constraint_type)}}">${{c.constraint_type}}</span></td><td><span class="badge ${{getBadgeClass(c.corpus)}}">${{c.corpus}}</span></td></tr>`).join('');
}}

function showConstraintDetail(constraintJson) {{
  const c = JSON.parse(constraintJson.replace(/\\\\'/g, "'")); const evidence = c.evidence || {{}};
  openModal('Constraint Details', `<div style="display: flex; gap: 8px; margin-bottom: 16px;"><span class="badge badge-other">${{c.task_id}}</span><span class="badge ${{getBadgeClass(c.constraint_type)}}">${{c.constraint_type}}</span><span class="badge ${{getBadgeClass(c.corpus)}}">${{c.corpus}}</span>${{getLinkageBadge(c.linkage_type)}}</div><h3 style="margin-bottom: 8px;">Constraint Text</h3><div class="detail-text" style="margin-bottom: 16px;">${{escapeHtml(c.text)}}</div><h3 style="margin-bottom: 8px;">Evidence Pointer</h3><table class="meta-table"><tr><td>Chunk ID</td><td><code style="font-size: 11px;">${{escapeHtml(evidence.chunk_id || '')}}</code></td></tr><tr><td>Char Range</td><td>${{evidence.char_start}} - ${{evidence.char_end}}</td></tr>${{evidence.text ? `<tr><td>Text Span</td><td style="font-style: italic;">"${{escapeHtml(truncate(evidence.text, 100))}}"</td></tr>` : ''}}</table><h3 style="margin: 24px 0 8px;">Source Question</h3><div class="detail-text">${{escapeHtml(c.question)}}</div>`);
}}

function renderValidation() {{
  const v = DATA.validation_summary;
  document.getElementById('validation').innerHTML = `
    <h2 style="margin-bottom: 8px;">HCSP Validation Summary</h2>
    <p style="color: var(--text-muted); margin-bottom: 24px; max-width: 1100px; line-height: 1.6;">
      <strong>Pipeline Stage:</strong> Quality Validation → Each HCSP task undergoes validation to ensure it requires multi-hop reasoning. We run three checks:
    </p>
    <div class="card" style="margin-bottom: 24px; background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);">
      <h3 style="margin-bottom: 16px;">📋 What the Validation Metrics Mean</h3>
      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
        <div style="background: white; padding: 16px; border-radius: 8px;">
          <div style="font-weight: 600; margin-bottom: 8px;">✅ Deterministic Pass</div>
          <div style="font-size: 13px; color: var(--text-muted);">The task has a unique, verifiable answer that can be extracted from the evidence. Tasks without clear answers are rejected.</div>
        </div>
        <div style="background: white; padding: 16px; border-radius: 8px;">
          <div style="font-weight: 600; margin-bottom: 8px;">📊 Ablation (Full Info)</div>
          <div style="font-size: 13px; color: var(--text-muted);">Given ALL evidence chunks, can the LLM answer correctly? This validates the evidence is sufficient. <strong>Expected:</strong> Should match the gold answer (or show the question is answerable).</div>
        </div>
        <div style="background: white; padding: 16px; border-radius: 8px;">
          <div style="font-weight: 600; margin-bottom: 8px;">🔒 Ablation (No Info)</div>
          <div style="font-size: 13px; color: var(--text-muted);">Can the LLM answer WITHOUT any evidence (just the question)? This validates the question requires the evidence. <strong>Expected:</strong> Should fail (NOT_ANSWERABLE) - otherwise the question is too easy.</div>
        </div>
      </div>
    </div>
    <div class="stats-grid">
      <div class="card"><div class="card-title">Total Tasks</div><div class="card-value">${{v.total_tasks}}</div></div>
      <div class="card"><div class="card-title">Passed Tasks</div><div class="card-value" style="color: #166534;">${{v.passed_tasks}}</div></div>
      <div class="card"><div class="card-title">Pass Rate</div><div class="card-value" style="color: ${{v.pass_rate > 0.8 ? '#166534' : '#dc2626'}};">${{(v.pass_rate * 100).toFixed(1)}}%</div></div>
      <div class="card"><div class="card-title">Ablation Full Info Pass</div><div class="card-value">${{v.ablation_full_pass}}</div><div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Tasks where LLM got correct answer with all evidence</div></div>
      <div class="card"><div class="card-title">Ablation No Info Pass</div><div class="card-value">${{v.ablation_no_info_pass}}</div><div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Tasks where LLM correctly said "NOT_ANSWERABLE" without evidence</div></div>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
      <div class="card">
        <h3 style="margin-bottom: 8px;">By Linkage Type</h3>
        <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 12px;">How local and web evidence connect:</p>
        <div style="display: flex; flex-direction: column; gap: 8px;">
          ${{Object.entries(v.linkage_counts || {{}}).map(([type, count]) => `<div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 6px;"><div>${{getLinkageBadge(type)}}<span style="font-size: 11px; color: var(--text-muted); margin-left: 8px;">${{LINKAGE_INFO[type]?.desc || ''}}</span></div><div style="font-weight: 600;">${{count}}</div></div>`).join('')}}
        </div>
      </div>
      <div class="card">
        <h3 style="margin-bottom: 8px;">By Mode</h3>
        <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 12px;">Source corpus distribution:</p>
        <div style="display: flex; flex-direction: column; gap: 8px;">
          ${{Object.entries(v.mode_counts || {{}}).map(([mode, count]) => `<div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: #f8f9fa; border-radius: 6px;"><div><span class="badge ${{getBadgeClass(mode)}}">${{mode}}</span><span style="font-size: 11px; color: var(--text-muted); margin-left: 8px;">${{mode === 'mixed' ? 'Both local + web evidence' : mode === 'local_only' ? 'Only local enterprise docs' : 'Only web corpus'}}</span></div><div style="font-weight: 600;">${{count}}</div></div>`).join('')}}
        </div>
      </div>
    </div>
    ${{v.failures?.length > 0 ? `<div class="card" style="margin-top: 24px;"><h3 style="margin-bottom: 8px; color: #dc2626;">Failed Tasks (${{v.failures.length}})</h3><p style="font-size: 12px; color: var(--text-muted); margin-bottom: 12px;">Tasks that did not pass deterministic validation - these are excluded from the final dataset.</p><table class="secrets-table"><thead><tr><th>Task ID</th><th>Question Preview</th><th>Reason</th></tr></thead><tbody>${{v.failures.map(f => `<tr><td><span class="badge badge-other">${{f.task_id}}</span></td><td style="font-size: 12px;">${{escapeHtml(f.question)}}</td><td><span class="badge" style="background: #fef2f2; color: #dc2626;">${{f.reason}}</span></td></tr>`).join('')}}</tbody></table></div>` : `<div class="card" style="margin-top: 24px; background: #f0fdf4;"><h3 style="margin-bottom: 8px; color: #166534;">✅ All Tasks Passed Validation</h3><p style="font-size: 13px; color: var(--text-muted);">Every generated HCSP task passed deterministic validation and is included in the dataset.</p></div>`}}
  `;
}}

function renderLogs() {{
  const logs = DATA.llm_logs; const runs = [...new Set(logs.map(l => l.run_id))].sort().reverse();
  let totalPrompt = 0, totalCompletion = 0; const byStage = {{}};
  logs.forEach(l => {{ totalPrompt += l.usage?.prompt_tokens || 0; totalCompletion += l.usage?.completion_tokens || 0; const stage = l.stage || 'unknown'; byStage[stage] = (byStage[stage] || 0) + (l.usage?.total_tokens || 0); }});
  document.getElementById('logs').innerHTML = `
    <div style="margin-bottom: 16px;">
      <h2 style="margin-bottom: 8px;">LLM Generation Logs</h2>
      <p style="color: var(--text-muted); max-width: 900px;"><strong>Pipeline Stage:</strong> All LLM calls → Every LLM API call during pipeline execution is logged here. Stages include: <strong>privacy_inventory</strong> (secret extraction), <strong>hcsp_bridge_extraction</strong> (finding cross-corpus links), <strong>hcsp_constraint_extraction</strong> (extracting facts from evidence), and <strong>hcsp_question_synthesis</strong> (generating final questions). Use this to audit LLM costs and debug generation issues.</p>
    </div>
    <div class="logs-summary"><div class="card"><div class="card-title">Total Entries</div><div class="card-value">${{logs.length.toLocaleString()}}</div></div><div class="card"><div class="card-title">Prompt Tokens</div><div class="card-value">${{totalPrompt.toLocaleString()}}</div></div><div class="card"><div class="card-title">Completion Tokens</div><div class="card-value">${{totalCompletion.toLocaleString()}}</div></div></div><div class="card" style="margin-bottom: 24px;"><h3 style="margin-bottom: 12px;">Tokens by Stage</h3><div style="display: flex; gap: 16px; flex-wrap: wrap;">${{Object.entries(byStage).map(([stage, tokens]) => `<div><strong>${{stage}}</strong>: ${{tokens.toLocaleString()}}</div>`).join('')}}</div></div><div class="filters-row"><select id="log-run-filter"><option value="">All Runs</option>${{runs.map(r => `<option value="${{r}}">${{r}}</option>`).join('')}}</select><select id="log-stage-filter"><option value="">All Stages</option>${{Object.keys(byStage).map(s => `<option value="${{s}}">${{s}}</option>`).join('')}}</select></div><div class="card"><table class="logs-table"><thead><tr><th>Timestamp</th><th>Run</th><th>Stage</th><th>Chunk ID</th><th>Prompt</th><th>Completion</th></tr></thead><tbody id="logs-tbody"></tbody></table></div>`;
  window.allLogs = logs; document.getElementById('log-run-filter').addEventListener('change', filterLogs); document.getElementById('log-stage-filter').addEventListener('change', filterLogs); filterLogs();
}}

function filterLogs() {{
  const run = document.getElementById('log-run-filter').value; const stage = document.getElementById('log-stage-filter').value;
  const filtered = window.allLogs.filter(l => {{ if (run && l.run_id !== run) return false; if (stage && l.stage !== stage) return false; return true; }});
  document.getElementById('logs-tbody').innerHTML = filtered.slice(0, 200).map((l, i) => `<tr onclick="showLogDetail(${{i}})"><td>${{l.timestamp_iso || new Date(l.timestamp * 1000).toISOString()}}</td><td><code style="font-size: 11px;">${{escapeHtml(truncate(l.run_id || '', 20))}}</code></td><td>${{escapeHtml(l.stage || '')}}</td><td><code style="font-size: 11px;">${{escapeHtml(truncate(l.chunk_id || l.task_id || '', 30))}}</code></td><td>${{l.usage?.prompt_tokens?.toLocaleString() || 0}}</td><td>${{l.usage?.completion_tokens?.toLocaleString() || 0}}</td></tr>`).join('');
  window.filteredLogs = filtered;
}}

function showLogDetail(index) {{
  const l = window.filteredLogs[index]; openModal('LLM Log Entry', `<table class="meta-table"><tr><td>Timestamp</td><td>${{l.timestamp_iso || new Date(l.timestamp * 1000).toISOString()}}</td></tr><tr><td>Run ID</td><td><code>${{escapeHtml(l.run_id || '')}}</code></td></tr><tr><td>Stage</td><td>${{escapeHtml(l.stage || '')}}</td></tr><tr><td>Model</td><td>${{escapeHtml(l.resolved_model || l.requested_model || '')}}</td></tr><tr><td>Chunk/Task ID</td><td><code>${{escapeHtml(l.chunk_id || l.task_id || '')}}</code></td></tr><tr><td>Prompt Tokens</td><td>${{l.usage?.prompt_tokens?.toLocaleString() || 0}}</td></tr><tr><td>Completion Tokens</td><td>${{l.usage?.completion_tokens?.toLocaleString() || 0}}</td></tr></table>`);
}}

function renderArchitecture() {{
  document.getElementById('architecture').innerHTML = `
    <h2 style="margin-bottom: 8px;">Pipeline Architecture</h2>
    <p style="color: var(--text-muted); margin-bottom: 24px; max-width: 1000px; line-height: 1.6;">The HCSP dataset generation pipeline transforms raw documents into multi-hop reasoning tasks. Each step is a separate Python script that reads input files and produces output files. Click on any step to see its source code location, input/output files, and LLM prompt templates (if applicable).</p>
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 16px 20px; border-radius: 12px; margin-bottom: 24px;">
      <h3 style="margin-bottom: 12px;">🎯 Pipeline Overview</h3>
      <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 13px;">
        <span style="padding: 6px 12px; background: white; border-radius: 6px;">📄 Raw Docs</span>
        <span>→</span>
        <span style="padding: 6px 12px; background: white; border-radius: 6px;">📦 Chunks</span>
        <span>→</span>
        <span style="padding: 6px 12px; background: white; border-radius: 6px;">🔗 Neighbors</span>
        <span>→</span>
        <span style="padding: 6px 12px; background: white; border-radius: 6px;">🔒 Secrets</span>
        <span>→</span>
        <span style="padding: 6px 12px; background: white; border-radius: 6px;">🌳 Research Trees</span>
        <span>→</span>
        <span style="padding: 6px 12px; background: white; border-radius: 6px;">❓ HCSP Tasks</span>
        <span>→</span>
        <span style="padding: 6px 12px; background: white; border-radius: 6px;">✅ Validation</span>
      </div>
    </div>
    <div class="pipeline-container">${{DATA.pipeline.map((step, i) => `<div class="pipeline-node" id="pipeline-${{i}}"><div class="pipeline-header" onclick="togglePipeline(${{i}})"><div class="pipeline-number">${{i + 1}}</div><div class="pipeline-info"><div class="pipeline-name">${{step.name}}</div><div class="pipeline-desc">${{step.description}}</div></div><div class="pipeline-arrow">&#9660;</div></div><div class="pipeline-body"><div class="pipeline-files"><div><h4>Source</h4><code>${{step.source}}</code></div><div><h4>Inputs</h4>${{step.inputs.map(i => `<code>${{i}}</code>`).join('')}}</div><div><h4>Outputs</h4>${{step.outputs.map(o => `<code>${{o}}</code>`).join('')}}</div></div>${{step.has_prompts ? `<div style="margin-top: 16px;"><h4 style="margin-bottom: 8px;">LLM Prompts Used</h4><p style="font-size: 12px; color: var(--text-muted); margin-bottom: 12px;">These are the prompt templates sent to the LLM during this pipeline stage:</p>${{step.prompt_keys.map(key => DATA.prompts[key] ? `<div class="prompt-box"><h4>${{key}}</h4><pre>${{escapeHtml(DATA.prompts[key])}}</pre></div>` : '').join('')}}</div>` : ''}}</div></div>`).join('')}}</div>
  `;
}}

function togglePipeline(index) {{ document.getElementById(`pipeline-${{index}}`).classList.toggle('expanded'); }}

renderOverview();
renderChunks();
renderTrees();
renderSecrets();
renderTasks();
renderHCSPTasks();
renderConstraints();
renderValidation();
renderLogs();
renderArchitecture();
</script>
</body>
</html>
'''


def main():
    parser = argparse.ArgumentParser(description="Generate HCSP dataset visualizer HTML")
    parser.add_argument(
        "--output",
        default=str(OUTPUTS_DIR / "visualizer_hcsp.html"),
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
    parser.add_argument(
        "--scopes-dir",
        default=str(SCOPES_DIR),
        help="Directory containing HCSP task outputs",
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

    # Load legacy tasks
    local_only = load_jsonl(OUTPUTS_DIR / "local_only.jsonl")
    web_only = load_jsonl(OUTPUTS_DIR / "web_only.jsonl")
    mixed = load_jsonl(OUTPUTS_DIR / "mixed.jsonl")
    print(f"  Legacy tasks: {len(local_only)} local, {len(web_only)} web, {len(mixed)} mixed")

    # Load HCSP tasks
    scopes_dir = Path(args.scopes_dir)
    hcsp_tasks_by_id = load_hcsp_tasks(scopes_dir)
    hcsp_tasks = []
    for tasks in hcsp_tasks_by_id.values():
        hcsp_tasks.extend(tasks)
    print(f"  HCSP tasks: {len(hcsp_tasks)} across {len(hcsp_tasks_by_id)} task IDs")

    # Extract constraints
    all_constraints = extract_all_constraints(hcsp_tasks_by_id)
    print(f"  Total constraints: {len(all_constraints)}")

    # Compute validation summary
    validation_summary = compute_validation_summary(hcsp_tasks_by_id)
    print(f"  Validation: {validation_summary['passed_tasks']}/{validation_summary['total_tasks']} passed")

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
        "hcsp_tasks": hcsp_tasks,
        "hcsp_tasks_by_id": hcsp_tasks_by_id,
        "all_constraints": all_constraints,
        "validation_summary": validation_summary,
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
