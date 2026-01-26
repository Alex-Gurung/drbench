#!/usr/bin/env python3
"""
Question-variant harness for privacy experiments.

Runs the full DrBench agent loop for multiple question variants,
collects search queries, and stores outputs under a run directory.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from drbench import task_loader, drbench_enterprise_space
from drbench.agents.drbench_agent.drbench_agent import DrBenchAgent
from drbench.config import RunConfig, set_run_config

WEB_ACTION = "web_search"
LOCAL_ACTION = "local_document_search"
WEB_TOOL = "internet_search"
LOCAL_TOOL = "local_document_search"


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:48] or "question"


def load_questions_file(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_variants(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []

    if isinstance(data.get("questions"), list):
        source = data["questions"]
        for idx, q in enumerate(source, 1):
            question = q.get("question") if isinstance(q, dict) else str(q)
            if not question:
                continue
            variants.append({
                "id": q.get("id") if isinstance(q, dict) else None,
                "question": question,
                "task": q.get("task") if isinstance(q, dict) else None,
            })
    elif isinstance(data.get("research_question_variants"), list):
        for q in data["research_question_variants"]:
            question = q.get("question") if isinstance(q, dict) else str(q)
            if not question:
                continue
            variants.append({
                "id": q.get("id") if isinstance(q, dict) else None,
                "question": question,
                "task": q.get("task") if isinstance(q, dict) else None,
            })
    elif isinstance(data.get("questions_tested"), list):
        for q in data["questions_tested"]:
            question = q.get("question") if isinstance(q, dict) else str(q)
            if not question:
                continue
            variants.append({
                "id": q.get("id") if isinstance(q, dict) else None,
                "question": question,
                "task": q.get("task") if isinstance(q, dict) else None,
            })
    else:
        raise ValueError("Unsupported questions file format.")

    for idx, v in enumerate(variants, 1):
        if not v.get("id"):
            v["id"] = f"q{idx}_{slugify(v['question'])}"
    return variants


def resolve_tasks(args_tasks: Optional[List[str]], data: Dict[str, Any]) -> List[str]:
    if args_tasks:
        return args_tasks
    if isinstance(data.get("tasks"), list):
        return list(data["tasks"])
    if data.get("task_id"):
        return [data["task_id"]]
    raise ValueError("No tasks specified. Use --task/--tasks or include tasks/task_id in questions file.")


def extract_queries_from_plan(plan_path: Path, query_types: str = "web") -> List[Dict[str, Any]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    include_web = query_types in ("web", "all")
    include_local = query_types in ("local", "all")

    searches: List[Dict[str, Any]] = []
    for action in plan.get("actions", []):
        action_type = action.get("type")
        actual = action.get("actual_output") or {}
        tool_name = actual.get("tool")

        is_web = include_web and (action_type == WEB_ACTION or tool_name == WEB_TOOL)
        is_local = include_local and (action_type == LOCAL_ACTION or tool_name == LOCAL_TOOL)
        if not is_web and not is_local:
            continue

        query = actual.get("query") or (action.get("parameters") or {}).get("query") or action.get("description") or ""
        if not query:
            continue

        searches.append({
            "query": query,
            "status": action.get("status", "unknown"),
            "tool": "web" if is_web else "local",
        })

    return searches


def save_searches(run_dir: Path, task_id: str, searches: List[Dict[str, Any]]) -> None:
    if not searches:
        return
    privacy_dir = run_dir / "privacy"
    privacy_dir.mkdir(exist_ok=True)
    path = privacy_dir / "web_searches.json"
    path.write_text(json.dumps({"task_id": task_id, "searches": searches}, indent=2), encoding="utf-8")


def save_tool_calls(run_dir: Path, task_id: str, plan_path: Path) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    calls = []
    for action in plan.get("actions", []):
        calls.append({
            "id": action.get("id"),
            "type": action.get("type"),
            "status": action.get("status", "unknown"),
            "parameters": action.get("parameters", {}),
            "created_in_iteration": action.get("created_in_iteration"),
            "created_from_research_step": action.get("created_from_research_step"),
            "dependencies": action.get("dependencies", []),
        })
    if not calls:
        return
    privacy_dir = run_dir / "privacy"
    privacy_dir.mkdir(exist_ok=True)
    (privacy_dir / "tool_calls.json").write_text(json.dumps({"task_id": task_id, "tool_calls": calls}, indent=2), encoding="utf-8")


def run_single_variant(
    task_id: str,
    question: str,
    question_id: str,
    base_dir: Path,
    cfg: RunConfig,
    use_enterprise: bool,
    query_types: str,
    auto_ports: bool,
    free_ports: bool,
) -> List[Dict[str, Any]]:
    run_dir = base_dir / task_id / question_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg.run_dir = run_dir
    set_run_config(cfg)

    task = task_loader.get_task_from_id(task_id=task_id)
    task_local_files = task.get_local_files_list()

    (run_dir / "question_used.json").write_text(
        json.dumps({"task_id": task_id, "question_id": question_id, "dr_question": question}, indent=2),
        encoding="utf-8",
    )

    env = None
    if use_enterprise:
        env = drbench_enterprise_space.DrBenchEnterpriseSearchSpace(
            task=task.get_path(),
            start_container=True,
            auto_ports=auto_ports,
            free_ports=free_ports,
        )

    try:
        agent = DrBenchAgent(
            model=cfg.model,
            max_iterations=cfg.max_iterations,
            concurrent_actions=cfg.concurrent_actions,
            verbose=cfg.verbose,
        )
        report = agent.generate_report(query=question, local_files=task_local_files, env=env)

        (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        session_dir = Path(agent.vector_store.storage_dir)
        for name in ["action_plan_final.json", "action_plan_initial.json", "research_plan.json"]:
            src = session_dir / name
            if src.exists():
                (run_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    finally:
        if env is not None:
            env.delete()

    plan_path = run_dir / "action_plan_final.json"
    if not plan_path.exists():
        plan_path = run_dir / "action_plan_initial.json"

    searches: List[Dict[str, Any]] = []
    if plan_path.exists():
        searches = extract_queries_from_plan(plan_path, query_types=query_types)
        if query_types in ("web", "all"):
            web_searches = [s for s in searches if s.get("tool") == "web"]
            save_searches(run_dir, task_id, web_searches)
        save_tool_calls(run_dir, task_id, plan_path)

    return searches


def _default_run_dir() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    ts = time.strftime("%Y%m%d_%H%M%S")
    return repo_root / "runs" / f"harness_{ts}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Question-variant harness")
    parser.add_argument("--task", type=str, help="Single task id")
    parser.add_argument("--tasks", nargs="*", help="Multiple task ids")
    parser.add_argument("--questions-file", type=Path, required=True, help="JSON file containing question variants")
    parser.add_argument("--query-types", choices=["web", "local", "all"], default="web")
    parser.add_argument("--mosaic", action="store_true", help="Save combined queries")
    parser.add_argument("--run-dir", type=str, help="Run directory root")
    parser.add_argument("--data-dir", type=str, help="Override DRBENCH_DATA_DIR")

    parser.add_argument("--model", type=str, required=True, help="LLM model name")
    parser.add_argument("--llm-provider", type=str, choices=["openai", "vllm", "openrouter", "azure", "together"], help="LLM provider")
    parser.add_argument("--embedding-provider", type=str, choices=["openai", "openrouter", "huggingface", "vllm"], help="Embedding provider")
    parser.add_argument("--embedding-model", type=str, help="Embedding model")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--concurrent-actions", type=int, default=3)
    parser.add_argument("--semantic-threshold", type=float, default=0.7)

    parser.add_argument("--no-web", action="store_true", help="Disable external web access")
    parser.add_argument("--no-log", action="store_true", help="Disable all logging")
    parser.add_argument("--no-log-searches", action="store_true")
    parser.add_argument("--no-log-prompts", action="store_true")
    parser.add_argument("--no-log-generations", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--enterprise", action="store_true", help="Enable enterprise container")
    parser.add_argument("--enterprise-auto-ports", action="store_true")
    parser.add_argument("--enterprise-free-ports", action="store_true")

    args = parser.parse_args()

    if args.data_dir:
        import os
        os.environ["DRBENCH_DATA_DIR"] = str(Path(args.data_dir))

    data = load_questions_file(args.questions_file)
    tasks = resolve_tasks(args.tasks or ([args.task] if args.task else None), data)
    variants = normalize_variants(data)

    base_dir = Path(args.run_dir) if args.run_dir else _default_run_dir()
    base_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig.from_cli(args)
    cfg.model = args.model
    cfg.llm_provider = args.llm_provider
    cfg.embedding_provider = args.embedding_provider
    cfg.embedding_model = args.embedding_model
    cfg.run_dir = base_dir
    set_run_config(cfg)
    (base_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    all_task_queries: Dict[str, List[Dict[str, Any]]] = {}

    for task_id in tasks:
        task_queries: List[Dict[str, Any]] = []
        for v in variants:
            if v.get("task") and v.get("task") != task_id:
                continue
            print(f"[INFO] Running {task_id} | {v['id']}")
            searches = run_single_variant(
                task_id=task_id,
                question=v["question"],
                question_id=v["id"],
                base_dir=base_dir,
                cfg=cfg,
                use_enterprise=args.enterprise,
                query_types=args.query_types,
                auto_ports=args.enterprise_auto_ports,
                free_ports=args.enterprise_free_ports,
            )
            task_queries.extend(searches)
        all_task_queries[task_id] = task_queries

        if args.mosaic and task_queries:
            mosaic_dir = base_dir / task_id / "mosaic"
            mosaic_dir.mkdir(parents=True, exist_ok=True)
            combined_path = mosaic_dir / "combined_queries.json"
            combined_path.write_text(json.dumps({"task_id": task_id, "queries": task_queries}, indent=2), encoding="utf-8")

    if args.mosaic and len(tasks) > 1:
        combined_queries: List[Dict[str, Any]] = []
        for task_id in tasks:
            combined_queries.extend(all_task_queries.get(task_id, []))
        if combined_queries:
            mosaic_dir = base_dir / "cross_task_mosaic"
            mosaic_dir.mkdir(parents=True, exist_ok=True)
            combined_path = mosaic_dir / "combined_queries.json"
            combined_path.write_text(json.dumps({"tasks": tasks, "queries": combined_queries}, indent=2), encoding="utf-8")

    print(f"[INFO] Harness output: {base_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
