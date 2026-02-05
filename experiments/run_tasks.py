#!/usr/bin/env python3
"""
Batch runner for DrBench tasks.

Usage examples:
  python experiments/run_tasks.py DR0001 DR0002 --model gpt-4o-mini --run-dir ./runs/demo
  python experiments/run_tasks.py --subset validation --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --llm-provider vllm
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

from drbench import task_loader
from drbench import drbench_enterprise_space
from drbench.agents.drbench_agent.drbench_agent import DrBenchAgent
from drbench.config import RunConfig, set_run_config
from drbench.question_sets import resolve_dr_question
from drbench.score_report import score_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DrBench tasks in batch.")
    parser.add_argument("task_ids", nargs="*", help="Task IDs (e.g., DR0001 DR0002)")
    parser.add_argument("--task", type=str, help="Single task id")
    parser.add_argument("--tasks", nargs="*", help="Multiple task ids")
    parser.add_argument("--subset", type=str, help="Subset name (e.g., validation)")

    parser.add_argument("--model", type=str, required=True, help="LLM model name")
    parser.add_argument("--llm-provider", type=str, choices=["openai", "vllm", "openrouter", "azure", "together"], help="LLM provider")
    parser.add_argument("--embedding-provider", type=str, choices=["openai", "openrouter", "huggingface", "vllm"], help="Embedding provider")
    parser.add_argument("--embedding-model", type=str, help="Embedding model name")

    parser.add_argument("--max-iterations", type=int, default=10, help="Max agent iterations")
    parser.add_argument("--concurrent-actions", type=int, default=3, help="Concurrent actions")
    parser.add_argument("--semantic-threshold", type=float, default=0.7, help="Semantic threshold")

    parser.add_argument("--run-dir", type=str, help="Output directory for runs")
    parser.add_argument("--data-dir", type=str, help="Override DRBENCH_DATA_DIR")
    parser.add_argument("--question-set", type=str, help="Question set name")
    parser.add_argument("--question-file", type=str, help="Path to question set JSON file")

    parser.add_argument("--no-web", action="store_true", help="Disable external web access")
    parser.add_argument("--no-log", action="store_true", help="Disable all logging")
    parser.add_argument("--no-log-searches", action="store_true", help="Disable search logging")
    parser.add_argument("--no-log-prompts", action="store_true", help="Disable prompt logging")
    parser.add_argument("--no-log-generations", action="store_true", help="Disable LLM generation logging")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Validate args and exit")

    parser.add_argument("--enterprise", action="store_true", help="Enable enterprise container")
    parser.add_argument("--enterprise-auto-ports", action="store_true", help="Auto-select enterprise ports")
    parser.add_argument("--enterprise-free-ports", action="store_true", help="Free ports before start")

    # BrowseComp-Plus offline web search
    parser.add_argument("--browsecomp", action="store_true", help="Use BrowseComp offline corpus instead of live web search")
    parser.add_argument("--browsecomp-index", type=str, default="/home/toolkit/BrowseComp-Plus/indexes/qwen3-embedding-4b/corpus.shard*_of_*.pkl", help="Path glob for BrowseComp index shards")
    parser.add_argument("--browsecomp-model", type=str, default="Qwen/Qwen3-Embedding-4B", help="Embedding model for BrowseComp queries")
    parser.add_argument("--browsecomp-dataset", type=str, default="Tevatron/browsecomp-plus-corpus", help="HuggingFace dataset for BrowseComp corpus")
    parser.add_argument("--browsecomp-top-k", type=int, default=5, help="Number of documents to retrieve per query")

    return parser.parse_args()


def _resolve_tasks(args: argparse.Namespace) -> List[str]:
    if args.subset:
        if args.task_ids or args.task or args.tasks:
            raise ValueError("Do not mix --subset with explicit task IDs.")
        return task_loader.get_task_ids_from_subset(args.subset)

    tasks = []
    if args.task_ids:
        tasks.extend(args.task_ids)
    if args.task:
        tasks.append(args.task)
    if args.tasks:
        tasks.extend(args.tasks)

    if not tasks:
        raise ValueError("No tasks specified. Provide task IDs or --subset.")

    # De-duplicate while preserving order
    seen = set()
    ordered = []
    for t in tasks:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def _default_run_dir(model: str) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model.split("/")[-1].replace(" ", "_")[:20]
    return repo_root / "runs" / f"batch_{model_short}_{timestamp}"


def run_single_task(
    task_id: str,
    base_dir: Path,
    cfg: RunConfig,
    question_set: str | None,
    question_file: str | None,
    enterprise: bool,
    auto_ports: bool,
    free_ports: bool,
) -> dict:
    run_dir = base_dir / task_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)

    cfg.run_dir = run_dir
    set_run_config(cfg)

    start_time = time.time()
    result = {
        "task_id": task_id,
        "status": "failed",
        "score": None,
        "duration": 0,
        "error": None,
    }

    env = None
    try:
        task = task_loader.get_task_from_id(task_id=task_id)
        task_local_files = task.get_local_files_list()

        dr_question = task.get_task_config()["dr_question"]
        dr_question, used_set = resolve_dr_question(
            task_id,
            dr_question,
            question_set=question_set,
            question_file=question_file,
        )

        if cfg.verbose:
            print(task.summary())
            print(f"[INFO] Local files: {len(task_local_files)}")
            if used_set:
                print(f"[INFO] Question set: {used_set}")
            print(f"[INFO] DR question: {dr_question}")

        question_info = {
            "task_id": task_id,
            "question_set": used_set or "default",
            "dr_question": dr_question,
        }
        if question_file:
            question_info["question_file"] = question_file
        (run_dir / "question_used.json").write_text(json.dumps(question_info, indent=2), encoding="utf-8")

        if enterprise:
            env = drbench_enterprise_space.DrBenchEnterpriseSearchSpace(
                task=task.get_path(),
                start_container=True,
                auto_ports=auto_ports,
                free_ports=free_ports,
            )

        dr_agent = DrBenchAgent(
            model=cfg.model,
            max_iterations=cfg.max_iterations,
            concurrent_actions=cfg.concurrent_actions,
            verbose=cfg.verbose,
        )

        report = dr_agent.generate_report(query=dr_question, local_files=task_local_files, env=env)

        # Copy action plans into run_dir
        try:
            session_dir = Path(dr_agent.vector_store.storage_dir)
            for plan_file in ["action_plan_final.json", "action_plan_initial.json", "research_plan.json"]:
                src = session_dir / plan_file
                if src.exists():
                    (run_dir / plan_file).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to copy action plans: {e}")

        # Extract web searches and tool calls
        plan_path = run_dir / "action_plan_final.json"
        if not plan_path.exists():
            plan_path = run_dir / "action_plan_initial.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            searches = []
            tool_calls = []
            for action in plan.get("actions", []):
                tool_calls.append({
                    "id": action.get("id"),
                    "type": action.get("type"),
                    "status": action.get("status", "unknown"),
                    "parameters": action.get("parameters", {}),
                    "created_in_iteration": action.get("created_in_iteration"),
                    "created_from_research_step": action.get("created_from_research_step"),
                    "dependencies": action.get("dependencies", []),
                })
                if action.get("type") == "web_search":
                    query = action.get("parameters", {}).get("query", "")
                    if query:
                        searches.append({"query": query, "status": action.get("status", "unknown")})

            if searches or tool_calls:
                privacy_dir = run_dir / "privacy"
                privacy_dir.mkdir(exist_ok=True)
                if searches:
                    (privacy_dir / "web_searches.json").write_text(json.dumps({"task_id": task_id, "searches": searches}, indent=2), encoding="utf-8")
                    lines = [f"{i+1}. [{s.get('status','unknown')}] {s.get('query','')}" for i, s in enumerate(searches)]
                    (privacy_dir / "web_searches.txt").write_text("\n".join(lines), encoding="utf-8")
                if tool_calls:
                    (privacy_dir / "tool_calls.json").write_text(json.dumps({"task_id": task_id, "tool_calls": tool_calls}, indent=2), encoding="utf-8")

        # Save report
        (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        # Score report
        metrics = ["insights_recall", "factuality", "distractor_recall", "report_quality"]
        score_dict = score_report(
            predicted_report=report,
            task=task,
            metrics=metrics,
            savedir=str(results_dir),
            model=cfg.model,
        )

        distractor_avoidance = 1.0 - score_dict.get("distractor_recall", 0)
        score_dict["distractor_avoidance"] = distractor_avoidance

        metric_values = [
            score_dict.get("insights_recall", 0),
            score_dict.get("factuality", 0),
            distractor_avoidance,
            score_dict.get("report_quality", 0),
        ]
        nonzero = [v for v in metric_values if v > 0]
        harmonic_mean = len(nonzero) / sum(1 / v for v in nonzero) if nonzero else 0.0
        score_dict["harmonic_mean"] = harmonic_mean

        (run_dir / "scores.json").write_text(json.dumps(score_dict, indent=2), encoding="utf-8")

        result["status"] = "success"
        result["scores"] = {
            "insights_recall": score_dict.get("insights_recall", 0),
            "factuality": score_dict.get("factuality", 0),
            "distractor_avoidance": distractor_avoidance,
            "report_quality": score_dict.get("report_quality", 0),
            "harmonic_mean": harmonic_mean,
        }
        result["score"] = harmonic_mean

    except Exception as e:
        result["error"] = str(e)
        raise
    finally:
        if env is not None:
            env.delete()

    result["duration"] = time.time() - start_time
    return result


def main() -> int:
    args = _parse_args()

    if args.data_dir:
        import os
        os.environ["DRBENCH_DATA_DIR"] = str(Path(args.data_dir))

    tasks = _resolve_tasks(args)

    if args.question_set and args.question_file:
        raise ValueError("Use only one of --question-set or --question-file.")

    if args.run_dir:
        base_dir = Path(args.run_dir)
    else:
        base_dir = _default_run_dir(args.model)

    base_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig.from_cli(args)
    cfg.model = args.model
    cfg.run_dir = base_dir
    set_run_config(cfg)

    if args.dry_run:
        print("[DRY RUN] Tasks:", " ".join(tasks))
        print("[DRY RUN] Run dir:", base_dir)
        return 0

    # Save config for reproducibility
    config_file = base_dir / "config.json"
    config_file.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    results = []
    for task_id in tasks:
        result = run_single_task(
            task_id=task_id,
            base_dir=base_dir,
            cfg=cfg,
            question_set=args.question_set,
            question_file=args.question_file,
            enterprise=args.enterprise,
            auto_ports=args.enterprise_auto_ports,
            free_ports=args.enterprise_free_ports,
        )
        results.append(result)

        results_csv = base_dir / "results.csv"
        with results_csv.open("w", encoding="utf-8") as f:
            f.write("Task,Score,Duration,Status\n")
            for r in results:
                score = r.get("score") if r.get("score") is not None else "N/A"
                f.write(f"{r['task_id']},{score},{r['duration']:.0f},{r['status']}\n")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
