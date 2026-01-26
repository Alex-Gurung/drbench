#!/usr/bin/env python3
"""
Explore DrBench evaluation questions from eval.json.

Shows the enterprise facts used for privacy evaluation:
  - Insights: private data the model SHOULD find (relevant to research question)
  - Distractors: private data the model should IGNORE (irrelevant)

Only enterprise_fact entries are shown (not external_fact, which is public web data).

Usage:
    python privacy/explore_questions.py                  # List all tasks with counts
    python privacy/explore_questions.py DR0001           # Show specific task details
    python privacy/explore_questions.py --insights-only  # Only insights
    python privacy/explore_questions.py --distractors-only

Data source: drbench/data/tasks/DR*/eval.json
"""

import argparse
import json
from pathlib import Path

from drbench.config import get_data_dir
from drbench.task_loader import get_task_from_id


def list_task_ids() -> list[str]:
    tasks_dir = get_data_dir()
    if not tasks_dir.exists():
        return []
    return sorted([p.name for p in tasks_dir.iterdir() if p.is_dir() and p.name.startswith("DR")])


def print_separator(char="=", width=80):
    print(char * width)


def print_header(text):
    print_separator()
    print(f"  {text}")
    print_separator()


def explore_task(task_id: str, show_insights=True, show_distractors=True, show_files=False):
    """Explore a single task's privacy questions."""
    task = get_task_from_id(task_id)
    task_config = task.get_task_config()
    eval_config = task.get_eval_config()

    # Header
    print_header(f"Task: {task_id}")
    print(f"\nCompany: {task_config['company_info']['name']}")
    print(f"Industry: {task_config['company_info']['industry']}")
    print(f"\nPersona: {task_config['persona']['name']} ({task_config['persona']['role']})")
    print(f"Department: {task_config['persona']['department']}")

    print(f"\n{'='*80}")
    print("DR QUESTION:")
    print(f"{'='*80}")
    print(f"\n{task_config['dr_question']}\n")

    # Separate insights and distractors
    qa_list = eval_config.get("dr_report_evaluation_qa", [])

    insights = [qa for qa in qa_list if qa.get("qa_type") == "insight"]
    distractors = [qa for qa in qa_list if qa.get("qa_type") == "distractor"]

    # Further split into internal (enterprise) vs external
    internal_insights = [qa for qa in insights if qa.get("type") == "enterprise_fact"]
    external_insights = [qa for qa in insights if qa.get("type") == "external_fact"]
    internal_distractors = [qa for qa in distractors if qa.get("type") == "enterprise_fact"]

    print(f"\nSummary: {len(internal_insights)} internal insights, {len(external_insights)} external insights, {len(internal_distractors)} distractors")

    if show_insights and internal_insights:
        print(f"\n{'='*80}")
        print(f"INTERNAL INSIGHTS ({len(internal_insights)} facts the model SHOULD find)")
        print(f"{'='*80}")

        for i, qa in enumerate(internal_insights, 1):
            print(f"\n[{qa['id']}] Insight {i}/{len(internal_insights)}")
            print(f"-" * 60)
            if qa.get("question"):
                print(f"Q: {qa['question']}")
            print(f"A: {qa['answer']}")
            if qa.get("justification"):
                print(f"Justification: {qa['justification']}")
            if show_files and qa.get("supporting_file_paths"):
                print(f"Files: {', '.join(qa['supporting_file_paths'])}")

    if show_insights and external_insights:
        print(f"\n{'='*80}")
        print(f"EXTERNAL INSIGHTS ({len(external_insights)} facts from web)")
        print(f"{'='*80}")

        for i, qa in enumerate(external_insights, 1):
            print(f"\n[{qa['id']}] External {i}/{len(external_insights)}")
            print(f"-" * 60)
            print(f"A: {qa['answer']}")
            if qa.get("supporting_urls"):
                print(f"URL: {qa['supporting_urls'][0]}")

    if show_distractors and internal_distractors:
        print(f"\n{'='*80}")
        print(f"DISTRACTORS ({len(internal_distractors)} facts the model should IGNORE)")
        print(f"{'='*80}")

        for i, qa in enumerate(internal_distractors, 1):
            print(f"\n[{qa['id']}] Distractor {i}/{len(internal_distractors)}")
            print(f"-" * 60)
            if qa.get("question"):
                print(f"Q: {qa['question']}")
            print(f"A: {qa['answer']}")
            if qa.get("justification"):
                print(f"Why irrelevant: {qa['justification']}")
            if show_files and qa.get("supporting_file_paths"):
                print(f"Files: {', '.join(qa['supporting_file_paths'])}")

    print()
    return {
        "internal_insights": len(internal_insights),
        "external_insights": len(external_insights),
        "distractors": len(internal_distractors),
    }


def main():
    parser = argparse.ArgumentParser(description="Explore DrBench privacy questions")
    parser.add_argument("task_id", nargs="?", help="Task ID (e.g., DR0001). If not provided, lists all tasks.")
    parser.add_argument("--insights-only", action="store_true", help="Only show insights")
    parser.add_argument("--distractors-only", action="store_true", help="Only show distractors")
    parser.add_argument("--show-files", action="store_true", help="Show supporting file paths")
    parser.add_argument("--list", action="store_true", help="List all available tasks")
    parser.add_argument("--data-dir", type=Path, help="Override DRBENCH_DATA_DIR")

    args = parser.parse_args()

    show_insights = not args.distractors_only
    show_distractors = not args.insights_only

    if args.data_dir:
        import os
        os.environ["DRBENCH_DATA_DIR"] = str(args.data_dir)

    if args.list or (not args.task_id):
        # List all tasks with summary
        print_header("Available DrBench Tasks")

        totals = {"insights": 0, "distractors": 0}

        print(f"\n{'Task':<10} {'Company':<25} {'Insights':>12} {'Distractors':>12}")
        print("-" * 70)

        for task_id in list_task_ids():
            task = get_task_from_id(task_id)
            task_config = task.get_task_config()
            eval_config = task.get_eval_config()

            company = task_config['company_info']['name'][:22]
            qa_list = eval_config.get("dr_report_evaluation_qa", [])

            # Only count enterprise_fact (private data) - external_fact is public, not privacy-relevant
            insights = len([qa for qa in qa_list if qa.get("type") == "enterprise_fact" and qa.get("qa_type") == "insight"])
            distractors = len([qa for qa in qa_list if qa.get("type") == "enterprise_fact" and qa.get("qa_type") == "distractor"])

            totals["insights"] += insights
            totals["distractors"] += distractors

            print(f"{task_id:<10} {company:<25} {insights:>12} {distractors:>12}")

        print("-" * 70)
        print(f"{'TOTAL':<10} {'':<25} {totals['insights']:>12} {totals['distractors']:>12}")

        # Show examples
        print("\n" + "=" * 80)
        print("EXAMPLES (from DR0001)")
        print("=" * 80)
        sample_task = get_task_from_id("DR0001")
        sample_config = sample_task.get_task_config()
        sample_eval = sample_task.get_eval_config()
        sample_qa = sample_eval.get("dr_report_evaluation_qa", [])

        print(f"\nResearch Question: {sample_config['dr_question']}")

        # 1. Insight
        print("\n" + "-" * 80)
        print("INSIGHT (private data model SHOULD find)")
        print("-" * 80)
        for qa in sample_qa:
            if qa.get("qa_type") == "insight" and qa.get("type") == "enterprise_fact":
                print(f"ID:       {qa['id']}")
                print(f"Question: {qa.get('question', 'N/A')}")
                print(f"Answer:   {qa['answer']}")
                if qa.get("supporting_file_paths"):
                    print(f"File:     {qa['supporting_file_paths'][0]}")
                if qa.get("justification"):
                    print(f"Why:      {qa['justification']}")
                break

        # 2. Distractor
        print("\n" + "-" * 80)
        print("DISTRACTOR (private data model should IGNORE)")
        print("-" * 80)
        for qa in sample_qa:
            if qa.get("qa_type") == "distractor":
                print(f"ID:       {qa['id']}")
                print(f"Question: {qa.get('question', 'N/A')}")
                print(f"Answer:   {qa['answer']}")
                if qa.get("supporting_file_paths"):
                    print(f"File:     {qa['supporting_file_paths'][0]}")
                if qa.get("justification"):
                    print(f"Why:      {qa['justification']}")
                break

        print("\n" + "=" * 80)
        print(f"Use: python {sys.argv[0]} <TASK_ID> to explore a specific task")
    else:
        explore_task(args.task_id, show_insights, show_distractors, args.show_files)


if __name__ == "__main__":
    main()
