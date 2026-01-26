#!/usr/bin/env python3
"""
Privacy Roleplay - Interactive test: can YOU infer private data from web searches?

You play the adversary. The script shows web search queries the agent made,
then asks you to answer private enterprise questions using ONLY those queries.
If you can answer correctly, that's a privacy leak.

This is the human version of what privacy_eval.py does with an LLM adversary.

Usage:
    python privacy_roleplay.py /path/to/batch           # Specific batch
    python privacy_roleplay.py --latest                 # Latest batch
    python privacy_roleplay.py --company "Lee"          # Filter by company

Controls:
    y - Yes, I can answer (then type your answer)
    n - No, cannot determine
    r - Review queries again
    s - Skip question
    q - Quit

Action plans are found by:
    1. Checking {batch}/DR*/action_plan_final.json (new runs)
    2. Matching session dirs by timestamp (old runs in outputs/vector_stores/)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from drbench.task_loader import get_task_from_id

# ANSI
B = '\033[1m'
D = '\033[2m'
R = '\033[0m'
GRN = '\033[92m'
YLW = '\033[93m'
RED = '\033[91m'
CYN = '\033[96m'

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
VECTOR_STORES = REPO_ROOT / "outputs" / "vector_stores"


def clear():
    os.system('clear' if os.name != 'nt' else 'cls')


def hr(char='-', width=70):
    print(char * width)


def header(text, width=70):
    print(f"\n{B}{'-'*width}{R}")
    print(f"{B}{text.center(width)}{R}")
    print(f"{B}{'-'*width}{R}\n")


def find_latest_batch(runs_dir: Path = DEFAULT_RUNS_DIR) -> Optional[Path]:
    """Find the most recent batch directory."""
    if not runs_dir.exists():
        return None
    batches = sorted(runs_dir.glob("batch_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return batches[0] if batches else None


def find_session_for_task(task_dir: Path, timeout: int = 600) -> Optional[Path]:
    """Find the session directory for a task using timestamp matching.

    The session is created at the start of generate_report(), and the task dir
    gets its final mtime when scoring completes. We match by finding the session
    whose mtime is closest to the task dir's mtime (within timeout seconds).
    """
    if not VECTOR_STORES.exists():
        return None

    task_mtime = task_dir.stat().st_mtime
    best_session = None
    best_diff = float('inf')

    for session_dir in VECTOR_STORES.iterdir():
        if not session_dir.is_dir():
            continue
        # Must have an action plan
        if not (session_dir / "action_plan_final.json").exists():
            if not (session_dir / "action_plan_initial.json").exists():
                continue

        diff = abs(session_dir.stat().st_mtime - task_mtime)
        if diff < best_diff and diff < timeout:
            best_diff = diff
            best_session = session_dir

    return best_session


def extract_web_searches(action_plan_path: Path) -> list[dict]:
    """Extract web_search calls from action plan."""
    with open(action_plan_path) as f:
        plan = json.load(f)

    searches = []
    for action in plan.get("actions", []):
        if action.get("type") == "web_search":
            query = action.get("parameters", {}).get("query", "")
            if query:
                searches.append({
                    "query": query,
                    "status": action.get("status", "unknown"),
                })
    return searches


def load_web_searches(task_dir: Path) -> Optional[list[dict]]:
    """Load web searches from privacy/web_searches.json if available."""
    path = task_dir / "privacy" / "web_searches.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data.get("searches", [])


def get_action_plan_for_task(task_dir: Path) -> Optional[Path]:
    """Get the action plan path for a task, checking run dir then session dir."""
    # First check if action plan is directly in run dir (new format)
    for name in ["action_plan_final.json", "action_plan_initial.json"]:
        path = task_dir / name
        if path.exists():
            return path

    # Fall back to session matching (old format)
    session = find_session_for_task(task_dir)
    if session:
        for name in ["action_plan_final.json", "action_plan_initial.json"]:
            path = session / name
            if path.exists():
                return path

    return None


def get_enterprise_questions(task_id: str) -> list[dict]:
    """Get enterprise fact questions for a task."""
    task = get_task_from_id(task_id)
    eval_config = task.get_eval_config()
    qa_list = eval_config.get("dr_report_evaluation_qa", [])
    return [qa for qa in qa_list if qa.get("type") == "enterprise_fact" and qa.get("question")]


def run_roleplay_batch(batch_dir: Path, company_filter: Optional[str] = None):
    """Roleplay across multiple tasks (aggregated queries)."""
    task_dirs = sorted(batch_dir.glob("DR*"))

    if not task_dirs:
        print(f"{RED}No task runs found in {batch_dir}{R}")
        return

    # Group by company
    companies = {}
    for td in task_dirs:
        task_id = td.name

        searches = load_web_searches(td)
        action_plan = None
        if searches is None:
            action_plan = get_action_plan_for_task(td)
            if not action_plan:
                print(f"{D}  {task_id}: no action plan found{R}")
                continue
            searches = extract_web_searches(action_plan)

        task = get_task_from_id(task_id)
        task_config = task.get_task_config()
        company = task_config['company_info']['name']

        if company_filter and company_filter.lower() not in company.lower():
            continue

        if company not in companies:
            companies[company] = {"tasks": [], "searches": [], "questions": []}

        companies[company]["tasks"].append(task_id)
        companies[company]["searches"].extend(searches)
        companies[company]["questions"].extend(get_enterprise_questions(task_id))

    if not companies:
        print(f"{RED}No matching tasks found{R}")
        return

    # Let user pick a company if multiple
    company_list = list(companies.keys())

    if len(company_list) > 1 and not company_filter:
        clear()
        header("SELECT COMPANY")
        for i, c in enumerate(company_list, 1):
            data = companies[c]
            print(f"  {B}{i}{R} {c} ({len(data['tasks'])} tasks, {len(data['searches'])} queries)")
        print()
        choice = input(f"  > ").strip()
        try:
            company = company_list[int(choice) - 1]
        except:
            company = company_list[0]
    else:
        company = company_list[0]

    data = companies[company]
    searches = data["searches"]
    questions = data["questions"]
    insights = [q for q in questions if q.get("qa_type") == "insight"]

    # Dedupe queries
    seen = set()
    unique_searches = []
    for s in searches:
        if s["query"] not in seen:
            seen.add(s["query"])
            unique_searches.append(s)

    clear()
    header("CROSS-TASK PRIVACY TEST")

    print(f"  {B}Company:{R}  {company}")
    print(f"  {B}Tasks:{R}    {', '.join(data['tasks'])}")
    print()
    print(f"  {B}Total Queries:{R}  {len(unique_searches)} unique web searches")
    print(f"  {B}Questions:{R}      {len(insights)} insight questions")
    print()
    print(f"  {D}This tests if COMBINED queries across tasks leak more.{R}")

    input(f"\n  {D}[Enter] View queries{R}")

    # Show queries
    clear()
    header(f"ALL QUERIES FOR {company.upper()}")

    for i, s in enumerate(unique_searches, 1):
        print(f"  {D}{i:2}.{R} {s['query']}")
    print()

    input(f"\n  {D}[Enter] Start questions{R}")

    # Questions
    correct = 0
    total = len(insights)

    for i, qa in enumerate(insights, 1):
        clear()
        print(f"{D}Question {i}/{total}{R}")
        hr()
        print()
        print(f"{B}Question:{R} {YLW}{qa['question']}{R}")
        print()
        print(f"  {B}y{R} Yes, I can answer")
        print(f"  {B}n{R} No, cannot determine")
        print(f"  {B}r{R} Review queries")
        print(f"  {B}s{R} Skip")
        print(f"  {B}q{R} Quit")

        while True:
            choice = input(f"\n  > ").strip().lower()

            if choice == 'q':
                return
            if choice == 's':
                break
            if choice == 'r':
                print()
                for j, s in enumerate(unique_searches, 1):
                    print(f"  {D}{j:2}.{R} {s['query']}")
                continue

            if choice == 'y':
                answer = input(f"\n  Your answer: ").strip()
                print(f"\n{B}Ground Truth:{R} {qa['answer']}")
                match = input(f"\n  Did you get it right? (y/n): ").strip().lower()
                if match == 'y':
                    print(f"  {RED}<- PRIVACY LEAK{R}")
                    correct += 1
                else:
                    print(f"  {GRN}<- Not leaked{R}")
                break

            if choice == 'n':
                print(f"\n{B}Ground Truth:{R} {qa['answer']}")
                print(f"  {GRN}<- Not leaked{R}")
                break

        input(f"\n  {D}[Enter] Next{R}")

    # Results
    clear()
    header("RESULTS")

    leak_rate = (correct / total * 100) if total > 0 else 0
    color = RED if leak_rate >= 50 else (YLW if leak_rate > 0 else GRN)

    print(f"  {B}Company:{R}     {company}")
    print(f"  {B}Leakage:{R}     {color}{correct}/{total}{R} ({leak_rate:.0f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Privacy Leakage Roleplay")
    parser.add_argument("run_dir", nargs="?", help="Path to batch directory")
    parser.add_argument("--latest", action="store_true", help="Use latest batch")
    parser.add_argument("--company", help="Filter by company name")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Runs directory for --latest")
    parser.add_argument("--data-dir", type=Path, help="Override DRBENCH_DATA_DIR")

    args = parser.parse_args()

    if args.data_dir:
        os.environ["DRBENCH_DATA_DIR"] = str(args.data_dir)

    if args.run_dir:
        run_path = Path(args.run_dir)
        if not run_path.exists():
            print(f"{RED}Path not found: {run_path}{R}")
            return
        run_roleplay_batch(run_path, args.company)

    elif args.latest or args.company:
        batch = find_latest_batch(args.runs_dir)
        if not batch:
            print(f"{RED}No batch found{R}")
            return
        run_roleplay_batch(batch, args.company)

    else:
        print()
        print(f"  {B}Usage:{R}")
        print(f"    python privacy_roleplay.py /path/to/batch    # Specific batch")
        print(f"    python privacy_roleplay.py --latest          # Latest batch")
        print(f"    python privacy_roleplay.py --company 'Lee'   # Filter by company")
        print()


if __name__ == "__main__":
    main()
