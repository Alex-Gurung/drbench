#!/usr/bin/env python3
"""Summarize chain building and chain privacy evaluation outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:  # pragma: no cover - exercised in plain fallback mode
    Console = None
    Panel = None
    Table = None


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "making_dataset_2" / "outputs"

VERIFY_CHECK_LABELS = {
    "no_docs_pass": "no_docs",
    "first_only_pass": "first_only",
    "last_only_pass": "last_only",
    "all_docs_pass": "all_docs",
}

EVAL_BUCKET_LABELS = {
    "agent_error": "Agent error",
    "no_actions_no_error": "No actions, no error",
    "actions_but_zero_required_docs_found": "Actions, 0 required docs",
    "partial_doc_retrieval_wrong_final": "Partial doc retrieval",
    "all_docs_found_but_wrong_final": "All docs found, wrong final",
    "final_correct": "Final correct",
}

EVAL_BUCKET_NOTES = {
    "agent_error": "Planner failed before actions; typically invalid JSON from the model.",
    "no_actions_no_error": "Agent returned an empty action plan and defaulted to NOT_FOUND.",
    "actions_but_zero_required_docs_found": "Searches ran, but none of the chain's required hop docs were retrieved.",
    "partial_doc_retrieval_wrong_final": "Some required docs were found, but missing hop coverage broke the chain.",
    "all_docs_found_but_wrong_final": "Retrieval succeeded, but the final answer was still wrong or badly formatted.",
    "final_correct": "Final answer matched under the run's current scoring rule.",
}

EVAL_BUCKET_ORDER = [
    "agent_error",
    "no_actions_no_error",
    "actions_but_zero_required_docs_found",
    "partial_doc_retrieval_wrong_final",
    "all_docs_found_but_wrong_final",
    "final_correct",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect chain generation and chain privacy evaluation outputs.",
    )
    parser.add_argument(
        "--chains",
        type=Path,
        help="Chain build JSONL. Defaults to the latest chains_*/chains.jsonl under outputs.",
    )
    parser.add_argument(
        "--eval",
        dest="eval_path",
        type=Path,
        help="Chain privacy eval JSONL. Defaults to the latest chain_privacy*.jsonl under outputs.",
    )
    parser.add_argument(
        "--top-tasks",
        type=int,
        default=12,
        help="How many task rows to show in the eval-by-task table.",
    )
    parser.add_argument(
        "--top-failures",
        type=int,
        default=8,
        help="How many verification failure patterns to show.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to write the computed summary as JSON.",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Force plain-text output even if rich is installed.",
    )
    return parser.parse_args()


def _latest_chain_path() -> Path | None:
    preferred = [
        path
        for path in OUTPUTS_DIR.glob("chains_*/chains.jsonl")
        if path.is_file()
    ]
    if preferred:
        return max(preferred, key=lambda path: path.stat().st_mtime)

    fallback = [
        path
        for path in OUTPUTS_DIR.glob("chains*.jsonl")
        if path.is_file() and path.name.startswith("chains")
    ]
    if fallback:
        return max(fallback, key=lambda path: path.stat().st_mtime)
    return None


def _latest_eval_path() -> Path | None:
    candidates = [
        path
        for path in OUTPUTS_DIR.glob("chain_privacy*.jsonl")
        if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_progress(eval_path: Path | None) -> dict[str, Any] | None:
    if eval_path is None:
        return None
    progress_path = eval_path.with_suffix(".progress.json")
    if not progress_path.exists():
        return None
    with progress_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _safe_div(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return numerator / denominator


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_seconds(value: float) -> str:
    if value >= 3600:
        return f"{value / 3600:.1f}h"
    if value >= 60:
        return f"{value / 60:.1f}m"
    return f"{value:.0f}s"


def _normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _inclusive_final_match(result: dict[str, Any]) -> bool:
    parsed_answers = result.get("agent_run", {}).get("parsed_answers", {})
    fallback_key = str(len(result.get("hops", [])))
    agent_answer = _normalize_text(parsed_answers.get("FINAL", parsed_answers.get(fallback_key, "")))
    truth = _normalize_text(result.get("global_answer", ""))
    return bool(
        agent_answer == truth
        or (truth and truth in agent_answer)
        or (agent_answer and agent_answer in truth)
    )


def _exact_final_match(result: dict[str, Any]) -> bool:
    parsed_answers = result.get("agent_run", {}).get("parsed_answers", {})
    fallback_key = str(len(result.get("hops", [])))
    agent_answer = _normalize_text(parsed_answers.get("FINAL", parsed_answers.get(fallback_key, "")))
    truth = _normalize_text(result.get("global_answer", ""))
    return bool(agent_answer and agent_answer == truth)


def classify_eval_result(result: dict[str, Any]) -> str:
    answer_eval = result.get("answer_eval", {})
    agent_run = result.get("agent_run", {})
    doc_retrieval = result.get("doc_retrieval", {})

    if answer_eval.get("final_correct"):
        return "final_correct"
    if agent_run.get("error"):
        return "agent_error"
    if agent_run.get("total_actions", 0) == 0:
        return "no_actions_no_error"
    if doc_retrieval.get("found_count", 0) == 0:
        return "actions_but_zero_required_docs_found"
    if doc_retrieval.get("found_count", 0) < doc_retrieval.get("total_count", 0):
        return "partial_doc_retrieval_wrong_final"
    return "all_docs_found_but_wrong_final"


def _pattern_sort_key(pattern: str) -> tuple[int, str]:
    return (len(pattern), pattern)


def summarize_build(
    chains: list[dict[str, Any]],
    eval_ids: set[str] | None = None,
    top_failures: int = 8,
) -> dict[str, Any]:
    eval_ids = eval_ids or set()
    total = len(chains)
    complete = sum(bool(chain.get("metadata", {}).get("complete")) for chain in chains)
    valid_ids = {
        chain.get("chain_id")
        for chain in chains
        if chain.get("verification", {}).get("is_valid")
    }
    valid_ids.discard(None)
    valid = len(valid_ids)

    llm_calls = [chain.get("metadata", {}).get("llm_calls", 0) for chain in chains]
    build_seconds = [chain.get("metadata", {}).get("elapsed_seconds", 0.0) for chain in chains]
    hop_counts = [len(chain.get("hops", [])) for chain in chains]

    by_pattern: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chain in chains:
        grouped[chain.get("pattern", "?")].append(chain)

    for pattern in sorted(grouped, key=_pattern_sort_key):
        subset = grouped[pattern]
        subset_valid_ids = {
            chain.get("chain_id")
            for chain in subset
            if chain.get("verification", {}).get("is_valid")
        }
        subset_valid_ids.discard(None)
        by_pattern.append(
            {
                "pattern": pattern,
                "total": len(subset),
                "complete": sum(bool(chain.get("metadata", {}).get("complete")) for chain in subset),
                "valid": len(subset_valid_ids),
                "eval_covered": len(subset_valid_ids & eval_ids),
                "avg_llm_calls": _safe_div(
                    sum(chain.get("metadata", {}).get("llm_calls", 0) for chain in subset),
                    len(subset),
                ),
                "avg_build_seconds": _safe_div(
                    sum(chain.get("metadata", {}).get("elapsed_seconds", 0.0) for chain in subset),
                    len(subset),
                ),
            }
        )

    invalid_chains = [
        chain for chain in chains if not chain.get("verification", {}).get("is_valid")
    ]
    failure_counter: Counter[tuple[str, ...]] = Counter()
    for chain in invalid_chains:
        verification = chain.get("verification", {})
        failed_checks = tuple(
            VERIFY_CHECK_LABELS[key]
            for key in VERIFY_CHECK_LABELS
            if not verification.get(key)
        )
        if not failed_checks:
            failed_checks = ("unknown",)
        failure_counter[failed_checks] += 1

    verification_failures = [
        {
            "failed_checks": ", ".join(combo),
            "count": count,
            "pct_invalid": _safe_div(count, len(invalid_chains)),
        }
        for combo, count in failure_counter.most_common(top_failures)
    ]

    return {
        "overview": {
            "total_chains": total,
            "complete_chains": complete,
            "complete_rate": _safe_div(complete, total),
            "valid_chains": valid,
            "valid_rate": _safe_div(valid, total),
            "eval_covered": len(valid_ids & eval_ids),
            "eval_coverage_rate": _safe_div(len(valid_ids & eval_ids), valid),
            "avg_hops": _safe_div(sum(hop_counts), total),
            "avg_llm_calls": _safe_div(sum(llm_calls), total),
            "avg_build_seconds": _safe_div(sum(build_seconds), total),
            "missing_eval_chain_ids": sorted(valid_ids - eval_ids),
        },
        "by_pattern": by_pattern,
        "verification_failures": verification_failures,
    }


def summarize_eval(results: list[dict[str, Any]], top_tasks: int = 12) -> dict[str, Any]:
    total = len(results)
    bucket_rows: list[dict[str, Any]] = []
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pattern: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)

    docs_found_total = 0
    docs_total = 0
    exact_final = 0
    inclusive_final = 0
    chain_complete = 0
    hop_accuracy_sum = 0.0
    total_actions = 0
    total_web = 0
    total_local = 0
    total_elapsed = 0.0
    error_counter: Counter[str] = Counter()

    for result in results:
        bucket = classify_eval_result(result)
        by_bucket[bucket].append(result)
        by_pattern[result.get("pattern", "?")].append(result)
        task_id = result.get("metadata", {}).get("task_id", "?")
        by_task[task_id].append(result)

        answer_eval = result.get("answer_eval", {})
        agent_run = result.get("agent_run", {})
        doc_retrieval = result.get("doc_retrieval", {})

        docs_found_total += doc_retrieval.get("found_count", 0)
        docs_total += doc_retrieval.get("total_count", 0)
        exact_final += int(_exact_final_match(result))
        inclusive_final += int(_inclusive_final_match(result))
        chain_complete += int(bool(answer_eval.get("chain_complete")))
        hop_accuracy_sum += answer_eval.get("hop_accuracy", 0.0)
        total_actions += agent_run.get("total_actions", 0)
        total_web += len(agent_run.get("web_searches", []))
        total_local += len(agent_run.get("local_searches", []))
        total_elapsed += agent_run.get("elapsed_seconds", 0.0)

        if agent_run.get("error"):
            error_counter[agent_run["error"]] += 1

    for bucket in EVAL_BUCKET_ORDER:
        subset = by_bucket.get(bucket, [])
        if not subset:
            continue
        subset_actions = sum(item.get("agent_run", {}).get("total_actions", 0) for item in subset)
        subset_searches = sum(
            len(item.get("agent_run", {}).get("web_searches", []))
            + len(item.get("agent_run", {}).get("local_searches", []))
            for item in subset
        )
        subset_docs_found = sum(item.get("doc_retrieval", {}).get("found_count", 0) for item in subset)
        subset_docs_total = sum(item.get("doc_retrieval", {}).get("total_count", 0) for item in subset)
        subset_hop_acc = sum(item.get("answer_eval", {}).get("hop_accuracy", 0.0) for item in subset)
        subset_elapsed = sum(item.get("agent_run", {}).get("elapsed_seconds", 0.0) for item in subset)

        bucket_rows.append(
            {
                "bucket": bucket,
                "label": EVAL_BUCKET_LABELS[bucket],
                "count": len(subset),
                "pct_total": _safe_div(len(subset), total),
                "avg_actions": _safe_div(subset_actions, len(subset)),
                "avg_searches": _safe_div(subset_searches, len(subset)),
                "doc_rate": _safe_div(subset_docs_found, subset_docs_total),
                "avg_hop_accuracy": _safe_div(subset_hop_acc, len(subset)),
                "avg_elapsed_seconds": _safe_div(subset_elapsed, len(subset)),
                "note": EVAL_BUCKET_NOTES[bucket],
            }
        )

    pattern_rows: list[dict[str, Any]] = []
    for pattern in sorted(by_pattern, key=_pattern_sort_key):
        subset = by_pattern[pattern]
        pattern_rows.append(
            {
                "pattern": pattern,
                "count": len(subset),
                "error_rate": _safe_div(
                    sum(bool(item.get("agent_run", {}).get("error")) for item in subset),
                    len(subset),
                ),
                "inclusive_final_rate": _safe_div(
                    sum(bool(item.get("answer_eval", {}).get("final_correct")) for item in subset),
                    len(subset),
                ),
                "exact_final_rate": _safe_div(
                    sum(_exact_final_match(item) for item in subset),
                    len(subset),
                ),
                "avg_hop_accuracy": _safe_div(
                    sum(item.get("answer_eval", {}).get("hop_accuracy", 0.0) for item in subset),
                    len(subset),
                ),
                "doc_rate": _safe_div(
                    sum(item.get("doc_retrieval", {}).get("found_count", 0) for item in subset),
                    sum(item.get("doc_retrieval", {}).get("total_count", 0) for item in subset),
                ),
                "avg_actions": _safe_div(
                    sum(item.get("agent_run", {}).get("total_actions", 0) for item in subset),
                    len(subset),
                ),
                "avg_web": _safe_div(
                    sum(len(item.get("agent_run", {}).get("web_searches", [])) for item in subset),
                    len(subset),
                ),
                "avg_local": _safe_div(
                    sum(len(item.get("agent_run", {}).get("local_searches", [])) for item in subset),
                    len(subset),
                ),
            }
        )

    task_rows: list[dict[str, Any]] = []
    for task_id, subset in by_task.items():
        task_rows.append(
            {
                "task_id": task_id,
                "count": len(subset),
                "error_rate": _safe_div(
                    sum(bool(item.get("agent_run", {}).get("error")) for item in subset),
                    len(subset),
                ),
                "inclusive_final_rate": _safe_div(
                    sum(bool(item.get("answer_eval", {}).get("final_correct")) for item in subset),
                    len(subset),
                ),
                "exact_final_rate": _safe_div(
                    sum(_exact_final_match(item) for item in subset),
                    len(subset),
                ),
                "avg_hop_accuracy": _safe_div(
                    sum(item.get("answer_eval", {}).get("hop_accuracy", 0.0) for item in subset),
                    len(subset),
                ),
                "doc_rate": _safe_div(
                    sum(item.get("doc_retrieval", {}).get("found_count", 0) for item in subset),
                    sum(item.get("doc_retrieval", {}).get("total_count", 0) for item in subset),
                ),
                "avg_actions": _safe_div(
                    sum(item.get("agent_run", {}).get("total_actions", 0) for item in subset),
                    len(subset),
                ),
            }
        )

    task_rows.sort(
        key=lambda row: (
            row["inclusive_final_rate"],
            row["avg_hop_accuracy"],
            row["doc_rate"],
            -row["error_rate"],
            -row["count"],
            row["task_id"],
        )
    )

    examples: list[dict[str, Any]] = []
    for bucket in EVAL_BUCKET_ORDER:
        subset = by_bucket.get(bucket, [])
        if not subset:
            continue
        example = subset[0]
        parsed_answers = example.get("agent_run", {}).get("parsed_answers", {})
        examples.append(
            {
                "bucket": EVAL_BUCKET_LABELS[bucket],
                "chain_id": example.get("chain_id", "?"),
                "task_id": example.get("metadata", {}).get("task_id", "?"),
                "pattern": example.get("pattern", "?"),
                "actions": example.get("agent_run", {}).get("total_actions", 0),
                "docs": (
                    f"{example.get('doc_retrieval', {}).get('found_count', 0)}"
                    f"/{example.get('doc_retrieval', {}).get('total_count', 0)}"
                ),
                "pred_final": parsed_answers.get("FINAL", ""),
                "truth_final": example.get("global_answer", ""),
            }
        )

    return {
        "overview": {
            "total_results": total,
            "agent_errors": sum(bool(item.get("agent_run", {}).get("error")) for item in results),
            "agent_error_rate": _safe_div(
                sum(bool(item.get("agent_run", {}).get("error")) for item in results),
                total,
            ),
            "inclusive_final_correct": inclusive_final,
            "inclusive_final_rate": _safe_div(inclusive_final, total),
            "exact_final_correct": exact_final,
            "exact_final_rate": _safe_div(exact_final, total),
            "chain_complete": chain_complete,
            "chain_complete_rate": _safe_div(chain_complete, total),
            "avg_hop_accuracy": _safe_div(hop_accuracy_sum, total),
            "doc_hit_rate": _safe_div(docs_found_total, docs_total),
            "company_name_leaks": sum(
                bool(item.get("privacy_eval", {}).get("company_name_leaked")) for item in results
            ),
            "secrets_leaked": sum(item.get("privacy_eval", {}).get("secrets_leaked", 0) for item in results),
            "secrets_total": sum(item.get("privacy_eval", {}).get("secrets_total", 0) for item in results),
            "avg_actions": _safe_div(total_actions, total),
            "avg_web": _safe_div(total_web, total),
            "avg_local": _safe_div(total_local, total),
            "avg_elapsed_seconds": _safe_div(total_elapsed, total),
        },
        "error_types": [
            {"error": error, "count": count, "pct_total": _safe_div(count, total)}
            for error, count in error_counter.most_common()
        ],
        "failure_classes": bucket_rows,
        "by_pattern": pattern_rows,
        "by_task": task_rows[:top_tasks],
        "examples": examples,
    }


def build_report(
    chains: list[dict[str, Any]],
    results: list[dict[str, Any]],
    progress: dict[str, Any] | None,
    top_tasks: int = 12,
    top_failures: int = 8,
) -> dict[str, Any]:
    eval_ids = {result.get("chain_id") for result in results if result.get("chain_id")}
    build_summary = summarize_build(chains, eval_ids=eval_ids, top_failures=top_failures)
    eval_summary = summarize_eval(results, top_tasks=top_tasks)
    return {
        "artifacts": {
            "build_total_rows": len(chains),
            "eval_total_rows": len(results),
            "progress_status": progress.get("status") if progress else None,
            "progress_processed": progress.get("processed") if progress else None,
            "progress_total": progress.get("total") if progress else None,
        },
        "build": build_summary,
        "evaluation": eval_summary,
    }


def _truncate(text: Any, width: int) -> str:
    value = "" if text is None else str(text)
    if len(value) <= width:
        return value
    return value[: max(0, width - 1)] + "…"


def _plain_table(title: str, columns: list[str], rows: list[list[str]]) -> str:
    widths = [len(column) for column in columns]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def _fmt_row(items: list[str]) -> str:
        return "  ".join(item.ljust(widths[idx]) for idx, item in enumerate(items))

    parts = [title, _fmt_row(columns), _fmt_row(["-" * width for width in widths])]
    parts.extend(_fmt_row(row) for row in rows)
    return "\n".join(parts)


def _render_plain(report: dict[str, Any], chains_path: Path, eval_path: Path) -> None:
    print("Artifacts")
    print(f"  chains: {chains_path}")
    print(f"  eval:   {eval_path}")
    progress_status = report["artifacts"].get("progress_status")
    if progress_status:
        print(
            "  progress:"
            f" {progress_status} "
            f"({report['artifacts'].get('progress_processed')}/"
            f"{report['artifacts'].get('progress_total')})"
        )
    print()

    build = report["build"]["overview"]
    print(
        _plain_table(
            "Build Overview",
            ["metric", "value"],
            [
                ["chains generated", str(build["total_chains"])],
                ["complete", f"{build['complete_chains']} ({_pct(build['complete_rate'])})"],
                ["valid", f"{build['valid_chains']} ({_pct(build['valid_rate'])})"],
                ["eval coverage", f"{build['eval_covered']} ({_pct(build['eval_coverage_rate'])})"],
                ["avg hops", f"{build['avg_hops']:.2f}"],
                ["avg llm calls", f"{build['avg_llm_calls']:.1f}"],
                ["avg build time", _fmt_seconds(build["avg_build_seconds"])],
            ],
        )
    )
    print()

    eval_overview = report["evaluation"]["overview"]
    print(
        _plain_table(
            "Eval Overview",
            ["metric", "value"],
            [
                ["rows", str(eval_overview["total_results"])],
                ["agent errors", f"{eval_overview['agent_errors']} ({_pct(eval_overview['agent_error_rate'])})"],
                ["final correct", f"{eval_overview['inclusive_final_correct']} ({_pct(eval_overview['inclusive_final_rate'])})"],
                ["exact final correct", f"{eval_overview['exact_final_correct']} ({_pct(eval_overview['exact_final_rate'])})"],
                ["chain complete", f"{eval_overview['chain_complete']} ({_pct(eval_overview['chain_complete_rate'])})"],
                ["avg hop accuracy", _pct(eval_overview["avg_hop_accuracy"])],
                ["doc hit rate", _pct(eval_overview["doc_hit_rate"])],
                ["company leaks", str(eval_overview["company_name_leaks"])],
                ["secrets leaked", f"{eval_overview['secrets_leaked']}/{eval_overview['secrets_total']}"],
                ["avg actions", f"{eval_overview['avg_actions']:.2f}"],
            ],
        )
    )
    print()

    bucket_rows = [
        [
            row["label"],
            str(row["count"]),
            _pct(row["pct_total"]),
            f"{row['avg_actions']:.2f}",
            f"{row['avg_searches']:.2f}",
            _pct(row["doc_rate"]),
            _pct(row["avg_hop_accuracy"]),
        ]
        for row in report["evaluation"]["failure_classes"]
    ]
    print(
        _plain_table(
            "Failure Breakdown",
            ["bucket", "count", "share", "avg actions", "avg searches", "doc rate", "hop acc"],
            bucket_rows,
        )
    )


def _render_rich(report: dict[str, Any], chains_path: Path, eval_path: Path) -> None:
    console = Console()
    progress_status = report["artifacts"].get("progress_status")
    progress_bits = []
    if progress_status:
        progress_bits.append(f"status={progress_status}")
        progress_bits.append(
            f"processed={report['artifacts'].get('progress_processed')}/"
            f"{report['artifacts'].get('progress_total')}"
        )

    artifacts_body = [
        f"[bold]chains[/bold]  {chains_path}",
        f"[bold]eval[/bold]    {eval_path}",
    ]
    if progress_bits:
        artifacts_body.append(f"[bold]progress[/bold] {' | '.join(progress_bits)}")
    console.print(Panel("\n".join(artifacts_body), title="Artifacts", expand=False))

    build_overview = report["build"]["overview"]
    build_table = Table(title="Chain Build Overview", show_lines=False)
    build_table.add_column("Metric")
    build_table.add_column("Value", justify="right")
    build_table.add_row("Chains generated", str(build_overview["total_chains"]))
    build_table.add_row(
        "Complete",
        f"{build_overview['complete_chains']} ({_pct(build_overview['complete_rate'])})",
    )
    build_table.add_row(
        "Valid",
        f"{build_overview['valid_chains']} ({_pct(build_overview['valid_rate'])})",
    )
    build_table.add_row(
        "Eval coverage",
        f"{build_overview['eval_covered']} ({_pct(build_overview['eval_coverage_rate'])})",
    )
    build_table.add_row("Avg hops", f"{build_overview['avg_hops']:.2f}")
    build_table.add_row("Avg LLM calls", f"{build_overview['avg_llm_calls']:.1f}")
    build_table.add_row("Avg build time", _fmt_seconds(build_overview["avg_build_seconds"]))
    console.print(build_table)

    pattern_table = Table(title="Build By Pattern")
    for column in [
        "Pattern",
        "Total",
        "Complete",
        "Valid",
        "Eval/Valid",
        "Avg LLM",
        "Avg Build",
    ]:
        pattern_table.add_column(column, justify="right" if column != "Pattern" else "left")
    for row in report["build"]["by_pattern"]:
        pattern_table.add_row(
            row["pattern"],
            str(row["total"]),
            _pct(_safe_div(row["complete"], row["total"])),
            _pct(_safe_div(row["valid"], row["total"])),
            f"{row['eval_covered']}/{row['valid']}",
            f"{row['avg_llm_calls']:.1f}",
            _fmt_seconds(row["avg_build_seconds"]),
        )
    console.print(pattern_table)

    verification_table = Table(title="Top Verification Failure Patterns")
    verification_table.add_column("Failed checks")
    verification_table.add_column("Count", justify="right")
    verification_table.add_column("% invalid", justify="right")
    for row in report["build"]["verification_failures"]:
        verification_table.add_row(
            row["failed_checks"],
            str(row["count"]),
            _pct(row["pct_invalid"]),
        )
    console.print(verification_table)

    if build_overview["missing_eval_chain_ids"]:
        missing_preview = ", ".join(build_overview["missing_eval_chain_ids"][:7])
        console.print(
            f"[yellow]Missing valid chains from eval:[/yellow] "
            f"{len(build_overview['missing_eval_chain_ids'])} "
            f"({missing_preview})"
        )

    eval_overview = report["evaluation"]["overview"]
    eval_table = Table(title="Chain Eval Overview")
    eval_table.add_column("Metric")
    eval_table.add_column("Value", justify="right")
    eval_table.add_row("Eval rows", str(eval_overview["total_results"]))
    eval_table.add_row(
        "Agent errors",
        f"{eval_overview['agent_errors']} ({_pct(eval_overview['agent_error_rate'])})",
    )
    eval_table.add_row(
        "Final correct",
        f"{eval_overview['inclusive_final_correct']} ({_pct(eval_overview['inclusive_final_rate'])})",
    )
    eval_table.add_row(
        "Exact final correct",
        f"{eval_overview['exact_final_correct']} ({_pct(eval_overview['exact_final_rate'])})",
    )
    eval_table.add_row(
        "Chain complete",
        f"{eval_overview['chain_complete']} ({_pct(eval_overview['chain_complete_rate'])})",
    )
    eval_table.add_row("Avg hop accuracy", _pct(eval_overview["avg_hop_accuracy"]))
    eval_table.add_row("Doc hit rate", _pct(eval_overview["doc_hit_rate"]))
    eval_table.add_row("Company name leaks", str(eval_overview["company_name_leaks"]))
    eval_table.add_row(
        "Secret leaks",
        f"{eval_overview['secrets_leaked']}/{eval_overview['secrets_total']}",
    )
    eval_table.add_row("Avg actions", f"{eval_overview['avg_actions']:.2f}")
    eval_table.add_row(
        "Avg searches",
        f"{eval_overview['avg_web'] + eval_overview['avg_local']:.2f}"
        f" ({eval_overview['avg_web']:.2f} web, {eval_overview['avg_local']:.2f} local)",
    )
    eval_table.add_row("Avg eval time", _fmt_seconds(eval_overview["avg_elapsed_seconds"]))
    console.print(eval_table)

    error_types = report["evaluation"]["error_types"]
    if error_types:
        error_table = Table(title="Agent Error Types")
        error_table.add_column("Error")
        error_table.add_column("Count", justify="right")
        error_table.add_column("% eval", justify="right")
        for row in error_types:
            error_table.add_row(
                _truncate(row["error"], 80),
                str(row["count"]),
                _pct(row["pct_total"]),
            )
        console.print(error_table)

    failure_table = Table(title="Why The Eval Looks Bad")
    failure_table.add_column("Bucket")
    failure_table.add_column("Count", justify="right")
    failure_table.add_column("Share", justify="right")
    failure_table.add_column("Avg actions", justify="right")
    failure_table.add_column("Avg searches", justify="right")
    failure_table.add_column("Doc rate", justify="right")
    failure_table.add_column("Hop acc", justify="right")
    failure_table.add_column("Interpretation")
    for row in report["evaluation"]["failure_classes"]:
        failure_table.add_row(
            row["label"],
            str(row["count"]),
            _pct(row["pct_total"]),
            f"{row['avg_actions']:.2f}",
            f"{row['avg_searches']:.2f}",
            _pct(row["doc_rate"]),
            _pct(row["avg_hop_accuracy"]),
            _truncate(row["note"], 58),
        )
    console.print(failure_table)

    eval_pattern_table = Table(title="Eval By Pattern")
    for column in [
        "Pattern",
        "Count",
        "Err %",
        "Final %",
        "Exact %",
        "Hop acc",
        "Doc rate",
        "Avg actions",
        "Avg web",
        "Avg local",
    ]:
        eval_pattern_table.add_column(column, justify="right" if column != "Pattern" else "left")
    for row in report["evaluation"]["by_pattern"]:
        eval_pattern_table.add_row(
            row["pattern"],
            str(row["count"]),
            _pct(row["error_rate"]),
            _pct(row["inclusive_final_rate"]),
            _pct(row["exact_final_rate"]),
            _pct(row["avg_hop_accuracy"]),
            _pct(row["doc_rate"]),
            f"{row['avg_actions']:.2f}",
            f"{row['avg_web']:.2f}",
            f"{row['avg_local']:.2f}",
        )
    console.print(eval_pattern_table)

    task_table = Table(title="Worst Tasks In Current Eval")
    task_table.add_column("Task")
    task_table.add_column("Count", justify="right")
    task_table.add_column("Err %", justify="right")
    task_table.add_column("Final %", justify="right")
    task_table.add_column("Exact %", justify="right")
    task_table.add_column("Hop acc", justify="right")
    task_table.add_column("Doc rate", justify="right")
    task_table.add_column("Avg actions", justify="right")
    for row in report["evaluation"]["by_task"]:
        task_table.add_row(
            row["task_id"],
            str(row["count"]),
            _pct(row["error_rate"]),
            _pct(row["inclusive_final_rate"]),
            _pct(row["exact_final_rate"]),
            _pct(row["avg_hop_accuracy"]),
            _pct(row["doc_rate"]),
            f"{row['avg_actions']:.2f}",
        )
    console.print(task_table)

    example_table = Table(title="Representative Chains")
    example_table.add_column("Bucket")
    example_table.add_column("Chain")
    example_table.add_column("Task")
    example_table.add_column("Pattern")
    example_table.add_column("Actions", justify="right")
    example_table.add_column("Docs", justify="right")
    example_table.add_column("Pred final")
    example_table.add_column("Truth")
    for row in report["evaluation"]["examples"]:
        example_table.add_row(
            row["bucket"],
            row["chain_id"],
            row["task_id"],
            row["pattern"],
            str(row["actions"]),
            row["docs"],
            _truncate(row["pred_final"], 22),
            _truncate(row["truth_final"], 22),
        )
    console.print(example_table)


def main() -> int:
    args = _parse_args()
    chains_path = args.chains or _latest_chain_path()
    eval_path = args.eval_path or _latest_eval_path()

    if chains_path is None:
        print("Could not find a chain build JSONL under making_dataset_2/outputs.", file=sys.stderr)
        return 1
    if eval_path is None:
        print("Could not find a chain privacy eval JSONL under making_dataset_2/outputs.", file=sys.stderr)
        return 1

    chains = _load_jsonl(chains_path)
    results = _load_jsonl(eval_path)
    progress = _load_progress(eval_path)
    report = build_report(
        chains=chains,
        results=results,
        progress=progress,
        top_tasks=args.top_tasks,
        top_failures=args.top_failures,
    )
    report["artifacts"].update(
        {
            "chains_path": str(chains_path),
            "eval_path": str(eval_path),
        }
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.no_rich or Console is None:
        _render_plain(report, chains_path=chains_path, eval_path=eval_path)
    else:
        _render_rich(report, chains_path=chains_path, eval_path=eval_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
