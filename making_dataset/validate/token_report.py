#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate token usage by stage.")
    parser.add_argument(
        "--log",
        required=True,
        help="Path to llm_generations.jsonl",
    )
    parser.add_argument(
        "--out-json",
        default="token_report.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--out-md",
        default="token_report.md",
        help="Output Markdown path",
    )
    return parser.parse_args()


def _accumulate(stats: Dict[str, Dict[str, int]], key: str, usage: Dict[str, int]) -> None:
    bucket = stats.setdefault(key, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    bucket["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
    bucket["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
    bucket["total_tokens"] += int(usage.get("total_tokens", 0) or 0)


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _write_md(path: Path, totals: Dict[str, Dict[str, int]], overall: Dict[str, int]) -> None:
    lines = []
    lines.append("# Token Usage Report")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Metric | Tokens |")
    lines.append("| --- | ---: |")
    lines.append(f"| Prompt | {overall['prompt_tokens']} |")
    lines.append(f"| Completion | {overall['completion_tokens']} |")
    lines.append(f"| Total | {overall['total_tokens']} |")
    lines.append("")
    lines.append("## By Stage")
    lines.append("")
    lines.append("| Stage | Prompt | Completion | Total |")
    lines.append("| --- | ---: | ---: | ---: |")
    for stage, usage in sorted(totals.items()):
        lines.append(
            f"| {stage} | {usage['prompt_tokens']} | {usage['completion_tokens']} | {usage['total_tokens']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    log_path = Path(args.log)
    records = _load_records(log_path)

    totals_by_stage: Dict[str, Dict[str, int]] = {}
    totals_by_model: Dict[str, Dict[str, int]] = {}
    overall = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    overall_stats = {"overall": overall}

    for record in records:
        usage = record.get("usage") or {}
        stage = record.get("stage") or "unknown"
        model = record.get("response_model") or record.get("resolved_model") or "unknown"

        _accumulate(totals_by_stage, stage, usage)
        _accumulate(totals_by_model, model, usage)
        _accumulate(overall_stats, "overall", usage)

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)

    report = {
        "overall": overall,
        "by_stage": totals_by_stage,
        "by_model": totals_by_model,
        "log_path": str(log_path),
    }
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_md(out_md, totals_by_stage, overall)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
