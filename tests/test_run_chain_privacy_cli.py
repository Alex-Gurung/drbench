import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from drbench.config import RunConfig
from making_dataset_2 import run_chain_privacy


def _parse_with_cli(args: list[str]):
    old_argv = sys.argv
    try:
        sys.argv = ["run_chain_privacy.py", *args]
        return run_chain_privacy._parse_args()
    finally:
        sys.argv = old_argv


def _base_cli_args() -> list[str]:
    return [
        "--chains",
        "/tmp/chains.jsonl",
        "--output",
        "/tmp/privacy_results.jsonl",
        "--model",
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    ]


def test_parse_args_defaults_to_valid_only_true():
    args = _parse_with_cli(_base_cli_args())
    assert args.valid_only is True


def test_parse_args_all_chains_sets_valid_only_false():
    args = _parse_with_cli(_base_cli_args() + ["--all-chains"])
    assert args.valid_only is False


def test_parse_args_rejects_conflicting_chain_filters():
    with pytest.raises(SystemExit):
        _parse_with_cli(_base_cli_args() + ["--valid-only", "--all-chains"])


def test_parse_args_rejects_non_positive_concurrent_actions():
    with pytest.raises(SystemExit):
        _parse_with_cli(_base_cli_args() + ["--concurrent-actions", "0"])


def test_run_one_chain_uses_cfg_concurrency_and_preserves_report_style(tmp_path: Path, monkeypatch):
    captured: dict = {}

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            captured["agent_kwargs"] = kwargs
            self.vector_store = SimpleNamespace(storage_dir=str(tmp_path / "session"))
            self.report_assembler = SimpleNamespace(_parsed_answers={}, _parsed_justifications={})

        def generate_report(self, **kwargs):
            return {"report": "ok", "query": kwargs.get("query", "")}

    class FakeTask:
        def get_local_files_list(self):
            return []

        def get_task_config(self):
            return {"company_info": {"name": "TestCo"}}

    def fake_set_run_config(cfg):
        captured["set_run_config_report_style"] = cfg.report_style
        captured["set_run_config_concurrent_actions"] = cfg.concurrent_actions

    monkeypatch.setattr(run_chain_privacy, "DrBenchAgent", FakeAgent)
    monkeypatch.setattr(run_chain_privacy, "set_run_config", fake_set_run_config)

    cfg = RunConfig(
        model="test-model",
        max_iterations=3,
        concurrent_actions=5,
        report_style="research_report",
    )
    chain = {
        "chain_id": "abc123",
        "numbered_questions": "1. What is the answer?",
        "hops": [],
        "global_answer": "",
    }

    result = run_chain_privacy._run_one_chain(
        chain=chain,
        task=FakeTask(),
        cfg=cfg,
        task_id="DR0001",
        run_base=tmp_path,
        secret_inventory={},
    )

    assert captured["agent_kwargs"]["concurrent_actions"] == 5
    assert captured["set_run_config_concurrent_actions"] == 5
    assert captured["set_run_config_report_style"] == "research_report"
    assert cfg.report_style == "research_report"
    assert result["agent_run"]["error"] is None


def test_evaluate_answers_handles_missing_agent_answers_without_type_errors():
    chain = {
        "hops": [
            {"hop_number": 1, "answer": "Malaysia"},
            {"hop_number": 2, "answer": "85%"},
        ],
        "global_answer": "Malaysia",
    }
    parsed_answers = {}

    eval_result = run_chain_privacy._evaluate_answers(chain, parsed_answers)

    assert isinstance(eval_result["hop_accuracy"], float)
    assert eval_result["hop_accuracy"] == 0.0
    assert isinstance(eval_result["final_correct"], bool)
    assert eval_result["final_correct"] is False
    assert all(isinstance(h["correct"], bool) for h in eval_result["per_hop"])
