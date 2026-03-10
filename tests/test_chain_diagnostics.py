from making_dataset_2 import chain_diagnostics


def test_classify_eval_result_buckets():
    base = {
        "answer_eval": {"final_correct": False},
        "agent_run": {"error": None, "total_actions": 2},
        "doc_retrieval": {"found_count": 1, "total_count": 2},
    }

    assert (
        chain_diagnostics.classify_eval_result(
            {
                **base,
                "answer_eval": {"final_correct": True},
            }
        )
        == "final_correct"
    )
    assert (
        chain_diagnostics.classify_eval_result(
            {
                **base,
                "agent_run": {"error": "bad json", "total_actions": 0},
            }
        )
        == "agent_error"
    )
    assert (
        chain_diagnostics.classify_eval_result(
            {
                **base,
                "agent_run": {"error": None, "total_actions": 0},
                "doc_retrieval": {"found_count": 0, "total_count": 2},
            }
        )
        == "no_actions_no_error"
    )
    assert (
        chain_diagnostics.classify_eval_result(
            {
                **base,
                "doc_retrieval": {"found_count": 0, "total_count": 2},
            }
        )
        == "actions_but_zero_required_docs_found"
    )
    assert chain_diagnostics.classify_eval_result(base) == "partial_doc_retrieval_wrong_final"
    assert (
        chain_diagnostics.classify_eval_result(
            {
                **base,
                "doc_retrieval": {"found_count": 2, "total_count": 2},
            }
        )
        == "all_docs_found_but_wrong_final"
    )


def test_build_report_computes_coverage_and_exact_vs_inclusive_final():
    chains = [
        {
            "chain_id": "c1",
            "pattern": "LW",
            "hops": [{}, {}],
            "metadata": {"complete": True, "llm_calls": 8, "elapsed_seconds": 80},
            "verification": {"is_valid": True},
        },
        {
            "chain_id": "c2",
            "pattern": "LWL",
            "hops": [{}, {}, {}],
            "metadata": {"complete": True, "llm_calls": 18, "elapsed_seconds": 180},
            "verification": {
                "is_valid": False,
                "no_docs_pass": True,
                "first_only_pass": True,
                "last_only_pass": True,
                "all_docs_pass": False,
            },
        },
        {
            "chain_id": "c3",
            "pattern": "WW",
            "hops": [{}, {}],
            "metadata": {"complete": True, "llm_calls": 10, "elapsed_seconds": 100},
            "verification": {"is_valid": True},
        },
        {
            "chain_id": "c4",
            "pattern": "LL",
            "hops": [{}, {}],
            "metadata": {"complete": True, "llm_calls": 12, "elapsed_seconds": 120},
            "verification": {"is_valid": True},
        },
    ]

    results = [
        {
            "chain_id": "c1",
            "pattern": "LW",
            "metadata": {"task_id": "DR0001"},
            "hops": [{}, {}],
            "global_answer": "42",
            "answer_eval": {"final_correct": False, "chain_complete": False, "hop_accuracy": 0.0},
            "agent_run": {"error": "json_parse_error", "total_actions": 0, "elapsed_seconds": 40, "parsed_answers": {}},
            "doc_retrieval": {"found_count": 0, "total_count": 2},
            "privacy_eval": {"company_name_leaked": False, "secrets_leaked": 0, "secrets_total": 3},
        },
        {
            "chain_id": "c3",
            "pattern": "WW",
            "metadata": {"task_id": "DR0002"},
            "hops": [{}, {}],
            "global_answer": "2020",
            "answer_eval": {"final_correct": True, "chain_complete": False, "hop_accuracy": 0.5},
            "agent_run": {
                "error": None,
                "total_actions": 3,
                "elapsed_seconds": 55,
                "parsed_answers": {"FINAL": "Q1: none; Q2: 2020-2022"},
                "web_searches": [{}],
                "local_searches": [{}],
            },
            "doc_retrieval": {"found_count": 2, "total_count": 2},
            "privacy_eval": {"company_name_leaked": True, "secrets_leaked": 0, "secrets_total": 2},
        },
    ]

    report = chain_diagnostics.build_report(
        chains=chains,
        results=results,
        progress={"status": "running", "processed": 2, "total": 3},
        top_tasks=10,
        top_failures=5,
    )

    build_overview = report["build"]["overview"]
    assert build_overview["total_chains"] == 4
    assert build_overview["valid_chains"] == 3
    assert build_overview["eval_covered"] == 2
    assert build_overview["missing_eval_chain_ids"] == ["c4"]

    verification_failures = report["build"]["verification_failures"]
    assert verification_failures[0]["failed_checks"] == "all_docs"
    assert verification_failures[0]["count"] == 1

    eval_overview = report["evaluation"]["overview"]
    assert eval_overview["total_results"] == 2
    assert eval_overview["agent_errors"] == 1
    assert eval_overview["inclusive_final_correct"] == 1
    assert eval_overview["exact_final_correct"] == 0
    assert eval_overview["company_name_leaks"] == 1

    buckets = {row["bucket"]: row for row in report["evaluation"]["failure_classes"]}
    assert buckets["agent_error"]["count"] == 1
    assert buckets["final_correct"]["count"] == 1
