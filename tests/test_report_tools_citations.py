from drbench.agents.drbench_agent.agent_tools.base import ResearchContext
from drbench.agents.drbench_agent.agent_tools.report_tools import ReportAssembler


class _FakeVectorStore:
    def __init__(self):
        self._results = [
            {
                "doc_id": "doc_finding",
                "similarity_score": 0.99,
                "content": "Synthetic finding that should not be preferred.",
                "metadata": {
                    "type": "research_finding",
                    "tool_used": "local_document_search",
                },
            },
            {
                "doc_id": "doc_internal",
                "similarity_score": 0.85,
                "content": "Average monthly store energy cost in Q3 2023 was 15%.",
                "metadata": {
                    "source_type": "local_document",
                    "file_path": "/tmp/q3-report.md",
                    "relative_path": "q3-report.md",
                },
            },
        ]

    def search(self, query: str, top_k: int = 10, use_semantic: bool = True):
        return self._results

    def get_document(self, doc_id: str):
        for result in self._results:
            if result["doc_id"] == doc_id:
                return {"metadata": result["metadata"], "content": result["content"]}
        return None


def test_concise_qa_report_uses_real_source_metadata_for_references(monkeypatch):
    prompts = []

    def fake_prompt_llm(*args, **kwargs):
        prompt = kwargs.get("prompt") if kwargs else args[-1]
        prompts.append(prompt)
        return (
            "ANSWER_1: 15%\n"
            "JUSTIFICATION_1: Found in the local report [DOC:doc_internal]\n\n"
            "ANSWER_FINAL: 15%\n"
            "JUSTIFICATION_FINAL: Same supporting report [DOC:doc_internal]"
        )

    monkeypatch.setattr(
        "drbench.agents.drbench_agent.agent_tools.report_tools.prompt_llm",
        fake_prompt_llm,
    )

    assembler = ReportAssembler(
        model="test-model",
        vector_store=_FakeVectorStore(),
        report_style="concise_qa",
    )
    context = ResearchContext(original_question="1. What percentage was reported?")

    report = assembler.generate_comprehensive_report(context)

    assert "doc_internal" in prompts[0]
    assert "doc_finding" not in prompts[0]
    assert "[^1]" in report
    assert "## References" in report
    assert "q3-report.md" in report
    assert "/tmp/q3-report.md" in report
    assert "Unknown source" not in report
