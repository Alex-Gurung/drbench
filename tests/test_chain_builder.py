import sys
from pathlib import Path
from types import SimpleNamespace

from making_dataset_2.data_loading import LocalDoc, Secret
from making_dataset_2.pipeline import chain_builder
from making_dataset_2.types import Chain, ChainState, HopRecord


class _FakeEntityIndex:
    def __init__(self, _nlp, docs):
        self.docs = docs

    @classmethod
    def load(cls, _cache_path, all_docs, nlp=None):
        return cls(nlp, all_docs)

    def save(self, _path):
        return None


def test_chain_builder_scopes_local_seed_pool_by_company_metadata(tmp_path: Path, monkeypatch):
    alpha_doc_id = "local/DR0001/files/alpha.md"
    beta_doc_id = "local/DR0006/files/beta.md"

    doc_lookup = {
        alpha_doc_id: LocalDoc(
            doc_id=alpha_doc_id,
            text="Alpha company planning document with enough text to exceed the minimum length." * 4,
            chunk_ids=["alpha-1"],
            meta={"task_id": "DR0001", "company_name": "Alpha Co"},
        ),
        beta_doc_id: LocalDoc(
            doc_id=beta_doc_id,
            text="Beta company planning document with enough text to exceed the minimum length." * 4,
            chunk_ids=["beta-1"],
            meta={"task_id": "DR0006", "company_name": "Beta Co"},
        ),
    }
    eligible = [
        Secret(
            chunk_id="alpha-1",
            doc_id=alpha_doc_id,
            question="What is the Alpha code name?",
            answer="Orion",
            secret_type="metric",
            justification="",
            quote="",
            doc_only_check={"with_doc": True},
        ),
        Secret(
            chunk_id="beta-1",
            doc_id=beta_doc_id,
            question="What is the Beta code name?",
            answer="Atlas",
            secret_type="metric",
            justification="",
            quote="",
            doc_only_check={"with_doc": True},
        ),
    ]

    captured = {}

    def fake_select_seed(secrets, lookup, **kwargs):
        captured["company"] = kwargs["company"]
        captured["doc_ids"] = {s.doc_id for s in secrets}
        secret = secrets[0]
        doc = lookup[secret.doc_id]
        hop = HopRecord(
            hop_number=1,
            hop_type="L",
            question=secret.question,
            answer=secret.answer,
            doc_id=secret.doc_id,
            doc_text=doc.text,
        )
        return ChainState(
            pattern=kwargs["pattern"],
            hop_history=[hop],
            global_question=secret.question,
            global_answer=secret.answer,
            used_doc_ids={secret.doc_id},
            task_id=doc.meta["task_id"],
            company=kwargs["company"],
        )

    def fake_build_one_chain(**kwargs):
        state = kwargs["state"]
        return Chain(
            chain_id="chain-1",
            pattern=state.pattern,
            hop_history=state.hop_history,
            global_question=state.global_question,
            global_answer=state.global_answer,
            metadata={"complete": False, "elapsed_seconds": 0.0, "n_jumps": 0, "llm_calls": 0},
        )

    args = SimpleNamespace(
        pattern="LW",
        patterns=None,
        n=1,
        output=str(tmp_path / "chains.jsonl"),
        model="dummy-model",
        base_url=None,
        api_key=None,
        secrets=str(tmp_path / "secrets.jsonl"),
        chunks_local=str(tmp_path / "chunks_local.jsonl"),
        chunks_web=[str(tmp_path / "chunks_web.jsonl")],
        task=None,
        company=None,
        spacy_model="fake_spacy_model",
        retrieval_mode="none",
        retrieval_k=10,
        workers=1,
        min_max_tokens=None,
        no_trace=True,
        resume=False,
        search_url=None,
        seed=0,
        verbose=False,
    )

    monkeypatch.setattr(chain_builder, "_parse_args", lambda: args)
    monkeypatch.setattr(chain_builder, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(chain_builder, "load_chunks_local", lambda _path: [])
    monkeypatch.setattr(chain_builder, "build_doc_lookup", lambda _chunks: doc_lookup)
    monkeypatch.setattr(chain_builder, "load_secrets", lambda _path: eligible)
    monkeypatch.setattr(chain_builder, "filter_seed_secrets", lambda secrets, lookup, **kwargs: eligible)
    monkeypatch.setattr(chain_builder, "_load_web_docs", lambda _paths: {})
    monkeypatch.setattr(chain_builder, "EntityIndex", _FakeEntityIndex)
    monkeypatch.setattr(chain_builder, "LLMClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(chain_builder, "select_seed", fake_select_seed)
    monkeypatch.setattr(chain_builder, "build_one_chain", fake_build_one_chain)
    monkeypatch.setitem(sys.modules, "spacy", SimpleNamespace(load=lambda _name: object()))

    rc = chain_builder.main()

    assert rc == 0
    assert captured["company"] == "Alpha Co"
    assert captured["doc_ids"] == {alpha_doc_id}
    assert (tmp_path / "chains.jsonl").exists()
