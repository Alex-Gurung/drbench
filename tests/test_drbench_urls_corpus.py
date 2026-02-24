import hashlib
import json
from pathlib import Path

import pytest

from making_dataset_2.drbench_urls import build_url_to_task_ids, doc_id_for_url, docid_for_url
from making_dataset_2.data_prep.chunk_web_drbench_urls import _emit_chunk_records, _parse_paragraph_blocks
from making_dataset_2.retrieval.hybrid import HybridSearcher


def test_docid_for_url_is_stable() -> None:
    url = "https://example.com/some/path?x=1"
    expected = "drbench_urls_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    assert docid_for_url(url) == expected
    assert doc_id_for_url(url).endswith("/" + expected)


def test_build_url_to_task_ids_maps_context_urls(tmp_path: Path) -> None:
    tasks_root = tmp_path / "tasks"
    (tasks_root / "DR0001").mkdir(parents=True)
    (tasks_root / "DR0002").mkdir(parents=True)

    url1 = "https://a.example.com/doc"
    url2 = "https://b.example.com/doc"

    (tasks_root / "DR0001" / "context.json").write_text(
        json.dumps({"url": {"url": url1}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (tasks_root / "DR0002" / "context.json").write_text(
        json.dumps({"url": {"url": url1}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (tasks_root / "SANITY0").mkdir(parents=True)
    (tasks_root / "SANITY0" / "context.json").write_text(
        json.dumps({"url": {"url": url2}}, ensure_ascii=False),
        encoding="utf-8",
    )

    mapping = build_url_to_task_ids(tasks_root)
    assert mapping[url1] == ["DR0001", "DR0002"]
    assert mapping[url2] == ["SANITY0"]


def test_chunking_emits_sequential_chunk_ids() -> None:
    doc = {
        "doc_id": "web/drbench_urls/drbench_urls_deadbeefdeadbeef",
        "url": "https://example.com/doc",
        "industry": "retail",
        "domain": "crm",
        "seed_date": "2024-01-01",
        "task_ids": ["DR0003"],
        "title": "Example Title",
        "text": (
            "# Example Title\n\n"
            "Para one with some words.\n\n"
            "## Section A\n\n"
            + ("word " * 120).strip()
            + "\n\n"
            "## Section B\n\n"
            + ("word " * 120).strip()
        ),
    }
    blocks = _parse_paragraph_blocks(doc["text"])
    recs = _emit_chunk_records(doc=doc, blocks=blocks, target_words=60, overlap_words=10, min_words=20)
    assert len(recs) >= 2
    assert recs[0]["chunk_id"].endswith("#0001")
    assert recs[1]["chunk_id"].endswith("#0002")
    assert all(r["doc_id"] == doc["doc_id"] for r in recs)
    assert all(r["meta"]["web_pool"] == "drbench_urls" for r in recs)


def test_bm25_retrieves_relevant_doc(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    doc_a = "web/drbench_urls/drbench_urls_aaaaaaaaaaaaaaaa"
    doc_b = "web/drbench_urls/drbench_urls_bbbbbbbbbbbbbbbb"
    rows = [
        {
            "chunk_id": f"{doc_a}#0001",
            "doc_id": doc_a,
            "source_type": "web",
            "text": "This chunk is about alpha widgets and only alpha widgets.",
            "offsets": {"start": 0, "end": 10},
            "meta": {"web_pool": "drbench_urls"},
        },
        {
            "chunk_id": f"{doc_b}#0001",
            "doc_id": doc_b,
            "source_type": "web",
            "text": "This chunk discusses beta gadgets exclusively.",
            "offsets": {"start": 0, "end": 10},
            "meta": {"web_pool": "drbench_urls"},
        },
    ]
    chunks_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    searcher = HybridSearcher(chunks_path=chunks_path, dense_index_path=None)
    hits = searcher.search("alpha widgets", mode="bm25", k=1)
    assert hits, "Expected at least one hit"
    assert hits[0].doc_id == doc_a

