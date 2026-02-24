# Retrieval (v2)

This folder contains retrieval utilities for `making_dataset_2/`.

There are **two supported retrieval setups**:

1. **BrowseComp-mirrored (recommended)**: Pyserini BM25 + Qwen3 dense shards
2. **Pure-Python v2 (development-only)**: in-memory BM25 + `.npz` dense index

The BrowseComp-mirrored path is what you want if you care about:
- Matching the existing DRBench/BrowseComp retrieval behavior
- Higher quality BM25 than the simple in-memory implementation
- Using the same Qwen3 embedding model family as the rest of the repo

## BrowseComp-Mirrored Indexes (Recommended)

### 0) Prereq: build docs + chunks

```bash
/home/toolkit/.mamba/envs/vllm013/bin/python -m making_dataset_2.data_prep.fetch_drbench_urls
/home/toolkit/.mamba/envs/vllm013/bin/python -m making_dataset_2.data_prep.chunk_web_drbench_urls
```

### 1) Export chunk corpus as a Pyserini JsonCollection

```bash
/home/toolkit/.mamba/envs/vllm013/bin/python -m making_dataset_2.retrieval.export_chunks_pyserini
```

This writes `id/contents` JSONL docs under:
- `making_dataset_2/outputs/indexes/drbench_urls_bm25_collection/`

### 2) Build a Lucene BM25 index (Pyserini)

Pyserini needs Java (`javac`). In this container, run with:

```bash
bash -lc 'source ~/initmamba.sh && /home/toolkit/.mamba/envs/vllm013/bin/python -m making_dataset_2.retrieval.build_bm25_index_pyserini'
```

This writes the Lucene index directory:
- `making_dataset_2/outputs/indexes/drbench_urls_bm25/`

### 3) Build Qwen3 dense shards (BrowseComp-compatible)

```bash
/home/toolkit/.mamba/envs/vllm013/bin/python -m making_dataset_2.retrieval.build_qwen3_dense_shards \\
  --model Qwen/Qwen3-Embedding-4B \\
  --num-shards 1
```

This writes shard pickles:
- `making_dataset_2/outputs/indexes/drbench_urls_qwen3_dense/corpus.shard*_of_*.pkl`

Search-time query encoding uses the BrowseComp query prefix:
- `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:`

Passage encoding uses:
- `passage_prefix=""` (empty), mirroring BrowseComp-Plus `qwen3-embed.md`.

### 4) Evaluate (BM25 / dense / BM25->dense rerank)

```bash
bash -lc 'source ~/initmamba.sh && /home/toolkit/.mamba/envs/vllm013/bin/python -m making_dataset_2.eval.retrieval_eval_drbench_urls_browsecomp \\
  --bm25-index making_dataset_2/outputs/indexes/drbench_urls_bm25 \\
  --dense-index-glob making_dataset_2/outputs/indexes/drbench_urls_qwen3_dense/corpus.shard*_of_*.pkl \\
  --dense-model Qwen/Qwen3-Embedding-4B \\
  --mode bm25_rerank_dense'
```

## Pure-Python v2 Retrieval (Development-Only)

This path is handy when you don't want Java/Pyserini and just want a quick
smoke-test loop. It is not intended to perfectly mirror BrowseComp.

Dense index build (writes `.npz`):

```bash
/home/toolkit/.mamba/envs/vllm013/bin/python -m making_dataset_2.retrieval.build_dense_index \\
  --backend openai_compatible \\
  --model Qwen/Qwen3-Embedding-4B \\
  --base-url http://127.0.0.1:8000/v1
```

Eval:

```bash
/home/toolkit/.mamba/envs/vllm013/bin/python -m making_dataset_2.eval.retrieval_eval_drbench_urls --mode hybrid
```

