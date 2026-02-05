# Local+Web Multi-Hop QA Dataset Generator — Implementation Plan

Date: 2026-02-03
Owner: Codex
Scope: Combine DrBench local vault (DR0001–DR0015) with BrowseComp‑Plus web corpus using prebuilt web indexes. Exclude SANITY0 and all qa_dict.json content. Use DrBench PDF/office extraction pipeline. Log all LLM token usage by stage.

## Status (as of 2026-02-04)
What exists and has been exercised end-to-end with small smoke runs:
- Local chunking: `outputs/chunks_local.jsonl` (~1457 chunks).
- Web chunking aligned to BrowseComp indexes: `outputs/chunks_web.jsonl` (100,195 docs; doc-level).
- Unified retrieval: local BM25 + web BM25 (Pyserini) + optional web dense (FAISS shards).
- Neighbor cache for local-only walks: `outputs/local_neighbors.jsonl`.
- Secret inventory with “doc-only” verification: `outputs/secret_inventory.jsonl` + per-run logs in `outputs/logs/<run_id>/`.
- Dataset generators:
  - Local-only: `generate/local_multihop_dataset.py`
  - Web-only: `generate/web_only_dataset.py`
  - Mixed numeric POC (3-way ablation): `generate/mixed_dataset_poc.py`
  - Mixed entity POC: `generate/mixed_entity_dataset_poc.py`
- Validators / inspection:
  - `validate/local_dataset_validate.py`
  - `validate/pretty_print_dataset.py`
  - `visualize.py` → `outputs/visualizer.html`

## Answerability Checks (“Ablations”)
We use two kinds of ablation-style checks to ensure questions actually require the intended evidence.

### A) Secret Inventory Doc-Only Check (local-only)
Implemented in: `generate/privacy_tagger.py` (enabled by default via `--doc-only-check`).

For each candidate secret item (a Q/A the model proposed for a local chunk), we run two fresh LLM calls:
- With-doc: answer the question using ONLY the chunk text (`stage=privacy_doc_answer`).
- No-doc: answer the same question with NO document access (`stage=privacy_nodoc_answer`).

We keep the item only if:
- With-doc != `NOT_ANSWERABLE`
- No-doc == `NOT_ANSWERABLE`

This is designed to filter out “guessable” questions (e.g., identity mapping or convention-based answers).

### B) Mixed Numeric 3-Way Corpus Ablation (needs both)
Implemented in: `generate/mixed_dataset_poc.py` (enabled by default via `--ablation-check`).

We synthesize a question whose answer is `abs(local_percent - web_percent)` and then verify:
- With both excerpts: must be answerable and numerically correct (`stage=mixed_answer_with_both`).
- Local-only: must be `NOT_ANSWERABLE` (`stage=mixed_answer_local_only`).
- Web-only: must be `NOT_ANSWERABLE` (`stage=mixed_answer_web_only`).

This enforces that the task really requires BOTH corpora (not solvable from one side alone).

### Notes
- Local-only dataset generation (`generate/local_multihop_dataset.py`) does NOT re-run ablations: it uses a final-hop secret Q/A that already passed the doc-only check in `secret_inventory.jsonl`.
- Web-only tasks (`generate/web_only_dataset.py`) are taken from BrowseComp-Plus; we optionally require the answer to appear as an exact substring in at least one hop doc (`--require-answer-span`).
- Mixed entity POC currently does not run a 3-way ablation; it relies on:
  - local seed secret being doc-only (from `secret_inventory.jsonl`)
  - web sub-QA being grounded (answer must appear verbatim in a passed excerpt)
  - TODO: add an explicit “needs both” ablation for entity tasks.

## Goals
- Build a verifiable multi-hop QA dataset with explicit trees and evidence.
- Enable privacy-leakage evaluation from external search queries.
- Use BrowseComp‑Plus prebuilt BM25 + Qwen3 dense indexes for web retrieval (default: `qwen3-embedding-0.6b`).
- Use vLLM Qwen3‑30B‑A3B‑Instruct‑2507 for local secret inventory extraction, with stage token accounting.

## Constraints (explicit)
- Exclude SANITY0 entirely.
- Do not ingest qa_dict.json or eval.json for corpus content.
- Local PDFs and docs must be extracted using DrBench’s ContentProcessor (no shortcuts).
- Web indexes are reused; no re-embedding required.
- Privacy tagging is local‑only and treats all names as private.
- Fail loud if token usage is missing from any LLM response.

## Data Sources
### Local vault
- Path: `nice_code/drbench/drbench/data/tasks/DR0001`–`DR0015` under `files/`
- File types: .md, .pdf, .csv, .json/.jsonl, .docx, .pptx, .xlsx, etc.
- Extraction: `drbench/agents/drbench_agent/agent_tools/content_processor.py`

### Web corpus
- Decrypted data: `BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl`
- Prebuilt indexes:
  - BM25: `BrowseComp-Plus/indexes/bm25/`
  - Dense: `BrowseComp-Plus/indexes/qwen3-embedding-0.6b/` (default; 4B/8B also available)

## Outputs
- `outputs/chunks.jsonl`
- `outputs/local_only.jsonl`
- `outputs/web_only.jsonl`
- `outputs/mixed.jsonl`
- `outputs/logs/<run_id>/llm_generations.jsonl`
- `outputs/logs/<run_id>/token_report.json`
- `outputs/logs/<run_id>/token_report.md`

## Commands
Use `/home/toolkit/.mamba/envs/vllm013/bin/python` for all runs.
If you already activated the `vllm013` environment, plain `python` is fine.

### Start vLLM server (background)
`VLLM_API_URL` is exported in `/home/toolkit/.bashrc` and `/home/toolkit/.bash_profile` sources it, so `bash -lc ...` runs also pick it up automatically.

```
source /home/toolkit/.bashrc

mkdir -p /home/toolkit/nice_code/drbench/logs
export VLLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
export VLLM_LOG_FILE="/home/toolkit/nice_code/drbench/logs/vllm_$(date +%Y%m%d_%H%M%S).log"

# Starts vLLM in the background and writes the PID to stdout (captured by nohup here).
nohup /home/toolkit/nice_code/drbench/scripts/start_vllm.sh >/dev/null 2>&1 & disown
```

Optional quick check:
```
curl -s http://localhost:8000/v1/models | head
```

### Local chunking
```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/data_prep/chunk_local.py \
  --prefer-md \
  --min-words 300
```

### Web chunking (BM25 contents as authoritative)
Recommended: use the cached `browsecomp-plus-corpus` so `chunks_web.jsonl` matches the prebuilt web indexes (BM25 + dense), which have **100,195** docs.

```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/data_prep/chunk_web.py \
  --source corpus_cache
```

Smoke (small):
```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/tests/smoke_chunk_web.py
```

### Merge chunk store
```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/data_prep/merge_chunks.py
```

### Web chunk size audit (detect overly large docs)
```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/validate/web_chunk_audit.py
```

### Local chunk size audit (detect overly short/large chunks)
```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/validate/local_chunk_audit.py
```

### Secret inventory (LLM)
```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/generate/privacy_tagger.py \
  --input /home/toolkit/nice_code/drbench/making_dataset/outputs/chunks_local.jsonl
```

Notes:
- By default, `privacy_tagger.py` runs a “doc-only” verification pass: it keeps a secret Q/A only if the model can answer *with* the chunk text and cannot answer *without* the chunk (`NOT_ANSWERABLE`).
- You can disable this for faster iteration with `--no-doc-only-check`.
- TODO: Consider forcing the with-doc verifier to output an extractive/short answer span (instead of a full sentence).
- NOTE: The script writes to a `*.partial` file first and only replaces `secret_inventory.jsonl` on success (to avoid clobbering a good inventory on a failed run).

### Split pooled local vault into scopes (per-task / per-company)
This converts the pooled local vault into smaller local corpora so the downstream datasets can be evaluated with existing DRBench per-task / per-company tooling.

Writes:
- `outputs/scopes/task/<DRxxxx>/chunks_local.jsonl`
- `outputs/scopes/task/<DRxxxx>/secret_inventory.jsonl`
- `outputs/scopes/company/<company>/chunks_local.jsonl`
- `outputs/scopes/company/<company>/secret_inventory.jsonl`

```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/data_prep/split_scopes.py \
  --chunks /home/toolkit/nice_code/drbench/making_dataset/outputs/chunks_local.jsonl \
  --secrets /home/toolkit/nice_code/drbench/making_dataset/outputs/secret_inventory.jsonl \
  --by task company
```

### Local neighbor cache (BM25, for tree walks)
```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/edges/build_local_neighbors.py \
  --chunks /home/toolkit/nice_code/drbench/making_dataset/outputs/chunks_local.jsonl \
  --output /home/toolkit/nice_code/drbench/making_dataset/outputs/local_neighbors.jsonl \
  --k 20
```

Scoped neighbor cache (example for one DRBench task):
```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/edges/build_local_neighbors.py \
  --chunks /home/toolkit/nice_code/drbench/making_dataset/outputs/scopes/task/DR0001/chunks_local.jsonl \
  --output /home/toolkit/nice_code/drbench/making_dataset/outputs/scopes/task/DR0001/local_neighbors.jsonl \
  --k 20
```

### Local-only multi-hop dataset (walk ends on a secret Q/A)
```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/generate/local_multihop_dataset.py \
  --chunks /home/toolkit/nice_code/drbench/making_dataset/outputs/chunks_local.jsonl \
  --neighbors /home/toolkit/nice_code/drbench/making_dataset/outputs/local_neighbors.jsonl \
  --secrets /home/toolkit/nice_code/drbench/making_dataset/outputs/secret_inventory.jsonl \
  --output /home/toolkit/nice_code/drbench/making_dataset/outputs/local_only.jsonl \
  --num-tasks 50 \
  --hops 4
```

Scoped local-only dataset (example for one DRBench task):
```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/generate/local_multihop_dataset.py \
  --chunks /home/toolkit/nice_code/drbench/making_dataset/outputs/scopes/task/DR0001/chunks_local.jsonl \
  --neighbors /home/toolkit/nice_code/drbench/making_dataset/outputs/scopes/task/DR0001/local_neighbors.jsonl \
  --secrets /home/toolkit/nice_code/drbench/making_dataset/outputs/scopes/task/DR0001/secret_inventory.jsonl \
  --output /home/toolkit/nice_code/drbench/making_dataset/outputs/scopes/task/DR0001/local_only.jsonl \
  --workspace-id drbench_task_DR0001_local_v1 \
  --num-tasks 20 \
  --hops 4
```

Entity-answer variant (prefers short entity-like strings):
```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/generate/local_multihop_dataset.py \
  --chunks /home/toolkit/nice_code/drbench/making_dataset/outputs/chunks_local.jsonl \
  --neighbors /home/toolkit/nice_code/drbench/making_dataset/outputs/local_neighbors.jsonl \
  --secrets /home/toolkit/nice_code/drbench/making_dataset/outputs/secret_inventory.jsonl \
  --output /home/toolkit/nice_code/drbench/making_dataset/outputs/local_only.entity.jsonl \
  --num-tasks 50 \
  --hops 4 \
  --answer-kind entity
```

### Validate local-only dataset (deterministic)
```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/validate/local_dataset_validate.py \
  --dataset /home/toolkit/nice_code/drbench/making_dataset/outputs/local_only.jsonl
```

Scoped validation (example for one DRBench task):
```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/validate/local_dataset_validate.py \
  --dataset /home/toolkit/nice_code/drbench/making_dataset/outputs/scopes/task/DR0001/local_only.jsonl \
  --chunks /home/toolkit/nice_code/drbench/making_dataset/outputs/scopes/task/DR0001/chunks_local.jsonl
```

### Smoke test: neighbors -> dataset -> validate (small)
Requires that `outputs/chunks_local.jsonl` and `outputs/secret_inventory.jsonl` already exist.
```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/tests/smoke_local_multihop.py
```

### Web-only dataset (BrowseComp-Plus tasks → web_only.jsonl)
This is a fast baseline that uses BrowseComp-Plus’s existing `query` + `answer` and wires them into our schema with 4+ web hops.

```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/generate/web_only_dataset.py \
  --num-tasks 50 \
  --hops 4
```

Entity-answer variant (filters out purely-numeric answers):
```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/generate/web_only_dataset.py \
  --num-tasks 50 \
  --hops 4 \
  --answer-kind entity
```

Smoke:
```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/tests/smoke_web_only_dataset.py
```

### Mixed dataset (POC, needs-both ablation enforced)
Generates a mixed task by combining:
- one **local secret** (from `secret_inventory.jsonl`)
- one **web percent** from a retrieved web doc (dense/BM25)

It asks a **single-answer computed question** (answer is a short string like `12 percentage points`) that requires BOTH values.

It enforces “needs both corpora” with a 3-way ablation check:
- (local+web) → answer
- (local only) → `NOT_ANSWERABLE`
- (web only) → `NOT_ANSWERABLE`

This script **uses vLLM** and **requires Java/Pyserini** (so run under `initmamba.sh`).

```
bash -lc 'source ~/.bashrc && source ~/initmamba.sh && /home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/generate/mixed_dataset_poc.py \
  --num-tasks 20 \
  --hops 4 \
  --web-backend bm25_rerank_dense'
```

Outputs:
- `outputs/mixed.jsonl`
- `outputs/logs/<run_id>/generation_report.json` (includes tokens/task estimate)
- `outputs/logs/<run_id>/generation_report.md`

Notes:
- By default, the generator asks vLLM to rewrite the dataset-facing question into a more natural/compositional style
  that avoids explicitly saying "internal/web/external". Disable with `--no-rewrite-question`.
- Web docs can be extremely long. `mixed_dataset_poc.py` does **not** require chunked web docs; it scans for a nearby `%`
  within keyword windows (see flags: `--max-web-chars`, `--web-window-chars`, `--web-max-windows`) and only passes short
  evidence excerpts to vLLM.

Smoke (requires vLLM running):
```
bash -lc 'source ~/.bashrc && source ~/initmamba.sh && /home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/tests/smoke_mixed_dataset_poc.py'
```

### Mixed dataset (Entity-Answer POC)
This is the current preferred mixed POC when you want **entity-string** answers instead of numeric computed diffs.

It uses:
- local secret inventory Q/A (to identify a seed entity)
- web retrieval seeded by that entity (cross-corpus dependency)
- vLLM to generate a grounded web sub-question + short entity answer from a web excerpt
- optional vLLM rewrite pass to make the final question more puzzle-like (disable with `--no-rewrite-question`)

Run:
```
bash -lc 'source ~/.bashrc && source ~/initmamba.sh && /home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/generate/mixed_entity_dataset_poc.py \
  --num-tasks 10 \
  --hops 4 \
  --web-backend bm25_rerank_dense \
  --output /home/toolkit/nice_code/drbench/making_dataset/outputs/mixed_entity.jsonl'
```

Smoke:
```
bash -lc 'source ~/.bashrc && source ~/initmamba.sh && /home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/tests/smoke_mixed_entity_dataset_poc.py'
```

### Pretty-print a dataset (Markdown)
Useful for quickly inspecting hop trees and evidence pointers.

```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/validate/pretty_print_dataset.py \
  --dataset /home/toolkit/nice_code/drbench/making_dataset/outputs/mixed.jsonl \
  --max-items 5
```

### Smoke test: web BM25 retrieval (requires Java)
Pyserini requires a working Java toolchain. If you see "Unable to find javac", run via:
```
bash -lc 'source ~/initmamba.sh && /home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/tests/smoke_web_bm25.py'
```

### Smoke test: unified local+web search (BM25)
```
bash -lc 'source ~/initmamba.sh && /home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/tests/smoke_unified_searcher.py'
```

### Smoke test: web dense retrieval (FAISS; no Java)
```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/tests/smoke_web_dense.py
```

### Smoke test: unified search with web dense enabled (requires Java)
```
bash -lc 'source ~/initmamba.sh && /home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/tests/smoke_unified_searcher_dense.py'
```

### BrowseComp retrieval eval (BM25 vs Dense vs Hybrid RRF)
Quick, offline sanity-check using BrowseComp `topics-qrels/*`.

```
bash -lc 'source ~/initmamba.sh && /home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/validate/browsecomp_retrieval_eval.py \
  --limit 50 \
  --ks 5,10,100'
```

### Token sanity check
```
/home/toolkit/.mamba/envs/vllm013/bin/python \
  /home/toolkit/nice_code/drbench/making_dataset/tests/token_usage_check.py \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507
```

## Logging & Token Accounting
### Logging directory
- Root: `nice_code/drbench/making_dataset/outputs/logs/<run_id>/`
- Each stage writes `llm_generations.jsonl` with per-request usage and metadata.

### Token usage requirements
- Every LLM call must log `prompt_tokens`, `completion_tokens`, and `total_tokens`.
- Missing usage → raise error (no silent fallback).

### Stage metadata fields
Each log record includes:
- `stage`: `privacy_inventory` | `query_gen` | `qa_synth` | `validator`
- `chunk_id` or `task_id`
- `model`, `provider`

### Token report
- `token_report.py` aggregates totals by stage.
- Outputs JSON + Markdown summary.

### vLLM token sanity test
- `tests/token_usage_check.py` sends a known prompt to vLLM and compares:
  - vLLM‑reported tokens vs HF tokenizer count (Qwen3 tokenizer)
- If mismatch exceeds threshold (e.g., >2%), **fail the run**.

## Pipeline Stages

```mermaid
flowchart LR
  A[DrBench files<br/>DR0001–DR0015] -->|ContentProcessor + semantic split| B[chunks_local.jsonl]
  C[BrowseComp-Plus corpus cache<br/>100,195 docs] -->|whole-doc alignment| D[chunks_web.jsonl]
  B --> E[chunks.jsonl]
  D --> E[chunks.jsonl]

  B --> F[local BM25 index<br/>(in-memory)]
  D --> G[web BM25 index<br/>(prebuilt Lucene)]
  D --> H[web dense index<br/>(prebuilt FAISS pickle shards)]

  B --> I[privacy_tagger.py<br/>secret_inventory.jsonl]
  B --> J[local_neighbors.jsonl]

  F --> K[local-only multihop generator]
  J --> K
  I --> K
  K --> L[local_only.jsonl]
  L --> M[validator]
```

## Task Scope (What “Local Vault” Means)
The spec sheet’s default is a **pooled local vault**: all DRBench tasks (DR0001–DR0015, excluding SANITY0) across all companies are ingested into a single chunk store. That means a generated **local-only** tree can (in principle) traverse:
- multiple DRBench tasks (different `DRxxxx`)
- multiple companies (Lee’s Market / MediConn / Elexion)

However, chunk metadata includes both:
- `meta.task_id` (e.g., `DR0003`)
- `meta.company` / `meta.company_name` (from `context.json`)

So we can generate variants later without re-extracting:
- **per-company vault**: constrain all local hops to a single `meta.company` and use a company-specific `workspace_id`
- **per-task vault**: constrain all local hops to a single `meta.task_id` (mostly useful for debugging)

Current default behavior:
- **Corpus**: pooled (all tasks + all companies).
- **Generators**: do not explicitly constrain by company/task *unless* you pass scoped inputs (see `data_prep/split_scopes.py`).

## Explicit QA Pipeline (How We Create One Task)
Below is the exact current “POC” pipeline for emitting a dataset item. The word “query” below refers to the retrieval query used by the generator to pick the next hop (stored in `tree.hops[*].edge.query` when applicable).

### A) Local-Only (`mode=local_only`)
Implemented in: `generate/local_multihop_dataset.py`
1. Input artifacts:
   - `outputs/chunks_local.jsonl`
   - `outputs/secret_inventory.jsonl` (doc-only verified secret Q/A per chunk)
   - `outputs/local_neighbors.jsonl` (BM25 neighbor cache)
2. Choose a **target chunk** that has at least one secret item.
3. Choose one secret item from that chunk:
   - `task.question` := secret `question`
   - `task.answer` := secret `answer`
4. Sample a multi-hop **path** that ends at the target chunk using the neighbor cache.
5. Deterministic evidence:
   - Find `answer` as a substring in the target chunk text and store `gold.answer_char_start/end`.
6. Privacy tags:
   - `required_secrets` contains the final secret Q/A used for the answer.
   - `unnecessary_secrets` contains other secrets from other local hops (if any).
7. Write a JSONL task record.

Note: Local-only trees currently do **not** store per-hop `edge.query` because the hop transitions come from the precomputed neighbor cache (built with “doc-as-query” BM25).

### B) Web-Only (`mode=web_only`)
Implemented in: `generate/web_only_dataset.py`
1. Stream BrowseComp-Plus tasks from `BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl`.
2. Take `task.question := rec.query` and `task.answer := rec.answer`.
3. Pick `hops` web docs from that task’s `gold_docs + evidence_docs` and store them as `tree.hops`.
4. Optional deterministic evidence:
   - Require that `answer` appears as an exact substring in at least one hop doc (`--require-answer-span`).
5. Write a JSONL task record.

### C) Mixed Numeric POC (`mode=mixed`, percent diff)
Implemented in: `generate/mixed_dataset_poc.py`
1. Pick a local secret item whose answer contains a percentage.
2. Use the local secret’s question (minus company names) as a **web retrieval query** to pick candidate web docs.
3. Inside candidate web docs, select a percentage near overlapping query terms.
4. Compute `answer := abs(local_percent - web_percent)` formatted as `"<N> percentage points"`.
5. Enforce “needs both” with **3-way ablation** (both/local-only/web-only) using vLLM.
6. Build a 4-hop tree and store the generator’s retrieval queries in `tree.hops[*].edge.query` where relevant.
7. Write a JSONL task record + generation report (tokens/task).

### D) Mixed Entity POC (`mode=mixed`, entity-string answer)
Implemented in: `generate/mixed_entity_dataset_poc.py`
1. Pick a local secret item whose answer looks like an entity string.
2. Use that secret answer as the **seed entity**, and retrieve a web doc about it.
3. Ask vLLM to produce a grounded **web sub-QA** (answer must be a short entity span in the provided web excerpt).
4. Ask vLLM to rewrite the final benchmark question so it’s “puzzle-like” and does not leak corpus hints.
5. Build a 4-hop mixed tree. (TODO: add explicit “needs both” ablation for entity tasks.)
6. Write a JSONL task record + generation report (tokens/task).

### 1) Local extraction + semantic chunking
- Walk local files under `files/` for DR0001–DR0015.
- Skip `qa_dict.json` and `file_dict.json` as content.
- Use `ContentProcessor` for extraction.
- Prefer `.md` when it exists alongside a PDF; use PDF only when no `.md` is available.
- Use semantic chunking:
  - Markdown: split by headings, but **merge adjacent sections** so chunks are at least ~`--min-words` (default 300) when possible. Each section contributes a `Title > Heading > Subheading` marker line.
  - Non‑MD: chunk by semantic splitter (sentence/paragraph), target ~512 tokens, 10% overlap.
- Doc IDs must remain informative after splitting. Use a stable format such as:
  - `local/<task_id>/<subdir>/<filename>` (e.g., `local/DR0001/IN002_pdf/food-safety-compliance.md`)
  - Chunk IDs append a zero‑padded index (e.g., `local/DR0001/IN002_pdf/food-safety-compliance.md#0003`).
- Store in `chunks.jsonl` with `source_type=local`.

### 2) Web doc map (for index reuse)
- Parse decrypted JSONL and collect all docs from `gold_docs`, `evidence_docs`, `negative_docs`.
- Deduplicate by `docid`.
- Store `docid -> text` map used to resolve FAISS results without HF download.
- Web chunks use **BM25 contents** as authoritative text (whole‑doc alignment).
- Note: This keeps web docs unchunked to align with the prebuilt BM25 index.

### 3) Retrieval
- Local retrieval: build BM25 + dense (Qwen3‑4B) over local chunks.
- Web retrieval: reuse BrowseComp‑Plus BM25 + FAISS.
- Unified searcher supports `corpus=local|web|all` and fuses BM25 + dense via RRF.

### 4) Tree generation
- Create multi‑hop trees (4–8 hops).
- For `mixed`, enforce at least one cross‑corpus dependency:
  - Extract value/entity from hop N and require hop N+1 query to include it across corpora.
- Allow multiple local or web hops in a row.

### 5) QA synthesis
- Deterministic templates to produce verifiable questions.
- Answer types: exact span, table cell, computed.
- Evidence pointers stored with char offsets into chunk text.

### 6) Privacy tagging (LLM‑assisted)
- Use vLLM Qwen3‑30B‑A3B‑Instruct‑2507 to generate **secret Q/A items** from **local chunks only**.
- All names are treated as private.
- Privacy inventory requests run concurrently (default `--concurrency 16`) to allow vLLM server batching.
- Model output format is block-based:
  `Question: ...`
  `Answer: ...`
  `Type: ...`
  `Justification: ...`
- A `Quote:` line is optional (legacy); we do not require span-anchored questions.
- Answers are grounded by checking that the answer text appears in the chunk (and via doc-only verification).
- Post-filters drop ungrounded or non-document-only items (e.g., email literals in questions).
- Guidelines are documented in `nice_code/drbench/making_dataset/PRIVACY_GUIDELINES.md`.
- Required secrets: spans used in evidence/compute inputs.
- Unnecessary secrets: spans in retrieved local chunks not used for answer.

### 7) Validation
Reject tasks that fail:
- hop count 4–8
- mode constraints
- missing or unverifiable evidence
- missing privacy metadata when local data is involved
- `mixed` without cross‑corpus dependency

## Milestones
1) Chunk store created for local + web (sanity checked).
2) Retrieval works for local/web/mixed.
3) 100–500 tasks per mode with validation pass.
4) Privacy tagging complete with token accounting.

## Open Questions
Execution notes:
- Use existing vLLM API configuration (via `VLLM_API_URL`) for runs.
- Run experiments with `/home/toolkit/.mamba/envs/vllm013/bin/python`.

## TODOs / Things To Test Next
High-value next steps (kept intentionally small and test-driven):
1) Mixed entity “needs both” ablation:
   - Add a check similar to the numeric POC: (both) answerable, (local-only) NOT_ANSWERABLE, (web-only) NOT_ANSWERABLE.
2) Question quality (InfoSeek/WebShaper style):
   - Strengthen rewrite prompts + add validators to avoid awkward “clue A / clue B” language and avoid leaking where to look.
3) Dense retrieval impact:
   - Run `validate/browsecomp_retrieval_eval.py` (BM25 vs dense vs hybrid RRF).
   - Add a small “custom queries” script comparing retrieval for our local chunks too (optional).
4) Full-run operational checks:
   - Run `privacy_tagger.py` on all ~1457 chunks with concurrency 16 and record wall time + token totals.
   - Spot-check the distribution of `secret_type` and the ratio of kept vs skipped items.

---

## Proposal: InfoSeek-Style Hierarchical CSP (HCSP) Task Generation (v2)
Date: 2026-02-04
Owner: (TBD)

### Summary
Our current generators produce tasks that are **verifiable** but often feel **templated/formulaic** because the dataset-facing `question` is usually a lightly-edited version of:
- a single secret inventory Q/A (local-only), or
- a BrowseComp-Plus query (web-only), or
- an explicit compute template (“absolute difference in percentage points…”) (mixed numeric POC).

This proposal reorients task creation around an **HCSP-style constraint tree**:
- A task is defined by a set of **constraints** (S₁…Sₙ) extracted from multiple hops.
- The solver must satisfy all constraints to identify the right target(s), then extract/compute the final answer.
- The dataset-facing question is generated by **blurring the parent node** (don’t name the target entity/doc directly; describe it via constraints).
- A quality gate verifies: **solvable with full information**, **not solvable with partial information**.

This should produce questions that read more like “puzzles” and less like templates, while keeping deterministic evidence pointers.

### Scope (what changes)
We implement a new “v2” path for *all three modes*:
- `local_only` HCSP generator
- `web_only` HCSP generator
- `mixed` HCSP generator

We keep the existing scripts as baselines and ship new scripts/output names (e.g., `*_hcsp.jsonl`) until v2 is stable.

### Assumption: re-chunking is allowed
Yes: we can recompute local chunks + neighbors + secret inventory if we want to change chunking parameters.
However, chunking is not the core requirement for HCSP; it’s an enabler for better constraint extraction.

Notes:
- `chunk_local.py` already supports “merge until at least ~`--min-words` (default 300) when possible”, but many source docs are intrinsically short (emails, slides, small tables), so some chunks will still be <300 words.
- If we do re-chunk, we must also rebuild:
  - `outputs/local_neighbors.jsonl`
  - `outputs/secret_inventory.jsonl`
  - any downstream datasets + validators that depend on char offsets.

### Definitions (for shared vocabulary)
- **Hop tree**: the provenance chain/subgraph over chunks (`tree.hops`) we already store.
- **Constraint**: a short statement grounded in a specific evidence span (chunk_id + char offsets), intended to be used as a “clue” in the question.
- **HCSP tree**: a small operator graph over constraints that defines how the answer is determined:
  - `INTERSECT` (entity/value must satisfy all constraints)
  - optional `COMPUTE` nodes (sum/diff/ratio/min/max/count) *when needed* for verifiability
- **Blur parent node**: do not reveal the target entity/doc name directly; use constraints to describe it.

### Privacy leakage controls (what we actually optimize)
The only hard requirement for “privacy leakage” tasks is:
- **Some local secret(s) must be required to solve** the task.

In practice, “required to solve” means the solver must **unroll** some number of secret facts during the solution (not merely encounter them in context).

Example (2 required secrets):
- Secret 1 (intermediate): the name/identifier of the internal group created in Q2.
- Secret 2 (final): the headcount/employee count for that group.
The benchmark question asks for the headcount but does not reveal the group name; the solver must first recover the group name (secret 1) and then use it to locate the headcount (secret 2).

To make this controllable, HCSP generation exposes two orthogonal knobs:
1) **Required secrets** (privacy leakage that is *necessary* for the answer)
   - Target: `min_required_local_secrets ≤ len(privacy.required_secrets) ≤ max_required_local_secrets`
   - Definition: a “required secret” is a secret fact that the solver must recover and use as:
     - the final answer evidence span, and/or
     - an intermediate value needed to retrieve/select the next hop (i.e., it appears in the *next-hop* `edge.query` or selection criteria).
   - Default policy (v2 MVP): required secrets come from **distinct hops** (distinct `chunk_id`s) so that unrolling is genuinely multi-hop.

2) **Unnecessary secrets** (privacy leakage that is *present in retrieved context* but not needed)
   - Target: `min_unnecessary_local_secrets ≤ len(privacy.unnecessary_secrets) ≤ max_unnecessary_local_secrets`
   - This is useful for “mosaic leakage” style evaluation: the agent may encounter extra secrets on the path even if not required.

Optional (explicit, tracked):
3) **Prompt leak budget** (privacy leakage in the dataset-facing question itself)
   - Default is to avoid including secret literals in the question (so leakage comes from retrieval, not from the prompt), but we can allow it:
     - `max_prompt_secret_literals` (default 0)
   - If non-zero, we record any leaked literals under `privacy.prompt_leaks` so downstream eval can filter/stratify.

### How “blurring” works (LLM-in-the-loop, like InfoSeek)
In InfoSeek, “blur” is not a deterministic `blur(node)` transform; it’s an LLM action that:
- mines additional “not strongly directional” constraints from the current page/chunk,
- adds them as constraint children (often with `entity=None`),
- rewrites the root question to reference constraints rather than naming the entity directly.

We mirror this by introducing an HCSP “builder loop” with actions:
- **Action A (Expand)**: pick a next hop via retrieval (local/web) and add it to the tree.
- **Action B (Blur / Constraint Mining)**: add 1–N constraint nodes grounded in the current hop’s text (with char offsets).
- **Action C (Rewrite Question)**: rewrite `task.question` to incorporate the constraint set naturally.

Python orchestrates retrieval + validation; the LLM authors the constraint text + question surface form.

### Target output schema (incremental, minimal diffs)
Keep existing top-level fields (`workspace_id`, `mode`, `question`, `answer`, `tree`, `gold`, `privacy`), and add:

- `tree.hops[*].edge.query` for **all** modes (including local-only) so we can audit “what query produced what hop”.
- `tree.hcsp` (new): explicit constraint/operator structure, with evidence pointers.
- `quality` (new): results of automated gates (deterministic + LLM-based + retrieval-based).

Proposed `tree.hcsp` shape (example):
```json
{
  "root": "A0",
  "nodes": {
    "S1": {"kind": "constraint", "text": "...", "evidence": {"chunk_id": "...", "char_start": 10, "char_end": 42}},
    "S2": {"kind": "constraint", "text": "...", "evidence": {"chunk_id": "...", "char_start": 88, "char_end": 120}},
    "A0": {"kind": "answer", "op": "INTERSECT", "inputs": ["S1","S2"], "answer_evidence": {"chunk_id": "...", "char_start": 500, "char_end": 504}}
  }
}
```

### Common generation pipeline (all modes)
1) **Sample a target** (answer + gold evidence)
   - Span-extractive answer (preferred) OR computed answer (fallback/secondary).
2) **Build hop tree**
   - Use existing retrieval / neighbor cache to get 4–8 hops.
   - Store `edge.query` for each hop (even when using neighbor cache, record the “doc-as-query” or extracted query).
   - For “required secrets” beyond the final answer, enforce that each intermediate secret is used to reach the next hop (e.g., by requiring the next hop to be retrieved with a query containing the intermediate secret’s answer string).
3) **Extract constraint candidates** (clues) from hop evidence
   - For each non-root hop (and optionally the root with answer-span masked), extract 1–3 candidate constraints.
   - Each constraint MUST be anchored to a span in a chunk (char offsets).
4) **Select an HCSP constraint set**
   - Choose 3–6 constraints that jointly:
     - do not contain the final answer string
     - do not contain banned literals (company name, doc_id, etc., depending on blur policy)
     - pass “necessity” checks (below).
5) **Synthesize a natural question** (LLM)
   - Input: constraint tree + the target ask (“what metric?”, “which entity?”, “what year?”), plus a banned-token list.
   - Output: one question that uses constraints naturally (no “clue A / clue B” phrasing), no corpus hints (“local/web”), no formula wording unless unavoidable.
6) **Quality assurance**
   - Deterministic checks (offsets, answer recompute, banned token leak).
   - LLM-based answerability checks (full info / partial info / no-doc).
   - Retrieval sensitivity checks (for INTERSECT-style tasks): removing a constraint should materially degrade retrieval to the correct evidence.
7) Emit dataset item + per-run report (tokens, acceptance, skip reasons).

### Quality gates (shared)
Deterministic:
- `answer` is verifiable from `gold` (exact span or compute inputs).
- All constraint spans are valid offsets in the referenced chunk.
- `question` must NOT contain the exact `answer` string (or normalized variants).
- “Banned literals” are now a **policy knob**:
  - optionally ban company names, doc titles/ids, file names, etc.
  - optionally allow some or all of these and record leaks under `privacy.prompt_leaks`.

LLM-based (vLLM):
- With full evidence snippets → answer must equal expected.
- With no evidence → must be `NOT_ANSWERABLE`.
- With partial evidence:
  - For `mixed`: “both required” ablation must pass (already done for numeric POC; add for entity POC + HCSP).
  - For HCSP-INTERSECT tasks: dropping a constraint’s evidence should yield `NOT_ANSWERABLE` or “multiple possible” (we can treat anything other than the exact answer as failure).

Retrieval-based (cheap proxy, especially for INTERSECT tasks):
- Run BM25 with the full synthesized question; the target evidence chunk (or doc) should be in top-K.
- Remove each constraint sentence from the question one at a time; the target rank should drop by ≥RANK_DELTA or fall out of top-K for at least M of the constraints.
  - This approximates “each constraint matters” without requiring a full symbolic solver.

### Local-only HCSP (`mode=local_only`)
Goal: multi-hop, puzzle-like question whose answer is verifiably anchored in the local vault, but the company/doc/entity is not named directly.

Target selection:
- Start from `secret_inventory.jsonl` (local secrets already doc-only verified).
- Prefer secret types that work well as “asks”: `kpi_numeric`, `dates`, `other_sensitive` (entity-like strings).

Hop tree:
- Use local neighbor cache OR dynamic BM25 retrieval.
- Ensure ≥1 hop is a “sibling” (same `doc_id`) when possible to increase coherent constraints.

Constraint extraction:
- Extract constraints from non-root hops + non-answer spans.
- Default blur policy for local-only:
  - ban `meta.company_name` and common aliases
  - ban literal filenames/doc_ids
  - ban the final answer string
  - allow *some* numeric/date facts only if they’re not the target answer (configurable).

HCSP styles (local-only):
1) **INTERSECT→EXTRACT** (preferred “InfoSeek-like”):
   - Constraints identify a specific report/section/initiative; answer is a metric from that target chunk.
2) **EXTRACT with intermediate secret(s)** (preferred for privacy control):
   - Final answer is extractive (string/number/entity).
   - One or more intermediate required secrets must be unrolled on distinct hops (e.g., entity/identifier → then attribute/value).
   - Avoid “compute/diff/ratio” tasks in the MVP unless we can keep questions natural.

### Web-only HCSP (`mode=web_only`)
We have two implementation options; start with A for speed.

A) **Seeded from BrowseComp-Plus tasks (fast path)**
- Use BrowseComp’s `query` + `answer` + `gold_docs/evidence_docs` as the starting point.
- Extract constraints from 2–5 evidence docs (excluding the doc where the answer span is found, if possible).
- Synthesize a new question via constraint tree and blur doc titles/entities where needed.
- Keep deterministic answer span requirement.

B) **Fully synthetic from the web corpus (slower path)**
- Sample a web doc and extract a target fact (entity/value).
- Retrieve supporting docs to extract additional constraints.
- Build HCSP tree and synthesize question.

### Mixed HCSP (`mode=mixed`)
Goal: tasks that *truly* require both corpora while staying natural/puzzle-like.

Targets:
- Prefer **entity/attribute** answers (InfoSeek-like) where:
  - local provides a seed entity or disambiguating constraint,
  - web provides the final answer span (or vice versa).
- Avoid computed “diff/ratio” tasks in the MVP.

Cross-corpus dependency requirements:
- At least one constraint from local and one from web in the HCSP tree.
- At least one retrieval query in hop N+1 must be generated from evidence in hop N (store in `edge.query`).

Quality gates (mixed-specific):
- 3-way ablation is mandatory:
  - with both excerpts → answerable
  - local-only → `NOT_ANSWERABLE`
  - web-only → `NOT_ANSWERABLE`

### Implementation plan (repo-level tasks)
New code (suggested files; adjust as needed):
1) `making_dataset/generate/hcsp/schema.py`
   - dataclasses / helpers for constraint nodes, compute nodes, evidence pointers.
2) `making_dataset/generate/hcsp/constraints.py`
   - extract candidate constraints from chunks (LLM + span offset enforcement).
3) `making_dataset/generate/hcsp/synthesize.py`
   - blur policy, banned-token enforcement, question generation prompt.
3a) `making_dataset/generate/hcsp/tree_builder.py`
   - hop tree construction (local/web/mixed) + cross-corpus bridge retrieval.
3b) `making_dataset/generate/hcsp/agent_loop.py`
   - InfoSeek-style builder loop (expand / blur / rewrite) that produces `tree.hcsp` + `task.question`.
4) `making_dataset/validate/hcsp_validate.py`
   - deterministic checks + vLLM answerability/ablation + retrieval sensitivity checks.
5) Generators:
   - `making_dataset/generate/local_hcsp_dataset.py`
   - `making_dataset/generate/web_hcsp_dataset.py`
   - `making_dataset/generate/mixed_hcsp_dataset.py`
6) Tests:
   - `making_dataset/tests/smoke_local_hcsp.py`
   - `making_dataset/tests/smoke_web_hcsp.py`
   - `making_dataset/tests/smoke_mixed_hcsp.py`

### Milestones (suggested execution order)
M0) Schema + validator scaffolding (1–2 days)
- Define `tree.hcsp` + `quality` fields and implement deterministic checks.
- Add a “no leaks” validator (answer literal + banned tokens).
- Add a small “LLM answer with evidence” harness we can reuse across modes.

M1) Local-only HCSP (2–4 days)
- Implement constraint extraction on local chunks (start with 1–2 constraints per hop).
- Implement blur policy (ban company name + doc ids + answer literal).
- Implement retrieval sensitivity check and a minimal partial-evidence ablation.
- Emit `outputs/local_only.hcsp.jsonl` and iterate until questions look good.

M2) Web-only HCSP seeded from BrowseComp (2–4 days)
- Rewrap BrowseComp tasks into HCSP form by extracting constraints from gold/evidence docs.
- Synthesize “blurred parent” questions (avoid doc titles when needed).
- Emit `outputs/web_only.hcsp.jsonl`.

M3) Mixed HCSP (3–6 days)
- Start from the existing mixed entity POC (it already has the right “shape”).
- Convert to HCSP constraint extraction + synthesis (not just rewrite).
- Add mandatory 3-way ablation for entity tasks.
- Emit `outputs/mixed.hcsp.jsonl`.

M4) Scale + measure (ongoing)
- Run 100–500 tasks/mode and track acceptance + failure modes.
- Add lightweight style metrics (question length distribution, repetition rate, banned-token violations, etc.).

### Rollout / A-B plan
1) Land HCSP v2 generators behind new output names (`local_only.hcsp.jsonl`, `web_only.hcsp.jsonl`, `mixed.hcsp.jsonl`).
2) Generate small samples (n=20 per mode) and inspect with `pretty_print_dataset.py` (update printer to show `tree.hcsp`).
3) Track acceptance + skip reasons; iterate prompts and constraint selection heuristics.
4) Once stable, decide whether to:
   - replace existing generators, or
   - keep both families (“templated baselines” + “HCSP v2”).

### Open design decisions (pick early)
- **Constraint eligibility**: are we allowed to include other local numeric/date facts in the question (besides the target answer), or do we restrict to “non-sensitive” clues?
- **Answer families per mode**: do we prioritize entity-string answers (more InfoSeek-like) or numeric KPI answers (closer to current privacy benchmark)?
- **Necessity definition**: do we require LLM partial-evidence `NOT_ANSWERABLE`, retrieval rank drop, or both?
- **Privacy knobs**: what are the default ranges for:
  - `min/max_required_local_secrets`
  - `min/max_unnecessary_local_secrets`
  - `max_prompt_secret_literals`?
- **Banned tokens**: do we always ban company names/doc titles, or do we allow them and just track via `privacy.prompt_leaks`?

### Main risks (and mitigations)
- **Constraint extraction is noisy** → require span offsets; keep 1–3 constraints/hop; discard low-confidence outputs.
- **“Puzzle-like” becomes “underspecified”** → enforce uniqueness proxies (retrieval sensitivity) and partial-evidence ablations.
- **Privacy leakage via question text** → add explicit leak checks; optionally store `privacy.prompt_leaks` if we intentionally allow some details.

### Success criteria (what “good” looks like)
- ≥70% of emitted tasks pass all deterministic + ablation checks (after tuning).
- Human spot-check: questions read natural and do not reveal the answer or explicit corpus hints.
- Mixed: ≥90% pass “needs both corpora” ablation.
- Retrieval sensitivity shows that at least 2 constraints per task are actually doing work (rank drops on removal).
