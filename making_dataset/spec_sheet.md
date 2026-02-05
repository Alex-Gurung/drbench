# Local+Web Multi‑Hop QA Dataset Generator (Privacy‑Leakage Benchmark)

## Objective
Create a **verifiable, multi‑hop question–answer dataset** over:
- **scoped local “vault(s)”** built from DRBench files (ignore DRBench’s original task prompts)
- a **static external corpus** (offline “web datastore,” e.g., BrowseComp‑Plus or a frozen snapshot)

Primary research use: evaluate **privacy leakage via external search queries** made by agents, including **mosaic leakage** across many tasks for the same vault.

### Local Vault Scopes (Important)
For easier evaluation with existing DRBench code, the local vault is **not one pooled mega‑vault by default**. Instead we support two first-class scope styles:
- **Per‑DRBench‑task vault**: one vault per `DRxxxx` task directory (e.g., `DR0001` only).
- **Per‑company vault**: one vault per company (merge all `DRxxxx` tasks belonging to that company).

Each vault scope corresponds to a distinct `workspace_id` (or a set of `workspace_id`s) and has its own local retrieval index / neighbor cache.
The web corpus is shared across all scopes.

---

## Inspirations (what we borrow)
- **InfoSeek**: generate **multi‑hop trees** with explicit hop structure and quality filtering, but **replace hyperlink (href) traversal with retrieval-based transitions**.
- **DeepDive**: **random walk/path sampling** idea to create long multi‑hop reasoning chains (we do it on retrieval-induced neighbors rather than a true KG).
- **BrowseComp‑Plus**: use a **fixed external corpus** for reproducible “web search” without live APIs.
- **DRBench**: use its heterogeneous local artifacts (PDF/CSV/etc.) as the private corpus.

---

## Deliverables
### 1) Dataset JSONL
Each line is one task:
- `workspace_id`: identifies the local vault scope (per‑task or per‑company)
- `mode`: `mixed | local_only | web_only`
- `question`: natural language
- `answer`: verifiable and short (string/number/boolean/short JSON)
- `tree`: explicit multi-hop structure (nodes/edges with provenance)
- `gold`: evidence pointers + optional compute program
- `privacy`: sensitive span inventory + per-task “required vs unnecessary” sensitive facts

### 2) Indices + optional edge cache
- BM25 index and dense index over **chunked text**
- optional “neighbor cache” (top‑K related chunks per chunk) for faster tree generation

---

## Key design rules (must satisfy)
### Verifiability
- Answers must be **deterministically checkable** from stored evidence:
  - exact span match, table cell match, or computed from extracted values + constants
- Avoid free-form reports.

### Question quality (high priority)
- Questions must **not** mention: local, web, internal, external, document, excerpt, chunk, corpus, snippet, evidence.
- Answers should typically be **entity strings or a short sentence**, not long multi-part outputs.
- Prefer **variable reasoning styles** (entity linking, temporal constraints, alias/renaming, “the X of the Y…”) over a single compute template (e.g., avoid overusing “difference of two numbers” tasks).

### Multi-hop structure
- Trees must have configurable hop count (e.g., 4–8).
- Allow **multiple local hops** and **multiple web hops** back-to-back (no bipartite constraint).

### Local↔Web interdependence (for `mixed`)
- Enforce at least one **dependency** where a later hop’s retrieval query/choice depends on info from earlier hops across corpora (avoid separable “local then web then merge”).

### Baselines
Generate the same schema in three modes:
- `local_only`: retrieval restricted to local index
- `web_only`: retrieval restricted to external index
- `mixed`: retrieval can use either index at each hop

---

## System architecture (what to implement)
### A) Data prep (chunk store)
1. Ingest DRBench files → extract text to chunks (start with easy: JSON/CSV/text; add PDF later).
2. Ingest external corpus docs → chunk similarly.
3. Write a canonical `chunks.jsonl`:
   - `chunk_id`, `doc_id`, `source_type` (`local|web`), `text`, optional `meta`.

4. Split local chunks + local secret inventory into scopes:
   - per‑task (`DRxxxx`)
   - per‑company (merged across tasks for the same company)

### B) Retrieval (hybrid)
Implement `retrieve(query, k, filters)` over the chunk store:
- Stage 1: **BM25** top `K_bm25` candidates (typical: 200)
- Stage 2: **dense rerank** of BM25 candidates → top `k` (optional; used for web when available)
- Optional: sampling from top‑N to add controlled noise/diversity.

Concrete implementation notes (current):
- Local BM25 is an in-memory BM25 index over scoped local chunks.
- Web BM25 uses the prebuilt BrowseComp‑Plus Lucene index (Pyserini).
- Web dense uses the prebuilt BrowseComp‑Plus FAISS shards with `Qwen/Qwen3-Embedding-0.6B` query encoding (EOS pooling + L2 normalize).
- For `bm25_rerank_dense`: we do BM25 → take top `web_bm25_candidates_k` docids (default 200) → dense rerank that candidate list.
- For `corpus=all`: we fuse local and web ranked lists via Reciprocal Rank Fusion (RRF) to avoid score-scale mismatch.

### C) Optional neighbor cache (precompute edges)
- For each chunk, store top‑K neighbors under:
  - `LL` (local→local), `LW` (local→web), `WW` (web→web)
- Recommended: compute `LL` + `LW` up-front; `WW` lazily/on-demand or doc-level only.

### D) Tree generator (no hrefs)
Replace “follow hyperlink” with “retrieve/select next chunk”:
- Choose a start node (local or web) based on the mode and scope.
- For each hop:
  - build a retrieval query from current state (see below),
  - retrieve candidates from the allowed index (based on `mode`),
  - pick next node (deterministic or sampled),
  - add an edge with a short natural-language `claim`.
- Terminate when hop count reached and answer is computable/extractable.

#### Make “hopping” explicit
At generation time we treat each hop transition as:
1. **State extraction**: extract a “bridge key” from the current node (usually an entity string or a short phrase).
2. **Query construction**: build `query = f(<bridge key> + <goal hint>)` (MVP: keyword/entity extraction).
3. **Retrieval**:
   - local scope: BM25 over scoped local chunks
   - web: BM25 or BM25→dense rerank over BrowseComp‑Plus
4. **Selection**: choose the next node from the top‑N candidates:
   - deterministic: rank‑1
   - sampled: uniform from top‑N (to add diversity)
5. **Edge recording**: store the query used and the selected `chunk_id`/`doc_id`.

For `mixed`, we also enforce at least one explicit cross‑corpus dependency by requiring that a bridge key extracted from a local hop is used directly in a web hop query (or vice‑versa).

**Query generation (MVP)**:
- keyword/entity extraction from current chunk + goal schema.
**Extension**:
- LLM proposes 3–10 search queries per chunk; use them offline to improve neighbor cache.

### E) Validators / filters (prefer deterministic)
Reject tasks that fail:
- hop count, mode constraints (all nodes local-only/web-only)
- missing evidence pointers
- answer not recomputable / not extractable
- `mixed` tasks that don’t show cross-corpus dependency (heuristic check)

(LLM-as-judge is optional and should not be required for the MVP.)

### F) Privacy metadata extraction
From local chunks, build `secret_inventory`:
- numeric KPIs/metrics, IDs/emails, names (if present), etc.
For each task:
- mark `required_secrets` vs `unnecessary_secrets` (based on gold evidence set).

This enables later evaluation from **external query logs** only.

---

## Suggested folder layout
- `data_prep/` (extract → chunk)
- `index/` (BM25 + dense build/load)
- `edges/` (neighbor cache build/load; lazy cache)
- `generate/` (tree generator + templates)
- `validate/` (deterministic validators)
- `schemas/` (JSON schema definitions + examples)
- `outputs/` (jsonl datasets per mode)

---

## MVP milestones
1. Build chunk store for local + external; hybrid retrieval works.
2. Generate **100–500** tasks per mode (`local_only`, `web_only`, `mixed`) with hop≥4.
3. Deterministic validation: recompute answers from evidence.
4. Privacy metadata present in dataset items (secret inventory + required/unnecessary tags).

---
