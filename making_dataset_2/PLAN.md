# Multi-Hop Chain Builder v2 — Detailed Pipeline Specification

Date: 2026-02-08

## 1. Goals

Build multi-hop QA chains that link **local enterprise documents** (L) with **public web documents** (W) using **iterative question composition**. Each chain is a sequence of hops where each question wraps the previous one:

```
Hop 1: Q1 → A1 (from Doc1)
Hop 2: Q2(slot_for_A1) → A2 (from Doc2)
Hop 3: Q3(slot_for_A2) → A3 (from Doc3)
...
```

The final question Q_n can only be answered by traversing all hops in order. If you don't know A1, you can't resolve Q2, so you can't get A2, so you can't resolve Q3, etc.

### Why we're rebuilding

The v1 pipeline (`making_dataset/generate/chain_builder.py`) produced **0/5 valid chains** across 4 iterations due to:

1. **Heuristic scoring couldn't catch hallucinations** — `score_bridge()` with proximity/slot/length heuristics still picked hallucinated facts (e.g., "Google developed Workday" from wrong document section). Heuristics are fundamentally the wrong tool for judging semantic quality.
2. **Numeric answers make terrible search queries** — "80%" or "5%" can't find relevant web documents; queries need to understand the *topic*, not just the raw answer.
3. **Model hallucinates across entities** — In multi-entity documents, model attributes facts from entity X's section to entity Y. Prompts didn't adequately guard against this.
4. **Validation too late** — Problems only discovered after the entire chain is built. No per-step quality gates.
5. **Hard to debug** — Complex interplay of query gen → search → bridge composition → scoring → validation made it hard to isolate which step was failing.

### Design principles for v2

1. **No heuristics, ever** — All quality judgments are made by the LLM with clear rubrics, or by simple deterministic checks (e.g., "answer != previous answer"). No numeric scoring, no proximity weights, no word-overlap ratios.
2. **Every LLM call is well-justified** — We don't care about minimizing calls; we care that each call has a clear purpose, a good prompt with rubric, and a well-defined output format. If we need 10 bridge attempts, that's fine.
3. **Better prompts instead of more code** — Clear rubrics with good/bad examples. Anti-hallucination instructions. Corpus-aware query generation.
4. **Test each step independently** — Each pipeline step has its own test harness. Only combine when each step works in isolation.
5. **Parallel generation, then select** — Generate bridges from multiple candidate docs in parallel, validate all survivors, then let the LLM pick the best one.

---

## 2. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUTS                                   │
│  - secret_inventory.jsonl (943 secrets, 15 tasks, 3 companies)  │
│  - chunks_local.jsonl (1,457 local chunks)                       │
│  - Web: BM25 index (Pyserini) + dense index (FAISS)            │
│  - Local: BM25 index + dense index (FAISS, to be built)        │
│  - LLM client (model-agnostic, e.g. vLLM, OpenAI-compat API)  │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  STEP 1: Select Seed Secret              │
│  (deterministic, no LLM)                 │
│  Pick a secret from inventory.           │
│  This gives us Q1, A1, Doc1.             │
│  Uses FULL DOCUMENT, not just chunk.     │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│  STEP 2: Generate Search Query               │
│  (1 LLM call)                                │
│  Turn chain context into a search query.     │
│  Prompt is CORPUS-AWARE: tells LLM whether   │
│  we're searching enterprise docs or public   │
│  web, so it can tailor the query.            │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│  STEP 3: Retrieve Candidate Documents        │
│  TWO-STAGE retrieval (both corpora):         │
│  Stage A: BM25 recall (k=200 candidates)     │
│  Stage B: Dense rerank (top-k from pool)     │
│  Filter out already-used doc_ids.            │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│  STEP 4: Compose Bridges (PARALLEL)          │
│  (1 LLM call per candidate, N in parallel)   │
│  Run bridge composition on top-N docs.       │
│  Each gets full chain history context.       │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│  STEP 5: Validate & Select Best Bridge       │
│  5a. Deterministic pre-checks (fast):        │
│      - Answer != prev? Prev answer NOT in Q? │
│  5b. LLM validation (1 call per survivor):   │
│      - Grounded in doc? Multi-hop? Natural?  │
│  5c. LLM selection (1 call):                │
│      - Pick best from all survivors          │
│  No survivors → back to Step 2 (retry)      │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  STEP 6: Update State & Repeat           │
│  (no LLM)                                │
│  Update chain history:                    │
│  - Add hop to per-hop Q/A list           │
│  - Update global composed Q/A            │
│  Repeat Steps 2-5 for each hop.          │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  STEP 7: Verify Complete Chain           │
│  (4 LLM calls)                           │
│  Test answerability under 4 conditions:  │
│  - No docs → NOT_ANSWERABLE             │
│  - First hop only → NOT_ANSWERABLE      │
│  - Last hop only → NOT_ANSWERABLE       │
│  - All hops → ANSWERABLE (correct)      │
│  Reject chain if any condition fails.    │
└──────────────────────────────────────────┘
```

---

## 3. Chain State: What We Track

As the chain grows, we maintain two parallel views:

### Per-hop Q/A list (local facts)
Each hop records the "local" question/answer at that step — what fact was established:
```
hop_history = [
  {hop: 1, type: "L", Q: "What HR platform does Lee's Market use?", A: "Workday", doc_id: "..."},
  {hop: 2, type: "W", Q: "When was Workday founded?", A: "2005", doc_id: "..."},
  {hop: 3, type: "L", Q: "What initiative did Lee's Market launch in 2005?", A: "Green Supply Chain", doc_id: "..."},
]
```

### Global composed question & answer
The composed question that wraps all hops — this is the final chain output:
```
global_question: "What initiative was launched in the same year the HR platform
                  used by Lee's Market was founded?"
global_answer: "Green Supply Chain"
```

Both views are passed to the bridge LLM so it can see:
- What individual facts have been established (hop_history)
- How they're currently composed (global_question)
- What needs to be extended next

---

## 4. Step-by-Step Specification

### Step 1: Select Seed Secret

**Purpose**: Pick a starting point — a local enterprise secret that will be the first hop of the chain.

**Inputs**:
- `secret_inventory.jsonl` — 221 document entries, each with nested secrets
- Pattern specification (e.g., `LW`, `LWL`, `LWLW`)
- Optional filters: `--task` (DR0001-DR0015), `--company`

**Process** (deterministic, no LLM):
1. Load all secrets from inventory
2. Apply filters (if specified):
   - `--task`: keep only secrets from that task
   - `--company`: keep only secrets from that company
3. Shuffle remaining secrets
4. For each secret, create initial state:
   - `Q = secret.question` (e.g., "What HR platform does Lee's Market use for onboarding?")
   - `A = secret.answer` (e.g., "Workday")
   - `Doc = full document text` (NOT just the chunk — the full document the secret came from)
   - `hop_type = "L"`
5. Initialize chain history:
   - `hop_history = [{hop: 1, type: "L", Q: Q, A: A, doc_id: ...}]`
   - `global_question = Q`
   - `global_answer = A`

**Output**: Initial `ChainState` with hop_history and global Q/A.

**Filtering rules**:
- Secret must have `doc_only_check = true` (verified answerable only with the document)
- Answer must be non-empty and 1-5 words
- Document text must be at least 200 characters

We do NOT filter by answer type (entity vs numeric) at this stage. Any secret is a valid seed — the query generation step (Step 2) handles both.

**Why full document, not chunk**: The secret inventory stores secrets at the chunk level, but chunks are narrow sections. The full document provides more context — other entities, metrics, and topics mentioned alongside the secret — which gives the bridge LLM more material to work with.

---

### Step 2: Generate Search Query

**Purpose**: Turn the current chain context into a search query that will find documents with topical overlap — documents where a bridge connection could exist. The prompt is **corpus-aware**: it tells the LLM whether we're searching a public web corpus or an enterprise knowledge base.

**Inputs**:
- `hop_history` — per-hop Q/A pairs so far
- `global_question` — the current composed question
- `global_answer` — the current answer
- `target_corpus` — `"web"` or `"local"` (from the chain pattern)

**Process** (1 LLM call):

**Prompt**:
```
Generate ONE search query to find documents that have topical overlap with the
current chain state. We want documents where a meaningful connection exists —
documents that discuss the current answer, related entities, or the broader topic.

CHAIN HISTORY (each hop is a fact established so far; together they form a
reasoning chain where each answer feeds into the next question):
{formatted_hop_history}

CURRENT GLOBAL STATE (the single composed question that requires ALL hops above
to answer, and its current answer):
- Question: {global_question}
- Answer: "{global_answer}"

TARGET CORPUS: {corpus_description}

{corpus_specific_instructions}

ANSWER TYPE GUIDE — classify the answer first, then build your query:
- PUBLIC ENTITY (Workday, Kubernetes, SAP, AWS): Include it in the query directly.
- PRIVATE IDENTIFIER (email, phone, internal URL, employee name): Do NOT include
  it. Search by the ROLE, TOPIC, or FUNCTION instead.
- NUMERIC METRIC (50%, 12 weeks, $2.3M): Do NOT search for the number. Search for
  industry BENCHMARKS or RESEARCH on the topic the metric describes.
- DATE / QUARTER (Q2 2024, January 2025): Do NOT search for the date. Search for
  the broader topic or initiative the date relates to.

EXAMPLES:

Chain: Q="What HR platform does Lee's Market use?" A="Workday"
Target: public web corpus
→ Answer type: PUBLIC ENTITY. Include it.
QUERY: Workday enterprise HR platform cloud adoption history

Chain: Q="What is the employee retention rate?" A="82%"
Target: public web corpus
→ Answer type: NUMERIC METRIC. Search topic, not number.
QUERY: employee retention rate retail industry benchmarks turnover statistics

Chain: Q="Who is the production manager?" A="michael.brown@elexion.com"
Target: public web corpus
→ Answer type: PRIVATE IDENTIFIER. Search by role/topic.
QUERY: manufacturing production manager role responsibilities industry trends

Chain: Q="What is the loyalty program spend increase?" A="50%"
Target: public web corpus
→ Answer type: NUMERIC METRIC. Search topic.
QUERY: retail loyalty program customer spending patterns member engagement

Chain: Q="When is the platform migration deadline?" A="Q2 2025"
Target: enterprise knowledge base
→ Answer type: DATE. Search the topic.
QUERY: enterprise software platform migration planning best practices timelines

Chain: Q="Who is the VP of store operations?" A="david.lee@leesmarket.com"
Target: enterprise knowledge base
→ Answer type: PRIVATE IDENTIFIER. Search by role.
QUERY: retail store operations management leadership organizational structure

RULES:
- The query should find documents where a CONNECTION to the current chain exists
- Focus on the TOPIC and DOMAIN, not the raw answer value
- DO NOT include: email addresses, personal names, phone numbers, internal URLs
- DO NOT include: fictional company names (Lee's Market, MediConn, Elexion) or
  their domains (leesmarket.com, elexionautomotive.com, mediconnsolutions.com)
- DO NOT include raw internal metrics with specific dates (e.g., "Q2 2024 25%")
- For private answers: think about what PUBLIC topic the answer relates to

Think step by step about what topics and entities in the chain would appear in
other documents, then put your final query inside <answer> tags:
<answer>your query here</answer>
```

**Corpus descriptions** (injected into `{corpus_description}` and `{corpus_specific_instructions}`):

For `target_corpus = "web"`:
```
corpus_description = "A public web corpus (~100K documents). Contains Wikipedia articles,
news, company pages, industry reports, and general reference content."

corpus_specific_instructions = """TIPS FOR WEB SEARCH:
- Use well-known public terms and proper nouns
- Include entity names, industry terminology, and factual keywords
- Think about what a Wikipedia article or news piece about this topic would contain"""
```

For `target_corpus = "local"`:
```
corpus_description = "An enterprise knowledge base containing internal documents from
retail, healthcare, and automotive companies. Contains reports, memos, presentations,
spreadsheets, and internal communications."

corpus_specific_instructions = """TIPS FOR ENTERPRISE SEARCH:
- Use business/industry terminology (KPIs, metrics, initiatives, vendors)
- Think about what internal reports or memos would discuss
- Include terms related to business operations, strategy, and tools"""
```

**LLM parameters**:
- Temperature: 0.3
- Max tokens: 300
- Stage: `"query_generation"`

**Output**: A single search query string.

**Post-processing**:
- Parse content inside `<answer>...</answer>` tags — this is the query
- If no `<answer>` tags found, fall back to last non-empty line
- Strip leading/trailing whitespace
- If empty after cleaning, fall back to using `global_answer` as the query directly

---

### Step 3: Retrieve Candidate Documents

**Purpose**: Find documents with topical overlap to the current chain state. Uses **two-stage hybrid retrieval**: BM25 + dense recall (union), then rank by dense cosine similarity.

**Inputs**:
- `query` — from Step 2
- `used_doc_ids` — documents already in the chain (to avoid reuse)
- Target corpus: `"web"` or `"local"` (from the chain pattern)

**Process**:

Two-stage hybrid per pool:
1. **Stage 1 — Broad recall**: BM25 retrieves top-200 by keyword match, dense retrieves top-200 by embedding similarity. Union the candidate sets.
2. **Stage 2 — Dense ranking**: Score all candidates by cosine similarity with the query embedding. Return top-k.

```python
# Each pool is a HybridSearcher instance
hits = pool.search(query, k=20, mode="hybrid", embedder=embedder)
```

**Web search — multi-pool**:
Web search queries two pools and merges results:
1. **drbench_urls pool** (27 fetched DrBench seed URLs, ~262 chunks) — high relevance, these are pages the tasks actually reference
2. **BrowseComp pool** (100K web docs) — broad coverage

```python
hits_drbench = drbench_urls_searcher.search(query, k=20, mode="hybrid", embedder=embedder)
hits_browsecomp = browsecomp_searcher.search(query, k=20, mode="hybrid", embedder=embedder)
hits = merge_results(hits_drbench, hits_browsecomp, k=20)
```

Dense cosine scores are directly comparable across pools because the same embedding model (Qwen3-Embedding-4B) is used everywhere. `merge_results()` deduplicates by doc_id, keeps the highest score, and returns top-k.

**Option**: For the first L→W hop, search drbench_urls pool only (since these URLs are specifically relevant to the DrBench tasks and more likely to yield good bridges). Fall back to full multi-pool search if no good bridges are found.

**Local search** — single pool:
```python
hits = local_searcher.search(query, k=20, mode="hybrid", embedder=embedder)
```

**Post-processing**:
1. Filter out `used_doc_ids`
2. Filter out documents shorter than 200 characters
3. Return up to 20 candidates

**Retrieval infrastructure** (all via `HybridSearcher`):

| Pool | Chunks | BM25 | Dense | Notes |
|------|--------|------|-------|-------|
| drbench_urls | `chunks_web_drbench_urls.jsonl` (262) | in-memory | numpy `.npz` | 27 DrBench seed URLs |
| BrowseComp | BrowseComp chunks JSONL (~100K) | in-memory | numpy `.npz` | To be converted from Pyserini/FAISS |
| Local | `chunks_local.jsonl` (1,457) | in-memory | numpy `.npz` | Enterprise docs |

All pools use the same embedding model (**Qwen/Qwen3-Embedding-4B**) so scores are comparable.

**Output**: Ordered list of `RetrievalHit(chunk_id, doc_id, score, text, meta)`, up to 20 candidates.

**Note on document-as-query**: We do NOT use the current document text as the search query. Research shows full-document BM25 queries are too noisy — they match on incidental terms rather than the entity/topic we care about. Step 2 generates a focused query from the chain context.

---

### Step 4: Compose Bridges (Parallel)

**Purpose**: For each of the top-N candidate documents, ask the LLM to compose the current question with new information from that document. Run these in **parallel** to maximize throughput.

**Inputs**:
- `hop_history` — per-hop Q/A pairs (so the LLM sees what facts exist)
- `global_question` — the current composed question
- `global_answer` — the current answer
- `candidate_docs` — top-N documents from Step 3 (N configurable, e.g. 10)

**Process** (N LLM calls in parallel):

For each candidate document, call the LLM with the bridge composition prompt. Collect all results.

**Prompt**:
```
You are building a multi-hop question chain. Given the chain history and a new
document, compose the NEXT question that extends the chain.

CHAIN HISTORY (each hop is a fact established so far; together they form a
reasoning chain where each answer feeds into the next question):
{formatted_hop_history}

CURRENT GLOBAL STATE (the single composed question that requires ALL hops above
to answer, and its current answer):
- Composed question: {global_question}
- Answer: "{global_answer}"

NEW CANDIDATE DOCUMENT:
{candidate_doc_text}

YOUR TASK:
1. Find an entity or concept in the new document that connects to the current
   chain (through the answer, the topic, or a shared entity).
2. Describe the current answer indirectly using context from the chain
   (the ANSWER_SLOT), WITHOUT naming it directly.
3. Write a new QUESTION that uses the ANSWER_SLOT and asks about a fact from
   the new document. The question should EXTEND the chain — answering it should
   require resolving all previous hops first.
4. Extract the ANSWER (1-5 words) from the new document.

EXAMPLE (2-hop chain, extending to 3rd hop):
  Chain history:
    Hop 1 (L): Q="What HR platform does Lee's Market use?" A="Workday"
    Hop 2 (W): Q="When was Workday founded?" A="2005"
  Global: Q="When was the HR platform Lee's Market uses founded?" A="2005"

  New doc: "In 2005, the retail industry saw a surge in digital transformation
            initiatives. Lee's Market launched its Green Supply Chain program
            that year, which reduced logistics costs by 15%..."

  ANSWER_SLOT: the year the HR platform Lee's Market uses was founded
  QUESTION: What initiative did Lee's Market launch in the year the HR platform
            they use was founded?
  ANSWER: Green Supply Chain

  Why good:
  - Connects through "2005" (shared between chain and new doc)
  - Requires knowing: Lee's Market uses Workday → Workday founded 2005 → what happened in 2005
  - Each hop is necessary — can't skip any
  - Answer is specific and appears in the document

BAD PATTERNS (output NO_BRIDGE instead):
- No meaningful connection between chain and new document → NO_BRIDGE
- Question can be answered without resolving previous hops → NO_BRIDGE
- Answer is generic (e.g., "improved", "increased", "data") → NO_BRIDGE
- Answer is same as current answer → NO_BRIDGE
- Would create compound question ("... and what...") → NO_BRIDGE

CRITICAL RULES:
- The new document must have a GENUINE connection to the current chain.
  If it doesn't, output NO_BRIDGE.
- The ANSWER must come from the new document. Do NOT infer or synthesize.
- If the document discusses multiple entities, make sure you're extracting
  facts about the RIGHT entity, not a different one mentioned nearby.
- The resulting question should require ALL previous hops to answer.
  A question that only needs the last hop is NOT multi-hop.

Think step by step:
1. What entities/concepts in the new document overlap with the chain?
2. Is the connection genuine and factual, or superficial?
3. What fact from the new document could extend the chain?
4. How would you describe the current answer indirectly?
5. Does the resulting question truly require ALL previous hops?

If no valid bridge exists, output:
<answer>NO_BRIDGE: <brief reason></answer>

Otherwise, output your final answer in <answer> tags:
<answer>
ANSWER_SLOT: <description of current answer using chain context>
QUESTION: <composed question extending the chain>
ANSWER: <1-5 words from the new document>
</answer>
```

**LLM parameters**:
- Temperature: 0.3
- Max tokens: 800
- Stage: `"bridge_composition"`

**Output**: List of `(candidate_doc, Bridge)` pairs — one per candidate that produced a valid response (not NO_BRIDGE).

---

### Step 5: Validate & Select Best Bridge

**Purpose**: From the parallel bridge proposals, filter out bad ones and select the best. Three sub-steps.

#### Step 5a: Deterministic Pre-Checks (no LLM)

Fast, cheap string checks that catch obvious failures:

| # | Check | Rationale |
|---|-------|-----------|
| 1 | `bridge.answer` is 1-5 words | Answers must be short and specific |
| 2 | `bridge.answer` != `global_answer` (case-insensitive) | Chain must progress — can't repeat the same answer |
| 3 | `global_answer` is NOT a substring of `bridge.answer` and vice versa | Prevents trivial variations ("Workday" → "Workday Inc.") |
| 4 | `global_answer` does NOT appear literally in `bridge.question` (case-insensitive) | Multi-hop requirement: question must describe answer indirectly |

If any check fails → discard this bridge candidate.

Note: we do NOT check verbatim answer-in-doc here. That's a grounding check better handled by the LLM in 5b, since small format variations (e.g., "Dave Duffield" vs "David Duffield", "$2.3B" vs "$2.3 billion") would cause false rejections.

#### Step 5b: LLM Validation (1 LLM call per surviving bridge)

For each bridge that passes 5a, ask the LLM to evaluate quality.

**Prompt**:
```
Evaluate this multi-hop question bridge for quality.

CHAIN HISTORY (each hop is a fact established so far; together they form a
reasoning chain where each answer feeds into the next question):
{formatted_hop_history}

CURRENT GLOBAL STATE (the single composed question that requires ALL hops above
to answer, and its current answer):
- Question: {global_question}
- Answer: "{global_answer}"

PROPOSED BRIDGE:
- Answer slot: {bridge.answer_slot}
- New question: {bridge.question}
- New answer: "{bridge.answer}"

DOCUMENT (where new answer comes from):
{candidate_doc_text_truncated}

Evaluate step by step on these criteria:

1. GROUNDED: Is "{bridge.answer}" actually present in or clearly derivable
   from the document? (Not hallucinated, not from a different entity's section.)
   Check carefully — if the document mentions multiple entities, verify the
   answer is attributed to the correct one.

2. GENUINE MULTI-HOP: Does answering the new question require resolving ALL
   previous hops? Walk through the reasoning chain step by step — count each
   hop needed. For hop N, the question should require N steps of reasoning.

3. REAL CONNECTION: Is the relationship between the chain and the new document
   genuine and factual? Not a superficial/coincidental connection?

4. ANSWER QUALITY: Is the answer specific and useful? (Named entities,
   specific numbers, dates, proper nouns are good. Generic words like
   "improved", "increased", "data" are bad.)

5. NATURAL QUESTION: Does the question read naturally? Could a real person
   ask this question?

Think through each criterion, then put your final verdict in <answer> tags:
<answer>
VERDICT: ACCEPT or REJECT
HOPS_REQUIRED: <number of reasoning steps needed to answer>
REASON: <brief explanation>
</answer>
```

**LLM parameters**:
- Temperature: 0.1 (consistent judgments)
- Max tokens: 600
- Stage: `"bridge_validation"`

Filter: keep only bridges with VERDICT=ACCEPT and HOPS_REQUIRED >= expected hop count.

#### Step 5c: LLM Selection (1 LLM call, only if multiple survivors)

If multiple bridges passed 5a+5b, ask the LLM to pick the best.

**Prompt**:
```
You have {N} candidate bridges for a multi-hop question chain. Pick the BEST one.

CHAIN HISTORY (each hop is a fact established so far; together they form a
reasoning chain where each answer feeds into the next question):
{formatted_hop_history}

CURRENT GLOBAL STATE (the single composed question that requires ALL hops above
to answer, and its current answer):
- Question: {global_question}
- Answer: "{global_answer}"

CANDIDATES (each survived validation — they are all acceptable, pick the best):
{formatted_candidates_with_numbers}

For each candidate, evaluate:

1. HOP DEPENDENCY STRENGTH: Try to answer the question without resolving the
   earlier hops. The best candidate is the one where skipping ANY hop makes
   the question unanswerable. Weakest: only the last hop is needed.

2. ANSWER SPECIFICITY: Specific named entities, dates, and proper nouns are
   better than numbers or short phrases. The answer should be something that
   is unlikely to appear in unrelated documents.

3. BRIDGEABILITY: Will this answer be easy to connect to MORE documents in
   future hops? (Named entities and domain-specific terms are easier to bridge
   from than generic numbers or years.)

4. QUESTION NATURALNESS: Does the question read like something a real person
   would ask? Awkward phrasing or overly convoluted structure is worse.

Think through each candidate on all 4 criteria, then output your choice:
<answer><candidate number, 1-indexed></answer>
```

**LLM parameters**:
- Temperature: 0.1
- Max tokens: 800
- Stage: `"bridge_selection"`

If only 1 survivor from 5b → skip 5c, use that one directly.
If 0 survivors → this hop failed, go back to Step 2 with different query or give up.

---

### Step 6: Update State & Repeat

**Purpose**: After a valid bridge is selected, update the chain state and prepare for the next hop.

**Process** (no LLM):
1. Add new hop to `hop_history`:
   ```
   hop_history.append({
       hop: len(hop_history) + 1,
       type: next_hop_type,
       Q: bridge.local_question,    # The "local" fact at this hop
       A: bridge.answer,
       doc_id: candidate_doc.doc_id,
   })
   ```
2. Update global state:
   ```
   global_question = bridge.question    # The new composed question
   global_answer = bridge.answer        # The new answer
   ```
3. Add `candidate_doc.doc_id` to `used_doc_ids`
4. Determine next hop type from pattern
5. If more hops needed → go back to Step 2
6. If pattern complete → go to Step 7

---

### Step 7: Verify Complete Chain

**Purpose**: Test that the final question truly requires ALL hops to answer, not just some subset.

**Inputs**:
- `global_question` — the final composed question
- `global_answer` — the expected final answer
- All document texts in the chain (ordered)

**Process** (4 LLM calls):

| Condition | Context provided | Expected result |
|-----------|-----------------|-----------------|
| No docs | None | NOT_ANSWERABLE |
| First hop only | Doc 1 only | NOT_ANSWERABLE |
| Last hop only | Doc N only | NOT_ANSWERABLE |
| All hops | All docs | ANSWERABLE (matching expected answer) |

**Prompt** (same for all 4, different context):
```
Given ONLY the context below, answer the question. If you cannot answer from the
provided context alone, respond with NOT_ANSWERABLE.

CONTEXT:
{context}

QUESTION: {question}

Think step by step:
1. What information does the question ask for?
2. What information is available in the context?
3. Can you trace a complete reasoning path from context to answer?
4. Is any intermediate fact missing that you would need?

Then put your final answer in <answer> tags:
- If answerable: <answer>ANSWERABLE: <your answer></answer>
- If not answerable: <answer>NOT_ANSWERABLE: <what information is missing></answer>
```

**LLM parameters**:
- Temperature: 0.1
- Max tokens: 500
- Stage: `"chain_verification"`

**Output**: `(is_valid: bool, reason: str)`

---

## 5. Prompt Templates Summary

| Step | Stage name | Temp | Max tokens | When called | Output format |
|------|-----------|------|------------|-------------|---------------|
| 2 | `query_generation` | 0.3 | 300 | Once per hop | `<answer>query</answer>` |
| 4 | `bridge_composition` | 0.3 | 800 | N times in parallel per hop | `<answer>ANSWER_SLOT/QUESTION/ANSWER</answer>` or `<answer>NO_BRIDGE</answer>` |
| 5b | `bridge_validation` | 0.1 | 600 | Once per bridge surviving 5a | `<answer>VERDICT/HOPS_REQUIRED/REASON</answer>` |
| 5c | `bridge_selection` | 0.1 | 800 | Once per hop (if multiple survivors) | `<answer>number</answer>` |
| 7 | `chain_verification` | 0.1 | 500 | 4 times per completed chain | `<answer>ANSWERABLE: answer</answer>` or `<answer>NOT_ANSWERABLE: reason</answer>` |

Each call is well-justified:
- **query_generation**: LLM understands the topic and target corpus better than any keyword template
- **bridge_composition**: The core creative step — only an LLM can compose questions
- **bridge_validation**: Grounding + semantic quality ("is this genuinely multi-hop?") can't be judged by string matching
- **bridge_selection**: Choosing "best" from valid candidates requires judgment
- **chain_verification**: End-to-end answerability test requires language understanding

---

## 6. Concrete End-to-End Example: 3-hop LWL Chain

### Hop 1 (L): Seed from enterprise docs

**Step 1** — select seed secret:
```
Secret from DR0004 (Lee's Market):
  Q1: "What HR platform does Lee's Market use for onboarding?"
  A1: "Workday"
  Doc: Full document from Lee's Market HR report (mentions Workday, SAP, other vendors)

hop_history = [{hop: 1, type: "L", Q: "What HR platform does Lee's Market use for onboarding?", A: "Workday"}]
global_question = "What HR platform does Lee's Market use for onboarding?"
global_answer = "Workday"
```

Next hop type from pattern "LWL": W (web)

### Building Hop 2 (W):

**Step 2** — generate query (corpus=web):
```
LLM input: chain history + global state + "public web corpus"
LLM output: "Workday enterprise HR platform cloud SaaS market competitors history"
```

**Step 3** — retrieve candidates:
```
BM25 recall (200 docs matching "Workday enterprise HR platform...") → dense rerank → top 20
Candidates include:
  Doc A: Wikipedia-style article about Workday Inc.
  Doc B: Industry report comparing HR platforms (Workday, SAP, Oracle)
  Doc C: News article about Workday's 2023 earnings
  ...
```

**Step 4** — compose bridges in parallel (say top 10):
```
Doc A → Bridge: SLOT="the HR platform Lee's Market uses", Q="In what year was the HR platform
         Lee's Market uses for onboarding founded?", A="2005"
Doc B → Bridge: SLOT="the HR platform Lee's Market uses", Q="What is the main competitor of
         the HR platform Lee's Market uses for onboarding in the mid-market segment?",
         A="SAP SuccessFactors"
Doc C → NO_BRIDGE (earnings info doesn't create a good multi-hop question)
...
```

**Step 5a** — deterministic pre-checks:
```
Both Doc A and Doc B bridges pass: answer != "Workday", "Workday" not in question, etc.
```

**Step 5b** — LLM validation:
```
Doc A bridge: ACCEPT, HOPS_REQUIRED=2 (need to know Workday → look up founding year)
Doc B bridge: ACCEPT, HOPS_REQUIRED=2 (need to know Workday → look up competitor)
```

**Step 5c** — LLM selection:
```
Picks Doc B (more interesting question, leads to named entity that's good for further bridging)
```

**Step 6** — update state:
```
hop_history = [
  {hop: 1, type: "L", Q: "What HR platform does Lee's Market use?", A: "Workday"},
  {hop: 2, type: "W", Q: "What is Workday's main mid-market competitor?", A: "SAP SuccessFactors"},
]
global_question = "What is the main mid-market competitor of the HR platform Lee's Market uses for onboarding?"
global_answer = "SAP SuccessFactors"
```

Next hop type: L (local)

### Building Hop 3 (L):

**Step 2** — generate query (corpus=local):
```
LLM input: chain history + global state + "enterprise knowledge base"
LLM output: "SAP SuccessFactors HR software implementation vendor evaluation"
```

**Step 3** — retrieve from local corpus:
```
BM25 recall (200 from 1,457 chunks) → dense rerank → top 20
Finds local doc from DR0004 that discusses vendor evaluations, mentions SAP
```

**Step 4** — compose bridges:
```
Local doc → Bridge: SLOT="the main mid-market competitor of the HR platform Lee's Market uses",
  Q="In what year did Lee's Market evaluate the main mid-market competitor of their current
     HR platform for potential migration?",
  A="2023"
```

**Step 5** — validate + select → passes

**Step 6** — update:
```
hop_history = [
  {hop: 1, type: "L", Q: "What HR platform does Lee's Market use?", A: "Workday"},
  {hop: 2, type: "W", Q: "What is Workday's main mid-market competitor?", A: "SAP SuccessFactors"},
  {hop: 3, type: "L", Q: "When did Lee's Market evaluate SAP SuccessFactors?", A: "2023"},
]
global_question = "In what year did Lee's Market evaluate the main mid-market competitor
                   of their current HR platform for potential migration?"
global_answer = "2023"
```

Pattern "LWL" complete → Step 7

### Step 7: Verify

```
Test 1: No docs → "NOT_ANSWERABLE" ✓ (no way to know any of this without docs)
Test 2: Doc 1 only (HR report) → "NOT_ANSWERABLE" ✓ (mentions Workday but not SAP competitor info)
Test 3: Doc 3 only (vendor eval) → "NOT_ANSWERABLE" ✓ (mentions SAP eval but not that it's Workday's competitor or that Lee's Market uses Workday)
Test 4: All 3 docs → "2023" ✓ (can trace: Workday → competitor is SAP → eval in 2023)
```

Chain is valid.

---

## 7. Lessons from v1 (What to Avoid)

### Problem 1: Heuristic scoring selected bad bridges
**v1**: `score_bridge()` with proximity/slot/length weights. Hallucinated bridges could still score high.
**v2**: No heuristic scoring. Deterministic pre-checks for obvious failures + LLM validation for semantic quality + LLM selection for best candidate.

### Problem 2: Numeric answers as search queries
**v1**: Used "80%" as a BM25 query → random documents containing "80%".
**v2**: LLM generates corpus-aware queries from full chain context — topic keywords, not raw values.

### Problem 3: Model hallucinations in multi-entity documents
**v1**: Model said "Google developed Workday" — mixing facts across document sections.
**v2**: Anti-hallucination instructions in bridge prompt + LLM validation explicitly checks grounding and entity attribution.

### Problem 4: answer == current_answer loops
**v1**: Model returned "Workday" when bridging from "Workday".
**v2**: Deterministic pre-check (answer != current answer, no substring containment).

### Problem 5: Composed questions didn't require multi-hop
**v1**: "When was Workday founded?" — answerable from last doc alone.
**v2**: LLM validation counts HOPS_REQUIRED and rejects if < expected. Chain verification (Step 7) tests with subsets of docs.

### Problem 6: Hard to debug
**v1**: Complex interleaved pipeline.
**v2**: Each step independently testable. Parallel generation makes it easy to see all candidates.

---

## 8. Data Structures

```python
@dataclass
class HopRecord:
    """One entry in the per-hop history."""
    hop_number: int
    hop_type: str     # "L" or "W"
    question: str     # The "local" question at this hop
    answer: str       # The answer at this hop
    doc_id: str
    doc_text: str     # Full document text

@dataclass
class ChainState:
    """Full state of a chain being built."""
    pattern: str                # e.g., "LWL"
    hop_history: list[HopRecord]  # Per-hop facts
    global_question: str        # Current composed question
    global_answer: str          # Current answer
    used_doc_ids: set[str]      # Docs already in chain
    company: str | None = None
    task_id: str | None = None

@dataclass
class Bridge:
    """A proposed bridge from Step 4."""
    answer_slot: str    # Description of current answer
    question: str       # New composed question
    answer: str         # New answer from the document
    doc_id: str
    doc_text: str

@dataclass
class Chain:
    """A complete, verified multi-hop chain."""
    chain_id: str
    pattern: str
    hop_history: list[HopRecord]
    global_question: str
    global_answer: str
    verification: dict
    metadata: dict       # Timing, LLM calls count, etc.
```

---

## 9. Evaluation Plan (Test Each Step Independently)

### Test 1: Query Generation Quality

**What**: Given 20 (question, answer) pairs from secret_inventory, generate search queries and check if they return relevant documents.

**How**:
```
For each (Q, A) pair:
  1. Generate query (Step 2) for both web and local corpus
  2. Search with two-stage retrieval (Step 3)
  3. Check: does top-10 include documents related to the topic?
```

**Success criteria**: At least 1 of top-10 results is topically relevant in >=70% of cases.

### Test 2: Bridge Composition Quality

**What**: Given 20 (chain_state, candidate_doc) pairs where we KNOW the doc is relevant, test if the LLM can compose a valid bridge.

**Setup**: Manually curate 20 good pairs from actual retrieval results.

**Success criteria**: Valid bridge produced for >=60% of known-good pairs.

### Test 3: Validation Accuracy

**What**: Given 20 bridge proposals (10 good, 10 bad), test if validation correctly accepts/rejects.

**Success criteria**:
- Deterministic pre-checks (5a): 100% accuracy on obvious failures
- LLM validation (5b): >=85% accuracy on semantic quality

### Test 4: End-to-End Chain Building

**What**: Run the full pipeline on 20 seeds and measure success rate.

**Success criteria**: >=5/20 valid chains (25%).

### Test 5: Chain Verification Accuracy

**What**: Given 10 known-good chains and 10 known-bad chains, test if verification correctly accepts/rejects.

**Success criteria**: >=90% accuracy.

---

## 10. Implementation Order

```
Phase 0: Infrastructure (PARTIALLY DONE)
  [DONE] Fetch + chunk drbench_urls pool (27 URLs → 262 chunks)
  [DONE] HybridSearcher with BM25 + dense hybrid retrieval
  [DONE] Dense index builder with Qwen3-Embedding-4B support
  [DONE] Retrieval eval harness (recall@k, MRR)
  - Build drbench_urls dense index with Qwen3-Embedding-4B (requires vLLM)
  - Build local dense index over chunks_local.jsonl
  - Convert BrowseComp to chunks JSONL + build dense index with Qwen3-Embedding-4B
  - Verify multi-pool web search works end-to-end

Phase 1: Scaffolding
  - Implement data structures (ChainState, HopRecord, Bridge, Chain)
  - Port data loading (secrets, chunks)
  - Full-document lookup from chunk_id → doc_id → full text

Phase 2: Step-by-step implementation + testing
  - Implement Step 2 (query generation) + run Test 1
  - Implement Step 4 (bridge composition, parallel) + run Test 2
  - Implement Step 5 (validate + select) + run Test 3
  - Implement Step 7 (verification) + run Test 5
  - Wire together Steps 1-7 + run Test 4

Phase 3: Iteration
  - Analyze failure modes from Test 4
  - Refine prompts based on actual failures
  - Re-run until success rate is acceptable
```

---

## 11. Infrastructure

### HybridSearcher
- Location: `making_dataset_2/retrieval/hybrid.py`
- Two-stage: BM25 recall + dense recall (union) → dense cosine ranking
- One instance per pool (drbench_urls, BrowseComp, local)
- `merge_results()` combines results from multiple pools by score

### Embedding Model
- **Qwen/Qwen3-Embedding-4B** — used across ALL pools for comparable scores
- Served via vLLM (`OpenAICompatibleBackend`) or loaded locally (`SentenceTransformerBackend`)
- Query embeddings computed at search time; document embeddings pre-built in `.npz` indexes

### Web Pools
- **drbench_urls**: 27 DrBench seed URLs → 262 chunks
  - Chunks: `making_dataset_2/outputs/chunks_web_drbench_urls.jsonl`
  - Dense: `making_dataset_2/outputs/indexes/drbench_urls_dense.npz`
  - Data prep: `data_prep/fetch_drbench_urls.py` → `chunk_web_drbench_urls.py` → `build_dense_index.py`
- **BrowseComp**: ~100K web docs
  - Chunks: **to be created** (convert from existing BrowseComp format to chunks JSONL)
  - Dense: **to be built** with Qwen3-Embedding-4B (replaces existing 0.6B index)

### Local Pool
- Chunks: `making_dataset/outputs/chunks_local.jsonl` (1,457 chunks)
- Dense: **to be built** in `making_dataset_2/outputs/indexes/local_dense.npz`

### LLM Client
- Location: `making_dataset/utils/vllm_client.py`
- Model-agnostic: wraps any OpenAI-compatible API
- `chat(messages, stage, **kwargs)` → returns ChatCompletion object

### Secret Inventory
- Location: `making_dataset/outputs/secret_inventory.jsonl`
- 221 document entries with nested secrets (943 total)
- Each secret: `{question, answer, secret_type, doc_only_check}`

---

## 12. Open Questions (To Resolve Before Coding)

1. **Parallel N**: How many candidate docs to run bridge composition on in parallel? (10? 20? all retrieved?)

2. **Document truncation**: Bridge prompt needs doc text. How much? First 3000 chars? Longer? Should we try to find the relevant section first?

3. **Verification strictness**: Should "fuzzy match" in Step 7 be exact string match, substring match, or LLM judgment?

4. **Chain length**: Start with 2-hop (LW) chains. Move to 3-hop after success rate is acceptable?

5. **Model choice**: Which model works best for bridge composition? Need to experiment.

6. ~~**Local dense index**: Which embedding model?~~ **RESOLVED**: Qwen3-Embedding-4B across all pools.

7. **LLM self-evaluation**: Step 5b uses the same model to judge its own compositions. Is this reliable, or should we use a different model for validation?
