# Handoff: Privacy Leakage in Deep Research Agents

## What This Project Is

We're studying whether deep research agents (like DrBench) leak private enterprise data through their web search queries. When an agent reads internal company docs and then searches the web, the queries it generates may contain company names, KPI numbers, or other private facts that an adversary monitoring search traffic could exploit.

There are **two parallel tracks** being developed:

1. **Atomic Privacy Testbed** (`experiments/atomic_privacy_testbed.py`) — tests whether a model's generated web queries leak data from local findings. Works with actual DrBench batch runs.
2. **Multi-hop QA Chain Builder** (`making_dataset_2/`) — constructs multi-hop reasoning chains that link local enterprise docs to public web docs through shared entities. These chains are the **dataset** we're building for evaluation.

---

## Track 1: Atomic Privacy Testbed

### How it works

1. Load a completed DrBench batch run (e.g., `runs/batch_Qwen3-30B_20260109/`)
2. Extract the local document syntheses (what the agent learned from enterprise docs)
3. Feed the research question + local findings to an LLM with a prompt template
4. The LLM generates web search queries
5. Check if those queries contain private info (company names, specific metrics)
6. Run an "adversary" LLM that tries to answer private questions from the queries alone

### Key constraint (from CLAUDE.md)

**Only use the `drbench_adaptive` prompt** — this matches the real DrBench agent behavior. Other prompts (`maximal_leakage`, `specific_validation`) are useful for calibration but don't represent realistic threats.

### What we've found

- Leakage is **stochastic** with the standard prompt: ~10-20% leak rate
- The same question can leak in one run but not another
- To properly measure: run each question 5-10 times
- Questions that embed metrics from local findings have higher leakage probability
- Company name leakage is the most common (model includes "Lee's Market" in queries)

### Running it

```bash
# Single task
python experiments/atomic_privacy_testbed.py \
    --batch ./runs/batch_Qwen3-30B-A3B-Instru_20260109_092111 \
    --task DR0001 \
    --model gpt-4o-mini --llm-provider openai

# All tasks
python experiments/atomic_privacy_testbed.py \
    --batch ./runs/batch_Qwen3-30B-A3B-Instru_20260109_092111 \
    --all \
    --model gpt-4o-mini --llm-provider openai

# Interactive mode (swap questions, re-run)
python experiments/atomic_privacy_testbed.py \
    --batch ./runs/batch_Qwen3-30B-A3B-Instru_20260109_092111 \
    --task DR0001 -i \
    --model gpt-4o-mini --llm-provider openai
```

### Prompt templates

Live in `experiments/prompts/`. The important ones:
- `drbench_adaptive.txt` — **the one to use** (matches real DrBench)
- `drbench_accurate.txt` — exact copy of DrBench's prompt, no local findings
- `drbench_with_context.txt` — same but injects local findings
- `maximal_leakage.txt` — deliberately maximizes leakage (calibration only)

---

## Track 2: Multi-hop QA Chain Builder

### The big picture

We're building a dataset of multi-hop questions that require reasoning across both **local enterprise docs** (L) and **public web docs** (W). The chain pattern is `L → W → W → L`:

```
Q1 → A1   (from local doc — private enterprise fact)
   ↓ A1 found in web doc
Q_hop1 → B  (from web doc — mentions A1, answer is new entity B)
   ↓ B connects to E in same doc
Q_hop2 → E  (from web doc — mentions B, answer is entity E)
   ↓ E appears in another privacy question Q3
Q3 → A3     (from local doc — another private enterprise fact)
```

This chain is dangerous because: a researcher asking Q1 leads to web searches containing A1, which through the web doc connects to E, which reveals the existence of Q3/A3 — a completely different private fact.

### Variable naming convention

- **A1**: Answer to seed privacy question Q1 (entity from local doc)
- **B**: Intermediate answer from Bridge Q1 (new fact from web doc)
- **E**: Bridge entity — appears in Q3's question text, connecting web back to local
- **A3**: Answer to target privacy question Q3 (different private fact)

### Data pipeline (`making_dataset_2/pipeline/`)

Full automated pipeline:

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `step1_seed.py` | Pick a privacy secret as seed (Q1/A1) |
| 2 | `step2_query.py` | Generate search query for web corpus |
| 3 | `step3_retrieve.py` | Retrieve candidate web docs |
| 4 | `step4_bridge.py` | LLM composes bridge questions from web doc |
| 5 | `step5_validate.py` | LLM validates bridge quality |
| 7 | `step7_verify.py` | End-to-end verification (need all docs to answer?) |
| * | `chain_builder.py` | Orchestrates all steps |

```bash
# Run automated chain building
/home/toolkit/.mamba/envs/vllm013/bin/python -m making_dataset_2.pipeline.chain_builder \
    --pattern LW \
    --model Qwen/Qwen3-30B \
    --base-url http://127.0.0.1:8000/v1 \
    --n 10 --output chains.jsonl
```

### Interactive chain builder (`making_dataset_2/eval/interactive_chain.py`)

Manual tool for building chains by hand. This is what we were actively working on.

```bash
/home/toolkit/.mamba/envs/vllm013/bin/python -m making_dataset_2.eval.interactive_chain
```

Features:
- **Search modes**: `[L]`ocal / `[W]`eb / `[B]`oth (BM25) / `[A]`uto-find (substring)
- **Auto-suggest**: When you pick a web doc, it uses spaCy NER to find privacy questions whose question text shares entities with the web doc
- **Two-hop bridge flow**: After picking a suggestion, shows context for both A1 and E, prompts for Bridge Q1 (→B) and Bridge Q2 (→E), then pre-fills Q3
- **Top-10 limiting** with "show more"

### chain_test.py — automated chain validation

Tests whether an LLM can generate valid two-hop bridge questions given (Q1, A1, web_doc, E, Q3).

```bash
/home/toolkit/.mamba/envs/vllm013/bin/python -m making_dataset_2.eval.chain_test \
    --model MiniMax-M2.1 \
    --base-url http://dns-xxx:8000/v1 \
    --n 10 --seed 42
```

Latest results (from tmp1.txt): **8/10 valid chains (80%)** with MiniMax-M2.1. Failures are usually `A1_NOT_IN_Q` (hop1 question doesn't mention A1) or `B_NOT_IN_Q` (hop2 question doesn't mention B).

---

## Data

### Files

| File | Count | Description |
|------|-------|-------------|
| `making_dataset/outputs/chunks_local.jsonl` | 1457 chunks | Enterprise docs (all 3 companies) |
| `making_dataset_2/outputs/chunks_web_drbench_urls.jsonl` | 274 chunks | Web docs from DrBench URLs |
| `making_dataset_2/outputs/secret_inventory.jsonl` | 221 entries → ~544 secrets | LLM-generated QA pairs from local docs |

### Secret inventory

Each entry in `secret_inventory.jsonl` has a `chunk_id`, `doc_id`, and a list of `secrets`. Each secret has:
- `question` / `answer` — the QA pair
- `secret_type` — `kpi_numeric`, `other_sensitive`, `names`, `emails`, etc.
- `doc_only_check` — `{with_doc: true/false, without_doc: true/false}` — whether the question is answerable with/without the doc

`filter_seed_secrets()` filters to secrets that:
- Are answerable with doc (`with_doc=true`)
- Have 1-5 word answers
- Are not names/emails
- Come from docs with 200+ chars

After filtering: ~544 eligible secrets.

### Companies and tasks

- **DR0001-DR0005**: Lee's Market (retail grocery)
- **DR0006-DR0010**: MediConn Solutions (healthcare IT)
- **DR0011-DR0015**: Elexion Automotive (electric vehicles)

### Best companies for chain-building

Based on entity overlap between local and web docs:
- **MediConn** (DR0006-0010) has the most overlap — CRM, telehealth, HIPAA, 2025, etc.
- **Elexion** (DR0011-0015) — ACC II regulations, CARB, EV market
- **Lee's Market** (DR0001-0005) — chatbots, Centennials, loyalty programs

---

## What Was Being Built When We Stopped

### interactive_chain.py — the active work

The interactive chain builder was being refined. Current state:

1. **Auto-find** works — substring match across both searchers' internal arrays
2. **spaCy NER suggestions** work — entities from web doc matched against pre-computed entities from privacy question text (not answers!)
3. **Two-hop bridge flow** works — shows A1 context + E context, prompts for Bridge Q1 + B and Bridge Q2 + E, then pre-fills Q3

### Design decisions that matter

**Why match entities in questions, not answers:**
The goal is to find a privacy question Q3 whose *question text* mentions entity E from the web doc. This is because:
- The question text is what determines if there's a reasoning path L→W→W→L
- If E appears in Q3's question, then knowing E (from the web doc) helps the researcher arrive at Q3, potentially leaking A3
- Matching answers would be backward — we need the chain to flow *through* the web doc

**Why spaCy NER instead of substring matching:**
Substring matching caused "9%" to match "69%" (the string "9%" appears inside "69%"). spaCy extracts entities as discrete tokens, so PERCENT entities like "9%" and "69%" are separate.

**Why two bridge questions instead of one:**
The chain needs A1→B→E as two reasoning steps in the web doc. A single bridge question would collapse this into one hop, losing the multi-hop structure. The pattern mirrors `chain_test.py`'s PROMPT_HOP1 and PROMPT_HOP2.

### What to work on next

1. **Run interactive_chain.py on MediConn tasks** — MediConn has the best entity overlap. Try DR0009 (CRM) as a starting point.

2. **Build a set of manually validated chains** — The interactive tool produces chains but they need human quality control. The automated pipeline (`chain_builder.py`) can produce many, but manual validation catches subtle issues.

3. **Connect chains back to privacy evaluation** — Once we have multi-hop chains, the next step is to test: does a DrBench agent following these chains actually leak more private data through web queries? This connects Track 1 and Track 2.

4. **chain_test.py quality improvements** — 80% valid is good but the 20% failures suggest the LLM prompts for hop generation could be tightened. Common failure: A1 not mentioned in Q_hop1.

5. **Scale up secret inventory** — Currently ~544 secrets from 221 entries. More secrets = more potential chains. Consider re-running `privacy_tagger.py` with improved prompts.

---

## Environment

- **Python**: `/home/toolkit/.mamba/envs/vllm013/bin/python` (has spaCy `en_core_web_lg`, all deps)
- **Default Python** does NOT have spaCy large model
- **Run from**: `/home/toolkit/nice_code/drbench/`
- **Git branch**: `privacy-eval-framework`

---

## Key Intuitions

### On privacy leakage

- The most dangerous leak is **company name + metric** in a single query (e.g., "Lee's Market 12% customer retention")
- Company name alone is a moderate leak (reveals what company is being researched)
- Metrics alone are low risk (numbers without context are ambiguous)
- The `drbench_adaptive` prompt leaks company names ~10-20% of the time — it's not zero but it's not catastrophic
- **Multi-hop chains amplify risk**: even if each individual query is clean, the *combination* of queries across a chain can reveal the reasoning path

### On chain building

- **Entity answers make better seeds than numeric answers.** "CRM" bridges to web docs about CRM systems. "12%" bridges to nothing useful on the web.
- `is_entity()` in `chain_test.py` filters: must have at least one 3+ letter alphabetic word, excludes pure numbers/percentages/emails/quarters.
- **Web doc quality matters hugely.** The 274 chunks from DrBench URLs are curated — they're the web pages the DrBench tasks reference. They have real overlap with enterprise topics.
- **Bridge quality depends on the LLM.** MiniMax-M2.1 produces 80% valid chains. Weaker models will be worse. The prompts in `chain_test.py` are relatively simple — just ask for Q containing A, answer B.
- **Verification (step 7) is the real test.** A chain is only valid if: the question is NOT answerable with zero docs, NOT answerable with only the first doc, NOT answerable with only the last doc, but IS answerable with all docs. This proves true multi-hop dependency.

### On the interactive tool

- The suggestion mechanism (spaCy NER overlap) produces many false positives. Years like "2025" appear in tons of questions and web docs. Entity types like ORG and PRODUCT are higher quality bridges.
- The two-hop bridge flow is designed to match what `chain_test.py` does automatically, but with human quality control.
- Pre-filling Q3 from the suggestion saves time and ensures the chain terminates at a real privacy question.

---

## Files Quick Reference

| Path | Purpose |
|------|---------|
| `making_dataset_2/eval/interactive_chain.py` | Interactive chain builder (the active tool) |
| `making_dataset_2/eval/chain_test.py` | Automated chain validation with LLM |
| `making_dataset_2/pipeline/chain_builder.py` | Full automated pipeline |
| `making_dataset_2/data_loading.py` | Load secrets, chunks, docs |
| `making_dataset_2/types.py` | Data structures (ChainState, HopRecord, Bridge, Chain) |
| `making_dataset_2/prompts.py` | All LLM prompt templates |
| `making_dataset_2/retrieval/hybrid.py` | HybridSearcher (BM25 + dense) |
| `experiments/atomic_privacy_testbed.py` | Atomic privacy testing |
| `experiments/prompts/drbench_adaptive.txt` | The prompt template that matters |
| `privacy_eval.py` | Evaluate privacy leakage from batch runs |
| `CLAUDE.md` | Project conventions and instructions |
