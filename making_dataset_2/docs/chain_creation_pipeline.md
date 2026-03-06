# Multi-Hop Chain Creation Pipeline

## Overview

The pipeline builds **multi-hop question-answer chains** that require reading
multiple documents in sequence. Each chain follows a **pattern** like `LWL`
indicating document types (L=local enterprise, W=web), and is validated to
ensure it truly requires ALL documents to answer correctly.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        chain_builder.py (orchestrator)                  │
│                                                                         │
│  ┌──────────────┐   For each transition in pattern (e.g. L→W, W→L):    │
│  │  SEED (step1) │                                                      │
│  │ L: random     │   ┌──────────────────────────────────────────────┐   │
│  │    secret      │   │              TRANSITION LOOP                 │   │
│  │ W: NER entity  │   │                                              │   │
│  │    + LLM Q/A   │   │  ┌─────────────┐    ┌────────────────────┐  │   │
│  └──────┬─────────┘   │  │ find_bridge  │    │ generate_question  │  │   │
│         │             │  │ (entity_     │───►│ _constrained       │  │   │
│         │             │  │  index.py +  │    │ (if needs_intra)   │  │   │
│         │             │  │  find_       │    └────────┬───────────┘  │   │
│         ▼             │  │  bridge.py)  │             │              │   │
│   ChainState          │  └──────┬──────┘    ┌────────▼───────────┐  │   │
│   (hop_history,       │         │           │ generate_question  │  │   │
│    global_answer,     │         │           │ _pick (inter-doc)  │  │   │
│    used_doc_ids)      │         │           └────────┬───────────┘  │   │
│         │             │         │                    │              │   │
│         │             │         │           ┌────────▼───────────┐  │   │
│         │             │         │           │ VALIDATION         │  │   │
│         │             │         │           │ • check_question   │  │   │
│         │             │         │           │   (deterministic)  │  │   │
│         │             │         │           │ • check_answerable │  │   │
│         │             │         │           │   _without_doc     │  │   │
│         │             │         │           │   (LLM trivial     │  │   │
│         │             │         │           │    gate)           │  │   │
│         │             │         │           └────────┬───────────┘  │   │
│         │             │         │                    │              │   │
│         │             │  Up to MAX_CANDIDATES=8      │              │   │
│         │             │  tried per transition        │              │   │
│         │             │                    ┌─────────▼──────────┐   │   │
│         │             │                    │ rank_questions     │   │   │
│         │             │                    │ (LLM judge, picks  │   │   │
│         │             │                    │  best from valid   │   │   │
│         │             │                    │  options)          │   │   │
│         │             │                    └─────────┬──────────┘   │   │
│         │             │                              │              │   │
│         │             │                    ┌─────────▼──────────┐   │   │
│         │             │                    │ COMMIT to state    │   │   │
│         │             │                    │ Update answer,     │   │   │
│         │             │                    │ doc, hop_history   │   │   │
│         │             │                    └────────────────────┘   │   │
│         │             └──────────────────────────────────────────────┘   │
│         │                                                               │
│         ▼                                                               │
│   ┌──────────────┐    ┌──────────────┐    ┌────────────────────────┐   │
│   │ format_      │───►│ VERIFY       │───►│ OUTPUT: Chain          │   │
│   │ numbered_    │    │ (step7)      │    │ {chain_id, pattern,    │   │
│   │ questions    │    │ 4 conditions │    │  hops, verification,   │   │
│   │ (N) refs     │    │ in parallel  │    │  metadata}             │   │
│   └──────────────┘    └──────────────┘    └────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Patterns

Each character represents one document hop. Transitions between consecutive
characters drive bridge finding.

| Pattern | Hops | Transitions | Description |
|---------|------|-------------|-------------|
| `LW`    | 2    | 1 (L→W)    | Local seed, bridge to web |
| `WL`    | 2    | 1 (W→L)    | Web seed, bridge to local |
| `LWL`   | 3    | 2 (L→W→L)  | Local → web → local |
| `WLW`   | 3    | 2 (W→L→W)  | Web → local → web |
| `LLWW`  | 4    | 3 (L→L→W→W)| Two local → two web |

## Pipeline Steps in Detail

### Step 1: Seed Selection (`step1_seed.py`)

**L seed (deterministic):**
- Pick a random secret from `secret_inventory.jsonl`
- Must be an "entity" (has uppercase/digits, ≤2 words, not email/date)
- Creates initial `HopRecord(hop_type="L")` with the secret as Q/A

**W seed (LLM-required):**
- Sample ≤200 random web docs
- Extract entities via spaCy NER
- Check if any entity appears in a local doc (bridgeability)
- LLM writes a Q/A pair for the entity
- Up to 20 attempts before giving up

### Step 2: Bridge Finding (`find_bridge.py`)

For each document transition, find candidate (target_doc, bridge_entity) pairs:

```
Priority order:
1. FAST PATH: current_answer appears in an unseen target doc
   → needs_intra=False (no extra question needed)
   → These candidates ranked first

2. ENTITY PATH: extract entities from current doc via spaCy NER
   → For each entity, check if it appears in any unseen target doc
   → Score by character proximity to current_answer in source doc
   → needs_intra=True (requires an intra-doc question)

3. BM25 PATH (optional): for large pools (>1000 docs)
   → Build query from context around current_answer (±200 chars)
   → Retrieve top-K docs, check for fast-path and entity-path matches
```

**Result:** Ordered list of `BridgeCandidate`:
```python
@dataclass
class BridgeCandidate:
    target_doc_id: str
    bridge_entity: str
    needs_intra: bool  # True = requires intra-doc Q to get to bridge_entity
    score: float       # Higher = better (proximity-based)
```

### Step 3: Question Generation (`step4_questions.py`)

For each bridge candidate (up to `MAX_CANDIDATES=8`):

**If `needs_intra` (bridge_entity ≠ current_answer):**
1. `generate_question_constrained(doc, current_answer, bridge_entity)`
   - Both sides known: prev_answer → target_answer in same doc
   - Prompt requires finding an exact sentence connecting them
   - prev_answer must be the "subject or key qualifier"
   - Result validated by `check_question()` + `check_answerable_without_doc()`

**Then always:**
2. `generate_question_pick(target_doc, bridge_entity, entity_list)`
   - bridge_entity is known, model picks best entity from list
   - Entities from target doc (via NER), excluding used ones
   - Result validated same way

**If multiple valid options:** LLM judge ranks them (`rank_questions`)

### Step 4: Validation (`step5_check.py`)

Two layers of validation, both must pass:

#### Deterministic checks (`check_question`):

| Check | Rule |
|-------|------|
| Non-empty | Question and answer must be present |
| Required phrase | Previous answer must appear in question text |
| Answer length | 1-5 words |
| Expected answer | Must match target (constrained questions only) |
| No duplicates | Answer ≠ any previous answer in chain |
| Not self-referential | Answer ≠ required phrase |
| Quote in doc | Quote must appear in source document |
| Answer in quote | Answer must appear in the quote |

#### LLM trivial gate (`check_answerable_without_doc`):

Ask the LLM to answer the question **without any document context**.
- If it can → question is **too easy** → rejected
- If it says `NOT_ANSWERABLE` → question requires context → kept

### Step 5: Formatting (`format_questions.py`)

Convert hop questions to numbered format with `(N)` back-references:

```
Input hops:
  Q1: "What HR platform does Lee's Market use?"           A: "Salesforce"
  Q2: "When was Salesforce founded?"                       A: "1999"
  Q3: "What did Lee's Market do in 1999?"                 A: "Opened first store"

Output:
  1. What HR platform does Lee's Market use?
  2. When was (1) founded?
  3 (final). What did Lee's Market do in (2)?
```

Replacement is longest-first to avoid partial matches. Raises `ValueError`
if any hop (after the first) doesn't contain a previous answer.

### Step 6: Verification (`step7_verify.py`)

Four answerability tests run **in parallel** (4 LLM calls):

| Condition | Context Provided | Expected |
|-----------|-----------------|----------|
| `no_docs` | (empty) | NOT_ANSWERABLE |
| `first_only` | Only 1st hop's doc | NOT_ANSWERABLE |
| `last_only` | Only last hop's doc | NOT_ANSWERABLE |
| `all_docs` | All docs | ANSWERABLE (with correct answer) |

**Context construction:** Each hop's context is the quote ± 500 chars
from the source doc. Falls back to first 6000 chars if no quote found.

**Validity:** All 4 conditions must pass. This ensures the chain requires
reading ALL documents — you can't shortcut with just one subset.

## Failure Modes

### What happens when a check fails

**Deterministic check fails (`check_question` returns error):**
- That candidate is **skipped**, next candidate is tried
- Up to `MAX_CANDIDATES=8` bridge candidates per transition
- If ALL candidates fail → transition fails → chain is **INCOMPLETE**

**Trivial gate fails (question too easy):**
- Same as above: candidate skipped, next tried

**No bridge candidates found:**
- Transition fails → chain is **INCOMPLETE**
- No retry — the chain is written as-is with `metadata.complete = false`

**Verification fails (step 7):**
- Chain is written with `verification.is_valid = false`
- No retry — the chain is preserved for debugging/analysis

### There is NO retry loop

Each `--n 30` generates exactly 30 chains per pattern. If a chain fails
partway through, it's written as incomplete. If verification fails, it's
written as invalid. The code does NOT:
- Re-attempt a failed chain with a different seed
- Retry with different bridge candidates after all 8 fail
- Loop until N valid chains are produced

To get more valid chains, increase `--n`. Typical valid rates:
- LW: ~90% (simple, 1 transition)
- WL: ~50% (web seed selection harder)
- LWL: ~47% (2 transitions)
- WLW: ~43% (2 transitions, web seed)
- LLWW: ~77% (3 transitions, but L→L bridging is easy)

## Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| `MAX_CANDIDATES` | 8 | `chain_builder.py` |
| `MAX_ENTITY_LIST` | 20 | `chain_builder.py` |
| `CONTEXT_WINDOW` | 4000 chars | `chain_builder.py` |
| `DOC_TEXT_LIMIT` | 8000 chars | `step4_questions.py` |
| `_VERIFY_DOC_LIMIT` | 6000 chars | `step7_verify.py` |
| `_QUOTE_CONTEXT` | 500 chars | `step7_verify.py` |

## Known Limitation: Back-Reference Independence

A question like:
```
3 (final). Who is the actor who plays the lead in The Wolf of Wall Street
           and was born in (2)?
```

This can be answered as "Leonardo DiCaprio" without knowing (2), because
the actor is uniquely identified by the movie alone. The `(2)` reference
is decorative, not load-bearing.

The current pipeline guards against this only partially:
- `check_answerable_without_doc` tests the **pre-formatted** question
  (with literal answers, not (N) references)
- Verification tests the **whole chain**, not individual questions

What's missing: a check that each `(N)` reference is **necessary** for
answering the question — i.e., the question should NOT be answerable if
the back-reference is removed or replaced with a placeholder.
