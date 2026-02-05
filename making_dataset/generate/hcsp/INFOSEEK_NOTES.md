# InfoSeek Methodology Notes

> **Implementation Status**: ✅ Complete (2026-02-04)
> See `/home/toolkit/.claude/plans/concurrent-tumbling-micali.md` for full implementation details.

Source: https://arxiv.org/abs/2509.00375 "Open Data Synthesis For Deep Research"

## Core Architecture: Dual-Agent System

### Planner Agent
- Strategic control over tree construction
- Selects which vertex to expand next
- Chooses which action (1-4) to execute
- Monitors complexity objectives
- Ensures balanced growth (sequential vs parallel reasoning)

### Browser Agent
Executes 4 action types:

1. **Action 1 (Initialization)**: Sample valid entity from Wikipedia as final answer, create root vertex r

2. **Action 2 (Constraint Addition)**: Extract k claims from entity's webpage that TOGETHER yield unique identification. This is the "blurring" - instead of naming entity, describe it with constraints.

3. **Action 3 (Depth Extension)**: Extract hyperlinks indicating dependencies → create child nodes. This extends the tree depth.

4. **Action 4 (Termination)**: Signal when tree achieves desired complexity.

## Tree Construction Process

**Recursive structure:**
- Base case: Single root vertex (the answer)
- Recursive step: Add new vertex w connected to existing vertex v with edge (v,w)
- Intermediate vertices are "blurred" with multiple constraints

**Key insight**: Each vertex is described by constraints, not named directly. The constraints must "taken together, yield the unique and determinate answer v."

## Blurring Process

For each vertex v:
1. Browser accesses v's webpage (Wikipedia)
2. Selects k claims that uniquely identify v
3. Claims become child vertices (constraints)
4. v is never named in the question - only described via constraints

Example from paper:
- Entity: Alan Turing
- NOT: "Who is Alan Turing?"
- YES: "Who is the mathematician born in a European city (English-speaking, >5M pop), who studied at Cambridge, earned PhD at Princeton in 1938?"

## Validation: Handling Edge Cases

**Underdetermined** (answer not unique):
- Even with all constraints, multiple valid answers
- Mitigation: Verifiability validation with ground-truth + distractor docs

**Overdetermined** (single constraint sufficient):
- One constraint alone yields answer
- Mitigation: Ensure candidate sets are mutually exclusive

**Difficulty filtering**:
- Remove samples answerable by parametric memory alone

## Question Generation

Final step: Convert tree structure to natural language question
- LLM receives "blurred vertex descriptions"
- Generates query requiring full tree traversal

## Application to Our Mixed-Mode HCSP

For local+web mixed mode:
1. **Root** = local secret (answer)
2. **Action 2 equivalent**: Extract constraints from local chunk
3. **Action 3 equivalent**: Find web chunks related to intermediate entities
4. **Blurring**: Replace intermediate entities with web-derived constraints

The key is INCREMENTAL construction:
- Don't generate everything at once
- Build tree step by step
- Validate at each step
- Planner decides when to add web context vs local context
