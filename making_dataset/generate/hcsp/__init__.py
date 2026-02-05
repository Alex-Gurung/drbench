# HCSP (Hierarchical Constraint Satisfaction Problem) task generation
# InfoSeek-style blurred multi-hop questions
#
# Implementation based on: https://arxiv.org/abs/2509.00375
# Full design doc: /home/toolkit/.claude/plans/concurrent-tumbling-micali.md
#
# Modules:
# - schema: Data classes (Constraint, Vertex, ResearchTree, HCSPTree, etc.)
# - planner: Planner agent (decides WHAT action to take)
# - browser: Browser agent (executes actions: INIT_ROOT, ADD_CONSTRAINTS, EXTEND_LOCAL, EXTEND_WEB)
# - tree_builder: Main loop integrating planner/browser
# - constraints: Constraint extraction from chunks
# - synthesize: Question synthesis from constraints
# - linkage: Linkage type detection (entity_chain, computational, selection, definitional)
#
# Key design decisions (from researcher feedback):
# 1. Root is a QUESTION node, not answer node - answer VALUE only in gold.answer
# 2. Minimal banning - only ban final answer string, not intermediate entities
# 3. Shape-based validation - ensure chain structure, web on critical path for mixed
