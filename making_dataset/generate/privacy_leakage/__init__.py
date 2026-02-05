# Privacy Leakage Task Generation
#
# Generates multi-hop QA tasks for testing agent privacy leakage.
# See QUESTION_EXAMPLES.md for target question styles.
# See SYSTEM_DESIGN.md for generation pipeline.
#
# Key question types:
# - 2-hop: Local → Web → Answer (simple)
# - 4-hop: Local → Web → Local → Web → Answer (preferred)
#
# The 4-hop questions require agents to discover private data
# while reasoning, creating opportunities to leak that data
# in web search queries.
