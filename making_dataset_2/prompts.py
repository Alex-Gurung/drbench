"""Prompt templates for the multi-hop chain builder.

The atomic question prompts (intra/inter) live in pipeline/step4_questions.py.
This module only contains the verification prompt used by step7.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Step 7: Chain Verification
# ---------------------------------------------------------------------------

CHAIN_VERIFICATION_TEMPLATE = """
This is a multi-hop question chain. Each question's answer feeds into the next \
question — references like (1) mean "the answer to question 1".

Your task: trace through ALL questions in order and provide the FINAL answer. \
You must answer every intermediate question to reach the final one. \
If ANY step cannot be answered from the context, respond NOT_ANSWERABLE.

CONTEXT:
{context}

QUESTION CHAIN:
{question}

Trace through each question using ONLY the context above:
- For each question, find the answer in the context
- Substitute answers into later questions where (N) appears
- If you cannot answer ANY question in the chain, stop and say NOT_ANSWERABLE

<answer>
If all questions answerable: ANSWERABLE: <final answer only>
If any step fails: NOT_ANSWERABLE: <which question failed and why>
</answer>"""


def build_verification_prompt(context: str, question: str) -> str:
    return CHAIN_VERIFICATION_TEMPLATE.format(
        context=context if context else "(no context provided)",
        question=question,
    )
