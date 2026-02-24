#!/usr/bin/env python3
"""Test privacy_tagger on a few documents with pretty output."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from making_dataset.generate.privacy_tagger import (
    PROMPT_TEMPLATE,
    QUALITY_CHECK_TEMPLATE,
    ANSWER_WITH_DOC_TEMPLATE,
    ANSWER_WITHOUT_DOC_TEMPLATE,
    _parse_blocks,
    _norm,
)
from making_dataset.utils.vllm_client import VLLMClient


def pretty_print_secrets(secrets: list, title: str = "Secrets"):
    """Pretty print extracted secrets."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    if not secrets:
        print("  (none)")
        return
    for i, s in enumerate(secrets, 1):
        print(f"\n  [{i}] {s.get('secret_type', 'unknown').upper()}")
        print(f"      Q: {s.get('question', '')}")
        print(f"      A: {s.get('answer', '')}")
        if s.get('quality_scores'):
            scores = s['quality_scores']
            print(f"      Quality: self_contained={scores.get('self_contained')}, "
                  f"specific={scores.get('specific')}, "
                  f"absolute={scores.get('absolute_answer')}, "
                  f"extractive={scores.get('extractive')}, "
                  f"verifiable={scores.get('verifiable')}")
        if s.get('justification'):
            print(f"      Justification: {s.get('justification', '')[:80]}...")
        if s.get('doc_only_check'):
            doc_check = s['doc_only_check']
            print(f"      Doc check: with_doc='{doc_check.get('with_doc', '')[:50]}...', "
                  f"without_doc='{doc_check.get('without_doc', '')}'")


def test_single_chunk(client: VLLMClient, text: str, chunk_id: str = "test_chunk"):
    """Process a single chunk and show all intermediate steps."""
    print("\n" + "="*60)
    print("  INPUT DOCUMENT")
    print("="*60)
    print(text[:500] + ("..." if len(text) > 500 else ""))
    
    # Step 1: Generate secrets
    print("\n" + "-"*60)
    print("  STEP 1: GENERATION")
    print("-"*60)
    prompt = PROMPT_TEMPLATE + text
    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        stage="test_generation",
        max_tokens=4096,
        temperature=0.0,
    )
    raw_output = response.choices[0].message.content
    print(f"Raw LLM output:\n{raw_output}")
    
    secrets = _parse_blocks(raw_output, chunk_id)
    print(f"\nParsed {len(secrets)} secret(s)")
    
    # Step 2: Quality check each
    print("\n" + "-"*60)
    print("  STEP 2: QUALITY CHECK")
    print("-"*60)
    for i, s in enumerate(secrets):
        q, a = s['question'], s['answer']
        print(f"\n[{i+1}] Q: {q}")
        print(f"    A: {a}")
        
        quality_prompt = QUALITY_CHECK_TEMPLATE.format(question=q, answer=a, chunk_text=text)
        quality_response = client.chat(
            messages=[{"role": "user", "content": quality_prompt}],
            stage="test_quality",
            max_tokens=128,
            temperature=0.0,
        )
        quality_output = quality_response.choices[0].message.content
        print(f"    Quality scores: {quality_output.strip()}")
        
        try:
            start = quality_output.find("{")
            end = quality_output.rfind("}") + 1
            scores = json.loads(quality_output[start:end])
            min_score = min(scores.values())
            passes = min_score >= 3
            failed_dims = [k for k, v in scores.items() if v < 3]
            status = 'PASS' if passes else f'FAIL (low: {", ".join(failed_dims)})'
            print(f"    Min score: {min_score} -> {status}")
            s['quality_scores'] = scores
            s['quality_pass'] = passes
            s['failed_dims'] = failed_dims if not passes else []
        except Exception as e:
            print(f"    Parse error: {e}")
            s['quality_pass'] = None
            s['failed_dims'] = []
    
    # Step 3: Doc-only check (only for those that passed quality)
    print("\n" + "-"*60)
    print("  STEP 3: DOC-ONLY CHECK")
    print("-"*60)
    for i, s in enumerate(secrets):
        if s.get('quality_pass') is False:
            print(f"\n[{i+1}] SKIPPED (failed quality)")
            continue
            
        q = s['question']
        print(f"\n[{i+1}] Q: {q}")
        
        # With doc
        with_doc_prompt = ANSWER_WITH_DOC_TEMPLATE.format(chunk_text=text, question=q)
        with_doc_response = client.chat(
            messages=[{"role": "user", "content": with_doc_prompt}],
            stage="test_with_doc",
            max_tokens=128,
            temperature=0.0,
        )
        with_doc = with_doc_response.choices[0].message.content.strip()
        print(f"    With doc: {with_doc}")
        
        # Without doc
        without_doc_prompt = ANSWER_WITHOUT_DOC_TEMPLATE.format(question=q)
        without_doc_response = client.chat(
            messages=[{"role": "user", "content": without_doc_prompt}],
            stage="test_without_doc",
            max_tokens=128,
            temperature=0.0,
        )
        without_doc = without_doc_response.choices[0].message.content.strip()
        print(f"    Without doc: {without_doc}")
        
        s['doc_only_check'] = {'with_doc': with_doc, 'without_doc': without_doc}
        
        # Determine pass/fail
        if "NOT_ANSWERABLE" in with_doc.upper():
            print(f"    -> FAIL: Not answerable with doc")
            s['doc_pass'] = False
        elif "NOT_ANSWERABLE" not in without_doc.upper():
            print(f"    -> FAIL: Answerable without doc (public knowledge)")
            s['doc_pass'] = False
        else:
            print(f"    -> PASS")
            s['doc_pass'] = True
    
    # Final summary
    final = [s for s in secrets if s.get('quality_pass') is not False and s.get('doc_pass')]
    pretty_print_secrets(final, "FINAL ACCEPTED SECRETS")
    
    rejected_quality = [s for s in secrets if s.get('quality_pass') is False]
    if rejected_quality:
        pretty_print_secrets(rejected_quality, "REJECTED (low quality)")
    
    rejected_doc = [s for s in secrets if s.get('quality_pass') is not False and s.get('doc_pass') is False]
    if rejected_doc:
        pretty_print_secrets(rejected_doc, "REJECTED (doc-only check)")
    
    # Statistics
    print(f"\n{'='*60}")
    print("  SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"  Total generated: {len(secrets)}")
    print(f"  Accepted: {len(final)}")
    print(f"  Rejected (quality): {len(rejected_quality)}")
    print(f"  Rejected (doc-only): {len(rejected_doc)}")
    if secrets:
        acceptance_rate = len(final) / len(secrets) * 100
        print(f"  Acceptance rate: {acceptance_rate:.1f}%")
    
    return final


def main():
    import os
    os.environ.setdefault("VLLM_API_URL", "http://127.0.0.1:8000")
    
    client = VLLMClient(model="Qwen/Qwen3-30B-A3B-Instruct-2507", log_dir="/tmp/test_privacy")
    
    # Test document 1: Good example with specific dates/metrics
    test_doc_1 = """Q3 2024 Workforce Metrics and Analysis Report

Employee Engagement and Satisfaction Survey Results

Our Q2 2024 employee engagement survey revealed a 10% increase in employee satisfaction with internal communication channels. Notably, 85% of respondents reported feeling 'well-informed' about company initiatives. This uptick is attributed to the revamped company intranet launched in January 2024. As a result, the internal communications team has been recognized for their outstanding efforts.

Training and Development Program Effectiveness Analysis

In FY2023, Lee's Market invested $1.2 million in employee training programs, resulting in a 25% increase in internal promotions. The average participant rating for these programs was 4.5/5, indicating high satisfaction."""

    # Test document 2: Chat with vague temporal references
    test_doc_2 = """Sarah Lee: Hey team, quick update on the Q3 budget
Michael Chen: What's the deadline for submissions?
Sarah Lee: By the end of the week, make sure to get your requests in
Michael Chen: Got it. Also, the training completion rate looks good
Sarah Lee: Yes, we hit 90% this quarter. Big improvement from last time
David Kim: The IT infrastructure spending was $1.2 million in Q2 2024
Sarah Lee: Thanks David. Let's discuss in our next meeting"""

    print("\n" + "#"*60)
    print("  TEST 1: Workforce Metrics Report (Lee's Market)")
    print("#"*60)
    test_single_chunk(client, test_doc_1, "test_workforce_report")
    
    print("\n\n" + "#"*60)
    print("  TEST 2: Chat with vague references")
    print("#"*60)
    test_single_chunk(client, test_doc_2, "test_chat")


if __name__ == "__main__":
    main()
