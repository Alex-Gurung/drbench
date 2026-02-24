#!/usr/bin/env python3
"""View privacy_tagger processing on real documents with full detail."""
from __future__ import annotations

import argparse
import json
import os
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
    _extract_text,
)
from making_dataset.utils.vllm_client import VLLMClient
from making_dataset.utils.run_context import get_log_dir


def print_section(title: str, char: str = "="):
    """Print a section header."""
    print(f"\n{char*80}")
    print(f"  {title}")
    print(f"{char*80}")


def print_prompt(title: str, prompt: str):
    """Print a prompt with clear formatting."""
    print(f"\n{'-'*80}")
    print(f"  {title}")
    print(f"{'-'*80}")
    print(prompt)
    print(f"{'-'*80}")


def view_document(
    client: VLLMClient,
    file_path: Path,
    task_id: str | None = None,
    company_name: str | None = None,
):
    """Process a single document and show all steps in detail."""
    
    # Extract text
    text = _extract_text(file_path)
    doc_id = f"local/{task_id}/{file_path.parent.name}/{file_path.name}" if task_id else str(file_path)
    
    print_section(f"PROCESSING DOCUMENT: {file_path.name}", "=")
    print(f"\nFile path: {file_path}")
    print(f"Document ID: {doc_id}")
    if task_id:
        print(f"Task ID: {task_id}")
    if company_name:
        print(f"Company: {company_name}")
    
    # Show full document
    print_section("FULL DOCUMENT TEXT", "=")
    print(text)
    
    # Step 1: Generation
    print_section("STEP 1: GENERATION", "=")
    generation_prompt = PROMPT_TEMPLATE + text
    print_prompt("Generation Prompt", generation_prompt)
    
    response = client.chat(
        messages=[{"role": "user", "content": generation_prompt}],
        stage="view_generation",
        max_tokens=4096,
        temperature=0.0,
    )
    raw_output = response.choices[0].message.content
    
    print(f"\n{'='*80}")
    print("  RAW LLM OUTPUT")
    print(f"{'='*80}")
    print(raw_output)
    
    secrets = _parse_blocks(raw_output, doc_id)
    print(f"\nParsed {len(secrets)} secret(s) from output")
    
    if not secrets:
        print("\nNo secrets extracted. Document may not contain private information.")
        return
    
    # Step 2: Quality check for each secret
    print_section("STEP 2: QUALITY CHECK", "=")
    for i, s in enumerate(secrets, 1):
        q = s['question']
        a = s['answer']
        secret_type = s.get('secret_type', 'unknown')
        justification = s.get('justification', '')
        
        print(f"\n{'#'*80}")
        print(f"  SECRET #{i}: {secret_type.upper()}")
        print(f"{'#'*80}")
        print(f"Question: {q}")
        print(f"Answer: {a}")
        print(f"Justification: {justification}")
        
        quality_prompt = QUALITY_CHECK_TEMPLATE.format(question=q, answer=a, chunk_text=text)
        print_prompt(f"Quality Check Prompt (Secret #{i})", quality_prompt)
        
        quality_response = client.chat(
            messages=[{"role": "user", "content": quality_prompt}],
            stage="view_quality",
            max_tokens=128,
            temperature=0.0,
        )
        quality_output = quality_response.choices[0].message.content
        
        print(f"\n{'='*80}")
        print("  QUALITY CHECK OUTPUT")
        print(f"{'='*80}")
        print(quality_output)
        
        try:
            start = quality_output.find("{")
            end = quality_output.rfind("}") + 1
            scores = json.loads(quality_output[start:end])
            min_score = min(scores.values())
            passes = min_score >= 3
            failed_dims = [k for k, v in scores.items() if v < 3]
            
            print(f"\n{'='*80}")
            print("  QUALITY SCORES")
            print(f"{'='*80}")
            for dim, score in scores.items():
                status = "✓" if score >= 3 else "✗"
                print(f"  {status} {dim}: {score}/5")
            print(f"\n  Min score: {min_score}")
            print(f"  Status: {'PASS' if passes else f'FAIL (low dimensions: {", ".join(failed_dims)})'}")
            
            s['quality_scores'] = scores
            s['quality_pass'] = passes
            s['failed_dims'] = failed_dims if not passes else []
        except Exception as e:
            print(f"\nERROR parsing quality scores: {e}")
            print(f"Raw output: {quality_output}")
            s['quality_pass'] = None
            s['failed_dims'] = []
    
    # Step 3: Doc-only check (only for those that passed quality)
    print_section("STEP 3: DOC-ONLY CHECK", "=")
    for i, s in enumerate(secrets, 1):
        if s.get('quality_pass') is False:
            print(f"\n{'#'*80}")
            print(f"  SECRET #{i}: SKIPPED (failed quality check)")
            print(f"{'#'*80}")
            continue
        
        q = s['question']
        print(f"\n{'#'*80}")
        print(f"  SECRET #{i}: DOC-ONLY CHECK")
        print(f"{'#'*80}")
        print(f"Question: {q}")
        
        # With doc
        with_doc_prompt = ANSWER_WITH_DOC_TEMPLATE.format(chunk_text=text, question=q)
        print_prompt(f"Answer WITH Document Prompt (Secret #{i})", with_doc_prompt)
        
        with_doc_response = client.chat(
            messages=[{"role": "user", "content": with_doc_prompt}],
            stage="view_with_doc",
            max_tokens=128,
            temperature=0.0,
        )
        with_doc = with_doc_response.choices[0].message.content.strip()
        
        print(f"\n{'='*80}")
        print("  ANSWER WITH DOCUMENT")
        print(f"{'='*80}")
        print(with_doc)
        
        # Without doc
        without_doc_prompt = ANSWER_WITHOUT_DOC_TEMPLATE.format(question=q)
        print_prompt(f"Answer WITHOUT Document Prompt (Secret #{i})", without_doc_prompt)
        
        without_doc_response = client.chat(
            messages=[{"role": "user", "content": without_doc_prompt}],
            stage="view_without_doc",
            max_tokens=128,
            temperature=0.0,
        )
        without_doc = without_doc_response.choices[0].message.content.strip()
        
        print(f"\n{'='*80}")
        print("  ANSWER WITHOUT DOCUMENT")
        print(f"{'='*80}")
        print(without_doc)
        
        s['doc_only_check'] = {'with_doc': with_doc, 'without_doc': without_doc}
        
        # Determine pass/fail
        if "NOT_ANSWERABLE" in with_doc.upper():
            print(f"\n  → FAIL: Not answerable with document")
            s['doc_pass'] = False
        elif "NOT_ANSWERABLE" not in without_doc.upper():
            print(f"\n  → FAIL: Answerable without document (public knowledge)")
            s['doc_pass'] = False
        else:
            print(f"\n  → PASS: Answerable only with document")
            s['doc_pass'] = True
    
    # Final summary
    print_section("FINAL SUMMARY", "=")
    final = [s for s in secrets if s.get('quality_pass') is not False and s.get('doc_pass')]
    rejected_quality = [s for s in secrets if s.get('quality_pass') is False]
    rejected_doc = [s for s in secrets if s.get('quality_pass') is not False and s.get('doc_pass') is False]
    
    print(f"\nTotal secrets generated: {len(secrets)}")
    print(f"  ✓ Accepted: {len(final)}")
    print(f"  ✗ Rejected (quality): {len(rejected_quality)}")
    print(f"  ✗ Rejected (doc-only): {len(rejected_doc)}")
    if secrets:
        acceptance_rate = len(final) / len(secrets) * 100
        print(f"\nAcceptance rate: {acceptance_rate:.1f}%")
    
    if final:
        print(f"\n{'='*80}")
        print("  ACCEPTED SECRETS")
        print(f"{'='*80}")
        for i, s in enumerate(final, 1):
            print(f"\n[{i}] {s.get('secret_type', 'unknown').upper()}")
            print(f"    Q: {s.get('question', '')}")
            print(f"    A: {s.get('answer', '')}")
            if s.get('quality_scores'):
                scores = s['quality_scores']
                print(f"    Quality: {scores}")
            if s.get('doc_only_check'):
                doc_check = s['doc_only_check']
                print(f"    Doc check: with_doc='{doc_check.get('with_doc', '')}', "
                      f"without_doc='{doc_check.get('without_doc', '')}'")
    
    if rejected_quality:
        print(f"\n{'='*80}")
        print("  REJECTED (LOW QUALITY)")
        print(f"{'='*80}")
        for i, s in enumerate(rejected_quality, 1):
            print(f"\n[{i}] {s.get('secret_type', 'unknown').upper()}")
            print(f"    Q: {s.get('question', '')}")
            print(f"    A: {s.get('answer', '')}")
            if s.get('quality_scores'):
                scores = s['quality_scores']
                print(f"    Quality scores: {scores}")
            if s.get('failed_dims'):
                print(f"    Failed dimensions: {', '.join(s['failed_dims'])}")
    
    if rejected_doc:
        print(f"\n{'='*80}")
        print("  REJECTED (DOC-ONLY CHECK)")
        print(f"{'='*80}")
        for i, s in enumerate(rejected_doc, 1):
            print(f"\n[{i}] {s.get('secret_type', 'unknown').upper()}")
            print(f"    Q: {s.get('question', '')}")
            print(f"    A: {s.get('answer', '')}")
            if s.get('doc_only_check'):
                doc_check = s['doc_only_check']
                print(f"    With doc: {doc_check.get('with_doc', '')}")
                print(f"    Without doc: {doc_check.get('without_doc', '')}")
    
    # Show log file location
    log_dir = get_log_dir()
    print(f"\n{'='*80}")
    print("  LOG FILES")
    print(f"{'='*80}")
    print(f"All LLM requests/responses logged to: {log_dir}")
    print(f"Look for files matching: view_*")
    
    return final


def main():
    parser = argparse.ArgumentParser(description="View privacy_tagger processing on real documents")
    parser.add_argument(
        "file",
        type=Path,
        help="Path to document file (.md, .txt, .jsonl, .pdf)",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="Task ID (e.g., DR0005) for document ID construction",
    )
    parser.add_argument(
        "--company",
        default=None,
        help="Company name for display",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Model name",
    )
    args = parser.parse_args()
    
    if not args.file.exists():
        print(f"Error: File not found: {args.file}")
        return 1
    
    os.environ.setdefault("VLLM_API_URL", "http://127.0.0.1:8000")
    
    client = VLLMClient(model=args.model, log_dir=get_log_dir())
    
    view_document(
        client=client,
        file_path=args.file,
        task_id=args.task_id,
        company_name=args.company,
    )
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
