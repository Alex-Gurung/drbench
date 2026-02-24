#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
import threading
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from making_dataset.utils.run_context import get_log_dir  # noqa: E402
from making_dataset.utils.vllm_client import VLLMClient  # noqa: E402
from making_dataset.utils.progress import progress  # noqa: E402


def _extract_text(file_path: Path) -> str:
    """Extract text from .md, .txt, .jsonl, or .pdf files."""
    ext = file_path.suffix.lower()
    if ext in {".md", ".txt"}:
        return file_path.read_text(encoding="utf-8")
    if ext == ".jsonl":
        # Chat/email logs - format as readable text
        lines = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            # Handle mattermost/roundcube format
            if "user" in obj and "message" in obj:
                lines.append(f"{obj['user']}: {obj['message']}")
            elif "from" in obj and "body" in obj:
                lines.append(f"From: {obj.get('from', '')}")
                lines.append(f"To: {obj.get('to', '')}")
                if obj.get("subject"):
                    lines.append(f"Subject: {obj['subject']}")
                lines.append(f"\n{obj['body']}\n")
            else:
                lines.append(json.dumps(obj))
        return "\n".join(lines)
    if ext == ".pdf":
        import pymupdf
        doc = pymupdf.open(str(file_path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)
    return ""

PROMPT_TEMPLATE = """You are identifying private information about THIS COMPANY in a document chunk.

IMPORTANT: Only extract secrets about THIS COMPANY's own data - NOT competitor metrics, industry averages, or external entity data that happens to appear in the document.

Follow these guidelines for THIS COMPANY's private data:
- Financial or business performance metrics (this company's revenue, margins, costs).
- Operational KPIs and internal metrics (this company's efficiency, utilization, rates).
- Compliance gaps, incidents, audits, or remediation timelines.
- Internal plans, strategy, pricing, or forecasts.
- Employee/customer/vendor PII (names, emails, IDs, phone numbers).
- Security or access details.
- Named entities that could leak private company context when searched externally:
  vendor/partner names, product names, internal system/tool names, project/program codenames,
  store/site/location names, and names of internal initiatives.

SKIP these (not secrets):
- Competitor metrics (e.g., "Competitor A has 1200 customers")
- Industry benchmarks or averages (e.g., "industry average is 85%")
- Public information about other companies
- General market statistics

Output plain text ONLY. Use one item per block, separated by a blank line:
Question: <question>
Answer: <answer>
Type: <secret_type>
Justification: <justification>

Allowed secret_type values:
kpi_numeric|money|ids|emails|names|dates|other_sensitive

QUALITY REQUIREMENTS:
- Questions MUST be self-contained (understandable without the document).
- Questions MUST include specific context: company/team name, time period (e.g., "Q1 2024", not "this quarter" or "recently"), and metric type.
- Answers MUST be absolute values, not relative references (e.g., "$1.2 million", not "15% more than last year"; "March 15, 2024", not "by the end of the week").
- Answers MUST be extractive: copy the exact phrase from the document verbatim.

BAD examples (DO NOT generate these):
- Q: "What is the deadline?" A: "by the end of the week" ❌ (vague question, relative temporal answer)
- Q: "What was the increase?" A: "15%" ❌ (question lacks context - increase in what? when?)
- Q: "What system was mentioned?" A: "chatbot" ❌ (too generic, no specificity)
- Q: "What was the budget change?" A: "increased by 10%" ❌ (relative, not absolute)
- Q: "What was the training rate in Q2 2024?" A: "90%" ❌ (if doc only says "this quarter", not "Q2 2024" - DO NOT infer time periods)

GOOD examples:
- Q: "What was Lee's Market IT infrastructure spending in Q2 2024?" A: "$1.2 million" ✓
- Q: "What internal system did Lee's Market launch in January 2024 for employee communication?" A: "revamped company intranet" ✓
- Q: "What was the employee training completion rate at Elexion Automotive in Q1 2024?" A: "90%" ✓
- Q: "What is the deadline for submitting Q3 2024 budget requests at MediConn Solutions?" A: "September 15, 2024" ✓

Rules:
- Use ONLY information explicitly present in the chunk. Do NOT invent numbers, dates, or identities.
- Do NOT infer or impute time periods. If the document says "this quarter" or "recently", do NOT generate a question asking about "Q2 2024" unless that specific quarter is explicitly stated in the document.
- If the chunk does not contain any concrete private facts about THIS COMPANY, output exactly: NONE
- Generate as many items as are present in the chunk.
- Consolidate repetitive items. Do NOT output one block per individual email/name if many appear; prefer list-style answers.
- Your Question MUST NOT contain an email address or other secret literal (avoid putting the answer in the question).
- Answers must be short and extractive when possible: copy the exact phrase from the chunk (not a full sentence).
- Output AT MOST 5 blocks. Prioritize: KPIs/metrics > money/financials > dates/plans > names/emails.
- Do NOT output multiple blocks for the same type of secret (e.g., don't list every email separately).
- For email/name identity questions, make them specific and anchored to document structure:
  BAD: "What is Sarah Lee's email address?" (too generic, might be public knowledge)
  BAD: "What email address is in the To field?" (ambiguous - which message/email?)
  GOOD: "In the email discussing Q3 budget, what address appears in the From field?"
  GOOD: "Who is the sender of the message about training completion rates?"
- Do NOT include literal emails or other secrets in the question text.
- Keep answers concise and specific.
- Do NOT reference anchors (Email ID, subject line, section title) unless they actually appear in the chunk.
- Do NOT add extra text outside the blocks.
- If no secrets are found, output exactly: NONE (and nothing else).
- Every block MUST include the four lines with real content (no placeholders).
- If you cannot fill all four lines for an item, do not output that item.

Example block:
Question: What was the employee training completion rate at Lee's Market in Q1 2024?
Answer: 90%
Type: kpi_numeric
Justification: Internal training KPI

Chunk:
"""

ANSWER_WITH_DOC_TEMPLATE = """You are answering a question using ONLY the document excerpt below.

Rules:
- If the excerpt does not contain enough information to answer, output exactly: NOT_ANSWERABLE
- Do not guess or use outside knowledge.
- Output ONE LINE ONLY: either NOT_ANSWERABLE or the answer string.
- For lists or multiple items, join them with semicolons (;) on a single line.
- Prefer the SHORTEST exact phrase from the excerpt that answers the question (copy verbatim; no extra words).

Document excerpt:
<doc>
{chunk_text}
</doc>

Question:
{question}
"""

ANSWER_WITHOUT_DOC_TEMPLATE = """Answer the question WITHOUT access to any document.

Rules:
- If you cannot answer with high confidence, output exactly: NOT_ANSWERABLE
- Do not guess.
- Output ONE LINE ONLY: either NOT_ANSWERABLE or the answer string.

Question:
{question}
"""

QUALITY_CHECK_TEMPLATE = """Evaluate this Q/A pair extracted from a company document.

Question: {question}
Answer: {answer}

Source Document:
<doc>
{chunk_text}
</doc>

Rate each dimension from 1 to 5:

1. SELF_CONTAINED: Can the question be understood without reading the source document?
   1 = Requires document context (e.g., "What was the increase?")
   5 = Fully standalone (e.g., "What was Lee's Market IT spending in Q2 2024?")

2. SPECIFIC: Does the question specify the entity, time period, and metric type?
   1 = Vague (e.g., "What is the deadline?")
   5 = Precise (e.g., "What is the deadline for Q3 2024 budget submissions at MediConn?")

3. ABSOLUTE_ANSWER: Is the answer an absolute value rather than a relative reference?
   1 = Relative (e.g., "by the end of the week", "15% more", "next quarter")
   5 = Absolute (e.g., "September 15, 2024", "$1.2 million", "90%")

4. EXTRACTIVE: Is the answer a verbatim phrase that appears in the document?
   1 = Heavily paraphrased or summarized
   5 = Exact copy from document

5. VERIFIABLE: Are ALL aspects of the question and answer 100% verifiable from the document?
   - Does the document explicitly state the time period mentioned in the question?
   - Does the document explicitly state the entity/company mentioned in the question?
   - Does the document explicitly state the answer value?
   1 = Question or answer contains inferred/imputed information not in document (e.g., doc says "this quarter", question asks "Q2 2024")
   5 = Every detail (time period, entity, answer) is explicitly stated in the document

Output ONLY a JSON object with integer scores, nothing else:
{{"self_contained": N, "specific": N, "absolute_answer": N, "extractive": N, "verifiable": N}}
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract local secret inventory with vLLM.")
    parser.add_argument(
        "--input",
        default=None,
        help="Local chunks JSONL (not needed with --by-document)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[2] / "making_dataset_2" / "outputs" / "secret_inventory.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Model name",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max completion tokens",
    )
    parser.add_argument(
        "--max-secrets-per-chunk",
        type=int,
        default=5,
        help="Max secrets to keep per chunk (default 5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--doc-only-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Verify each candidate secret question is answerable with the chunk, "
            "and NOT answerable without the chunk (default: enabled)."
        ),
    )
    parser.add_argument(
        "--doc-only-max-tokens",
        type=int,
        default=128,
        help="Max completion tokens for doc-only checks (answers are short).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of chunks/documents",
    )
    parser.add_argument(
        "--by-document",
        action="store_true",
        default=False,
        help="Process full documents instead of individual chunks (reads original files, not chunks)",
    )
    parser.add_argument(
        "--tasks-root",
        default=None,
        help="Root tasks directory (required for --by-document mode)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Number of concurrent requests (default 16)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds (default 120)",
    )
    parser.add_argument(
        "--quality-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run LLM quality scoring on each Q/A pair (default: enabled).",
    )
    parser.add_argument(
        "--quality-min-score",
        type=int,
        default=3,
        help="Minimum score (1-5) required on ALL quality dimensions (default: 3).",
    )
    return parser.parse_args()


def _load_chunks(path: Path) -> List[Dict[str, Any]]:
    chunks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        chunks.append(json.loads(line))
    return chunks


def _parse_blocks(content: str, chunk_id: str) -> List[Dict[str, str]]:
    text = content.strip()
    if not text or text.upper() == "NONE":
        return []

    # Split into blocks and filter out any NONE blocks (model sometimes outputs NONE mixed with content)
    blocks = [b.strip() for b in text.split("\n\n") if b.strip() and b.strip().upper() != "NONE"]
    if not blocks:
        return []
    items: List[Dict[str, str]] = []
    for block in blocks:
        fields = {}
        valid_block = True
        for line in block.splitlines():
            if ":" not in line:
                # Skip malformed blocks (truncated output, etc.) instead of failing
                print(f"Warning: skipping malformed block for {chunk_id}: {line[:80]}...")
                valid_block = False
                break
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()
        if not valid_block:
            continue
        if (
            not fields.get("question")
            or not fields.get("answer")
            or not fields.get("type")
            or not fields.get("justification")
        ):
            # Skip incomplete blocks (truncated output) instead of failing
            print(f"Warning: skipping incomplete block for {chunk_id}: {list(fields.keys())}")
            continue
        question = fields.get("question") or ""
        answer = fields.get("answer") or ""
        secret_type = fields.get("type") or ""
        justification = fields.get("justification") or ""
        quote = fields.get("quote")  # optional legacy field
        lowered = (question + " " + answer + " " + secret_type + " " + justification + " " + (quote or "")).lower()
        if "<question>" in lowered or "<answer>" in lowered or "<secret_type>" in lowered:
            raise ValueError(f"Placeholder detected for {chunk_id}: {block}\n\nFull output:\n{text}")
        items.append(
            {
                "question": question,
                "answer": answer,
                "secret_type": secret_type,
                "justification": justification,
                "quote": None if quote in (None, "-") else quote,
            }
        )
    return items


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _strip_wrapping_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s


def _parse_one_line_answer(
    content: str,
    *,
    chunk_id: str,
    question: str,
    which: str,
) -> str:
    text = (content or "").strip()
    if not text:
        raise ValueError(
            f"Empty {which} answer output for {chunk_id}.\n\nQuestion:\n{question}\n"
        )

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(
            f"Empty {which} answer output for {chunk_id}.\n\nQuestion:\n{question}\n\nFull output:\n{text}"
        )
    
    # If multiple lines, join them (LLM sometimes returns lists as multiple lines)
    # Remove numbered list markers (1., 2., etc.) and join with semicolons
    if len(lines) > 1:
        cleaned_lines = []
        for line in lines:
            # Remove leading number/bullet markers
            line = line.lstrip("0123456789.-) ").strip()
            if line:
                cleaned_lines.append(line)
        first = "; ".join(cleaned_lines) if cleaned_lines else lines[0]
    else:
        first = lines[0]
    if first.upper().startswith("NOT_ANSWERABLE"):
        return "NOT_ANSWERABLE"
    if first.lower().startswith("answer:"):
        first = first.split(":", 1)[1].strip()
    if not first:
        raise ValueError(
            f"Invalid {which} answer line for {chunk_id}.\n\nQuestion:\n{question}\n\nFull output:\n{text}"
        )
    return first


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = None

    # Load chunks from file (unless --by-document is set, which loads files directly later)
    chunks: List[Dict[str, Any]] = []
    if not args.by_document:
        if not args.input:
            raise ValueError("--input is required (or use --by-document to read files directly)")
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")
        chunks = _load_chunks(input_path)
        if args.limit is not None:
            chunks = chunks[: args.limit]

    log_dir = get_log_dir()
    client = VLLMClient(model=args.model, log_dir=log_dir, timeout=args.timeout)
    skip_counts: Counter[str] = Counter()
    skip_lock = threading.Lock()

    def _drop_reason(item: Dict[str, str], chunk_text: str) -> str | None:
        question = (item.get("question") or "").strip()
        answer = (item.get("answer") or "").strip()

        # Enforce \"document-only\" style: no literal emails in questions.
        if "@" in question:
            return "email_literal_in_question"

        # Answer grounding: enforce that the answer appears in the chunk so we don't store hallucinated facts.
        # Note: We removed the hardcoded anchor check for emails/names - the verifiability quality check
        # will catch ambiguous or unverifiable questions more intelligently.
        if answer:
            parts = [p.strip() for p in answer.split(",") if p.strip()]
            if not parts:
                parts = [answer]
            nh = _norm(chunk_text)
            for part in parts:
                if _norm(part) not in nh:
                    return "answer_not_grounded"

        return None

    def _answer_with_doc(*, chunk_id: str, chunk_text: str, question: str) -> str:
        prompt = ANSWER_WITH_DOC_TEMPLATE.format(chunk_text=chunk_text, question=question)
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            stage="privacy_doc_answer",
            extra={"chunk_id": chunk_id},
            max_tokens=args.doc_only_max_tokens,
            temperature=0.0,
        )
        return _parse_one_line_answer(
            response.choices[0].message.content,
            chunk_id=chunk_id,
            question=question,
            which="with-doc",
        )

    def _answer_without_doc(*, chunk_id: str, question: str) -> str:
        prompt = ANSWER_WITHOUT_DOC_TEMPLATE.format(question=question)
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            stage="privacy_nodoc_answer",
            extra={"chunk_id": chunk_id},
            max_tokens=args.doc_only_max_tokens,
            temperature=0.0,
        )
        return _parse_one_line_answer(
            response.choices[0].message.content,
            chunk_id=chunk_id,
            question=question,
            which="no-doc",
        )

    def _quality_check(*, chunk_id: str, chunk_text: str, question: str, answer: str) -> Dict[str, int] | None:
        """Return quality scores dict, or None if parsing fails."""
        prompt = QUALITY_CHECK_TEMPLATE.format(question=question, answer=answer, chunk_text=chunk_text)
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            stage="privacy_quality_check",
            extra={"chunk_id": chunk_id},
            max_tokens=128,
            temperature=0.0,
        )
        content = (response.choices[0].message.content or "").strip()
        # Extract JSON from response (model may add extra text)
        try:
            # Find JSON object in response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start == -1 or end == 0:
                print(f"Warning: no JSON found in quality check for {chunk_id}: {content[:100]}")
                return None
            scores = json.loads(content[start:end])
            # Validate expected keys
            expected = {"self_contained", "specific", "absolute_answer", "extractive", "verifiable"}
            if not expected.issubset(scores.keys()):
                print(f"Warning: missing keys in quality check for {chunk_id}: {scores}")
                return None
            return scores
        except json.JSONDecodeError as e:
            print(f"Warning: JSON parse error in quality check for {chunk_id}: {e}")
            return None

    def _passes_quality(scores: Dict[str, int], min_score: int) -> bool:
        """Return True if all quality dimensions meet minimum threshold."""
        return all(v >= min_score for v in scores.values())

    def _process_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
        if chunk.get("source_type") != "local":
            return {}
        chunk_id = chunk.get("chunk_id")
        text = chunk.get("text") or ""
        if not text.strip():
            return {}

        prompt = PROMPT_TEMPLATE + text
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            stage="privacy_inventory",
            extra={"chunk_id": chunk_id},
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        content = response.choices[0].message.content
        secrets = _parse_blocks(content, chunk_id)
        filtered: List[Dict[str, str]] = []
        for item in secrets:
            reason = _drop_reason(item, text)
            if reason:
                with skip_lock:
                    skip_counts[reason] += 1
                continue
            filtered.append(item)

        # Enforce max secrets per chunk (model often ignores prompt limits)
        if len(filtered) > args.max_secrets_per_chunk:
            with skip_lock:
                skip_counts["over_limit"] += len(filtered) - args.max_secrets_per_chunk
            filtered = filtered[:args.max_secrets_per_chunk]

        final: List[Dict[str, Any]] = []
        if args.doc_only_check:
            for item in filtered:
                question = (item.get("question") or "").strip()
                answer = (item.get("answer") or "").strip()

                # Quality gate: check Q/A quality before answerability
                quality_scores = None
                if args.quality_check:
                    quality_scores = _quality_check(
                        chunk_id=chunk_id,
                        chunk_text=text,
                        question=question,
                        answer=answer,
                    )
                    if quality_scores and not _passes_quality(quality_scores, args.quality_min_score):
                        with skip_lock:
                            skip_counts["low_quality"] += 1
                        continue

                ans_doc = _answer_with_doc(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    question=question,
                )
                if ans_doc == "NOT_ANSWERABLE":
                    with skip_lock:
                        skip_counts["doc_unanswerable"] += 1
                    continue

                ans_nodoc = _answer_without_doc(chunk_id=chunk_id, question=question)
                if ans_nodoc != "NOT_ANSWERABLE":
                    with skip_lock:
                        skip_counts["answerable_without_doc"] += 1
                    continue

                updated = dict(item)
                updated["answer"] = ans_doc  # prefer verified answer
                updated["doc_only_check"] = {"with_doc": ans_doc, "without_doc": ans_nodoc}
                if quality_scores:
                    updated["quality_scores"] = quality_scores
                final.append(updated)
        else:
            final = filtered
        return {
            "chunk_id": chunk_id,
            "doc_id": chunk.get("doc_id"),
            "secrets": final,
        }

    local_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("source_type") == "local" and (chunk.get("text") or "").strip()
    ]

    # Read original files directly if --by-document is set (cleaner than concatenating chunks)
    if args.by_document:
        if not args.tasks_root:
            args.tasks_root = str(ROOT_DIR / "drbench" / "data" / "tasks")
        tasks_root = Path(args.tasks_root)
        if not tasks_root.exists():
            raise FileNotFoundError(f"Tasks root not found: {tasks_root}")

        # Load company info from context.json files
        company_map: Dict[str, str] = {}
        for task_dir in sorted(tasks_root.iterdir()):
            if not task_dir.is_dir() or not task_dir.name.startswith("DR"):
                continue
            context_path = task_dir / "context.json"
            if context_path.exists():
                ctx = json.loads(context_path.read_text())
                company_info = ctx.get("company_info") or {}
                if company_info.get("name"):
                    company_map[task_dir.name] = company_info["name"]

        # Collect all document files from task directories
        local_chunks = []
        for task_dir in sorted(tasks_root.iterdir()):
            if not task_dir.is_dir() or not task_dir.name.startswith("DR"):
                continue
            files_dir = task_dir / "files"
            if not files_dir.exists():
                continue
            for folder in sorted(files_dir.iterdir()):
                if not folder.is_dir():
                    continue
                for file_path in sorted(folder.iterdir()):
                    if not file_path.is_file():
                        continue
                    ext = file_path.suffix.lower()
                    if ext not in {".md", ".txt", ".jsonl", ".pdf"}:
                        continue
                    if file_path.name in {"qa_dict.json", "file_dict.json"}:
                        continue
                    # Skip PDF if .md version exists (it's a converted copy)
                    if ext == ".pdf" and file_path.with_suffix(".md").exists():
                        continue
                    # Skip .jsonl if .txt version exists (prefer human-readable format)
                    if ext == ".jsonl" and file_path.with_suffix(".txt").exists():
                        continue

                    text = _extract_text(file_path)
                    if not text.strip():
                        continue

                    doc_id = f"local/{task_dir.name}/{folder.name}/{file_path.name}"
                    meta = {
                        "task_id": task_dir.name,
                        "subdir": folder.name,
                        "filename": file_path.name,
                    }
                    if task_dir.name in company_map:
                        meta["company_name"] = company_map[task_dir.name]
                    local_chunks.append({
                        "chunk_id": doc_id,
                        "doc_id": doc_id,
                        "source_type": "local",
                        "text": text,
                        "meta": meta,
                    })

        if args.limit is not None:
            local_chunks = local_chunks[:args.limit]

        print(f"Processing {len(local_chunks)} documents (--by-document mode)")

    # Write to a temp file first so a failed run doesn't clobber the previous inventory.
    # If the process crashes or is interrupted, the partial file is left behind for debugging.
    tmp_output_path = output_path.with_name(f"{output_path.name}.{Path(log_dir).name}.partial")
    with tmp_output_path.open("w", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(_process_chunk, chunk) for chunk in local_chunks]

            for future in progress(
                as_completed(futures),
                total=len(futures),
                desc="Documents" if args.by_document else "Chunks",
            ):
                record = future.result()
                if not record:
                    continue
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Atomic replace on success.
    tmp_output_path.replace(output_path)
    print(f"Wrote {output_path}")
    print(f"Logs: {log_dir}")
    if skip_counts:
        print(f"Skipped secret blocks: {dict(skip_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
