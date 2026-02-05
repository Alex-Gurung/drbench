#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from making_dataset.index.unified_searcher import UnifiedSearcher  # noqa: E402
from making_dataset.utils.run_context import get_log_dir  # noqa: E402
from making_dataset.utils.vllm_client import VLLMClient  # noqa: E402


ANSWER_WITH_EVIDENCE_TEMPLATE = """You are answering a question using ONLY the evidence excerpts below.

Rules:
- If you cannot answer using ONLY the excerpts, output exactly: NOT_ANSWERABLE
- Do not guess. Do not use outside knowledge.
- Output ONE LINE ONLY: either NOT_ANSWERABLE or the final answer string.

Evidence excerpts:
<excerpt_a>
{local_excerpt}
</excerpt_a>

<excerpt_b>
{web_excerpt}
</excerpt_b>

Question:
{question}
"""

REWRITE_QUESTION_TEMPLATE = """You are writing a multi-hop benchmark question.

Constraints:
- Do NOT mention: local, web, internal, external, document, excerpt, chunk, corpus, snippet, evidence.
- Do NOT include percent signs in the question.
- Do NOT include either of the two percentage values (even without a % sign).
- The question MUST ask for the absolute difference between two percentages, in percentage points.
- Make it sound natural and compositional (puzzle-like), and reference distinctive details from BOTH snippets
  so someone could find them by searching, but keep it to <= 6 sentences.

You are given two short snippets. Each contains one highlighted percentage like <<25%>>.

Snippet A:
{local_excerpt_hl}

Snippet B:
{web_excerpt_hl}

Write the question. Output ONLY the question text.
"""


FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# NOTE: Don't use `\b` around `%` (it's a non-word char, so `\b` often fails).
PERCENT_RE = re.compile(r"(\d{1,4}(?:\.\d+)?)\s*%")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "POC mixed dataset generator: combine one local secret % with one web % and ask a computed question "
            "that requires BOTH, enforced with 3-way ablation."
        )
    )
    parser.add_argument(
        "--local-chunks",
        default="/home/toolkit/nice_code/drbench/making_dataset/outputs/chunks_local.jsonl",
        help="Local chunks JSONL.",
    )
    parser.add_argument(
        "--secrets",
        default="/home/toolkit/nice_code/drbench/making_dataset/outputs/secret_inventory.jsonl",
        help="secret_inventory.jsonl (from privacy_tagger.py).",
    )
    parser.add_argument(
        "--output",
        default="/home/toolkit/nice_code/drbench/making_dataset/outputs/mixed.jsonl",
        help="Output dataset JSONL path.",
    )
    parser.add_argument(
        "--workspace-id",
        default="drbench_mixed_poc_v1",
        help="Workspace identifier.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=20,
        help="Number of tasks to generate (default: 20).",
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=4,
        help="Hop count (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed (default: 0).",
    )
    parser.add_argument(
        "--web-bm25-index",
        default="/home/toolkit/BrowseComp-Plus/indexes/bm25",
        help="BrowseComp-Plus BM25 index path (Pyserini).",
    )
    parser.add_argument(
        "--web-backend",
        default="bm25_rerank_dense",
        choices=["bm25", "dense", "bm25_rerank_dense"],
        help="Web retrieval backend for UnifiedSearcher.",
    )
    parser.add_argument(
        "--web-dense-index-glob",
        default="/home/toolkit/BrowseComp-Plus/indexes/qwen3-embedding-0.6b/corpus.shard*_of_4.pkl",
        help="Dense index shards glob (used if --web-backend != bm25).",
    )
    parser.add_argument(
        "--web-bm25-candidates-k",
        type=int,
        default=200,
        help="BM25 candidate depth for bm25->dense rerank (default: 200).",
    )
    parser.add_argument(
        "--max-web-chars",
        type=int,
        default=200_000,
        help=(
            "Max chars of a web doc BODY to scan when looking for candidate percentages (default: 200k). "
            "If a doc is longer, we still consider it, but only scan an early prefix for speed."
        ),
    )
    parser.add_argument(
        "--web-window-chars",
        type=int,
        default=2500,
        help="Keyword-window size in chars when searching for percentages near query terms (default: 2500).",
    )
    parser.add_argument(
        "--web-max-windows",
        type=int,
        default=5,
        help="Max number of keyword windows per web doc to scan for percentages (default: 5).",
    )
    parser.add_argument(
        "--min-web-overlap",
        type=int,
        default=1,
        help=(
            "Require this many keyword overlaps between the local question and the chosen web-percent context "
            "(default: 1; set to 0 to be less strict)."
        ),
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="vLLM model name.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max completion tokens for ablation checks (default: 128).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0).",
    )
    parser.add_argument(
        "--limit-secrets",
        type=int,
        default=None,
        help="Optional limit on number of secret-bearing chunks loaded (for smoke runs).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Max sampling attempts (default: max(200, num_tasks*50)).",
    )
    parser.add_argument(
        "--ablation-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run 3-way (both/local-only/web-only) ablation checks with vLLM (default: enabled).",
    )
    parser.add_argument(
        "--rewrite-question",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use vLLM to rewrite the final dataset question into a more compositional, natural style "
            "(default: enabled; requires vLLM)."
        ),
    )
    parser.add_argument(
        "--rewrite-max-tokens",
        type=int,
        default=256,
        help="Max completion tokens for question rewrite (default: 256).",
    )
    parser.add_argument(
        "--allow-identity-secrets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow secret_type in {names, emails} (default: disabled; focus on KPIs/metrics).",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_chunks_map(path: Path) -> Dict[str, Dict[str, Any]]:
    chunk_map: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("chunk_id")
            if not cid:
                raise ValueError(f"Missing chunk_id in chunks file: {path}")
            chunk_map[str(cid)] = obj
    return chunk_map


def _strip_wrapping_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s


def _highlight_first(text: str, needle: str) -> str:
    """
    Mark the first occurrence of `needle` with <<...>> to make the ablation prompt unambiguous.
    If not found, return the original text.
    """
    if not needle:
        return text
    i = text.find(needle)
    if i < 0:
        return text
    return text[:i] + f"<<{needle}>>" + text[i + len(needle) :]


def _find_percent_match(text: str, value: float) -> Optional[Tuple[str, int, int]]:
    """
    Find an exact %-span in `text` whose numeric value matches `value` (within tolerance).
    Returns (matched_text, char_start, char_end) or None.
    """
    best = None
    for m in PERCENT_RE.finditer(text):
        try:
            v = float(m.group(1))
        except Exception:
            continue
        if abs(v - value) <= 1e-6:
            return m.group(0), m.start(), m.end()
        # Keep the closest match as a fallback (useful when vLLM answers round/truncate).
        dv = abs(v - value)
        if best is None or dv < best[0]:
            best = (dv, m.group(0), m.start(), m.end())
    if best is None:
        return None
    dv, s, a, b = best
    # Only accept a near match (e.g., 70 vs 70.0). Avoid drifting to unrelated percents.
    if dv <= 0.1:
        return s, a, b
    return None


def _parse_frontmatter(text: str) -> Dict[str, str]:
    m = FRONTMATTER_RE.search(text)
    if not m:
        return {}
    body = m.group(1)
    meta: Dict[str, str] = {}
    for line in body.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        meta[k.strip().lower()] = v.strip()
    return meta


def _parse_percent(s: str) -> Optional[float]:
    m = PERCENT_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _format_percentage_points(x: float) -> str:
    # Keep answers short + stable.
    if abs(x - round(x)) < 1e-6:
        return f"{int(round(x))} percentage points"
    return f"{x:.1f} percentage points"


def _strip_company_names(s: str) -> str:
    # Local company names do not exist in the web corpus and harm retrieval.
    # Keep this conservative; it's better to under-strip than to destroy meaning.
    out = s
    out = re.sub(r"\bLee's Market\b", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\bMediConn\b", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\bElexion\b", "", out, flags=re.IGNORECASE)
    # Clean up possessives like "Lee's Market's".
    out = re.sub(r"\b('s)\b", "", out)
    return " ".join(out.split())


def _merge_windows(windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not windows:
        return []
    ws = sorted(windows, key=lambda x: (x[0], x[1]))
    merged: List[Tuple[int, int]] = [ws[0]]
    for a, b in ws[1:]:
        pa, pb = merged[-1]
        if a <= pb:
            merged[-1] = (pa, max(pb, b))
        else:
            merged.append((a, b))
    return merged


def _answer_norm(s: str) -> str:
    s = (s or "").strip()
    # Allow trivial punctuation variance.
    if s.endswith("."):
        s = s[:-1].strip()
    return s


def _parse_first_number(s: str) -> Optional[float]:
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _answer_one_line(
    client: VLLMClient,
    *,
    stage: str,
    chunk_id: str,
    question: str,
    local_excerpt: str,
    web_excerpt: str,
    max_tokens: int,
    temperature: float,
) -> Tuple[str, Dict[str, int]]:
    prompt = ANSWER_WITH_EVIDENCE_TEMPLATE.format(
        local_excerpt=local_excerpt.strip(),
        web_excerpt=web_excerpt.strip(),
        question=question.strip(),
    )
    resp = client.chat(
        messages=[{"role": "user", "content": prompt}],
        stage=stage,
        extra={"chunk_id": chunk_id},
        max_tokens=max_tokens,
        temperature=temperature,
    )
    usage = resp.usage
    usage_dict = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    content = (resp.choices[0].message.content or "").strip()
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    # vLLM models sometimes ignore the "ONE LINE ONLY" instruction and emit an explanation
    # plus a final NOT_ANSWERABLE. Be strict about semantics (answerable vs not) but don't
    # crash the whole run on formatting issues.
    if "NOT_ANSWERABLE" in content.upper() or any(ln.upper().startswith("NOT_ANSWERABLE") for ln in lines):
        return "NOT_ANSWERABLE", usage_dict
    if not content:
        raise ValueError("Empty model output")
    return content, usage_dict


def _one_line(s: str) -> str:
    return " ".join((s or "").split())


def _is_good_rewrite(question: str, *, banned_numbers: Optional[List[float]] = None) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    ql = q.lower()
    # Don't leak corpus hints in the final question.
    forbidden = (
        " local",
        " web",
        "internal",
        "external",
        "document",
        "excerpt",
        "chunk",
        "corpus",
        "snippet",
        "evidence",
    )
    if any(tok in ql for tok in forbidden):
        return False
    # Avoid leaking the answer percent values.
    if "%" in q:
        return False
    if banned_numbers:
        nums = re.findall(r"\d+(?:\.\d+)?", q)
        for ns in nums:
            try:
                v = float(ns)
            except Exception:
                continue
            for b in banned_numbers:
                try:
                    bv = float(b)
                except Exception:
                    continue
                if abs(v - bv) <= 0.1:
                    return False
    # Keep the compute target explicit.
    if "difference" not in ql:
        return False
    if "percentage point" not in ql:
        return False
    return True


def _rewrite_dataset_question(
    client: VLLMClient,
    *,
    chunk_id: str,
    local_excerpt_hl: str,
    web_excerpt_hl: str,
    max_tokens: int,
    temperature: float,
) -> Tuple[str, Dict[str, int]]:
    prompt = REWRITE_QUESTION_TEMPLATE.format(
        local_excerpt_hl=local_excerpt_hl.strip(),
        web_excerpt_hl=web_excerpt_hl.strip(),
    )
    resp = client.chat(
        messages=[{"role": "user", "content": prompt}],
        stage="mixed_question_rewrite",
        extra={"chunk_id": chunk_id},
        max_tokens=max_tokens,
        temperature=temperature,
    )
    usage = resp.usage
    usage_dict = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    content = (resp.choices[0].message.content or "").strip()
    return _one_line(content), usage_dict


def main() -> int:
    args = _parse_args()
    local_chunks_path = Path(args.local_chunks)
    secrets_path = Path(args.secrets)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.num_tasks <= 0:
        raise ValueError("--num-tasks must be > 0")
    if args.hops < 2:
        raise ValueError("--hops must be >= 2")
    if not local_chunks_path.exists():
        raise FileNotFoundError(f"Local chunks not found: {local_chunks_path}")
    if not secrets_path.exists():
        raise FileNotFoundError(f"Secrets file not found: {secrets_path}")

    # Unified search (local BM25 + web BM25, optionally web dense).
    searcher = UnifiedSearcher(
        local_chunks_path=str(local_chunks_path),
        web_bm25_index_path=args.web_bm25_index,
        web_dense_index_glob=(args.web_dense_index_glob if args.web_backend != "bm25" else None),
    )

    chunk_map = _load_chunks_map(local_chunks_path)

    secrets_recs = _load_jsonl(secrets_path)
    if args.limit_secrets is not None:
        secrets_recs = secrets_recs[: args.limit_secrets]

    candidates: List[Tuple[str, Dict[str, Any]]] = []
    seen_secret_items = 0
    seen_with_q_a = 0
    seen_skipped_identity = 0
    seen_with_percent_answer = 0
    for rec in secrets_recs:
        cid = rec.get("chunk_id")
        if not cid:
            continue
        for item in rec.get("secrets") or []:
            seen_secret_items += 1
            if not (item.get("question") and item.get("answer")):
                continue
            seen_with_q_a += 1
            st = (item.get("secret_type") or "").strip().lower()
            if not args.allow_identity_secrets and st in {"emails", "names"}:
                seen_skipped_identity += 1
                continue
            # For the POC we only build computed mixed questions from percent-valued secrets.
            if _parse_percent(str(item.get("answer") or "")) is None:
                continue
            seen_with_percent_answer += 1
            candidates.append((str(cid), item))

    if not candidates:
        raise ValueError(
            "No suitable secret items found.\n"
            f"- secrets_path: {secrets_path}\n"
            f"- secret_records_loaded: {len(secrets_recs)}\n"
            f"- secret_items_seen: {seen_secret_items}\n"
            f"- secret_items_with_question_answer: {seen_with_q_a}\n"
            f"- secret_items_skipped_identity: {seen_skipped_identity}\n"
            f"- secret_items_with_percent_answer: {seen_with_percent_answer}\n\n"
            "Did you run privacy_tagger.py (and keep its default --doc-only-check)?"
        )

    log_dir = get_log_dir()
    client = None
    if args.ablation_check or args.rewrite_question:
        client = VLLMClient(model=args.model, log_dir=log_dir)

    rng = random.Random(args.seed)

    usage_by_stage: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    skip_counts: Dict[str, int] = defaultdict(int)

    tasks: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = (
        int(args.max_attempts)
        if args.max_attempts is not None
        else max(200, args.num_tasks * 50)
    )

    for _ in tqdm(range(max_attempts), desc="Generating mixed tasks", unit="try"):
        if len(tasks) >= args.num_tasks:
            break
        attempts += 1

        local_cid, secret = rng.choice(candidates)
        local_chunk = chunk_map.get(local_cid)
        if not local_chunk:
            skip_counts["missing_local_chunk"] += 1
            continue
        local_text = local_chunk.get("text") or ""
        if not local_text.strip():
            skip_counts["empty_local_text"] += 1
            continue

        local_question = (secret.get("question") or "").strip()
        local_answer = (secret.get("answer") or "").strip()
        if not local_question or not local_answer:
            skip_counts["missing_local_q_or_a"] += 1
            continue

        local_value = _parse_percent(local_answer)
        if local_value is None:
            skip_counts["local_answer_no_percent"] += 1
            continue
        local_match = _find_percent_match(local_text, local_value)
        if local_match is None:
            skip_counts["local_percent_not_found_in_chunk"] += 1
            continue
        local_percent_str, lq_start, lq_end = local_match

        # Build a web query that is derived from the local question (dependency), but does not include the answer.
        # Strip local-only names that would hurt web retrieval.
        web_query = _strip_company_names(local_question)

        web_hits = searcher.search(
            web_query,
            corpus="web",
            k=50,
            web_backend=args.web_backend,
            web_bm25_candidates_k=args.web_bm25_candidates_k,
        )
        if not web_hits:
            skip_counts["web_no_hits"] += 1
            continue

        # Pick a web doc + a percent inside it. Heuristic: choose a percent whose surrounding context
        # shares keywords with the local question.
        q_words = [
            w
            for w in re.findall(r"[a-z]{4,}", local_question.lower())
            if w
            not in {
                "what",
                "which",
                "when",
                "where",
                "who",
                "whom",
                "this",
                "that",
                "from",
                "with",
                "into",
                "then",
                "than",
                "after",
                "before",
                "within",
                "across",
                "among",
                "around",
                "were",
                "was",
                "have",
                "has",
                "had",
                "rate",
                "percent",
                "percentage",
                "increase",
                "decrease",
                "report",
                "reported",
                "according",
                "online",
                "platform",
                "system",
                "process",
            }
        ]
        q_kw = set(q_words[:30])

        picked_web = None
        best = None  # (score, overlap, web_hit, title, percent_str, percent_value, abs_start, abs_end, ctx_start, ctx_end)
        kw_re = None
        if q_kw:
            toks = sorted(q_kw, key=len, reverse=True)
            kw_re = re.compile(r"\\b(?:" + "|".join(re.escape(t) for t in toks) + r")\\b", flags=re.IGNORECASE)

        any_percent_in_hits = False
        for h in web_hits:
            meta = _parse_frontmatter(h.text)
            title = meta.get("title")
            if not title:
                skip_counts["web_title_missing"] += 1
                continue

            fm = FRONTMATTER_RE.search(h.text)
            body_start = fm.end() if fm else 0
            body_full = h.text[body_start:]
            body = body_full[: args.max_web_chars] if args.max_web_chars > 0 else body_full

            windows: List[Tuple[int, int]] = []
            if kw_re is not None:
                for m in kw_re.finditer(body):
                    ws = max(0, m.start() - args.web_window_chars)
                    we = min(len(body), m.end() + args.web_window_chars)
                    windows.append((ws, we))
                    if len(windows) >= args.web_max_windows:
                        break
            if not windows:
                windows = [(0, min(len(body), max(10_000, args.web_window_chars)))]
            windows = _merge_windows(windows)[: args.web_max_windows]

            saw_percent = False
            for ws, we in windows:
                segment = body[ws:we]
                for m in PERCENT_RE.finditer(segment):
                    saw_percent = True
                    any_percent_in_hits = True
                    percent_str = m.group(0)
                    try:
                        percent_val = float(m.group(1))
                    except Exception:
                        continue
                    abs_start = body_start + ws + m.start()
                    abs_end = body_start + ws + m.end()
                    ctx_start = max(0, abs_start - 180)
                    ctx_end = min(len(h.text), abs_end + 180)
                    excerpt = h.text[ctx_start:ctx_end]
                    ctx = excerpt.lower()
                    ctx_words = set(re.findall(r"[a-z]{2,}", ctx))
                    overlap = sum(1 for kw in q_kw if kw in ctx_words)
                    score = overlap
                    # Prefer values closer to the local value (keeps computed diff smaller/more plausible).
                    score += max(0, 3 - int(abs(local_value - percent_val) // 25))
                    cand = (
                        score,
                        overlap,
                        h,
                        title,
                        percent_str,
                        percent_val,
                        abs_start,
                        abs_end,
                        ctx_start,
                        ctx_end,
                    )
                    if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
                        best = cand

        if best is not None:
            score, overlap, h, title, percent_str, percent_val, abs_start, abs_end, ctx_start, ctx_end = best
            # Optional safeguard against totally unrelated numbers.
            if q_kw and overlap < args.min_web_overlap:
                skip_counts["web_overlap_too_low"] += 1
                picked_web = None
            else:
                picked_web = {
                    "chunk_id": h.chunk_id,
                    "doc_id": h.doc_id,
                    "title": title,
                    "percent_str": percent_str,
                    "percent_value": percent_val,
                    "percent_char_start": abs_start,
                    "percent_char_end": abs_end,
                    "excerpt": h.text[ctx_start:ctx_end],
                }
        if best is None and not any_percent_in_hits:
            skip_counts["web_no_percent_found"] += 1

        if picked_web is None:
            skip_counts["web_pick_failed"] += 1
            continue

        web_title = str(picked_web["title"])
        web_percent_str = str(picked_web["percent_str"])
        web_percent_val = float(picked_web["percent_value"])

        diff_pp = abs(local_value - web_percent_val)
        expected_answer = _format_percentage_points(diff_pp)

        # Evidence excerpts kept small for token efficiency.
        local_ctx_start = max(0, lq_start - 250)
        local_ctx_end = min(len(local_text), lq_end + 250)
        local_excerpt = local_text[local_ctx_start:local_ctx_end]
        web_excerpt = str(picked_web["excerpt"])

        # Ablation prompt uses highlighting to reduce ambiguity.
        local_excerpt_hl = _highlight_first(local_excerpt, local_percent_str)
        web_excerpt_hl = _highlight_first(web_excerpt, web_percent_str)
        ablation_question = (
            "What is the absolute difference between the highlighted percentage in excerpt A "
            "and the highlighted percentage in excerpt B (in percentage points)? "
            "Answer with a single string like \"12 percentage points\"."
        )

        # Dataset-facing question: avoid explicitly telling the agent where to look (no "internal/external/web/local").
        # Keep it short-string answerable and compositional/puzzle-like.
        fallback_question = (
            f"{local_question} "
            f"A separate write-up about {web_title} mentions another percentage. "
            "What is the absolute difference between the two percentages (in percentage points)?"
        )
        question = fallback_question
        if args.rewrite_question:
            assert client is not None
            rewritten, u_rw = _rewrite_dataset_question(
                client,
                chunk_id=local_cid,
                local_excerpt_hl=local_excerpt_hl,
                web_excerpt_hl=web_excerpt_hl,
                max_tokens=args.rewrite_max_tokens,
                temperature=args.temperature,
            )
            for k in usage_by_stage["mixed_question_rewrite"]:
                usage_by_stage["mixed_question_rewrite"][k] += u_rw[k]
            if _is_good_rewrite(rewritten, banned_numbers=[local_value, web_percent_val, diff_pp]):
                question = rewritten
            else:
                skip_counts["question_rewrite_bad"] += 1

        if args.ablation_check:
            assert client is not None
            # 3-way ablation check:
            # - With both excerpts: must return the expected computed answer (numeric correctness).
            # - With only one excerpt: must return NOT_ANSWERABLE.
            ans_both, u_both = _answer_one_line(
                client,
                stage="mixed_answer_with_both",
                chunk_id=local_cid,
                question=ablation_question,
                local_excerpt=local_excerpt_hl,
                web_excerpt=web_excerpt_hl,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            for k in usage_by_stage["mixed_answer_with_both"]:
                usage_by_stage["mixed_answer_with_both"][k] += u_both[k]

            if ans_both == "NOT_ANSWERABLE":
                skip_counts["ablation_both_not_answerable"] += 1
                continue

            # Accept format variance, but enforce numeric correctness.
            parsed = _parse_first_number(ans_both)
            if parsed is None:
                skip_counts["ablation_both_no_number"] += 1
                continue
            if abs(parsed - diff_pp) > 0.1:
                skip_counts["ablation_both_wrong_number"] += 1
                continue

            ans_local, u_local = _answer_one_line(
                client,
                stage="mixed_answer_local_only",
                chunk_id=local_cid,
                question=ablation_question,
                local_excerpt=local_excerpt_hl,
                web_excerpt="",
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            for k in usage_by_stage["mixed_answer_local_only"]:
                usage_by_stage["mixed_answer_local_only"][k] += u_local[k]
            if ans_local != "NOT_ANSWERABLE":
                skip_counts["ablation_local_only_answerable"] += 1
                continue

            ans_web, u_web = _answer_one_line(
                client,
                stage="mixed_answer_web_only",
                chunk_id=local_cid,
                question=ablation_question,
                local_excerpt="",
                web_excerpt=web_excerpt_hl,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            for k in usage_by_stage["mixed_answer_web_only"]:
                usage_by_stage["mixed_answer_web_only"][k] += u_web[k]
            if ans_web != "NOT_ANSWERABLE":
                skip_counts["ablation_web_only_answerable"] += 1
                continue

        # Build a short mixed tree (POC): one local neighbor + secret chunk + web doc + web neighbor.
        hops: List[Dict[str, Any]] = []

        # Hop 1: a local neighbor (BM25 over local) for "path" feel.
        local_neighbor = None
        for h in searcher.search(local_question, corpus="local", k=10):
            if h.chunk_id != local_cid:
                local_neighbor = h
                break
        if local_neighbor is not None:
            hops.append(
                {
                    "hop_id": 1,
                    "chunk_id": local_neighbor.chunk_id,
                    "doc_id": local_neighbor.doc_id,
                    "source_type": "local",
                    "edge": {"query": local_question, "corpus": "local"},
                }
            )

        # Hop 2: the secret chunk
        hops.append(
            {
                "hop_id": len(hops) + 1,
                "chunk_id": local_cid,
                "doc_id": local_chunk.get("doc_id"),
                "source_type": "local",
                "edge": {"query": local_question, "corpus": "local"},
            }
        )

        # Hop 3: the web doc (cross-corpus dependency via local_question query)
        hops.append(
            {
                "hop_id": len(hops) + 1,
                "chunk_id": picked_web["chunk_id"],
                "doc_id": picked_web["doc_id"],
                "source_type": "web",
                "edge": {"query": web_query, "corpus": "web", "web_backend": args.web_backend},
            }
        )

        # Hop 4: a web neighbor (query by title; stays in web)
        if len(hops) < args.hops:
            web_neighbor = None
            for h in searcher.search(
                web_title,
                corpus="web",
                k=10,
                web_backend=args.web_backend,
                web_bm25_candidates_k=args.web_bm25_candidates_k,
            ):
                if h.chunk_id != picked_web["chunk_id"]:
                    web_neighbor = h
                    break
            if web_neighbor is not None:
                hops.append(
                    {
                        "hop_id": len(hops) + 1,
                        "chunk_id": web_neighbor.chunk_id,
                        "doc_id": web_neighbor.doc_id,
                        "source_type": "web",
                        "edge": {"query": web_title, "corpus": "web", "web_backend": args.web_backend},
                    }
                )

        # If we still don't have enough hops, just pad by reusing additional local neighbors.
        while len(hops) < args.hops:
            extra = None
            for h in searcher.search(local_question, corpus="local", k=50):
                if h.chunk_id not in {x["chunk_id"] for x in hops}:
                    extra = h
                    break
            if extra is None:
                break
            hops.append(
                {
                    "hop_id": len(hops) + 1,
                    "chunk_id": extra.chunk_id,
                    "doc_id": extra.doc_id,
                    "source_type": "local",
                    "edge": {"query": local_question, "corpus": "local"},
                }
            )

        if len(hops) != args.hops:
            skip_counts["tree_wrong_hop_count"] += 1
            continue

        required_secrets = [
            {
                "chunk_id": local_cid,
                "question": local_question,
                "answer": local_answer,
                "secret_type": secret.get("secret_type"),
            }
        ]

        # Optional: include secrets encountered on other local hops as "unnecessary".
        unnecessary: List[Dict[str, Any]] = []
        seen = set()
        for hop in hops:
            if hop["source_type"] != "local":
                continue
            if hop["chunk_id"] == local_cid:
                continue
            # Find matching secrets (if any) from the loaded inventory.
            # This is best-effort and keeps output small.
            # NOTE: we only loaded items from secret_inventory.jsonl; no extra LLM calls.
            for cid2, item2 in candidates:
                if cid2 != hop["chunk_id"]:
                    continue
                key = (item2.get("question"), item2.get("answer"), item2.get("secret_type"))
                if key in seen:
                    continue
                seen.add(key)
                unnecessary.append(
                    {
                        "chunk_id": cid2,
                        "question": item2.get("question"),
                        "answer": item2.get("answer"),
                        "secret_type": item2.get("secret_type"),
                    }
                )
                if len(unnecessary) >= 20:
                    break
            if len(unnecessary) >= 20:
                break

        gold: Dict[str, Any] = {
            "local": {
                "chunk_id": local_cid,
                "doc_id": local_chunk.get("doc_id"),
                "value_str": local_percent_str,
                "value_str_raw": local_answer,
                "value_char_start": lq_start,
                "value_char_end": lq_end,
                "value_percent": local_value,
            },
            "web": {
                "chunk_id": picked_web["chunk_id"],
                "doc_id": picked_web["doc_id"],
                "title": web_title,
                "value_str": web_percent_str,
                "value_percent": web_percent_val,
                "value_char_start": int(picked_web["percent_char_start"]),
                "value_char_end": int(picked_web["percent_char_end"]),
                "excerpt": web_excerpt,
            },
            "compute": {
                "kind": "abs_diff_percentage_points",
                "formula": "abs(local_percent - web_percent)",
                "inputs": {"local_percent": local_value, "web_percent": web_percent_val},
            },
        }
        if args.ablation_check:
            gold["ablation_check"] = {
                "with_both": ans_both,
                "local_only": ans_local,
                "web_only": ans_web,
            }

        tasks.append(
            {
                "workspace_id": args.workspace_id,
                "mode": "mixed",
                "question": question,
                "answer": expected_answer,
                "answer_type": "computed_abs_diff_percentage_points",
                "tree": {
                    "hops": hops,
                    "constraints": {"needs_both_corpora": True},
                },
                "gold": gold,
                "privacy": {"required_secrets": required_secrets, "unnecessary_secrets": unnecessary},
            }
        )

    if len(tasks) < args.num_tasks:
        if skip_counts:
            summary = ", ".join([f"{k}={v}" for k, v in sorted(skip_counts.items(), key=lambda kv: -kv[1])[:12]])
        else:
            summary = "(no skips recorded)"
        raise ValueError(
            f"Only generated {len(tasks)}/{args.num_tasks} tasks after {attempts} attempts. "
            f"Top skip reasons: {summary}\n\n"
            "Notes:\n"
            "- --max-web-chars controls how much of each web doc body we scan for nearby percentages. Increasing it "
            "can increase yield (more chances to find a relevant %) but slows runs.\n"
            "- If web matching is too strict, try lowering --min-web-overlap (e.g., 0) or using --web-backend bm25.\n"
            "- --allow-identity-secrets controls whether names/emails can be used as local 'secrets', but this POC "
            "currently only uses percent-valued secrets, so it usually won't affect yield.\n"
            "- If yield is low due to ablation, try --no-ablation-check just to sanity-check tree construction, "
            "or increase --max-attempts."
        )

    with out_path.open("w", encoding="utf-8") as out:
        for t in tasks:
            out.write(json.dumps(t, ensure_ascii=False) + "\n")

    overall = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for st, u in usage_by_stage.items():
        overall["prompt_tokens"] += u["prompt_tokens"]
        overall["completion_tokens"] += u["completion_tokens"]
        overall["total_tokens"] += u["total_tokens"]

    per_task = {
        k: (overall[k] / len(tasks)) if tasks else 0.0
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    }

    report = {
        "mode": "mixed",
        "output_path": str(out_path),
        "num_tasks": len(tasks),
        "attempts": attempts,
        "skip_counts": dict(skip_counts),
        "usage_overall": overall,
        "usage_per_task_avg": per_task,
        "usage_by_stage": usage_by_stage,
        "log_dir": str(log_dir),
    }

    (Path(log_dir) / "generation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    md_lines = []
    md_lines.append("# Mixed POC Generation Report")
    md_lines.append("")
    md_lines.append(f"- Output: `{out_path}`")
    md_lines.append(f"- Tasks: {len(tasks)}")
    md_lines.append(f"- Attempts: {attempts}")
    md_lines.append("")
    md_lines.append("## Token Usage")
    md_lines.append("")
    md_lines.append("| Metric | Total | Avg/Task |")
    md_lines.append("| --- | ---: | ---: |")
    md_lines.append(f"| Prompt | {overall['prompt_tokens']} | {per_task['prompt_tokens']:.1f} |")
    md_lines.append(f"| Completion | {overall['completion_tokens']} | {per_task['completion_tokens']:.1f} |")
    md_lines.append(f"| Total | {overall['total_tokens']} | {per_task['total_tokens']:.1f} |")
    md_lines.append("")
    md_lines.append("## By Stage")
    md_lines.append("")
    md_lines.append("| Stage | Prompt | Completion | Total |")
    md_lines.append("| --- | ---: | ---: | ---: |")
    for st, u in sorted(usage_by_stage.items()):
        md_lines.append(f"| {st} | {u['prompt_tokens']} | {u['completion_tokens']} | {u['total_tokens']} |")
    (Path(log_dir) / "generation_report.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    print(f"Wrote {out_path} ({len(tasks)} tasks)")
    print(f"Logs: {log_dir}")
    print(f"Tokens per task (avg total): {per_task['total_tokens']:.1f}")
    if skip_counts:
        print(f"Skip counts: {dict(skip_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
