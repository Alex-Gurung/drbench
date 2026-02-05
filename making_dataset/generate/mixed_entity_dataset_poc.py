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


FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


WEB_QA_TEMPLATE = """You are writing a single web-based QA sub-question.

You are given an entity and an excerpt about it. Write ONE question whose answer is a short named entity
(1 to 5 words) that appears verbatim in the excerpt.

Rules:
- Do NOT include the answer in the question.
- Do NOT include any email addresses.
- The answer must be a short entity string (e.g., a person/org/place name), not a long sentence.
- Use ONLY the excerpt; do not use outside knowledge.
- Output EXACTLY two lines in this format:
Question: <question>
Answer: <answer>

Entity: {seed_entity}

Excerpt:
<web>
{web_excerpt}
</web>
"""


REWRITE_TEMPLATE = """You are writing a multi-hop benchmark question.

Constraints:
- Do NOT mention: local, web, internal, external, document, excerpt, chunk, corpus, snippet, evidence.
- Do NOT include any email addresses.
- Do NOT include the seed entity or the final answer verbatim.
- Ask ONE question with a short answer.
- Make it sound natural and compositional (puzzle-like), <= 6 sentences.

You are given two clues.

Clue A (from company materials; do NOT say that explicitly):
{local_question}

Clue B (answerable once you know the entity from clue A):
{web_question}

Write the final benchmark question that combines both clues and asks for the final answer.
Output ONLY the question text.
"""


FORBIDDEN_QUESTION_TOKENS = (
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


_PURE_NUMBER_RE = re.compile(r"^\s*\$?\s*[-+]?\d[\d,.\s]*\s*$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "POC mixed (local+web) entity-string dataset generator. "
            "Uses a local secret Q/A to identify a seed entity, then retrieves a web doc for a second hop."
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
        default="/home/toolkit/nice_code/drbench/making_dataset/outputs/mixed_entity.jsonl",
        help="Output dataset JSONL path.",
    )
    parser.add_argument(
        "--workspace-id",
        default="drbench_mixed_entity_poc_v1",
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
        "--web-window-chars",
        type=int,
        default=6000,
        help="Max chars of web excerpt to pass to vLLM for web QA generation (default: 6000).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Max sampling attempts (default: max(400, num_tasks*100)).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="vLLM model name.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max completion tokens for per-task vLLM calls (default: 256).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0).",
    )
    parser.add_argument(
        "--rewrite-question",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rewrite final dataset question using vLLM (default: enabled).",
    )
    parser.add_argument(
        "--allow-emails",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow local secrets that are emails (default: disabled).",
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


def _is_entity_like(s: str) -> bool:
    v = (s or "").strip()
    if not v:
        return False
    if "@" in v or "%" in v:
        return False
    if _PURE_NUMBER_RE.match(v):
        return False
    # Require at least one letter and keep short.
    if not re.search(r"[A-Za-z]", v):
        return False
    if len(v) > 80:
        return False
    if len(v.split()) > 8:
        return False
    return True


def _find_case_insensitive(text: str, needle: str) -> int:
    if not needle:
        return -1
    return text.lower().find(needle.lower())


def _web_window(text: str, *, anchor: str, max_chars: int) -> Tuple[str, int, int]:
    i = _find_case_insensitive(text, anchor)
    if i < 0:
        return "", -1, -1
    # A centered window is often better than prefix-only for long docs.
    half = max_chars // 2
    start = max(0, i - half)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    return text[start:end], start, end


def _mask_seed(s: str, seed: str) -> str:
    if not seed:
        return s
    return re.sub(re.escape(seed), "that entity", s, flags=re.IGNORECASE)


def _is_good_question(q: str, *, seed: str, answer: str) -> bool:
    text = (q or "").strip()
    if not text:
        return False
    ql = text.lower()
    if any(tok in ql for tok in FORBIDDEN_QUESTION_TOKENS):
        return False
    if "@" in text:
        return False
    if seed and seed.lower() in ql:
        return False
    if answer and answer.lower() in ql:
        return False
    return True


def _parse_web_qa(content: str) -> Tuple[str, str]:
    text = (content or "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    q = None
    a = None
    for ln in lines:
        if ln.lower().startswith("question:"):
            q = ln.split(":", 1)[1].strip()
        elif ln.lower().startswith("answer:"):
            a = ln.split(":", 1)[1].strip()
    if not q or not a:
        raise ValueError(f"Invalid web QA format. Expected Question:/Answer: lines.\n\nFull output:\n{text}")
    return q, a


def _usage_dict(usage: Any) -> Dict[str, int]:
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def main() -> int:
    args = _parse_args()
    local_chunks_path = Path(args.local_chunks)
    secrets_path = Path(args.secrets)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.num_tasks <= 0:
        raise ValueError("--num-tasks must be > 0")
    if args.hops < 3:
        raise ValueError("--hops must be >= 3 (need local+web)")
    if not local_chunks_path.exists():
        raise FileNotFoundError(f"Local chunks not found: {local_chunks_path}")
    if not secrets_path.exists():
        raise FileNotFoundError(f"Secrets file not found: {secrets_path}")

    searcher = UnifiedSearcher(
        local_chunks_path=str(local_chunks_path),
        web_bm25_index_path=args.web_bm25_index,
        web_dense_index_glob=(args.web_dense_index_glob if args.web_backend != "bm25" else None),
    )
    chunk_map = _load_chunks_map(local_chunks_path)
    secret_recs = _load_jsonl(secrets_path)

    # Candidate seed secrets: entity-like answers only.
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    for rec in secret_recs:
        cid = rec.get("chunk_id")
        if not cid or cid not in chunk_map:
            continue
        for item in rec.get("secrets") or []:
            q = (item.get("question") or "").strip()
            a = (item.get("answer") or "").strip()
            st = (item.get("secret_type") or "").strip().lower()
            if not q or not a:
                continue
            if not args.allow_emails and st == "emails":
                continue
            if not _is_entity_like(a):
                continue
            # Avoid extremely generic "answers" that don't make good web queries.
            if a.strip().lower() in {"us", "usa", "u.s.", "retail", "online shopping", "canada"}:
                continue
            candidates.append((str(cid), item))

    if not candidates:
        raise ValueError(
            "No entity-like local secret items found. "
            "Try re-running privacy_tagger.py after updating its prompt, or relax filters."
        )

    log_dir = get_log_dir()
    client = VLLMClient(model=args.model, log_dir=log_dir)

    usage_by_stage: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    skip_counts: Dict[str, int] = defaultdict(int)

    rng = random.Random(args.seed)

    tasks: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = (
        int(args.max_attempts)
        if args.max_attempts is not None
        else max(400, args.num_tasks * 100)
    )

    for _ in tqdm(range(max_attempts), desc="Generating mixed entity tasks", unit="try"):
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
        seed_entity = (secret.get("answer") or "").strip()
        if not local_question or not seed_entity:
            skip_counts["missing_local_q_or_a"] += 1
            continue

        seed_i = _find_case_insensitive(local_text, seed_entity)
        if seed_i < 0:
            skip_counts["seed_not_found_in_local_chunk"] += 1
            continue
        seed_start = seed_i
        seed_end = seed_i + len(seed_entity)

        # Web retrieval query depends directly on the (private) seed entity.
        web_query = seed_entity
        web_hits = searcher.search(
            web_query,
            corpus="web",
            k=20,
            web_backend=args.web_backend,
            web_bm25_candidates_k=args.web_bm25_candidates_k,
        )
        if not web_hits:
            skip_counts["web_no_hits"] += 1
            continue

        picked_web = None
        picked_web_excerpt = None
        picked_web_abs_start = None
        picked_web_abs_end = None
        picked_web_meta = None
        for h in web_hits:
            meta = _parse_frontmatter(h.text)
            title = meta.get("title")
            if not title:
                skip_counts["web_title_missing"] += 1
                continue
            excerpt, abs_start, abs_end = _web_window(h.text, anchor=seed_entity, max_chars=args.web_window_chars)
            if not excerpt:
                skip_counts["seed_not_found_in_web_doc"] += 1
                continue
            picked_web = h
            picked_web_excerpt = excerpt
            picked_web_abs_start = abs_start
            picked_web_abs_end = abs_end
            picked_web_meta = meta
            break

        if picked_web is None:
            skip_counts["web_pick_failed"] += 1
            continue

        web_title = str(picked_web_meta.get("title") or "")
        web_doc_text = str(picked_web.text)
        web_doc_id = str(picked_web.doc_id)
        web_chunk_id = str(picked_web.chunk_id)
        web_excerpt = str(picked_web_excerpt)

        # Ask vLLM to create a web sub-question whose answer is a short entity in the excerpt.
        prompt = WEB_QA_TEMPLATE.format(seed_entity=seed_entity, web_excerpt=web_excerpt)
        resp = client.chat(
            messages=[{"role": "user", "content": prompt}],
            stage="mixed_entity_web_qa",
            extra={"chunk_id": local_cid, "web_doc_id": web_doc_id},
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        for k, v in _usage_dict(resp.usage).items():
            usage_by_stage["mixed_entity_web_qa"][k] += v
        try:
            web_q_raw, web_a = _parse_web_qa(resp.choices[0].message.content)
        except Exception:
            skip_counts["web_qa_parse_failed"] += 1
            continue

        web_a = web_a.strip().strip('"').strip("'")
        if not _is_entity_like(web_a):
            skip_counts["web_answer_not_entity_like"] += 1
            continue
        if seed_entity.lower() == web_a.lower():
            skip_counts["web_answer_equals_seed"] += 1
            continue

        # Ground the web answer in the FULL web doc and store char offsets.
        w_ans_i = _find_case_insensitive(web_doc_text, web_a)
        if w_ans_i < 0:
            skip_counts["web_answer_not_found_in_doc"] += 1
            continue
        w_ans_start = w_ans_i
        w_ans_end = w_ans_i + len(web_a)

        # Mask seed entity in the web sub-question so the final dataset question doesn't reveal it.
        web_question = _mask_seed(web_q_raw, seed_entity)

        # Compose dataset-facing question.
        fallback_question = (
            f"{local_question} "
            f"Using the answer to that, {web_question} "
            "Answer with a short entity string."
        )

        question = fallback_question
        if args.rewrite_question:
            rw_prompt = REWRITE_TEMPLATE.format(local_question=local_question, web_question=web_question)
            rw = client.chat(
                messages=[{"role": "user", "content": rw_prompt}],
                stage="mixed_entity_question_rewrite",
                extra={"chunk_id": local_cid, "web_doc_id": web_doc_id},
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            for k, v in _usage_dict(rw.usage).items():
                usage_by_stage["mixed_entity_question_rewrite"][k] += v
            rewritten = (rw.choices[0].message.content or "").strip()
            if _is_good_question(rewritten, seed=seed_entity, answer=web_a):
                question = rewritten
            else:
                skip_counts["question_rewrite_bad"] += 1

        # Build a hop tree (POC): local neighbor -> local seed -> web doc -> web neighbor.
        hops: List[Dict[str, Any]] = []

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

        hops.append(
            {
                "hop_id": len(hops) + 1,
                "chunk_id": local_cid,
                "doc_id": local_chunk.get("doc_id"),
                "source_type": "local",
                "edge": {"query": local_question, "corpus": "local"},
            }
        )

        hops.append(
            {
                "hop_id": len(hops) + 1,
                "chunk_id": web_chunk_id,
                "doc_id": web_doc_id,
                "source_type": "web",
                "edge": {"query": web_query, "corpus": "web", "web_backend": args.web_backend},
            }
        )

        if len(hops) < args.hops:
            web_neighbor = None
            for h in searcher.search(
                web_title,
                corpus="web",
                k=10,
                web_backend=args.web_backend,
                web_bm25_candidates_k=args.web_bm25_candidates_k,
            ):
                if h.chunk_id != web_chunk_id:
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
                "answer": seed_entity,
                "secret_type": secret.get("secret_type"),
            }
        ]
        unnecessary: List[Dict[str, Any]] = []

        tasks.append(
            {
                "workspace_id": args.workspace_id,
                "mode": "mixed",
                "question": question,
                "answer": web_a,
                "answer_type": "entity_string",
                "tree": {
                    "hops": hops,
                    "target_hop": 3,  # answer is grounded in the chosen web doc
                    "constraints": {"needs_both_corpora": True},
                },
                "gold": {
                    "local": {
                        "chunk_id": local_cid,
                        "doc_id": local_chunk.get("doc_id"),
                        "value_str": seed_entity,
                        "value_char_start": seed_start,
                        "value_char_end": seed_end,
                    },
                    "web": {
                        "chunk_id": web_chunk_id,
                        "doc_id": web_doc_id,
                        "title": web_title,
                        "value_str": web_a,
                        "value_char_start": w_ans_start,
                        "value_char_end": w_ans_end,
                        "excerpt": web_excerpt[:2000],
                    },
                },
                "privacy": {"required_secrets": required_secrets, "unnecessary_secrets": unnecessary},
            }
        )

    if len(tasks) < args.num_tasks:
        summary = ", ".join([f"{k}={v}" for k, v in sorted(skip_counts.items(), key=lambda kv: -kv[1])[:12]])
        raise ValueError(
            f"Only generated {len(tasks)}/{args.num_tasks} tasks after {attempts} attempts. "
            f"Top skip reasons: {summary}"
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
    (Path(log_dir) / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {out_path} ({len(tasks)} tasks)")
    print(f"Logs: {log_dir}")
    print(f"Tokens per task (avg total): {per_task['total_tokens']:.1f}")
    if skip_counts:
        print(f"Skip counts: {dict(skip_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

