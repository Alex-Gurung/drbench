#!/usr/bin/env python3
"""
Explore retrieval pipeline for 4-hop question generation.

Given a local secret, this script:
1. Generates query variants based on the secret's concepts
2. Retrieves web documents via BM25/dense
3. Optionally uses LLM to rank candidates and discover bridges
4. Exports results to HTML for manual review

Usage:
    python -m making_dataset.explore_retrieval --random-secret --vllm-url http://127.0.0.1:8000
    python -m making_dataset.explore_retrieval --chunk-stats
    python -m making_dataset.explore_retrieval --random-secret --no-llm --html results.html
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR.parent))

from making_dataset.index.unified_searcher import UnifiedSearcher
from making_dataset.utils.vllm_client import VLLMClient

# Default paths
DEFAULT_CHUNKS_LOCAL = ROOT_DIR / "outputs" / "chunks_local.jsonl"
DEFAULT_SECRET_INVENTORY = ROOT_DIR / "outputs" / "secret_inventory.jsonl"
DEFAULT_WEB_BM25_INDEX = "/home/toolkit/BrowseComp-Plus/indexes/bm25"


@dataclass
class Secret:
    chunk_id: str
    question: str
    answer: str
    secret_type: str
    quote: str
    company: str
    task_id: str


@dataclass
class WebHit:
    docid: str
    score: float
    text: str


@dataclass
class Bridge:
    bridge_type: str
    local_value: str
    web_value: str
    relationship: str
    question: str
    answer: str
    reasoning: str


# LLM ranking prompt - select best bridge candidates from retrieved docs
RANK_CANDIDATES_PROMPT = """Given this private enterprise metric and {n_docs} web documents, rank them by bridge potential.

PRIVATE METRIC:
Question: {secret_question}
Answer: {secret_answer}

WEB DOCUMENTS:
{docs_text}

Which documents contain comparable metrics that could form a meaningful relationship (ratio, comparison, threshold, rank, category) with the private metric?

Output format:
RANKING: [best_idx, second_idx, ...] (1-indexed)
BEST_REASON: <why the top doc is most promising, 15 words max>
"""

# Full bridge discovery prompt
BRIDGE_DISCOVERY_PROMPT = """You are analyzing a local enterprise secret and a web document to find meaningful relationships.

LOCAL SECRET (private company data):
Question: {secret_question}
Answer: {secret_answer}
Context: {secret_quote}

WEB DOCUMENT (public information):
{web_text}

Find a meaningful relationship between the local secret and something specific in the web document.

Valid relationship types:
- RATIO: Local value compared to web value (e.g., "Company X uses 10x more")
- COMPARISON: Local vs web benchmark (e.g., "Above industry average of Y%")
- THRESHOLD: Local qualifies for something (e.g., "Exceeds minimum requirement of Z")
- RANK: Local's position in a list (e.g., "Ranks #N among...")
- CATEGORY: Local fits a category (e.g., "Qualifies as mid-size company")

If you find a meaningful relationship, output:
BRIDGE_TYPE: <type>
LOCAL_VALUE: <the specific value from the local secret>
WEB_VALUE: <the related value from web document>
RELATIONSHIP: <how they connect>
QUESTION: <a question that requires BOTH the local secret AND web fact to answer>
ANSWER: <the answer to your question>
REASONING: <brief explanation>

If NO meaningful relationship exists, output:
NO_BRIDGE: <brief reason>
"""


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def task_to_company(task_id: str) -> str:
    """DR0001-DR0005: Lee's Market, DR0006-DR0010: MediConn, DR0011-DR0015: Elexion"""
    num = int(task_id[2:])
    if num <= 5:
        return "Lee's Market"
    elif num <= 10:
        return "MediConn Solutions"
    else:
        return "Elexion Automotive"


def load_secrets(path: Path, secret_type: str | None = None) -> list[Secret]:
    """Load secrets from inventory."""
    secrets = []
    for rec in load_jsonl(path):
        chunk_id = rec["chunk_id"]
        task_id = chunk_id.split("/")[1]
        company = task_to_company(task_id)
        for s in rec.get("secrets", []):
            if secret_type and s.get("secret_type") != secret_type:
                continue
            secrets.append(Secret(
                chunk_id=chunk_id,
                question=s.get("question", ""),
                answer=s.get("answer", ""),
                secret_type=s.get("secret_type", ""),
                quote=s.get("quote") or "",
                company=company,
                task_id=task_id,
            ))
    return secrets


def generate_queries(secret: Secret) -> list[str]:
    """Generate query variants for retrieving related web docs."""
    queries = []
    question = secret.question.lower()
    quote = (secret.quote or "").lower()
    combined = question + " " + quote

    # Industry
    industry = {
        "Lee's Market": "retail grocery",
        "MediConn Solutions": "healthcare",
        "Elexion Automotive": "automotive manufacturing",
    }.get(secret.company, "")

    # Concept patterns
    patterns = [
        (r"(retention rate|employee retention)", "employee retention rate"),
        (r"(cost reduction|cost savings)", "cost reduction"),
        (r"(energy usage|energy consumption|kWh)", "energy consumption"),
        (r"(revenue|sales)", "company revenue"),
        (r"(profit margin|margin)", "profit margin"),
        (r"(customer satisfaction)", "customer satisfaction"),
        (r"(incident|incidents)", "incident rate"),
        (r"(compliance|regulatory)", "compliance rate"),
        (r"(training|completion rate)", "training completion rate"),
        (r"(turnover)", "employee turnover"),
        (r"(productivity)", "employee productivity"),
        (r"(efficiency)", "operational efficiency"),
        (r"(defect|quality)", "quality defect rate"),
    ]

    concept = None
    for pattern, name in patterns:
        if re.search(pattern, combined):
            concept = name
            break

    if concept:
        queries = [
            f"{industry} {concept}",
            f"{concept} benchmark",
            f"{concept} industry average",
            f"{concept} statistics",
        ]
    else:
        # Fallback: key words from question
        stopwords = {"what", "was", "the", "for", "in", "a", "an", "of", "to", "is", "how"}
        words = [w for w in re.findall(r'\b[a-z]+\b', question) if w not in stopwords and len(w) > 3]
        if words:
            phrase = " ".join(words[:3])
            queries = [f"{industry} {phrase}", f"{phrase} benchmark"]

    if not queries:
        queries = [f"{industry} KPI benchmarks", f"{industry} performance metrics"]

    # Dedupe
    seen = set()
    return [q for q in queries if not (q in seen or seen.add(q))][:5]


def search_web(query: str, searcher: UnifiedSearcher, k: int, backend: str) -> list[WebHit]:
    """Search web corpus."""
    hits = searcher.search(query, corpus="web", k=k, web_backend=backend)
    return [WebHit(docid=h.doc_id, score=h.score, text=h.text) for h in hits]


def rank_candidates(secret: Secret, hits: list[WebHit], client: VLLMClient) -> list[int]:
    """Use LLM to rank web hits by bridge potential. Returns indices in ranked order."""
    docs_text = "\n\n".join(
        f"[{i+1}] {hit.text[:800]}..." for i, hit in enumerate(hits[:5])
    )
    prompt = RANK_CANDIDATES_PROMPT.format(
        secret_question=secret.question,
        secret_answer=secret.answer,
        n_docs=min(5, len(hits)),
        docs_text=docs_text,
    )
    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        stage="rank_candidates",
        max_tokens=200,
        temperature=0.0,
    )
    output = response.choices[0].message.content

    # Parse ranking
    match = re.search(r"RANKING:\s*\[([^\]]+)\]", output)
    if match:
        indices = [int(x.strip()) - 1 for x in match.group(1).split(",") if x.strip().isdigit()]
        return [i for i in indices if 0 <= i < len(hits)]
    return list(range(len(hits)))


def discover_bridge(secret: Secret, hit: WebHit, client: VLLMClient) -> Bridge | None:
    """Use LLM to discover bridge between secret and web doc."""
    prompt = BRIDGE_DISCOVERY_PROMPT.format(
        secret_question=secret.question,
        secret_answer=secret.answer,
        secret_quote=secret.quote[:500] if secret.quote else "",
        web_text=hit.text[:3000],
    )
    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        stage="bridge_discovery",
        max_tokens=500,
        temperature=0.3,
    )
    output = response.choices[0].message.content

    if "NO_BRIDGE" in output:
        return None

    def extract(field: str) -> str:
        m = re.search(rf"{field}:\s*(.+?)(?:\n[A-Z_]+:|$)", output, re.DOTALL)
        return m.group(1).strip() if m else ""

    bridge_type = extract("BRIDGE_TYPE")
    if not bridge_type:
        return None

    return Bridge(
        bridge_type=bridge_type,
        local_value=extract("LOCAL_VALUE"),
        web_value=extract("WEB_VALUE"),
        relationship=extract("RELATIONSHIP"),
        question=extract("QUESTION"),
        answer=extract("ANSWER"),
        reasoning=extract("REASONING"),
    )


def print_chunk_stats(chunks_path: Path):
    """Print chunk size statistics."""
    chunks = load_jsonl(chunks_path)
    words = [len(c.get("text", "").split()) for c in chunks]
    words.sort()
    n = len(words)
    print("=== CHUNK STATISTICS ===")
    print(f"Total: {n}")
    print(f"Avg words: {sum(words)/n:.0f}, Median: {words[n//2]}")
    print(f"Min: {min(words)}, Max: {max(words)}")


def print_secret_stats(secrets_path: Path):
    """Print secret type statistics."""
    secrets = load_secrets(secrets_path)
    by_type: dict[str, int] = {}
    by_company: dict[str, int] = {}
    for s in secrets:
        by_type[s.secret_type] = by_type.get(s.secret_type, 0) + 1
        by_company[s.company] = by_company.get(s.company, 0) + 1

    print(f"\n=== SECRET STATISTICS ({len(secrets)} total) ===")
    print("\nBy type:")
    for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    print("\nBy company:")
    for co, c in sorted(by_company.items(), key=lambda x: -x[1]):
        print(f"  {co}: {c}")


def explore_secret(
    secret: Secret,
    searcher: UnifiedSearcher,
    client: VLLMClient | None,
    k_hits: int,
    web_backend: str,
) -> dict:
    """Explore retrieval and bridge discovery for a single secret. Returns result dict."""
    result = {
        "secret": {
            "chunk_id": secret.chunk_id,
            "company": secret.company,
            "task_id": secret.task_id,
            "type": secret.secret_type,
            "question": secret.question,
            "answer": secret.answer,
            "quote": secret.quote[:300] if secret.quote else "",
        },
        "queries": [],
        "hits": [],
        "ranking": [],
        "bridges": [],
    }

    print(f"\n{'='*70}")
    print(f"SECRET: {secret.question}")
    print(f"ANSWER: {secret.answer}")
    print(f"COMPANY: {secret.company} | TYPE: {secret.secret_type}")

    # Generate queries
    queries = generate_queries(secret)
    result["queries"] = queries
    print(f"\nQUERIES: {queries}")

    # Search
    all_hits = []
    for query in queries[:3]:
        hits = search_web(query, searcher, k=k_hits, backend=web_backend)
        print(f"\n[{query}] -> {len(hits)} hits")
        for hit in hits[:3]:
            preview = hit.text[:150].replace("\n", " ")
            print(f"  {hit.docid}: {preview}...")
            if hit not in all_hits:
                all_hits.append(hit)

    result["hits"] = [{"docid": h.docid, "score": h.score, "text": h.text[:500]} for h in all_hits]

    if not all_hits:
        print("\nNo hits found")
        return result

    if client is None:
        print("\nNo LLM - skipping ranking and bridge discovery")
        return result

    # LLM ranking
    print("\nRANKING candidates...")
    ranked_indices = rank_candidates(secret, all_hits, client)
    result["ranking"] = ranked_indices
    print(f"  Order: {[i+1 for i in ranked_indices]}")

    # Bridge discovery on top candidates
    print("\nBRIDGE DISCOVERY:")
    for idx in ranked_indices[:3]:
        hit = all_hits[idx]
        print(f"  Checking [{idx+1}] {hit.docid}...")
        bridge = discover_bridge(secret, hit, client)
        if bridge:
            print(f"    FOUND: {bridge.bridge_type} - {bridge.relationship}")
            result["bridges"].append({
                "hit_idx": idx,
                "docid": hit.docid,
                "type": bridge.bridge_type,
                "local_value": bridge.local_value,
                "web_value": bridge.web_value,
                "relationship": bridge.relationship,
                "question": bridge.question,
                "answer": bridge.answer,
                "reasoning": bridge.reasoning,
            })
        else:
            print(f"    No bridge")

    print(f"\nSUMMARY: {len(result['bridges'])} bridges found")
    return result


def export_html(results: list[dict], output_path: Path):
    """Export results to HTML file."""
    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Retrieval Exploration Results</title>
<style>
body {{ font-family: system-ui; max-width: 1200px; margin: 0 auto; padding: 20px; }}
.secret {{ background: #f5f5f5; border-radius: 8px; padding: 20px; margin: 20px 0; }}
.secret h2 {{ margin-top: 0; color: #333; }}
.meta {{ color: #666; font-size: 0.9em; }}
.queries {{ background: #e8f4f8; padding: 10px; border-radius: 4px; margin: 10px 0; }}
.hit {{ border-left: 3px solid #ccc; padding-left: 10px; margin: 10px 0; }}
.hit.ranked-1 {{ border-color: #4caf50; }}
.bridge {{ background: #e8f5e9; padding: 15px; border-radius: 4px; margin: 10px 0; }}
.bridge h4 {{ margin: 0 0 10px 0; color: #2e7d32; }}
pre {{ background: #f0f0f0; padding: 10px; overflow-x: auto; font-size: 0.85em; }}
</style>
</head><body>
<h1>Retrieval Exploration Results</h1>
<p>Generated {len(results)} secret explorations</p>
"""

    for i, r in enumerate(results):
        s = r["secret"]
        html += f"""
<div class="secret">
<h2>#{i+1}: {s['question']}</h2>
<div class="meta">
<strong>Answer:</strong> {s['answer']}<br>
<strong>Company:</strong> {s['company']} | <strong>Type:</strong> {s['type']} | <strong>Task:</strong> {s['task_id']}
</div>
<div class="queries"><strong>Queries:</strong> {', '.join(r['queries'])}</div>
"""
        # Hits
        if r["hits"]:
            html += "<h3>Web Hits</h3>"
            ranking = r.get("ranking", [])
            for j, hit in enumerate(r["hits"]):
                rank_class = "ranked-1" if ranking and ranking[0] == j else ""
                html += f"""<div class="hit {rank_class}">
<strong>[{j+1}]</strong> {hit['docid']} (score: {hit['score']:.2f})
<pre>{hit['text'][:400]}...</pre>
</div>"""

        # Bridges
        if r["bridges"]:
            html += "<h3>Bridges Found</h3>"
            for b in r["bridges"]:
                html += f"""<div class="bridge">
<h4>{b['type']}: {b['relationship']}</h4>
<p><strong>Local:</strong> {b['local_value']} | <strong>Web:</strong> {b['web_value']}</p>
<p><strong>Question:</strong> {b['question']}</p>
<p><strong>Answer:</strong> {b['answer']}</p>
<p><em>{b['reasoning']}</em></p>
</div>"""

        html += "</div>"

    html += "</body></html>"
    output_path.write_text(html)
    print(f"\nExported to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-stats", action="store_true")
    parser.add_argument("--secret-stats", action="store_true")
    parser.add_argument("--random-secret", action="store_true")
    parser.add_argument("--secret-type", type=str)
    parser.add_argument("--vllm-url", type=str)
    parser.add_argument("--vllm-model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--k-hits", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--chunks-path", type=str)
    parser.add_argument("--secrets-path", type=str)
    parser.add_argument("--web-index-path", type=str)
    parser.add_argument("--web-backend", choices=["bm25", "dense", "bm25_rerank_dense"], default="bm25")
    parser.add_argument("--html", type=str, help="Export results to HTML file")
    args = parser.parse_args()

    chunks_path = Path(args.chunks_path) if args.chunks_path else DEFAULT_CHUNKS_LOCAL
    secrets_path = Path(args.secrets_path) if args.secrets_path else DEFAULT_SECRET_INVENTORY
    web_index = args.web_index_path or DEFAULT_WEB_BM25_INDEX

    if args.seed is not None:
        random.seed(args.seed)

    if args.chunk_stats:
        print_chunk_stats(chunks_path)
        return

    if args.secret_stats:
        print_secret_stats(secrets_path)
        return

    if not args.random_secret:
        print("Use --random-secret, --chunk-stats, or --secret-stats")
        return

    # Initialize
    print("Loading searcher...")
    searcher = UnifiedSearcher(local_chunks_path=str(chunks_path), web_bm25_index_path=web_index)
    print(f"  Web docs: {searcher.web_bm25.num_docs}")

    client = None
    if not args.no_llm and args.vllm_url:
        os.environ["VLLM_API_URL"] = args.vllm_url
        client = VLLMClient(model=args.vllm_model, api_url=args.vllm_url)
        print(f"  LLM: {args.vllm_model}")

    secrets = load_secrets(secrets_path, args.secret_type)
    print(f"  Secrets: {len(secrets)}")

    sample = random.sample(secrets, min(args.n, len(secrets)))
    results = []
    for secret in sample:
        result = explore_secret(secret, searcher, client, args.k_hits, args.web_backend)
        results.append(result)

    if args.html:
        export_html(results, Path(args.html))


if __name__ == "__main__":
    main()
