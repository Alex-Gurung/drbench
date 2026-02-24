#!/usr/bin/env python3
from __future__ import annotations

"""Fetch and extract DRBench seed URLs into a local web corpus.

Inputs:
- `drbench/data/contexts/urls.json` (seed URLs + metadata)
- `drbench/data/tasks/*/context.json` (optional URL -> task_id mapping)

Outputs (defaults under `making_dataset_2/outputs/`):
- `docs_web_drbench_urls.jsonl`: one record per URL (success or failure)
- `drbench_urls_corpus/downloads/`: raw downloads (html/pdf/bin)
- `drbench_urls_corpus/extracted/`: extracted plaintext `.txt`

Notes:
- This script requires outbound internet access.
- HTML extraction prefers `trafilatura` or `readability-lxml` if installed, then
  falls back to a BS4 heuristic that tries to isolate the main content.
"""

import argparse
import json
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from making_dataset.utils.progress import progress  # noqa: E402
from making_dataset_2.drbench_urls import (  # noqa: E402
    WEB_POOL,
    build_url_to_task_ids,
    doc_id_for_docid,
    docid_for_url,
    load_seed_urls,
)


DEFAULT_URLS_JSON = ROOT_DIR / "drbench" / "data" / "contexts" / "urls.json"
DEFAULT_TASKS_ROOT = ROOT_DIR / "drbench" / "data" / "tasks"
DEFAULT_OUTPUT_DOCS = ROOT_DIR / "making_dataset_2" / "outputs" / "docs_web_drbench_urls.jsonl"
DEFAULT_CORPUS_DIR = ROOT_DIR / "making_dataset_2" / "outputs" / "drbench_urls_corpus"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except Exception:
        return str(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and extract drbench seed URLs into a local corpus.")
    parser.add_argument("--urls-json", default=str(DEFAULT_URLS_JSON), help="Path to drbench contexts/urls.json")
    parser.add_argument("--tasks-root", default=str(DEFAULT_TASKS_ROOT), help="Path to drbench data/tasks root")
    parser.add_argument(
        "--output-docs",
        default=str(DEFAULT_OUTPUT_DOCS),
        help="Output docs JSONL (one record per URL, includes failures).",
    )
    parser.add_argument(
        "--corpus-dir",
        default=str(DEFAULT_CORPUS_DIR),
        help="Sidecar dir for raw downloads and extracted text files.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="Per-request timeout.")
    parser.add_argument("--retries", type=int, default=2, help="Number of retries on transient failures.")
    parser.add_argument("--max-workers", type=int, default=4, help="Concurrent fetch workers.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Skip successful URLs.")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if files/records already exist.")
    parser.add_argument(
        "--user-agent",
        default="Mozilla/5.0 (compatible; drbench-url-corpus/1.0; +https://github.com/ServiceNow/drbench)",
        help="User-Agent header to send.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of URLs to process.")
    return parser.parse_args()


def _load_completed_success_urls(output_docs: Path) -> set[str]:
    if not output_docs.exists():
        return set()
    done: set[str] = set()
    with output_docs.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = (obj.get("url") or "").strip()
            text = obj.get("text") or ""
            err = obj.get("error")
            status = obj.get("http_status")
            if url and text.strip() and not err and isinstance(status, int) and 200 <= status < 300:
                done.add(url)
    return done


def _guess_raw_extension(url: str, content_type: str) -> str:
    ct = (content_type or "").lower()
    path = urlparse(url).path.lower()
    if "application/pdf" in ct or path.endswith(".pdf"):
        return ".pdf"
    if "html" in ct or path.endswith((".html", ".htm")):
        return ".html"
    if ct.startswith("text/"):
        return ".txt"
    return ".bin"


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines).strip()
    # Compress blank lines, but preserve paragraph breaks.
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def _content_score(el) -> float:
    txt = el.get_text(" ", strip=True)
    if len(txt) < 200:
        return 0.0
    link_txt = sum(len(a.get_text(" ", strip=True)) for a in el.find_all("a"))
    p_count = len(el.find_all("p"))
    li_count = len(el.find_all("li"))
    h_count = len(el.find_all(["h1", "h2", "h3"]))
    # Favor dense, paragraph-heavy regions and penalize link-heavy nav-like blocks.
    return (len(txt) - 0.5 * link_txt) + 200.0 * p_count + 50.0 * li_count + 100.0 * h_count


def _pick_main_container(soup: BeautifulSoup):
    for tag in ("article", "main"):
        el = soup.find(tag)
        if el and len(el.get_text(" ", strip=True)) >= 200:
            return el

    body = soup.body or soup
    best = body
    best_score = _content_score(best)

    for el in body.find_all(["article", "main", "section", "div"], recursive=True):
        score = _content_score(el)
        if score > best_score:
            best = el
            best_score = score

    return best


def _extract_markdownish_from_soup(container) -> str:
    blocks: list[str] = []
    for el in container.find_all(["h1", "h2", "h3", "p", "li"], recursive=True):
        if el.name == "p" and el.find_parent("li") is not None:
            continue
        text = el.get_text(" ", strip=True)
        if not text or len(text) < 2:
            continue
        if el.name in {"h1", "h2", "h3"}:
            level = {"h1": 1, "h2": 2, "h3": 3}[el.name]
            blocks.append("#" * level + " " + text)
        elif el.name == "li":
            blocks.append("- " + text)
        else:
            blocks.append(text)

    out_lines: list[str] = []
    prev_is_list = False
    for b in blocks:
        is_heading = b.startswith("#")
        is_list = b.startswith("- ")
        if not out_lines:
            out_lines.append(b)
            prev_is_list = is_list
            continue
        if is_heading:
            out_lines.append("")
            out_lines.append(b)
            prev_is_list = False
            continue
        if is_list:
            if not prev_is_list:
                out_lines.append("")
            out_lines.append(b)
            prev_is_list = True
            continue
        # paragraph
        if prev_is_list:
            out_lines.append("")
        out_lines.append("")
        out_lines.append(b)
        prev_is_list = False

    return "\n".join(out_lines).strip()


def _extract_html_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    # Remove common boilerplate containers.
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe"]):
        tag.decompose()
    for tag_name in ("header", "footer", "nav", "aside", "form"):
        for tag in soup.find_all(tag_name):
            tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    container = _pick_main_container(soup)
    text = _extract_markdownish_from_soup(container)
    if not title:
        # Try to infer from first heading.
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
    return text, title


def _extract_text_from_response(response: requests.Response, url: str) -> tuple[str, str, str]:
    """Return (text, title, extractor_name)."""
    content_type = (response.headers.get("content-type") or "").lower()
    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        return "", "", "pdf"

    html = response.text or ""

    # Prefer trafilatura if installed (boilerplate removal).
    try:
        import trafilatura  # type: ignore

        extracted = trafilatura.extract(
            html,
            output_format="markdown",
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
        if extracted and len(extracted.strip()) >= 200:
            text = _normalize_text(extracted)
            # Title still comes from HTML; trafilatura metadata is optional.
            _, title = _extract_html_text(html)
            return text, title, "trafilatura"
    except Exception:
        pass

    # Fallback: readability-lxml (if installed) and then BS4 extraction.
    try:
        from readability import Document  # type: ignore

        doc = Document(html)
        summary_html = doc.summary(html_partial=True)
        text, title = _extract_html_text(summary_html)
        text = _normalize_text(text)
        if text and len(text.strip()) >= 200:
            if not title:
                title = (doc.short_title() or "").strip()
            return text, title, "readability"
    except Exception:
        pass

    # Fallback: simple BS4 heuristic.
    text, title = _extract_html_text(html)
    text = _normalize_text(text)
    return text, title, "bs4"


def _extract_text_from_pdf(pdf_path: Path) -> str:
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    try:
        pages = [page.get_text() for page in doc]
    finally:
        doc.close()
    return _normalize_text("\n\n".join(pages))


def _ensure_dirs(corpus_dir: Path) -> tuple[Path, Path]:
    downloads_dir = corpus_dir / "downloads"
    extracted_dir = corpus_dir / "extracted"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    return downloads_dir, extracted_dir


def _should_skip_url(
    *,
    url: str,
    done_urls: set[str],
    raw_path: Path,
    extracted_path: Path,
    force: bool,
) -> bool:
    if force:
        return False
    if url in done_urls:
        return True
    if raw_path.exists() and extracted_path.exists():
        return True
    return False


def _fetch_with_retries(
    session: requests.Session,
    url: str,
    *,
    timeout_seconds: float,
    retries: int,
) -> requests.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, timeout=timeout_seconds)
            # Retry on transient server/rate-limit errors.
            if resp.status_code in {429} or resp.status_code >= 500:
                if attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
            return resp
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(2 ** attempt)
    assert last_exc is not None
    raise last_exc


def _build_base_record(seed: dict[str, Any], *, url: str, task_ids: list[str]) -> dict[str, Any]:
    docid = docid_for_url(url)
    doc_id = doc_id_for_docid(docid)
    return {
        "doc_id": doc_id,
        "docid": docid,
        "url": url,
        "industry": seed.get("industry"),
        "domain": seed.get("domain"),
        "seed_date": seed.get("date"),
        "task_ids": task_ids,
        "example_dr_questions": seed.get("example_dr_questions") or [],
        "insights": seed.get("insights") or [],
    }


def _process_one(
    seed: dict[str, Any],
    *,
    user_agent: str,
    url_to_task_ids: dict[str, list[str]],
    downloads_dir: Path,
    extracted_dir: Path,
    timeout_seconds: float,
    retries: int,
    force: bool,
    done_urls: set[str],
) -> dict[str, Any]:
    url = (seed.get("url") or "").strip()
    task_ids = url_to_task_ids.get(url, [])
    record: dict[str, Any] = _build_base_record(seed, url=url, task_ids=task_ids)

    fetched_at = _utc_now_iso()
    record["fetched_at"] = fetched_at

    docid = record["docid"]
    # Determine paths up-front so skip logic is stable.
    path = urlparse(url).path.lower()
    raw_ext_guess = ".pdf" if path.endswith(".pdf") else ".html"
    raw_path_guess = downloads_dir / f"{docid}{raw_ext_guess}"
    extracted_path = extracted_dir / f"{docid}.txt"

    if _should_skip_url(
        url=url,
        done_urls=done_urls,
        raw_path=raw_path_guess,
        extracted_path=extracted_path,
        force=force,
    ):
        record["skipped"] = True
        record["raw_path"] = _safe_relpath(raw_path_guess)
        record["extracted_path"] = _safe_relpath(extracted_path)
        return record

    # Create a per-thread session (requests.Session is not thread-safe).
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept": "*/*"})

    try:
        response = _fetch_with_retries(session, url, timeout_seconds=timeout_seconds, retries=retries)
        record["http_status"] = int(response.status_code)
        content_type = response.headers.get("content-type") or ""
        record["content_type"] = content_type

        ext = _guess_raw_extension(url, content_type)
        raw_path = downloads_dir / f"{docid}{ext}"
        raw_path.write_bytes(response.content)
        record["raw_path"] = _safe_relpath(raw_path)

        if not (200 <= int(response.status_code) < 300):
            record["error"] = f"HTTP {response.status_code}"
            record["extracted_path"] = _safe_relpath(extracted_path)
            return record

        title = ""
        extractor = ""
        if ext == ".pdf":
            text = _extract_text_from_pdf(raw_path)
            extractor = "pymupdf"
        else:
            text, title, extractor = _extract_text_from_response(response, url)

        text = _normalize_text(text)
        record["extractor"] = extractor
        if title:
            record["title"] = title

        if not text.strip():
            record["error"] = "Empty extracted text"
            record["extracted_path"] = _safe_relpath(extracted_path)
            return record

        extracted_path.write_text(text, encoding="utf-8")
        record["extracted_path"] = _safe_relpath(extracted_path)
        record["text"] = text
        return record

    except Exception as exc:
        record["error"] = str(exc)
        record.setdefault("http_status", None)
        record.setdefault("content_type", None)
        record["raw_path"] = _safe_relpath(raw_path_guess)
        record["extracted_path"] = _safe_relpath(extracted_path)
        return record


def _iter_seeds(seeds: list[dict[str, Any]], limit: Optional[int]) -> Iterable[dict[str, Any]]:
    if limit is None:
        return seeds
    return seeds[: max(0, limit)]


def main() -> int:
    args = _parse_args()
    urls_json_path = Path(args.urls_json)
    tasks_root = Path(args.tasks_root)
    output_docs = Path(args.output_docs)
    corpus_dir = Path(args.corpus_dir)

    output_docs.parent.mkdir(parents=True, exist_ok=True)
    downloads_dir, extracted_dir = _ensure_dirs(corpus_dir)

    seeds = load_seed_urls(urls_json_path)
    url_to_task_ids = build_url_to_task_ids(tasks_root)

    done_urls: set[str] = set()
    if args.resume and not args.force:
        done_urls = _load_completed_success_urls(output_docs)

    # Append when resuming; otherwise overwrite.
    mode = "a" if (args.resume and output_docs.exists()) else "w"
    processed = 0
    with output_docs.open(mode, encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
            futures = []
            for seed in _iter_seeds(seeds, args.limit):
                url = (seed.get("url") or "").strip()
                if not url:
                    continue
                futures.append(
                    ex.submit(
                        _process_one,
                        seed,
                        user_agent=args.user_agent,
                        url_to_task_ids=url_to_task_ids,
                        downloads_dir=downloads_dir,
                        extracted_dir=extracted_dir,
                        timeout_seconds=float(args.timeout_seconds),
                        retries=int(args.retries),
                        force=bool(args.force),
                        done_urls=done_urls,
                    )
                )

            for fut in progress(as_completed(futures), total=len(futures), desc=f"Fetch {WEB_POOL}"):
                rec = fut.result()
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                processed += 1

    print(f"Wrote {processed} records to {output_docs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
