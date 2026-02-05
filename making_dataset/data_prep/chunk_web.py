#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from making_dataset.utils.progress import progress  # noqa: E402

DEFAULT_INPUT = Path("/home/toolkit/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl")
DEFAULT_CORPUS_CACHE_ROOT = Path("/transformers_cache/datasets/Tevatron___browsecomp-plus-corpus")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create web chunks from BrowseComp-Plus JSONL.")
    parser.add_argument(
        "--source",
        choices=["decrypted_jsonl", "corpus_cache"],
        default="decrypted_jsonl",
        help=(
            "Where to read web docs from. "
            "'decrypted_jsonl' uses BrowseComp-Plus tasks file (subset of docs). "
            "'corpus_cache' uses the cached Tevatron/browsecomp-plus-corpus dataset (full 100,195 docs)."
        ),
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Decrypted BrowseComp-Plus JSONL (used when --source=decrypted_jsonl)",
    )
    parser.add_argument(
        "--corpus-cache-root",
        default=str(DEFAULT_CORPUS_CACHE_ROOT),
        help=(
            "Root dir for the cached Tevatron/browsecomp-plus-corpus dataset "
            "(used when --source=corpus_cache)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of docs written (useful for smoke tests).",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "chunks_web.jsonl"),
        help="Output JSONL path",
    )
    return parser.parse_args()


def _extract_frontmatter(text: str) -> Dict[str, str]:
    if not text.startswith("---"):
        return {}
    parts = text.split("\n")
    if len(parts) < 3:
        return {}
    # find closing ---
    try:
        end_idx = parts[1:].index("---") + 1
    except ValueError:
        return {}
    frontmatter_lines = parts[1:end_idx]
    meta: Dict[str, str] = {}
    for line in frontmatter_lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key:
            meta[key] = value
    return meta


def _iter_docs_from_decrypted(path: Path) -> Iterator[Dict]:
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in progress(f, total=None, desc="BrowseComp entries"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            docs = (row.get("gold_docs") or []) + (row.get("evidence_docs") or []) + (row.get("negative_docs") or [])
            for doc in docs:
                docid = str(doc.get("docid"))
                if not docid or docid in seen:
                    continue
                seen.add(docid)
                yield doc


def _find_corpus_cache_arrows(root: Path) -> list[Path]:
    # Typical path:
    # /transformers_cache/datasets/Tevatron___browsecomp-plus-corpus/default/0.0.0/<hash>/*.arrow
    patterns = [
        str(root / "**" / "browsecomp-plus-corpus-*.arrow"),
        str(root / "**" / "*.arrow"),
    ]
    hits: list[Path] = []
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            hits.append(Path(p))
        if hits:
            break
    hits = [p for p in hits if p.is_file()]
    return sorted(set(hits))


def _iter_docs_from_corpus_cache(root: Path) -> Iterator[Dict]:
    try:
        import pyarrow.ipc as ipc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required to read the cached browsecomp-plus-corpus dataset."
        ) from exc

    arrows = _find_corpus_cache_arrows(root)
    if not arrows:
        raise FileNotFoundError(
            f"No .arrow files found under corpus cache root: {root}\n"
            "Expected a cached Tevatron/browsecomp-plus-corpus dataset."
        )

    for arrow_path in arrows:
        with arrow_path.open("rb") as f:
            reader = ipc.RecordBatchStreamReader(f)
            for batch in reader:
                # Columns: docid, text, url
                col_docid = batch.column("docid")
                col_text = batch.column("text")
                col_url = batch.column("url") if "url" in batch.schema.names else None
                for i in range(batch.num_rows):
                    docid = col_docid[i].as_py()
                    text = col_text[i].as_py()
                    url = col_url[i].as_py() if col_url is not None else None
                    yield {"docid": docid, "text": text, "url": url}


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    corpus_cache_root = Path(args.corpus_cache_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "decrypted_jsonl":
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

    with output_path.open("w", encoding="utf-8") as out:
        n = 0
        if args.source == "decrypted_jsonl":
            doc_iter = _iter_docs_from_decrypted(input_path)
        else:
            doc_iter = _iter_docs_from_corpus_cache(corpus_cache_root)

        pbar = None
        if args.source == "corpus_cache":
            # Use a manual tqdm so the bar stays accurate even if we early-break on --limit.
            try:
                from tqdm import tqdm  # type: ignore

                pbar = tqdm(total=args.limit, desc="Web docs", ncols=100)
            except Exception:
                pbar = None

        for doc in doc_iter:
            docid = str(doc.get("docid"))
            text = doc.get("text") or ""
            if not text.strip():
                continue
            meta = _extract_frontmatter(text)
            url = doc.get("url")
            if url:
                meta.setdefault("url", url)
            doc_id = f"web/{docid}"
            record = {
                "chunk_id": f"{doc_id}#0001",
                "doc_id": doc_id,
                "source_type": "web",
                "text": text,
                "offsets": None,
                "meta": meta,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            if pbar is not None:
                pbar.update(1)
            if args.limit is not None and n >= args.limit:
                break

        if pbar is not None:
            pbar.close()

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
