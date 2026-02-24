#!/usr/bin/env python3
from __future__ import annotations

"""Build BrowseComp-compatible dense index shards for a chunk pool.

This writes pickle shards compatible with `making_dataset.index.web_dense.WebDenseSearcher`:
  pickle.dump((reps: float32[n, dim], lookup: list[str]), f)

Encoding settings are chosen to mirror BrowseComp-Plus:
- pooling: eos (last non-pad token)
- normalize: True (L2)
- passage_prefix: "" (empty)
- passage_max_len: 4096 (default)

Docids:
- We use `chunk_id` as the dense docid, so retrieval returns chunk-level IDs.
  The parent doc can be recovered with `chunk_id.split('#', 1)[0]`.
"""

import argparse
import math
import os
import pickle
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from making_dataset_2.retrieval.chunks import iter_chunks  # noqa: E402

DEFAULT_CHUNKS = ROOT_DIR / "making_dataset_2" / "outputs" / "chunks_web_drbench_urls.jsonl"
DEFAULT_OUT_DIR = ROOT_DIR / "making_dataset_2" / "outputs" / "indexes" / "drbench_urls_qwen3_dense"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Qwen3 dense shards (BrowseComp-compatible pickles).")
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS), help="Input chunks JSONL")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for shard pickles")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Embedding-4B",
        help="Embedding model name/path (default: Qwen/Qwen3-Embedding-4B)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for encoding passages (default: cpu; use cuda:0 if available).",
    )
    parser.add_argument("--passage-prefix", default="", help="Passage prefix (BrowseComp: empty string).")
    parser.add_argument("--passage-max-len", type=int, default=4096, help="Max passage tokens (default: 4096).")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size (default: 32).")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of output shards (default: 1).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of chunks to embed")
    return parser.parse_args()


def _iter_valid_chunks(chunks_path: Path, *, limit: int | None) -> Iterator[tuple[str, str]]:
    n = 0
    for rec in iter_chunks(chunks_path):
        chunk_id = str(rec.get("chunk_id") or "").strip()
        text = str(rec.get("text") or "").strip()
        if not chunk_id or not text:
            continue
        yield chunk_id, text
        n += 1
        if limit is not None and n >= limit:
            break


def _chunked(it: Iterable[Any], size: int) -> Iterator[list[Any]]:
    buf: list[Any] = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def _resolve_hf_snapshot_dir(model_name_or_path: str) -> str:
    """Resolve an HF model ID to a local snapshot dir under $TRANSFORMERS_CACHE.

    We typically run offline. Passing a local snapshot directory avoids any HF
    Hub calls and matches `making_dataset/index/web_dense.py` behavior.
    """
    p = Path(model_name_or_path)
    if p.exists():
        return str(p)

    cache_root = Path(os.getenv("TRANSFORMERS_CACHE", "")).expanduser()
    if not cache_root.exists():
        raise FileNotFoundError(
            f"TRANSFORMERS_CACHE not found ({cache_root}). "
            f"Pass a local model snapshot path instead of '{model_name_or_path}'."
        )

    repo_dir = cache_root / f"models--{model_name_or_path.replace('/', '--')}"
    snaps_dir = repo_dir / "snapshots"
    if not snaps_dir.exists():
        raise FileNotFoundError(
            f"Could not resolve model '{model_name_or_path}' under {snaps_dir}. "
            "Pass a local model snapshot path instead."
        )

    snaps = sorted([p for p in snaps_dir.iterdir() if p.is_dir()])
    if not snaps:
        raise FileNotFoundError(f"No snapshots found for model '{model_name_or_path}' under {snaps_dir}.")
    return str(snaps[-1])


def _encode_passages(
    *,
    model,
    tokenizer,
    texts: list[str],
    device: str,
    passage_prefix: str,
    passage_max_len: int,
    batch_size: int,
) -> np.ndarray:
    import torch

    ctx = torch.amp.autocast(device_type="cuda") if device.startswith("cuda") else nullcontext()
    outs: list[np.ndarray] = []
    with ctx:
        with torch.no_grad():
            for batch_texts in _chunked(texts, int(batch_size)):
                batch_in = [passage_prefix + t for t in batch_texts]
                toks = tokenizer(
                    batch_in,
                    padding=True,
                    truncation=True,
                    max_length=int(passage_max_len),
                    return_tensors="pt",
                )
                toks = {k: v.to(device) for k, v in toks.items()}
                reps = model.encode_passage(toks)
                outs.append(reps.detach().cpu().numpy())
    mat = np.concatenate(outs, axis=0)
    return np.asarray(mat, dtype=np.float32)


def main() -> int:
    args = _parse_args()
    chunks_path = Path(args.chunks)
    output_dir = Path(args.output_dir)

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    num_shards = int(args.num_shards)
    if num_shards <= 0:
        raise ValueError("--num-shards must be >= 1")

    # First pass: count.
    total = sum(1 for _ in _iter_valid_chunks(chunks_path, limit=int(args.limit) if args.limit else None))
    if total <= 0:
        raise ValueError(f"No valid chunks found in {chunks_path}")

    shard_size = int(math.ceil(total / float(num_shards)))
    print(f"Chunks: {total} | shards: {num_shards} | shard_size: {shard_size}")

    # Load model and tokenizer once.
    import torch
    from tevatron.retriever.modeling.dense import DenseModel
    from transformers import AutoTokenizer

    device = str(args.device)
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    model_path = _resolve_hf_snapshot_dir(str(args.model))

    model = DenseModel.load(
        model_path,
        pooling="eos",
        normalize=True,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    output_dir.mkdir(parents=True, exist_ok=True)

    # Second pass: encode sequentially into shards.
    it = _iter_valid_chunks(chunks_path, limit=int(args.limit) if args.limit else None)
    for shard_idx in range(1, num_shards + 1):
        shard_items = [x for _, x in zip(range(shard_size), it)]
        if not shard_items:
            break

        lookup = [cid for cid, _ in shard_items]
        texts = [t for _, t in shard_items]

        reps = _encode_passages(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            passage_prefix=str(args.passage_prefix),
            passage_max_len=int(args.passage_max_len),
            batch_size=int(args.batch_size),
        )
        if reps.shape[0] != len(lookup):
            raise ValueError("Embedding count mismatch")

        out_path = output_dir / f"corpus.shard{shard_idx}_of_{num_shards}.pkl"
        with out_path.open("wb") as f:
            pickle.dump((reps, lookup), f)
        print(f"Wrote shard: {out_path} (n={len(lookup)}, dim={reps.shape[1]})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
