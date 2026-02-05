from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import faiss  # type: ignore
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class WebDenseHit:
    docid: str
    score: float
    text: Optional[str] = None


def _resolve_hf_snapshot_dir(model_name_or_path: str) -> str:
    """
    We run offline (no HF Hub). If the caller passes an HF model ID like
    "Qwen/Qwen3-Embedding-0.6B", resolve it to a local snapshot directory under
    $TRANSFORMERS_CACHE (default: /transformers_cache/transformers).
    """
    p = Path(model_name_or_path)
    if p.exists():
        return str(p)

    cache_root = Path(os.getenv("TRANSFORMERS_CACHE", "")).expanduser()
    if not cache_root.exists():
        raise FileNotFoundError(
            f"TRANSFORMERS_CACHE not found ({cache_root}). "
            f"Pass a local model path instead of '{model_name_or_path}'."
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
        raise FileNotFoundError(
            f"No snapshots found for model '{model_name_or_path}' under {snaps_dir}."
        )
    # Usually there's a single snapshot; if multiple, pick the last (lexicographically).
    return str(snaps[-1])


class Qwen3EosEmbedder:
    """
    Minimal Qwen3 embedding encoder matching BrowseComp-Plus index settings:
    - pool: eos (last non-pad token)
    - normalize: L2
    - query prefix: instruct prefix (see BrowseComp-Plus scripts_build_index/qwen3-embed.md)
    """

    def __init__(
        self,
        *,
        model_name_or_path: str,
        query_prefix: str,
        query_max_len: int = 512,
        normalize: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = _resolve_hf_snapshot_dir(model_name_or_path)
        self.query_prefix = query_prefix
        self.query_max_len = int(query_max_len)
        self.normalize = bool(normalize)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # NOTE: use local snapshot path to avoid any HF Hub calls.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="left")
        self.model = AutoModel.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def encode_queries(self, queries: list[str], *, batch_size: int = 8) -> np.ndarray:
        if not queries:
            return np.zeros((0, 0), dtype=np.float32)

        outs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch_q = queries[i : i + batch_size]
                batch_text = [self.query_prefix + q for q in batch_q]
                toks = self.tokenizer(
                    batch_text,
                    padding=True,
                    truncation=True,
                    max_length=self.query_max_len,
                    return_tensors="pt",
                )
                toks = {k: v.to(self.device) for k, v in toks.items()}
                out = self.model(**toks, return_dict=True)
                last_hidden = out.last_hidden_state  # (B, T, D)
                attn = toks.get("attention_mask")
                if attn is None:
                    raise ValueError("Tokenizer output missing attention_mask")
                idx = attn.sum(dim=1) - 1  # last non-pad token index
                pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), idx]
                pooled = pooled.float()
                if self.normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                outs.append(pooled.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(outs, axis=0)


class WebDenseSearcher:
    def __init__(
        self,
        *,
        index_glob: str,
        model_name_or_path: str = "Qwen/Qwen3-Embedding-0.6B",
        normalize: bool = True,
        query_prefix: str = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        query_max_len: int = 512,
        encoder_batch_size: int = 8,
        doc_store: Optional[Any] = None,
    ) -> None:
        self.index_glob = index_glob
        self.model_name_or_path = model_name_or_path
        self.normalize = bool(normalize)
        self.query_prefix = query_prefix
        self.query_max_len = int(query_max_len)
        self.encoder_batch_size = int(encoder_batch_size)
        self.doc_store = doc_store

        files = sorted(glob.glob(index_glob))
        if not files:
            raise FileNotFoundError(f"No index shards matched: {index_glob}")

        reps_list: list[np.ndarray] = []
        lookup: list[str] = []
        for fp in files:
            with open(fp, "rb") as f:
                reps, shard_lookup = pickle.load(f)
            reps = np.asarray(reps, dtype=np.float32)
            if reps.ndim != 2:
                raise ValueError(f"Invalid reps shape in {fp}: {reps.shape}")
            reps_list.append(reps)
            lookup.extend([str(x) for x in shard_lookup])

        self.lookup = lookup
        self.dim = int(reps_list[0].shape[1])

        self._reps = np.concatenate(reps_list, axis=0)
        if self._reps.shape[0] != len(self.lookup):
            raise ValueError("Dense index reps/lookup length mismatch")

        self.docid_to_pos = {docid: i for i, docid in enumerate(self.lookup)}

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self._reps)

        self.embedder = Qwen3EosEmbedder(
            model_name_or_path=self.model_name_or_path,
            query_prefix=self.query_prefix,
            query_max_len=self.query_max_len,
            normalize=self.normalize,
        )

    def search_many(self, queries: list[str], *, k: int = 10) -> list[list[WebDenseHit]]:
        if not queries:
            return []
        q_reps = self.embedder.encode_queries(queries, batch_size=self.encoder_batch_size)
        scores, idxs = self.index.search(q_reps, k)
        out: list[list[WebDenseHit]] = []
        for row_scores, row_idxs in zip(scores, idxs):
            hits: list[WebDenseHit] = []
            for score, idx in zip(row_scores, row_idxs):
                if idx < 0:
                    continue
                docid = self.lookup[int(idx)]
                hits.append(WebDenseHit(docid=docid, score=float(score)))
            out.append(hits)
        return out

    def search(self, query: str, *, k: int = 10) -> list[WebDenseHit]:
        return self.search_many([query], k=k)[0]

    def rerank_docids(self, query: str, docids: list[str]) -> list[WebDenseHit]:
        """
        Rerank a given docid candidate set using dense dot product.
        Useful for BM25->dense rerank hybrid retrieval.
        """
        if not docids:
            return []
        q = self.embedder.encode_queries([query], batch_size=1)[0]  # (D,)
        cand_pos = [self.docid_to_pos.get(str(d)) for d in docids]
        keep = [(d, p) for d, p in zip(docids, cand_pos) if p is not None]
        if not keep:
            return []
        cand_docids = [str(d) for d, _ in keep]
        cand_vecs = self._reps[[int(p) for _, p in keep]]
        scores = cand_vecs @ q.astype(np.float32)
        order = np.argsort(-scores)
        return [WebDenseHit(docid=cand_docids[i], score=float(scores[i])) for i in order]

    def get_document(self, docid: str) -> Optional[dict[str, Any]]:
        if self.doc_store is None:
            return None
        return self.doc_store.get_document(docid)

