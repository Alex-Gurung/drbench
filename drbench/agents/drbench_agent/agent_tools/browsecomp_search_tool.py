"""
BrowseComp-Plus search tool for DrBench.

Provides offline web search using a fixed FAISS-indexed corpus (BrowseComp-Plus).
Uses dense retrieval with Qwen3-Embedding models for query encoding.

The embedding model runs on CPU by default to avoid GPU memory conflicts with vLLM.
"""

from __future__ import annotations

import glob
import logging
import pickle
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from tevatron.retriever.searcher import FaissFlatSearcher
from tevatron.retriever.modeling.dense import DenseModel

from .base import ResearchContext, Tool
from drbench.internet_search_logging import log_internet_search
from drbench.config import RunConfig, get_run_config

logger = logging.getLogger(__name__)


class BrowseCompSearchTool(Tool):
    """Tool for searching the BrowseComp-Plus fixed corpus using dense retrieval.

    This tool provides reproducible, offline web search as an alternative to
    live Serper searches. Results come from a fixed corpus indexed with FAISS.

    The embedding model runs on CPU by default to avoid GPU memory conflicts
    with vLLM serving the main LLM.
    """

    def __init__(
        self,
        config: RunConfig,
        vector_store: Any = None,
        device: str = "cpu",
    ):
        """Initialize BrowseComp search tool.

        Args:
            config: RunConfig with BrowseComp settings (index_glob, model_name, etc.)
            vector_store: Optional vector store to store retrieved documents
            device: Device for embedding model ("cpu" by default to avoid vLLM conflicts)
        """
        self.config = config
        self.vector_store = vector_store
        self.device = device

        self.searcher: Optional[FaissFlatSearcher] = None
        self.lookup: Optional[List[str]] = None
        self.dataset = None
        self.docid_to_idx: Dict[str, int] = {}
        self.docid_to_url: Dict[str, str] = {}
        self.model = None
        self.tokenizer = None

        # Task prefix for query encoding
        self.task_prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

        self._initialize()

    @property
    def purpose(self) -> str:
        # Match InternetSearchTool's purpose so agent treats it as regular web search
        return """External market research, competitive intelligence, and public data analysis.
        IDEAL FOR: Market trends, competitor analysis, industry reports, public research papers, news articles, regulatory information, and technology comparisons.
        USE WHEN: Research requires public/external sources, competitor benchmarking, market validation, industry context, or recent developments.
        PARAMETERS: query (specific search terms work best - e.g., 'AI market size 2024', 'competitor pricing strategies', 'regulatory changes fintech')
        OUTPUTS: Search results with URLs, snippets, and relevant content that gets automatically processed and stored for synthesis."""

    def _initialize(self) -> None:
        """Initialize FAISS index, embedding model, and corpus."""
        logger.info("Initializing BrowseComp-Plus search tool...")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Index: {self.config.browsecomp_index_glob}")
        logger.info(f"  Model: {self.config.browsecomp_model_name}")

        self._load_faiss_index()
        self._load_model_and_tokenizer()
        self._load_dataset()

        logger.info("BrowseComp-Plus search tool initialized.")

    def _load_faiss_index(self) -> None:
        """Load FAISS index from pickle shards."""
        def pickle_load(path: str):
            with open(path, "rb") as f:
                reps, lookup = pickle.load(f)
            return np.array(reps), lookup

        index_files = sorted(glob.glob(self.config.browsecomp_index_glob))
        if not index_files:
            raise RuntimeError(
                f"No index shards found for pattern: {self.config.browsecomp_index_glob}"
            )

        logger.info(f"Loading {len(index_files)} index shards...")
        reps0, lookup0 = pickle_load(index_files[0])
        self.searcher = FaissFlatSearcher(reps0)
        self.lookup = list(lookup0)

        for path in index_files[1:]:
            reps, shard_lookup = pickle_load(path)
            self.searcher.add(reps)
            self.lookup.extend(shard_lookup)

        logger.info(f"Loaded index with {len(self.lookup)} documents.")

    def _load_model_and_tokenizer(self) -> None:
        """Load embedding model and tokenizer on specified device."""
        # Use float32 for CPU, float16 for GPU
        torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

        logger.info(f"Loading embedding model on {self.device}...")
        self.model = DenseModel.load(
            self.config.browsecomp_model_name,
            pooling="eos",
            normalize=True,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.browsecomp_model_name,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _load_dataset(self) -> None:
        """Load corpus dataset from HuggingFace."""
        dataset_name = self.config.browsecomp_dataset_name
        dataset_path = Path(dataset_name)

        if dataset_path.exists():
            logger.info(f"Loading local corpus file: {dataset_path}")
            self.dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        else:
            logger.info(f"Loading dataset: {dataset_name}")
            self.dataset = load_dataset(dataset_name, split="train")

        for idx, row in enumerate(self.dataset):
            docid = row.get("docid")
            if docid is None:
                continue
            self.docid_to_idx[docid] = idx
            self.docid_to_url[docid] = row.get("url") or ""

        logger.info(f"Loaded {len(self.docid_to_idx)} documents from corpus.")

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query string to embedding vector."""
        batch = self.tokenizer(
            self.task_prefix + query,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Use autocast for GPU, no-op context for CPU
        if self.device.startswith("cuda"):
            ctx = torch.amp.autocast(device_type="cuda")
        else:
            ctx = nullcontext()

        with ctx:
            with torch.no_grad():
                reps = self.model.encode_query(batch)

        return reps.cpu().numpy()

    def _get_doc(self, docid: str) -> Dict[str, Any]:
        """Retrieve full document from corpus by docid."""
        idx = self.docid_to_idx.get(docid)
        if idx is None:
            return {"docid": docid, "url": "", "text": ""}

        row = self.dataset[int(idx)]
        return {
            "docid": docid,
            "url": row.get("url") or "",
            "text": row.get("text") or "",
        }

    def execute(self, query: str, context: ResearchContext) -> Dict[str, Any]:
        """Execute BrowseComp search and return results."""
        if not self.searcher or not self.lookup:
            output = self.create_error_output(
                "browsecomp_search",
                query,
                "Searcher not initialized",
            )
            log_internet_search(
                tool="browsecomp_search",
                query=query,
                params={"top_k": self.config.browsecomp_top_k},
                result=output,
                extra={"source": "browsecomp"},
            )
            return output

        try:
            # Encode query
            q_reps = self._encode_query(query)

            # Search FAISS index
            scores, indices = self.searcher.search(q_reps, self.config.browsecomp_top_k)
            scores = scores[0]
            indices = indices[0]

            # Retrieve documents (truncate to avoid blowing up adaptive planning context)
            max_chars = self.config.browsecomp_max_chars
            results = []
            for score, idx in zip(scores, indices):
                docid = self.lookup[int(idx)]
                doc = self._get_doc(docid)
                text = doc.get("text") or ""
                if len(text) > max_chars:
                    text = text[:max_chars] + "\n\n[truncated]"
                results.append({
                    "docid": docid,
                    "score": float(score),
                    "url": doc.get("url"),
                    "text": text,
                })

            # Store in vector store if available
            content_stored_count = 0
            if self.vector_store and results:
                for i, result in enumerate(results):
                    if result.get("text"):
                        doc_id = self.vector_store.store_document(
                            content=result["text"],
                            metadata={
                                "type": "browsecomp_result",
                                "query": query,
                                "url": result.get("url", ""),
                                "docid": result.get("docid"),
                                "score": result.get("score"),
                                "search_rank": i + 1,
                                "timestamp": datetime.now().isoformat(),
                            },
                        )
                        if doc_id:
                            content_stored_count += 1

            output = self.create_success_output(
                tool_name="browsecomp_search",
                query=query,
                results=results,
                data_retrieved=len(results) > 0,
                results_count=len(results),
                source="browsecomp",
                content_stored_in_vector=content_stored_count,
                stored_in_vector=content_stored_count > 0,
            )

            log_internet_search(
                tool="browsecomp_search",
                query=query,
                params={"top_k": self.config.browsecomp_top_k},
                result=output,
                extra={"source": "browsecomp"},
            )
            return output

        except Exception as exc:
            logger.error(f"BrowseComp search failed: {exc}")
            output = self.create_error_output(
                "browsecomp_search",
                query,
                f"Search failed: {exc}",
            )
            log_internet_search(
                tool="browsecomp_search",
                query=query,
                params={"top_k": self.config.browsecomp_top_k},
                result=output,
                extra={"source": "browsecomp"},
            )
            return output
