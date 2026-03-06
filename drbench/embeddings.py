"""
Unified embedding provider for DrBench.

Supports multiple backends:
- OpenAI (default)
- OpenRouter
- HuggingFace/SentenceTransformers (local, requires [hf] extra)
- vLLM server

Configuration:
    DRBENCH_EMBEDDING_PROVIDER: "openai" | "openrouter" | "huggingface" | "vllm"
    DRBENCH_EMBEDDING_MODEL: Model name (provider-specific)
    DRBENCH_EMBEDDING_DEVICE: Device for HuggingFace (e.g., "cuda:0", "cpu")
    VLLM_EMBEDDING_URL: URL for vLLM embedding server
"""

import logging
import os
import threading
from typing import List, Optional

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

# Default models for each provider
DEFAULT_MODELS = {
    "openai": "text-embedding-ada-002",
    "openrouter": "openai/text-embedding-ada-002",
    "huggingface": "Qwen/Qwen3-Embedding-4B",
    "vllm": "Qwen/Qwen3-Embedding-4B",
}

# Embedding dimensions for known models
EMBEDDING_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "Qwen/Qwen3-Embedding-4B": 2560,
    "Qwen/Qwen3-Embedding-8B": 4096,
}

# Cached HuggingFace model (loading is expensive)
_hf_model = None
_hf_model_name = None
_hf_lock = threading.Lock()


def _get_openai_embeddings(texts: List[str], model: str) -> List[List[float]]:
    """Generate embeddings using OpenAI API."""
    from openai import OpenAI

    # Use OpenAI's endpoint directly - OPENAI_BASE_URL may point to vLLM for LLM
    # which doesn't support embeddings. Users wanting vLLM embeddings should use
    # DRBENCH_EMBEDDING_PROVIDER=vllm instead.
    client = OpenAI(
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def _get_openrouter_embeddings(texts: List[str], model: str) -> List[List[float]]:
    """Generate embeddings using OpenRouter."""
    from openai import OpenAI

    openrouter_api_url = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    client = OpenAI(
        base_url=openrouter_api_url,
        api_key=openrouter_api_key,
    )
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def _get_huggingface_embeddings(texts: List[str], model: str) -> List[List[float]]:
    """Generate embeddings using SentenceTransformers."""
    global _hf_model, _hf_model_name

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "SentenceTransformers not installed. Install with: pip install drbench[hf]"
        )

    # Load model if not cached or different model requested (double-checked lock)
    if _hf_model is None or _hf_model_name != model:
        with _hf_lock:
            if _hf_model is None or _hf_model_name != model:
                logger.info(f"Loading embedding model: {model}")

                try:
                    import torch
                except ImportError:
                    raise ImportError(
                        "PyTorch not installed. Install with: pip install drbench[hf]"
                    )

                embedding_device = os.getenv("DRBENCH_EMBEDDING_DEVICE")
                force_cpu = bool(embedding_device and embedding_device.lower().startswith("cpu"))

                if torch.cuda.is_available() and not force_cpu:
                    num_gpus = torch.cuda.device_count()
                    default_device = f"cuda:{num_gpus - 1}" if num_gpus > 1 else "cuda:0"
                    embedding_device = embedding_device or default_device
                    logger.info(f"Loading embedding model on device: {embedding_device}")

                    try:
                        _hf_model = SentenceTransformer(
                            model,
                            device=embedding_device,
                            model_kwargs={
                                "attn_implementation": "flash_attention_2",
                                "torch_dtype": torch.float16,
                            },
                            tokenizer_kwargs={"padding_side": "left"},
                        )
                    except Exception as e:
                        logger.warning(f"Could not load with flash_attention_2: {e}")
                        _hf_model = SentenceTransformer(
                            model,
                            device=embedding_device,
                            tokenizer_kwargs={"padding_side": "left"},
                        )
                else:
                    if force_cpu:
                        logger.info("Loading embedding model on CPU (DRBENCH_EMBEDDING_DEVICE=cpu)")
                    _hf_model = SentenceTransformer(
                        model,
                        device="cpu" if force_cpu else None,
                        tokenizer_kwargs={"padding_side": "left"},
                    )

                _hf_model_name = model
                logger.info(f"Embedding model loaded: {model}")

    embeddings = _hf_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.tolist()


def _get_vllm_embeddings(texts: List[str], model: str) -> List[List[float]]:
    """Generate embeddings using vLLM server."""

    base_url = os.getenv("VLLM_EMBEDDING_URL")
    if not base_url:
        raise ValueError(
            "VLLM_EMBEDDING_URL must be set when using DRBENCH_EMBEDDING_PROVIDER=vllm"
        )

    client = OpenAI(base_url=f"{base_url}/v1", api_key="not-needed")
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def get_embeddings(
    texts: List[str],
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of texts to embed
        model: Model name (uses provider default if not specified)
        provider: Provider override ("openai", "openrouter", "huggingface", "vllm")

    Returns:
        List of embedding vectors

    Raises:
        ValueError: If provider is unknown or model/provider mismatch
    """
    if not texts:
        return []

    # Determine provider (CLI overrides via RunConfig)
    from drbench.config import get_run_config
    cfg = get_run_config()
    provider = provider or cfg.get_embedding_provider()
    if not provider:
        provider = os.getenv("DRBENCH_EMBEDDING_PROVIDER", "openai")
    provider = provider.lower()

    # Get model - use RunConfig override or env, then default
    if model is None:
        model = cfg.get_embedding_model() or os.getenv("DRBENCH_EMBEDDING_MODEL", DEFAULT_MODELS.get(provider))

    # Validate model/provider compatibility - NO silent switching
    if model and model.startswith("text-embedding-") and provider not in ("openai", "openrouter"):
        raise ValueError(
            f"Model '{model}' is an OpenAI model but provider is '{provider}'. "
            f"Either set DRBENCH_EMBEDDING_PROVIDER=openai or use a {provider}-compatible model."
        )

    # Route to provider
    if provider == "openai":
        return _get_openai_embeddings(texts, model)
    elif provider == "openrouter":
        return _get_openrouter_embeddings(texts, model)
    elif provider == "huggingface":
        return _get_huggingface_embeddings(texts, model)
    elif provider == "vllm":
        return _get_vllm_embeddings(texts, model)
    else:
        raise ValueError(
            f"Unknown embedding provider: '{provider}'. "
            f"Valid options: openai, openrouter, huggingface, vllm"
        )


def _normalize_embedding_model(model: Optional[str]) -> Optional[str]:
    """Normalize model name for dimension lookup."""
    if not model:
        return None
    if model.startswith("openai/"):
        return model.split("/", 1)[1]
    return model


def get_embedding_dimension(model: Optional[str] = None) -> int:
    """Get the embedding dimension for a model."""
    if model:
        normalized = _normalize_embedding_model(model)
        return EMBEDDING_DIMENSIONS.get(normalized, 1536)

    # Use default model for current provider (RunConfig override if present)
    from drbench.config import get_run_config
    cfg = get_run_config()
    provider = (cfg.get_embedding_provider() or os.getenv("DRBENCH_EMBEDDING_PROVIDER", "openai")).lower()
    default_model = cfg.get_embedding_model() or os.getenv("DRBENCH_EMBEDDING_MODEL", DEFAULT_MODELS.get(provider))
    normalized = _normalize_embedding_model(default_model)
    return EMBEDDING_DIMENSIONS.get(normalized, 1536)


def create_zero_embeddings(count: int, model: Optional[str] = None) -> List[np.ndarray]:
    """Create zero embeddings as fallback when API fails."""
    dim = get_embedding_dimension(model)
    return [np.zeros(dim) for _ in range(count)]
