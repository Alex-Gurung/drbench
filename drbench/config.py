"""
Centralized configuration management for DrBench.

Configuration Philosophy:
- Secrets/Infrastructure: Environment variables only (OPENAI_API_KEY, VLLM_API_URL, etc.)
- Providers: Environment defaults, CLI overrides (DRBENCH_LLM_PROVIDER, DRBENCH_EMBEDDING_PROVIDER)
- Run parameters: CLI only, no env vars (--model, --max-iterations, --run-dir)

This separation ensures:
- Secrets never appear in CLI history
- Experiments are reproducible (CLI args are explicit)
- No need to juggle env vars between runs
"""

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)
else:
    load_dotenv(override=True)

# =============================================================================
# Secrets (env-only, never from CLI)
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
VLLM_API_KEY = os.getenv("VLLM_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")

# =============================================================================
# Infrastructure (env-only)
# =============================================================================
VLLM_API_URL = os.getenv("VLLM_API_URL")
VLLM_EMBEDDING_URL = os.getenv("VLLM_EMBEDDING_URL")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
DRBENCH_DATA_DIR = os.getenv("DRBENCH_DATA_DIR")
DRBENCH_DOCKER_IMAGE = os.getenv("DRBENCH_DOCKER_IMAGE", "drbench-services")
DRBENCH_DOCKER_TAG = os.getenv("DRBENCH_DOCKER_TAG", "latest")

# =============================================================================
# Providers (env defaults, CLI overrides)
# =============================================================================
# LLM provider: openai | vllm | openrouter | azure
DRBENCH_LLM_PROVIDER = os.getenv("DRBENCH_LLM_PROVIDER", "openai")

# Embedding provider: openai | openrouter | huggingface | vllm
DRBENCH_EMBEDDING_PROVIDER = os.getenv("DRBENCH_EMBEDDING_PROVIDER", "openai")
DRBENCH_EMBEDDING_MODEL = os.getenv("DRBENCH_EMBEDDING_MODEL")
DRBENCH_EMBEDDING_DEVICE = os.getenv("DRBENCH_EMBEDDING_DEVICE")

# =============================================================================
# RunConfig - Runtime configuration for task execution
# =============================================================================
# Run parameters come from CLI args, not env vars.
# Use RunConfig() with explicit values from argparse.

@dataclass
class RunConfig:
    """Runtime configuration for DrBench task execution.

    All run parameters should be passed explicitly via CLI args.
    Use from_cli() to create from argparse Namespace.

    Example:
        # From CLI args
        args = parser.parse_args()
        cfg = RunConfig.from_cli(args)
        set_run_config(cfg)

        # Direct instantiation
        cfg = RunConfig(
            model="gpt-4",
            max_iterations=10,
            run_dir=Path("./runs/my_experiment"),
        )
    """

    # Model (CLI: --model, required for experiments)
    model: Optional[str] = None

    # Execution parameters (CLI: --max-iterations, --concurrent-actions)
    max_iterations: int = 10
    concurrent_actions: int = 3

    # Semantic search (CLI: --semantic-threshold)
    semantic_threshold: float = 0.7

    # Output directory (CLI: --run-dir, required for logging)
    run_dir: Optional[Path] = None

    # Logging flags (CLI: --no-log-searches, --no-log-prompts, etc.)
    log_searches: bool = True  # Default ON
    log_prompts: bool = True   # Default ON
    log_generations: bool = True  # Default ON
    verbose: bool = False

    # Web access control (CLI: --no-web)
    no_web: bool = False

    # Provider overrides (CLI: --llm-provider, --embedding-provider)
    # If None, uses env var defaults
    llm_provider: Optional[str] = None
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None

    # Report style (CLI: --report-style)
    report_style: str = "research_report"  # "research_report" or "concise_qa"

    # BrowseComp-Plus retrieval (CLI: --browsecomp, --browsecomp-index, etc.)
    # When enabled, replaces live internet search with fixed corpus retrieval
    browsecomp_enabled: bool = False
    browsecomp_index_glob: str = "/home/toolkit/BrowseComp-Plus/indexes/qwen3-embedding-4b/corpus.shard*_of_*.pkl"
    browsecomp_model_name: str = "Qwen/Qwen3-Embedding-4B"
    browsecomp_dataset_name: str = "Tevatron/browsecomp-plus-corpus"
    browsecomp_top_k: int = 5
    browsecomp_max_chars: int = 8000  # Max chars per result (~2k tokens)

    def get_llm_provider(self) -> str:
        """Get LLM provider, falling back to env var."""
        return self.llm_provider or DRBENCH_LLM_PROVIDER

    def get_embedding_provider(self) -> str:
        """Get embedding provider, falling back to env var."""
        return self.embedding_provider or DRBENCH_EMBEDDING_PROVIDER

    def get_embedding_model(self) -> Optional[str]:
        """Get embedding model, falling back to env var."""
        return self.embedding_model or DRBENCH_EMBEDDING_MODEL

    @classmethod
    def from_cli(cls, args) -> "RunConfig":
        """Create config from argparse Namespace.

        Expected args attributes:
            model, max_iterations, concurrent_actions, semantic_threshold,
            run_dir, no_log, no_log_searches, no_log_prompts, no_log_generations,
            verbose, llm_provider, embedding_provider, embedding_model,
            browsecomp, browsecomp_index, browsecomp_model, browsecomp_dataset, browsecomp_top_k
        """
        return cls(
            model=getattr(args, "model", None),
            max_iterations=getattr(args, "max_iterations", 10),
            concurrent_actions=getattr(args, "concurrent_actions", 3),
            semantic_threshold=getattr(args, "semantic_threshold", 0.7),
            run_dir=Path(args.run_dir) if getattr(args, "run_dir", None) else None,
            log_searches=not getattr(args, "no_log_searches", False) and not getattr(args, "no_log", False),
            log_prompts=not getattr(args, "no_log_prompts", False) and not getattr(args, "no_log", False),
            log_generations=not getattr(args, "no_log_generations", False) and not getattr(args, "no_log", False),
            verbose=getattr(args, "verbose", False),
            no_web=getattr(args, "no_web", False),
            llm_provider=getattr(args, "llm_provider", None),
            embedding_provider=getattr(args, "embedding_provider", None),
            embedding_model=getattr(args, "embedding_model", None),
            browsecomp_enabled=getattr(args, "browsecomp", False),
            browsecomp_index_glob=getattr(args, "browsecomp_index", "/home/toolkit/BrowseComp-Plus/indexes/qwen3-embedding-4b/corpus.shard*_of_*.pkl"),
            browsecomp_model_name=getattr(args, "browsecomp_model", "Qwen/Qwen3-Embedding-4B"),
            browsecomp_dataset_name=getattr(args, "browsecomp_dataset", "Tevatron/browsecomp-plus-corpus"),
            browsecomp_top_k=getattr(args, "browsecomp_top_k", 5),
            browsecomp_max_chars=getattr(args, "browsecomp_max_chars", 8000),
            report_style=getattr(args, "report_style", "research_report"),
        )

    def to_dict(self) -> dict:
        """Serialize config for logging."""
        return {
            "model": self.model,
            "max_iterations": self.max_iterations,
            "concurrent_actions": self.concurrent_actions,
            "semantic_threshold": self.semantic_threshold,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "log_searches": self.log_searches,
            "log_prompts": self.log_prompts,
            "log_generations": self.log_generations,
            "no_web": self.no_web,
            "llm_provider": self.get_llm_provider(),
            "embedding_provider": self.get_embedding_provider(),
            "embedding_model": self.get_embedding_model(),
            "browsecomp_enabled": self.browsecomp_enabled,
            "browsecomp_index_glob": self.browsecomp_index_glob,
            "browsecomp_model_name": self.browsecomp_model_name,
            "browsecomp_top_k": self.browsecomp_top_k,
            "browsecomp_max_chars": self.browsecomp_max_chars,
            "report_style": self.report_style,
        }


# Thread-local config — each thread gets its own RunConfig so parallel
# workers can have different run_dir without clobbering each other.
_thread_local = threading.local()


def get_run_config() -> RunConfig:
    """Get the current thread's run config.

    Returns existing config or creates default.
    For experiments, always call set_run_config() first with CLI args.
    """
    cfg = getattr(_thread_local, "run_config", None)
    if cfg is None:
        cfg = RunConfig()
        _thread_local.run_config = cfg
    return cfg


def set_run_config(config: RunConfig) -> None:
    """Set the current thread's run config.

    Call this at the start of a run with CLI-parsed config.
    """
    _thread_local.run_config = config


# =============================================================================
# Path utilities
# =============================================================================
REPO_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "drbench" / "data" / "tasks"


def get_data_dir() -> Path:
    """Get the task data directory.

    Resolved at call time so CLI can set DRBENCH_DATA_DIR before use.
    """
    data_dir = os.getenv("DRBENCH_DATA_DIR")
    if data_dir:
        return Path(data_dir)
    return DEFAULT_DATA_DIR


# =============================================================================
# Validation utilities
# =============================================================================

def validate_required_keys(required_keys: list[str]) -> bool:
    """Validate that required environment variables are set."""
    missing = [k for k in required_keys if not globals().get(k)]
    if missing:
        print(f"Warning: Missing required environment variables: {', '.join(missing)}")
        return False
    return True


def validate_provider_config(provider: str) -> None:
    """Validate that required config is set for a provider.

    Raises:
        ValueError: If required config is missing
    """
    if provider == "vllm":
        # Read from env directly (not module-level cache) to support runtime export
        if not os.getenv("VLLM_API_URL"):
            raise ValueError(
                f"DRBENCH_LLM_PROVIDER={provider} requires VLLM_API_URL to be set"
            )
    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError(
                f"DRBENCH_LLM_PROVIDER={provider} requires OPENAI_API_KEY to be set"
            )
    elif provider == "openrouter":
        if not OPENROUTER_API_KEY:
            raise ValueError(
                f"DRBENCH_LLM_PROVIDER={provider} requires OPENROUTER_API_KEY to be set"
            )
    elif provider == "azure":
        if not AZURE_API_KEY or not AZURE_ENDPOINT:
            raise ValueError(
                f"DRBENCH_LLM_PROVIDER={provider} requires AZURE_API_KEY and AZURE_ENDPOINT"
            )
