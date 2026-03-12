# suppress warnings
import warnings

warnings.filterwarnings("ignore")

# import libraries
import argparse
import logging
import os
import textwrap
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from together import Together

# Import centralized configuration
from drbench import config
from drbench.config import RunConfig, get_run_config, set_run_config
from drbench.openrouter_logging import log_openrouter_generation

logger = logging.getLogger(__name__)

# Available providers and example models
AVAILABLE_PROVIDERS = ["openai", "openrouter", "vllm", "together", "azure"]
EXAMPLE_MODELS = {
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "openrouter": ["anthropic/claude-3.5-sonnet", "openai/gpt-4o"],
    "vllm": ["Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"],
    "together": ["meta-llama/Llama-3.3-70B-Instruct-Turbo"],
    "azure": ["gpt-4o"],
}


class AIAgentManager:
    """Manager class for AI agent operations.

    Uses explicit provider routing via DRBENCH_LLM_PROVIDER environment variable.
    No URL sniffing or model-based guessing.

    Args:
        model: Model name to use
        provider: LLM provider override (defaults to DRBENCH_LLM_PROVIDER env var)
        api_key: API key override (defaults to provider-specific env var)
        api_url: API URL override for vllm/openrouter (defaults to env var)
        max_tokens: Maximum tokens for generation
        temperature: Temperature for generation
        with_linebreak: Whether to wrap output with linebreaks
    """

    def __init__(
        self,
        model: str,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        with_linebreak: bool = False,
    ):
        # Determine provider explicitly - NO URL sniffing
        cfg = get_run_config()
        self.provider = provider or cfg.get_llm_provider()

        if self.provider not in AVAILABLE_PROVIDERS:
            raise ValueError(
                f"Unknown provider: '{self.provider}'. "
                f"Set DRBENCH_LLM_PROVIDER to one of: {', '.join(AVAILABLE_PROVIDERS)}"
            )

        self.client = None
        self.model = model
        self.actual_model = model  # May be overridden for openrouter (prefix stripped)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.with_linebreak = with_linebreak

        # Set up credentials based on explicit provider
        self._setup_credentials(api_key, api_url)
        self._initialize_client()

    def _setup_credentials(self, api_key: Optional[str], api_url: Optional[str]):
        """Set up API credentials based on provider."""
        if self.provider == "openai":
            self.api_key = api_key or config.OPENAI_API_KEY
            self.api_url = None
            if not self.api_key:
                raise ValueError(
                    "DRBENCH_LLM_PROVIDER=openai requires OPENAI_API_KEY to be set"
                )

        elif self.provider == "openrouter":
            self.api_key = api_key or config.OPENROUTER_API_KEY
            self.api_url = api_url or getattr(config, "OPENROUTER_API_URL", "https://openrouter.ai/api/v1")
            if not self.api_key:
                raise ValueError(
                    "DRBENCH_LLM_PROVIDER=openrouter requires OPENROUTER_API_KEY to be set"
                )

        elif self.provider == "vllm":
            self.api_key = api_key or config.VLLM_API_KEY or "not-needed"
            self.api_url = api_url or os.getenv("VLLM_API_URL")
            if not self.api_url:
                raise ValueError(
                    "DRBENCH_LLM_PROVIDER=vllm requires VLLM_API_URL to be set"
                )

        elif self.provider == "together":
            self.api_key = api_key or config.TOGETHER_API_KEY
            self.api_url = None
            if not self.api_key:
                raise ValueError(
                    "DRBENCH_LLM_PROVIDER=together requires TOGETHER_API_KEY to be set"
                )

        elif self.provider == "azure":
            self.api_key = api_key or config.AZURE_API_KEY
            self.api_url = api_url or config.AZURE_ENDPOINT
            if not self.api_key or not self.api_url:
                raise ValueError(
                    "DRBENCH_LLM_PROVIDER=azure requires AZURE_API_KEY and AZURE_ENDPOINT"
                )

    def _initialize_client(self):
        """Initialize the appropriate client for the provider."""
        if self.provider == "openai":
            self.client = OpenAI(api_key=self.api_key)

        elif self.provider == "openrouter":
            self.client = OpenAI(base_url=self.api_url, api_key=self.api_key)
            self.actual_model = self.model.removeprefix("openrouter/")

        elif self.provider == "vllm":
            self.client = OpenAI(base_url=f"{self.api_url}/v1", api_key=self.api_key)

        elif self.provider == "together":
            self.client = Together(api_key=self.api_key)

        elif self.provider == "azure":
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.api_url,
                api_version=config.AZURE_API_VERSION or "2024-02-15-preview",
            )

    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return AVAILABLE_PROVIDERS

    def get_structured_response_function(self) -> Any:
        """Get the structured response function"""
        if self.provider == "together":
            return self.client.chat.completions.create
        return self.client.beta.chat.completions.parse

    def prompt_llm(
        self,
        prompt: str,
        response_format: Optional[Any] = None,
        return_json: bool = False,
    ) -> str | Any:
        """
        Prompt the LLM and get a response.

        Args:
            prompt: The text prompt to send to the model
            response_format: Optional Pydantic model for structured output
            return_json: If True, extract JSON from response

        Returns:
            Generated text response or parsed response if response_format is provided
        """
        # Use logging, not print
        logger.debug(f"Prompting {self.model} via {self.provider}")

        if not self.client:
            raise ValueError("Client not initialized. Please provide a valid API key.")

        try:
            if response_format:
                structured_response_function = self.get_structured_response_function()
                response = structured_response_function(
                    model=self.actual_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_format=(
                        {
                            "type": "json_schema",
                            "schema": response_format.model_json_schema(),
                        }
                        if self.provider == "together"
                        else response_format
                    ),
                )
                if self.provider == "together":
                    output = response.choices[0].message.content
                    output = response_format.model_validate_json(output)
                else:
                    output = response.choices[0].message.parsed
                log_openrouter_generation(
                    response,
                    request_kind="chat.completions.parse",
                    requested_model=self.model,
                    resolved_model=self.actual_model,
                    source="gen_agent.prompt_llm",
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.actual_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                log_openrouter_generation(
                    response,
                    request_kind="chat.completions",
                    requested_model=self.model,
                    resolved_model=self.actual_model,
                    source="gen_agent.prompt_llm",
                )
                output = response.choices[0].message.content

            if self.with_linebreak:
                output = textwrap.fill(output, width=80)

            if return_json:
                return extract_json_from_response(output)

            return output

        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise

    def generate_text(self, prompt: str) -> str:
        """Generate text using specified parameters"""
        return self.prompt_llm(prompt)

    def generate_quiz_content(
        self, topic: str, difficulty: str = "intermediate", question_count: int = 5
    ) -> str:
        """Generate educational quiz content"""
        prompt = f"""Create a {difficulty} level quiz about {topic} with {question_count} questions. 
        Include multiple choice questions with explanations for each answer."""
        return self.prompt_llm(prompt)

    def analyze_content(self, text: str, analysis_type: str = "summary") -> str:
        """Analyze and process text content"""
        prompt = f"""Perform a {analysis_type} analysis of the following text:
        
        {text}
        
        Please provide a detailed {analysis_type}."""
        return self.prompt_llm(prompt)

    def provide_tutoring(
        self, subject: str, concept: str, difficulty_level: str = "beginner"
    ) -> str:
        """Provide educational tutoring and explanations"""
        prompt = f"""Act as a {difficulty_level} level tutor for {subject}. 
        Explain the concept of {concept} in a clear, educational manner with examples."""
        return self.prompt_llm(prompt)

    def process_document(
        self, document_path: str, processing_type: str = "summarize"
    ) -> str:
        """Process documents (placeholder for document processing)"""
        prompt = f"""Process the document at {document_path} and {processing_type} its content."""
        return self.prompt_llm(prompt)


# Global instance for backward compatibility
agent_manager = None


def initialize_agent_manager(
    model: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    with_linebreak: bool = False,
) -> AIAgentManager:
    """Initialize the global agent manager.

    Args:
        model: Model name to use
        provider: LLM provider (defaults to DRBENCH_LLM_PROVIDER)
        api_key: API key override
        max_tokens: Maximum tokens for generation
        temperature: Temperature for generation
        with_linebreak: Whether to wrap output

    Returns:
        Initialized AIAgentManager instance
    """
    global agent_manager
    agent_manager = AIAgentManager(
        model=model,
        provider=provider,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        with_linebreak=with_linebreak,
    )
    return agent_manager


def get_available_providers() -> List[str]:
    """Get available LLM providers"""
    return AVAILABLE_PROVIDERS


# Backward compatibility function
def prompt_llm(prompt: str, with_linebreak: bool = False, model: str = None) -> str:
    """
    Legacy function for backward compatibility.
    Requires initialize_agent_manager() to be called first.
    """
    global agent_manager
    if not agent_manager:
        raise ValueError(
            "Agent manager not initialized. Call initialize_agent_manager() first."
        )

    return agent_manager.prompt_llm(prompt)


def main():
    """Main function to demonstrate agent capabilities"""
    parser = argparse.ArgumentParser(description="AI Agent Manager")
    parser.add_argument(
        "--model", type=str, required=True, help="Model name to use"
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        help=f"LLM provider (default: DRBENCH_LLM_PROVIDER env var). Options: {', '.join(AVAILABLE_PROVIDERS)}"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a 3 line post about pizza",
        help="Text prompt to send to the model",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Output directory for logs (default: ./runs/gen_agent_<timestamp>)",
    )
    parser.add_argument(
        "--list-providers", action="store_true", help="List available providers"
    )

    args = parser.parse_args()

    if args.list_providers:
        print("\nAvailable Providers:")
        print("-" * 50)
        for provider in AVAILABLE_PROVIDERS:
            examples = EXAMPLE_MODELS.get(provider, [])
            print(f"• {provider}: {', '.join(examples[:2])}")
        print()
        return

    # Configure run dir for default-on logging
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = Path(__file__).resolve().parent.parent / "runs" / f"gen_agent_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    set_run_config(RunConfig(model=args.model, run_dir=run_dir, llm_provider=args.provider))

    # Initialize agent manager
    manager = initialize_agent_manager(
        model=args.model,
        provider=args.provider,
    )

    # Example usage
    print(f"\nUsing model: {args.model}")
    print(f"Provider: {manager.provider}")
    print(f"Prompt: {args.prompt}")
    print("-" * 80)

    try:
        response = manager.prompt_llm(args.prompt)
        print("\nResponse:")
        print(response)
    except Exception as e:
        print(f"\nError: {e}")

    print("-" * 80)


def extract_json_from_response(response: str) -> List[Dict[str, Any]]:
    """Extract JSON from AI response text"""
    import json
    import re

    # Try to find JSON array or object (original approach)
    json_match = re.search(r"(\[.*\]|\{.*\})", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try parsing the entire response (original approach)
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        # Apply more sophisticated cleaning and extraction only after initial attempts fail
        response_cleaned = response.strip()

        # Remove common introductory phrases
        intro_patterns = [
            r"^Here is the generated.*?:\s*",
            r"^Here's the.*?:\s*",
            r"^The.*?is:\s*",
            r"^Generated.*?:\s*",
            r"^Output:\s*",
            r"^Result:\s*",
            r"^Response:\s*",
        ]

        for pattern in intro_patterns:
            response_cleaned = re.sub(
                pattern, "", response_cleaned, flags=re.IGNORECASE | re.MULTILINE
            )

        response_cleaned = response_cleaned.strip()

        # Try to find JSON array or object with cleaned response
        json_match = re.search(r"(\[.*\]|\{.*\})", response_cleaned, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try cleaning up common JSON issues
                json_str = re.sub(r",(\s*[\]\}])", r"\1", json_str)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        # Try parsing the entire cleaned response
        try:
            return json.loads(response_cleaned)
        except json.JSONDecodeError:
            # Last attempt: line-by-line extraction
            lines = response_cleaned.split("\n")
            json_lines = []
            bracket_count = 0
            brace_count = 0
            started = False

            for line in lines:
                stripped = line.strip()
                if not started and (
                    stripped.startswith("[") or stripped.startswith("{")
                ):
                    started = True

                if started:
                    json_lines.append(line)
                    bracket_count += line.count("[") - line.count("]")
                    brace_count += line.count("{") - line.count("}")

                    # If we've closed all brackets/braces, we're done
                    if bracket_count <= 0 and brace_count <= 0 and started:
                        break

            if json_lines:
                try:
                    json_str = "\n".join(json_lines)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

            raise ValueError(
                f"Could not extract valid JSON from response\nResponse:\n{response}"
            )
