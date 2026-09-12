"""LLM provider abstraction for DocuGraph AI.

Supports LLM-agnostic configuration with local-first priority:
- Ollama (local, default if available)
- OpenAI (cloud)
- Anthropic (cloud)

Usage:
    from docugraph.core.llm import get_llm_client, is_ollama_available

    # Auto-detect best available provider
    client = get_llm_client()

    # Or specify explicitly
    client = get_llm_client(provider=LLMProvider.OLLAMA)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, cast

import httpx

from docugraph.core.config import LLMProvider, get_config


class LLMProviderBase(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used."""
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            Generated text response
        """
        ...

    @abstractmethod
    def get_graphiti_client(self) -> Any:
        """Get a Graphiti-compatible LLM client.

        Returns:
            A client compatible with Graphiti's llm_client parameter
        """
        ...


class OllamaLLM(LLMProviderBase):
    """Ollama LLM provider for local inference."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._temperature = temperature

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Generate using Ollama API."""
        async with httpx.AsyncClient() as client:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = await client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature or self._temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=120.0,
            )
            response.raise_for_status()
            content: str = response.json()["message"]["content"]
            return content

    def get_graphiti_client(self) -> Any:
        """Get Graphiti-compatible Ollama client."""
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        llm_config = LLMConfig(
            api_key="ollama",
            model=self._model,
            small_model=self._model,
            base_url=f"{self._base_url}/v1",
            temperature=self._temperature,
        )
        return OpenAIGenericClient(config=llm_config)


class OpenAILLM(LLMProviderBase):
    """OpenAI LLM provider."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._temperature = temperature

        if not self._api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Generate using OpenAI API."""
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("openai package required: pip install openai") from e

        client = AsyncOpenAI(api_key=self._api_key)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=self._model,
            messages=cast("Any", messages),
            temperature=temperature or self._temperature,
            max_tokens=max_tokens,
        )
        content: str = response.choices[0].message.content or ""
        return content

    def get_graphiti_client(self) -> Any:
        """Get Graphiti-compatible OpenAI client."""
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_client import OpenAIClient

        llm_config = LLMConfig(
            api_key=self._api_key,
            model=self._model,
            small_model="gpt-4o-mini",
            temperature=self._temperature,
        )
        return OpenAIClient(config=llm_config)


class AnthropicLLM(LLMProviderBase):
    """Anthropic LLM provider."""

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._temperature = temperature

        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable required")

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Generate using Anthropic API."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ImportError("anthropic package required: pip install anthropic") from e

        client = AsyncAnthropic(api_key=self._api_key)
        response = await client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature or self._temperature,
        )
        text: str = response.content[0].text
        return text

    def get_graphiti_client(self) -> Any:
        """Get Graphiti-compatible Anthropic client."""
        from graphiti_core.llm_client.anthropic_client import AnthropicClient
        from graphiti_core.llm_client.config import LLMConfig

        llm_config = LLMConfig(
            api_key=self._api_key,
            model=self._model,
            small_model=self._model,
            temperature=self._temperature,
        )
        return AnthropicClient(config=llm_config)


def is_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running locally.

    Args:
        base_url: Ollama server URL

    Returns:
        True if Ollama is available
    """
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def get_available_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Get list of available Ollama models.

    Args:
        base_url: Ollama server URL

    Returns:
        List of model names
    """
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
    except Exception:  # nosec B110 -- Ollama may be down; empty list is the answer
        pass
    return []


def get_llm_provider(
    provider: LLMProvider | None = None,
    model: str | None = None,
) -> LLMProviderBase:
    """Get an LLM provider instance.

    Auto-detects the best available provider if not specified.
    Priority: Ollama (local) -> OpenAI -> Anthropic

    Args:
        provider: Specific provider to use, or None for auto-detection
        model: Specific model to use, or None for provider default

    Returns:
        LLM provider instance

    Raises:
        RuntimeError: If no LLM provider is available
    """
    config = get_config()

    # Use config if not specified
    if provider is None:
        provider = config.llm.provider

    # Auto-detect provider
    if provider == LLMProvider.AUTO:
        if is_ollama_available():
            provider = LLMProvider.OLLAMA
        elif os.environ.get("OPENAI_API_KEY"):
            provider = LLMProvider.OPENAI
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = LLMProvider.ANTHROPIC
        else:
            raise RuntimeError(
                "No LLM provider available. Options:\n"
                "  1. Start Ollama locally: ollama serve\n"
                "  2. Set OPENAI_API_KEY environment variable\n"
                "  3. Set ANTHROPIC_API_KEY environment variable\n"
                "For local-only operation, install Ollama: https://ollama.ai"
            )

    # Get model from config if not specified
    if model is None or model == "auto":
        model = config.llm.model if config.llm.model != "auto" else None

    temperature = config.llm.temperature

    # Create provider instance
    if provider == LLMProvider.OLLAMA:
        return OllamaLLM(
            model=model or "llama3.2",
            temperature=temperature,
        )
    elif provider == LLMProvider.OPENAI:
        return OpenAILLM(
            model=model or "gpt-4o-mini",
            temperature=temperature,
        )
    elif provider == LLMProvider.ANTHROPIC:
        return AnthropicLLM(
            model=model or "claude-3-haiku-20240307",
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_llm_client(
    provider: LLMProvider | None = None,
    model: str | None = None,
) -> LLMProviderBase:
    """Alias for get_llm_provider for backward compatibility."""
    return get_llm_provider(provider=provider, model=model)


def get_graphiti_llm_client(
    provider: LLMProvider | None = None,
    model: str | None = None,
) -> Any:
    """Get a Graphiti-compatible LLM client.

    This returns the native Graphiti client format for use with
    Graphiti's knowledge graph operations.

    Args:
        provider: Specific provider to use, or None for auto-detection
        model: Specific model to use, or None for provider default

    Returns:
        Graphiti-compatible LLM client
    """
    llm = get_llm_provider(provider=provider, model=model)
    return llm.get_graphiti_client()
