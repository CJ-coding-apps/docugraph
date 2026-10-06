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

# One literal each, because the preflight check and the client it builds have
# to agree on which server and which model they are talking about.
#
# These are the only model names left in this package, and they are here for the
# one provider that can be asked what it has. Ollama runs on this machine, keeps
# its models in a local store, and answers `GET /api/tags` with the list;
# `ollama pull llama3.2` is a command any reader can run. A cloud provider
# cannot be treated the same way: a name written here is a name the provider
# will retire on its own schedule, and after that date every call fails with a
# 404 that says nothing about which line of what config put it there. So the
# cloud providers have no default at all -- they take the model from
# `llm.model` and say so when it is missing. See `_cloud_model_problem`.
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "llama3.2"


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
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature, or None to leave the provider's
                own default in place. None is the default because current
                reasoning models reject the parameter outright when the value
                is not theirs, so sending one the user never asked for is a
                call that cannot succeed.
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


def _make_resilient_graphiti_client(llm_config: Any) -> Any:
    """Build a Graphiti OpenAIGenericClient that tolerates loose JSON shapes.

    Graphiti parses each extraction response as ``Model(**json.loads(text))``.
    Many OpenAI-compatible endpoints (Ollama/llama.cpp-served models) do not
    strictly honor the requested ``json_schema`` and return a bare JSON array
    (e.g. ``[{...}]``) instead of the wrapping object
    (``{"extracted_entities": [...]}``). That makes ``Model(**list)`` raise
    ``TypeError: argument after ** must be a mapping, not list``.

    This subclass coerces such a bare list back into the single list-typed
    field the target ``response_model`` declares, so local-LLM graph
    extraction works without requiring a strict-schema cloud model.
    """
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    if not llm_config.model:
        # There used to be a `or "gpt-4.1-mini"` here, which was the worst of both
        # worlds: it named an OpenAI model on a client built only for local
        # OpenAI-compatible servers, so an Ollama user without a model configured
        # got a request for a name Ollama could not have, and the error said
        # "model not found" rather than "no model configured".
        raise ValueError(
            "A Graphiti LLM client needs a model name, and this one was built "
            "without one. OllamaLLM/OpenAILLM/AnthropicLLM all resolve a model "
            "in their constructors, so reaching here means a client was built "
            "directly from an empty LLMConfig."
        )

    class ResilientGraphitiClient(OpenAIGenericClient):
        async def _generate_response(
            self,
            messages: Any,
            response_model: Any = None,
            max_tokens: Any = None,
            model_size: Any = None,  # noqa: ARG002 -- Graphiti-mandated signature
        ) -> dict[str, Any]:
            # Reimplements the parent request loop with tolerant JSON parsing:
            # OpenAI-compatible models served via Ollama/llama.cpp often wrap
            # their output in prose or a ```json fence, or emit a bare list,
            # which the parent's strict json.loads(...) + Model(**result) path
            # can't handle. We extract the JSON payload and coerce its shape.
            import openai as _openai
            from graphiti_core.llm_client.errors import EmptyResponseError, RateLimitError

            openai_messages: list[dict[str, str]] = []
            for m in messages:
                m.content = self._clean_input(m.content)
                if m.role in ("user", "system"):
                    openai_messages.append({"role": m.role, "content": m.content})

            token_limit: int = int(max_tokens or self.max_tokens)
            # `max_tokens` and not `max_completion_tokens`: this client only ever
            # talks to an OpenAI-*compatible* local server (Ollama, llama.cpp),
            # which is the endpoint that kept `max_tokens`. The managed OpenAI API
            # is reached through graphiti_core's own OpenAIClient, not here.
            # `temperature` is omitted rather than sent as None when unset, so an
            # unset value cannot reach the server as a null the server must
            # interpret.
            request: dict[str, Any] = {
                "model": self.model,
                "messages": openai_messages,
                "max_tokens": token_limit,
                "response_format": self._build_response_format(response_model),
            }
            if self.temperature is not None:
                request["temperature"] = self.temperature
            try:
                response = await self.client.chat.completions.create(**request)
                text = response.choices[0].message.content or ""
                if not text:
                    raise EmptyResponseError("LLM returned an empty response")
                parsed = self._loads_tolerant(self._strip_code_fences(text))
                return self._coerce_to_schema(parsed, response_model)
            except _openai.RateLimitError as e:
                raise RateLimitError from e

        @staticmethod
        def _loads_tolerant(text: str) -> Any:
            """json.loads, but fall back to the first JSON value embedded in prose."""
            import json as _json
            import re as _re

            try:
                return _json.loads(text)
            except _json.JSONDecodeError:
                # Grab the outermost {...} or [...] block a chatty model wrapped
                # its answer in (e.g. a reasoning preamble before the JSON).
                match = _re.search(r"(\{.*\}|\[.*\])", text, _re.DOTALL)
                if match:
                    return _json.loads(match.group(1))
                raise

        @classmethod
        def _coerce_to_schema(cls, result: Any, response_model: Any) -> dict[str, Any]:
            if response_model is None:
                return result if isinstance(result, dict) else {}

            # Find the model's sole list-typed field and its item model, if any.
            list_field, item_model = cls._list_field(response_model)

            # A bare list: attach it to that list field.
            if isinstance(result, list):
                if list_field is None:
                    return {}
                result = {list_field: result}

            if not isinstance(result, dict):
                return {}

            # Normalize each list item's keys to the item model's field names
            # (e.g. models that emit `entity_name` where the schema wants
            # `name`). Only *missing* fields are filled, so exact-match fields
            # like `source_entity_name` are never disturbed.
            if list_field and item_model and isinstance(result.get(list_field), list):
                result[list_field] = [
                    cls._normalize_item(item, item_model) if isinstance(item, dict) else item
                    for item in result[list_field]
                ]
            return result

        @staticmethod
        def _list_field(model: Any) -> tuple[str | None, Any]:
            import typing

            for name, field in model.model_fields.items():
                ann = field.annotation
                if "list" in str(ann).lower():
                    args = typing.get_args(ann)
                    item_model = args[0] if args and hasattr(args[0], "model_fields") else None
                    return name, item_model
            return None, None

        @staticmethod
        def _normalize_item(item: dict[str, Any], item_model: Any) -> dict[str, Any]:
            fields = set(item_model.model_fields)
            present = {k for k in item if k in fields}
            out = dict(item)
            for field_name in fields - present:
                # Find an unclaimed key that clearly aliases this field, e.g.
                # `entity_name` -> `name`. Suffix match keeps it conservative.
                for key in list(out):
                    if key in fields:
                        continue
                    if key.endswith("_" + field_name) or key == field_name:
                        out[field_name] = out.pop(key)
                        break
            return out

    return ResilientGraphitiClient(config=llm_config)


def _graphiti_temperature(value: float | None) -> float:
    """Hand Graphiti a temperature that may be unset.

    Graphiti annotates ``LLMConfig.temperature`` as ``float`` and then branches
    on ``temperature is not None`` before putting it on the request -- so None is
    how its own implementation says "leave the model's default alone", even
    though the signature does not admit it. The conversion lives here so that
    one mismatch is described once rather than at three call sites.

    Passing its 1.0 default instead is not equivalent: Graphiti skips the
    parameter only for the reasoning models it recognises by prefix, and a
    temperature nobody chose is still a temperature the model may reject.
    """
    return cast("float", value)


class OllamaLLM(LLMProviderBase):
    """Ollama LLM provider for local inference."""

    def __init__(
        self,
        model: str = OLLAMA_DEFAULT_MODEL,
        base_url: str = OLLAMA_DEFAULT_BASE_URL,
        temperature: float | None = None,
        small_model: str | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._temperature = temperature
        # Graphiti's cheap pass runs on the same local model unless told
        # otherwise, which is the honest default for a single-GPU box.
        self._small_model = small_model or model

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

            # `temperature or self._temperature` was wrong for 0.0, which is
            # falsy: asking for a deterministic run silently got whatever the
            # instance default was. Compare against None explicitly.
            effective_temperature = self._temperature if temperature is None else temperature
            options: dict[str, Any] = {"num_predict": max_tokens}
            if effective_temperature is not None:
                options["temperature"] = effective_temperature

            response = await client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": False,
                    "options": options,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            content: str = response.json()["message"]["content"]
            return content

    def get_graphiti_client(self) -> Any:
        """Get Graphiti-compatible Ollama client."""
        from graphiti_core.llm_client.config import LLMConfig

        llm_config = LLMConfig(
            api_key="ollama",
            model=self._model,
            small_model=self._small_model,
            base_url=f"{self._base_url}/v1",
            temperature=_graphiti_temperature(self._temperature),
        )
        return _make_resilient_graphiti_client(llm_config)


class OpenAILLM(LLMProviderBase):
    """OpenAI LLM provider."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        small_model: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._temperature = temperature

        if not self._api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
        if not model:
            # No default on purpose. See the constants at the top of this module.
            raise ValueError(
                "OpenAI needs a model name and none was configured. Set "
                "`llm.model` in docugraph.yaml (or DOCUGRAPH_LLM__MODEL), e.g. "
                "a model id from https://platform.openai.com/docs/models."
            )
        self._model = model
        self._small_model = small_model or model

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
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # `max_completion_tokens`, not `max_tokens`. Chat Completions has two
        # output-token parameters and they are not interchangeable: the o-series
        # and the GPT-5 family reject `max_tokens` outright ("Unsupported
        # parameter"), while the older models accept it and ignore the newer
        # name. The newer name is the one both generations understand.
        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        # Temperature is sent only when it is set, and the same reasoning models
        # reject any value other than their own fixed default -- so a 0.0 that
        # nobody chose would fail the call. `temperature or self._temperature`
        # additionally got 0.0 wrong, since 0.0 is falsy.
        effective_temperature = self._temperature if temperature is None else temperature
        if effective_temperature is not None:
            request["temperature"] = effective_temperature

        response = await client.chat.completions.create(**request)
        content: str = response.choices[0].message.content or ""
        return content

    def get_graphiti_client(self) -> Any:
        """Get Graphiti-compatible OpenAI client."""
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_client import OpenAIClient

        llm_config = LLMConfig(
            api_key=self._api_key,
            model=self._model,
            small_model=self._small_model,
            temperature=_graphiti_temperature(self._temperature),
        )
        return OpenAIClient(config=llm_config)


class AnthropicLLM(LLMProviderBase):
    """Anthropic LLM provider."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        small_model: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._temperature = temperature

        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable required")
        if not model:
            # No default on purpose; see the constants at the top of this module.
            raise ValueError(
                "Anthropic needs a model name and none was configured. Set "
                "`llm.model` in docugraph.yaml (or DOCUGRAPH_LLM__MODEL), e.g. "
                "a model id from https://docs.anthropic.com/en/docs/about-claude/models."
            )
        self._model = model
        self._small_model = small_model or model

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
        temperature: float | None = None,  # noqa: ARG002 -- LLMProviderBase declares it
        max_tokens: int = 4096,
    ) -> str:
        """Generate using Anthropic API."""
        try:
            from anthropic import AsyncAnthropic
            from anthropic.types import TextBlock
        except ImportError as e:
            raise ImportError("anthropic package required: pip install anthropic") from e

        client = AsyncAnthropic(api_key=self._api_key)
        # `max_tokens` is required here, and is the only optional-looking
        # parameter sent. `temperature` is not sent, and must not be added back.
        # The 1.x SDK removed it from Messages.create's signature and rejects it
        # before the request is built -- `TypeError: AsyncMessages.create() got
        # an unexpected keyword argument 'temperature'` -- so passing it made
        # every Anthropic call fail at runtime, whatever the type checker said.
        # The argument stays in the signature because LLMProviderBase declares it
        # and the OpenAI and Ollama providers do honour it; on this provider it is
        # inert, which is a difference worth knowing rather than one to paper
        # over by smuggling it through `extra_body`.
        #
        # `system` is omitted rather than sent as "", which is a real difference:
        # an empty string is a system prompt saying nothing, and it is carried in
        # the request whether or not the caller wanted one.
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            request["system"] = system_prompt

        response = await client.messages.create(**request)
        # `content` is a union of block types and only TextBlock carries `.text`.
        # Reading `content[0].text` assumed the first block was text, which stops
        # being true as soon as a reply leads with a thinking or tool-use block --
        # and then the caller got an AttributeError naming no provider and no
        # model. Collect the text blocks instead, and if there are none, say what
        # came back rather than returning an empty string that reads as success.
        text = "".join(block.text for block in response.content if isinstance(block, TextBlock))
        if not text:
            kinds = ", ".join(sorted({type(block).__name__ for block in response.content}))
            raise RuntimeError(
                f"Anthropic returned no text for model {self._model!r}; "
                f"content blocks were: {kinds or 'none'}."
            )
        return text

    def get_graphiti_client(self) -> Any:
        """Get Graphiti-compatible Anthropic client."""
        from graphiti_core.llm_client.anthropic_client import AnthropicClient
        from graphiti_core.llm_client.config import LLMConfig

        llm_config = LLMConfig(
            api_key=self._api_key,
            model=self._model,
            small_model=self._small_model,
            temperature=_graphiti_temperature(self._temperature),
        )
        return AnthropicClient(config=llm_config)


def is_ollama_available(base_url: str = OLLAMA_DEFAULT_BASE_URL) -> bool:
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


def get_available_ollama_models(base_url: str = OLLAMA_DEFAULT_BASE_URL) -> list[str]:
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


class LLMUnavailableError(RuntimeError):
    """No LLM can serve the requested model.

    Raised for both "no provider is set up" and "the provider is reachable but
    does not have the model pulled". The second case is the reason this exists:
    an Ollama server that is running without the configured model passes every
    reachability check, then fails deep inside Graphiti with a provider-specific
    404 traceback. The graph tools claim to degrade gracefully; this is what
    makes that true.

    Subclasses RuntimeError so existing ``except RuntimeError`` handlers keep
    catching it.
    """


def _model_is_available(wanted: str, available: list[str]) -> bool:
    """Whether `wanted` names a model the Ollama server actually has.

    Ollama tags its models, so a configured ``llama3.2`` is stored as
    ``llama3.2:latest``. Compare on the untagged name, but only when the
    request is itself untagged -- an explicit ``llama3.2:7b`` must match
    exactly, or we would send a request for a tag that is not there.
    """
    for name in available:
        if name == wanted:
            return True
        if ":" not in wanted and name.split(":", 1)[0] == wanted:
            return True
    return False


def _diagnose_ollama(wanted_model: str, base_url: str) -> tuple[str | None, list[str]]:
    """Why Ollama cannot serve `wanted_model`, plus what it does have.

    Returns ``(None, installed)`` when Ollama can serve the request.
    """
    if not is_ollama_available(base_url):
        return f"Ollama is not reachable at {base_url}", []

    available = get_available_ollama_models(base_url)
    if not available:
        return f"Ollama at {base_url} is running but reports no models installed", []

    if not _model_is_available(wanted_model, available):
        return (
            f"Ollama at {base_url} is running but does not have the model "
            f"'{wanted_model}' (installed: {', '.join(available[:5])})",
            available,
        )
    return None, available


# --- Cloud models: none by default, and checked against the provider's list ---

OPENAI_MODELS_URL = "https://api.openai.com/v1/models"
ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
# Anthropic requires an explicit API version on every request; there is no
# unversioned form. Pinned here rather than floated, because the shape of the
# response this reads is part of the version.
ANTHROPIC_VERSION = "2023-06-01"

# Where a user sets the model, so the error can name the exact variable rather
# than describing where it lives.
_MODEL_SETTING = {
    LLMProvider.OPENAI: "DOCUGRAPH_LLM__MODEL",
    LLMProvider.ANTHROPIC: "DOCUGRAPH_LLM__MODEL",
}
_MODEL_DOCS = {
    LLMProvider.OPENAI: "https://platform.openai.com/docs/models",
    LLMProvider.ANTHROPIC: "https://docs.anthropic.com/en/docs/about-claude/models",
}

# One verification per (provider, model) per process. Model catalogs change on
# the provider's schedule and not this process's, and the MCP server is
# long-lived, so without this every graph call would spend a round trip asking
# a question whose answer cannot have changed. Keyed without the API key, which
# is deliberately not held anywhere but the client: a process reads its keys
# once, so a second key for the same provider is not a case that arises.
_MODEL_LIST_CACHE: dict[tuple[str, str], list[str]] = {}


def _list_openai_models(api_key: str) -> list[str]:
    """Every model id OpenAI will serve this key."""
    response = httpx.get(
        OPENAI_MODELS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10.0,
    )
    response.raise_for_status()
    data = response.json().get("data", [])
    return [str(entry["id"]) for entry in data]


def _list_anthropic_models(api_key: str) -> list[str]:
    """Every model id Anthropic will serve this key.

    Unlike OpenAI's, this endpoint is paginated, so a single GET returns the
    first page and quietly omits the rest -- which would report a model the
    account can plainly use as "not found".
    """
    headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}
    ids: list[str] = []
    after: str | None = None
    # Bounded: the catalog is a few pages, and a server that keeps saying
    # `has_more` must not spin here.
    for _ in range(10):
        params: dict[str, Any] = {"limit": 100}
        if after:
            params["after_id"] = after
        response = httpx.get(ANTHROPIC_MODELS_URL, headers=headers, params=params, timeout=10.0)
        response.raise_for_status()
        payload = response.json()
        ids += [str(entry["id"]) for entry in payload.get("data", [])]
        after = payload.get("last_id")
        if not payload.get("has_more") or not after:
            break
    return ids


def _cloud_model_problem(provider: LLMProvider, model: str | None, api_key: str) -> str | None:
    """Why `provider` cannot serve `model`, or None if it can.

    Two questions, in order: is a model named at all, and does the provider
    still offer it. The second is the one worth a network call -- a model id is
    the one piece of configuration whose validity expires without anything in
    this repository changing, and the failure it produces at call time is a 404
    from inside Graphiti that names neither the config nor the file.
    """
    setting = _MODEL_SETTING[provider]
    if not model or model == "auto":
        return (
            f"no model is configured for {provider.value}: cloud providers have no "
            f"default here, because a name compiled into this package is retired on "
            f"the provider's schedule and then every call fails with a 404 that "
            f"names no config. Set {setting} to a model id from {_MODEL_DOCS[provider]}. "
            f"Leave it unset to keep using Ollama."
        )

    cached = _MODEL_LIST_CACHE.get((provider.value, model))
    if cached is None:
        try:
            cached = (
                _list_openai_models(api_key)
                if provider == LLMProvider.OPENAI
                else _list_anthropic_models(api_key)
            )
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status in (401, 403):
                return f"the {provider.value} API rejected the key ({status})"
            return (
                f"could not list {provider.value} models to check '{model}' "
                f"(HTTP {status} from {OPENAI_MODELS_URL if provider == LLMProvider.OPENAI else ANTHROPIC_MODELS_URL})"
            )
        except httpx.HTTPError as e:
            return (
                f"could not reach the {provider.value} model list to check '{model}' "
                f"({e.__class__.__name__}); the model is unverified, not confirmed"
            )
        _MODEL_LIST_CACHE[(provider.value, model)] = cached

    if model in cached:
        return None
    shown = ", ".join(cached[:10]) or "none"
    return (
        f"{provider.value} does not list a model named '{model}' "
        f"(available: {shown}{', ...' if len(cached) > 10 else ''})"
    )


def _provider_problem(provider: LLMProvider, model: str | None) -> tuple[str | None, list[str]]:
    """Why `provider` cannot serve `model` right now, plus Ollama's model list."""
    if provider == LLMProvider.OLLAMA:
        return _diagnose_ollama(model or OLLAMA_DEFAULT_MODEL, OLLAMA_DEFAULT_BASE_URL)
    if provider == LLMProvider.OPENAI:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "OPENAI_API_KEY is not set", []
        return _cloud_model_problem(provider, model, api_key), []
    if provider == LLMProvider.ANTHROPIC:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return "ANTHROPIC_API_KEY is not set", []
        return _cloud_model_problem(provider, model, api_key), []
    return f"Unsupported LLM provider: {provider}", []


def _unavailable_error(
    problems: list[str],
    model: str | None,
    ollama_models: list[str],
) -> LLMUnavailableError:
    """Build the actionable error for an unusable LLM configuration."""
    lines = [
        "The knowledge-graph tools need an LLM, and none is available.",
        "",
        "Why not:",
    ]
    lines += [f"  - {problem}" for problem in problems]
    lines += [
        "",
        "To fix, pick one:",
        f"    ollama pull {model or OLLAMA_DEFAULT_MODEL}",
        "    export OPENAI_API_KEY=sk-...   # and set DOCUGRAPH_LLM__MODEL",
        "    export ANTHROPIC_API_KEY=sk-ant-...   # and set DOCUGRAPH_LLM__MODEL",
    ]
    if ollama_models:
        lines += [
            "",
            "Ollama already has a model you could use instead:",
            f"    DOCUGRAPH_LLM__MODEL={ollama_models[0]}",
        ]
    lines += [
        "",
        "Vector and keyword search need no LLM: `docugraph search`, `search_docs` "
        "and `hybrid_search` work as they are.",
    ]
    return LLMUnavailableError("\n".join(lines))


def get_llm_provider(
    provider: LLMProvider | None = None,
    model: str | None = None,
) -> LLMProviderBase:
    """Get an LLM provider instance, validating that it can actually serve.

    Auto-detects the best available provider if not specified.
    Priority: Ollama (local) -> OpenAI -> Anthropic

    Auto-detection checks the *model*, not just the service, so an Ollama
    server that is running without the configured model falls through to a
    cloud key instead of being selected and then failing at call time.

    Args:
        provider: Specific provider to use, or None for auto-detection
        model: Specific model to use, or None for provider default

    Returns:
        LLM provider instance

    Raises:
        LLMUnavailableError: If the selected provider cannot serve the
            requested model -- unreachable, missing model, or missing key.
    """
    config = get_config()

    # Use config if not specified
    if provider is None:
        provider = config.llm.provider

    # Get model from config if not specified. "auto" is the sentinel for "the
    # provider's own default" and resolves to None, which a cloud provider now
    # reports as a missing setting rather than papering over with a hard-coded
    # name.
    if model is None or model == "auto":
        model = config.llm.model if config.llm.model != "auto" else None

    temperature = config.llm.temperature
    # Graphiti runs a second, cheaper model for summarization and edge dedup.
    # Unset means "the same as the main model", resolved per provider below --
    # never a name compiled in here.
    small_model = config.llm.small_model

    if provider == LLMProvider.AUTO:
        candidates = (LLMProvider.OLLAMA, LLMProvider.OPENAI, LLMProvider.ANTHROPIC)
        problems: list[str] = []
        ollama_models: list[str] = []
        for candidate in candidates:
            problem, models = _provider_problem(candidate, model)
            if problem is None:
                provider = candidate
                break
            problems.append(problem)
            ollama_models = ollama_models or models
        else:
            raise _unavailable_error(problems, model, ollama_models)
    else:
        problem, ollama_models = _provider_problem(provider, model)
        if problem is not None:
            raise _unavailable_error([problem], model, ollama_models)

    # Create provider instance. The cloud branches pass `model` through
    # unchanged, None included: the constructors are the single place that
    # decides what a missing model means, so a fallback here could not disagree
    # with the preflight above.
    if provider == LLMProvider.OLLAMA:
        return OllamaLLM(
            model=model or OLLAMA_DEFAULT_MODEL,
            temperature=temperature,
            small_model=small_model,
        )
    elif provider == LLMProvider.OPENAI:
        return OpenAILLM(
            model=model,
            temperature=temperature,
            small_model=small_model,
        )
    elif provider == LLMProvider.ANTHROPIC:
        return AnthropicLLM(
            model=model,
            temperature=temperature,
            small_model=small_model,
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
