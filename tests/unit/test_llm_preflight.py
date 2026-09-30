"""The graph tools check for a usable LLM up front, and say what to do.

The defect these guard against: with Ollama running but the configured model not
pulled, every reachability check passed, the graph store opened its Kùzu
database, and the failure surfaced from deep inside Graphiti as a
provider-specific 404. The CLI printed that traceback and still exited 0; the MCP
server returned `Error adding to graph: Error code: 404 - model 'llama3.2' not
found`. Neither told the user which command would fix it.

The fix is a single check in `get_llm_provider`, which both the CLI and the MCP
server reach through `get_graphiti_llm_client`. These tests pin the two setups
that were measured by hand: Ollama up without the model, and no key at all.
"""

import pytest
from click.testing import CliRunner

import docugraph.core.llm as llm
from docugraph.core.config import LLMProvider, reset_config
from docugraph.core.llm import (
    OLLAMA_DEFAULT_BASE_URL,
    OLLAMA_DEFAULT_MODEL,
    LLMUnavailableError,
    get_llm_provider,
)
from docugraph.interfaces.cli import EXIT_LLM_UNAVAILABLE
from docugraph.interfaces.cli import main as cli

# A model that is installed but is not the one docugraph asks for.
OTHER_MODEL = "glm-5.1:cloud"


@pytest.fixture(autouse=True)
def _fresh_config():
    """Rebuild the cached config around every test in this module."""
    reset_config()
    yield
    reset_config()


def _no_api_keys(monkeypatch) -> None:
    """Clear every key and provider override that could satisfy the preflight."""
    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DOCUGRAPH_LLM__PROVIDER",
        "DOCUGRAPH_LLM__MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


def _fake_ollama(monkeypatch, *, reachable: bool, models: list[str]) -> None:
    """Stand in for the two probes that talk to the Ollama server."""

    def _available(_base_url: str = "") -> bool:
        return reachable

    def _installed(_base_url: str = "") -> list[str]:
        return list(models)

    monkeypatch.setattr(llm, "is_ollama_available", _available)
    monkeypatch.setattr(llm, "get_available_ollama_models", _installed)


def _llm_goes_missing(monkeypatch) -> None:
    """Make the graph store fail its LLM preflight, as it would on a bare box."""
    import docugraph.storage.graph_store as graph_store

    def _no_llm(*_args, **_kwargs):
        raise LLMUnavailableError(
            "The knowledge-graph tools need an LLM, and none is available.\n"
            f"  - Ollama at {OLLAMA_DEFAULT_BASE_URL} is running but does not have the "
            f"model '{OLLAMA_DEFAULT_MODEL}'\n"
            f"    ollama pull {OLLAMA_DEFAULT_MODEL}\n"
            "    export OPENAI_API_KEY=sk-..."
        )

    monkeypatch.setattr(graph_store, "get_graphiti_llm_client", _no_llm)


class TestNoLlmIsAnActionableError:
    def test_ollama_down_and_no_keys(self, monkeypatch):
        _no_api_keys(monkeypatch)
        _fake_ollama(monkeypatch, reachable=False, models=[])

        with pytest.raises(LLMUnavailableError) as excinfo:
            get_llm_provider()

        message = str(excinfo.value)
        # Every candidate is named, with why it was rejected.
        assert OLLAMA_DEFAULT_BASE_URL in message
        assert "OPENAI_API_KEY is not set" in message
        assert "ANTHROPIC_API_KEY is not set" in message
        # ...and each has a remedy.
        assert f"ollama pull {OLLAMA_DEFAULT_MODEL}" in message
        assert "export OPENAI_API_KEY" in message
        assert "export ANTHROPIC_API_KEY" in message
        # The claim that search still works is part of the message, so it
        # cannot silently go stale without failing here.
        assert "docugraph search" in message

    def test_ollama_up_without_the_model_names_the_model(self, monkeypatch):
        """The measured setup: server reachable, model not pulled."""
        _no_api_keys(monkeypatch)
        _fake_ollama(monkeypatch, reachable=True, models=[OTHER_MODEL])

        with pytest.raises(LLMUnavailableError) as excinfo:
            get_llm_provider()

        message = str(excinfo.value)
        assert f"does not have the model '{OLLAMA_DEFAULT_MODEL}'" in message
        # It says what IS installed, and how to use it instead of pulling.
        assert OTHER_MODEL in message
        assert f"DOCUGRAPH_LLM__MODEL={OTHER_MODEL}" in message

    def test_ollama_up_with_no_models_at_all(self, monkeypatch):
        _no_api_keys(monkeypatch)
        _fake_ollama(monkeypatch, reachable=True, models=[])

        with pytest.raises(LLMUnavailableError) as excinfo:
            get_llm_provider()

        assert "no models installed" in str(excinfo.value)

    def test_ollama_down_and_no_alternative_hint(self, monkeypatch):
        """With nothing installed, no 'use a model you have' line is offered."""
        _no_api_keys(monkeypatch)
        _fake_ollama(monkeypatch, reachable=True, models=[])

        with pytest.raises(LLMUnavailableError) as excinfo:
            get_llm_provider()

        assert "DOCUGRAPH_LLM__MODEL" not in str(excinfo.value)

    def test_explicit_openai_without_a_key(self, monkeypatch):
        _no_api_keys(monkeypatch)

        with pytest.raises(LLMUnavailableError) as excinfo:
            get_llm_provider(provider=LLMProvider.OPENAI)

        assert "OPENAI_API_KEY is not set" in str(excinfo.value)

    def test_explicit_anthropic_without_a_key(self, monkeypatch):
        _no_api_keys(monkeypatch)

        with pytest.raises(LLMUnavailableError) as excinfo:
            get_llm_provider(provider=LLMProvider.ANTHROPIC)

        assert "ANTHROPIC_API_KEY is not set" in str(excinfo.value)

    def test_the_error_stays_a_runtime_error(self, monkeypatch):
        """The existing `except RuntimeError` handlers must keep catching it."""
        _no_api_keys(monkeypatch)
        _fake_ollama(monkeypatch, reachable=False, models=[])
        assert issubclass(LLMUnavailableError, RuntimeError)


class TestAutoDetectionPrefersAUsableProvider:
    def test_a_running_ollama_without_the_model_falls_through_to_a_key(self, monkeypatch):
        """`auto` means usable, not merely listening on the port."""
        _no_api_keys(monkeypatch)
        _fake_ollama(monkeypatch, reachable=True, models=[OTHER_MODEL])
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        provider = get_llm_provider()

        assert provider.provider_name == "openai"

    def test_a_running_ollama_with_the_model_is_used(self, monkeypatch):
        _no_api_keys(monkeypatch)
        # Ollama tags its models, so the pulled name carries a tag.
        _fake_ollama(monkeypatch, reachable=True, models=[f"{OLLAMA_DEFAULT_MODEL}:latest"])

        provider = get_llm_provider()

        assert provider.provider_name == "ollama"
        assert provider.model_name == OLLAMA_DEFAULT_MODEL


class TestModelNameMatching:
    def test_untagged_request_matches_a_tagged_model(self):
        assert llm._model_is_available("llama3.2", ["llama3.2:latest"])

    def test_tagged_request_requires_that_exact_tag(self):
        # Asking for :7b must not be satisfied by :latest -- Ollama would 404.
        assert not llm._model_is_available("llama3.2:7b", ["llama3.2:latest"])

    def test_an_unrelated_model_does_not_match(self):
        assert not llm._model_is_available("llama3.2", ["mistral:latest", "nomic-embed-text"])

    def test_exact_name_always_matches(self):
        assert llm._model_is_available("mistral:7b", ["mistral:7b"])


# --- The two surfaces a user actually meets -------------------------------


class TestCliReportsCleanly:
    def test_graph_add_exits_three_without_a_traceback(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DOCUGRAPH_STORAGE__DATA_DIR", str(tmp_path))
        _llm_goes_missing(monkeypatch)

        result = CliRunner().invoke(cli, ["graph", "add", "FastAPI uses Starlette."])

        assert result.exit_code == EXIT_LLM_UNAVAILABLE
        # A deliberate SystemExit, not an exception that happened to exit non-zero.
        assert isinstance(result.exception, SystemExit)
        assert "Traceback" not in result.output
        assert f"ollama pull {OLLAMA_DEFAULT_MODEL}" in result.output
        assert "OPENAI_API_KEY" in result.output

    def test_graph_search_exits_three_without_a_traceback(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DOCUGRAPH_STORAGE__DATA_DIR", str(tmp_path))
        _llm_goes_missing(monkeypatch)

        result = CliRunner().invoke(cli, ["graph", "search", "web frameworks"])

        assert result.exit_code == EXIT_LLM_UNAVAILABLE
        assert isinstance(result.exception, SystemExit)
        assert "Traceback" not in result.output


class TestMcpReturnsAnActionableError:
    async def test_graph_add_is_an_error_result_that_says_what_to_do(self, monkeypatch):
        _llm_goes_missing(monkeypatch)
        from docugraph.interfaces.mcp_server import _graph_add

        result = await _graph_add({"content": "FastAPI uses Starlette."})

        assert result.isError is True
        text = result.content[0].text
        assert f"ollama pull {OLLAMA_DEFAULT_MODEL}" in text
        assert "OPENAI_API_KEY" in text
        # Not the raw provider text this replaced.
        assert "404" not in text
        assert "Error code" not in text

    async def test_graph_query_is_an_error_result_that_says_what_to_do(self, monkeypatch):
        _llm_goes_missing(monkeypatch)
        from docugraph.interfaces.mcp_server import _graph_query

        result = await _graph_query({"query": "what is starlette"})

        assert result.isError is True
        assert f"ollama pull {OLLAMA_DEFAULT_MODEL}" in result.content[0].text
