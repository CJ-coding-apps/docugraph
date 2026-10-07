"""What the cloud providers actually put on the wire.

A signature the type checker reads and a signature the SDK enforces at runtime
are two different claims, and this module exists because they disagreed.
`anthropic>=0.18.0` then had no upper bound, so pip installed whatever was
current -- and in the 1.x line `Messages.create` no longer takes `temperature`.
Passing it raised `TypeError: AsyncMessages.create() got an unexpected keyword
argument 'temperature'` before a request was ever built, so every Anthropic call
failed at runtime while `mypy` under CI's `--extra dev` sync saw nothing at all,
because anthropic was not installed and the import was treated as missing.

The `cloud` extra is now capped below each SDK's next major, so that particular
break cannot arrive as an automatic install again. The cap is a decision, not a
guarantee: it is this file that says what the pinned version actually accepts,
and it is the reason the cap is 2 rather than 3.

So each test asks the installed SDK a question over a mocked HTTP transport. The
client is the genuine article -- only its transport is replaced -- so the SDK
still validates every argument the provider passes, and the body the SDK built is
what gets asserted. That is the only place the two providers' request shapes can
be checked: a double for the client would agree with whatever the provider does.

The shapes themselves are not cosmetic. OpenAI has two names for the output-token
limit and they are not interchangeable -- `max_tokens` is rejected outright by the
o-series and the GPT-5 family -- and the same reasoning models reject any
temperature other than their own, so a `0.0` nobody chose fails the call. Both
providers are therefore tested twice over: once with nothing configured, and once
with a temperature set.

Anthropic and OpenAI are imported per test rather than at module level. An
`importorskip` at the top would let an absent anthropic skip the OpenAI tests
too, which is how a test suite goes quiet -- on a dev-only install that is
exactly what happened here.
"""

from __future__ import annotations

import importlib
import inspect
import json
from typing import Any

import pytest

from docugraph.core.llm import AnthropicLLM, OpenAILLM

# Model names appear in this file as fixtures, which the requirement allows:
# this is the one place that has to name a model to prove the package itself no
# longer does. `o3-mini` stands for the family that rejects `max_tokens` and any
# temperature it did not choose; `gpt-4o` for the family that predates both rules.
REASONING_MODEL = "o3-mini"
STANDARD_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-3-haiku-20240307"

# Anthropic's own model of a reply. Small, but shaped like the real thing, because
# the provider reads the text blocks out of `content` and a looser double would
# hide a change in how those blocks are nested.
ANTHROPIC_REPLY = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": CLAUDE_MODEL,
    "content": [{"type": "text", "text": "the answer"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}

OPENAI_REPLY = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": STANDARD_MODEL,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "the answer"},
            "finish_reason": "stop",
        }
    ],
}


def _http_client_module(sdk_name: str) -> Any:
    """The httpx module the installed SDK will accept for a mock transport.

    anthropic 1.5 rejects an ``httpx.AsyncClient`` outright -- ``Invalid
    `http_client` argument; `httpx.AsyncClient` is from the `httpx` package, but
    this SDK uses `httpx2` `` -- so the double has to be built from whichever
    module this SDK names. Both expose the same MockTransport/AsyncClient
    surface, so only the import differs; openai wants the ordinary httpx.
    """
    names = ("httpx2", "httpx") if sdk_name == "anthropic" else ("httpx", "httpx2")
    for name in names:
        try:
            return importlib.import_module(name)
        except ImportError:  # pragma: no cover -- only if neither is installed
            continue
    pytest.skip("no httpx module available to build a mock transport")


def _anthropic() -> Any:
    """The anthropic module, skipping only the tests that need it."""
    return pytest.importorskip("anthropic")


def _install_mock_transport(
    sdk: str, monkeypatch: pytest.MonkeyPatch, reply: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Make the SDK build a real client that talks to nobody.

    Returns the list of request bodies the SDK serialised. Asserting on those
    rather than on the call arguments is the point: a keyword the SDK quietly
    drops, or one it adds on its own, is visible here and nowhere else.
    """
    module = _anthropic() if sdk == "anthropic" else pytest.importorskip("openai")
    httpx = _http_client_module(sdk)
    client_class = module.AsyncAnthropic if sdk == "anthropic" else module.AsyncOpenAI
    body = (ANTHROPIC_REPLY if sdk == "anthropic" else OPENAI_REPLY) if reply is None else reply
    sent: list[dict[str, Any]] = []

    def handler(request: Any) -> Any:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=body)

    # Captured before the patch: the factory below replaces the module attribute,
    # so a call through `module.AsyncOpenAI` inside it would find itself.
    real_client = client_class

    def factory(*, api_key: str, **_kwargs: Any) -> Any:
        return real_client(
            api_key=api_key,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

    monkeypatch.setattr(module, client_class.__name__, factory)
    return sent


class TestAnthropicCallShape:
    async def test_generate_returns_the_text_the_sdk_sent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The defect itself: this raised TypeError before the fix.

        Nothing here asserts *which* arguments the provider passes, on purpose.
        The SDK is the authority on that, and it says so by raising.
        """
        sent = _install_mock_transport("anthropic", monkeypatch)

        text = await _anthropic_provider().generate("what is the answer?")

        assert text == "the answer"
        assert len(sent) == 1, "expected exactly one request"
        assert sent[0]["messages"] == [{"role": "user", "content": "what is the answer?"}]

    async def test_the_call_carries_only_parameters_the_sdk_advertises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No parameter may reach the wire that the SDK does not declare.

        The runtime rejection covers the arguments the SDK validates, but there is
        a way around that which looks harmless: `extra_body` is passed through
        untouched, so a removed parameter routed through it would leave the local
        call succeeding and the request still going out with a field the API may
        no longer accept. This checks the wire, not the call.
        """
        sent = _install_mock_transport("anthropic", monkeypatch)

        await _anthropic_provider().generate("what is the answer?")

        assert sent, "the mock transport saw no request, so this check proves nothing"

        anthropic = _anthropic()
        client = anthropic.AsyncAnthropic(api_key="sk-ant-test-not-a-real-key")
        advertised = set(inspect.signature(client.messages.create).parameters)

        unexpected = sorted(set(sent[0]) - advertised)
        assert unexpected == [], (
            f"the request carries {unexpected}, which Messages.create does not declare; "
            "either the SDK has changed or the parameter is being smuggled past it"
        )


class TestAnthropicSendsOnlyWhatWasConfigured:
    """`max_tokens` always; `system` and `temperature` only on request."""

    async def test_max_tokens_is_always_sent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Anthropic has no equivalent of OpenAI's two names, and no default.

        Omitting it is a 400, so this is the one parameter that is never
        conditional.
        """
        sent = _install_mock_transport("anthropic", monkeypatch)

        await _anthropic_provider().generate("q", max_tokens=1234)

        assert sent[0]["max_tokens"] == 1234

    async def test_an_empty_system_prompt_is_omitted_not_sent_blank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`""` is a system prompt saying nothing, and it travels either way.

        The difference matters because the field is present in the request as
        soon as it is passed: a caller who wanted no system prompt would have
        sent an empty one.
        """
        sent = _install_mock_transport("anthropic", monkeypatch)

        await _anthropic_provider().generate("q", system_prompt="")
        assert "system" not in sent[0]

        await _anthropic_provider().generate("q", system_prompt="be terse")
        assert sent[1]["system"] == "be terse"

    async def test_a_configured_temperature_is_not_put_on_the_wire(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """This provider's `temperature` is inert, and that is the contract.

        The parameter stays in the shared signature because the OpenAI and Ollama
        providers honour it. Here the 1.x SDK removed it, so a value that was
        accepted into `LLMConfig` must still not reach the request -- and must not
        be smuggled through `extra_body` to reach it either.
        """
        sent = _install_mock_transport("anthropic", monkeypatch)

        await _anthropic_provider(temperature=0.3).generate("q", temperature=0.9)

        assert "temperature" not in sent[0]


class TestOpenAIRequestShape:
    """One token-limit name for both model generations, and no invented temperature.

    The two families disagree about both parameters, and the disagreement is
    silent in opposite directions: older models accept `max_tokens` and ignore
    `max_completion_tokens`, the reasoning models reject `max_tokens` and any
    temperature they did not choose. Sending the newer name and omitting an unset
    temperature is the intersection -- the shape both accept.
    """

    async def test_a_reasoning_model_is_asked_without_a_temperature(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The case that used to 400 for every user of an o-series model."""
        sent = _install_mock_transport("openai", monkeypatch)

        await _openai_provider(REASONING_MODEL).generate("q")

        assert "temperature" not in sent[0], (
            "a reasoning model rejects any temperature other than its own, so an "
            "unset one must not be sent at all"
        )

    async def test_a_standard_model_is_asked_the_same_way(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The standard case must not diverge: one shape, both generations."""
        sent = _install_mock_transport("openai", monkeypatch)

        await _openai_provider(STANDARD_MODEL).generate("q")

        assert "temperature" not in sent[0]

    async def test_a_configured_temperature_is_sent_for_either_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What is configured is sent, verbatim, including a deliberate 0.0.

        `temperature or self._temperature` used to lose a 0.0, since 0.0 is
        falsy -- so the one value most likely to be chosen deliberately was the
        one silently dropped. Whether a *particular* model accepts it is the
        user's question, not one this layer can answer.
        """
        sent = _install_mock_transport("openai", monkeypatch)

        await _openai_provider(STANDARD_MODEL, temperature=0.0).generate("q")
        assert sent[0]["temperature"] == 0.0

        await _openai_provider(STANDARD_MODEL).generate("q", temperature=0.7)
        assert sent[1]["temperature"] == 0.7

    @pytest.mark.parametrize("model", [REASONING_MODEL, STANDARD_MODEL])
    async def test_the_token_limit_uses_the_name_both_generations_accept(
        self, monkeypatch: pytest.MonkeyPatch, model: str
    ) -> None:
        """`max_completion_tokens`, never `max_tokens`.

        Not interchangeable, and not a preference: the reasoning models raise
        "Unsupported parameter" on `max_tokens`, while the older ones accept it
        and ignore the newer name. The newer name is the only one that works for
        both, which is why this is asserted rather than reasoned about.
        """
        sent = _install_mock_transport("openai", monkeypatch)

        await _openai_provider(model).generate("q", max_tokens=2048)

        assert sent[0]["max_completion_tokens"] == 2048
        assert "max_tokens" not in sent[0], (
            "the legacy name is rejected outright by the o-series and GPT-5 family"
        )

    async def test_an_empty_system_prompt_is_omitted_not_sent_blank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent = _install_mock_transport("openai", monkeypatch)

        await _openai_provider(STANDARD_MODEL).generate("q", system_prompt="")
        assert [m["role"] for m in sent[0]["messages"]] == ["user"]

        await _openai_provider(STANDARD_MODEL).generate("q", system_prompt="be terse")
        assert [m["role"] for m in sent[1]["messages"]] == ["system", "user"]


class TestAnthropicReplyShape:
    """Which content block the text is read from.

    A second defect at the same call site, found by the same change: switching
    CI's sync to `--all-extras` made mypy see anthropic's real block union and
    reject `response.content[0].text` eleven times over. The type error was the
    smaller half -- at runtime a reply that leads with a thinking or tool-use
    block made that line raise `AttributeError: 'ThinkingBlock' object has no
    attribute 'text'`, naming neither the provider nor the model.
    """

    async def test_text_is_read_from_the_text_block_not_the_first_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reply = {
            **ANTHROPIC_REPLY,
            "content": [
                {"type": "thinking", "thinking": "let me consider", "signature": "sig"},
                {"type": "text", "text": "the answer"},
            ],
        }
        _install_mock_transport("anthropic", monkeypatch, reply)

        assert await _anthropic_provider().generate("what is the answer?") == "the answer"

    async def test_a_reply_with_no_text_block_names_what_came_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Better an error that says what happened than an empty string.

        Returning "" here would read downstream as a successful, empty answer,
        which is the shape of failure this codebase keeps having to remove.
        """
        reply = {
            **ANTHROPIC_REPLY,
            "content": [
                {"type": "thinking", "thinking": "let me consider", "signature": "sig"},
            ],
        }
        _install_mock_transport("anthropic", monkeypatch, reply)

        with pytest.raises(RuntimeError) as excinfo:
            await _anthropic_provider().generate("what is the answer?")

        message = str(excinfo.value)
        assert "ThinkingBlock" in message, message
        assert CLAUDE_MODEL in message, message


def _anthropic_provider(**kwargs: Any) -> AnthropicLLM:
    """A provider built the way production builds one: with a model named."""
    return AnthropicLLM(model=CLAUDE_MODEL, api_key="sk-ant-test-not-a-real-key", **kwargs)


def _openai_provider(model: str = STANDARD_MODEL, **kwargs: Any) -> OpenAILLM:
    return OpenAILLM(model=model, api_key="sk-test-not-a-real-key", **kwargs)
