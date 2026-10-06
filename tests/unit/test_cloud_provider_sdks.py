"""The cloud provider calls match what the installed SDK actually accepts.

A signature the type checker reads and a signature the SDK enforces at runtime
are two different claims, and this module exists because they disagreed.
`anthropic>=0.18.0` has no upper bound, so pip installs whatever is current --
and in the 1.x line `Messages.create` no longer takes `temperature`. Passing it
raised `TypeError: AsyncMessages.create() got an unexpected keyword argument
'temperature'` before a request was ever built, so every Anthropic call failed at
runtime while `mypy` under CI's `--extra dev` sync saw nothing at all, because
anthropic was not installed and the import was treated as missing.

So this asks the installed SDK two questions, over a mocked HTTP transport:
does the call the provider makes succeed, and does it carry anything the SDK
does not advertise? The second is what keeps the fix from being undone by the
plausible-looking alternative -- smuggling the parameter through `extra_body`,
which the SDK would not reject locally but which would then be sent to an API
that may no longer accept it either.

Both tests skip when anthropic is absent, which is why CI's type-check job syncs
with `--all-extras` and runs this file: on a dev-only install they would pass by
never running.
"""

from __future__ import annotations

import importlib
import inspect
import json
from typing import Any

import pytest

from docugraph.core.llm import AnthropicLLM

anthropic = pytest.importorskip("anthropic")

# The SDK's own model of a reply. Small, but shaped like the real thing, because
# the provider reads `response.content[0].text` and a looser double would hide a
# change in how content blocks are nested.
REPLY = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-haiku-20240307",
    "content": [{"type": "text", "text": "the answer"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


def _async_httpx() -> Any:
    """The httpx module the installed SDK will accept for a mock transport.

    anthropic 1.5 rejects an ``httpx.AsyncClient`` outright -- ``Invalid
    `http_client` argument; `httpx.AsyncClient` is from the `httpx` package, but
    this SDK uses `httpx2` `` -- so the double has to be built from whichever
    module this SDK names. Both expose the same MockTransport/AsyncClient
    surface, so only the import differs.
    """
    for name in ("httpx2", "httpx"):
        try:
            return importlib.import_module(name)
        except ImportError:  # pragma: no cover -- only if neither is installed
            continue
    pytest.skip("no httpx module available to build a mock transport")


def _install_mock_transport(
    monkeypatch: pytest.MonkeyPatch, reply: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Make ``AsyncAnthropic`` build a real client that talks to nobody.

    The client is the genuine article -- only its transport is replaced -- so the
    SDK still validates every argument the provider passes, which is the whole
    point. Returns the list of request bodies the SDK built.
    """
    httpx = _async_httpx()
    # Captured before the patch: the factory below replaces the module attribute,
    # so a call through `anthropic.AsyncAnthropic` inside it would find itself.
    real_client = anthropic.AsyncAnthropic
    body = REPLY if reply is None else reply
    sent: list[dict[str, Any]] = []

    def handler(request: Any) -> Any:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=body)

    def factory(*, api_key: str, **_kwargs: Any) -> Any:
        return real_client(
            api_key=api_key,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

    monkeypatch.setattr(anthropic, "AsyncAnthropic", factory)
    return sent


def _provider() -> AnthropicLLM:
    return AnthropicLLM(api_key="sk-ant-test-not-a-real-key")


class TestAnthropicCallShape:
    async def test_generate_returns_the_text_the_sdk_sent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The defect itself: this raised TypeError before the fix.

        Nothing here asserts *which* arguments the provider passes, on purpose.
        The SDK is the authority on that, and it says so by raising.
        """
        sent = _install_mock_transport(monkeypatch)

        text = await _provider().generate("what is the answer?")

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
        sent = _install_mock_transport(monkeypatch)

        await _provider().generate("what is the answer?")

        assert sent, "the mock transport saw no request, so this check proves nothing"

        client = anthropic.AsyncAnthropic(api_key="sk-ant-test-not-a-real-key")
        advertised = set(inspect.signature(client.messages.create).parameters)

        unexpected = sorted(set(sent[0]) - advertised)
        assert unexpected == [], (
            f"the request carries {unexpected}, which Messages.create does not declare; "
            "either the SDK has changed or the parameter is being smuggled past it"
        )


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
            **REPLY,
            "content": [
                {"type": "thinking", "thinking": "let me consider", "signature": "sig"},
                {"type": "text", "text": "the answer"},
            ],
        }
        _install_mock_transport(monkeypatch, reply)

        assert await _provider().generate("what is the answer?") == "the answer"

    async def test_a_reply_with_no_text_block_names_what_came_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Better an error that says what happened than an empty string.

        Returning "" here would read downstream as a successful, empty answer,
        which is the shape of failure this codebase keeps having to remove.
        """
        reply = {
            **REPLY,
            "content": [
                {"type": "thinking", "thinking": "let me consider", "signature": "sig"},
            ],
        }
        _install_mock_transport(monkeypatch, reply)

        with pytest.raises(RuntimeError) as excinfo:
            await _provider().generate("what is the answer?")

        message = str(excinfo.value)
        assert "ThinkingBlock" in message, message
        assert "claude-3-haiku-20240307" in message, message
