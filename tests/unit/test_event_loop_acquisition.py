"""How each entry point gets an event loop, and what it must not do to get one.

Three sync entry points used to reach for ``asyncio.get_event_loop()`` and
juggled the three cases it can be in -- no loop, a stopped loop, a running loop
-- by hand. That lookup is wrong in both directions once anything in the process
has called ``asyncio.run()``: it raises ``RuntimeError`` when no loop is current,
and inside a running loop it hands back a loop that cannot be driven with
``run_until_complete``. Neither failure reads as a loop problem from the caller's
side, so each site grew its own workaround.

The ruled design, applied to every site: one async implementation, awaited by
async callers, and a sync wrapper that does ``asyncio.run(...)`` for callers with
no loop. The sync wrapper is *not* a bridge -- it does not spawn a thread to run
the coroutine while a loop is already running, because that hides the caller's
mistake behind a thread. It raises, which is the signal to await instead.

The tests come in both contexts, because the two fail differently: after a prior
``asyncio.run()`` there is no current loop at all, and inside a running loop
there is one that cannot be driven.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from typing import Any

import pytest

from docugraph.agents.tools.graph_query import query_entities_sync
from docugraph.core.llm import LLMUnavailableError
from docugraph.core.models import Chunk, SearchResult
from docugraph.interfaces.mcp_server import _search_docs
from docugraph.retrieval.reranker import LLMReranker
from docugraph.storage.graph_store import GraphStore

GRAPH_UNAVAILABLE_REASON = "No LLM can serve the requested model"

# A refused sync wrapper is a coroutine object that nobody awaited -- it is
# created as asyncio.run's argument and released when the call is refused. The
# warning is how that refusal surfaces, not a separate failure.
refusal_noise = pytest.mark.filterwarnings("ignore::RuntimeWarning")


async def _a_prior_async_call() -> None:
    """Stands in for any earlier async caller in the process.

    ``asyncio.run`` is what such a caller leaves behind: after it returns, there
    is no current event loop. Nothing about this coroutine matters; the running
    of it does.
    """


def _result(content: str, score: float = 0.5) -> SearchResult:
    return SearchResult(chunk=Chunk(document_id="d", content=content), score=score)


# --------------------------------------------------------------------------
# GraphStore: the one site reachable from both directions with no fallback
# --------------------------------------------------------------------------


class _FakeGraphiti:
    """Counts closes; close() is the only thing GraphStore's exit needs."""

    def __init__(self) -> None:
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1


class _StubGraphStore(GraphStore):
    """A GraphStore that is already initialized, with a counted fake inside."""

    def __init__(self, db_path: Path | str | None, graphiti: _FakeGraphiti) -> None:
        super().__init__(db_path=db_path)
        self._graphiti = graphiti
        self._initialized = True


class TestGraphStoreExit:
    def test_the_sync_context_manager_closes_after_a_loop_ran(self, tmp_path: Path) -> None:
        """The CLI case. ``__exit__`` had no fallback, so this used to raise.

        ``get_event_loop()`` finds no current loop once anything has called
        ``asyncio.run()``, so ``with GraphStore(...)`` failed at exit in any
        process that had already run something async -- with a message about
        event loops, naming nothing about the store.
        """
        asyncio.run(_a_prior_async_call())
        graphiti = _FakeGraphiti()

        with _StubGraphStore(tmp_path / "graph", graphiti):
            pass

        assert graphiti.closed == 1

    async def test_the_async_context_manager_closes_inside_a_running_loop(
        self, tmp_path: Path
    ) -> None:
        """The in-loop form, which is what an async caller uses instead."""
        graphiti = _FakeGraphiti()

        async with _StubGraphStore(tmp_path / "graph", graphiti):
            pass

        assert graphiti.closed == 1

    @refusal_noise
    async def test_the_sync_exit_refuses_to_close_from_inside_a_loop(self, tmp_path: Path) -> None:
        """The contract that makes the previous test the right way to do this.

        Driving a second loop from inside a running one is not possible, and the
        sync exit does not pretend otherwise by handing the work to a thread. It
        raises, and the store is left open rather than half-closed.
        """
        graphiti = _FakeGraphiti()
        store = _StubGraphStore(tmp_path / "graph", graphiti)

        with pytest.raises(RuntimeError, match="running event loop"):
            store.__exit__(None, None, None)

        assert graphiti.closed == 0


# --------------------------------------------------------------------------
# query_entities_sync: the wrapper must not run the query behind a thread
# --------------------------------------------------------------------------


class _FailingGraphStore:
    """A graph store whose search fails the way an unconfigured LLM does."""

    async def search(self, **_kwargs: Any) -> list[dict[str, Any]]:
        raise LLMUnavailableError(GRAPH_UNAVAILABLE_REASON)

    async def close(self) -> None:
        pass


class TestQueryEntitiesSync:
    def test_it_reports_the_real_failure_after_a_loop_ran(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A contract test, not the guard against the old pattern.

        The old wrapper passed this too: its bare ``except RuntimeError`` caught
        the failed loop lookup and re-ran the query in a fresh loop, so the
        caller still got the real error. What it cost was a *second* execution
        of whatever raised -- here harmless, in a writing tool not. The test
        below is the one that tells the two apart.
        """
        monkeypatch.setattr(
            "docugraph.storage.graph_store.GraphStore", lambda *_a, **_k: _FailingGraphStore()
        )
        asyncio.run(_a_prior_async_call())

        with pytest.raises(LLMUnavailableError) as excinfo:
            query_entities_sync("zorblat")

        assert GRAPH_UNAVAILABLE_REASON in str(excinfo.value)

    @refusal_noise
    async def test_the_sync_wrapper_refuses_to_run_on_a_side_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inside a loop the wrapper raises instead of quietly forking.

        The old wrapper handed the coroutine to a ThreadPoolExecutor here, which
        worked and *was* the bug: the caller learned nothing about having called a
        sync entry point from async code, and the query ran on a thread nobody was
        waiting for.
        """
        opened: list[str] = []

        def _record(*_args: Any, **_kwargs: Any) -> _FailingGraphStore:
            opened.append("constructed")
            return _FailingGraphStore()

        monkeypatch.setattr("docugraph.storage.graph_store.GraphStore", _record)

        with pytest.raises(RuntimeError):
            query_entities_sync("zorblat")

        assert opened == [], "the coroutine must not have been reached at all"

    def test_the_query_runs_once_even_when_it_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The old wrapper could run the query twice.

        With a current-but-idle loop in place, the old wrapper drove it with
        ``run_until_complete``. A ``RuntimeError`` raised *inside* the query then
        looked exactly like a failed loop lookup, so the bare ``except
        RuntimeError`` caught it and ran the whole query again in a fresh loop.
        ``LLMUnavailableError`` subclasses ``RuntimeError`` (``llm.py:439``), as
        does the store's own "no API key" error, so this was reachable in
        ordinary use -- and for a writing tool, "ran twice" is data loss.
        """
        attempts: list[str] = []

        class _CountingStore(_FailingGraphStore):
            async def search(self, **_kwargs: Any) -> list[dict[str, Any]]:
                attempts.append("search")
                return await super().search(**_kwargs)

        monkeypatch.setattr(
            "docugraph.storage.graph_store.GraphStore", lambda *_a, **_k: _CountingStore()
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with pytest.raises(LLMUnavailableError):
                query_entities_sync("zorblat")
        finally:
            asyncio.set_event_loop(None)
            loop.close()

        assert attempts == ["search"], "the query must not be run a second time"


# --------------------------------------------------------------------------
# LLMReranker: the same shape, plus the short-circuit that needs no loop
# --------------------------------------------------------------------------


class _FakeLLM:
    """Scores every document 7/10, and counts the calls it was asked for."""

    BATCH_SIZE = 5  # LLMReranker's default: one generate() per batch of five

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, **_kwargs: Any) -> str:
        self.calls += 1
        return "\n".join(["7"] * self.BATCH_SIZE)


def _use_llm(monkeypatch: pytest.MonkeyPatch) -> _FakeLLM:
    llm = _FakeLLM()
    monkeypatch.setattr("docugraph.core.llm.get_llm_provider", lambda: llm)
    return llm


class TestLLMReranker:
    async def test_it_can_be_awaited_inside_a_running_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The capability the sync-only API did not have."""
        llm = _use_llm(monkeypatch)

        reranked = await LLMReranker().arerank("q", [_result("a", 0.1), _result("b", 0.9)])

        assert llm.calls == 1, "one scoring call covers the batch"
        assert reranked[0].chunk.content == "b", "0.9 original beats 0.1 at equal LLM score"

    def test_the_sync_rerank_works_after_a_loop_ran(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = _use_llm(monkeypatch)
        asyncio.run(_a_prior_async_call())

        reranked = LLMReranker().rerank("q", [_result("a", 0.1)])

        assert llm.calls == 1
        assert len(reranked) == 1

    @refusal_noise
    async def test_the_sync_rerank_refuses_inside_a_running_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm = _use_llm(monkeypatch)

        with pytest.raises(RuntimeError, match="running event loop"):
            LLMReranker().rerank("q", [_result("a", 0.1)])

        assert llm.calls == 0

    def test_rerank_runs_once_even_when_the_model_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same double execution, through the reranker's own wrapper.

        The failure has to come from outside ``_score_batch``, which catches
        everything and falls back to neutral scores; resolving the provider is
        the step that raises, and it does so exactly once per rerank.
        """
        attempts: list[str] = []

        def _no_llm() -> Any:
            attempts.append("get_llm_provider")
            raise LLMUnavailableError(GRAPH_UNAVAILABLE_REASON)

        monkeypatch.setattr("docugraph.core.llm.get_llm_provider", _no_llm)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with pytest.raises(LLMUnavailableError):
                LLMReranker().rerank("q", [_result("a", 0.1)])
        finally:
            asyncio.set_event_loop(None)
            loop.close()

        assert attempts == ["get_llm_provider"], "the work must not be done a second time"

    async def test_an_empty_result_set_reranks_without_a_loop_or_a_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Why the short-circuit sits in the wrapper, not in the coroutine.

        Returning an empty list is not worth driving a loop for, so it happens
        before ``asyncio.run`` -- which is what lets it work inside a running
        loop, where the wrapper would otherwise refuse.
        """

        def _no_llm() -> Any:
            raise AssertionError("no model should be resolved for an empty result set")

        monkeypatch.setattr("docugraph.core.llm.get_llm_provider", _no_llm)

        assert LLMReranker().rerank("q", []) == []
        assert await LLMReranker().arerank("q", []) == []


# --------------------------------------------------------------------------
# The MCP server: a blocking tool must not stall every other request
# --------------------------------------------------------------------------


class _SlowVectorStore:
    """A vector store that takes its time, the way a cold index does."""

    def __init__(self, delay: float) -> None:
        self._delay = delay

    def search(self, _query: str, top_k: int) -> list[SearchResult]:
        time.sleep(self._delay)
        return [_result("The zorblat token calibrates the flux capacitor.")][:top_k]


class TestSearchDocsDoesNotBlock:
    async def test_a_slow_search_leaves_the_loop_free_to_work(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The server is one process: a blocking call freezes all of it.

        Nothing in the logs would say why, because the request that blocks logs
        nothing while it is blocking.
        """
        monkeypatch.setattr(
            "docugraph.interfaces.mcp_server.get_vector_store",
            lambda: _SlowVectorStore(delay=0.2),
        )

        ticks = 0

        async def _ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        ticker = asyncio.create_task(_ticker())
        try:
            result = await _search_docs({"query": "zorblat", "top_k": 1})
        finally:
            ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ticker

        text = "\n".join(item.text for item in result.content)
        assert "flux capacitor" in text, "the search itself must have worked"
        # 0.2s of blocking against a 10 ms ticker: a search that occupied the loop
        # yields zero ticks, one that hands off to a thread yields ~20. Five is
        # far from both, so the threshold is not a timing race.
        assert ticks >= 5, "the search occupied the event loop instead of yielding it"
