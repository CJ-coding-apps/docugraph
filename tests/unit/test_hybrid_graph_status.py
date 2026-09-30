"""A hybrid search must say when its graph leg did not run.

The defect these guard against: ``HybridRetriever.search()`` discarded every
graph failure -- the store could not be opened, could not be queried, or the
search could not be run -- in three separate ``except: pass`` blocks. A caller
that asked for graph results got vector and keyword results back with nothing
to say the graph half had been dropped, which is indistinguishable from a graph
that simply had nothing to say.

Ruled behaviour: ``mode="all"`` with ``include_graph=true`` and the graph
unavailable must return ``graph_available: false`` plus a reason.
"""

from docugraph.core.llm import LLMUnavailableError
from docugraph.core.models import Chunk, SearchResult
from docugraph.interfaces.mcp_server import _hybrid_search
from docugraph.retrieval.hybrid_search import (
    HybridRetriever,
    HybridSearchConfig,
    SearchMode,
)

GRAPH_UNAVAILABLE_REASON = "No LLM can serve the requested model"


class _FakeVectorStore:
    """Enough of VectorStore for fusion and result building to produce a hit."""

    def __init__(self, chunk: Chunk | None) -> None:
        self._chunk = chunk

    def search(self, _query, top_k, filter_expr=None):
        assert filter_expr is None, "no filters were supplied, so none are built"
        return self._hits(0.5)[:top_k]

    def search_hybrid(self, _query, top_k, filter_expr=None):
        assert filter_expr is None, "no filters were supplied, so none are built"
        return self._hits(0.4)[:top_k]

    def _hits(self, score: float) -> list[SearchResult]:
        if self._chunk is None:
            return []
        return [SearchResult(chunk=self._chunk, score=score)]

    def get_chunk(self, chunk_id):
        return self._chunk if self._chunk is not None and chunk_id == self._chunk.id else None


class _FakeGraphStore:
    """Stands in for GraphStore; ``search`` is async, as the real one is."""

    def __init__(self, facts: list[dict]) -> None:
        self._facts = facts

    async def search(self, _query, num_results):
        return self._facts[:num_results]


def _chunk() -> Chunk:
    return Chunk(
        document_id="doc-1",
        content="The zorblat token calibrates the flux capacitor.",
        metadata={"title": "Calibration Guide", "source_path": "guide.md"},
    )


def _graph_store_fails(monkeypatch) -> None:
    """Make opening the graph store raise, the way a missing LLM does."""

    def _raise(*_args, **_kwargs):
        raise LLMUnavailableError(GRAPH_UNAVAILABLE_REASON)

    monkeypatch.setattr("docugraph.storage.graph_store.GraphStore", _raise)


def _retriever(chunk: Chunk | None, *, include_graph: bool) -> HybridRetriever:
    return HybridRetriever(
        vector_store=_FakeVectorStore(chunk),
        config=HybridSearchConfig(include_graph=include_graph),
    )


class TestGraphStatusOnTheRetriever:
    async def test_mode_all_reports_the_graph_did_not_run(self, monkeypatch):
        """The ruled case: asked for the graph, got none, and are told why.

        Async because the graph leg is reached by a running loop -- the MCP
        server's situation. Called with no loop at all, ``search()`` fails
        earlier, in its own loop lookup, and reports that instead.
        """
        _graph_store_fails(monkeypatch)
        retriever = _retriever(_chunk(), include_graph=True)

        results = retriever.search("zorblat", top_k=5, mode=SearchMode.ALL)

        assert retriever.graph_status.requested is True
        assert retriever.graph_status.available is False
        assert GRAPH_UNAVAILABLE_REASON in (retriever.graph_status.reason or "")
        assert results, "a failed graph leg must not remove the vector/keyword results"

    async def test_available_graph_is_reported_as_available(self, monkeypatch):
        """The other direction, so `available` cannot pass as a constant."""
        monkeypatch.setattr(
            "docugraph.storage.graph_store.GraphStore",
            lambda *_a, **_k: _FakeGraphStore([{"uuid": "fact-1"}]),
        )
        retriever = _retriever(_chunk(), include_graph=True)

        retriever.search("zorblat", top_k=5, mode=SearchMode.ALL)

        assert retriever.graph_status.requested is True
        assert retriever.graph_status.available is True
        assert retriever.graph_status.reason is None

    def test_not_requested_reports_nothing(self, monkeypatch):
        """Default hybrid mode never asks for the graph, so there is no note."""
        _graph_store_fails(monkeypatch)
        retriever = _retriever(_chunk(), include_graph=True)

        retriever.search("zorblat", top_k=5, mode=SearchMode.HYBRID)

        assert retriever.graph_status.requested is False
        assert retriever.graph_status.reason is None

    def test_requested_but_disabled_by_config_says_so(self, monkeypatch):
        """`mode="all"` without `include_graph` is a request that was refused."""
        _graph_store_fails(monkeypatch)
        retriever = _retriever(_chunk(), include_graph=False)

        retriever.search("zorblat", top_k=5, mode=SearchMode.ALL)

        assert retriever.graph_status.requested is True
        assert retriever.graph_status.available is False
        assert "include_graph" in (retriever.graph_status.reason or "")


class TestGraphStatusOnTheTool:
    async def test_tool_reports_graph_unavailable_with_a_reason(self, monkeypatch):
        _graph_store_fails(monkeypatch)
        chunk = _chunk()
        monkeypatch.setattr(
            "docugraph.retrieval.hybrid_search.HybridRetriever",
            lambda *, config: HybridRetriever(vector_store=_FakeVectorStore(chunk), config=config),
        )

        result = await _hybrid_search({"query": "zorblat", "mode": "all", "include_graph": True})
        text = "\n".join(item.text for item in result.content)

        assert getattr(result, "isError", False) is False, "degrading is not an error"
        assert "graph_available: false" in text
        assert GRAPH_UNAVAILABLE_REASON in text
        assert "Calibration Guide" in text, "the results themselves must still be returned"

    async def test_tool_carries_the_note_when_there_are_no_results(self, monkeypatch):
        """The most invisible path: an empty answer that hid a dropped graph."""
        _graph_store_fails(monkeypatch)
        monkeypatch.setattr(
            "docugraph.retrieval.hybrid_search.HybridRetriever",
            lambda *, config: HybridRetriever(vector_store=_FakeVectorStore(None), config=config),
        )

        result = await _hybrid_search({"query": "zorblat", "mode": "all", "include_graph": True})
        text = "\n".join(item.text for item in result.content)

        assert "No results found." in text
        assert "graph_available: false" in text

    async def test_tool_stays_quiet_when_the_graph_was_not_requested(self, monkeypatch):
        _graph_store_fails(monkeypatch)
        chunk = _chunk()
        monkeypatch.setattr(
            "docugraph.retrieval.hybrid_search.HybridRetriever",
            lambda *, config: HybridRetriever(vector_store=_FakeVectorStore(chunk), config=config),
        )

        result = await _hybrid_search({"query": "zorblat", "mode": "hybrid"})
        text = "\n".join(item.text for item in result.content)

        # Without these two, the test would also pass on a hard error, whose
        # text happens not to contain "graph_available" either.
        assert getattr(result, "isError", False) is False
        assert "Calibration Guide" in text, "the search itself must have worked"
        assert "graph_available" not in text
