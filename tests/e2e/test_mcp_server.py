"""End-to-end tests for the MCP stdio server.

Spawns the real `docugraph-mcp` server over stdio via the mcp client SDK and
exercises it exactly as Claude Code would. Uses a temp data dir so no state
leaks; the default fastembed model downloads once (cached), so these tests
touch the network on a cold cache.
"""

import os
import sys

import pytest

# The server initializes a VectorStore/embedder on tool calls; skip cleanly if
# the mcp client SDK isn't importable for some reason.
mcp = pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

EXPECTED_TOOLS = {
    "search_docs",
    "hybrid_search",
    "crawl_url",
    "index_git",
    "memory_store",
    "memory_recall",
    "graph_query",
    "graph_add",
    "get_stats",
}


def _text(result) -> str:
    parts = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)


@pytest.fixture
def server_params(tmp_path):
    """StdioServerParameters launching docugraph-mcp against a temp data dir."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "DOCUGRAPH_STORAGE__DATA_DIR": str(tmp_path / "data"),
        # Reuse the developer/CI fastembed cache if present to avoid re-download.
        "FASTEMBED_CACHE_PATH": os.environ.get(
            "FASTEMBED_CACHE_PATH", os.path.expanduser("~/.cache/fastembed")
        ),
    }
    # Prefer the console script; fall back to `python -m` if not on PATH.
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "docugraph.interfaces.mcp_server"],
        env=env,
    )


@pytest.mark.asyncio
async def test_lists_exactly_nine_tools(server_params):
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = (await session.list_tools()).tools
        names = {t.name for t in tools}

    assert names == EXPECTED_TOOLS
    assert len(tools) == 9


@pytest.mark.asyncio
async def test_hybrid_search_schema_has_rerank_and_include_graph(server_params):
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = {t.name: t for t in (await session.list_tools()).tools}
        props = tools["hybrid_search"].inputSchema["properties"]

    assert "rerank" in props
    assert "include_graph" in props


@pytest.mark.asyncio
async def test_memory_round_trip(server_params):
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        await session.call_tool("memory_store", {"key": "phase9.key", "value": "round-trip-ok"})
        recalled = await session.call_tool("memory_recall", {"key": "phase9.key"})

    assert "round-trip-ok" in _text(recalled)


@pytest.mark.asyncio
async def test_get_stats_reports_fastembed(server_params):
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        stats = await session.call_tool("get_stats", {})

    text = _text(stats)
    assert "bge-small-en-v1.5" in text
    assert "total_chunks" in text


@pytest.mark.asyncio
async def test_graph_query_degrades_gracefully_without_llm(server_params):
    """Without an LLM configured, graph_query must not crash the session.

    It should either return guidance (isError) or an empty/valid result if an
    LLM happens to be reachable on the test host — never take down the server.
    """
    # Ensure no cloud LLM key leaks in from the environment for this check.
    server_params.env.pop("OPENAI_API_KEY", None)
    server_params.env.pop("ANTHROPIC_API_KEY", None)

    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        result = await session.call_tool("graph_query", {"query": "what depends on x?"})
        # Session still alive afterwards: a follow-up call must succeed.
        stats = await session.call_tool("get_stats", {})

    text = _text(result)
    if getattr(result, "isError", False):
        # Degraded path: message should guide the user toward an LLM.
        assert any(kw in text for kw in ("LLM", "Ollama", "OPENAI_API_KEY"))
    assert "total_chunks" in _text(stats)
