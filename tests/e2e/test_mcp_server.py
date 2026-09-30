"""End-to-end tests for the MCP stdio server.

Spawns the real `docugraph-mcp` server over stdio via the mcp client SDK and
exercises it exactly as Claude Code would. Uses a temp data dir so no state
leaks; the default fastembed model downloads once (cached), so these tests
touch the network on a cold cache.
"""

import json
import os
import sys
from pathlib import Path

import pytest

# The server initializes a VectorStore/embedder on tool calls; skip cleanly if
# the mcp client SDK isn't importable for some reason.
mcp = pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

# The frozen tool surface, captured from the real stdio wire. Regenerate
# deliberately (and say so in the commit) when the surface really changes — a
# rename or schema edit must not pass silently, which is why the comparison is
# exact rather than a membership test.
SNAPSHOT_PATH = Path(__file__).parent / "tools_snapshot.json"


def _tool_surface(tool) -> dict:
    """The frozen part of one tool: what a client sees and classifies on.

    Descriptions are deliberately excluded — the published fingerprint is names,
    schemas and annotations. Extend this deliberately, not by accident.
    """
    annotations = tool.annotations
    return {
        "name": tool.name,
        "inputSchema": tool.inputSchema,
        "annotations": (
            None
            if annotations is None
            else {
                "readOnlyHint": annotations.readOnlyHint,
                "destructiveHint": annotations.destructiveHint,
            }
        ),
    }


def _live_surface(tools) -> list:
    """The advertised surface, in the same order/shape as the snapshot."""
    return sorted((_tool_surface(t) for t in tools), key=lambda t: t["name"])


def _saved_surface() -> list:
    return json.loads(SNAPSHOT_PATH.read_text())


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
async def test_tool_surface_matches_frozen_snapshot(server_params):
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        live = _live_surface((await session.list_tools()).tools)

    saved = _saved_surface()
    assert [t["name"] for t in live] == [t["name"] for t in saved], (
        f"The set of advertised tools changed. If that is intended, re-freeze {SNAPSHOT_PATH.name}."
    )
    for live_tool, saved_tool in zip(live, saved, strict=True):
        assert live_tool == saved_tool, (
            f"Tool {live_tool['name']!r} no longer matches {SNAPSHOT_PATH.name}. "
            "If the change is intended, re-freeze the snapshot."
        )


@pytest.mark.asyncio
async def test_every_tool_is_annotated(server_params):
    """A tool with no approval hints would reach a client unclassified."""
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = (await session.list_tools()).tools

    for tool in tools:
        assert tool.annotations is not None, f"Tool {tool.name!r} has no annotations"
        assert tool.annotations.readOnlyHint is not None, (
            f"Tool {tool.name!r} does not declare readOnlyHint"
        )


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
