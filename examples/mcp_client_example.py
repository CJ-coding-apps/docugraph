#!/usr/bin/env python3
"""Example: driving the DocuGraph AI MCP server as a real stdio client.

DocuGraph AI ships an MCP stdio server (`docugraph-mcp`). This example spawns
it as a subprocess and talks to it over JSON-RPC using the `mcp` client SDK —
the same protocol Claude Code and Cursor use. It doubles as a small live
smoke test: list the tools, store/recall a memory value, and fetch stats.

The server exposes nine tools:
    search_docs    - vector search over indexed documentation
    hybrid_search  - vector + keyword (+ optional graph) fusion; `rerank` opt-in
    crawl_url      - crawl and index a URL
    index_git      - clone and index a git repository
    memory_store   - store a value in repo/branch-scoped agent memory
    memory_recall  - retrieve a stored value
    graph_query    - query the documentation knowledge graph (needs an LLM)
    graph_add      - extract entities/relationships into the graph (needs an LLM)
    get_stats      - indexing statistics

Run:
    python examples/mcp_client_example.py

MCP client configuration (Claude Code / Cursor) — add to your MCP config:
    {
      "mcpServers": {
        "docugraph": { "command": "docugraph-mcp", "env": {} }
      }
    }

On macOS Intel, launch it via Docker instead:
    {
      "mcpServers": {
        "docugraph": {
          "command": "docker",
          "args": ["run", "-i", "--rm",
                   "-v", "docugraph-data:/data/docugraph",
                   "docugraph-ai-v1", "docugraph-mcp"]
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _text(result: object) -> str:
    """Extract concatenated text content from a CallToolResult."""
    parts: list[str] = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)


async def main() -> None:
    # Spawn the installed `docugraph-mcp` console script over stdio.
    # The SDK launches the child with a scrubbed environment, so pass through
    # the vars the server actually reads (data dir, model cache, API keys) —
    # otherwise DOCUGRAPH_* set in your shell won't reach the server.
    passthrough = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("DOCUGRAPH_")
        or key in {"FASTEMBED_CACHE_PATH", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
    }
    params = StdioServerParameters(command="docugraph-mcp", args=[], env=passthrough or None)

    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        # 1. Discover tools.
        tools = (await session.list_tools()).tools
        print(f"Server exposes {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}")

        # 2. Memory round-trip.
        print("\nStoring a memory value...")
        await session.call_tool(
            "memory_store",
            {"key": "example.note", "value": "docugraph mcp works"},
        )
        recalled = await session.call_tool("memory_recall", {"key": "example.note"})
        print(f"Recalled: {_text(recalled)}")

        # 3. Stats.
        stats = await session.call_tool("get_stats", {})
        print(f"\nStats:\n{_text(stats)}")

        # 4. Search (only meaningful once you've indexed something).
        #    hybrid_search accepts `rerank: true` to run the local
        #    cross-encoder (downloads ~1.8 GB on first use).
        hits = await session.call_tool(
            "hybrid_search",
            {"query": "how to configure embeddings", "top_k": 3},
        )
        print(f"\nSearch results:\n{_text(hits)}")


if __name__ == "__main__":
    # Some tool outputs are plain text rather than JSON — fine for a demo.
    with contextlib.suppress(json.JSONDecodeError):
        asyncio.run(main())
