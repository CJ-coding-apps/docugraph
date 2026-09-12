#!/usr/bin/env python3
"""Example: Using DocuGraph AI as an MCP Server.

This example shows how to configure DocuGraph AI as an MCP server
for use with Claude Code, Cursor, or other MCP-compatible clients.

The MCP server provides these tools:
- search_docs: Search indexed documentation
- hybrid_search: Combined vector + keyword search
- crawl_url: Crawl and index a URL
- index_git: Clone and index a git repository
- memory_store: Store information for later
- memory_recall: Retrieve stored information
- graph_query: Query the knowledge graph
- graph_add: Add content to the knowledge graph
- get_stats: Get indexing statistics

Usage:
    # Start the MCP server
    docugraph mcp-server

    # Or run directly
    python -m docugraph.interfaces.mcp_server
"""

import json

# Example Claude Code MCP configuration
CLAUDE_CODE_CONFIG = {
    "mcpServers": {
        "docugraph": {
            "command": "docugraph",
            "args": ["mcp-server"],
            "env": {
                # Optional: customize data directory
                # "DOCUGRAPH_STORAGE__DATA_DIR": "/path/to/data",
                # Optional: use OpenAI for graph features
                # "OPENAI_API_KEY": "your-key-here",
            },
        }
    }
}

# Example Cursor MCP configuration
CURSOR_CONFIG = {
    "mcpServers": {
        "docugraph": {
            "command": "docugraph",
            "args": ["mcp-server"],
        }
    }
}

# Docker-based configuration (for macOS Intel)
DOCKER_CONFIG = {
    "mcpServers": {
        "docugraph": {
            "command": "docker",
            "args": [
                "compose",
                "-f",
                "/path/to/docugraph-ai/docker-compose.yml",
                "run",
                "--rm",
                "cli",
                "mcp-server",
            ],
        }
    }
}


def print_config():
    """Print example configurations."""
    print("DocuGraph AI - MCP Server Configuration Examples")
    print("=" * 60)

    print("\n1. Claude Code Configuration")
    print("   Add to ~/.claude/claude_desktop_config.json:")
    print("-" * 60)
    print(json.dumps(CLAUDE_CODE_CONFIG, indent=2))

    print("\n\n2. Cursor Configuration")
    print("   Add to cursor settings:")
    print("-" * 60)
    print(json.dumps(CURSOR_CONFIG, indent=2))

    print("\n\n3. Docker-based Configuration (macOS Intel)")
    print("   For systems where LanceDB isn't available natively:")
    print("-" * 60)
    print(json.dumps(DOCKER_CONFIG, indent=2))

    print("\n\nAvailable MCP Tools:")
    print("-" * 60)
    tools = [
        ("search_docs", "Search indexed documentation", "query, top_k"),
        ("hybrid_search", "Combined vector + keyword search", "query, mode, include_graph"),
        ("crawl_url", "Crawl and index a URL", "url, max_pages"),
        ("index_git", "Clone and index a git repository", "url, ref, include_code"),
        ("memory_store", "Store information for later", "key, value, repository, branch"),
        ("memory_recall", "Retrieve stored information", "key, repository, branch"),
        ("graph_query", "Query the knowledge graph", "query, num_results"),
        ("graph_add", "Add content to the knowledge graph", "content, name, group_id"),
        ("get_stats", "Get indexing statistics", ""),
    ]

    for name, desc, params in tools:
        print(f"\n  {name}")
        print(f"    Description: {desc}")
        if params:
            print(f"    Parameters: {params}")

    print("\n\nQuick Start:")
    print("-" * 60)
    print("""
1. Install DocuGraph AI:
   pip install docugraph-ai

2. Index some documentation:
   docugraph crawl https://docs.python.org/3/library/asyncio.html

3. Start the MCP server:
   docugraph mcp-server

4. Configure your MCP client (Claude Code, Cursor, etc.)

5. Use natural language to search:
   "Search my docs for async error handling"
""")


if __name__ == "__main__":
    print_config()
