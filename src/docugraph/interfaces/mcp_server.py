"""MCP (Model Context Protocol) server for DocuGraph AI.

Provides tools for Claude Code and other MCP-compatible clients:
- search_docs: Search indexed documentation
- hybrid_search: Combine vector, keyword, and graph search
- crawl_url: Crawl and index a URL
- index_git: Clone and index a git repository
- memory_store: Store values in agent memory
- memory_recall: Retrieve values from agent memory
- graph_query: Query the knowledge graph
- get_stats: Get indexing statistics
"""

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
)

from docugraph.storage.vector_store import VectorStore

# Initialize server
server = Server("docugraph")

# Global vector store instance
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Get or create the vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        # Search tools
        Tool(
            name="search_docs",
            description=(
                "Search indexed documentation for relevant content. "
                "Returns chunks of documentation that match the query, "
                "with relevance scores and source information."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documentation",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="hybrid_search",
            description=(
                "Perform hybrid search combining vector similarity, keyword matching, "
                "and optionally knowledge graph traversal for more comprehensive results."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 10)",
                        "default": 10,
                    },
                    "mode": {
                        "type": "string",
                        "description": "Search mode: vector, keyword, hybrid, or all (default: hybrid)",
                        "enum": ["vector", "keyword", "hybrid", "all"],
                        "default": "hybrid",
                    },
                    "include_graph": {
                        "type": "boolean",
                        "description": "Include knowledge graph search (default: false)",
                        "default": False,
                    },
                    "rerank": {
                        "type": "boolean",
                        "description": (
                            "Rerank fused results with the local cross-encoder "
                            "(BAAI/bge-reranker-v2-m3, multilingual). Downloads "
                            "~1.8 GB on first use. Opt-in; off by default."
                        ),
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        ),
        # Indexing tools
        Tool(
            name="crawl_url",
            description=(
                "Crawl a URL and index its content for future searches. "
                "Use this to add new documentation to the knowledge base."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to crawl and index",
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "Maximum pages to crawl (default: 1)",
                        "default": 1,
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="index_git",
            description=(
                "Clone a git repository and index its documentation. "
                "Useful for adding project documentation to the knowledge base."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Git repository URL (HTTPS or SSH)",
                    },
                    "name": {
                        "type": "string",
                        "description": "Custom name for the repository (optional)",
                    },
                    "ref": {
                        "type": "string",
                        "description": "Branch, tag, or commit to checkout (optional)",
                    },
                    "include_code": {
                        "type": "boolean",
                        "description": "Include code files, not just docs (default: false)",
                        "default": False,
                    },
                },
                "required": ["url"],
            },
        ),
        # Memory tools
        Tool(
            name="memory_store",
            description=(
                "Store information in agent memory for later recall. "
                "Memory is scoped by repository and branch for context isolation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Key to store the value under",
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to store (can be JSON for complex data)",
                    },
                    "repository": {
                        "type": "string",
                        "description": "Repository scope (default: 'default')",
                        "default": "default",
                    },
                    "branch": {
                        "type": "string",
                        "description": "Branch scope (default: 'main')",
                        "default": "main",
                    },
                },
                "required": ["key", "value"],
            },
        ),
        Tool(
            name="memory_recall",
            description=(
                "Retrieve information from agent memory. "
                "Use this to recall previous decisions, context, or stored data."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Key to retrieve (or pattern with % wildcard)",
                    },
                    "repository": {
                        "type": "string",
                        "description": "Repository scope (default: 'default')",
                        "default": "default",
                    },
                    "branch": {
                        "type": "string",
                        "description": "Branch scope (default: 'main')",
                        "default": "main",
                    },
                    "list_all": {
                        "type": "boolean",
                        "description": "List all keys instead of getting a specific value",
                        "default": False,
                    },
                },
                "required": ["key"],
            },
        ),
        # Graph tools
        Tool(
            name="graph_query",
            description=(
                "Query the knowledge graph to find entities and relationships. "
                "Returns facts extracted from indexed documentation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for the knowledge graph",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 10)",
                        "default": 10,
                    },
                    "group_id": {
                        "type": "string",
                        "description": "Group ID to filter by (optional)",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="graph_add",
            description=(
                "Add content to the knowledge graph. Extracts entities and relationships using LLM."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to extract entities from",
                    },
                    "name": {
                        "type": "string",
                        "description": "Episode name (optional)",
                    },
                    "group_id": {
                        "type": "string",
                        "description": "Group ID for scoping (default: 'default')",
                        "default": "default",
                    },
                },
                "required": ["content"],
            },
        ),
        # Utility tools
        Tool(
            name="get_stats",
            description="Get statistics about the indexed documentation.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """Handle tool calls."""
    handlers = {
        "search_docs": _search_docs,
        "hybrid_search": _hybrid_search,
        "crawl_url": _crawl_url,
        "index_git": _index_git,
        "memory_store": _memory_store,
        "memory_recall": _memory_recall,
        "graph_query": _graph_query,
        "graph_add": _graph_add,
        "get_stats": _get_stats,
    }

    handler = handlers.get(name)
    if handler:
        return await handler(arguments)

    return CallToolResult(
        content=[TextContent(type="text", text=f"Unknown tool: {name}")],
        isError=True,
    )


async def _search_docs(arguments: dict[str, Any]) -> CallToolResult:
    """Search indexed documentation."""
    query = arguments.get("query", "")
    top_k = arguments.get("top_k", 5)

    if not query:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: query is required")],
            isError=True,
        )

    try:
        vector_store = get_vector_store()
        results = vector_store.search(query, top_k=top_k)

        if not results:
            return CallToolResult(
                content=[TextContent(type="text", text="No results found.")],
            )

        # Format results
        output_parts = [f"Found {len(results)} results for: {query}\n"]

        for i, result in enumerate(results, 1):
            chunk = result.chunk
            source = chunk.metadata.get("source_url") or chunk.metadata.get(
                "source_path", "Unknown"
            )
            title = chunk.metadata.get("title", "Untitled")
            section = chunk.metadata.get("section_title", "")

            output_parts.append(f"\n--- Result {i} (score: {result.score:.4f}) ---")
            output_parts.append(f"Title: {title}")
            if section:
                output_parts.append(f"Section: {section}")
            output_parts.append(f"Source: {source}")
            output_parts.append(f"\n{chunk.content}\n")

        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(output_parts))],
        )

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error searching: {str(e)}")],
            isError=True,
        )


async def _hybrid_search(arguments: dict[str, Any]) -> CallToolResult:
    """Perform hybrid search."""
    query = arguments.get("query", "")
    top_k = arguments.get("top_k", 10)
    mode = arguments.get("mode", "hybrid")
    include_graph = arguments.get("include_graph", False)
    rerank = arguments.get("rerank", False)

    if not query:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: query is required")],
            isError=True,
        )

    try:
        from docugraph.retrieval.hybrid_search import (
            FusionStrategy,
            HybridRetriever,
            HybridSearchConfig,
            SearchMode,
        )

        mode_map = {
            "vector": SearchMode.VECTOR,
            "keyword": SearchMode.KEYWORD,
            "hybrid": SearchMode.HYBRID,
            "all": SearchMode.ALL,
        }

        config = HybridSearchConfig(include_graph=include_graph)
        retriever = HybridRetriever(config=config)

        results = retriever.search(
            query=query,
            top_k=top_k,
            mode=mode_map.get(mode, SearchMode.HYBRID),
            fusion=FusionStrategy.RRF,
        )

        if rerank and results:
            # Opt-in cross-encoder rerank (downloads ~1.8 GB on first use).
            from docugraph.retrieval.reranker import FastembedReranker

            reranker = FastembedReranker()
            reranked = reranker.rerank(
                query=query,
                results=[r.to_search_result() for r in results],
                top_k=top_k,
            )
            # Reorder the HybridResult objects (which carry source attribution)
            # and update each score to the blended rerank score so the reported
            # score matches the reranked order.
            by_id = {r.chunk.id: r for r in results}
            reordered = []
            for rr in reranked:
                hit = by_id.get(rr.chunk.id)
                if hit is not None:
                    hit.score = rr.final_score
                    reordered.append(hit)
            results = reordered

        if not results:
            return CallToolResult(
                content=[TextContent(type="text", text="No results found.")],
            )

        output_parts = [f"Found {len(results)} results for: {query}\n"]

        for i, result in enumerate(results, 1):
            chunk = result.chunk
            source = chunk.metadata.get("source_url") or chunk.metadata.get(
                "source_path", "Unknown"
            )
            title = chunk.metadata.get("title", "Untitled")

            output_parts.append(f"\n--- Result {i} (score: {result.score:.4f}) ---")
            output_parts.append(f"Title: {title}")
            output_parts.append(f"Source: {source}")
            output_parts.append(f"Found via: {', '.join(result.sources)}")
            output_parts.append(f"\n{chunk.content}\n")

        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(output_parts))],
        )

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error in hybrid search: {str(e)}")],
            isError=True,
        )


async def _crawl_url(arguments: dict[str, Any]) -> CallToolResult:
    """Crawl a URL and index its content."""
    url = arguments.get("url", "")
    max_pages = arguments.get("max_pages", 1)

    if not url:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: url is required")],
            isError=True,
        )

    try:
        from docugraph.ingestion.chunker import Chunker
        from docugraph.ingestion.crawler import DocCrawler

        crawler = DocCrawler()
        chunker = Chunker()
        vector_store = get_vector_store()

        # Crawl
        if max_pages == 1:
            documents = [await crawler.crawl_single(url)]
        else:
            documents = await crawler.crawl_site(url, max_pages=max_pages)

        # Chunk and index
        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        count = vector_store.add_chunks(all_chunks)

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Successfully indexed {count} chunks from {len(documents)} pages.\nTotal chunks in store: {vector_store.count()}",
                )
            ],
        )

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error crawling: {str(e)}")],
            isError=True,
        )


async def _index_git(arguments: dict[str, Any]) -> CallToolResult:
    """Clone and index a git repository."""
    url = arguments.get("url", "")
    name = arguments.get("name")
    ref = arguments.get("ref")
    include_code = arguments.get("include_code", False)

    if not url:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: url is required")],
            isError=True,
        )

    try:
        from docugraph.ingestion.chunker import Chunker
        from docugraph.ingestion.git_indexer import GitIndexer, GitIndexerConfig

        config = GitIndexerConfig(
            clone_depth=1,  # Shallow clone for speed
            ref=ref,
            include_code=include_code,
        )

        indexer = GitIndexer(config=config)
        chunker = Chunker()
        vector_store = get_vector_store()

        # Clone and index
        repo_info, documents = indexer.clone_and_index(url, name=name, ref=ref)

        # Chunk and index
        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        count = vector_store.add_chunks(all_chunks)

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        f"Successfully indexed {count} chunks from {len(documents)} files.\n"
                        f"Repository: {repo_info.name} @ {repo_info.branch or 'HEAD'}\n"
                        f"Commit: {(repo_info.commit_hash or 'N/A')[:12]}\n"
                        f"Total chunks in store: {vector_store.count()}"
                    ),
                )
            ],
        )

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error indexing git repo: {str(e)}")],
            isError=True,
        )


async def _memory_store(arguments: dict[str, Any]) -> CallToolResult:
    """Store a value in agent memory."""
    key = arguments.get("key", "")
    value = arguments.get("value", "")
    repository = arguments.get("repository", "default")
    branch = arguments.get("branch", "main")

    if not key:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: key is required")],
            isError=True,
        )

    try:
        from docugraph.storage.memory_store import MemoryStore

        store = MemoryStore()

        # Try to parse as JSON
        try:
            parsed_value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed_value = value

        store.set(key, parsed_value, repository=repository, branch=branch)

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Stored: {key} = {value}\nScope: {repository}/{branch}",
                )
            ],
        )

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error storing memory: {str(e)}")],
            isError=True,
        )


async def _memory_recall(arguments: dict[str, Any]) -> CallToolResult:
    """Retrieve a value from agent memory."""
    key = arguments.get("key", "")
    repository = arguments.get("repository", "default")
    branch = arguments.get("branch", "main")
    list_all = arguments.get("list_all", False)

    if not key and not list_all:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: key is required")],
            isError=True,
        )

    try:
        from docugraph.storage.memory_store import MemoryStore

        store = MemoryStore()

        if list_all or key == "*":
            keys = store.list_keys(repository=repository, branch=branch)
            if not keys:
                return CallToolResult(
                    content=[
                        TextContent(type="text", text=f"No memories found in {repository}/{branch}")
                    ],
                )
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Memory keys in {repository}/{branch}:\n"
                        + "\n".join(f"  - {k}" for k in keys),
                    )
                ],
            )

        value = store.get(key, repository=repository, branch=branch)

        if value is None:
            return CallToolResult(
                content=[TextContent(type="text", text=f"No value found for key: {key}")],
            )

        # Format value
        value_str = json.dumps(value, indent=2) if isinstance(value, (dict, list)) else str(value)

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"{key} = {value_str}\nScope: {repository}/{branch}",
                )
            ],
        )

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error recalling memory: {str(e)}")],
            isError=True,
        )


async def _graph_query(arguments: dict[str, Any]) -> CallToolResult:
    """Query the knowledge graph."""
    query = arguments.get("query", "")
    num_results = arguments.get("num_results", 10)
    group_id = arguments.get("group_id")

    if not query:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: query is required")],
            isError=True,
        )

    try:
        from docugraph.storage.graph_store import GraphStore

        store = GraphStore()
        group_ids = [group_id] if group_id else None

        try:
            results = await store.search(query, group_ids=group_ids, num_results=num_results)
        finally:
            await store.close()

        if not results:
            return CallToolResult(
                content=[TextContent(type="text", text="No results found in knowledge graph.")],
            )

        output_parts = [f"Found {len(results)} facts for: {query}\n"]

        for i, r in enumerate(results, 1):
            output_parts.append(f"{i}. {r['fact']}")
            if r.get("valid_at"):
                output_parts.append(f"   Valid: {r['valid_at']}")
            output_parts.append("")

        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(output_parts))],
        )

    except RuntimeError as e:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        f"Graph store requires LLM. Error: {str(e)}\n\n"
                        "To use the knowledge graph:\n"
                        "  Local: Install Ollama (https://ollama.ai) and run: ollama serve\n"
                        "  Cloud: Set OPENAI_API_KEY or ANTHROPIC_API_KEY"
                    ),
                )
            ],
            isError=True,
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error querying graph: {str(e)}")],
            isError=True,
        )


async def _graph_add(arguments: dict[str, Any]) -> CallToolResult:
    """Add content to the knowledge graph."""
    content = arguments.get("content", "")
    name = arguments.get("name")
    group_id = arguments.get("group_id", "default")

    if not content:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: content is required")],
            isError=True,
        )

    try:
        from docugraph.storage.graph_store import GraphStore

        store = GraphStore()

        try:
            episode_name = name or f"episode_{hash(content) % 10000}"
            result = await store.add_episode(
                name=episode_name,
                content=content,
                source_type="text",
                group_id=group_id,
            )
        finally:
            await store.close()

        output_parts = [
            f"Added to knowledge graph: {result['episode_uuid']}",
            f"Extracted {result['entities_count']} entities",
            f"Extracted {result['relationships_count']} relationships",
        ]

        if result["entities"]:
            output_parts.append("\nEntities:")
            for e in result["entities"][:5]:
                output_parts.append(f"  - {e['name']}")
            if len(result["entities"]) > 5:
                output_parts.append(f"  ... and {len(result['entities']) - 5} more")

        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(output_parts))],
        )

    except RuntimeError as e:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        f"Graph store requires LLM. Error: {str(e)}\n\n"
                        "To use the knowledge graph:\n"
                        "  Local: Install Ollama (https://ollama.ai) and run: ollama serve\n"
                        "  Cloud: Set OPENAI_API_KEY or ANTHROPIC_API_KEY"
                    ),
                )
            ],
            isError=True,
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error adding to graph: {str(e)}")],
            isError=True,
        )


async def _get_stats(_arguments: dict[str, Any]) -> CallToolResult:
    """Get indexing statistics."""
    try:
        from docugraph.core.config import get_config

        config = get_config()
        vector_store = get_vector_store()

        stats = {
            "total_chunks": vector_store.count(),
            "data_directory": str(config.storage.data_dir),
            "embedding_model": config.embeddings.model,
            "embedding_provider": config.embeddings.provider.value,
        }

        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(stats, indent=2))],
        )

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error getting stats: {str(e)}")],
            isError=True,
        )


def run_server() -> None:
    """Run the MCP server."""
    asyncio.run(_run_server())


async def _run_server() -> None:
    """Run the MCP server with stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    """Entry point for the MCP server."""
    run_server()


if __name__ == "__main__":
    main()
