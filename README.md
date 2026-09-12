# DocuGraph AI v1

Documentation RAG for coding agents: vector search, a documentation
knowledge graph, agent memory, and an **MCP server** — with **torch-free
local embeddings** (pure ONNX via [fastembed](https://github.com/qdrant/fastembed)).

The MCP stdio server is the primary surface: it plugs directly into Claude
Code and other MCP clients. The CLI is for out-of-band index bootstrapping
(crawl / index / search / stats / memory / graph / config).

## Features

- **Multi-source ingestion**: crawl web docs, index local files, clone & index git repos, parse API specs
- **Hybrid search**: vector similarity + keyword (BM25), fused with RRF
- **Documentation knowledge graph**: entities/relationships over your docs (Kuzu via Graphiti)
- **Agent memory**: persistent, repository/branch-scoped key-value store
- **Optional reranking**: local multilingual cross-encoder (BAAI/bge-reranker-v2-m3), opt-in
- **Torch-free**: default embeddings and reranking run on ONNX — no PyTorch in the dependency tree

## Quick start

### Installation

```bash
# With uv (recommended)
uv pip install docugraph-ai-v1

# Or with pip
pip install docugraph-ai-v1

# For development
git clone https://github.com/docugraph/docugraph-ai-v1
cd docugraph-ai-v1
uv sync --extra dev
```

First use downloads the default embedding model (`BAAI/bge-small-en-v1.5`,
~130 MB ONNX) and caches it under `FASTEMBED_CACHE_PATH` (default
`~/.cache/fastembed`).

### MCP integration with Claude Code (primary surface)

Add to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "docugraph": {
      "command": "docugraph-mcp",
      "env": {}
    }
  }
}
```

The server speaks JSON-RPC over stdio and exposes nine tools:

| Tool | Purpose |
|------|---------|
| `search_docs` | Vector search over indexed documentation |
| `hybrid_search` | Vector + keyword (+ optional graph) fusion; `rerank` opt-in |
| `crawl_url` | Crawl a URL and index it |
| `index_git` | Clone and index a git repository |
| `memory_store` | Store a value in repo/branch-scoped agent memory |
| `memory_recall` | Retrieve a stored value |
| `graph_query` | Query the documentation knowledge graph |
| `graph_add` | Extract entities/relationships from text into the graph |
| `get_stats` | Indexing statistics |

### CLI (index bootstrap + one-off commands)

```bash
# Crawl and index documentation
docugraph crawl https://fastapi.tiangolo.com/tutorial/first-steps/

# Index a local docs directory
docugraph index local ./docs

# Search indexed content
docugraph search "how to create a path operation"
docugraph search "async error handling" --top-k 10

# Rerank with the local multilingual cross-encoder (downloads ~1.8 GB once)
docugraph search "path operations" --rerank

# Documentation knowledge graph (requires an LLM — see below)
docugraph graph add "FastAPI depends on Starlette and Pydantic."
docugraph graph search "what does FastAPI depend on?"

# Agent memory
docugraph memory set build.status green
docugraph memory get build.status

# Stats + reset
docugraph stats
docugraph clear

# Run the MCP server manually (usually launched by the MCP client)
docugraph mcp-server
```

### Python API

```python
import asyncio
from docugraph.ingestion.crawler import DocCrawler
from docugraph.ingestion.chunker import Chunker
from docugraph.storage.vector_store import VectorStore


async def main():
    crawler = DocCrawler()
    doc = await crawler.crawl_single("https://docs.python.org/3/tutorial/")

    chunker = Chunker()
    chunks = chunker.chunk_document(doc)

    store = VectorStore()  # embeds with fastembed by default
    store.add_chunks(chunks)

    results = store.search("how to use lists", top_k=5)
    for r in results:
        print(f"{r.score:.4f}: {r.chunk.content[:100]}...")


asyncio.run(main())
```

## Configuration

Create `~/.docugraph/config.yaml` (all fields optional — defaults shown):

```yaml
storage:
  data_dir: ~/.docugraph/data

embeddings:
  provider: auto            # auto | fastembed | openai | ollama | cohere | sentence-transformers
  model: BAAI/bge-small-en-v1.5
  # dimensions: auto-detected from the model
  batch_size: 32

crawler:
  max_concurrent: 5
  rate_limit: 2.0
  respect_robots: true
```

`provider: auto` resolves to: **OpenAI** (if `OPENAI_API_KEY` is set and the
`openai` package is installed) → **fastembed** (local ONNX, the default) →
**sentence-transformers** (only if you have it installed). Any field can be
overridden by environment variable, e.g. `DOCUGRAPH_EMBEDDINGS__MODEL` or
`DOCUGRAPH_EMBEDDINGS__PROVIDER`.

See [docs/configuration.md](docs/configuration.md) for the full reference,
including the embedding-model consistency guard and the graph LLM requirement.

## The documentation knowledge graph

DocuGraph builds a temporal entity/relationship graph over your docs (Kuzu,
via [Graphiti](https://github.com/getzep/graphiti)). Embeddings for the graph
are served by the same torch-free fastembed adapter — **no embedding service
required**. However, Graphiti performs entity extraction with an **LLM**, so
`graph add` / `graph_query` need an LLM configured (Ollama locally, or an
OpenAI/Anthropic key). Without one, the graph tools degrade gracefully with a
clear message; vector and keyword search are unaffected.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Interfaces                             │
│                 MCP server  │  CLI                          │
├─────────────────────────────────────────────────────────────┤
│                      Retrieval Layer                        │
│        Hybrid Search  │  Optional Rerank  │  Context        │
├─────────────────────────────────────────────────────────────┤
│                      Storage Layer                          │
│        LanceDB (Vectors)  │  Kuzu (Graph)  │  Memory        │
├─────────────────────────────────────────────────────────────┤
│                      Ingestion Pipeline                     │
│           Crawl4AI  │  Chunker  │  Graphiti extraction      │
└─────────────────────────────────────────────────────────────┘
```

## Technology stack

- **Embeddings & reranking**: fastembed (pure ONNX, no torch) — default `bge-small-en-v1.5`; opt-in reranker `bge-reranker-v2-m3`
- **Vector database**: LanceDB (embedded)
- **Graph database**: Kuzu via Graphiti (temporal knowledge graphs)
- **Web crawler**: Crawl4AI (JS rendering, clean Markdown)
- **MCP**: Model Context Protocol for Claude Code integration
- Optional cloud/local backends: OpenAI, Cohere, Ollama (extras)

## License

MIT
