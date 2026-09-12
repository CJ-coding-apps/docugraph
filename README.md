# DocuGraph AI

Intelligent coding assistant with knowledge graphs, vector search, and agent memory for custom documentation RAG.

## Features

- **Multi-source Ingestion**: Crawl web documentation, index local files, parse API specs
- **Hybrid Search**: Combine vector similarity and keyword search for better results
- **Knowledge Graph**: Build entity relationships for contextual understanding
- **Agent Memory**: Persistent, branch-aware memory for coding sessions
- **Multiple Interfaces**: CLI, REST API, and MCP server for Claude Code integration

## Quick Start

### Installation

```bash
# Install with uv (recommended)
uv pip install docugraph-ai

# Or with pip
pip install docugraph-ai

# For development
git clone https://github.com/docugraph/docugraph-ai
cd docugraph-ai
uv sync
```

### Basic Usage

```bash
# Crawl and index documentation
docugraph crawl https://fastapi.tiangolo.com/tutorial/first-steps/

# Search indexed content
docugraph search "how to create a path operation"

# Start the REST API server
docugraph serve

# Start MCP server for Claude Code
docugraph mcp-server
```

### Python API

```python
import asyncio
from docugraph.ingestion.crawler import DocCrawler
from docugraph.ingestion.chunker import Chunker
from docugraph.storage.vector_store import VectorStore

async def main():
    # Crawl documentation
    crawler = DocCrawler()
    doc = await crawler.crawl_single("https://docs.python.org/3/tutorial/")

    # Chunk the document
    chunker = Chunker()
    chunks = chunker.chunk_document(doc)

    # Index in vector store
    store = VectorStore()
    store.add_chunks(chunks)

    # Search
    results = store.search("how to use lists", top_k=5)
    for r in results:
        print(f"{r.score:.4f}: {r.chunk.content[:100]}...")

asyncio.run(main())
```

### MCP Integration with Claude Code

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

Then in Claude Code:
- Use `search_docs` to search indexed documentation
- Use `crawl_url` to add new documentation to the index
- Use `get_stats` to see indexing statistics

## Configuration

Create `~/.docugraph/config.yaml`:

```yaml
storage:
  data_dir: ~/.docugraph/data

embeddings:
  provider: sentence-transformers  # local-first
  model: all-MiniLM-L6-v2
  device: auto

crawler:
  max_concurrent: 5
  rate_limit: 2.0
  respect_robots: true

server:
  host: 0.0.0.0
  port: 8000
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interfaces                         │
│              CLI  │  REST API  │  MCP Server                │
├─────────────────────────────────────────────────────────────┤
│                     Retrieval Layer                         │
│           Hybrid Search  │  Reranking  │  Context           │
├─────────────────────────────────────────────────────────────┤
│                     Storage Layer                           │
│         LanceDB (Vectors)  │  Kuzu (Graph)  │  Memory       │
├─────────────────────────────────────────────────────────────┤
│                     Ingestion Pipeline                      │
│           Crawl4AI  │  Chunker  │  Entity Extraction        │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

- **Vector Database**: LanceDB (embedded, fast)
- **Graph Database**: Kuzu via Graphiti (temporal knowledge graphs)
- **Web Crawler**: Crawl4AI (JS rendering, clean Markdown)
- **Embeddings**: sentence-transformers (local-first), with OpenAI/Ollama fallback
- **API**: FastAPI with async support
- **MCP**: Model Context Protocol for Claude Code integration

## License

MIT
