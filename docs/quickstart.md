# DocuGraph AI Quickstart

Get started with DocuGraph AI in under 5 minutes.

## Installation

### Using pip

```bash
pip install docugraph-ai
```

### Using uv (recommended)

```bash
uv pip install docugraph-ai
```

### For macOS Intel (x86_64)

LanceDB doesn't have native wheels for macOS Intel. Use Docker instead:

```bash
git clone https://github.com/yourorg/docugraph-ai.git
cd docugraph-ai
docker-compose up -d
```

## Basic Usage

### 1. Index Documentation

#### From a URL

```bash
# Crawl a single page
docugraph crawl https://fastapi.tiangolo.com/tutorial/first-steps/

# Crawl multiple pages
docugraph crawl https://docs.python.org/3/library/asyncio.html --max-pages 10
```

#### From Local Files

```bash
# Index a directory
docugraph index-local ./docs

# Index with specific patterns
docugraph index-local ./docs --patterns "*.md" --patterns "*.rst"
```

#### From a Git Repository

```bash
# Clone and index
docugraph index-git https://github.com/tiangolo/fastapi.git

# Include code files
docugraph index-git https://github.com/yourorg/yourrepo.git --include-code
```

### 2. Search

```bash
# Basic search
docugraph search "how to handle async errors"

# With more results
docugraph search "authentication" --top-k 10
```

### 3. Check Stats

```bash
docugraph stats
```

## Using with Claude Code / MCP

DocuGraph AI can be used as an MCP server with Claude Code, Cursor, or other MCP-compatible clients.

### Configuration

Add to your MCP configuration (e.g., `~/.claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "docugraph": {
      "command": "docugraph",
      "args": ["mcp-server"]
    }
  }
}
```

### Available Tools

Once configured, you can use natural language to:

- **Search documentation**: "Search my docs for error handling patterns"
- **Crawl new content**: "Index the React documentation"
- **Store context**: "Remember that we're using JWT for auth"
- **Query the knowledge graph**: "What entities are related to FastAPI?"

## REST API

Start the API server:

```bash
docugraph serve --port 8000
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/stats` | GET | Get indexing statistics |
| `/v1/search` | POST | Search documents |
| `/v1/search/hybrid` | POST | Hybrid search |
| `/v1/crawl` | POST | Crawl and index URL |
| `/v1/index/local` | POST | Index local files |
| `/v1/index/git` | POST | Index git repository |
| `/v1/memory` | POST/GET | Store/retrieve memory |
| `/v1/graph/search` | POST | Query knowledge graph |

### Example: Search

```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "async error handling", "top_k": 5}'
```

## Python API

```python
from docugraph.storage.vector_store import VectorStore
from docugraph.ingestion.crawler import DocCrawler
from docugraph.ingestion.chunker import Chunker

# Initialize
crawler = DocCrawler()
chunker = Chunker()
vector_store = VectorStore()

# Crawl and index
async def index_docs():
    doc = await crawler.crawl_single("https://example.com/docs")
    chunks = chunker.chunk_document(doc)
    vector_store.add_chunks(chunks)

# Search
results = vector_store.search("my query", top_k=5)
for result in results:
    print(f"Score: {result.score}")
    print(f"Content: {result.chunk.content[:200]}...")
```

## Configuration

Create `~/.docugraph/config.yaml`:

```yaml
# Storage
storage:
  data_dir: ~/.docugraph/data

# Embeddings (local-first)
embeddings:
  provider: sentence-transformers
  model: all-MiniLM-L6-v2

# Server
server:
  host: 0.0.0.0
  port: 8000
```

Or use environment variables:

```bash
export DOCUGRAPH_STORAGE__DATA_DIR=/custom/path
export DOCUGRAPH_EMBEDDINGS__MODEL=BAAI/bge-base-en-v1.5
```

## Next Steps

- [Configuration Reference](configuration.md)
- [API Reference](api-reference.md)
- [Examples](../examples/)
