# DocuGraph AI v1 Quickstart

Get started with DocuGraph AI in under 5 minutes.

## Installation

### Using uv (recommended)

```bash
uv pip install docugraph-ai-v1
```

### Using pip

```bash
pip install docugraph-ai-v1
```

First use downloads the default embedding model (`BAAI/bge-small-en-v1.5`,
~130 MB, pure ONNX — no torch) into `FASTEMBED_CACHE_PATH`
(default `~/.cache/fastembed`).

### For macOS Intel (x86_64)

LanceDB and onnxruntime wheel coverage for macOS Intel is limited on newer
releases; the dependency pins target versions with x86_64 wheels. If you hit
install issues, use Docker:

```bash
git clone https://github.com/docugraph/docugraph-ai-v1.git
cd docugraph-ai-v1
# Bootstrap an index into the shared volume:
docker compose run --rm cli index local /path/to/docs
# Run the MCP stdio server (see the Docker section below)
```

## Basic usage

### 1. Index documentation

#### From a URL

```bash
# Crawl a single page
docugraph crawl https://fastapi.tiangolo.com/tutorial/first-steps/

# Crawl multiple pages
docugraph crawl https://docs.python.org/3/library/asyncio.html --max-pages 10
```

#### From local files

```bash
# Index a directory (recursive by default)
docugraph index local ./docs

# Index with specific patterns
docugraph index local ./docs --pattern "*.md" --pattern "*.rst"
```

#### From a git repository

```bash
# Clone and index
docugraph index git https://github.com/tiangolo/fastapi.git

# Include code files, pin a ref
docugraph index git https://github.com/yourorg/yourrepo.git --ref main --include-code
```

### 2. Search

```bash
# Basic search
docugraph search "how to handle async errors"

# More results
docugraph search "authentication" --top-k 10

# Rerank with the local multilingual cross-encoder
# (downloads bge-reranker-v2-m3, ~1.8 GB, on first use)
docugraph search "authentication" --rerank
```

### 3. Check stats

```bash
docugraph stats
```

## Using with Claude Code / MCP (primary surface)

DocuGraph AI is designed to run as an MCP stdio server for Claude Code,
Cursor, or any MCP-compatible client.

### Configuration

Add to your MCP configuration (e.g. `~/.claude/claude_desktop_config.json`):

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

(`docugraph mcp-server` is an equivalent alias if you prefer the subcommand
form.)

### Available tools

Nine tools are exposed. In natural language you can:

- **Search documentation**: "Search my docs for error handling patterns"
- **Hybrid search**: "Find auth docs across vector and keyword search"
- **Crawl / index**: "Index the React documentation", "Index this git repo"
- **Store context**: "Remember that we're using JWT for auth"
- **Query the doc graph**: "What entities are related to FastAPI?" *(needs an LLM — see [configuration.md](configuration.md))*

## Documentation knowledge graph

```bash
# Requires an LLM for entity extraction (Ollama, or an OpenAI/Anthropic key).
docugraph graph add "FastAPI depends on Starlette and Pydantic."
docugraph graph search "what does FastAPI depend on?"
```

Graph *embeddings* are served locally by fastembed (no embedding service).
Only entity **extraction** needs an LLM; without one, graph tools return a
clear guidance message and vector/keyword search continue to work.

## Agent memory

```bash
docugraph memory set build.status green
docugraph memory get build.status
docugraph memory list
```

Memory is scoped by repository and branch.

## Docker

The MCP server speaks stdio — it is not an HTTP service. Run it with
`docker run -i`:

```bash
IMG=$(docker compose build -q cli)
docker run -i --rm -v docugraph-data:/data/docugraph "$IMG" docugraph-mcp
```

Use the `cli` compose service for one-off index bootstrapping into the shared
`docugraph-data` volume.

## Python API

```python
from docugraph.storage.vector_store import VectorStore
from docugraph.ingestion.crawler import DocCrawler
from docugraph.ingestion.chunker import Chunker

crawler = DocCrawler()
chunker = Chunker()
vector_store = VectorStore()  # embeds with fastembed by default


async def index_docs():
    doc = await crawler.crawl_single("https://example.com/docs")
    chunks = chunker.chunk_document(doc)
    vector_store.add_chunks(chunks)


results = vector_store.search("my query", top_k=5)
for result in results:
    print(f"Score: {result.score}")
    print(f"Content: {result.chunk.content[:200]}...")
```

## Configuration

Create `~/.docugraph/config.yaml`:

```yaml
storage:
  data_dir: ~/.docugraph/data

embeddings:
  provider: auto                 # auto | fastembed | openai | ollama | cohere | sentence-transformers
  model: BAAI/bge-small-en-v1.5
```

Or use environment variables:

```bash
export DOCUGRAPH_STORAGE__DATA_DIR=/custom/path
export DOCUGRAPH_EMBEDDINGS__MODEL=BAAI/bge-base-en-v1.5
```

> **Note:** changing the embedding model after indexing will fail loudly at
> search time (the store records which model wrote its vectors). Run
> `docugraph clear` and re-index when you switch models.

## Next steps

- [Configuration Reference](configuration.md)
- [Examples](../examples/)
