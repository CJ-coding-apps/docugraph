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
pip install docugraph
```

**Requires Python 3.11–3.13.** CI runs the suite on all three, on Linux. Python
3.14 is excluded deliberately, for two independent reasons: `kuzu` 0.11.3 — the
final release, and the version this package pins — publishes no 3.14 wheels for
macOS or Windows, and fastembed's 3.14 requirement (`onnxruntime>=1.24.2`) cannot
be satisfied alongside the `onnxruntime<1.24` cap this package carries.

Or work from a checkout — clone this repository, then from its root:

```bash
# With uv (recommended)
uv sync --extra dev

# Or with pip
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

First use downloads the default embedding model (`BAAI/bge-small-en-v1.5`,
~130 MB ONNX) and caches it under `FASTEMBED_CACHE_PATH`. Set that variable to
keep the weights somewhere durable; unset, fastembed puts them in a
`fastembed_cache` subdirectory of the system temp directory, which the OS is
free to clear.

**URL crawling needs a browser that the Python install does not include.**
Crawl4AI drives Playwright's Chromium, so run this once before using
`docugraph crawl` or the `crawl_url` tool:

```bash
playwright install chromium
# On Linux, add --with-deps to install the browser's system libraries too:
#   playwright install --with-deps chromium
```

This is ~190 MB and is cached, so it is a one-off. `index local`, `index git`,
`search`, `memory` and `stats` do not need it — only fetching a URL does.

### MCP integration with Claude Code (primary surface)

Add to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "docugraph": {
      "command": "/absolute/path/to/docugraph/.venv/bin/docugraph-mcp",
      "env": {}
    }
  }
}
```

Use the **absolute path** to the installed script. An MCP client launches the
server directly rather than through a shell, so it does not pick up the venv
that `uv sync` created — a bare `"command": "docugraph-mcp"` fails with
`No such file or directory` unless that directory happens to be on the client's
`PATH`. `uv sync` installs the script at `.venv/bin/docugraph-mcp` inside the
repository (`Scripts\docugraph-mcp.exe` on Windows); substitute your real path.

If you would rather not hardcode it, this is equivalent and needs no path
change when the checkout moves:

```json
{
  "mcpServers": {
    "docugraph": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/docugraph", "run", "docugraph-mcp"],
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
| `crawl_url` | Crawl a URL and index it ¹ |
| `index_git` | Clone and index a git repository |
| `memory_store` | Store a value in repo/branch-scoped agent memory |
| `memory_recall` | Retrieve a stored value |
| `graph_query` | Query the documentation knowledge graph ² |
| `graph_add` | Extract entities/relationships from text into the graph ² |
| `get_stats` | Indexing statistics |

¹ Needs Playwright's Chromium — see the installation step above.
² Needs an LLM configured — Ollama locally, or an OpenAI/Anthropic key. See
[the knowledge graph section](#the-documentation-knowledge-graph).

### CLI (index bootstrap + one-off commands)

Run these from the repository root. `uv sync` installs the `docugraph` command
into the project's virtual environment but does not put that directory on your
`PATH`, so each command below is prefixed with `uv run`. If you activated the
virtual environment (`source .venv/bin/activate`, as the pip instructions do),
drop the prefix and run them as bare `docugraph …`.

```bash
# Crawl and index documentation (needs the Chromium install from above)
uv run docugraph crawl https://fastapi.tiangolo.com/tutorial/first-steps/

# Index a local docs directory
uv run docugraph index local ./docs

# Search indexed content
uv run docugraph search "how to create a path operation"
uv run docugraph search "async error handling" --top-k 10

# Rerank with the local multilingual cross-encoder (downloads about 2 GB once)
uv run docugraph search "path operations" --rerank

# Documentation knowledge graph — needs an LLM (Ollama locally, or an
# OPENAI_API_KEY/ANTHROPIC_API_KEY); see "The documentation knowledge graph"
uv run docugraph graph add "FastAPI depends on Starlette and Pydantic."
uv run docugraph graph search "what does FastAPI depend on?"

# Agent memory
uv run docugraph memory set build.status green
uv run docugraph memory get build.status

# Stats + reset
uv run docugraph stats
uv run docugraph clear

# Configuration
uv run docugraph config show
uv run docugraph config init

# Run the MCP server manually (usually launched by the MCP client)
uv run docugraph mcp-server
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
  provider: auto            # auto | fastembed | openai | ollama | cohere
  model: BAAI/bge-small-en-v1.5   # fastembed's default; see the note below
  # dimensions: auto-detected from the model
  batch_size: 32

crawler:
  max_concurrent: 5
  rate_limit: 2.0
  respect_robots: true
```

`provider: auto` resolves to **fastembed** (local ONNX) — nothing about your
environment changes that, so documents are not sent anywhere for embedding
unless you name a cloud provider. Set `provider: openai` (or `cohere`) to send
them there; that is the request, and it needs a model named with it: the entry
above is fastembed's own default and no cloud provider has one. Any field can be
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

### KuzuDB version

`kuzu` is pinned to `==0.11.3` in `pyproject.toml` (reached through
`graphiti-core[kuzu]`, which requires `kuzu>=0.11.3`). Kuzu is archived upstream —
0.11.3 is the final release — and its on-disk format is not stable before 1.0, so a
database written by one Kuzu version may not be readable by another. Treat the pin
as a migration boundary, not a floor: to move versions, upgrade and then re-ingest
rather than pointing the new version at the old graph file.

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

Apache-2.0 — see `LICENSE`.
