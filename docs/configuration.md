# Configuration Reference

DocuGraph AI can be configured via:
1. Configuration file (`~/.docugraph/config.yaml`)
2. Environment variables (prefixed with `DOCUGRAPH_`)
3. Command-line arguments

## Configuration File

Default location: `~/.docugraph/config.yaml`

```yaml
# =============================================================================
# Storage Configuration
# =============================================================================
storage:
  # Directory for all data (vectors, graph, memory)
  data_dir: ~/.docugraph/data

  # Vector database backend
  vector_db: lancedb  # Currently only lancedb supported

  # Graph database backend (via Graphiti)
  graph_db: kuzu      # Currently only kuzu supported

# =============================================================================
# Embeddings Configuration (local-first, torch-free by default)
# =============================================================================
embeddings:
  # Provider. Options:
  #   auto                  - OpenAI (if OPENAI_API_KEY + openai installed)
  #                           -> fastembed -> sentence-transformers
  #   fastembed             - local, pure ONNX, no torch (default backend)
  #   openai                - OpenAI API (needs the 'cloud' extra + key)
  #   ollama                - local Ollama server (needs the 'ollama' extra)
  #   cohere                - Cohere API (needs the 'cloud' extra + key)
  #   sentence-transformers - only if you have it installed (pulls in torch)
  provider: auto

  # Model name (provider-specific)
  model: BAAI/bge-small-en-v1.5   # fastembed default, 384 dims

  # Alternative fastembed models:
  # - BAAI/bge-base-en-v1.5     (better quality, 768 dims)
  # - text-embedding-3-small    (OpenAI; provider: openai)

  # Embedding dimensions (auto-detected from the model)
  dimensions: null

  # Batch size for embedding generation
  batch_size: 32

# =============================================================================
# LLM Configuration (required only for the documentation knowledge graph)
# =============================================================================
llm:
  # Provider selection. Options: auto, ollama, openai, anthropic
  provider: auto      # Tries: ollama -> openai -> anthropic
  model: auto

  # Explicit configuration:
  # provider: ollama
  # model: llama3.2
  #
  # provider: openai
  # model: gpt-4o-mini
  #
  # provider: anthropic
  # model: claude-haiku-4-5-20251001

  temperature: 0.0
  max_tokens: 4096

# =============================================================================
# Crawler Configuration
# =============================================================================
crawler:
  max_concurrent: 5
  rate_limit: 2.0            # requests per second
  user_agent: DocuGraph-AI/0.1.0
  respect_robots: true
  cache_ttl: 86400          # 24 hours
  timeout: 30               # seconds
```

> There is no `server:` or `mcp:` section. The MCP server speaks stdio and is
> launched by the MCP client (or `docugraph-mcp` / `docugraph mcp-server`);
> there is no HTTP transport and no REST API.

## The embedding-model consistency guard

The vector store records which embedding model wrote its vectors (in an
`embedding_model` column). When you open an existing index, DocuGraph checks:

- the stored vector dimension matches the current embedder, **and**
- the stored model **identity** matches the current `model`.

The identity check matters because different models can share a dimension —
for example `all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5` are both 384-dim,
so a dimension-only check would silently mix incompatible vectors. If either
check fails, search raises a clear error telling you to run `docugraph clear`
and re-index. Indexes created before this column existed are also rejected.

## Reranking (opt-in)

Reranking is **off by default**. When enabled, DocuGraph uses a local
multilingual cross-encoder, `BAAI/bge-reranker-v2-m3` (pure ONNX via
fastembed). Its weights are **~1.8 GB** and download on first use into
`FASTEMBED_CACHE_PATH`, so it is always an explicit opt-in — nothing large is
ever downloaded implicitly.

- CLI: `docugraph search "..." --rerank`
- MCP: call `hybrid_search` with `"rerank": true`

## Environment Variables

All options can be set via `DOCUGRAPH_SECTION__KEY=value`:

```bash
# Storage
export DOCUGRAPH_STORAGE__DATA_DIR=/custom/data/path

# Embeddings
export DOCUGRAPH_EMBEDDINGS__PROVIDER=fastembed
export DOCUGRAPH_EMBEDDINGS__MODEL=BAAI/bge-base-en-v1.5

# LLM (for the graph; requires API keys for cloud providers)
export DOCUGRAPH_LLM__PROVIDER=openai
export DOCUGRAPH_LLM__MODEL=gpt-4o-mini

# API keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# fastembed model cache (bge-small + any opt-in reranker weights)
export FASTEMBED_CACHE_PATH=~/.cache/fastembed

# Crawler
export DOCUGRAPH_CRAWLER__MAX_CONCURRENT=10
export DOCUGRAPH_CRAWLER__RATE_LIMIT=5.0
```

## Provider-Specific Configuration

### fastembed (default — local, no torch)

```yaml
embeddings:
  provider: fastembed
  model: BAAI/bge-small-en-v1.5   # fast, 384 dims
  # model: BAAI/bge-base-en-v1.5  # better quality, 768 dims
```

No API keys required. Models download automatically to
`FASTEMBED_CACHE_PATH`. `bge`-family models are given the retrieval
query-instruction prefix automatically for best relevance.

### Ollama (local)

```yaml
embeddings:
  provider: ollama
  model: nomic-embed-text

llm:
  provider: ollama
  model: llama3.2
```

Requires the `ollama` extra and a running server (`ollama serve`).

### OpenAI

```yaml
embeddings:
  provider: openai
  model: text-embedding-3-small

llm:
  provider: openai
  model: gpt-4o-mini
```

Requires the `cloud` extra and `export OPENAI_API_KEY=sk-...`.

### Cohere

```yaml
embeddings:
  provider: cohere
  model: embed-english-v3.0
```

Requires the `cloud` extra and `export COHERE_API_KEY=...`.

### Anthropic (graph LLM only)

```yaml
llm:
  provider: anthropic
  model: claude-haiku-4-5-20251001
```

Requires the `cloud` extra and `export ANTHROPIC_API_KEY=sk-ant-...`.

### sentence-transformers (optional, pulls in torch)

Not a managed dependency. If you install `sentence-transformers` yourself, it
works as a provider (`provider: sentence-transformers`) — but it brings torch
into your environment, which the default fastembed path avoids.

## The graph requires an LLM

The documentation knowledge graph (Graphiti/Kuzu) uses **embeddings served
locally by fastembed** — no embedding service needed. But Graphiti performs
**entity extraction with an LLM**, so `graph add` / `graph_query` need an LLM
configured (Ollama locally, or an OpenAI/Anthropic key). Without one, the
graph tools degrade gracefully and vector/keyword search are unaffected.

## Docker Configuration

Configuration is passed via environment variables:

```yaml
# docker/docker-compose.yml (excerpt)
services:
  cli:
    environment:
      - DOCUGRAPH_STORAGE__DATA_DIR=/data/docugraph
      - FASTEMBED_CACHE_PATH=/data/docugraph/models
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
```

The MCP server is not a compose service — run it with `docker run -i`
(see the [quickstart](quickstart.md#docker)).

## Configuration Precedence

1. Command-line arguments (highest priority)
2. Environment variables
3. Configuration file
4. Default values (lowest priority)
