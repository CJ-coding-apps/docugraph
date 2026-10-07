# Configuration Reference

DocuGraph AI can be configured via:
1. Configuration file (`~/.docugraph/config.yaml`)
2. Environment variables (prefixed with `DOCUGRAPH_`)
3. Command-line arguments

> **Every model name in this document is an example.** DocuGraph ships no cloud
> model id: nothing in `src/` names one, and `tests/unit/test_no_model_ids_in_source.py`
> fails the build if that changes. A cloud provider is used only when you name
> both the provider and the model; naming the provider alone is an error that
> says which setting is missing. The local backends are the exception and do
> carry a default each, because their models are a download away rather than an
> account away — those defaults are listed below.

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
  #   auto                  - fastembed -> sentence-transformers. Local only:
  #                           an OPENAI_API_KEY in the environment does not
  #                           change this, so no document text leaves the
  #                           machine unless you name a cloud provider below.
  #   fastembed             - local, pure ONNX, no torch (default backend)
  #   openai                - OpenAI API (needs the 'cloud' extra + key)
  #   ollama                - local Ollama server (needs the 'ollama' extra)
  #   cohere                - Cohere API (needs the 'cloud' extra + key)
  #   sentence-transformers - only if you have it installed (pulls in torch)
  provider: auto

  # Model name (provider-specific). Leave this out and each *local* backend
  # falls back to its own model -- fastembed to BAAI/bge-small-en-v1.5 (384
  # dims, ~130 MB on first use), sentence-transformers to all-MiniLM-L6-v2,
  # Ollama to nomic-embed-text (pull it first).
  #
  # There is no default for a *cloud* provider, and leaving `model` unset with
  # `provider: openai` or `cohere` is an error rather than a guess. An embedding
  # model fixes the vector space of the whole index, so a name this package
  # picked would be a silent, hard-to-undo choice made for you -- and the wrong
  # one, since the local default is a name no cloud API has heard of.
  #
  # Examples (any model your provider offers will do; these are not defaults):
  # - BAAI/bge-base-en-v1.5     (fastembed, better quality, 768 dims)
  # - text-embedding-3-small    (OpenAI; set provider: openai)
  model: BAAI/bge-small-en-v1.5

  # Embedding dimensions. Left null, they are measured from the first embedding
  # the model returns, which is the model's own answer to the question. Set it
  # only to skip that first call; a wrong value here does not fail, it writes
  # vectors of the wrong width, so the measured path is the safer one.
  dimensions: null

  # Batch size for embedding generation
  batch_size: 32

# =============================================================================
# LLM Configuration (required only for the documentation knowledge graph)
# =============================================================================
llm:
  # Provider selection. Options: auto, ollama, openai, anthropic
  provider: auto      # Tries: ollama -> openai -> anthropic

  # The model used for extraction and reranking.
  #
  # `auto` (the shipped value) and "unset" both mean the same thing: resolve a
  # model from whatever provider is actually usable. Ollama contributes its own
  # default (llama3.2, pulled on request); a cloud provider contributes nothing,
  # because it has no default here -- for those, this must name a model.
  #
  # A cloud model is checked against the provider's own models-list endpoint
  # before the first call, so a name that has been retired fails with
  # "not found; available: ..." and the list of names that do exist, rather than
  # as a 404 from deep inside the graph layer. If the endpoint cannot be
  # reached, the model is reported as unverified rather than assumed good.
  #
  # Examples (not defaults; set one to use that provider):
  # provider: ollama
  # model: llama3.2
  #
  # provider: openai
  # model: gpt-4o-mini
  #
  # provider: anthropic
  # model: claude-haiku-4-5-20251001
  model: auto

  # A second, cheaper model for the summarisation and edge-deduplication passes
  # the knowledge graph runs. Unset means "the same as `model`", which is what it
  # did before this setting existed -- the difference is that the fallback is now
  # whatever you chose rather than a model name compiled into the package.
  small_model: null

  # Unset by default, and sent to the provider only when you set it. That is not
  # tidiness: OpenAI's reasoning models (o-series, GPT-5) reject any temperature
  # other than their own fixed default, so a shipped 0.0 made every call to one
  # fail with `unsupported value`. Set `0.0` explicitly for deterministic local
  # output. On Anthropic this setting is accepted and not sent; see below.
  temperature: null

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
fastembed). Its weights are **about 2 GB** and download on first use into
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

# LLM (for the graph; requires API keys for cloud providers).
# Naming a cloud provider means naming a model with it: there is no default.
export DOCUGRAPH_LLM__PROVIDER=openai
export DOCUGRAPH_LLM__MODEL=<a model id from your provider's docs>

# API keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# fastembed model cache (bge-small + any opt-in reranker weights).
# Worth setting: without it fastembed uses a `fastembed_cache` subdirectory
# of the system temp directory, which the OS is free to clear.
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
  # model: BAAI/bge-base-en-v1.5  # an example: better quality, 768 dims
```

No API keys required. Omitting `model` uses the backend's own default,
`BAAI/bge-small-en-v1.5` (fast, 384 dims). Models download automatically to
`FASTEMBED_CACHE_PATH`. `bge`-family models are given the retrieval
query-instruction prefix automatically for best relevance.

### Ollama (local)

```yaml
embeddings:
  provider: ollama
  # model: nomic-embed-text   # the backend's default; an example, not a requirement

llm:
  provider: ollama
  # model: llama3.2           # the backend's default, pulled on request
```

Requires the `ollama` extra and a running server (`ollama serve`). Both models
above are Ollama's own defaults for their slots and need not be written down.

### OpenAI

```yaml
embeddings:
  provider: openai
  model: text-embedding-3-small   # an example — required, there is no default

llm:
  provider: openai
  model: gpt-4o-mini              # an example — required, there is no default
```

Requires the `cloud` extra and `export OPENAI_API_KEY=sk-...`. Both model names
are required and checked against OpenAI's models list; the examples above are
neither exhaustive nor maintained by this package.

### Cohere

```yaml
embeddings:
  provider: cohere
  model: embed-english-v3.0   # an example — required, there is no default
```

Requires the `cloud` extra and `export COHERE_API_KEY=...`. Cohere is reachable
here for embeddings only: there is no Cohere LLM provider.

### Anthropic (graph LLM only)

```yaml
llm:
  provider: anthropic
  model: claude-haiku-4-5-20251001   # an example — required, there is no default
```

Requires the `cloud` extra and `export ANTHROPIC_API_KEY=sk-ant-...`.

`temperature` is **ignored for Anthropic.** The 1.x SDK dropped the
parameter from its request type, so it is not sent: the setting is accepted in
your config and has no effect on the request, rather than raising and failing
the call. OpenAI and Ollama honour it as usual. If you need deterministic
output on Anthropic, that is a property of the model and the prompt, not of a
sampling setting here.

Because this provider's `temperature` never reaches the request, leaving it
unset costs nothing here — but it is still worth leaving unset everywhere, so
that switching providers does not silently start sending a sampling value the
new model rejects.

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
