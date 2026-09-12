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
  # Directory for all data (vectors, graphs, memory)
  data_dir: ~/.docugraph/data

  # Vector database backend
  vector_db: lancedb  # Currently only lancedb supported

  # Graph database backend (via Graphiti)
  graph_db: kuzu  # Currently only kuzu supported

# =============================================================================
# Embeddings Configuration
# =============================================================================
embeddings:
  # Embedding provider (local-first)
  # Options: sentence-transformers, ollama, openai, huggingface
  provider: sentence-transformers

  # Model name (provider-specific)
  model: all-MiniLM-L6-v2

  # Alternative models:
  # - BAAI/bge-base-en-v1.5 (better quality, 768 dims)
  # - nomic-ai/nomic-embed-text-v1.5 (excellent quality)
  # - text-embedding-3-small (OpenAI)

  # Embedding dimensions (auto-detected from model)
  dimensions: auto

  # Batch size for embedding generation
  batch_size: 32

  # Device for local models
  device: auto  # Options: auto, cpu, cuda, mps

# =============================================================================
# LLM Configuration (for entity extraction, reranking)
# =============================================================================
llm:
  # Provider selection
  # Options: auto, ollama, openai, anthropic
  provider: auto  # Tries: ollama -> openai -> anthropic

  # Model (provider-specific)
  model: auto

  # Explicit configuration:
  # provider: ollama
  # model: llama3.2
  #
  # provider: openai
  # model: gpt-4o-mini
  #
  # provider: anthropic
  # model: claude-3-haiku-20240307

  # Generation settings
  temperature: 0.0
  max_tokens: 2000

# =============================================================================
# Crawler Configuration
# =============================================================================
crawler:
  # Maximum concurrent requests
  max_concurrent: 5

  # Rate limiting (requests per second)
  rate_limit: 2.0

  # User agent string
  user_agent: DocuGraph-AI/1.0

  # Respect robots.txt
  respect_robots: true

  # Cache TTL in seconds (24 hours)
  cache_ttl: 86400

# =============================================================================
# Server Configuration
# =============================================================================
server:
  # API server host
  host: 0.0.0.0

  # API server port
  port: 8000

  # Enable debug mode
  debug: false

  # CORS origins (list)
  cors_origins:
    - "*"

# =============================================================================
# MCP Configuration
# =============================================================================
mcp:
  # Transport type
  transport: stdio  # Options: stdio, http

  # HTTP port (if transport is http)
  port: 3000
```

## Environment Variables

All configuration options can be set via environment variables using the pattern:
`DOCUGRAPH_SECTION__KEY=value`

### Examples

```bash
# Storage
export DOCUGRAPH_STORAGE__DATA_DIR=/custom/data/path

# Embeddings
export DOCUGRAPH_EMBEDDINGS__PROVIDER=openai
export DOCUGRAPH_EMBEDDINGS__MODEL=text-embedding-3-small

# LLM (requires API keys)
export DOCUGRAPH_LLM__PROVIDER=openai
export DOCUGRAPH_LLM__MODEL=gpt-4o-mini

# API Keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Server
export DOCUGRAPH_SERVER__HOST=0.0.0.0
export DOCUGRAPH_SERVER__PORT=9000
export DOCUGRAPH_SERVER__DEBUG=true

# Crawler
export DOCUGRAPH_CRAWLER__MAX_CONCURRENT=10
export DOCUGRAPH_CRAWLER__RATE_LIMIT=5.0
```

## Provider-Specific Configuration

### Sentence Transformers (Default)

```yaml
embeddings:
  provider: sentence-transformers
  model: all-MiniLM-L6-v2  # Fast, 384 dimensions
  # model: BAAI/bge-base-en-v1.5  # Better quality, 768 dims
  device: auto  # Uses GPU if available
```

No API keys required. Models are downloaded automatically.

### Ollama (Local LLM)

```yaml
embeddings:
  provider: ollama
  model: nomic-embed-text

llm:
  provider: ollama
  model: llama3.2
```

Requires Ollama to be running: `ollama serve`

### OpenAI

```yaml
embeddings:
  provider: openai
  model: text-embedding-3-small

llm:
  provider: openai
  model: gpt-4o-mini
```

Requires: `export OPENAI_API_KEY=sk-...`

### Anthropic

```yaml
llm:
  provider: anthropic
  model: claude-3-haiku-20240307
```

Requires: `export ANTHROPIC_API_KEY=sk-ant-...`

## Docker Configuration

When using Docker, configuration is passed via environment variables:

```yaml
# docker-compose.yml
services:
  docugraph:
    environment:
      - DOCUGRAPH_STORAGE__DATA_DIR=/data/docugraph
      - DOCUGRAPH_SERVER__HOST=0.0.0.0
      - DOCUGRAPH_SERVER__PORT=8000
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
```

## Configuration Precedence

1. Command-line arguments (highest priority)
2. Environment variables
3. Configuration file
4. Default values (lowest priority)
