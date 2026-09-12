# API Reference

DocuGraph AI provides a REST API for programmatic access to all features.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required. For production deployments, consider adding a reverse proxy with authentication.

---

## Health & Stats

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

### GET /v1/stats

Get indexing statistics.

**Response:**
```json
{
  "total_chunks": 1234,
  "data_directory": "/home/user/.docugraph/data",
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_provider": "sentence-transformers"
}
```

---

## Search

### POST /v1/search

Search indexed documentation using vector similarity.

**Request:**
```json
{
  "query": "how to handle async errors",
  "top_k": 5,
  "filter": null
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| query | string | Yes | - | Search query text |
| top_k | integer | No | 5 | Number of results (1-100) |
| filter | object | No | null | Filter criteria |

**Response:**
```json
{
  "query": "how to handle async errors",
  "results": [
    {
      "chunk": {
        "id": "chunk-123",
        "document_id": "doc-456",
        "content": "Error handling in async code...",
        "start_char": 0,
        "end_char": 500,
        "metadata": {
          "source_url": "https://example.com/docs",
          "title": "Error Handling"
        },
        "created_at": "2024-01-15T10:30:00Z"
      },
      "score": 0.89,
      "highlights": []
    }
  ],
  "total": 1,
  "took_ms": 45.2
}
```

### POST /v1/search/hybrid

Hybrid search combining vector, keyword, and optionally graph search.

**Request:**
```json
{
  "query": "authentication patterns",
  "top_k": 10,
  "mode": "hybrid",
  "fusion": "rrf",
  "include_graph": false
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| query | string | Yes | - | Search query |
| top_k | integer | No | 10 | Number of results |
| mode | string | No | "hybrid" | Search mode: vector, keyword, hybrid, all |
| fusion | string | No | "rrf" | Fusion strategy: rrf, weighted, interleaved |
| include_graph | boolean | No | false | Include knowledge graph search |

**Response:**
```json
{
  "query": "authentication patterns",
  "results": [
    {
      "chunk": { ... },
      "score": 0.92,
      "sources": ["vector", "keyword"],
      "vector_score": 0.88,
      "keyword_score": 0.95,
      "graph_score": null
    }
  ],
  "total": 5,
  "took_ms": 78.5
}
```

---

## Indexing

### POST /v1/crawl

Crawl a URL and index its content.

**Request:**
```json
{
  "url": "https://fastapi.tiangolo.com/tutorial/",
  "max_pages": 10,
  "pattern": "/tutorial/"
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| url | string | Yes | - | URL to crawl |
| max_pages | integer | No | 1 | Maximum pages to crawl (1-1000) |
| pattern | string | No | null | URL pattern filter (regex) |

**Response:**
```json
{
  "url": "https://fastapi.tiangolo.com/tutorial/",
  "pages_crawled": 10,
  "chunks_indexed": 156,
  "total_chunks": 1234
}
```

### POST /v1/index/local

Index local documentation files.

**Request:**
```json
{
  "path": "/path/to/docs",
  "recursive": true,
  "patterns": ["*.md", "*.rst"]
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| path | string | Yes | - | Path to directory |
| recursive | boolean | No | true | Recursively index subdirectories |
| patterns | array | No | null | File patterns to include |

**Response:**
```json
{
  "source": "/path/to/docs",
  "files_indexed": 25,
  "chunks_indexed": 340,
  "total_chunks": 1574
}
```

### POST /v1/index/git

Clone and index a git repository.

**Request:**
```json
{
  "url": "https://github.com/org/repo.git",
  "name": "my-repo",
  "ref": "main",
  "shallow": true,
  "include_code": false
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| url | string | Yes | - | Git repository URL |
| name | string | No | null | Custom name for the repository |
| ref | string | No | null | Branch, tag, or commit to checkout |
| shallow | boolean | No | false | Shallow clone (depth=1) |
| include_code | boolean | No | false | Include code files |

**Response:**
```json
{
  "source": "my-repo @ main",
  "files_indexed": 45,
  "chunks_indexed": 520,
  "total_chunks": 2094
}
```

### GET /v1/repos

List all cloned repositories.

**Response:**
```json
[
  {
    "name": "fastapi",
    "url": "https://github.com/tiangolo/fastapi.git",
    "branch": "master",
    "commit": "abc123def456",
    "local_path": "/home/user/.docugraph/data/repos/fastapi"
  }
]
```

### DELETE /v1/index

Clear all indexed content.

**Response:**
```json
{
  "status": "cleared"
}
```

---

## Memory

### POST /v1/memory

Store a value in agent memory.

**Request:**
```json
{
  "key": "auth_approach",
  "value": "JWT with refresh tokens",
  "repository": "my-project",
  "branch": "main"
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| key | string | Yes | - | Memory key |
| value | any | Yes | - | Value to store (any JSON type) |
| repository | string | No | "default" | Repository scope |
| branch | string | No | "main" | Branch scope |

**Response:**
```json
{
  "key": "auth_approach",
  "value": "JWT with refresh tokens",
  "repository": "my-project",
  "branch": "main",
  "created_at": "2024-01-15T10:30:00Z",
  "accessed_at": "2024-01-15T10:30:00Z"
}
```

### GET /v1/memory/{key}

Retrieve a value from agent memory.

**Query Parameters:**
- `repository` (default: "default")
- `branch` (default: "main")

**Response:**
```json
{
  "key": "auth_approach",
  "value": "JWT with refresh tokens",
  "repository": "my-project",
  "branch": "main"
}
```

### GET /v1/memory

List all memory keys for a scope.

**Query Parameters:**
- `repository` (default: "default")
- `branch` (default: "main")

**Response:**
```json
{
  "keys": ["auth_approach", "database_config", "api_version"],
  "repository": "my-project",
  "branch": "main",
  "total": 3
}
```

### DELETE /v1/memory/{key}

Delete a memory entry.

**Query Parameters:**
- `repository` (default: "default")
- `branch` (default: "main")

**Response:**
```json
{
  "key": "auth_approach",
  "deleted": true
}
```

---

## Knowledge Graph

### POST /v1/graph/add

Add content to the knowledge graph. Extracts entities and relationships using LLM.

**Request:**
```json
{
  "content": "FastAPI is a modern Python web framework built on Starlette and Pydantic.",
  "name": "fastapi-overview",
  "group_id": "docs",
  "source_type": "text"
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| content | string | Yes | - | Content to extract entities from |
| name | string | No | null | Episode name |
| group_id | string | No | "default" | Group ID for scoping |
| source_type | string | No | "text" | Content type (text/json) |

**Response:**
```json
{
  "episode_uuid": "ep-123",
  "entities_count": 4,
  "relationships_count": 2,
  "entities": [
    {"name": "FastAPI", "type": "concept"},
    {"name": "Starlette", "type": "concept"},
    {"name": "Pydantic", "type": "concept"},
    {"name": "Python", "type": "concept"}
  ]
}
```

### POST /v1/graph/search

Search the knowledge graph.

**Request:**
```json
{
  "query": "What is FastAPI built on?",
  "group_ids": ["docs"],
  "num_results": 10
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| query | string | Yes | - | Search query |
| group_ids | array | No | null | Group IDs to filter by |
| num_results | integer | No | 10 | Maximum results (1-100) |

**Response:**
```json
{
  "query": "What is FastAPI built on?",
  "results": [
    {
      "fact": "FastAPI is built on Starlette",
      "valid_at": "2024-01-15T10:30:00Z"
    },
    {
      "fact": "FastAPI uses Pydantic for validation",
      "valid_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 2
}
```

### DELETE /v1/graph

Clear the knowledge graph.

**Response:**
```json
{
  "status": "cleared"
}
```

---

## Error Responses

All endpoints return standard error responses:

```json
{
  "detail": "Error message here"
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Resource doesn't exist |
| 422 | Validation Error - Invalid request body |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Requires LLM for graph features |
