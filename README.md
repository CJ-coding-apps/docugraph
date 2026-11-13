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
- **Measured retrieval**: a committed adversarial benchmark scores every search mode per failure category, with a regression gate (`docugraph bench`)

## Quick start

### Installation

A core install indexes and searches local documentation, and serves MCP:

```bash
pip install docugraph
```

Crawling web pages and the documentation knowledge graph are extras, so a user
who needs neither does not download a headless browser or an archived graph
database:

```bash
pip install 'docugraph[crawl]'   # crawl_url, docugraph crawl (also needs a browser — below)
pip install 'docugraph[graph]'   # graph_query, graph_add, docugraph graph (Kuzu via Graphiti)
pip install 'docugraph[graph,ladybug]'  # the same, on the opt-in LadybugDB backend (experimental)
pip install 'docugraph[all]'     # crawl + graph + cloud + ollama
```

Using a feature whose extra is absent fails with one sentence naming it — for
example `This feature needs the 'graph' extra: pip install 'docugraph[graph]'` —
rather than an import traceback.

**Requires Python 3.11–3.13.** CI runs the suite on all three, on Linux and on
x86_64 macOS. Python 3.14 is excluded deliberately, for two independent reasons:
`kuzu` 0.11.3 — the final release, and the version this package pins — publishes
no 3.14 wheels for macOS or Windows, and fastembed's 3.14 requirement
(`onnxruntime>=1.24.2`) cannot be satisfied alongside the `onnxruntime<1.24` cap
this package carries.

Or work from a checkout — clone this repository, then from its root:

```bash
# With uv (recommended; the extras match the Docker image)
uv sync --all-extras

# Or with pip
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,crawl,graph]'
```

First use downloads the default embedding model (`BAAI/bge-small-en-v1.5`,
~130 MB ONNX) and caches it under `FASTEMBED_CACHE_PATH`. Set that variable to
keep the weights somewhere durable; unset, fastembed puts them in a
`fastembed_cache` subdirectory of the system temp directory, which the OS is
free to clear.

**URL crawling needs the `crawl` extra and a browser that neither extra
includes.** Crawl4AI drives Playwright's Chromium, so install the extra and run
this once before using `docugraph crawl` or the `crawl_url` tool:

```bash
playwright install chromium
# On Linux, add --with-deps to install the browser's system libraries too:
#   playwright install --with-deps chromium
```

This is ~190 MB and is cached, so it is a one-off. `index local`, `index git`,
`search`, `memory` and `stats` do not need it — only fetching a URL does.

**URLs that an agent supplies are treated as untrusted.** `crawl_url` and
`index_git` refuse, before fetching or cloning, a value that is not `http(s)`
(or, for git, `https`/`ssh`), that begins with `-`, that is not a public
address, that leaves an allowlist you set, or that would write outside the
repository directory. Refusals are one line — `Refused: private_address — …` —
and appear in the tool result with `isError`. A site crawl is capped at 50 pages
per call. Set `crawler.allow_private_networks: true` to crawl an intranet, and
`crawler.allowed_domains` to narrow what may be reached; see
[docs/configuration.md](https://github.com/CJ-coding-apps/docugraph/blob/main/docs/configuration.md#url-policy).

### MCP integration with Claude Code (primary surface)

The MCP server is a console script, so where it lives depends on how you
installed the package. For an MCP client, an isolated tool install is the least
fuss: it puts the script somewhere stable that does not move with a project
directory.

Install both feature extras. `crawl_url` and `docugraph crawl` need `crawl`;
`graph_query`, `graph_add` and `docugraph graph` need `graph`. A core-only
install still starts the server and advertises all nine tools, so the gap only
appears when one of those features is used — hence both extras here:

```bash
# pipx
pipx install 'docugraph[crawl,graph]'

# or with uv:
uv tool install 'docugraph[crawl,graph]'
```

The `crawl` extra installs Playwright but not its browser, and the `playwright`
command lives inside the isolated environment rather than on your `PATH`. Run
it with that environment's own interpreter, so the browser revision matches the
Playwright version the extra installed rather than whatever a fresh resolution
picks today:

```bash
# pipx
"$(pipx environment --value PIPX_LOCAL_VENVS)/docugraph/bin/python" -m playwright install chromium

# uv
"$(uv tool dir)/docugraph/bin/python" -m playwright install chromium
```

That is ~190 MB into the per-user Playwright cache, downloaded once. Skip it if
you will only use `index local`, `index git`, `search`, `memory` and `stats`.

Then ask your shell where it landed:

```bash
which docugraph-mcp         # macOS, Linux
where docugraph-mcp         # Windows
```

and use that exact path in your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "docugraph": {
      "command": "<the path `which docugraph-mcp` printed>",
      "env": {}
    }
  }
}
```

An MCP client launches the server directly rather than through a shell, so it
does not inherit your `PATH`: the short form `"command": "docugraph-mcp"` fails
with `No such file or directory` even when that directory is on it. That is why
the path is spelled out.

If you would rather not manage an install, this is equivalent and works
straight from a checkout, with no path to update when it moves:

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

From a checkout, clone this repository and run `uv sync --all-extras`
(`pip install -e '.[dev,crawl,graph]'` works too). The script is then at
`.venv/bin/docugraph-mcp` (`Scripts\docugraph-mcp.exe` on Windows), and that is
the path to give the client.

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

`docugraph.__all__` is the supported surface. The models, `Config`, `UrlPolicy`
and `MissingExtraError` are always importable; the heavier names are resolved on
first access, so `import docugraph` stays cheap.

```python
import asyncio

from docugraph import Chunker, DocCrawler, VectorStore


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
  allow_private_networks: false   # opt in to intranet/loopback targets
  allowed_domains: []             # empty = any public host
```

Graphiti telemetry is **off** unless you set `GRAPHITI_TELEMETRY_ENABLED`
yourself; an explicit value is never overridden.

`provider: auto` resolves to **fastembed** (local ONNX) — nothing about your
environment changes that, so documents are not sent anywhere for embedding
unless you name a cloud provider. Set `provider: openai` (or `cohere`) to send
them there; that is the request, and it needs a model named with it: the entry
above is fastembed's own default and no cloud provider has one. Any field can be
overridden by environment variable, e.g. `DOCUGRAPH_EMBEDDINGS__MODEL` or
`DOCUGRAPH_EMBEDDINGS__PROVIDER`.

See [docs/configuration.md](https://github.com/CJ-coding-apps/docugraph/blob/main/docs/configuration.md) for the full reference,
including the embedding-model consistency guard and the graph LLM requirement.

## Network activity

DocuGraph runs locally, but these connections are possible — each only in the situation named:

- **Model downloads.** `fastembed` (the default embedder) downloads its ONNX model from Hugging
  Face on first use, and the reranker (about 2 GB) when you first enable it. Afterwards the model is
  read from the local cache. Nothing about your documents is sent.
- **Crawl and clone targets.** `crawl_url` fetches the pages you point it at, and `index_git`
  clones the repository URL you give it. The MCP tools treat those values as untrusted: pages must be
  `http`/`https` and resolve to a global address, git remotes must pass the same check, and
  `crawler.allow_private_networks` is the opt-in for an intranet or a local docs server.
- **A named LLM or embedding provider.** Only if you name one: `embeddings.provider: openai` (or
  `cohere`) sends the text being embedded there, and the graph tools send prompts to the configured
  LLM — OpenAI, Anthropic, or Ollama.
- **Ollama**, when configured — to your Ollama server, localhost by default.
- **The Ladybug FTS extension**, only from the two Ladybug commands. `docugraph graph fetch-fts`
  downloads one file from `extension.ladybugdb.com` and checks it against a pinned sha256;
  `docugraph graph install-fts --unverified` runs Ladybug's own `INSTALL`, which may also download —
  and, in some Ladybug versions, execute an installer in-process. Neither runs from an MCP tool, and
  opening a graph never installs or downloads anything.
- **Graphiti telemetry**, to PostHog, **off unless you opt in** with
  `GRAPHITI_TELEMETRY_ENABLED=true`.

CLI crawls and clones are user-directed: they are exempt from the scheme, DNS and address rules,
but the repository-name containment and the git hardening apply to every caller.

## The documentation knowledge graph

DocuGraph builds a temporal entity/relationship graph over your docs (Kuzu,
via [Graphiti](https://github.com/getzep/graphiti)). Embeddings for the graph
are served by the same torch-free fastembed adapter — **no embedding service
required**. However, Graphiti performs entity extraction with an **LLM**, so
`graph add` / `graph_query` need an LLM configured (Ollama locally, or an
OpenAI/Anthropic key). Without one, the graph tools degrade gracefully with a
clear message; vector and keyword search are unaffected.

### Optional: LadybugDB backend (experimental)

The graph runs on Kùzu 0.11.3 through
[Graphiti](https://github.com/getzep/graphiti). Kùzu is archived upstream, so
0.4.0 adds LadybugDB as a second backend that has to be enabled explicitly.
**Kùzu remains the default**, a new install uses it, and nothing changes for
anyone who does not set `graph_db`.

```bash
pip install 'docugraph[graph,ladybug]'
docugraph graph fetch-fts            # verified FTS extension for this version and platform
export DOCUGRAPH_STORAGE__GRAPH_DB=ladybug   # or graph_db: ladybug in the config file
```

Ladybug stores the graph in a single file, `graphiti.lbdb`, beside Kùzu's
`graphiti.kuzu` directory. A graph built with Kùzu cannot be opened by Ladybug;
copy it explicitly first — the command verifies per-table counts and does not
change your setting:

```bash
docugraph graph migrate-to-ladybug
```

With `graph_db: ladybug` and an old Kùzu graph still present, every graph
operation refuses with one sentence naming the migrate command instead of
starting an empty graph beside it. To go back, set `graph_db: kuzu` — the Kùzu
graph was never touched.

**One engine per process.** The two engines register the same native type
names, so once either one has been loaded the other cannot be imported in the
same Python process (`generic_type: type "Database" is already registered!`),
whichever is imported first. DocuGraph imports only the driver of the configured
backend, so ordinary use never meets this; it matters if your own code imports
`kuzu` next to a Ladybug-backed graph, or if a long-running process tries to
switch `graph_db` at runtime — use a new process for the other engine. The
migration command runs each engine in its own child process for this reason.

**Full-text search is an extension on Ladybug.** DocuGraph downloads the file
itself and loads it by path: `graph fetch-fts` checks it against a SHA-256 pinned
in this package for your Ladybug version and platform, and confirms it loads
before keeping it. That pin makes tampering detectable; it is not an upstream
signature. If no verified file is available (nothing fetched, no pin, or a
mismatch), the backend refuses to start and states the options — including
`docugraph graph install-fts --unverified` for a platform with no pin, an
interactive command that says plainly what Ladybug's `INSTALL` does. MCP tools
never download anything.

**Promotion criteria.** Ladybug becomes the default in a later release only
when all of these hold: (1) graphiti 0.30.2's 25 Kùzu-passing tests pass under
this package's Ladybug driver in CI; (2) the Ladybug graph job is green on every
leg; (3) the migration job passes; (4) CJ has used the backend for real work for
a period he chooses, with no unresolved defects; and (5) no extension dependency
remains unaddressed, or a safe install mechanism exists for each one.

## Retrieval benchmark

Retrieval quality is measured, not asserted. `benchmarks/retrieval/` commits a
hand-authored documentation corpus for a fictional SDK plus graded relevance
judgments, built adversarially around seven failure modes: exact-token lookups
(error codes, parameter names), paraphrase queries with no lexical overlap,
keyword-stuffed distractor pages, near-sibling semantic distractors, version-
pinned questions, acronym/synonym gaps, and "X without Y" constraint queries.
Each query records the trap it sets in a `design_note`, and a manifest pins the
dataset by hash — editing it is a deliberate act that invalidates baselines.
The corpus holds 91 documents and 88 queries: 56 eight-per-failure-mode
queries, plus 16 multi-hop and 16 aggregation queries added for the graph
benchmark. Support-set queries declare a `required` set — every document a
complete answer needs — which `set_recall@k` and `full_support@k` score; the
same set is recorded in `gold_subgraph.json` together with the joining entity
and required facts. The keyword-trap distractors are built to cover many
*distinct* query terms rather than repeat one (BM25 saturates a repeated term,
so a one-token trap does not bind), and the semantic-distractor queries share
few exact tokens with their target so that only meaning — not lexical overlap —
can recover them.

```bash
# Score every arm (vector-only, BM25-only, LanceDB-native hybrid, RRF, weighted)
# against a temporary store built from the committed corpus:
docugraph bench run --json report.json

# Hold a report against a committed baseline; exits 1 when nDCG@10 drops by
# more than the tolerance, overall or in any single category:
docugraph bench compare benchmarks/retrieval/baselines/bge-small-en-v1.5.json report.json
```

Reported metrics are recall@1/5/10, MRR@10, and nDCG@10 (graded, linear gain),
macro-averaged overall and per category. The per-category table is the point:
a fusion change that wins on average while losing exact-code lookup shows up
as the regression it is. Baselines are per embedding model and live in
`benchmarks/retrieval/baselines/`; the `Retrieval benchmark` workflow re-runs
the suite weekly against the committed baseline.

Honest scope: the corpus is small and purpose-built, so scores are comparable
across docugraph changes — not with BEIR or any external leaderboard. The
first thing this benchmark caught predates it: before v0.2.0 no FTS index was
ever created, so "keyword" search silently returned vector results and
RRF fused the vector ranking with itself. The benchmark's `fts` arm and the
hybrid arm's refuse-to-score-on-fallback check exist so that class of silent
degradation cannot come back unnoticed.

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

## Status and support

DocuGraph is pre-1.0. It uses semantic versioning for what that can mean before
1.0: a minor release may break an interface, and the CHANGELOG's breaking
changes are listed first.

**Supported interfaces** are the MCP tools in the table above, the CLI, the
names in `docugraph.__all__`, and the settings documented in
[docs/configuration.md](https://github.com/CJ-coding-apps/docugraph/blob/main/docs/configuration.md).
Anything else under `docugraph.*` is internal and may move between minor
releases.

**Tested platforms**: Linux and macOS 15 on x86_64, with Python 3.11, 3.12 and
3.13, in both the core-only and the all-extras install. Other platforms and
architectures — Windows, Linux ARM, Apple silicon — are not tested here and are
not claimed.

**Kùzu** is archived upstream; this package pins the final release (0.11.3) and
it is only installed by the `graph` extra. The pin is a migration boundary: see
[KuzuDB version](#kuzudb-version).

**Security reports** go to the process in
[docs/SECURITY.md](https://github.com/CJ-coding-apps/docugraph/blob/main/docs/SECURITY.md),
not to a public issue.

## Acknowledgements

This project uses Crawl4AI (https://github.com/unclecode/crawl4ai) for web data extraction.

- [fastembed](https://github.com/qdrant/fastembed) — local ONNX embeddings, and the optional reranker.
- [Graphiti](https://github.com/getzep/graphiti) — temporal knowledge-graph extraction.
- [LanceDB](https://github.com/lancedb/lancedb) — the embedded vector store.
- [Kùzu](https://github.com/kuzudb/kuzu) — the embedded graph database.

## License

Apache-2.0 — see `LICENSE`, and `NOTICE` for the Crawl4AI attribution.
