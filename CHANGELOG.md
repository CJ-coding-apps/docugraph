# Changelog

Notable changes to docugraph. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/spec/v2.0.0.html).

docugraph is pre-1.0: a minor release may change a tool's behaviour or a setting's meaning, and a
patch release will not. Breaking changes are listed as breaking either way, because a reader of this
file is entitled to know which of their own setup stops working. The supported interfaces are the MCP
tools, the CLI commands and the configuration settings; the Python modules behind them are not a
public API.

## [Unreleased]

### Security

- **`crawl_url` and `index_git` now judge model-supplied URLs before acting.** Pages must be
  `http`/`https`; git remotes must be `https`, `ssh`, or `user@host:path`. A value beginning with `-`
  is refused as `option_like` rather than passed to git. A host is refused unless it resolves to a
  global address (`private_address` covers loopback, link-local, `0.0.0.0` and the metadata
  address), unless `crawler.allow_private_networks` is on. `crawler.allowed_domains` narrows what may
  be reached. A refusal is a tool error whose text is `Refused: <reason> — <detail>`.
- **Redirects are enforced hop by hop.** Chromium follows a redirect without re-entering a Playwright
  route handler, so each `Location` is resolved and judged before the next request is made. A page
  whose final URL is refused is dropped and reported (`Dropped N page(s) by policy:`) and never
  indexed.
- **Service workers cannot bypass the route handler.** The page-context hook removes
  `navigator.serviceWorker` in every frame before page scripts run. Websockets are closed.
- **`index_git` cannot write outside the repository directory.** The name — or the one derived from
  the URL — must be a single path component, and the resolved destination must stay inside the
  configured repos directory. An escaping name is refused as `path_escape`, not renamed.
- **Git runs hardened under the policy.** `--` precedes the URL and path, refs are validated against
  `git-check-ref-format`, and clones carry `protocol.file.allow=never`, `protocol.ext.allow=never`
  and `http.followRedirects=false`.
- **A site crawl is capped at 50 pages per MCP tool call**, and `max_pages` below 1 or not an integer
  is an error rather than a crawl.
- **Graphiti telemetry is off by default.** `GRAPHITI_TELEMETRY_ENABLED` is set to `false` when
  absent; an explicit value is respected.

### Fixed

- `docugraph index git` now exits non-zero with `Error: Refused: …` when a repository name or ref is
  refused. Previously a refused path reached the generic `RuntimeError` branch, printed a note, and
  exited 0.

### Changed

- `CrawlerConfig` gains `allow_private_networks` (default `false`) and `allowed_domains` (default
  empty).

## [0.2.0] - 2026-10-23

### Added

- A retrieval-quality benchmark with a committed corpus and a regression gate. It scores
  vector-only, BM25-only, LanceDB-native hybrid, RRF and weighted arms on the same queries and fails
  when nDCG@10 drops past the tolerance.

### Fixed

- The full-text index was never built, so keyword search silently returned nothing; it is now built
  and BM25 is real. The hybrid path records when it falls back to the vector leg.

## [0.1.1] - 2026-10-08

Maintenance release — no changes to the package's code or behaviour.

### Fixed

- The release workflow publishes the container image,
  `ghcr.io/cj-coding-apps/docugraph:v0.1.1` (and `:latest`). 0.1.0 has no image: the workflow built
  the tag from the repository name with its original capitalisation, which GHCR rejects.

### Added

- `pyproject.toml` declares project URLs, so the PyPI page links to the repository, issues and
  releases.

## [0.1.0] - 2026-10-08

First public release, on PyPI.

### Added

- Documentation RAG for coding agents: crawl or index documentation, build a documentation knowledge
  graph and a repository/branch-scoped agent memory, and serve all of it to MCP clients over stdio.
  The surface is nine tools (`crawl_url`, `index_git`, `search_docs`, `hybrid_search`, `graph_add`,
  `graph_query`, `memory_store`, `memory_recall`, `get_stats`) and a CLI for out-of-band index
  bootstrapping.
- Embeddings run in-process on ONNX: `embeddings.provider: auto` resolves to fastembed and never
  consults the environment, so documents are not sent anywhere to be embedded unless a cloud provider
  is named.
- Optional extras: `cloud` (OpenAI, Anthropic and Cohere SDKs), `ollama` (local LLM) and `dev`
  (test and lint tooling).

### Notes

- Python 3.11–3.13 on Linux. `kuzu` is pinned `==0.11.3`, the final release upstream, and is treated
  as a migration boundary rather than a floor.
- URL crawling needs Playwright's Chromium (`playwright install chromium`).
