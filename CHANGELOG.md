# Changelog

Notable changes to docugraph. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/spec/v2.0.0.html).

docugraph is pre-1.0: a minor release may change a tool's behaviour or a setting's meaning, and a
patch release will not. Breaking changes are listed as breaking either way, because a reader of this
file is entitled to know which of their own setup stops working. The supported interfaces are the MCP
tools, the CLI commands, the configuration settings and the names in `docugraph.__all__`; every other
module under `docugraph.*` is internal.

## [Unreleased]

## [0.5.0] - 2026-11-13

### Added

- **A retrieval-quality benchmark, and its committed baseline.**
  `python -m docugraph.evals.graph_report` scores the retrieval arms (vector, FTS, the RRF hybrids,
  the weighted fusion and the graph variants) over a 104-query corpus and writes
  `benchmarks/retrieval/baselines/bge-small-en-v1.5-graph.json`; the corpus and the extraction
  fixture it reads are committed under `benchmarks/retrieval/`, so the bench runs with no key and no
  network reachability to an LLM. `python -m docugraph.evals.build_graph_fixture` builds that
  fixture. CI runs the bench on every push and asserts that two fresh processes in the same job are
  byte-identical.

### Fixed

- **Extraction with an Ollama model writes nodes again.** Graphiti's client defaulted to
  `json_schema` output enforcement, which Ollama ignores for the configured model; extraction failed
  with a pydantic `ValidationError` before any node was written. The Ollama path now requests
  `json_object` mode, where Graphiti appends the response schema to the prompt.
- **A model that repeats the schema is retried once, then reported.** A `properties` echo envelope is
  unwrapped (only when the target model declares no `properties` field and the envelope carries
  nothing but schema keys), one corrective message is sent, and a second echo raises
  `EmptyResponseError` instead of ingesting a fabricated value.

### Notes

- **The multi-hop retrieval gate is NOT MET, and the ceiling is recorded with it.** The
  pre-registered comparison of `rrf+graph-2hop` against `rrf` measured nDCG@10 −0.0056
  [−0.0194, +0.0050] and set_recall@10 +0.0000, against thresholds of +0.05 and +0.10. The
  comparator already reaches set_recall@10 = 1.000, full_support@10 = 1.000 and nDCG@10 = 0.984 on
  the 32 multi-hop queries, so the maximum attainable gain was +0.016 nDCG@10 and 0 set_recall@10:
  the set is saturated at k=10 over 107 documents, which is not evidence that graph retrieval cannot
  help multi-hop. No headroom rule was pre-registered, so the outcome is not relabelled. Nothing is
  built on the graph leg in this release.

## [0.4.0] - 2026-11-09

### Added

- **LadybugDB as an opt-in graph backend.** `storage.graph_db` now accepts `kuzu` (default, unchanged)
  or `ladybug`, installed with `pip install 'docugraph[graph,ladybug]'`. On Ladybug, full-text search
  needs its FTS extension: `docugraph graph fetch-fts` downloads it and verifies it against a checksum
  pinned in this package for your Ladybug version and platform. If no verified file is available the
  backend refuses to start with one sentence naming the fix, rather than running with degraded search.
- **`docugraph graph migrate-to-ladybug`.** Copies an existing Kùzu graph into a Ladybug database file,
  verifies per-table row counts, and leaves the Kùzu graph and the configured backend untouched.
- **`docugraph graph install-fts --unverified`.** The explicit, interactive fallback for a version or
  platform with no pinned checksum. It states what it does and what Ladybug's `INSTALL` may execute
  before it runs. The MCP tools never perform either download.

## [0.3.0] - 2026-11-08

### Breaking

- **A core install no longer includes the crawler or the knowledge graph.** `crawl4ai`, `playwright`,
  `graphiti-core[kuzu]` and `kuzu` moved out of `dependencies` and into two extras:
  `docugraph[crawl]` and `docugraph[graph]`. `docugraph[all]` composes all four user-facing extras.
  A core install still indexes local files, clones and indexes git repositories, searches, keeps
  agent memory and serves MCP.
- **A feature whose extra is absent fails with one sentence, not a traceback.**
  `This feature needs the 'crawl' extra: pip install 'docugraph[crawl]', then: playwright install
  chromium` (and the same form for `graph`). The MCP server returns it as a tool error with
  `isError`; the CLI prints it to stderr and exits 1; `docugraph --help` and `import docugraph` work
  with core only.
- **`crawler.allow_private_networks = true` no longer permits link-local or cloud-metadata
  addresses.** Ordinary private ranges (10.0.0.0/8, 192.168.0.0/16, loopback) are admitted as before.

### Security

- **A private IPv4 is judged through every IPv6 spelling that can carry it.** 0.2.1 checked the IPv6
  wrapper's `is_global` flag, which reports global for `64:ff9b::10.0.0.5` and `::10.0.0.5`; the
  embedded IPv4 is now extracted — IPv4-mapped, IPv4-compatible, both NAT64 prefixes, 6to4 and
  Teredo (server and client) — and refused when it is not global. This closes a default-policy gap
  in 0.2.1; it was found in review and no advisory is reissued.
- **Link-local and cloud-metadata addresses are refused whatever `allow_private_networks` says.**
  169.254.0.0/16, fe80::/10, `fd00:ec2::254` and `100.100.100.200` are no longer reachable by turning
  the intranet setting on.
- **A crawl refuses to run unguarded.** If the installed crawl4ai is missing the hook, retry switch
  or browser-less fallback the guards pin, the tool returns `Refused: unsupported_crawler` before
  the browser starts; an error while registering the hook becomes the same refusal, not a crash.

### Added

- The browser security suites run in CI against a real Chromium, and a missing browser fails the job
  rather than skipping the tests (`DOCUGRAPH_REQUIRE_BROWSER=1`).
- **A declared Python surface.** `docugraph.__all__` lists the supported names; the heavier ones are
  resolved on first access, so `import docugraph` does not import an extra. The package ships
  `py.typed`, and the README states the supported interfaces, tested platforms and version policy.
- `MissingExtraError` (an `ImportError` subclass) and `docugraph._optional.require`, the single place
  that names an absent extra.

### Changed

- `crawl4ai` is bounded to `>=0.9.3,<0.10` and `playwright` is declared at `>=1.48,<1.63` — the
  versions whose hook table and routing API the guards are written against.
- `fastembed` is capped at `<0.9`, `pyarrow` at `<26` and `pandas` at `<4`, each with the reason and
  the condition for lifting it recorded beside it. `onnxruntime<1.24` and `lancedb<0.26` stay.
- `typer` was an unused dependency and is removed.
- Package metadata is a PEP 639 license expression (`Apache-2.0`), with the redundant `License ::`
  classifier removed, and classifiers and project URLs that name the supported platforms, the
  changelog file and the security policy.
- `tests/` is linted by ruff alongside `src/` and `scripts/`.

### Fixed

- A redirected page is recorded under its **final** URL and its relative links are resolved against
  that URL, in both crawl paths. Previously a relative link on a redirected page was resolved
  against the requested URL (LOW-5 from the 0.2.1 review).

## [0.2.1] - 2026-11-07

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
  `navigator.serviceWorker` in every frame before page scripts run. Websockets are closed. Routing is
  registered on the browser context, so popups, frames and workers are judged by the same handler.
- **crawl4ai's anti-bot replay cannot escape the guards.** Its retry navigates through the same hooked
  context and is routed; its browser-less `fallback_fetch_function` would run outside every guard, so
  the retry and the fallback are pinned off in both crawl paths.
- **An existing clone is updated only from the origin that was judged.** When the clone destination
  already exists, its `origin` must equal the requested URL and pass the same policy; a mismatch is
  refused as `path_escape`. Pull and the fetch fallback carry the same protocol and redirect lockdown
  as the clone.
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
- `crawl_site` now counts pages it **attempts**, not pages that produced a document, so a page that
  fails to scrape still consumes the budget. This changes the CLI too: `docugraph crawl --max-pages`
  now limits attempted pages.
- Under a URL policy the crawl cache is not read, so a page cached by an earlier unpoliced crawl is
  never served without its final URL being judged. Cached documents are still written.

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

[Unreleased]: https://github.com/CJ-coding-apps/docugraph/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/CJ-coding-apps/docugraph/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/CJ-coding-apps/docugraph/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/CJ-coding-apps/docugraph/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/CJ-coding-apps/docugraph/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/CJ-coding-apps/docugraph/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/CJ-coding-apps/docugraph/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/CJ-coding-apps/docugraph/releases/tag/v0.1.0
