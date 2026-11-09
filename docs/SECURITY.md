# Security policy

## Reporting a vulnerability

Report suspected vulnerabilities through GitHub's
[private vulnerability reporting](https://github.com/CJ-coding-apps/docugraph/security/advisories/new),
not as a public issue. If you cannot use that form, open an issue that says only that you have a security
report to share, and wait for a maintainer to open a private channel — please do not put the details in
the issue.

Useful to include: the version, whether the MCP server or the CLI is involved, what you did, and what
happened. A reproduction is worth more than a description.

## Supported versions

DocuGraph is pre-1.0. Only the most recent release is supported; fixes land there and are not backported.
Please reproduce against the latest version before reporting.

## What this project does with your data

Worth stating plainly, because it decides what is and is not a vulnerability here.

- **The index is local.** The document index, the knowledge graph and agent memory are built and stored
  on the machine that runs the command.
- **The MCP server runs over stdio** as a child process of the client that starts it. It does not open a
  network port.
- **Embedding is local.** `embeddings.provider` defaults to `auto`, which resolves to `fastembed` — an
  in-process ONNX model that touches no network — and then to `sentence-transformers` if it is installed.
  Nothing about the environment changes this: an `OPENAI_API_KEY` in your shell does not select a cloud
  backend, so the text of your documents is not sent anywhere for embedding. To send it somewhere, name the
  provider (`openai`, `cohere`); that is the request. That distinction is deliberate, and a change that
  makes `auto` consult the environment again is a vulnerability.
- **Every other provider is named, not inferred.** `ollama` talks to your Ollama server (localhost by
  default); `openai` and `cohere` talk to those services; the `cloud` extra's LLM features send your prompts
  to whichever of OpenAI or Anthropic you configure. Each requires its own extra to be installed and its
  provider named in the config.
- **Graphiti telemetry is off unless you opt in.** graphiti-core reads
  `GRAPHITI_TELEMETRY_ENABLED` at construction time and treats an absent variable as enabled; DocuGraph
  sets it to `false` when it is absent, so a fresh install sends no telemetry. An explicit value is left
  alone. Set it to `true` to opt in.
- **A missing extra is an error, not a fallback.** The `crawl` and `graph` extras add capability; without
  one, the feature says which extra to install and stops. No path substitutes a weaker implementation when
  an extra is absent, and a feature never silently degrades to a different transport. `hybrid_search`
  reports `graph_available: false` with that same sentence as the reason, which is a reported omission,
  not a silent one.

## The Ladybug backend and its full-text-search extension

The opt-in `ladybug` backend is the one place where docugraph loads a native extension, so the rules are
stated plainly.

- **DocuGraph never runs Ladybug's `INSTALL`** as part of opening a graph. The backend loads exactly one
  file, with `LOAD EXTENSION '<path>'`, from docugraph's own cache.
- **That file is verified first.** `docugraph graph fetch-fts` downloads it from the extension repository
  and checks its SHA-256 against a value pinned in this package for the installed Ladybug version and
  platform. It loads the file once to confirm it works, then keeps it. A download that does not match is
  deleted, never loaded.
- **A pin is tamper-evident, not a signature.** It records the bytes this release was tested with; it
  does not authenticate the server, and it cannot detect a compromise that happened before the pin was
  measured. CI re-downloads the artifact on every run and fails if the bytes change, which is the only
  independent check behind the pinned values.
- **The unverified path is explicit and interactive.** `docugraph graph install-fts --unverified` runs
  Ladybug's own `INSTALL FTS`. It states, before doing so, that the download is unverified and that
  Ladybug's `INSTALL` can, in some versions, download and execute an installer in-process. It refuses
  without a terminal confirmation unless `--yes` is passed. Afterwards docugraph records the digest of
  exactly what was installed, and later loads accept only those bytes.
- **FTS is required, not optional, on this backend.** If the extension is missing, unloadable, or its
  index cannot be created, the Ladybug backend refuses to start with one sentence naming the fix. It
  does not run with degraded graph search.
- **MCP tools never approve a download.** Both `fetch-fts` and `install-fts` are CLI commands; an MCP tool
  call that finds no usable extension returns the refusal instead.

## URL and repository guards

The MCP tools accept values chosen by a model, so both `crawl_url` and `index_git` treat their input as
untrusted and decide before any effect:

- The scheme allowlist, a leading `-`, DNS resolution and the global-address rule are enforced before
  the cache, the browser, git or the vector store is touched.
- **Link-local and cloud-metadata addresses are refused whatever `allow_private_networks` says.** The
  setting means "this intranet", never "this machine's cloud credentials": 169.254.0.0/16 and
  fe80::/10 (including the AWS instance metadata address), `fd00:ec2::254` and `100.100.100.200`
  stay refused. Every IPv6 literal is also decomposed into the IPv4 addresses it can carry, and each
  of those is judged by the same rule: IPv4-mapped (`::ffff:a.b.c.d`), IPv4-compatible
  (`::a.b.c.d`), NAT64 (`64:ff9b::/96` and `64:ff9b:1::/48`), 6to4 (`2002::/16`) and Teredo
  (`2001::/32`, both the server and the client). 0.2.1 judged only the IPv6 wrapper, so a private IPv4
  spelled as `[64:ff9b::10.0.0.5]` was reported global and passed the default policy; the embedded
  address is what is judged now.
- Redirects are walked hop by hop. Each `Location` is judged before its request is made, because
  Chromium otherwise follows a redirect internally without re-entering the route handler. A page whose
  final URL is refused is dropped and reported, never indexed.
- Inside the browser every request in the **context** is routed, not only the page that started the
  crawl: sub-resource URLs are judged, popups (`window.open`, `target=_blank`), workers and frames pass
  through the same handler, non-network schemes are refused, and websockets are closed. A page script
  cannot register a service worker — the context hook removes `navigator.serviceWorker` in every frame
  before page scripts run — so a worker cannot fetch a blocked address outside the route handler. That
  removal is defence in depth; the route handler is the primary control. crawl4ai's anti-bot retry
  navigates through the same hooked context and is covered by these guards; its browser-less
  `fallback_fetch_function` would run outside every guard, so the retry and the fallback are pinned off
  in both crawl paths.
- **The guards are checked before the browser starts.** If the installed crawl4ai lacks the hook, the
  retry switch or the browser-less fallback they pin, the crawl is refused as `unsupported_crawler`
  rather than run unguarded; an error while registering the hook — crawl4ai raises `Invalid hook
  type` for a name it does not have — becomes the same refusal, never a crash. The guards are bound
  to crawl4ai 0.9.x's hook table and to Playwright 1.48+ (the websocket routing API), and both are
  capped below their next minor so a library release cannot silently reopen either.
- Git runs with `--` before the URL and path, a closed ref grammar, `protocol.file.allow=never`,
  `protocol.ext.allow=never` and `http.followRedirects=false`. The clone destination is resolved and must
  stay inside the configured repos directory; a name that escapes it is refused, not renamed. When a
  clone directory already exists, its **origin** must equal the requested URL and pass the same policy,
  and the pull and fetch fallback run under the same `-c` lockdown — the requested URL is not what pull
  would contact.

The browser suites in `tests/integration` run against a real Chromium in CI, and there a missing
browser is a failure rather than a skip (`DOCUGRAPH_REQUIRE_BROWSER=1`): a guard that only ever skips
is a guard nobody runs.

Two limits are stated here rather than hidden. **DNS rebinding between the policy check and the
browser's own resolution is narrowed, not closed:** the route-level re-check catches a request that
resolves differently on the second look, but a name that changes after that look can still be
fetched. And **the CLI is exempt from the scheme, DNS and address rules only** — `docugraph crawl` and
`docugraph index git` fetch and clone exactly what the person at the keyboard asked for. The
destination-containment, `--` and ref-grammar rules apply to every caller. The address policy above is
the MCP tools' contract, where the input is chosen by a model.

Anything that breaks one of those statements — an index written outside the configured storage
directory, a provider request made when no cloud provider was named in the config, a path escaping the
configured root — is a vulnerability, and we would want to hear about it.

## Out of scope

Findings that depend on an attacker already running code as the user who runs DocuGraph; and the content
of documents you chose to index.
