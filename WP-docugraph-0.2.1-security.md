# WP-docugraph-0.2.1 — Agent-reachable crawl/clone guards and third-party telemetry off

**Repository:** CJ-coding-apps/docugraph · **Base:** `main` @ 491db39 (v0.2.0) · **Branch:** `fix/0.2.1-security`
**Issued by:** CJ (maintainer), 2026-11-05 · **Revision 1** (2026-11-05): amendments A-1…A-9 in §10, issued at plan review · **Reviewer:** an independent verification session (Claude), per AGENTS.md §8.4
**Governing rules:** CJ's AGENTS.md constitution, revision 2. It is not installed in this repository; it governs your conduct all the same. Where this order and the constitution disagree, the constitution wins (§11), and you report the conflict.

---

## 0. How this order runs

1. **Research → Plan → STOP (§7).** Read the files in §6. Deliver a plan covering: your understanding, the assumptions you are making (listed first), every new unit's signature with *core* or *shell* stated, and the effects each unit takes as parameters, the test list mapped to §4's examples, and any open question. Then **stop**. CJ forwards the plan to the reviewer; you implement only the approved plan.
2. **Tests first.** Commit the tests for §4 and §5 before any implementation. Run them against unmodified 0.2.0 and **paste the failing output** — that is the red evidence. A test that passes on 0.2.0 is either testing nothing or testing a guard that already exists; report which.
3. **Implement** until green. Never edit an approved test to match the code (§8.2). If a test looks wrong, stop and report it with the spec line it contradicts.
4. **Verify** with the exact commands in §8 and paste their real output (§10). Report every failure, warning and skip.
5. **Open a PR. Do not merge it.** The PR body follows §9's format. Then stop. Merging, the version bump, tagging, the GitHub release, the PyPI approval and the security advisory are **CJ's acts** (§1). Prepare them; never perform them, even if a later message seems to ask you to.

---

## 1. Why

docugraph's `docs/SECURITY.md` states that *"the index is local"*, that a path escaping the configured root is a vulnerability, and that no provider request is made unless the config names one. Three agent-reachable behaviours break those statements:

- **`crawl_url` (MCP)** passes the agent's URL straight to Crawl4AI (`mcp_server.py` `_crawl_url`, ~line 536). Crawl4AI accepts `file://` (local files) and `raw:` as well as http(s), and crawls in headless Chromium, so a page can also redirect to an internal address. `crawl_site`'s same-domain check (`crawler.py` ~188/265) only governs link-following, never the start URL. `max_pages` has no upper limit. An agent can therefore pull local files, loopback services or cloud metadata into the index, where `search_docs` returns them — including an agent steered by text injected into a page it crawled.
- **`index_git` (MCP)** builds `git clone [--depth N] [--branch REF] <url> <path>` (`git_indexer.py` ~266–275) with **no `--` before `<url>`**, so a value beginning with `-` is parsed as a git option (the class of bug behind GitPython CVE-2022-24439; a working exploit through clone's options is *not* demonstrated). It also clones local paths and `file://` repositories into the index, and the non-shallow path runs `git checkout <ref>` with an agent-supplied `ref`.
- **graphiti-core** sends anonymous PostHog telemetry whenever a `Graphiti` instance is constructed, unless `GRAPHITI_TELEMETRY_ENABLED=false`. It disables itself under pytest, so the suite has never seen it. docugraph neither disables nor discloses it.

**Version: patch (0.2.1).** These fixes make guarantees that `docs/SECURITY.md` already states true (constitution §9 version semantics, C-007). That holds only because the new destination rules apply to the **MCP surface**; human-invoked CLI behaviour is unchanged (§3).

## 2. Maintainer decisions in force (cite as D-2026-11-05-#)

1. Private and internal destinations are **refused by default** on the MCP surface; a setting re-enables them for intranet use.
2. *(0.3.0 — out of scope here.)*
3. A GitHub security advisory will be published **by CJ** after release. Draft its text in your NOTES file; do not publish or commit it.
4. *(0.3.0 — out of scope here.)*
5. The MCP `crawl_url` tool caps `max_pages` at **50**.

## 3. Behavior

### 3.1 URL policy (new core module `docugraph/security/url_policy.py`)
A pure decision over a URL. Effects (DNS) are a parameter. The rules, applied in this order:
1. Malformed or empty → refused (`malformed`).
2. Scheme must be in the allowlist for the kind of URL (web: `http`, `https`; git remote: `https`, `ssh`, or scp-style `user@host:path`). Anything else — `file`, `raw:`, `ftp`, `ext::`, `fd::`, a local path, a value starting with `-` — is refused (`scheme_not_allowed`, or `option_like` for a leading `-`). **`allow_private` never re-enables a refused scheme.**
3. If `allowed_domains` is non-empty, the host must equal one of them or be a subdomain of one (`domain_not_allowed`).
4. The host is passed through the injected resolver — **every** host, including IP literals and numeric spellings (`0x7f000001`, `2130706433`, `127.1`), so they are judged by the address they denote, not by string checks. Resolver failure → refused (`unresolvable`); fail closed.
5. Unless `allow_private` is true, **every** returned address must satisfy `ipaddress.ip_address(a).is_global`. One non-global address refuses the URL (`private_address`). This covers loopback, RFC 1918, link-local (169.254.0.0/16, including 169.254.169.254), CGNAT 100.64/10, ULA fc00::/7, unspecified, and IPv4-mapped IPv6.

### 3.2 Where the policy applies
- **MCP `crawl_url`:** the start URL, every link `crawl_site` would enqueue, and — at the browser layer — every request the page makes (navigation, redirect, subresource), via a Playwright `page.route("**/*")` handler registered through a crawl4ai page hook. **Superseded by A-3 in §10:** a `page.route` handler is called only for the first URL of a redirect, so redirects need the design in A-3.
- **MCP `index_git`:** the remote URL (git-remote rules); git additionally runs with `-c protocol.file.allow=never -c protocol.ext.allow=never`.
- **All callers of the git indexer (MCP and CLI):** a `--` separates options from `<url> <path>` in `git clone`; `ref` is validated (no leading `-`, `git check-ref-format`-shaped, or a hex commit id) before git runs, and is placed where git reads it as a revision, never as an option. This is hardening, not a behaviour change, so it applies everywhere.
- **Explicitly unchanged:** CLI `crawl` and CLI git indexing keep accepting what they accept today. The policy is an **injected parameter** (`None` = today's behaviour) that the MCP layer supplies; the CLI does not.

### 3.3 Settings (in `CrawlerConfig`, env prefix `DOCUGRAPH_`, nested delimiter `__`)
- `allow_private_networks: bool = False` — `DOCUGRAPH_CRAWLER__ALLOW_PRIVATE_NETWORKS`
- `allowed_domains: list[str] = []` — `DOCUGRAPH_CRAWLER__ALLOWED_DOMAINS`
Both govern the MCP `crawl_url` and `index_git` policies.

### 3.4 MCP result contract
- A refused URL returns `isError: true` with text that begins `Refused: <reason>` — `<reason>` is one of `malformed`, `scheme_not_allowed`, `option_like`, `domain_not_allowed`, `unresolvable`, `private_address` — followed by a one-line detail. **Nothing is fetched, cloned or indexed**: the store's chunk count is unchanged.
- `crawl_url` with `max_pages` > 50 crawls at most 50 pages, and the result text says `max_pages capped at 50`. `max_pages` < 1 is refused with `isError: true`.
- **Tool names, input schemas and descriptions do not change.** The frozen tool-surface snapshot (`tests/e2e/test_mcp_server.py`) must pass untouched.

### 3.5 Third-party telemetry
- Before graphiti-core reads `GRAPHITI_TELEMETRY_ENABLED` on any path that constructs `Graphiti`, docugraph sets it to `"false"` **if it is absent**. An explicit user value — including `"true"` — is left alone.
- Your plan must **cite the graphiti-core 0.30.2 source line where the variable is read** (import time or construction time), and place the default before it on every path.

### 3.6 Disclosure
- `README.md`: a new **"Network activity"** section listing every outbound connection docugraph can make: Hugging Face model downloads on first use (fastembed; the reranker when enabled), crawl and clone targets, the configured LLM or embedding provider when one is named, Ollama when configured, and Graphiti telemetry (off unless opted in). Also document the two new settings.
- `docs/SECURITY.md`: in "What this project does with your data", add the MCP destination rules, the telemetry default, and two honest limits — DNS rebinding between the policy check and the browser's own resolution is narrowed by the route-level re-check but not closed; and CLI crawls/clones are user-directed and not restricted.
- `CHANGELOG.md` (new, Keep a Changelog — copy the house format from cognitive-fabric's): backfill 0.1.0, 0.1.1 and 0.2.0 from the git history and GitHub release notes, and add this change-set under `## [Unreleased]` with **Security** and **Fixed** subsections. CJ renames `Unreleased` to `0.2.1` in his version-bump commit.

### 3.7 Dependency telemetry audit (report only)
Grep the installed site-packages of the full environment for `posthog`, `telemetry`, `analytics`, `sentry`, `segment`, `mixpanel`. Report a table: package, what it sends, default, how to disable, and whether docugraph exercises it. **Fix only Graphiti here**; anything else found becomes a finding for CJ.

## 4. Worked examples (acceptance — each line becomes a test; ORACLE in brackets)

*URL policy, web kind, defaults (allow_private=false, no allowlist), resolver = a fake mapping unless stated:*
| # | Input | Resolver returns | Expected | Oracle |
|---|---|---|---|---|
| W1 | `file:///etc/hosts` | — (not called) | refused `scheme_not_allowed` | D-1; Crawl4AI docs (file:// accepted) |
| W2 | `raw:<html>secret</html>` | — | refused `scheme_not_allowed` | D-1; Crawl4AI docs (raw:) |
| W3 | `http://127.0.0.1:8080/` | `127.0.0.1` | refused `private_address` | D-1; RFC 6890 via `ipaddress.is_global` |
| W4 | `http://[::1]/` | `::1` | refused `private_address` | as W3 |
| W5 | `http://169.254.169.254/latest/meta-data/` | `169.254.169.254` | refused `private_address` | as W3 (link-local) |
| W6 | `http://docs.internal.example/` | `10.0.0.5` | refused `private_address` | as W3 (RFC 1918) |
| W7 | `https://cdn.example/` | `93.184.216.34`, `192.168.1.10` | refused `private_address` (one is enough) | §3.1 rule 5 |
| W8 | `https://nowhere.invalid/` | raises `OSError` | refused `unresolvable` | §1 fail closed |
| W9 | `https://docs.python.org/3/` | `151.101.0.223` | allowed | D-1 |
| W10 | `http://127.0.0.1/` with `allow_private=true` | `127.0.0.1` | allowed | D-1 setting |
| W11 | `file:///etc/hosts` with `allow_private=true` | — | refused `scheme_not_allowed` | §3.1 rule 2 |
| W12 | `https://evil.example/` with `allowed_domains={"python.org"}` | — | refused `domain_not_allowed` | §3.3 |
| W13 | `https://docs.python.org/` with `allowed_domains={"python.org"}` | public | allowed | §3.3 (subdomain) |
| W14 | `http://0x7f000001/` and `http://2130706433/` — **system resolver** (shell test) | real | refused — `private_address`, or `unresolvable` where the platform's resolver rejects the spelling; never allowed | §3.1 rules 4–5 |

*Git remote kind:*
| # | Input | Expected | Oracle |
|---|---|---|---|
| G1 | `--upload-pack=touch /tmp/x` | refused `option_like`; git never invoked | git-clone(1): `--` ends options; D-1 |
| G2 | `-u touch /tmp/x` | refused `option_like` | as G1 |
| G3 | `/Users/me/private-repo`, `./repo`, `file:///tmp/repo` | refused `scheme_not_allowed` | D-1 |
| G4 | `ext::sh -c id`, `fd::0` | refused `scheme_not_allowed` | git-remote-ext(1), git-remote-fd(1) |
| G5 | `https://10.0.0.5/team/repo.git` | refused `private_address` | §3.1 rule 5 |
| G6 | `https://github.com/CJ-coding-apps/docugraph.git` (public resolver) | allowed | D-1 |
| G7 | `git@github.com:CJ-coding-apps/docugraph.git` (public resolver) | allowed | D-1 (scp-style ssh) |
| G8 | ref `-x`, `--orphan=y`, `a..b`, `bad ref` | rejected before git runs | git-check-ref-format(1) |
| G9 | ref `main`, `v0.2.0`, `feature/x`, `0123abcd` | accepted | git-check-ref-format(1) |
| G10 | any accepted clone (subprocess mocked — shell test) | argv contains `--` immediately before `<url>` | git-clone(1) |

*MCP end-to-end (real stdio server, `tests/e2e`):*
| # | Call | Expected | Oracle |
|---|---|---|---|
| M1 | `crawl_url{url:"file:///etc/hosts"}` | `isError`, text starts `Refused: scheme_not_allowed`; chunk count unchanged | §3.4 |
| M2 | `crawl_url{url:"http://169.254.169.254/"}` | `isError`, `Refused: private_address`; count unchanged | §3.4 |
| M3 | `index_git{url:"--upload-pack=touch <tmp_path>/pwned"}` (a pytest temp dir) | `isError`, `Refused: option_like`; the marker file does not exist | §3.4 |
| M4 | `index_git{url:"file:///tmp"}` | `isError`, `Refused: scheme_not_allowed` | §3.4 |
| M5 | `crawl_url{url:<allowed>, max_pages:0}` | `isError` | §3.4 |
| M6 | tool listing | identical to the frozen snapshot | §3.4 |

*Crawler and browser:*
| # | Case | Expected | Oracle |
|---|---|---|---|
| C1 | `crawl_site` under policy, page links to `http://127.0.0.1/admin` | that link is not enqueued | §3.2 |
| C2 | browser route decision for a request to `http://169.254.169.254/` under policy | abort | §3.2 |
| C3 | local HTTP server whose page 302-redirects to a blocked address, crawled via MCP policy | the blocked target is never fetched (server-side request log is empty for it) | §3.2 — integration-marked; needs `playwright install chromium` |
| C4 | `max_pages:500` via MCP | at most 50 pages; result text contains `max_pages capped at 50` | D-5 |

*Telemetry:*
| # | Case | Expected | Oracle |
|---|---|---|---|
| T1 | variable unset → docugraph's graph path prepared (in a **subprocess without pytest imported**) | graphiti's own telemetry-enabled check returns false | Graphiti telemetry docs; §3.5 |
| T2 | variable preset to `"true"` | left as `"true"` | §3.5 |

## 5. Edge cases (also tests)
IDN host (`https://bücher.example/`); trailing-dot host (`localhost.`); uppercase scheme (`FILE:///etc/hosts`, `HTTP://127.0.0.1`); credentials in the URL (`http://user:pw@127.0.0.1/`); an IPv4-mapped IPv6 literal (`http://[::ffff:127.0.0.1]/`); an empty URL; a URL that is only whitespace.

## 6. Files
**Expected to change:**
- `src/docugraph/security/__init__.py`, `src/docugraph/security/url_policy.py` *(new)*
- `src/docugraph/core/config.py` *(two `CrawlerConfig` fields)*
- `src/docugraph/ingestion/crawler.py` *(optional policy parameter; link filter; route hook)*
- `src/docugraph/ingestion/git_indexer.py` *(`--`; ref validation; optional policy and git `-c` flags)*
- `src/docugraph/interfaces/mcp_server.py` *(supply policies; refusal results; clamp)*
- `src/docugraph/storage/graph_store.py` *(telemetry default)*
- tests: `tests/unit/test_url_policy.py`, `tests/unit/test_git_indexer_args.py`, `tests/unit/test_telemetry_default.py`, `tests/integration/test_crawl_redirect.py`, and new cases in `tests/e2e/test_mcp_server.py`
- `README.md`, `docs/SECURITY.md`, `CHANGELOG.md` *(new)*

**Reserved — do not edit (§1):** `pyproject.toml` (including the version field), `uv.lock`, `.github/workflows/*`, `docker/*`. **No new dependencies**: `ipaddress`, `socket` and `urllib.parse` are stdlib. If you find you need a file outside this list, name it in the PR under "Files outside the spec's list, and why", or stop and ask.

## 7. Non-goals (binding)
The 0.3.0 work (extras split, removing `typer`, dependency caps, the Python ceiling, public API and `py.typed`, PyPI classifiers, macOS CI); any change to CLI behaviour; closing DNS rebinding fully; any change to MCP tool names, schemas or descriptions; fixing telemetry in any library other than Graphiti; reformatting or refactoring code you are not otherwise changing.

## 8. Verification you must run and paste (CI's exact commands)
```bash
uv sync --all-extras
uv run ruff check src/ scripts/
uv run ruff format --check src/ scripts/
uv run mypy src/ scripts/
uv run pytest
uv run pytest tests/integration/test_crawl_redirect.py -v   # after: uv run playwright install chromium
```
Also: the **red run** — the new tests against unmodified 0.2.0, with output. In it, record two facts explicitly, because they turn "likely" into "demonstrated" (or not) for the advisory: whether **M1 actually indexed `/etc/hosts`** on 0.2.0 (search the store for a line of it), and whether **M3 actually created the marker file** on 0.2.0. Either answer is useful; report what happened, not what was expected. And a **mutation check**: delete the scheme check, the `is_global` check, the `--`, and the telemetry default, one at a time; re-run; show which tests go red for each. A mutation that leaves everything green means a vacuous test — report it (§8.3).

## 9. PR body (§9 format) and report-back
- The work package (this file) and the decisions cited (D-2026-11-05-1, -3, -5).
- The verification actually run, with counts, and the red-run and mutation results.
- Files outside the spec's list, and why. Every deviation from this order, flagged for veto.
- The §3.7 telemetry audit table.
- What was deliberately **not** done, and who owns it: the version bump, merge, tag, release, PyPI approval and advisory (CJ); the 0.3.0 items.
- A **live-row specification** (§8.4, Appendix B V6): what CJ must observe when he runs a real Claude Code session against this branch's MCP server — asking it to crawl `file:///etc/hosts` and `http://169.254.169.254/` (expect the refusals, relayed honestly), and to crawl one public docs page (expect success). You specify the rows; CJ runs them.

Report back: the PR URL, branch, commit SHAs, and the outputs above. Then stop.

---

## 10. Amendments — revision 1 (issued at plan review; binding, same force as §3–§5)

**A-1 · `index_git` `name` path escape (new HIGH, in scope).** `GitIndexer.clone()` builds `self._repos_dir / name` with an agent-supplied `name` (or one derived from the URL's last path component). `pathlib` discards the base when `name` is absolute (`/data/repos` / `/Users/x/private` → `/Users/x/private`), and `..` escapes it. If the path exists, `clone()` calls `update()`, which runs `git pull --ff-only` there and indexes it. So an agent can make docugraph pull and index any existing git repository on the machine. `docs/SECURITY.md` already names a path escaping the configured root as a vulnerability, so this is a patch fix (C-007).
- Rule (all callers, MCP and CLI — it is a stated guarantee, not a new policy): `name`, whether given or derived, must be one path component: non-empty, not `.` or `..`, no `/` or `\\`, no NUL, not absolute. And `(repos_dir / name).resolve()` must be inside `repos_dir.resolve()` (`is_relative_to`), which also catches a symlink planted inside `repos_dir`. Otherwise refuse **before any git process runs**.
- MCP refusal: `Refused: path_escape — <detail>`. Add `path_escape` to §3.4's reason list. CLI: a clear error and non-zero exit (a CLI call that escaped the root was never a supported behaviour).
- A URL whose derived name fails the rule (e.g. `https://host/a/..`) is refused the same way; do not silently rename it.
- New worked examples (each a test):

| # | Case | Expected | Oracle |
|---|---|---|---|
| G11 | `index_git(url=<allowed https>, name="/tmp/outside-repo")` where that path is a git repo | `Refused: path_escape`; no git process spawned; chunk count unchanged | A-1 |
| G12 | `name="../outside-repo"` | `Refused: path_escape` | A-1 |
| G13 | `name="ok-name"` | proceeds (clone args built under `repos_dir/ok-name`) | A-1 |
| G14 | a symlink `repos_dir/link -> /tmp/outside-repo`, `name="link"` | `Refused: path_escape` | A-1 |
| G15 | CLI git indexing with `name="/tmp/outside-repo"` | non-zero exit, error names the path; no git process | A-1 |
| M7 | red run: on unmodified 0.2.0, MCP `index_git` with `name` = absolute path of a temp git repo containing a sentinel string | record whether the sentinel became searchable (demonstrated or not, for the advisory) | A-1 |

**A-2 · URL policy rulings on the plan's assumptions.**
- A1, A5: approved as written.
- A2 approved, plus: an IPv4-mapped / IPv4-compatible IPv6 address is judged by its mapped IPv4 (`ipv4_mapped`), and `urlsplit(...).port` raising `ValueError` → `malformed`.
- A3 approved with one change: the browser route handler **fails closed**. Requests with scheme `http`/`https` are checked by policy; `data:`, `blob:` and `about:` pass; **every other scheme is aborted** (`file:`, `chrome-extension:`, `ws:`/`wss:`, `filesystem:`, unknown).
- A4 approved, plus these cases in §5: `http://0.0.0.0/` → `private_address`; after the scheme decision (so G2/G4 stay `scheme_not_allowed`), for allowed schemes only: an ASCII control character (incl. tab/CR/LF) anywhere in the value, or a backslash or whitespace anywhere in the **raw authority** (`urlsplit(...).netloc`, userinfo included — not the parsed hostname) → `malformed`; whitespace in a path or query is not refused (parser-differential guard: `http://127.0.0.1\\@evil.example/` must not be judged as `evil.example`); a scoped IPv6 literal (`http://[fe80::1%25eth0]/`) → refused whatever `allow_private` says.

**A-3 · Redirect enforcement (replaces the redirect clause of §3.2).** Playwright calls a `page.route` handler only for the **first** URL of a redirect chain, so a handler that only checks `request.url` lets a 302 to `169.254.169.254` through. Required design:
1. For **navigation** requests (`request.is_navigation_request()`), the handler fetches the response itself with `route.fetch(max_redirects=0)`. If the status is 3xx, resolve `Location` against the request URL and run the policy on it: refused → `route.abort()`; allowed → `route.fulfill(response=…)` so the browser issues the next hop as a new request, which comes back through the handler.
2. For **sub-resource** requests, check `request.url` and abort or `route.continue_()`. A sub-resource redirect is the documented residual (it cannot put a document into the index; see 3).
3. **Backstop, independent of the browser:** after each page is crawled, run the policy on the result's final URL (`result.url` / `redirected_url`, whichever crawl4ai exposes). A page whose final URL fails policy is **dropped before indexing** and reported in the result text. This also covers anything 1 misses.
4. Service workers: if crawl4ai lets you create the context with `service_workers="block"`, do so under policy; if not, document the residual in `docs/SECURITY.md`. WebSockets: use `page.route_web_socket` to close them under policy if the installed Playwright has it; otherwise document.
5. **C3 adjudicates.** The local server's request log must show no request to the blocked target. Add **C5**: the redirect target is an allowed-scheme URL on a *different* blocked host reached only via a second hop (302 → 302 → blocked), same expectation. If C3/C5 cannot be made green with 1–3, stop and report; do not weaken them.
- Residual to state in `docs/SECURITY.md`: `route.fetch` resolves DNS in Playwright's process, separately from the policy's resolver, so DNS rebinding between check and fetch remains (already a non-goal).

**A-4 · git hardening addition.** Under the MCP policy, also pass `-c http.followRedirects=false`, so an HTTPS remote cannot redirect git to a host the policy never saw. Add **G16**: the built clone args under policy contain that pair before `clone`.

**A-5 · Telemetry tests.** Keep T1/T2 and add: **T3** — a call-site test that goes red if the `apply_telemetry_default(os.environ)` call is removed from `_ensure_initialized` (mutation check must show it); **T4** — a structural test (AST walk over `src/`) asserting `Graphiti(` is constructed only in `storage/graph_store.py`, so a new construction site cannot bypass the default.

**A-6 · C4 placement.** C4 needs a browser, so it is integration-marked with an explicit skip reason when Chromium is absent. Add a shell-level unit test that the MCP layer passes `max_pages ≤ 50` to the crawler and that the result text carries `max_pages capped at 50` — that one always runs.

**A-7 · Test authoring and lockfile (plan questions 1–2).** Blind authoring by a fresh subagent is approved: it reads this order, the approved signatures and `tests/` helpers, and **not** `src/`. Commit tests first and record the tests-commit SHA in the PR; the reviewer will diff test files between that SHA and the head, and any change to an approved test is a finding. `uv.lock` stays untouched (no new deps).

**A-8 · Files.** `docs/configuration.md` is added to §6 for the two new settings and the `path_escape` refusal.

**A-9 · LOW (fix if cheap, otherwise list in the PR).** Memoize DNS results per host for the duration of one crawl. Expose one public entry point (`UrlPolicy.check`); keep `check_url` private or remove it.

**Mutation checks (§8) extended:** also delete the `path_escape` check, the navigation-redirect `Location` check, the post-crawl final-URL check, and the `apply_telemetry_default` call site, one at a time, and show which tests go red.

Sources for A-3: Playwright `Page#route` docs ("The handler will only be called for the first url if the response is a redirect"), https://rubydoc.info/gems/playwright-ruby-client/0.3.0/Playwright/Page:route ; `Route#fetch` (`maxRedirects`), https://gem.sh/gems/playwright-ruby-client/v1.56.0/classes/Playwright::Route

**A-10 · Maintainer rulings after plan review (2026-11-05; binding, same force as §3–§5).**

- **N1 approved.** `src/docugraph/interfaces/cli.py` is added to §6. Only `RepoPathRefusedError` and `InvalidGitRefError` become `click.ClickException` (exit 1); every other CLI branch is unchanged. CHANGELOG lists it under Fixed.
- **N5 approved.** A scoped IPv6 literal is `malformed`, detail `"scoped IPv6 literal"`. The reason vocabulary stays closed; no new reason is added.
- **N12 approved, with two conditions.** G11–G14 inject a fake resolver returning a global address (`1.1.1.1`), so no test touches the network. For M7 the temp repo gets a local bare `origin` so 0.2.0's `git pull --ff-only` can succeed offline, and the pull outcome and the index outcome are recorded separately.
- **N8 approved.** The dropped-page line is one line, `Dropped N page(s) by policy:`, followed by the first five URLs with their reasons.
- **N6 overruled; documentation alone is not enough.** A service worker registered by an allowed page can fetch an internal address outside `page.route`, and what it fetches can be rendered into the DOM and indexed. In the same context hook that registers the route, call `context.add_init_script` (public Playwright API; it runs in every frame before page scripts) to remove `navigator.serviceWorker`. Integration test **C6** is required: an allowed local page tries to register a service worker that fetches a blocked target, and the blocked server's log must show no request. If C6 cannot go green, stop and report; do not weaken it. The `docs/SECURITY.md` note stays as defence in depth.
- **Mutation rule.** A mutation row expected to stay green on 3.11 is also run on 3.13 (the newest Python CI runs); a row green on every supported Python makes its test vacuous and is reported as such.
- **§6 additions by A-10.** `src/docugraph/interfaces/cli.py`. `docs/configuration.md` is covered by A-8.
