"""Crawl guards at the pure/unit level: link filter, route hook, backstop, clamp.

C1 exercises ``enqueue_targets`` with a dict-backed resolver. C2 exercises
``make_page_route_hook`` with fake Playwright route/request objects -- every
effect (abort, continue, fetch, fulfill) is recorded, so the decision can be
adjudicated without a browser. The redirect rows encode A-3: the handler must
fetch the navigation itself with ``max_redirects=0`` and inspect ``Location``.

The backstop and clamp rows go through ``_crawl_url`` with the crawler and the
vector store replaced, so they always run and touch nothing.
"""

from types import SimpleNamespace

import pytest

from docugraph.core.config import reset_config
from docugraph.ingestion import crawler as crawler_module
from docugraph.ingestion.crawler import (
    enqueue_targets,
    final_page_url,
    make_page_route_hook,
)
from docugraph.interfaces import mcp_server
from docugraph.security.url_policy import (
    RefusalReason,
    UrlKind,
    UrlPolicy,
    UrlRefusal,
)

PRIVATE_URL = "http://169.254.169.254/"


# ---------------------------------------------------------------------------
# Helpers: policy, fake route/request, a store that absorbs anything
# ---------------------------------------------------------------------------


def _mapping(table: dict[str, tuple[str, ...]]):
    def resolve(host: str) -> tuple[str, ...]:
        return table[host]

    return resolve


def _web_policy(table: dict[str, tuple[str, ...]], **kwargs) -> UrlPolicy:
    return UrlPolicy(kind=UrlKind.WEB, resolver=_mapping(table), **kwargs)


class _Request:
    def __init__(self, url: str, navigation: bool) -> None:
        self.url = url
        self._navigation = navigation

    def is_navigation_request(self) -> bool:
        return self._navigation


class _Response:
    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self.headers = headers or {}


class _Route:
    """A Playwright Route stand-in that records everything it is asked to do."""

    def __init__(
        self,
        url: str,
        *,
        navigation: bool = True,
        response: _Response | None = None,
        fetch_error: Exception | None = None,
    ) -> None:
        self.request = _Request(url, navigation)
        self.calls: list[str] = []
        self.fetch_kwargs: dict | None = None
        self.fulfilled_with = None
        self._response = response
        self._fetch_error = fetch_error

    async def abort(self) -> None:
        self.calls.append("abort")

    async def continue_(self) -> None:
        self.calls.append("continue_")

    async def fetch(self, **kwargs) -> _Response:
        self.calls.append("fetch")
        self.fetch_kwargs = kwargs
        if self._fetch_error is not None:
            raise self._fetch_error
        return self._response if self._response is not None else _Response(200)

    async def fulfill(self, **kwargs) -> None:
        self.calls.append("fulfill")
        self.fulfilled_with = kwargs.get("response")


class _Null(int):
    """Zero that can be awaited, called, indexed and iterated.

    ``_crawl_url``'s indexing effect is not under test here; the rows need a
    store that never blocks or fails regardless of which method the handler
    calls on it (sync or async).
    """

    def __new__(cls) -> "_Null":
        return int.__new__(cls, 0)

    def __await__(self):
        if False:  # makes this a generator without yielding
            yield
        return self

    def __call__(self, *args, **kwargs) -> "_Null":
        return _Null()

    def __getattr__(self, name: str) -> "_Null":
        return _Null()

    def __getitem__(self, key) -> "_Null":
        return _Null()

    def __iter__(self):
        return iter(())


def _result_text(result) -> str:
    return "\n".join(
        item.text
        for item in getattr(result, "content", [])
        if getattr(item, "text", None) is not None
    )


@pytest.fixture
def mcp_env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DOCUGRAPH_STORAGE__DATA_DIR", str(tmp_path / "data"))
    reset_config()
    yield
    reset_config()


def _fake_web_policy() -> UrlPolicy:
    return UrlPolicy(kind=UrlKind.WEB, resolver=lambda host: ("1.1.1.1",))


def _patch_mcp_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop ``_crawl_url`` from resolving, embedding or indexing anything."""
    monkeypatch.setattr(mcp_server, "_web_policy", _fake_web_policy)
    monkeypatch.setattr(mcp_server, "get_vector_store", lambda: _Null())


def _install_fake_crawler(
    monkeypatch: pytest.MonkeyPatch,
    recorder: dict,
    *,
    site_docs: list | None = None,
    on_site=None,
) -> None:
    """Replace DocCrawler everywhere the handler can have bound it."""

    class FakeCrawler:
        def __init__(self, *args, **kwargs) -> None:
            recorder["constructed"] = True

        async def crawl_site(self, url: str, max_pages: int = 100, **kwargs):
            recorder["crawl_site"] = True
            recorder["max_pages"] = max_pages
            if on_site is not None:
                on_site(kwargs)
            return [] if site_docs is None else site_docs

        async def crawl_single(self, url: str, **kwargs):
            recorder["crawl_single"] = True
            return [] if site_docs is None else site_docs

    monkeypatch.setattr(mcp_server, "DocCrawler", FakeCrawler, raising=False)
    monkeypatch.setattr(crawler_module, "DocCrawler", FakeCrawler)


# ---------------------------------------------------------------------------
# C1: enqueue_targets
# ---------------------------------------------------------------------------


def test_c1_private_link_is_not_enqueued() -> None:
    """ORACLE: C1 / §3.2 — the policy decides each candidate link."""
    policy = _web_policy({"example.com": ("127.0.0.1",)})
    targets = enqueue_targets(
        [{"href": "/admin"}],
        page_url="http://example.com/",
        base_netloc="example.com",
        url_pattern=None,
        policy=policy,
    )
    assert targets == []


def test_c1_public_link_is_enqueued() -> None:
    """ORACLE: C1 / §3.2."""
    policy = _web_policy({"example.com": ("1.1.1.1",)})
    targets = enqueue_targets(
        [{"href": "/docs"}],
        page_url="http://example.com/",
        base_netloc="example.com",
        url_pattern=None,
        policy=policy,
    )
    assert targets == ["http://example.com/docs"]


def test_c1_empty_offsite_and_fragment_links_are_handled() -> None:
    """ORACLE: C1 / §3.2 — empty href, other netloc, and query/fragment stripping."""
    policy = _web_policy({"example.com": ("1.1.1.1",)})
    targets = enqueue_targets(
        [
            {"href": ""},
            {"href": "https://other.example/x"},
            {"href": "/docs?page=2#section"},
            {"href": "guide#top"},
        ],
        page_url="http://example.com/",
        base_netloc="example.com",
        url_pattern=None,
        policy=policy,
    )
    assert targets == ["http://example.com/docs", "http://example.com/guide"]


def test_c1_no_policy_reproduces_the_old_behaviour() -> None:
    """ORACLE: C1 / §3.2 — the policy is injected; None means today's rules."""
    targets = enqueue_targets(
        [{"href": "/admin"}, {"href": "https://other.example/x"}, {"href": ""}],
        page_url="http://example.com/",
        base_netloc="example.com",
        url_pattern=None,
        policy=None,
    )
    assert targets == ["http://example.com/admin"]


# ---------------------------------------------------------------------------
# C2: the browser route hook
# ---------------------------------------------------------------------------


def _hook_policy() -> UrlPolicy:
    return _web_policy(
        {
            "docs.example": ("1.1.1.1",),
            "cdn.example": ("1.1.1.1",),
            "169.254.169.254": ("169.254.169.254",),
        }
    )


async def test_c2_private_navigation_is_aborted() -> None:
    """ORACLE: C2 / §3.2 — a navigation to a link-local address is aborted."""
    hook = make_page_route_hook(_hook_policy())
    route = _Route(PRIVATE_URL, navigation=True)

    await hook(route)

    assert "abort" in route.calls
    assert "fetch" not in route.calls


async def test_c2_allowed_navigation_is_not_aborted() -> None:
    """ORACLE: C2 / A-3 — the handler fetches the navigation itself."""
    hook = make_page_route_hook(_hook_policy())
    route = _Route("https://docs.example/page", navigation=True, response=_Response(200))

    await hook(route)

    assert "abort" not in route.calls
    assert route.fetch_kwargs is not None
    assert route.fetch_kwargs.get("max_redirects") == 0
    assert "fulfill" in route.calls


async def test_c2_data_scheme_passes() -> None:
    """ORACLE: C2 / A-3 — data:, blob: and about: are not network requests."""
    hook = make_page_route_hook(_hook_policy())
    route = _Route("data:text/html,<p>hi</p>", navigation=True)

    await hook(route)

    assert "continue_" in route.calls
    assert "abort" not in route.calls


async def test_c2_file_scheme_is_aborted() -> None:
    """ORACLE: C2 / A-3 — every other scheme fails closed."""
    hook = make_page_route_hook(_hook_policy())
    route = _Route("file:///etc/hosts", navigation=True)

    await hook(route)

    assert "abort" in route.calls
    assert "continue_" not in route.calls


async def test_c2_private_subresource_is_aborted() -> None:
    """ORACLE: C2 / §3.2 — sub-resource requests are checked by URL."""
    hook = make_page_route_hook(_hook_policy())
    route = _Route(PRIVATE_URL, navigation=False)

    await hook(route)

    assert "abort" in route.calls
    assert "fetch" not in route.calls


async def test_c2_public_subresource_continues() -> None:
    """ORACLE: C2 / §3.2."""
    hook = make_page_route_hook(_hook_policy())
    route = _Route("https://cdn.example/app.js", navigation=False)

    await hook(route)

    assert "continue_" in route.calls
    assert "abort" not in route.calls


async def test_c2_redirect_to_private_address_is_aborted() -> None:
    """ORACLE: C2 / A-3 — page.route sees only the first URL of a redirect."""
    hook = make_page_route_hook(_hook_policy())
    route = _Route(
        "https://docs.example/redirect",
        navigation=True,
        response=_Response(302, {"location": PRIVATE_URL}),
    )

    await hook(route)

    assert "abort" in route.calls
    assert "fulfill" not in route.calls


async def test_c2_redirect_to_allowed_address_is_fulfilled() -> None:
    """ORACLE: C2 / A-3 — the next hop re-enters the handler as a new request."""
    hook = make_page_route_hook(_hook_policy())
    response = _Response(302, {"location": "https://docs.example/next"})
    route = _Route("https://docs.example/redirect", navigation=True, response=response)

    await hook(route)

    assert "abort" not in route.calls
    assert "fulfill" in route.calls
    assert route.fulfilled_with is response


async def test_c2_fetch_failure_fails_closed() -> None:
    """ORACLE: C2 / A-3 — route.fetch raising aborts rather than continuing."""
    hook = make_page_route_hook(_hook_policy())
    route = _Route(
        "https://docs.example/page",
        navigation=True,
        fetch_error=RuntimeError("network failed"),
    )

    await hook(route)

    assert "abort" in route.calls
    assert "fulfill" not in route.calls


# ---------------------------------------------------------------------------
# final_page_url: the backstop's input
# ---------------------------------------------------------------------------


def test_final_page_url_prefers_the_redirected_url() -> None:
    """ORACLE: A-3.3 — the page is judged by where it ended up."""
    result = SimpleNamespace(redirected_url="http://169.254.169.254/", url="http://ok.example/")
    assert final_page_url(result, "http://ok.example/") == "http://169.254.169.254/"


def test_final_page_url_falls_back_to_url() -> None:
    """ORACLE: A-3.3 — no redirect means the result's own url."""
    result = SimpleNamespace(redirected_url=None, url="http://ok.example/")
    assert final_page_url(result, "http://requested.example/") == "http://ok.example/"


# ---------------------------------------------------------------------------
# The backstop and the A-6 clamp, through _crawl_url
# ---------------------------------------------------------------------------


async def test_a3_dropped_page_is_reported_in_the_result_text(
    mcp_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ORACLE: A-3.3 / N8 — a refused final URL is dropped and counted."""
    _patch_mcp_effects(monkeypatch)
    recorder: dict = {}

    def _refuse(kwargs: dict) -> None:
        callback = kwargs.get("on_refused")
        assert callback is not None, "crawl_site must receive on_refused"
        callback(
            PRIVATE_URL,
            UrlRefusal(RefusalReason.PRIVATE_ADDRESS, "link-local address"),
        )

    doc = SimpleNamespace(
        url="https://docs.example/",
        title="Docs",
        content="hello",
        markdown="hello",
        html="<html>hello</html>",
        cleaned_html="<html>hello</html>",
        metadata={},
        links={},
        success=True,
    )
    _install_fake_crawler(monkeypatch, recorder, site_docs=[doc], on_site=_refuse)

    result = await mcp_server._crawl_url(
        {"url": "https://docs.example/", "max_pages": 2}
    )

    text = _result_text(result)
    assert "Dropped 1 page(s) by policy:" in text, text
    assert PRIVATE_URL in text, text
    assert "private_address" in text, text


async def test_a6_max_pages_is_capped_at_50(
    mcp_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ORACLE: A-6 / D-5 — the shell-level clamp always runs."""
    _patch_mcp_effects(monkeypatch)
    recorder: dict = {}
    _install_fake_crawler(monkeypatch, recorder, site_docs=[])

    result = await mcp_server._crawl_url(
        {"url": "https://docs.example/", "max_pages": 500}
    )

    assert recorder.get("crawl_site") is True
    assert recorder.get("max_pages") == 50
    assert "max_pages capped at 50" in _result_text(result)


@pytest.mark.parametrize("value", [0, -1, "abc"])
async def test_a6_max_pages_below_one_or_not_an_int_is_refused(
    value, mcp_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ORACLE: A-6 / §3.4 — max_pages < 1 is an isError, never a crawl."""
    _patch_mcp_effects(monkeypatch)
    recorder: dict = {}
    _install_fake_crawler(monkeypatch, recorder, site_docs=[])

    result = await mcp_server._crawl_url({"url": "https://docs.example/", "max_pages": value})

    assert result.isError
    assert "max_pages" in _result_text(result)
    assert "crawl_site" not in recorder and "crawl_single" not in recorder


async def test_a6_max_pages_one_uses_crawl_single(
    mcp_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ORACLE: A-6 — one page is the existing single-page path."""
    _patch_mcp_effects(monkeypatch)
    recorder: dict = {}
    _install_fake_crawler(monkeypatch, recorder, site_docs=[])

    await mcp_server._crawl_url({"url": "https://docs.example/", "max_pages": 1})

    assert recorder.get("crawl_single") is True
    assert "crawl_site" not in recorder


# ---------------------------------------------------------------------------
# Round-1 review: the post-crawl backstop (MED-4) and the policy cache (LOW-3)
# ---------------------------------------------------------------------------


class _FakeResult:
    """The slice of a crawl4ai result the backstop reads."""

    def __init__(self, *, redirected_url: str, links: dict | None = None) -> None:
        self.url = "https://docs.example/"
        self.redirected_url = redirected_url
        self.success = True
        self.metadata = {}
        self.markdown = "# Docs"
        self.html = "<html><body>Docs</body></html>"
        self.links = links if links is not None else {"internal": []}


class _FakeAsyncCrawler:
    """A browser replacement: ``arun`` returns a result with a chosen final URL."""

    result: _FakeResult

    def __init__(self, *args, **kwargs) -> None:
        self.crawler_strategy = SimpleNamespace(set_hook=lambda *a, **k: None)

    async def __aenter__(self) -> "_FakeAsyncCrawler":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def arun(self, url: str | None = None, config=None):
        return self.result


def _install_fake_browser(monkeypatch: pytest.MonkeyPatch, result: _FakeResult) -> None:
    import crawl4ai

    _FakeAsyncCrawler.result = result
    monkeypatch.setattr(crawl4ai, "AsyncWebCrawler", _FakeAsyncCrawler)


def _backstop_policy() -> UrlPolicy:
    return _web_policy(
        {
            "docs.example": ("1.1.1.1",),
            "169.254.169.254": ("169.254.169.254",),
        }
    )


async def test_a33_single_page_with_a_blocked_final_url_is_refused(
    mcp_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ORACLE: A-3.3 / MED-4 — the backstop raises even when the route hook saw nothing."""
    _install_fake_browser(monkeypatch, _FakeResult(redirected_url=PRIVATE_URL))
    crawler = crawler_module.DocCrawler(url_policy=_backstop_policy())

    with pytest.raises(crawler_module.UrlRefusedError) as excinfo:
        await crawler.crawl_single("https://docs.example/", use_cache=False)

    assert excinfo.value.refusal.reason is RefusalReason.PRIVATE_ADDRESS


async def test_a33_site_crawl_drops_a_page_with_a_blocked_final_url(
    mcp_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ORACLE: A-3.3 / MED-4 — the page is dropped, reported, and not indexed."""
    _install_fake_browser(monkeypatch, _FakeResult(redirected_url=PRIVATE_URL))
    crawler = crawler_module.DocCrawler(url_policy=_backstop_policy())
    dropped: list[tuple[str, UrlRefusal]] = []

    documents = await crawler.crawl_site(
        "https://docs.example/",
        max_pages=2,
        use_cache=False,
        on_refused=lambda url, refusal: dropped.append((url, refusal)),
    )

    assert documents == []
    assert len(dropped) == 1, dropped
    assert dropped[0][0] == PRIVATE_URL, dropped
    assert dropped[0][1].reason is RefusalReason.PRIVATE_ADDRESS


async def test_low3_policy_does_not_serve_a_cached_page(
    mcp_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ORACLE: LOW-3 — a cached page has no backstop, so a policy crawl skips the cache."""
    _install_fake_browser(
        monkeypatch,
        _FakeResult(redirected_url="https://docs.example/", links={}),
    )
    crawler = crawler_module.DocCrawler(url_policy=_backstop_policy())

    cache_reads: list[str] = []
    real_get_cached = crawler._get_cached

    def _record(url: str):
        cache_reads.append(url)
        return real_get_cached(url)

    monkeypatch.setattr(crawler, "_get_cached", _record)

    documents = await crawler.crawl_site(
        "https://docs.example/", max_pages=1, use_cache=True
    )

    assert documents, "the fake browser returned a page"
    assert cache_reads == [], (
        "a policy crawl must not serve a page whose final URL was never judged: "
        f"{cache_reads!r}"
    )


# ---------------------------------------------------------------------------
# Round-2 review: LOW-5 deferred to 0.3.0 (MED-5)
# ---------------------------------------------------------------------------


class _RedirectAwareFakeCrawler:
    """Answers per URL and records every request a site crawl makes."""

    calls: list[str] = []

    def __init__(self, *args, **kwargs) -> None:
        self.crawler_strategy = SimpleNamespace(set_hook=lambda *a, **k: None)

    async def __aenter__(self) -> "_RedirectAwareFakeCrawler":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def arun(self, url: str | None = None, config=None):
        assert url is not None
        _RedirectAwareFakeCrawler.calls.append(url)
        if url == "http://example.com/":
            return _FakeResult(
                redirected_url="http://www.example.com/",
                links={"internal": [{"href": "/page"}]},
            )
        return _FakeResult(redirected_url=url, links={"internal": []})


async def test_med5_a_relative_link_is_enqueued_when_the_start_redirects_to_another_host(
    mcp_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ORACLE: MED-5 — LOW-5 is deferred, so links resolve against the start URL.

    The link filter compares a candidate's netloc with the *start* URL's. If a
    relative link were resolved against the redirected URL of an apex -> www
    crawl, the candidate would be on another host and dropped: the crawl would
    stop after one page. This pins the 0.2.0 resolution the round-2 review
    asked to restore; LOW-5 (final-URL attribution) is deferred to 0.3.0.
    """
    import crawl4ai

    _RedirectAwareFakeCrawler.calls = []
    monkeypatch.setattr(crawl4ai, "AsyncWebCrawler", _RedirectAwareFakeCrawler)

    crawler = crawler_module.DocCrawler()  # no policy: the CLI path
    await crawler.crawl_site("http://example.com/", max_pages=5, use_cache=False)

    assert _RedirectAwareFakeCrawler.calls == [
        "http://example.com/",
        "http://example.com/page",
    ], (
        "the relative link of a page that redirected to another host was not "
        f"enqueued: {_RedirectAwareFakeCrawler.calls!r}"
    )


# ---------------------------------------------------------------------------
# Round-4 review: crawl4ai's anti-bot replay (retry / browser-less fallback)
# ---------------------------------------------------------------------------


async def test_round4_both_crawl_paths_pin_crawl4ais_replay_off(
    mcp_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ORACLE: round-4 item 3 — the browser-less fallback must stay impossible.

    The retry itself is guarded: it calls ``crawler_strategy.crawl`` again, and
    the ``on_page_context_created`` hook docugraph installs runs for it (proven
    against the real browser in tests/integration/test_crawl_antibot.py). The
    fallback is not: ``await fallback_fetch_function(url)`` is a plain HTTP
    fetch in Python, outside the browser and outside every guard. crawl4ai
    0.9.3 defaults both off, and this pins them so a future default cannot
    switch the fallback on. The stand-in config models exactly that future
    release; without the pin both attributes stay on.
    """
    import crawl4ai

    real_config = crawl4ai.CrawlerRunConfig
    seen: list = []

    def _future_default_config(*args, **kwargs):
        """A crawl4ai whose anti-bot replay is on by default."""
        kwargs.setdefault("max_retries", 3)
        kwargs.setdefault("fallback_fetch_function", lambda _url: "")
        config = real_config(*args, **kwargs)
        seen.append(config)
        return config

    monkeypatch.setattr(crawl4ai, "CrawlerRunConfig", _future_default_config)
    _install_fake_browser(
        monkeypatch, _FakeResult(redirected_url="https://docs.example/")
    )

    crawler = crawler_module.DocCrawler()  # no policy: the CLI path too
    await crawler.crawl_single("https://docs.example/", use_cache=False)
    await crawler.crawl_site("https://docs.example/", max_pages=1, use_cache=False)

    assert len(seen) == 2, "both crawl paths must build a crawl4ai config"
    for config in seen:
        assert config.max_retries == 0, (
            "the anti-bot retry was not pinned off: "
            f"{getattr(config, 'max_retries', None)!r}"
        )
        assert config.fallback_fetch_function is None, (
            "crawl4ai's browser-less fallback fetch was not pinned off: "
            f"{getattr(config, 'fallback_fetch_function', None)!r}"
        )
