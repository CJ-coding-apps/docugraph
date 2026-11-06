"""C4 (page cap through MCP) and C6 (service workers cannot bypass the policy).

Both need a real headless Chromium; the skip reason names the install command.
C4 goes through ``_crawl_url`` with the vector store stubbed, so the claim is
about the cap the shell applies, not about embeddings. C6's adjudicator is the
blocked server's log: a registered service worker would fetch it outside
``page.route``, so the log must stay empty.
"""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from docugraph.core.config import reset_config
from docugraph.ingestion.crawler import DocCrawler
from docugraph.interfaces import mcp_server
from docugraph.security.url_policy import UrlKind, UrlPolicy


def _chromium_installed() -> bool:
    """True when Playwright's Chromium executable is present on this machine."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception:  # pragma: no cover - playwright is a declared dependency
        return False
    try:
        with sync_playwright() as playwright:
            return Path(playwright.chromium.executable_path).exists()
    except Exception:  # pragma: no cover - a broken browser install is a skip
        return False


pytestmark = pytest.mark.skipif(
    not _chromium_installed(),
    reason="Chromium not installed; run `uv run playwright install chromium`",
)


class _Null(int):
    """Zero that can be awaited, called and indexed; see test_crawl_guard."""

    def __new__(cls) -> "_Null":
        return int.__new__(cls, 0)

    def __await__(self):
        if False:
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


def _make_handler(routes: dict[str, tuple[int, dict[str, str], bytes]], log: list[str]):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            log.append(self.path)
            status, headers, body = routes.get(
                self.path, (404, {"Content-Type": "text/plain"}, b"not found")
            )
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

    return Handler


def _start(
    host: str,
    routes: dict[str, tuple[int, dict[str, str], bytes]],
    log: list[str],
) -> tuple[ThreadingHTTPServer, Thread]:
    server = ThreadingHTTPServer((host, 0), _make_handler(routes, log))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop(server: ThreadingHTTPServer, thread: Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _html(body: str) -> tuple[int, dict[str, str], bytes]:
    return 200, {"Content-Type": "text/html; charset=utf-8"}, body.encode()


def _result_text(result) -> str:
    return "\n".join(
        item.text
        for item in getattr(result, "content", [])
        if getattr(item, "text", None) is not None
    )


@pytest.fixture
def private_crawl_config(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """The intranet setting on, with the data dir isolated and config re-read."""
    monkeypatch.setenv("DOCUGRAPH_STORAGE__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DOCUGRAPH_CRAWLER__ALLOW_PRIVATE_NETWORKS", "true")
    monkeypatch.delenv("DOCUGRAPH_CRAWLER__ALLOWED_DOMAINS", raising=False)
    reset_config()
    yield
    monkeypatch.delenv("DOCUGRAPH_CRAWLER__ALLOW_PRIVATE_NETWORKS", raising=False)
    reset_config()


async def test_c4_mcp_caps_the_crawl_at_fifty_pages(
    private_crawl_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ORACLE: C4 / A-6 / D-5 — 60 links, max_pages 500, at most 50 fetched."""
    page_count = 60
    routes: dict[str, tuple[int, dict[str, str], bytes]] = {}
    links = "".join(f'<a href="/page/{i}">page {i}</a>' for i in range(page_count))
    routes["/"] = _html(f"<html><body>{links}</body></html>")
    for i in range(page_count):
        routes[f"/page/{i}"] = _html(f"<html><body>page {i}</body></html>")

    log: list[str] = []
    server, thread = _start("127.0.0.1", routes, log)
    port = server.server_address[1]

    monkeypatch.setattr(mcp_server, "get_vector_store", lambda: _Null())
    try:
        result = await asyncio.wait_for(
            mcp_server._crawl_url(
                {"url": f"http://127.0.0.1:{port}/", "max_pages": 500}
            ),
            timeout=240,
        )
    finally:
        _stop(server, thread)

    page_requests = sum(1 for path in log if path == "/" or path.startswith("/page/"))
    assert page_requests >= 1, "the crawler never fetched anything"
    assert page_requests <= 50, (
        f"the shell crawled {page_requests} pages; max_pages must cap at 50"
    )
    assert "max_pages capped at 50" in _result_text(result)


async def test_c6_a_service_worker_cannot_reach_a_blocked_host() -> None:
    """ORACLE: C6 / A-10 — the init script removes navigator.serviceWorker.

    Two adjudicators, because they fail differently. The page reports what it
    saw (``sw:undefined`` when the init script ran, ``sw:object`` when it did
    not) and the blocked host's log must stay empty. Measured: without the init
    script a top-level page registers the worker and its install handler does
    reach the blocked host, so this is a real request, not a hypothetical one.
    ``_KEEP_OPEN`` holds the document open long enough for that request to be
    attempted before the crawler closes the browser.

    The blocked host is named ``localhost`` and answered by a server on
    127.0.0.1, rather than bound to 127.0.0.2. On macOS a connection to a
    non-primary loopback alias fails inside the OS (the client's selected
    source address equals the alias, and the server's accepted socket raises
    ``ENOTCONN``), so a ``127.0.0.2`` listener is unreachable whether or not the
    policy holds -- the log would be empty for the wrong reason. The name is
    reachable here and refused by ``allowed_domains``.
    """
    blocked_log: list[str] = []
    blocked_routes = {
        "/blocked": (
            200,
            {
                "Content-Type": "text/plain",
                "Access-Control-Allow-Origin": "*",
            },
            b"blocked",
        )
    }
    blocked_server, blocked_thread = _start("127.0.0.1", blocked_routes, blocked_log)
    blocked_port = blocked_server.server_address[1]

    start_log: list[str] = []
    start_routes = {
        "/": _html(
            "<html><head><title>fixture</title></head><body>"
            "<p>This fixture page carries enough readable text for Crawl4AI's "
            "structural check to accept it as a real document.</p><script>"
            "document.title = 'sw:' + typeof navigator.serviceWorker;"
            "if (navigator.serviceWorker) {"
            "  navigator.serviceWorker.register('/sw.js');"
            "}"
            f"{_KEEP_OPEN}"
            "</script>page</body></html>"
        ),
        "/sw.js": (
            200,
            {"Content-Type": "application/javascript"},
            (
                "self.addEventListener('install', (event) => {"
                f"  event.waitUntil(fetch('http://localhost:{blocked_port}/blocked'));"
                "});"
            ).encode(),
        ),
    }
    start_server, start_thread = _start("127.0.0.1", start_routes, start_log)
    start_port = start_server.server_address[1]

    policy = UrlPolicy(
        kind=UrlKind.WEB,
        allow_private=True,
        allowed_domains=("127.0.0.1",),
    )
    crawler = DocCrawler(url_policy=policy)
    document = None
    try:
        try:
            document = await asyncio.wait_for(
                crawler.crawl_single(
                    f"http://127.0.0.1:{start_port}/", use_cache=False
                ),
                timeout=90,
            )
        except Exception:  # noqa: BLE001 - the log is the adjudicator
            pass
        # Give a registered worker time to run its install handler.
        await asyncio.sleep(1.0)
        assert "/" in start_log, "the crawler never requested the start page"
        assert document is not None and document.title == "sw:undefined", (
            "the init script did not remove navigator.serviceWorker before page "
            f"scripts ran; the page reported {getattr(document, 'title', None)!r}"
        )
        assert blocked_log == [], (
            "a service worker contacted the blocked host outside page.route: "
            f"{blocked_log}"
        )
    finally:
        _stop(start_server, start_thread)
        _stop(blocked_server, blocked_thread)


# ---------------------------------------------------------------------------
# Round-1 review: frames, popups and workers (MED-2/MED-3: C6b, C7, C8)
# ---------------------------------------------------------------------------

# Blocking the load event keeps the page open while the request is attempted:
# crawl4ai waits for domcontentloaded, and the crawler closes the browser as
# soon as arun returns, so a request issued after that point proves nothing.
_KEEP_OPEN = "const end = Date.now() + 3000; while (Date.now() < end) {}"


def _blocked_routes() -> dict[str, tuple[int, dict[str, str], bytes]]:
    return {
        "/blocked": (
            200,
            {
                "Content-Type": "text/plain",
                "Access-Control-Allow-Origin": "*",
            },
            b"blocked",
        )
    }


async def _crawl_ignoring_the_outcome(crawler: DocCrawler, url: str) -> None:
    try:
        await asyncio.wait_for(crawler.crawl_single(url, use_cache=False), timeout=90)
    except Exception:  # noqa: BLE001 - the server log is the adjudicator
        pass


async def test_c6b_service_worker_from_an_about_blank_iframe_cannot_reach_the_blocked_host() -> None:
    """ORACLE: MED-3 / C6b — the init script must also reach an about:blank frame.

    About:blank cannot register a worker at all (measured: ``InvalidStateError``,
    because the document URL is not a fetchable script base), so the blocked
    log alone would stay empty whether or not the init script ran. The frame's
    own ``navigator`` is the adjudicator: ``frame-sw:undefined`` proves the init
    script reached it before the frame's scripts. The blocked host is named
    ``localhost`` for the reason given in C6.
    """
    blocked_log: list[str] = []
    blocked_server, blocked_thread = _start("127.0.0.1", _blocked_routes(), blocked_log)
    blocked_port = blocked_server.server_address[1]

    start_log: list[str] = []
    start_routes = {
        "/": _html(
            "<html><head><title>fixture</title></head><body>"
            "<p>This fixture page carries enough readable text for Crawl4AI's "
            "structural check to accept it as a real document.</p>"
            "<iframe id='frame'></iframe><script>"
            "const frame = document.getElementById('frame');"
            "const sw = frame.contentWindow.navigator.serviceWorker;"
            "document.title = 'frame-sw:' + typeof sw;"
            "if (sw) { try { sw.register('/sw.js'); } catch (error) {} }"
            f"{_KEEP_OPEN}"
            "</script>page</body></html>"
        ),
        "/sw.js": (
            200,
            {"Content-Type": "application/javascript"},
            (
                "self.addEventListener('install', (event) => {"
                f"  event.waitUntil(fetch('http://localhost:{blocked_port}/blocked'));"
                "});"
            ).encode(),
        ),
    }
    start_server, start_thread = _start("127.0.0.1", start_routes, start_log)
    start_port = start_server.server_address[1]

    policy = UrlPolicy(
        kind=UrlKind.WEB,
        allow_private=True,
        allowed_domains=("127.0.0.1",),
    )
    crawler = DocCrawler(url_policy=policy)
    document = None
    try:
        try:
            document = await asyncio.wait_for(
                crawler.crawl_single(f"http://127.0.0.1:{start_port}/", use_cache=False),
                timeout=90,
            )
        except Exception:  # noqa: BLE001 - the log is the adjudicator
            pass
        assert "/" in start_log, "the crawler never requested the start page"
        assert document is not None and document.title == "frame-sw:undefined", (
            "the init script did not reach the about:blank frame; the frame "
            f"reported {getattr(document, 'title', None)!r}"
        )
        assert blocked_log == [], (
            "a service worker in an about:blank frame contacted the blocked host: "
            f"{blocked_log}"
        )
    finally:
        _stop(start_server, start_thread)
        _stop(blocked_server, blocked_thread)


async def test_c7_popup_requests_are_routed() -> None:
    """ORACLE: MED-2 / C7 — window.open opens a new page; the context routing must cover it."""
    blocked_log: list[str] = []
    blocked_server, blocked_thread = _start("127.0.0.1", _blocked_routes(), blocked_log)
    blocked_port = blocked_server.server_address[1]

    start_log: list[str] = []
    start_routes = {
        "/": _html(
            "<html><body><script>"
            f"window.open('http://localhost:{blocked_port}/blocked');"
            f"{_KEEP_OPEN}"
            "</script>page</body></html>"
        )
    }
    start_server, start_thread = _start("127.0.0.1", start_routes, start_log)
    start_port = start_server.server_address[1]

    policy = UrlPolicy(
        kind=UrlKind.WEB,
        allow_private=True,
        allowed_domains=("127.0.0.1",),
    )
    crawler = DocCrawler(url_policy=policy)
    try:
        await _crawl_ignoring_the_outcome(crawler, f"http://127.0.0.1:{start_port}/")
        assert "/" in start_log, "the crawler never requested the start page"
        assert blocked_log == [], (
            "a popup contacted the blocked host outside the policy: "
            f"{blocked_log}"
        )
    finally:
        _stop(start_server, start_thread)
        _stop(blocked_server, blocked_thread)


async def test_c8_dedicated_worker_requests_are_routed() -> None:
    """ORACLE: MED-2 / C8 — a Worker's fetch must pass the same route handler."""
    blocked_log: list[str] = []
    blocked_server, blocked_thread = _start("127.0.0.1", _blocked_routes(), blocked_log)
    blocked_port = blocked_server.server_address[1]

    start_log: list[str] = []
    start_routes = {
        "/": _html(
            "<html><body><script>"
            "const worker = new Worker('/worker.js');"
            f"{_KEEP_OPEN}"
            "</script>page</body></html>"
        ),
        "/worker.js": (
            200,
            {"Content-Type": "application/javascript"},
            f"fetch('http://localhost:{blocked_port}/blocked').catch(() => {{}});".encode(),
        ),
    }
    start_server, start_thread = _start("127.0.0.1", start_routes, start_log)
    start_port = start_server.server_address[1]

    policy = UrlPolicy(
        kind=UrlKind.WEB,
        allow_private=True,
        allowed_domains=("127.0.0.1",),
    )
    crawler = DocCrawler(url_policy=policy)
    try:
        await _crawl_ignoring_the_outcome(crawler, f"http://127.0.0.1:{start_port}/")
        assert "/" in start_log, "the crawler never requested the start page"
        assert blocked_log == [], (
            "a dedicated worker contacted the blocked host outside the policy: "
            f"{blocked_log}"
        )
    finally:
        _stop(start_server, start_thread)
        _stop(blocked_server, blocked_thread)
