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
    """ORACLE: C6 / A-10 — the init script removes navigator.serviceWorker."""
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
    blocked_server, blocked_thread = _start("127.0.0.2", blocked_routes, blocked_log)
    blocked_port = blocked_server.server_address[1]

    start_log: list[str] = []
    start_routes = {
        "/": _html(
            "<html><body><script>"
            "navigator.serviceWorker.register('/sw.js');"
            "</script>page</body></html>"
        ),
        "/sw.js": (
            200,
            {"Content-Type": "application/javascript"},
            (
                "self.addEventListener('install', (event) => {"
                f"  event.waitUntil(fetch('http://127.0.0.2:{blocked_port}/blocked'));"
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
    try:
        try:
            await asyncio.wait_for(
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
        assert blocked_log == [], (
            "a service worker contacted the blocked host outside page.route: "
            f"{blocked_log}"
        )
    finally:
        _stop(start_server, start_thread)
        _stop(blocked_server, blocked_thread)
