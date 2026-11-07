"""Round-4 item 3: crawl4ai's anti-bot retry stays inside the guarded context.

crawl4ai >= 0.9 classifies a page as anti-bot and retries ``crawl()`` up to
``CrawlerRunConfig.max_retries`` times; after the retries it can hand the URL
to ``fallback_fetch_function``. The retry calls
``crawler_strategy.crawl(url, config)`` again, which runs the same
``on_page_context_created`` hook docugraph installs, so the retry navigates in
a page and context that carry the context route, the websocket routing and the
init script. The fallback is a plain HTTP fetch run in Python, outside the
browser and outside every guard, so docugraph pins both off
(``_pin_replay_guards``).

This test drives the retry path directly: the first request answers with an
anti-bot shell, every later request answers 302 to a blocked host, and the
retry must not reach it.

These tests need Chromium; the skip reason names the install command.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from docugraph.ingestion.crawler import _install_page_guards
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

# No visible text and no content element: two structural signals, so crawl4ai
# classifies this as an anti-bot shell and enters the retry path.
_ANTIBOT_SHELL = b"<html><head><script>var a = 1;</script></head><body></body></html>"


def _answer(
    handler: BaseHTTPRequestHandler,
    status: int,
    headers: dict[str, str],
    body: bytes,
) -> None:
    handler.send_response(status)
    for name, value in headers.items():
        handler.send_header(name, value)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


async def test_antibot_retry_navigation_is_routed() -> None:
    """ORACLE: round 4 item 3 — the retry must not reach the blocked host."""
    blocked_log: list[str] = []

    class BlockedHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            blocked_log.append(self.path)
            _answer(self, 200, {"Content-Type": "text/plain"}, b"blocked")

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

    blocked_server = ThreadingHTTPServer(("127.0.0.1", 0), BlockedHandler)
    blocked_thread = Thread(target=blocked_server.serve_forever, daemon=True)
    blocked_thread.start()
    blocked_port = blocked_server.server_address[1]

    start_log: list[str] = []

    class StartHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            start_log.append(self.path)
            if self.path != "/":
                _answer(self, 404, {"Content-Type": "text/plain"}, b"not found")
                return
            if start_log.count("/") == 1:
                _answer(
                    self,
                    200,
                    {"Content-Type": "text/html; charset=utf-8"},
                    _ANTIBOT_SHELL,
                )
                return
            _answer(
                self,
                302,
                {"Location": f"http://localhost:{blocked_port}/blocked"},
                b"",
            )

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

    start_server = ThreadingHTTPServer(("127.0.0.1", 0), StartHandler)
    start_thread = Thread(target=start_server.serve_forever, daemon=True)
    start_thread.start()
    start_port = start_server.server_address[1]

    policy = UrlPolicy(
        kind=UrlKind.WEB,
        allow_private=True,
        allowed_domains=("127.0.0.1",),
    )

    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

    try:
        async with AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False)) as crawler:
            _install_page_guards(crawler, policy)
            result = await crawler.arun(
                url=f"http://127.0.0.1:{start_port}/",
                config=CrawlerRunConfig(
                    wait_until="domcontentloaded",
                    max_retries=1,
                ),
            )

        page_requests = [path for path in start_log if path == "/"]
        assert len(page_requests) == 2, (
            "the anti-bot retry did not run; crawl4ai's retry path was not "
            f"exercised ({start_log!r})"
        )
        assert result.success is False, (
            "the retry was expected to be aborted by the policy; the crawl "
            f"reported {getattr(result, 'error_message', None)!r}"
        )
        assert "Blocked by anti-bot" in (getattr(result, "error_message", "") or ""), (
            "the first attempt was not classified as anti-bot: "
            f"{getattr(result, 'error_message', None)!r}"
        )
        assert blocked_log == [], (
            f"crawl4ai's anti-bot retry navigated outside docugraph's guards: {blocked_log}"
        )
    finally:
        start_server.shutdown()
        start_server.server_close()
        start_thread.join(timeout=5)
        blocked_server.shutdown()
        blocked_server.server_close()
        blocked_thread.join(timeout=5)
