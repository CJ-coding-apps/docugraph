"""C3/C5: redirect enforcement, adjudicated by the blocked server's request log.

The crawler runs a real headless Chromium against three loopback servers. The
policy's resolver is the real one -- every target is a numeric loopback address,
so DNS is local. The blocked server's log is the adjudicator: it must be empty
for the address the policy refused, whatever the browser returned.

These tests need Chromium; the skip reason names the install command.
"""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from docugraph.ingestion.crawler import DocCrawler
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
            # Silence the default per-request stderr line; the log above is ours.
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


def _redirect(location: str) -> tuple[int, dict[str, str], bytes]:
    return 302, {"Location": location}, b""


async def _crawl_without_caring_about_the_error(crawler: DocCrawler, url: str) -> None:
    """Run one crawl; a refusal may surface as an exception or a failed result.

    The adjudicator is the server log, so either surface is acceptable here.
    """
    try:
        await asyncio.wait_for(crawler.crawl_single(url, use_cache=False), timeout=90)
    except Exception:  # noqa: BLE001 - the refusal's surface is not the claim
        return


async def test_c3_redirect_to_a_blocked_host_is_never_fetched() -> None:
    """ORACLE: C3 / §3.2, A-3 — a 302 to a refused host must not reach it."""
    blocked_log: list[str] = []
    blocked_server, blocked_thread = _start(
        "127.0.0.2", {"/blocked": _html("<html>secret</html>")}, blocked_log
    )
    blocked_port = blocked_server.server_address[1]

    start_log: list[str] = []
    start_server, start_thread = _start(
        "127.0.0.1",
        {"/": _redirect(f"http://127.0.0.2:{blocked_port}/blocked")},
        start_log,
    )
    start_port = start_server.server_address[1]

    policy = UrlPolicy(
        kind=UrlKind.WEB,
        allow_private=True,
        allowed_domains=("127.0.0.1",),
    )
    crawler = DocCrawler(url_policy=policy)

    try:
        await _crawl_without_caring_about_the_error(
            crawler, f"http://127.0.0.1:{start_port}/"
        )
        assert "/" in start_log, "the crawler never requested the start page"
        assert blocked_log == [], (
            "the blocked host was contacted despite the policy refusal: "
            f"{blocked_log}"
        )
    finally:
        _stop(start_server, start_thread)
        _stop(blocked_server, blocked_thread)


async def test_c5_second_hop_to_a_blocked_host_is_never_fetched() -> None:
    """ORACLE: C5 / A-3 — 302 -> 302 -> blocked; the final host stays untouched."""
    blocked_log: list[str] = []
    blocked_server, blocked_thread = _start(
        "127.0.0.3", {"/blocked": _html("<html>secret</html>")}, blocked_log
    )
    blocked_port = blocked_server.server_address[1]

    middle_log: list[str] = []
    middle_server, middle_thread = _start(
        "127.0.0.2",
        {"/step2": _redirect(f"http://127.0.0.3:{blocked_port}/blocked")},
        middle_log,
    )
    middle_port = middle_server.server_address[1]

    start_log: list[str] = []
    start_server, start_thread = _start(
        "127.0.0.1",
        {"/": _redirect(f"http://127.0.0.2:{middle_port}/step2")},
        start_log,
    )
    start_port = start_server.server_address[1]

    policy = UrlPolicy(
        kind=UrlKind.WEB,
        allow_private=True,
        allowed_domains=("127.0.0.1", "127.0.0.2"),
    )
    crawler = DocCrawler(url_policy=policy)

    try:
        await _crawl_without_caring_about_the_error(
            crawler, f"http://127.0.0.1:{start_port}/"
        )
        assert "/" in start_log, "the crawler never requested the start page"
        assert middle_log.count("/step2") == 1, (
            "the allowed middle hop must be served exactly once: " f"{middle_log}"
        )
        assert blocked_log == [], (
            "the blocked host was contacted through the second hop: "
            f"{blocked_log}"
        )
    finally:
        _stop(start_server, start_thread)
        _stop(middle_server, middle_thread)
        _stop(blocked_server, blocked_thread)
