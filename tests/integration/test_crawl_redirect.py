"""C3/C5: redirect enforcement, adjudicated by the blocked server's request log.

The crawler runs a real headless Chromium against three servers, all bound to
127.0.0.1 and told apart by the host name in the URL (``127.0.0.1``, ``127.1``,
``localhost``). A second loopback alias such as 127.0.0.2 cannot be used on
macOS: connecting to a non-primary alias makes the OS pick that same address as
the client's source, and the connection dies (the server's accepted socket
raises ``ENOTCONN``), so a 127.0.0.2 listener is unreachable regardless of the
policy and the log would be empty for the wrong reason. The three names above
all reach 127.0.0.1 and are refused by ``allowed_domains``.

This suite fails when DOCUGRAPH_REQUIRE_BROWSER=1 and Chromium is missing.
"""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from docugraph.ingestion.crawler import DocCrawler
from docugraph.security.url_policy import UrlKind, UrlPolicy

pytestmark = pytest.mark.usefixtures("require_chromium")


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
        "127.0.0.1", {"/blocked": _html("<html>secret</html>")}, blocked_log
    )
    blocked_port = blocked_server.server_address[1]

    start_log: list[str] = []
    start_server, start_thread = _start(
        "127.0.0.1",
        {"/": _redirect(f"http://localhost:{blocked_port}/blocked")},
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
        await _crawl_without_caring_about_the_error(crawler, f"http://127.0.0.1:{start_port}/")
        assert "/" in start_log, "the crawler never requested the start page"
        assert blocked_log == [], (
            f"the blocked host was contacted despite the policy refusal: {blocked_log}"
        )
    finally:
        _stop(start_server, start_thread)
        _stop(blocked_server, blocked_thread)


async def test_c5_second_hop_to_a_blocked_host_is_never_fetched() -> None:
    """ORACLE: C5 / A-3 — 302 -> 302 -> blocked; the final host stays untouched."""
    blocked_log: list[str] = []
    blocked_server, blocked_thread = _start(
        "127.0.0.1", {"/blocked": _html("<html>secret</html>")}, blocked_log
    )
    blocked_port = blocked_server.server_address[1]

    middle_log: list[str] = []
    middle_server, middle_thread = _start(
        "127.0.0.1",
        {"/step2": _redirect(f"http://localhost:{blocked_port}/blocked")},
        middle_log,
    )
    middle_port = middle_server.server_address[1]

    start_log: list[str] = []
    start_server, start_thread = _start(
        "127.0.0.1",
        {"/": _redirect(f"http://127.1:{middle_port}/step2")},
        start_log,
    )
    start_port = start_server.server_address[1]

    policy = UrlPolicy(
        kind=UrlKind.WEB,
        allow_private=True,
        allowed_domains=("127.0.0.1", "127.1"),
    )
    crawler = DocCrawler(url_policy=policy)

    try:
        await _crawl_without_caring_about_the_error(crawler, f"http://127.0.0.1:{start_port}/")
        assert "/" in start_log, "the crawler never requested the start page"
        assert middle_log.count("/step2") == 1, (
            f"the allowed middle hop must be served exactly once: {middle_log}"
        )
        assert blocked_log == [], (
            f"the blocked host was contacted through the second hop: {blocked_log}"
        )
    finally:
        _stop(start_server, start_thread)
        _stop(middle_server, middle_thread)
        _stop(blocked_server, blocked_thread)


async def test_p8_redirect_hop_to_a_metadata_host_is_aborted() -> None:
    """ORACLE: A-2 / P8 — a redirect hop to a metadata host is never fetched."""
    blocked_log: list[str] = []
    blocked_server, blocked_thread = _start(
        "127.0.0.1", {"/blocked": _html("<html>secret</html>")}, blocked_log
    )
    blocked_port = blocked_server.server_address[1]

    start_log: list[str] = []
    start_server, start_thread = _start(
        "127.0.0.1",
        {"/start": _redirect(f"http://localhost:{blocked_port}/blocked")},
        start_log,
    )
    start_port = start_server.server_address[1]

    def _resolve(host: str) -> tuple[str, ...]:
        if host == "localhost":
            return ("169.254.169.254",)
        return ("127.0.0.1",)

    policy = UrlPolicy(kind=UrlKind.WEB, allow_private=True, resolver=_resolve)
    crawler = DocCrawler(url_policy=policy)

    try:
        # As in C3/C5, the adjudicator is the blocked server's log: an aborted
        # navigation surfaces either as a refusal or as a failed result, and the
        # claim is that the hop was never made.
        await _crawl_without_caring_about_the_error(crawler, f"http://127.0.0.1:{start_port}/start")
        assert blocked_log == [], f"the redirect hop to a metadata host was fetched: {blocked_log}"
    finally:
        _stop(start_server, start_thread)
        _stop(blocked_server, blocked_thread)
