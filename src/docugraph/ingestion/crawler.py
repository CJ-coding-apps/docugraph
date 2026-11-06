"""Web crawler using Crawl4AI."""

import asyncio
import hashlib
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin, urlparse, urlsplit

from docugraph.core.config import CrawlerConfig, get_config
from docugraph.core.models import ContentType, Document
from docugraph.security.url_policy import (
    UrlPolicy,
    UrlRefusal,
    UrlRefusedError,
    is_pass_through_scheme,
)

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECT_HOPS = 10

# Runs in every frame before page scripts (A-10). A service worker registered
# by an allowed page fetches outside page.route, so the registration surface is
# removed rather than policed.
_SERVICE_WORKER_REMOVAL_SCRIPT = """
(() => {
  try {
    Object.defineProperty(Navigator.prototype, 'serviceWorker', {
      configurable: true,
      get: () => undefined,
    });
  } catch (error) {
    try { delete Navigator.prototype.serviceWorker; } catch (ignored) {}
  }
})();
"""


def final_page_url(result: Any, requested_url: str) -> str:
    """Where a crawl actually ended up (A-3.3); the backstop judges this value."""
    return getattr(result, "redirected_url", None) or getattr(result, "url", None) or requested_url


def enqueue_targets(
    links: Iterable[Mapping[str, Any]],
    *,
    page_url: str,
    base_netloc: str,
    url_pattern: re.Pattern[str] | None,
    policy: UrlPolicy | None,
) -> list[str]:
    """Turn a page's links into the next crawl targets.

    The rules are the ones ``crawl_site`` always applied (empty href, same
    netloc, optional regex, query and fragment stripped) plus, when a policy is
    injected, one decision per resulting URL. ``policy=None`` is exactly the
    old behaviour.
    """
    targets: list[str] = []
    for link in links:
        href = link.get("href", "")
        if not href:
            continue

        full_url = urljoin(page_url, href)
        parsed = urlparse(full_url)

        if parsed.netloc != base_netloc:
            continue
        if url_pattern is not None and not url_pattern.search(full_url):
            continue

        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if clean_url in targets:
            continue
        if policy is not None and policy.check(clean_url) is not None:
            continue

        targets.append(clean_url)
    return targets


def _header(response: Any, name: str) -> str:
    headers = getattr(response, "headers", None) or {}
    return headers.get(name) or headers.get(name.title()) or ""


def make_page_route_hook(policy: UrlPolicy) -> Callable[[Any], Awaitable[None]]:
    """Build the ``page.route`` handler that enforces the policy (A-3).

    Playwright routes a redirect chain one request at a time and the handler
    never sees the ``Location`` hop, so a navigation is fetched with
    ``max_redirects=0`` and its ``Location`` judged here. Sub-resources are
    judged by their own URL. Every other scheme fails closed except the
    non-network ``data:``/``blob:``/``about:``.
    """

    async def handler(route: Any) -> None:
        request = route.request
        url = request.url
        scheme = urlsplit(url).scheme.lower()

        if scheme not in {"http", "https"}:
            if is_pass_through_scheme(scheme):
                await route.continue_()
            else:
                await route.abort()
            return

        refusal = await asyncio.to_thread(policy.check, url)
        if refusal is not None:
            await route.abort()
            return

        if not request.is_navigation_request():
            await route.continue_()
            return

        # Chromium follows a redirect internally, without re-entering the route
        # handler (measured), so the chain is walked here: every hop is judged
        # before its request is made, and only the final response is handed to
        # the browser.
        current_url = url
        try:
            response = await route.fetch(max_redirects=0)
        except Exception:
            await route.abort()
            return

        for _ in range(_MAX_REDIRECT_HOPS):
            if getattr(response, "status", None) not in _REDIRECT_STATUSES:
                break

            location = _header(response, "location")
            if not location:
                break

            target = urljoin(current_url, location)
            if await asyncio.to_thread(policy.check, target) is not None:
                await route.abort()
                return
            if target == current_url:
                # Self-redirect: no new host can be reached; hand it back.
                break

            current_url = target
            try:
                response = await route.fetch(url=current_url, max_redirects=0)
            except Exception:
                await route.abort()
                return
        else:
            await route.abort()
            return

        await route.fulfill(response=response)

    return handler


async def _close_websocket(websocket: Any) -> None:
    """Websocket routing is present only on newer Playwright; close anyway."""
    await websocket.close()


def _install_page_guards(crawler: Any, policy: UrlPolicy) -> None:
    """Register the route, websocket and service-worker guards on one crawler.

    The route handler is registered on the **context**, not the page: a page
    that calls ``window.open`` opens a new page whose requests a page-level
    route never sees (MED-2). The page-level registration remains only as a
    fallback for a hook that was given no context.
    """
    route_hook = make_page_route_hook(policy)

    async def on_page_context_created(
        page: Any, context: Any = None, _config: Any = None, **_kwargs: Any
    ) -> None:
        target = context if context is not None else page
        await target.route("**/*", route_hook)

        route_web_socket = getattr(target, "route_web_socket", None)
        if route_web_socket is not None:
            await route_web_socket("**/*", _close_websocket)

        if context is not None:
            await context.add_init_script(_SERVICE_WORKER_REMOVAL_SCRIPT)

    crawler.crawler_strategy.set_hook("on_page_context_created", on_page_context_created)


def _pin_replay_guards(crawler_config: Any) -> None:
    """Turn crawl4ai's anti-bot replay off (round-4 item 3).

    crawl4ai >= 0.9 retries a page it classifies as blocked up to
    ``max_retries`` times. That retry calls ``crawler_strategy.crawl`` again,
    which runs ``on_page_context_created``, so it navigates inside the same
    guarded context and is covered by the route, websocket and init-script
    guards. ``fallback_fetch_function`` is different: it is a plain HTTP fetch
    run in Python, outside the browser and outside every guard, and a future
    crawl4ai default must not be able to switch it on. Both are already off by
    default; this pins them. Older crawl4ai has neither attribute.
    """
    if hasattr(crawler_config, "max_retries"):
        crawler_config.max_retries = 0
    if hasattr(crawler_config, "fallback_fetch_function"):
        crawler_config.fallback_fetch_function = None


class DocCrawler:
    """Web crawler for documentation sites using Crawl4AI."""

    def __init__(
        self,
        config: CrawlerConfig | None = None,
        url_policy: UrlPolicy | None = None,
    ):
        """Initialize the crawler.

        Args:
            config: Crawler configuration. If None, uses default from config.
            url_policy: Optional URL policy. When present, every URL the
                crawler asks the browser for is judged by it, and a refusal is
                raised (single page) or dropped (site crawl).
        """
        self._config = config or get_config().crawler
        self._cache_dir = get_config().storage.data_dir / "crawl_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._url_policy = url_policy

    def _check_url(self, url: str) -> UrlRefusal | None:
        """The policy decision, or ``None`` when there is no policy."""
        if self._url_policy is None:
            return None
        return self._url_policy.check(url)

    def _get_cache_key(self, url: str) -> str:
        """Generate a cache key for a URL."""
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def _get_cached(self, url: str) -> Document | None:
        """Get cached document if available and fresh."""
        cache_key = self._get_cache_key(url)
        cache_file = self._cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        import json

        try:
            with open(cache_file) as f:
                data = json.load(f)

            # Check TTL. A cache entry written before timestamps carried a
            # timezone parses as naive, so it is read as UTC rather than
            # subtracted from an aware "now" (which raises).
            cached_at = datetime.fromisoformat(data["cached_at"])
            if cached_at.tzinfo is None:
                cached_at = cached_at.replace(tzinfo=UTC)
            age = (datetime.now(UTC) - cached_at).total_seconds()
            if age > self._config.cache_ttl:
                return None

            return Document(**data["document"])
        except Exception:
            return None

    def _save_cache(self, url: str, document: Document) -> None:
        """Save document to cache."""
        cache_key = self._get_cache_key(url)
        cache_file = self._cache_dir / f"{cache_key}.json"

        import json

        data = {
            "cached_at": datetime.now(UTC).isoformat(),
            "document": document.model_dump(mode="json"),
        }

        with open(cache_file, "w") as f:
            json.dump(data, f)

    async def crawl_single(
        self,
        url: str,
        use_cache: bool = True,
    ) -> Document:
        """Crawl a single URL.

        Args:
            url: The URL to crawl.
            use_cache: Whether to use cached results.

        Returns:
            The crawled document.
        """
        # The start URL is judged before the cache and before the browser.
        refusal = await asyncio.to_thread(self._check_url, url)
        if refusal is not None:
            raise UrlRefusedError(refusal)

        # Under a policy a cached page has no final URL to judge, and it may
        # have been written by an unpoliced CLI crawl (LOW-3), so it is skipped.
        if use_cache and self._url_policy is None:
            cached = self._get_cached(url)
            if cached:
                return cached

        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        from crawl4ai.content_filter_strategy import PruningContentFilter
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
        )

        crawler_config = CrawlerRunConfig(
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                    threshold=0.48,
                    threshold_type="fixed",
                    min_word_threshold=0,
                )
            ),
            wait_until="domcontentloaded",
        )
        _pin_replay_guards(crawler_config)

        async with AsyncWebCrawler(config=browser_config) as crawler:
            if self._url_policy is not None:
                _install_page_guards(crawler, self._url_policy)

            result = await crawler.arun(url=url, config=crawler_config)

            if not result.success:
                raise RuntimeError(f"Failed to crawl {url}: {result.error_message}")

            # Backstop: the page is judged by where it ended up (A-3.3).
            final_url = final_page_url(result, url)
            refusal = await asyncio.to_thread(self._check_url, final_url)
            if refusal is not None:
                raise UrlRefusedError(refusal)

            # Extract title from metadata or content
            title = result.metadata.get("title") if result.metadata else None
            if not title and result.markdown:
                # Try to extract from first heading
                lines = result.markdown.split("\n")
                for line in lines:
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break

            document = Document(
                source_url=url,
                title=title,
                content=result.markdown.fit_markdown
                if hasattr(result.markdown, "fit_markdown")
                else result.markdown,
                content_type=ContentType.MARKDOWN,
                metadata={
                    "raw_html_length": len(result.html) if result.html else 0,
                    "links_count": len(result.links.get("internal", [])) if result.links else 0,
                    "crawled_at": datetime.now(UTC).isoformat(),
                },
            )

            # Cache the result
            if use_cache:
                self._save_cache(url, document)

            return document

    async def crawl_site(
        self,
        start_url: str,
        max_pages: int = 100,
        url_pattern: str | None = None,
        use_cache: bool = True,
        *,
        on_refused: Callable[[str, UrlRefusal], None] | None = None,
    ) -> list[Document]:
        """Crawl a documentation site starting from a URL.

        Args:
            start_url: The starting URL.
            max_pages: Maximum number of pages to crawl.
            url_pattern: Optional regex pattern to filter URLs.
            use_cache: Whether to use cached results.
            on_refused: Called with ``(url, refusal)`` for each page dropped by
                the policy. Effects stay with the caller; the crawler only
                reports.

        Returns:
            List of crawled documents.
        """
        refusal = await asyncio.to_thread(self._check_url, start_url)
        if refusal is not None:
            raise UrlRefusedError(refusal)

        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        from crawl4ai.content_filter_strategy import PruningContentFilter
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
        )

        crawler_config = CrawlerRunConfig(
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                    threshold=0.48,
                    threshold_type="fixed",
                    min_word_threshold=0,
                )
            ),
            wait_until="domcontentloaded",
        )
        _pin_replay_guards(crawler_config)

        # Parse base domain for filtering
        parsed_start = urlparse(start_url)
        base_domain = parsed_start.netloc
        parsed_start.path.rsplit("/", 1)[0] if "/" in parsed_start.path else ""

        visited: set[str] = set()
        to_visit: list[str] = [start_url]
        documents: list[Document] = []

        url_regex = re.compile(url_pattern) if url_pattern else None

        async with AsyncWebCrawler(config=browser_config) as crawler:
            if self._url_policy is not None:
                _install_page_guards(crawler, self._url_policy)

            # The cap is on pages processed (fetched or served from cache), not
            # on pages that yielded a document: an unparseable page must not
            # let the crawl run past its limit.
            while to_visit and len(visited) < max_pages:
                # Get next URL to crawl
                url = to_visit.pop(0)

                if url in visited:
                    continue

                visited.add(url)

                # Under a policy a cached page has no final URL to judge, and
                # it may have been written by an unpoliced CLI crawl (LOW-3).
                if use_cache and self._url_policy is None:
                    cached = self._get_cached(url)
                    if cached:
                        documents.append(cached)
                        continue

                try:
                    result = await crawler.arun(url=url, config=crawler_config)

                    if not result.success:
                        continue

                    # Backstop: the page is judged by where it ended up (A-3.3);
                    # a refused page is dropped and reported, never indexed.
                    final_url = final_page_url(result, url)
                    refusal = await asyncio.to_thread(self._check_url, final_url)
                    if refusal is not None:
                        if on_refused is not None:
                            on_refused(final_url, refusal)
                        continue

                    # Extract title
                    title = result.metadata.get("title") if result.metadata else None
                    if not title and result.markdown:
                        lines = result.markdown.split("\n")
                        for line in lines:
                            if line.startswith("# "):
                                title = line[2:].strip()
                                break

                    content = (
                        result.markdown.fit_markdown
                        if hasattr(result.markdown, "fit_markdown")
                        else result.markdown
                    )

                    document = Document(
                        source_url=url,
                        title=title,
                        content=content,
                        content_type=ContentType.MARKDOWN,
                        metadata={
                            "raw_html_length": len(result.html) if result.html else 0,
                            "crawled_at": datetime.now(UTC).isoformat(),
                        },
                    )

                    documents.append(document)

                    if use_cache:
                        self._save_cache(url, document)

                    # Extract and queue internal links
                    if result.links and "internal" in result.links:
                        targets = await asyncio.to_thread(
                            enqueue_targets,
                            result.links["internal"],
                            page_url=url,
                            base_netloc=base_domain,
                            url_pattern=url_regex,
                            policy=self._url_policy,
                        )
                        for clean_url in targets:
                            if clean_url not in visited and clean_url not in to_visit:
                                to_visit.append(clean_url)

                except Exception as e:
                    # Log error but continue crawling
                    print(f"Error crawling {url}: {e}")
                    continue

                # Respect rate limit
                await asyncio.sleep(1.0 / self._config.rate_limit)

        return documents

    def crawl_single_sync(self, url: str, use_cache: bool = True) -> Document:
        """Synchronous wrapper for crawl_single."""
        return asyncio.run(self.crawl_single(url, use_cache))

    def crawl_site_sync(
        self,
        start_url: str,
        max_pages: int = 100,
        url_pattern: str | None = None,
        use_cache: bool = True,
        *,
        on_refused: Callable[[str, UrlRefusal], None] | None = None,
    ) -> list[Document]:
        """Synchronous wrapper for crawl_site."""
        return asyncio.run(
            self.crawl_site(
                start_url,
                max_pages,
                url_pattern,
                use_cache,
                on_refused=on_refused,
            )
        )


def get_crawler() -> DocCrawler:
    """Get the default crawler instance."""
    return DocCrawler()
