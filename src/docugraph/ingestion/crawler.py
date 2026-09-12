"""Web crawler using Crawl4AI."""

import asyncio
import hashlib
from datetime import datetime
from urllib.parse import urljoin, urlparse

from docugraph.core.config import CrawlerConfig, get_config
from docugraph.core.models import ContentType, Document


class DocCrawler:
    """Web crawler for documentation sites using Crawl4AI."""

    def __init__(self, config: CrawlerConfig | None = None):
        """Initialize the crawler.

        Args:
            config: Crawler configuration. If None, uses default from config.
        """
        self._config = config or get_config().crawler
        self._cache_dir = get_config().storage.data_dir / "crawl_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

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

            # Check TTL
            cached_at = datetime.fromisoformat(data["cached_at"])
            age = (datetime.utcnow() - cached_at).total_seconds()
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
            "cached_at": datetime.utcnow().isoformat(),
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
        # Check cache first
        if use_cache:
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

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=crawler_config)

            if not result.success:
                raise RuntimeError(f"Failed to crawl {url}: {result.error_message}")

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
                    "crawled_at": datetime.utcnow().isoformat(),
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
    ) -> list[Document]:
        """Crawl a documentation site starting from a URL.

        Args:
            start_url: The starting URL.
            max_pages: Maximum number of pages to crawl.
            url_pattern: Optional regex pattern to filter URLs.
            use_cache: Whether to use cached results.

        Returns:
            List of crawled documents.
        """
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

        # Parse base domain for filtering
        parsed_start = urlparse(start_url)
        base_domain = parsed_start.netloc
        parsed_start.path.rsplit("/", 1)[0] if "/" in parsed_start.path else ""

        visited: set[str] = set()
        to_visit: list[str] = [start_url]
        documents: list[Document] = []

        import re

        url_regex = re.compile(url_pattern) if url_pattern else None

        async with AsyncWebCrawler(config=browser_config) as crawler:
            while to_visit and len(documents) < max_pages:
                # Get next URL to crawl
                url = to_visit.pop(0)

                if url in visited:
                    continue

                visited.add(url)

                # Check cache
                if use_cache:
                    cached = self._get_cached(url)
                    if cached:
                        documents.append(cached)
                        continue

                try:
                    result = await crawler.arun(url=url, config=crawler_config)

                    if not result.success:
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
                            "crawled_at": datetime.utcnow().isoformat(),
                        },
                    )

                    documents.append(document)

                    if use_cache:
                        self._save_cache(url, document)

                    # Extract and queue internal links
                    if result.links and "internal" in result.links:
                        for link in result.links["internal"]:
                            href = link.get("href", "")
                            if not href:
                                continue

                            # Resolve relative URLs
                            full_url = urljoin(url, href)
                            parsed = urlparse(full_url)

                            # Filter to same domain
                            if parsed.netloc != base_domain:
                                continue

                            # Filter by pattern if provided
                            if url_regex and not url_regex.search(full_url):
                                continue

                            # Skip anchors and query params
                            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

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
    ) -> list[Document]:
        """Synchronous wrapper for crawl_site."""
        return asyncio.run(self.crawl_site(start_url, max_pages, url_pattern, use_cache))


def get_crawler() -> DocCrawler:
    """Get the default crawler instance."""
    return DocCrawler()
