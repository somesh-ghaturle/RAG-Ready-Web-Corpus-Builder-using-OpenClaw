"""Async web crawler with politeness controls, robots.txt, and rate limiting."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import defaultdict
from typing import AsyncIterator, Optional
from urllib.parse import urljoin, urlparse

import httpx
from robotexclusionrulesparser import RobotExclusionRulesParser
from tenacity import retry, stop_after_attempt, wait_exponential

from rag_corpus_builder.config import CrawlConfig
from rag_corpus_builder.models import CrawlResult

logger = logging.getLogger(__name__)


class RobotsCache:
    """Cache and check robots.txt rules per domain."""

    def __init__(self, user_agent: str, timeout: float = 10.0) -> None:
        self._user_agent = user_agent
        self._timeout = timeout
        self._cache: dict[str, RobotExclusionRulesParser] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, url: str, client: httpx.AsyncClient) -> bool:
        """Check if the URL is allowed by robots.txt."""
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        async with self._lock:
            if domain not in self._cache:
                parser = RobotExclusionRulesParser()
                robots_url = f"{domain}/robots.txt"
                try:
                    resp = await client.get(robots_url, timeout=self._timeout)
                    if resp.status_code == 200:
                        parser.parse(resp.text)
                    else:
                        # If robots.txt is missing/errored, allow everything
                        parser.parse("")
                except Exception:
                    logger.debug("Could not fetch robots.txt for %s", domain)
                    parser.parse("")
                self._cache[domain] = parser

            return self._cache[domain].is_allowed(self._user_agent, url)


class DomainThrottler:
    """Per-domain rate limiting."""

    def __init__(self, delay_seconds: float) -> None:
        self._delay = delay_seconds
        self._last_request: dict[str, float] = defaultdict(float)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def wait(self, url: str) -> None:
        domain = urlparse(url).netloc
        async with self._locks[domain]:
            elapsed = time.monotonic() - self._last_request[domain]
            if elapsed < self._delay:
                await asyncio.sleep(self._delay - elapsed)
            self._last_request[domain] = time.monotonic()


class WebCrawler:
    """Async breadth-first web crawler with configurable politeness."""

    def __init__(self, config: CrawlConfig) -> None:
        self.config = config
        self._visited: set[str] = set()
        self._robots = RobotsCache(config.user_agent, timeout=config.timeout_seconds)
        self._throttler = DomainThrottler(config.delay_seconds)
        self._excluded_re = [re.compile(p, re.IGNORECASE) for p in config.excluded_patterns]
        self._allowed_domains: set[str] = set()
        self._pages_crawled = 0
        self._pages_failed = 0
        self._pages_skipped = 0

    def _normalise_url(self, url: str) -> str:
        """Strip fragments and trailing slashes for deduplication."""
        parsed = urlparse(url)
        path = parsed.path.rstrip("/") or "/"
        return f"{parsed.scheme}://{parsed.netloc}{path}"
        # Intentionally drop query params? No — keep them for now
        # return f"{parsed.scheme}://{parsed.netloc}{path}?{parsed.query}" if parsed.query else ...

    def _normalise_url_full(self, url: str) -> str:
        """Normalise URL keeping query params but dropping fragments."""
        parsed = urlparse(url)
        path = parsed.path.rstrip("/") or "/"
        base = f"{parsed.scheme}://{parsed.netloc}{path}"
        if parsed.query:
            base += f"?{parsed.query}"
        return base

    def _is_excluded(self, url: str) -> bool:
        return any(pattern.search(url) for pattern in self._excluded_re)

    def _is_allowed_domain(self, url: str) -> bool:
        if not self._allowed_domains:
            return True
        domain = urlparse(url).netloc
        return any(domain == d or domain.endswith(f".{d}") for d in self._allowed_domains)

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        """Extract and resolve links from HTML (lightweight regex-based)."""
        links: list[str] = []
        for match in re.finditer(r'<a\s+[^>]*href=["\']([^"\']+)["\']', html, re.IGNORECASE):
            href = match.group(1).strip()
            if href.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue
            absolute = urljoin(base_url, href)
            normalised = self._normalise_url_full(absolute)
            if normalised.startswith(("http://", "https://")):
                links.append(normalised)
        return links

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def _fetch(self, client: httpx.AsyncClient, url: str) -> httpx.Response:
        """Fetch a URL with retry logic."""
        return await client.get(
            url,
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
        )

    async def crawl(self) -> AsyncIterator[CrawlResult]:
        """Run the crawl and yield results as they arrive."""
        if not self.config.seed_urls:
            logger.warning("No seed URLs configured — nothing to crawl.")
            return

        # Determine allowed domains
        if self.config.allowed_domains:
            self._allowed_domains = set(self.config.allowed_domains)
        else:
            self._allowed_domains = {urlparse(u).netloc for u in self.config.seed_urls}

        # BFS queue: (url, depth, parent_url)
        queue: asyncio.Queue[tuple[str, int, Optional[str]]] = asyncio.Queue()
        for url in self.config.seed_urls:
            normalised = self._normalise_url_full(url)
            self._visited.add(normalised)
            await queue.put((normalised, 0, None))

        results: list[CrawlResult] = []
        semaphore = asyncio.Semaphore(self.config.concurrency)

        headers = {
            "User-Agent": self.config.user_agent,
            **self.config.headers,
        }

        async with httpx.AsyncClient(headers=headers) as client:
            active_tasks: set[asyncio.Task[None]] = set()

            async def process_url(url: str, depth: int, parent: Optional[str]) -> None:
                nonlocal results
                async with semaphore:
                    if self._pages_crawled >= self.config.max_pages:
                        return

                    # Robots.txt check
                    if self.config.respect_robots_txt:
                        if not await self._robots.is_allowed(url, client):
                            self._pages_skipped += 1
                            logger.debug("Blocked by robots.txt: %s", url)
                            return

                    # Rate limiting
                    await self._throttler.wait(url)

                    start_time = time.monotonic()
                    try:
                        response = await self._fetch(client, url)
                        elapsed_ms = (time.monotonic() - start_time) * 1000

                        content_type = response.headers.get("content-type", "")
                        if "text/html" not in content_type and "application/xhtml" not in content_type:
                            logger.debug("Skipping non-HTML: %s (%s)", url, content_type)
                            return

                        result = CrawlResult(
                            url=str(response.url),
                            status_code=response.status_code,
                            html=response.text,
                            headers=dict(response.headers),
                            depth=depth,
                            parent_url=parent,
                            response_time_ms=elapsed_ms,
                        )
                        results.append(result)
                        self._pages_crawled += 1
                        logger.info(
                            "[%d/%d] Crawled %s (%d, %.0fms)",
                            self._pages_crawled,
                            self.config.max_pages,
                            url,
                            response.status_code,
                            elapsed_ms,
                        )

                        # Discover new links
                        if depth < self.config.max_depth:
                            for link in self._extract_links(response.text, str(response.url)):
                                norm_link = self._normalise_url_full(link)
                                if (
                                    norm_link not in self._visited
                                    and self._is_allowed_domain(norm_link)
                                    and not self._is_excluded(norm_link)
                                ):
                                    self._visited.add(norm_link)
                                    await queue.put((norm_link, depth + 1, url))

                    except Exception as exc:
                        self._pages_failed += 1
                        logger.warning("Failed to crawl %s: %s", url, exc)

            # Process the BFS queue
            while not queue.empty() or active_tasks:
                # Launch tasks from queue
                while not queue.empty() and self._pages_crawled + len(active_tasks) < self.config.max_pages:
                    url, depth, parent = await queue.get()
                    task = asyncio.create_task(process_url(url, depth, parent))
                    active_tasks.add(task)
                    task.add_done_callback(active_tasks.discard)

                # Wait for at least one task to complete
                if active_tasks:
                    done, _ = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                    for t in done:
                        active_tasks.discard(t)
                        if t.exception():
                            logger.error("Task error: %s", t.exception())

                # Check if we've hit the max
                if self._pages_crawled >= self.config.max_pages:
                    break

            # Wait for remaining tasks
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)

        logger.info(
            "Crawl complete: %d pages crawled, %d failed, %d skipped (robots.txt)",
            self._pages_crawled,
            self._pages_failed,
            self._pages_skipped,
        )

        for result in results:
            yield result

    @property
    def stats(self) -> dict[str, int]:
        return {
            "pages_crawled": self._pages_crawled,
            "pages_failed": self._pages_failed,
            "pages_skipped_robots": self._pages_skipped,
            "urls_discovered": len(self._visited),
        }
