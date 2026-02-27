"""Content extraction: convert raw HTML into structured documents."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import trafilatura
from bs4 import BeautifulSoup, Tag

from rag_corpus_builder.config import ExtractionConfig
from rag_corpus_builder.models import CrawlResult, ExtractedDocument

logger = logging.getLogger(__name__)


class ContentExtractor:
    """Extract clean, structured content from crawled HTML pages."""

    def __init__(self, config: ExtractionConfig) -> None:
        self.config = config

    def extract(self, crawl_result: CrawlResult) -> ExtractedDocument | None:
        """Extract structured content from a crawl result. Returns None if below quality threshold."""
        html = crawl_result.html
        url = crawl_result.url

        # Try trafilatura first (best at boilerplate removal)
        main_content = self._extract_with_trafilatura(html, url)

        # Fallback to BS4 if trafilatura fails
        if not main_content and self.config.fallback_to_raw:
            main_content = self._extract_with_bs4(html)

        if not main_content or len(main_content) < self.config.min_content_length:
            logger.debug("Skipping %s — content too short (%d chars)", url, len(main_content or ""))
            return None

        soup = BeautifulSoup(html, "lxml")

        doc = ExtractedDocument(
            url=url,
            title=self._extract_title(soup),
            description=self._extract_description(soup),
            author=self._extract_author(soup),
            published_date=self._extract_date(soup),
            main_content=main_content,
            raw_html=html if True else "",  # controlled later by export config
            crawled_at=crawl_result.crawled_at,
        )

        if self.config.extract_metadata:
            doc.metadata = self._extract_metadata(soup)

        if self.config.extract_tables:
            doc.tables = self._extract_tables(soup)

        if self.config.extract_code_blocks:
            doc.code_blocks = self._extract_code_blocks(soup)

        if self.config.extract_links:
            doc.links = self._extract_links(soup, url)

        doc.headings = self._extract_headings(soup)
        doc.images = self._extract_images(soup, url)
        doc.content_hash = hashlib.sha256(main_content.encode()).hexdigest()
        doc.word_count = len(main_content.split())

        return doc

    def _extract_with_trafilatura(self, html: str, url: str) -> str | None:
        """Use trafilatura for main content extraction."""
        try:
            result = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=True,
                include_links=False,
                include_images=False,
                favor_precision=True,
                deduplicate=True,
            )
            return result
        except Exception as exc:
            logger.debug("Trafilatura extraction failed for %s: %s", url, exc)
            return None

    def _extract_with_bs4(self, html: str) -> str:
        """Fallback extraction using BeautifulSoup."""
        soup = BeautifulSoup(html, "lxml")

        # Remove noise elements
        for tag in soup.find_all(["script", "style", "nav", "header", "footer", "aside", "iframe", "noscript"]):
            tag.decompose()

        # Try to find main content area
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(attrs={"role": "main"})
            or soup.find("div", class_=re.compile(r"content|article|post|entry", re.I))
            or soup.body
        )

        if main is None:
            return ""

        text = main.get_text(separator="\n", strip=True)
        # Collapse excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def _extract_title(self, soup: BeautifulSoup) -> str:
        # og:title > <title> > h1
        og = soup.find("meta", property="og:title")
        if og and isinstance(og, Tag) and og.get("content"):
            return str(og["content"]).strip()
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text(strip=True)
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return ""

    def _extract_description(self, soup: BeautifulSoup) -> str:
        for attrs in [
            {"property": "og:description"},
            {"name": "description"},
            {"name": "Description"},
        ]:
            tag = soup.find("meta", attrs=attrs)
            if tag and isinstance(tag, Tag) and tag.get("content"):
                return str(tag["content"]).strip()
        return ""

    def _extract_author(self, soup: BeautifulSoup) -> str:
        for attrs in [
            {"name": "author"},
            {"property": "article:author"},
            {"name": "Author"},
        ]:
            tag = soup.find("meta", attrs=attrs)
            if tag and isinstance(tag, Tag) and tag.get("content"):
                return str(tag["content"]).strip()
        return ""

    def _extract_date(self, soup: BeautifulSoup) -> str | None:
        for attrs in [
            {"property": "article:published_time"},
            {"name": "date"},
            {"name": "publish_date"},
            {"property": "og:updated_time"},
        ]:
            tag = soup.find("meta", attrs=attrs)
            if tag and isinstance(tag, Tag) and tag.get("content"):
                return str(tag["content"]).strip()
        time_tag = soup.find("time")
        if time_tag and isinstance(time_tag, Tag) and time_tag.get("datetime"):
            return str(time_tag["datetime"]).strip()
        return None

    def _extract_metadata(self, soup: BeautifulSoup) -> dict[str, Any]:
        """Extract all meta tags into a dictionary."""
        meta: dict[str, Any] = {}
        for tag in soup.find_all("meta"):
            if not isinstance(tag, Tag):
                continue
            name = tag.get("name") or tag.get("property") or tag.get("http-equiv")
            content = tag.get("content")
            if name and content:
                meta[str(name)] = str(content)
        # Also extract canonical URL
        canonical = soup.find("link", rel="canonical")
        if canonical and isinstance(canonical, Tag) and canonical.get("href"):
            meta["canonical_url"] = str(canonical["href"])
        return meta

    def _extract_headings(self, soup: BeautifulSoup) -> list[dict[str, str]]:
        headings: list[dict[str, str]] = []
        for level in range(1, 7):
            for h in soup.find_all(f"h{level}"):
                text = h.get_text(strip=True)
                if text:
                    headings.append({"level": str(level), "text": text})
        return headings

    def _extract_tables(self, soup: BeautifulSoup) -> list[str]:
        """Convert HTML tables to markdown format."""
        tables: list[str] = []
        for table in soup.find_all("table"):
            md = self._table_to_markdown(table)
            if md:
                tables.append(md)
        return tables

    def _table_to_markdown(self, table: Tag) -> str:
        """Convert an HTML table element to markdown."""
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = []
            for td in tr.find_all(["th", "td"]):
                cells.append(td.get_text(strip=True).replace("|", "\\|"))
            if cells:
                rows.append(cells)

        if not rows:
            return ""

        # Normalize column count
        max_cols = max(len(r) for r in rows)
        for row in rows:
            row.extend([""] * (max_cols - len(row)))

        lines: list[str] = []
        # Header
        lines.append("| " + " | ".join(rows[0]) + " |")
        lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        # Body
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def _extract_code_blocks(self, soup: BeautifulSoup) -> list[dict[str, str]]:
        blocks: list[dict[str, str]] = []
        for pre in soup.find_all("pre"):
            code_tag = pre.find("code")
            if code_tag:
                lang = ""
                classes = code_tag.get("class", [])
                if isinstance(classes, list):
                    for cls in classes:
                        if cls.startswith(("language-", "lang-")):
                            lang = cls.split("-", 1)[1]
                            break
                blocks.append({"language": lang, "code": code_tag.get_text()})
            else:
                blocks.append({"language": "", "code": pre.get_text()})
        return blocks

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[dict[str, str]]:
        links: list[dict[str, str]] = []
        for a in soup.find_all("a", href=True):
            href = str(a["href"])
            text = a.get_text(strip=True)
            if href.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue
            if not href.startswith(("http://", "https://")):
                from urllib.parse import urljoin
                href = urljoin(base_url, href)
            links.append({"href": href, "text": text})
        return links

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> list[dict[str, str]]:
        images: list[dict[str, str]] = []
        for img in soup.find_all("img"):
            src = img.get("src", "")
            alt = img.get("alt", "")
            if src:
                if not src.startswith(("http://", "https://", "data:")):
                    from urllib.parse import urljoin
                    src = urljoin(base_url, src)
                images.append({"src": src, "alt": alt})
        return images
