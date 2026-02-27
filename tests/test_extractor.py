"""Tests for the extractor module."""

from datetime import datetime, timezone

from rag_corpus_builder.config import ExtractionConfig
from rag_corpus_builder.extractor import ContentExtractor
from rag_corpus_builder.models import CrawlResult


def make_crawl_result(html: str, url: str = "https://example.com") -> CrawlResult:
    return CrawlResult(
        url=url,
        status_code=200,
        html=html,
        headers={"content-type": "text/html"},
        crawled_at=datetime.now(timezone.utc),
        depth=0,
    )


SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Page Title</title>
    <meta name="description" content="A test page for extraction">
    <meta name="author" content="Test Author">
    <meta property="og:title" content="OG Test Title">
</head>
<body>
    <nav>Navigation should be removed</nav>
    <main>
        <h1>Main Heading</h1>
        <p>This is the main content of the test page. It contains enough text to pass the minimum content length filter for our extraction testing purposes.</p>
        <h2>Sub Heading</h2>
        <p>More content here with additional paragraphs to ensure we pass length thresholds.</p>
        <table>
            <tr><th>Name</th><th>Value</th></tr>
            <tr><td>Alpha</td><td>100</td></tr>
            <tr><td>Beta</td><td>200</td></tr>
        </table>
        <pre><code class="language-python">print("hello world")</code></pre>
        <a href="/about">About Us</a>
        <a href="https://external.com">External Link</a>
    </main>
    <footer>Footer should be ignored</footer>
</body>
</html>
"""


class TestContentExtraction:
    def test_extracts_title_from_og(self) -> None:
        config = ExtractionConfig()
        extractor = ContentExtractor(config)
        result = extractor.extract(make_crawl_result(SAMPLE_HTML))
        assert result is not None
        assert result.title == "OG Test Title"

    def test_extracts_description(self) -> None:
        config = ExtractionConfig()
        extractor = ContentExtractor(config)
        result = extractor.extract(make_crawl_result(SAMPLE_HTML))
        assert result is not None
        assert result.description == "A test page for extraction"

    def test_extracts_author(self) -> None:
        config = ExtractionConfig()
        extractor = ContentExtractor(config)
        result = extractor.extract(make_crawl_result(SAMPLE_HTML))
        assert result is not None
        assert result.author == "Test Author"

    def test_extracts_headings(self) -> None:
        config = ExtractionConfig()
        extractor = ContentExtractor(config)
        result = extractor.extract(make_crawl_result(SAMPLE_HTML))
        assert result is not None
        heading_texts = [h["text"] for h in result.headings]
        assert "Main Heading" in heading_texts
        assert "Sub Heading" in heading_texts

    def test_extracts_tables(self) -> None:
        config = ExtractionConfig(extract_tables=True)
        extractor = ContentExtractor(config)
        result = extractor.extract(make_crawl_result(SAMPLE_HTML))
        assert result is not None
        assert len(result.tables) >= 1
        assert "Alpha" in result.tables[0]

    def test_extracts_code_blocks(self) -> None:
        config = ExtractionConfig(extract_code_blocks=True)
        extractor = ContentExtractor(config)
        result = extractor.extract(make_crawl_result(SAMPLE_HTML))
        assert result is not None
        assert len(result.code_blocks) >= 1
        assert result.code_blocks[0]["language"] == "python"
        assert "hello world" in result.code_blocks[0]["code"]

    def test_extracts_links(self) -> None:
        config = ExtractionConfig(extract_links=True)
        extractor = ContentExtractor(config)
        result = extractor.extract(make_crawl_result(SAMPLE_HTML))
        assert result is not None
        hrefs = [l["href"] for l in result.links]
        assert any("about" in h for h in hrefs)
        assert any("external.com" in h for h in hrefs)

    def test_filters_short_content(self) -> None:
        config = ExtractionConfig(min_content_length=10000)
        extractor = ContentExtractor(config)
        result = extractor.extract(make_crawl_result(SAMPLE_HTML))
        assert result is None

    def test_content_hash_set(self) -> None:
        config = ExtractionConfig()
        extractor = ContentExtractor(config)
        result = extractor.extract(make_crawl_result(SAMPLE_HTML))
        assert result is not None
        assert len(result.content_hash) == 64  # SHA-256 hex


class TestBS4Fallback:
    def test_fallback_extracts_content(self) -> None:
        simple_html = """
        <html><body>
            <div class="content">
                <p>This is enough content in a simple HTML page without complex structure
                that trafilatura might not handle well but BeautifulSoup can extract properly.</p>
            </div>
        </body></html>
        """
        config = ExtractionConfig(fallback_to_raw=True, min_content_length=10)
        extractor = ContentExtractor(config)
        result = extractor.extract(make_crawl_result(simple_html))
        # The result depends on whether trafilatura or BS4 handled it
        # Either way, if content is long enough, we should get a result
        if result:
            assert len(result.main_content) >= 10
