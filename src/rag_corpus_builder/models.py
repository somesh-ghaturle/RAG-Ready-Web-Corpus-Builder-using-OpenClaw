"""Shared data models used across pipeline stages."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class CrawlResult(BaseModel):
    """Raw result from the crawler."""

    url: str
    status_code: int
    html: str
    headers: dict[str, str] = Field(default_factory=dict)
    crawled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    depth: int = 0
    parent_url: Optional[str] = None
    response_time_ms: float = 0.0


class ExtractedDocument(BaseModel):
    """Document after content extraction."""

    url: str
    title: str = ""
    description: str = ""
    author: str = ""
    published_date: Optional[str] = None
    language: Optional[str] = None
    main_content: str = ""
    headings: list[dict[str, str]] = Field(default_factory=list)  # [{level, text}]
    links: list[dict[str, str]] = Field(default_factory=list)  # [{href, text}]
    tables: list[str] = Field(default_factory=list)  # markdown tables
    code_blocks: list[dict[str, str]] = Field(default_factory=list)  # [{language, code}]
    images: list[dict[str, str]] = Field(default_factory=list)  # [{src, alt}]
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_html: str = ""
    content_hash: str = ""
    word_count: int = 0
    crawled_at: Optional[datetime] = None


class ProcessedDocument(BaseModel):
    """Document after preprocessing."""

    url: str
    title: str = ""
    clean_text: str = ""
    language: str = ""
    language_confidence: float = 0.0
    content_hash: str = ""
    word_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_duplicate: bool = False
    crawled_at: Optional[datetime] = None


class DocumentChunk(BaseModel):
    """A single chunk of a processed document."""

    chunk_id: str = ""
    document_url: str = ""
    document_title: str = ""
    text: str = ""
    token_count: int = 0
    chunk_index: int = 0
    total_chunks: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None
    content_hash: str = ""


class PipelineStats(BaseModel):
    """Pipeline execution statistics."""

    urls_discovered: int = 0
    pages_crawled: int = 0
    pages_failed: int = 0
    pages_skipped_robots: int = 0
    pages_extracted: int = 0
    pages_filtered_language: int = 0
    pages_deduplicated: int = 0
    total_chunks: int = 0
    total_tokens: int = 0
    embeddings_generated: int = 0
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return 0.0
