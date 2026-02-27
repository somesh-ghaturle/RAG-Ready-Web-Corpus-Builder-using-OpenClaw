"""Configuration models for the pipeline."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class ChunkStrategy(str, Enum):
    """Chunking strategies available."""

    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"
    SENTENCE = "sentence"


class ExportFormat(str, Enum):
    """Output formats."""

    JSONL = "jsonl"
    PARQUET = "parquet"
    HF_DATASET = "hf_dataset"


class CrawlConfig(BaseModel):
    """Crawler configuration."""

    seed_urls: list[str] = Field(default_factory=list, description="Starting URLs to crawl")
    max_pages: int = Field(default=100, ge=1, le=100_000, description="Maximum pages to crawl")
    max_depth: int = Field(default=3, ge=0, le=20, description="Maximum link-follow depth")
    concurrency: int = Field(default=5, ge=1, le=50, description="Concurrent requests")
    delay_seconds: float = Field(default=1.0, ge=0.0, le=60.0, description="Delay between requests per domain")
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0, description="Request timeout")
    respect_robots_txt: bool = Field(default=True, description="Honour robots.txt rules")
    user_agent: str = Field(
        default="RAGCorpusBuilder/0.1 (+https://github.com/rag-corpus-builder)",
        description="User-Agent header",
    )
    allowed_domains: list[str] = Field(default_factory=list, description="Restrict crawl to these domains (empty = follow seed domains)")
    excluded_patterns: list[str] = Field(
        default_factory=lambda: [
            r".*\.(png|jpg|jpeg|gif|svg|ico|css|js|woff|woff2|ttf|eot|mp4|mp3|pdf|zip|tar|gz)$",
            r".*/login.*",
            r".*/signup.*",
            r".*/cart.*",
        ],
        description="URL regex patterns to skip",
    )
    headers: dict[str, str] = Field(default_factory=dict, description="Extra HTTP headers")


class ExtractionConfig(BaseModel):
    """Content extraction configuration."""

    extract_tables: bool = Field(default=True, description="Extract HTML tables as markdown")
    extract_code_blocks: bool = Field(default=True, description="Preserve code blocks")
    extract_links: bool = Field(default=True, description="Extract links with anchor text")
    extract_metadata: bool = Field(default=True, description="Extract meta tags, og:, etc.")
    min_content_length: int = Field(default=50, ge=0, description="Drop pages with fewer chars of main content")
    fallback_to_raw: bool = Field(default=True, description="If trafilatura fails, fall back to BS4 extraction")


class PreprocessConfig(BaseModel):
    """Text preprocessing configuration."""

    lowercase: bool = Field(default=False, description="Convert text to lowercase")
    remove_extra_whitespace: bool = Field(default=True, description="Collapse multiple spaces/newlines")
    min_language_confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Language detection confidence threshold")
    target_languages: list[str] = Field(
        default_factory=lambda: ["en"],
        description="Keep only pages in these languages (ISO 639-1 codes). Empty = keep all.",
    )
    dedup_enabled: bool = Field(default=True, description="Enable content deduplication")
    dedup_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="SimHash similarity threshold for dedup")


class ChunkConfig(BaseModel):
    """Chunking configuration."""

    strategy: ChunkStrategy = Field(default=ChunkStrategy.RECURSIVE, description="Chunking strategy")
    chunk_size: int = Field(default=512, ge=64, le=8192, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=64, ge=0, le=2048, description="Overlap between chunks in tokens")
    tokenizer: str = Field(default="cl100k_base", description="Tiktoken encoding name for token counting")

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_less_than_size(cls, v: int, info: object) -> int:
        data = info.data if hasattr(info, "data") else {}  # type: ignore[union-attr]
        chunk_size = data.get("chunk_size", 512)
        if v >= chunk_size:
            msg = f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            raise ValueError(msg)
        return v


class EmbeddingConfig(BaseModel):
    """Embedding configuration (optional)."""

    enabled: bool = Field(default=False, description="Generate embeddings for chunks")
    model_name: str = Field(default="all-MiniLM-L6-v2", description="Sentence-Transformers model")
    batch_size: int = Field(default=64, ge=1, le=1024, description="Embedding batch size")
    device: str = Field(default="cpu", description="Device: cpu, cuda, mps")
    normalize: bool = Field(default=True, description="L2-normalize embeddings")


class ExportConfig(BaseModel):
    """Export configuration."""

    format: ExportFormat = Field(default=ExportFormat.JSONL, description="Output format")
    output_dir: Path = Field(default=Path("output"), description="Output directory")
    filename_prefix: str = Field(default="corpus", description="Output filename prefix")
    include_raw_html: bool = Field(default=False, description="Include raw HTML in output")
    include_embeddings: bool = Field(default=False, description="Include embeddings in output")
    compress: bool = Field(default=False, description="Gzip compress output files")


class PipelineConfig(BaseSettings):
    """Top-level pipeline configuration."""

    crawl: CrawlConfig = Field(default_factory=CrawlConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    log_level: str = Field(default="INFO", description="Logging level")

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def from_env_and_file(cls, path: Optional[str | Path] = None) -> PipelineConfig:
        """Load from YAML file (if given), with env-var overrides."""
        if path and Path(path).exists():
            return cls.from_yaml(path)
        return cls()

    def to_yaml(self, path: str | Path) -> None:
        """Persist current config to YAML."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
