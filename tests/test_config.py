"""Tests for the config module."""

import tempfile
from pathlib import Path

import pytest

from rag_corpus_builder.config import (
    ChunkConfig,
    ChunkStrategy,
    ExportFormat,
    PipelineConfig,
)


class TestPipelineConfig:
    def test_default_config(self) -> None:
        config = PipelineConfig()
        assert config.crawl.max_pages == 100
        assert config.chunk.strategy == ChunkStrategy.RECURSIVE
        assert config.export.format == ExportFormat.JSONL

    def test_yaml_roundtrip(self) -> None:
        config = PipelineConfig()
        config.crawl.seed_urls = ["https://example.com"]
        config.crawl.max_pages = 42

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            config.to_yaml(f.name)
            loaded = PipelineConfig.from_yaml(f.name)

        assert loaded.crawl.seed_urls == ["https://example.com"]
        assert loaded.crawl.max_pages == 42

    def test_chunk_overlap_validation(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkConfig(chunk_size=100, chunk_overlap=200)

    def test_from_env_and_file_no_file(self) -> None:
        config = PipelineConfig.from_env_and_file(path=None)
        assert config.crawl.max_pages == 100


class TestChunkConfig:
    def test_valid_config(self) -> None:
        config = ChunkConfig(chunk_size=256, chunk_overlap=32)
        assert config.chunk_size == 256
        assert config.chunk_overlap == 32

    def test_all_strategies(self) -> None:
        for strategy in ChunkStrategy:
            config = ChunkConfig(strategy=strategy)
            assert config.strategy == strategy
