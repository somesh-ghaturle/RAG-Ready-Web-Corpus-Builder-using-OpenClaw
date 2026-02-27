"""Tests for the exporter module."""

import json
import tempfile
from pathlib import Path

from rag_corpus_builder.config import ExportConfig, ExportFormat
from rag_corpus_builder.exporter import DatasetExporter
from rag_corpus_builder.models import DocumentChunk


def make_chunks(n: int = 3) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id=f"chunk_{i}",
            document_url=f"https://example.com/page{i}",
            document_title=f"Page {i}",
            text=f"This is the text content of chunk number {i} which is long enough to be meaningful.",
            token_count=20,
            chunk_index=i,
            total_chunks=n,
            content_hash=f"hash_{i}",
            metadata={"language": "en"},
        )
        for i in range(n)
    ]


class TestJSONLExport:
    def test_exports_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(format=ExportFormat.JSONL, output_dir=tmpdir, filename_prefix="test")
            exporter = DatasetExporter(config)
            chunks = make_chunks()
            path = exporter.export(chunks)

            assert path.exists()
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 3

            record = json.loads(lines[0])
            assert "chunk_id" in record
            assert "text" in record

    def test_compressed_jsonl(self) -> None:
        import gzip

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(
                format=ExportFormat.JSONL, output_dir=tmpdir, filename_prefix="test", compress=True
            )
            exporter = DatasetExporter(config)
            chunks = make_chunks()
            path = exporter.export(chunks)

            assert path.name.endswith(".jsonl.gz")
            with gzip.open(path, "rt") as f:
                lines = f.readlines()
            assert len(lines) == 3


class TestParquetExport:
    def test_exports_parquet(self) -> None:
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(format=ExportFormat.PARQUET, output_dir=tmpdir, filename_prefix="test")
            exporter = DatasetExporter(config)
            chunks = make_chunks()
            path = exporter.export(chunks)

            assert path.suffix == ".parquet"
            table = pq.read_table(path)
            assert table.num_rows == 3
            assert "text" in table.column_names


class TestEmptyExport:
    def test_empty_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(format=ExportFormat.JSONL, output_dir=tmpdir, filename_prefix="empty")
            exporter = DatasetExporter(config)
            path = exporter.export([])
            assert path.exists()
            with open(path) as f:
                assert f.read().strip() == ""
