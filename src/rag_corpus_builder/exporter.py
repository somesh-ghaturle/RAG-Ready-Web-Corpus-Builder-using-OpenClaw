"""Export document chunks to various formats."""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

from rag_corpus_builder.config import ExportConfig, ExportFormat
from rag_corpus_builder.models import DocumentChunk, PipelineStats

logger = logging.getLogger(__name__)


class DatasetExporter:
    """Export processed chunks to JSONL, Parquet, or HuggingFace Dataset."""

    def __init__(self, config: ExportConfig) -> None:
        self.config = config
        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        chunks: Sequence[DocumentChunk],
        stats: PipelineStats | None = None,
    ) -> Path:
        """Export chunks using the configured format. Returns the output path."""
        fmt = self.config.format

        if fmt == ExportFormat.JSONL:
            return self._export_jsonl(chunks, stats)
        elif fmt == ExportFormat.PARQUET:
            return self._export_parquet(chunks, stats)
        elif fmt == ExportFormat.HF_DATASET:
            return self._export_hf_dataset(chunks, stats)
        else:
            raise ValueError(f"Unsupported export format: {fmt}")

    def _chunk_to_dict(self, chunk: DocumentChunk) -> dict[str, Any]:
        """Convert a chunk to a serialisable dictionary."""
        data: dict[str, Any] = {
            "chunk_id": chunk.chunk_id,
            "document_url": chunk.document_url,
            "document_title": chunk.document_title,
            "text": chunk.text,
            "token_count": chunk.token_count,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "content_hash": chunk.content_hash,
            "metadata": json.dumps(chunk.metadata) if chunk.metadata else "{}",
        }
        if self.config.include_embeddings and chunk.embedding is not None:
            data["embedding"] = chunk.embedding
        return data

    def _export_jsonl(
        self,
        chunks: Sequence[DocumentChunk],
        stats: PipelineStats | None,
    ) -> Path:
        """Export to JSONL (optionally gzipped)."""
        suffix = ".jsonl.gz" if self.config.compress else ".jsonl"
        path = self._output_dir / f"{self.config.filename_prefix}{suffix}"

        opener = gzip.open if self.config.compress else open
        mode = "wt" if self.config.compress else "w"

        with opener(path, mode, encoding="utf-8") as f:  # type: ignore[call-overload]
            for chunk in chunks:
                line = json.dumps(self._chunk_to_dict(chunk), ensure_ascii=False)
                f.write(line + "\n")

        # Write stats sidecar
        if stats:
            stats_path = self._output_dir / f"{self.config.filename_prefix}_stats.json"
            with open(stats_path, "w") as sf:
                json.dump(stats.model_dump(mode="json"), sf, indent=2, default=str)

        logger.info("Exported %d chunks to %s", len(chunks), path)
        return path

    def _export_parquet(
        self,
        chunks: Sequence[DocumentChunk],
        stats: PipelineStats | None,
    ) -> Path:
        """Export to Parquet."""
        path = self._output_dir / f"{self.config.filename_prefix}.parquet"

        records = [self._chunk_to_dict(chunk) for chunk in chunks]

        if not records:
            logger.warning("No chunks to export")
            # Write empty parquet
            table = pa.table(
                {
                    "chunk_id": pa.array([], type=pa.string()),
                    "text": pa.array([], type=pa.string()),
                }
            )
            pq.write_table(table, path)
            return path

        # Build schema dynamically
        columns: dict[str, list[Any]] = {}
        for key in records[0]:
            columns[key] = [r.get(key) for r in records]

        table = pa.table(columns)
        pq.write_table(
            table,
            path,
            compression="snappy" if self.config.compress else "none",
        )

        if stats:
            stats_path = self._output_dir / f"{self.config.filename_prefix}_stats.json"
            with open(stats_path, "w") as sf:
                json.dump(stats.model_dump(mode="json"), sf, indent=2, default=str)

        logger.info("Exported %d chunks to %s", len(chunks), path)
        return path

    def _export_hf_dataset(
        self,
        chunks: Sequence[DocumentChunk],
        stats: PipelineStats | None,
    ) -> Path:
        """Export as a HuggingFace Dataset (Arrow format on disk)."""
        path = self._output_dir / f"{self.config.filename_prefix}_dataset"

        records = [self._chunk_to_dict(chunk) for chunk in chunks]

        if not records:
            logger.warning("No chunks to export")
            ds = Dataset.from_dict({"chunk_id": [], "text": []})
        else:
            columns: dict[str, list[Any]] = {}
            for key in records[0]:
                columns[key] = [r.get(key) for r in records]
            ds = Dataset.from_dict(columns)

        ds.save_to_disk(str(path))

        if stats:
            stats_path = self._output_dir / f"{self.config.filename_prefix}_stats.json"
            with open(stats_path, "w") as sf:
                json.dump(stats.model_dump(mode="json"), sf, indent=2, default=str)

        logger.info("Exported %d chunks as HF Dataset to %s", len(chunks), path)
        return path
