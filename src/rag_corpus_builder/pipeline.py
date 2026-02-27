"""Pipeline orchestrator: wires together all stages."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from rag_corpus_builder.chunker import DocumentChunker
from rag_corpus_builder.config import PipelineConfig
from rag_corpus_builder.crawler import WebCrawler
from rag_corpus_builder.embedder import EmbeddingGenerator
from rag_corpus_builder.exporter import DatasetExporter
from rag_corpus_builder.extractor import ContentExtractor
from rag_corpus_builder.models import (
    DocumentChunk,
    ExtractedDocument,
    PipelineStats,
    ProcessedDocument,
)
from rag_corpus_builder.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)
console = Console()


class Pipeline:
    """End-to-end RAG corpus building pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.stats = PipelineStats()

    async def run(self) -> None:
        """Execute the full pipeline."""
        self.stats.started_at = datetime.now(timezone.utc)

        console.rule("[bold blue]RAG Corpus Builder Pipeline[/bold blue]")
        console.print(f"[dim]Seed URLs: {', '.join(self.config.crawl.seed_urls)}[/dim]")
        console.print(f"[dim]Max pages: {self.config.crawl.max_pages} | Depth: {self.config.crawl.max_depth}[/dim]")
        console.print(f"[dim]Chunk strategy: {self.config.chunk.strategy.value} | Size: {self.config.chunk.chunk_size} tokens[/dim]")
        console.print(f"[dim]Export format: {self.config.export.format.value}[/dim]")
        console.print()

        # Stage 1: Crawl
        console.rule("[bold cyan]Stage 1: Crawling[/bold cyan]")
        crawl_results = await self._crawl()
        console.print(f"[green]✓[/green] Crawled {len(crawl_results)} pages")
        console.print()

        if not crawl_results:
            console.print("[yellow]No pages crawled. Exiting.[/yellow]")
            return

        # Stage 2: Extract
        console.rule("[bold cyan]Stage 2: Content Extraction[/bold cyan]")
        extracted = self._extract(crawl_results)
        console.print(f"[green]✓[/green] Extracted {len(extracted)} documents")
        console.print()

        if not extracted:
            console.print("[yellow]No content extracted. Exiting.[/yellow]")
            return

        # Stage 3: Preprocess
        console.rule("[bold cyan]Stage 3: Preprocessing[/bold cyan]")
        processed = self._preprocess(extracted)
        console.print(f"[green]✓[/green] {len(processed)} documents after preprocessing")
        console.print()

        if not processed:
            console.print("[yellow]All documents filtered out. Exiting.[/yellow]")
            return

        # Stage 4: Chunk
        console.rule("[bold cyan]Stage 4: Chunking[/bold cyan]")
        chunks = self._chunk(processed)
        console.print(f"[green]✓[/green] Created {len(chunks)} chunks")
        console.print()

        # Stage 5: Embed (optional)
        if self.config.embedding.enabled:
            console.rule("[bold cyan]Stage 5: Embedding[/bold cyan]")
            chunks = self._embed(chunks)
            console.print(f"[green]✓[/green] Generated embeddings for {len(chunks)} chunks")
            console.print()

        # Stage 6: Export
        console.rule("[bold cyan]Stage 6: Export[/bold cyan]")
        self.stats.finished_at = datetime.now(timezone.utc)
        output_path = self._export(chunks)
        console.print(f"[green]✓[/green] Exported to {output_path}")
        console.print()

        # Summary
        self._print_summary()

    async def _crawl(self) -> list:
        """Run the crawler."""
        crawler = WebCrawler(self.config.crawl)
        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Crawling...", total=self.config.crawl.max_pages)

            async for result in crawler.crawl():
                results.append(result)
                progress.update(task, completed=len(results))

        stats = crawler.stats
        self.stats.urls_discovered = stats["urls_discovered"]
        self.stats.pages_crawled = stats["pages_crawled"]
        self.stats.pages_failed = stats["pages_failed"]
        self.stats.pages_skipped_robots = stats["pages_skipped_robots"]
        return results

    def _extract(self, crawl_results: list) -> list[ExtractedDocument]:
        """Extract content from crawled pages."""
        extractor = ContentExtractor(self.config.extraction)
        extracted: list[ExtractedDocument] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting content...", total=len(crawl_results))

            for cr in crawl_results:
                doc = extractor.extract(cr)
                if doc is not None:
                    extracted.append(doc)
                progress.update(task, advance=1)

        self.stats.pages_extracted = len(extracted)
        return extracted

    def _preprocess(self, docs: list[ExtractedDocument]) -> list[ProcessedDocument]:
        """Preprocess and filter documents."""
        preprocessor = TextPreprocessor(self.config.preprocess)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Preprocessing...", total=len(docs))
            processed: list[ProcessedDocument] = []
            for doc in docs:
                result = preprocessor.process(doc)
                if result is not None:
                    processed.append(result)
                progress.update(task, advance=1)

        filtered_count = len(docs) - len(processed)
        self.stats.pages_filtered_language = filtered_count
        self.stats.pages_deduplicated = len(docs) - len(processed)
        return processed

    def _chunk(self, docs: list[ProcessedDocument]) -> list[DocumentChunk]:
        """Chunk documents."""
        chunker = DocumentChunker(self.config.chunk)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Chunking...", total=len(docs))
            all_chunks: list[DocumentChunk] = []
            for doc in docs:
                chunks = chunker.chunk_document(doc)
                all_chunks.extend(chunks)
                progress.update(task, advance=1)

        self.stats.total_chunks = len(all_chunks)
        self.stats.total_tokens = sum(c.token_count for c in all_chunks)
        return all_chunks

    def _embed(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Generate embeddings for chunks."""
        embedder = EmbeddingGenerator(self.config.embedding)
        result = embedder.embed_chunks(chunks)
        self.stats.embeddings_generated = len(result)
        return result

    def _export(self, chunks: list[DocumentChunk]) -> str:
        """Export chunks to the configured format."""
        exporter = DatasetExporter(self.config.export)
        path = exporter.export(chunks, self.stats)
        return str(path)

    def _print_summary(self) -> None:
        """Print pipeline execution summary."""
        console.rule("[bold green]Pipeline Complete[/bold green]")
        console.print()

        from rich.table import Table

        table = Table(title="Pipeline Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white", justify="right")

        table.add_row("URLs Discovered", str(self.stats.urls_discovered))
        table.add_row("Pages Crawled", str(self.stats.pages_crawled))
        table.add_row("Pages Failed", str(self.stats.pages_failed))
        table.add_row("Pages Skipped (robots.txt)", str(self.stats.pages_skipped_robots))
        table.add_row("Pages Extracted", str(self.stats.pages_extracted))
        table.add_row("Pages Filtered/Deduped", str(self.stats.pages_deduplicated))
        table.add_row("Total Chunks", str(self.stats.total_chunks))
        table.add_row("Total Tokens", f"{self.stats.total_tokens:,}")
        if self.stats.embeddings_generated:
            table.add_row("Embeddings Generated", str(self.stats.embeddings_generated))
        table.add_row("Duration", f"{self.stats.duration_seconds:.1f}s")

        console.print(table)
        console.print()


def run_pipeline(config: PipelineConfig) -> None:
    """Convenience function to run the pipeline synchronously."""
    pipeline = Pipeline(config)
    asyncio.run(pipeline.run())
