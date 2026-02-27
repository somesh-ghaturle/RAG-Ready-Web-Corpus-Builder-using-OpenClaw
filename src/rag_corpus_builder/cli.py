"""CLI interface for RAG Corpus Builder."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

from rag_corpus_builder.config import (
    ChunkStrategy,
    ExportFormat,
    PipelineConfig,
)
from rag_corpus_builder.pipeline import run_pipeline

console = Console()


def setup_logging(level: str) -> None:
    """Configure rich logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


@click.group()
@click.version_option(package_name="rag-corpus-builder")
def main() -> None:
    """RAG-Ready Web Corpus Builder — convert websites into searchable document datasets."""
    pass


@main.command()
@click.argument("urls", nargs=-1, required=True)
@click.option("--config", "-c", "config_path", type=click.Path(exists=True), help="YAML config file")
@click.option("--max-pages", "-n", type=int, default=100, show_default=True, help="Max pages to crawl")
@click.option("--max-depth", "-d", type=int, default=3, show_default=True, help="Max crawl depth")
@click.option("--concurrency", type=int, default=5, show_default=True, help="Concurrent requests")
@click.option("--delay", type=float, default=1.0, show_default=True, help="Per-domain delay (seconds)")
@click.option(
    "--chunk-strategy",
    type=click.Choice([s.value for s in ChunkStrategy], case_sensitive=False),
    default="recursive",
    show_default=True,
    help="Chunking strategy",
)
@click.option("--chunk-size", type=int, default=512, show_default=True, help="Chunk size in tokens")
@click.option("--chunk-overlap", type=int, default=64, show_default=True, help="Chunk overlap in tokens")
@click.option(
    "--format", "-f",
    "export_format",
    type=click.Choice([f.value for f in ExportFormat], case_sensitive=False),
    default="jsonl",
    show_default=True,
    help="Output format",
)
@click.option("--output", "-o", type=click.Path(), default="output", show_default=True, help="Output directory")
@click.option("--embed/--no-embed", default=False, show_default=True, help="Generate embeddings")
@click.option("--embed-model", type=str, default="all-MiniLM-L6-v2", show_default=True, help="Embedding model name")
@click.option("--language", "-l", multiple=True, default=["en"], show_default=True, help="Target languages (ISO 639-1)")
@click.option("--no-robots", is_flag=True, default=False, help="Ignore robots.txt")
@click.option("--compress", is_flag=True, default=False, help="Compress output")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), default="INFO", show_default=True)
def crawl(
    urls: tuple[str, ...],
    config_path: str | None,
    max_pages: int,
    max_depth: int,
    concurrency: int,
    delay: float,
    chunk_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    export_format: str,
    output: str,
    embed: bool,
    embed_model: str,
    language: tuple[str, ...],
    no_robots: bool,
    compress: bool,
    log_level: str,
) -> None:
    """Crawl URLs, extract content, and build a RAG-ready corpus.

    Pass one or more seed URLs as arguments:

        rag-corpus crawl https://docs.python.org https://wiki.python.org
    """
    setup_logging(log_level)

    # Load config from file or build from CLI args
    if config_path:
        config = PipelineConfig.from_yaml(config_path)
        # Override seed URLs if provided on command line
        if urls:
            config.crawl.seed_urls = list(urls)
    else:
        config = PipelineConfig(
            crawl={
                "seed_urls": list(urls),
                "max_pages": max_pages,
                "max_depth": max_depth,
                "concurrency": concurrency,
                "delay_seconds": delay,
                "respect_robots_txt": not no_robots,
            },
            extraction={},
            preprocess={
                "target_languages": list(language),
            },
            chunk={
                "strategy": chunk_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
            embedding={
                "enabled": embed,
                "model_name": embed_model,
            },
            export={
                "format": export_format,
                "output_dir": output,
                "compress": compress,
            },
            log_level=log_level,
        )

    try:
        run_pipeline(config)
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user.[/yellow]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n[red]Pipeline failed: {exc}[/red]")
        logging.getLogger(__name__).exception("Pipeline error")
        sys.exit(1)


@main.command("init-config")
@click.option("--output", "-o", type=click.Path(), default="pipeline_config.yaml", show_default=True)
def init_config(output: str) -> None:
    """Generate a default YAML configuration file."""
    config = PipelineConfig()
    config.to_yaml(output)
    console.print(f"[green]✓[/green] Default config written to {output}")
    console.print("[dim]Edit the file and run: rag-corpus crawl --config pipeline_config.yaml URL[/dim]")


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
def inspect(input_path: str) -> None:
    """Inspect an exported corpus file (JSONL or Parquet)."""
    import json

    from rich.table import Table

    path = Path(input_path)

    if path.suffix == ".jsonl" or path.name.endswith(".jsonl.gz"):
        import gzip

        opener = gzip.open if path.name.endswith(".gz") else open
        records = []
        with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[call-overload]
            for line in f:
                records.append(json.loads(line))

        console.print(f"[bold]JSONL Corpus: {path.name}[/bold]")
        console.print(f"Total chunks: {len(records)}")
        console.print()

        if records:
            table = Table(title="Sample Chunks (first 5)", show_lines=True)
            table.add_column("ID", style="cyan", max_width=16)
            table.add_column("URL", style="blue", max_width=40)
            table.add_column("Tokens", justify="right")
            table.add_column("Text Preview", max_width=60)

            for rec in records[:5]:
                table.add_row(
                    rec.get("chunk_id", ""),
                    rec.get("document_url", ""),
                    str(rec.get("token_count", "")),
                    rec.get("text", "")[:120] + "...",
                )
            console.print(table)

    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        pf = pq.read_table(path)
        console.print(f"[bold]Parquet Corpus: {path.name}[/bold]")
        console.print(f"Total chunks: {pf.num_rows}")
        console.print(f"Columns: {pf.column_names}")
        console.print()

        if pf.num_rows > 0:
            df = pf.slice(0, min(5, pf.num_rows)).to_pandas()
            console.print(df.to_string())

    else:
        console.print(f"[yellow]Unsupported file format: {path.suffix}[/yellow]")


if __name__ == "__main__":
    main()
