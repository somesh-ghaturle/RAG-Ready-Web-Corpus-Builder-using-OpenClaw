"""Background task runner for pipeline execution."""

from __future__ import annotations

import asyncio
import logging
import threading
import traceback
from datetime import datetime, timezone

from django.utils import timezone as dj_timezone

logger = logging.getLogger(__name__)

# Global registry of running jobs
_running_jobs: dict[str, threading.Thread] = {}
_job_lock = threading.Lock()


def is_job_running(job_id: str) -> bool:
    with _job_lock:
        thread = _running_jobs.get(job_id)
        return thread is not None and thread.is_alive()


def cancel_job(job_id: str) -> bool:
    """Mark a job as cancelled. The pipeline checks this flag periodically."""
    from web.dashboard.models import CrawlJob
    try:
        job = CrawlJob.objects.get(pk=job_id)
        if job.status == CrawlJob.Status.RUNNING:
            job.status = CrawlJob.Status.CANCELLED
            job.finished_at = dj_timezone.now()
            job.append_log("[CANCELLED] Job cancelled by user.")
            job.save()
            return True
    except CrawlJob.DoesNotExist:
        pass
    return False


def start_pipeline_job(job_id: str) -> None:
    """Launch the pipeline in a background thread."""
    with _job_lock:
        if job_id in _running_jobs and _running_jobs[job_id].is_alive():
            logger.warning("Job %s is already running", job_id)
            return

        thread = threading.Thread(
            target=_run_pipeline_thread,
            args=(str(job_id),),
            name=f"pipeline-{job_id[:8]}",
            daemon=True,
        )
        _running_jobs[job_id] = thread
        thread.start()


def _run_pipeline_thread(job_id: str) -> None:
    """Execute the pipeline (runs in a background thread)."""
    import django
    django.setup()

    from web.dashboard.models import CrawlJob, CrawlJobChunk
    from rag_corpus_builder.config import PipelineConfig
    from rag_corpus_builder.crawler import WebCrawler
    from rag_corpus_builder.extractor import ContentExtractor
    from rag_corpus_builder.preprocessor import TextPreprocessor
    from rag_corpus_builder.chunker import DocumentChunker
    from rag_corpus_builder.embedder import EmbeddingGenerator
    from rag_corpus_builder.exporter import DatasetExporter

    try:
        job = CrawlJob.objects.get(pk=job_id)
    except CrawlJob.DoesNotExist:
        logger.error("Job %s not found", job_id)
        return

    job.status = CrawlJob.Status.RUNNING
    job.started_at = dj_timezone.now()
    job.progress = 0
    job.current_stage = "Initializing"
    job.save()
    job.append_log(f"[START] Pipeline started at {job.started_at.isoformat()}")

    try:
        config = PipelineConfig(**job.config_json)
        _execute_pipeline(job, config)
    except Exception as exc:
        job.status = CrawlJob.Status.FAILED
        job.error_message = str(exc)
        job.finished_at = dj_timezone.now()
        job.append_log(f"[ERROR] {traceback.format_exc()}")
        job.save()
        logger.exception("Pipeline job %s failed", job_id)
    finally:
        with _job_lock:
            _running_jobs.pop(job_id, None)


def _execute_pipeline(job, config) -> None:
    """Run all pipeline stages, updating the job record along the way."""
    from web.dashboard.models import CrawlJob, CrawlJobChunk
    from rag_corpus_builder.crawler import WebCrawler
    from rag_corpus_builder.extractor import ContentExtractor
    from rag_corpus_builder.preprocessor import TextPreprocessor
    from rag_corpus_builder.chunker import DocumentChunker
    from rag_corpus_builder.embedder import EmbeddingGenerator
    from rag_corpus_builder.exporter import DatasetExporter

    def check_cancelled():
        job.refresh_from_db(fields=["status"])
        return job.status == CrawlJob.Status.CANCELLED

    async def check_cancelled_async():
        from asgiref.sync import sync_to_async
        await sync_to_async(job.refresh_from_db)(fields=["status"])
        return job.status == CrawlJob.Status.CANCELLED

    # ── Stage 1: Crawl ──
    job.current_stage = "Crawling"
    job.progress = 5
    job.save(update_fields=["current_stage", "progress", "updated_at"])
    job.append_log(f"[CRAWL] Starting crawl of {len(config.crawl.seed_urls)} seed URLs...")

    crawler = WebCrawler(config.crawl)
    crawl_results = []

    loop = asyncio.new_event_loop()
    try:
        async def collect_crawl():
            async for result in crawler.crawl():
                crawl_results.append(result)
                if await check_cancelled_async():
                    return

        loop.run_until_complete(collect_crawl())
    finally:
        loop.close()

    if check_cancelled():
        return

    stats = crawler.stats
    job.pages_crawled = stats["pages_crawled"]
    job.pages_failed = stats["pages_failed"]
    job.progress = 20
    job.save(update_fields=["pages_crawled", "pages_failed", "progress", "updated_at"])
    job.append_log(f"[CRAWL] Crawled {len(crawl_results)} pages ({stats['pages_failed']} failed)")

    if not crawl_results:
        job.status = CrawlJob.Status.COMPLETED
        job.finished_at = dj_timezone.now()
        job.progress = 100
        job.append_log("[DONE] No pages crawled. Pipeline finished with empty results.")
        job.save()
        return

    # ── Stage 2: Extract ──
    job.current_stage = "Extracting"
    job.progress = 30
    job.save(update_fields=["current_stage", "progress", "updated_at"])
    job.append_log("[EXTRACT] Extracting content from HTML...")

    extractor = ContentExtractor(config.extraction)
    extracted = []
    for cr in crawl_results:
        doc = extractor.extract(cr)
        if doc is not None:
            extracted.append(doc)

    if check_cancelled():
        return

    job.pages_extracted = len(extracted)
    job.progress = 50
    job.save(update_fields=["pages_extracted", "progress", "updated_at"])
    job.append_log(f"[EXTRACT] Extracted {len(extracted)} documents")

    if not extracted:
        job.status = CrawlJob.Status.COMPLETED
        job.finished_at = dj_timezone.now()
        job.progress = 100
        job.append_log("[DONE] No content extracted. Pipeline finished.")
        job.save()
        return

    # ── Stage 3: Preprocess ──
    job.current_stage = "Preprocessing"
    job.progress = 55
    job.save(update_fields=["current_stage", "progress", "updated_at"])
    job.append_log("[PREPROCESS] Cleaning, deduplicating, filtering...")

    preprocessor = TextPreprocessor(config.preprocess)
    processed = preprocessor.process_batch(extracted)

    if check_cancelled():
        return

    job.progress = 65
    job.save(update_fields=["progress", "updated_at"])
    job.append_log(f"[PREPROCESS] {len(processed)} documents after preprocessing ({len(extracted) - len(processed)} filtered)")

    if not processed:
        job.status = CrawlJob.Status.COMPLETED
        job.finished_at = dj_timezone.now()
        job.progress = 100
        job.append_log("[DONE] All documents filtered out. Pipeline finished.")
        job.save()
        return

    # ── Stage 4: Chunk ──
    job.current_stage = "Chunking"
    job.progress = 70
    job.save(update_fields=["current_stage", "progress", "updated_at"])
    job.append_log(f"[CHUNK] Chunking with strategy={config.chunk.strategy.value}, size={config.chunk.chunk_size}")

    chunker = DocumentChunker(config.chunk)
    chunks = chunker.chunk_batch(processed)

    if check_cancelled():
        return

    job.total_chunks = len(chunks)
    job.total_tokens = sum(c.token_count for c in chunks)
    job.progress = 80
    job.save(update_fields=["total_chunks", "total_tokens", "progress", "updated_at"])
    job.append_log(f"[CHUNK] Created {len(chunks)} chunks ({job.total_tokens:,} tokens)")

    # ── Stage 5: Embed (optional) ──
    if config.embedding.enabled:
        job.current_stage = "Embedding"
        job.progress = 85
        job.save(update_fields=["current_stage", "progress", "updated_at"])
        job.append_log(f"[EMBED] Generating embeddings with {config.embedding.model_name}...")

        embedder = EmbeddingGenerator(config.embedding)
        chunks = embedder.embed_chunks(chunks)
        job.append_log(f"[EMBED] Generated {len(chunks)} embeddings")

    if check_cancelled():
        return

    # ── Stage 6: Export ──
    job.current_stage = "Exporting"
    job.progress = 90
    job.save(update_fields=["current_stage", "progress", "updated_at"])
    job.append_log(f"[EXPORT] Exporting to {config.export.format.value}...")

    exporter = DatasetExporter(config.export)
    from rag_corpus_builder.models import PipelineStats
    pipeline_stats = PipelineStats(
        urls_discovered=stats.get("urls_discovered", 0),
        pages_crawled=job.pages_crawled,
        pages_failed=job.pages_failed,
        pages_extracted=job.pages_extracted,
        total_chunks=job.total_chunks,
        total_tokens=job.total_tokens,
        started_at=job.started_at,
        finished_at=dj_timezone.now(),
    )
    output_path = exporter.export(chunks, pipeline_stats)
    job.output_path = str(output_path)
    job.append_log(f"[EXPORT] Written to {output_path}")

    # ── Store chunks in DB for browsing ──
    job.current_stage = "Storing chunks"
    job.progress = 95
    job.save(update_fields=["current_stage", "progress", "output_path", "updated_at"])

    chunk_objects = []
    for c in chunks:
        chunk_objects.append(CrawlJobChunk(
            job=job,
            chunk_id=c.chunk_id,
            document_url=c.document_url,
            document_title=c.document_title,
            text=c.text,
            token_count=c.token_count,
            chunk_index=c.chunk_index,
            total_chunks=c.total_chunks,
            content_hash=c.content_hash,
            metadata_json=c.metadata,
        ))
    CrawlJobChunk.objects.bulk_create(chunk_objects, batch_size=500)
    job.append_log(f"[STORE] Stored {len(chunk_objects)} chunks in database")

    # ── Done ──
    job.status = CrawlJob.Status.COMPLETED
    job.finished_at = dj_timezone.now()
    job.progress = 100
    job.current_stage = "Complete"
    job.save()
    job.append_log(f"[DONE] Pipeline completed in {job.duration_display}")
