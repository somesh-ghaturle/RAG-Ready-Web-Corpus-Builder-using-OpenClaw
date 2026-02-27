"""Django models for crawl job tracking."""

import json
import uuid

from django.db import models
from django.utils import timezone


class CrawlJob(models.Model):
    """A single pipeline execution."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, help_text="Human-friendly job name")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING, db_index=True)

    # Configuration (stored as JSON)
    config_json = models.JSONField(default=dict, help_text="Full PipelineConfig as JSON")

    # Quick-access config fields
    seed_urls = models.TextField(help_text="Newline-separated seed URLs")
    max_pages = models.PositiveIntegerField(default=100)
    max_depth = models.PositiveIntegerField(default=3)
    chunk_strategy = models.CharField(max_length=30, default="recursive")
    chunk_size = models.PositiveIntegerField(default=512)
    export_format = models.CharField(max_length=20, default="jsonl")

    # Execution tracking
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Results & stats
    pages_crawled = models.PositiveIntegerField(default=0)
    pages_failed = models.PositiveIntegerField(default=0)
    pages_extracted = models.PositiveIntegerField(default=0)
    total_chunks = models.PositiveIntegerField(default=0)
    total_tokens = models.PositiveBigIntegerField(default=0)
    output_path = models.CharField(max_length=500, blank=True, default="")

    # Logs
    log_text = models.TextField(blank=True, default="", help_text="Pipeline execution log")
    error_message = models.TextField(blank=True, default="")

    # Progress (0-100)
    progress = models.PositiveSmallIntegerField(default=0)
    current_stage = models.CharField(max_length=50, blank=True, default="")

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Crawl Job"
        verbose_name_plural = "Crawl Jobs"

    def __str__(self):
        return f"{self.name} ({self.status})"

    @property
    def duration_seconds(self):
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        if self.started_at:
            return (timezone.now() - self.started_at).total_seconds()
        return 0

    @property
    def duration_display(self):
        s = self.duration_seconds
        if s < 60:
            return f"{s:.0f}s"
        if s < 3600:
            return f"{s / 60:.1f}m"
        return f"{s / 3600:.1f}h"

    @property
    def seed_urls_list(self):
        return [u.strip() for u in self.seed_urls.split("\n") if u.strip()]

    @property
    def display_name(self):
        return self.name or f"Job {str(self.id)[:8]}"

    @property
    def elapsed_display(self):
        return self.duration_display

    @property
    def log_lines(self):
        if not self.log_text:
            return []
        return self.log_text.strip().split("\n")

    @property
    def log_output(self):
        return self.log_text

    def append_log(self, message):
        self.log_text += message + "\n"
        self.save(update_fields=["log_text", "updated_at"])


class CrawlJobChunk(models.Model):
    """Stores individual chunks from a completed crawl job for browsing."""

    job = models.ForeignKey(CrawlJob, on_delete=models.CASCADE, related_name="chunks")
    chunk_id = models.CharField(max_length=64, db_index=True)
    document_url = models.URLField(max_length=2000)
    document_title = models.CharField(max_length=500, blank=True, default="")
    text = models.TextField()
    token_count = models.PositiveIntegerField(default=0)
    chunk_index = models.PositiveIntegerField(default=0)
    total_chunks = models.PositiveIntegerField(default=0)
    content_hash = models.CharField(max_length=64, blank=True, default="")
    metadata_json = models.JSONField(default=dict)

    class Meta:
        ordering = ["document_url", "chunk_index"]
        verbose_name = "Chunk"

    def __str__(self):
        return f"{self.chunk_id} ({self.document_title})"
