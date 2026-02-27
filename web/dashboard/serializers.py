"""DRF serializers for the REST API."""

from rest_framework import serializers

from web.dashboard.models import CrawlJob, CrawlJobChunk


class CrawlJobListSerializer(serializers.ModelSerializer):
    duration = serializers.CharField(source="duration_display", read_only=True)

    class Meta:
        model = CrawlJob
        fields = [
            "id", "name", "status", "seed_urls", "max_pages", "max_depth",
            "chunk_strategy", "chunk_size", "export_format",
            "pages_crawled", "pages_extracted", "total_chunks", "total_tokens",
            "progress", "current_stage", "duration",
            "created_at", "started_at", "finished_at",
        ]


class CrawlJobDetailSerializer(serializers.ModelSerializer):
    duration = serializers.CharField(source="duration_display", read_only=True)
    config = serializers.JSONField(source="config_json", read_only=True)

    class Meta:
        model = CrawlJob
        fields = [
            "id", "name", "status", "seed_urls", "max_pages", "max_depth",
            "chunk_strategy", "chunk_size", "export_format", "config",
            "pages_crawled", "pages_failed", "pages_extracted",
            "total_chunks", "total_tokens",
            "progress", "current_stage", "output_path",
            "log_text", "error_message", "duration",
            "created_at", "started_at", "finished_at",
        ]


class CrawlJobCreateSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=255)
    seed_urls = serializers.ListField(child=serializers.URLField())
    max_pages = serializers.IntegerField(min_value=1, max_value=100000, default=100)
    max_depth = serializers.IntegerField(min_value=0, max_value=20, default=3)
    concurrency = serializers.IntegerField(min_value=1, max_value=50, default=5)
    delay_seconds = serializers.FloatField(min_value=0, max_value=60, default=1.0)
    respect_robots = serializers.BooleanField(default=True)
    chunk_strategy = serializers.ChoiceField(
        choices=["recursive", "sentence", "semantic", "sliding_window"],
        default="recursive",
    )
    chunk_size = serializers.IntegerField(min_value=64, max_value=8192, default=512)
    chunk_overlap = serializers.IntegerField(min_value=0, max_value=2048, default=64)
    export_format = serializers.ChoiceField(
        choices=["jsonl", "parquet", "hf_dataset"],
        default="jsonl",
    )
    target_languages = serializers.ListField(
        child=serializers.CharField(max_length=5),
        default=["en"],
    )


class ChunkSerializer(serializers.ModelSerializer):
    class Meta:
        model = CrawlJobChunk
        fields = [
            "chunk_id", "document_url", "document_title",
            "text", "token_count", "chunk_index", "total_chunks",
            "content_hash", "metadata_json",
        ]
