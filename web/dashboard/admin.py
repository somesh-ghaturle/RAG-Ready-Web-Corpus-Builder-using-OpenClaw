from django.contrib import admin
from web.dashboard.models import CrawlJob, CrawlJobChunk


@admin.register(CrawlJob)
class CrawlJobAdmin(admin.ModelAdmin):
    list_display = ["name", "status", "pages_crawled", "total_chunks", "created_at"]
    list_filter = ["status"]
    search_fields = ["name", "seed_urls"]
    readonly_fields = ["id", "created_at", "updated_at"]


@admin.register(CrawlJobChunk)
class CrawlJobChunkAdmin(admin.ModelAdmin):
    list_display = ["chunk_id", "job", "document_title", "token_count"]
    list_filter = ["job"]
    search_fields = ["text", "document_title", "document_url"]
