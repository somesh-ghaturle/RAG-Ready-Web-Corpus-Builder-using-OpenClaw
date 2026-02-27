"""REST API views."""

from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from web.dashboard.models import CrawlJob, CrawlJobChunk
from web.dashboard.serializers import (
    ChunkSerializer,
    CrawlJobCreateSerializer,
    CrawlJobDetailSerializer,
    CrawlJobListSerializer,
)
from web.dashboard.tasks import cancel_job, start_pipeline_job


class CrawlJobViewSet(viewsets.ModelViewSet):
    """
    API endpoint for crawl jobs.

    list:    GET /api/jobs/
    create:  POST /api/jobs/
    detail:  GET /api/jobs/{id}/
    delete:  DELETE /api/jobs/{id}/
    status:  GET /api/jobs/{id}/status/
    cancel:  POST /api/jobs/{id}/cancel/
    chunks:  GET /api/jobs/{id}/chunks/
    """

    queryset = CrawlJob.objects.all()
    lookup_field = "pk"

    def get_serializer_class(self):
        if self.action == "list":
            return CrawlJobListSerializer
        if self.action == "create":
            return CrawlJobCreateSerializer
        return CrawlJobDetailSerializer

    def create(self, request, *args, **kwargs):
        serializer = CrawlJobCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        d = serializer.validated_data

        config_dict = {
            "crawl": {
                "seed_urls": d["seed_urls"],
                "max_pages": d["max_pages"],
                "max_depth": d["max_depth"],
                "concurrency": d["concurrency"],
                "delay_seconds": d["delay_seconds"],
                "respect_robots_txt": d["respect_robots"],
            },
            "extraction": {},
            "preprocess": {
                "target_languages": d.get("target_languages", ["en"]),
            },
            "chunk": {
                "strategy": d["chunk_strategy"],
                "chunk_size": d["chunk_size"],
                "chunk_overlap": d["chunk_overlap"],
            },
            "embedding": {"enabled": False},
            "export": {
                "format": d["export_format"],
                "output_dir": "output",
            },
            "log_level": "INFO",
        }

        job = CrawlJob.objects.create(
            name=d["name"],
            seed_urls="\n".join(d["seed_urls"]),
            max_pages=d["max_pages"],
            max_depth=d["max_depth"],
            chunk_strategy=d["chunk_strategy"],
            chunk_size=d["chunk_size"],
            export_format=d["export_format"],
            config_json=config_dict,
        )

        start_pipeline_job(str(job.id))

        return Response(
            CrawlJobDetailSerializer(job).data,
            status=status.HTTP_201_CREATED,
        )

    @action(detail=True, methods=["get"])
    def status(self, request, pk=None):
        job = self.get_object()
        return Response({
            "id": str(job.id),
            "status": job.status,
            "progress": job.progress,
            "current_stage": job.current_stage,
            "pages_crawled": job.pages_crawled,
            "pages_extracted": job.pages_extracted,
            "total_chunks": job.total_chunks,
            "total_tokens": job.total_tokens,
            "duration": job.duration_display,
        })

    @action(detail=True, methods=["post"])
    def cancel(self, request, pk=None):
        job = self.get_object()
        cancelled = cancel_job(str(job.id))
        return Response({"cancelled": cancelled})

    @action(detail=True, methods=["get"])
    def chunks(self, request, pk=None):
        job = self.get_object()
        chunks = CrawlJobChunk.objects.filter(job=job)

        # Optional search
        q = request.query_params.get("q")
        if q:
            chunks = chunks.filter(text__icontains=q)

        doc_url = request.query_params.get("doc")
        if doc_url:
            chunks = chunks.filter(document_url=doc_url)

        page = self.paginate_queryset(chunks)
        if page is not None:
            serializer = ChunkSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = ChunkSerializer(chunks, many=True)
        return Response(serializer.data)
