"""Dashboard views for the web UI."""

import json
from pathlib import Path

from django.conf import settings
from django.db.models import Sum, Count, Avg, Q
from django.http import JsonResponse, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from web.dashboard.forms import CrawlJobForm
from web.dashboard.models import CrawlJob, CrawlJobChunk
from web.dashboard.tasks import cancel_job, start_pipeline_job


def dashboard(request):
    """Main dashboard — overview of all jobs and stats."""
    jobs = CrawlJob.objects.all()[:20]

    # Aggregate stats
    totals = CrawlJob.objects.filter(status=CrawlJob.Status.COMPLETED).aggregate(
        total_pages=Sum("pages_crawled"),
        total_chunks=Sum("total_chunks"),
        total_tokens=Sum("total_tokens"),
        total_jobs=Count("id"),
    )
    running_count = CrawlJob.objects.filter(status=CrawlJob.Status.RUNNING).count()

    context = {
        "jobs": jobs,
        "totals": {
            "pages": totals["total_pages"] or 0,
            "chunks": totals["total_chunks"] or 0,
            "tokens": totals["total_tokens"] or 0,
            "jobs": totals["total_jobs"] or 0,
        },
        "running_count": running_count,
    }
    return render(request, "dashboard/index.html", context)


def new_job(request):
    """Create a new crawl job."""
    if request.method == "POST":
        form = CrawlJobForm(request.POST)
        if form.is_valid():
            config_dict = form.build_config_dict()
            d = form.cleaned_data

            job = CrawlJob.objects.create(
                name=d["name"],
                seed_urls=d["seed_urls"],
                max_pages=d["max_pages"],
                max_depth=d["max_depth"],
                chunk_strategy=d["chunk_strategy"],
                chunk_size=d["chunk_size"],
                export_format=d["export_format"],
                config_json=config_dict,
            )

            # Start the pipeline in background
            start_pipeline_job(str(job.id))

            return redirect("job_detail", job_id=job.id)
    else:
        form = CrawlJobForm()

    return render(request, "dashboard/new_job.html", {"form": form})


def job_detail(request, job_id):
    """View details, logs, and status of a job."""
    job = get_object_or_404(CrawlJob, pk=job_id)

    chunks_qs = CrawlJobChunk.objects.filter(job=job)
    chunks_count = chunks_qs.count()
    chunks = chunks_qs[:20]
    total_tokens = chunks_qs.aggregate(t=Sum("token_count"))["t"] or 0

    context = {
        "job": job,
        "chunks": chunks,
        "chunks_count": chunks_count,
        "total_tokens": total_tokens,
        "config_display": json.dumps(job.config_json, indent=2),
        "log_lines": job.log_lines,
    }
    return render(request, "dashboard/job_detail.html", context)


def job_status_api(request, job_id):
    """AJAX endpoint for live job status polling."""
    job = get_object_or_404(CrawlJob, pk=job_id)
    return JsonResponse({
        "id": str(job.id),
        "status": job.status,
        "progress": job.progress,
        "current_stage": job.current_stage,
        "pages_crawled": job.pages_crawled,
        "pages_extracted": job.pages_extracted,
        "total_chunks": job.total_chunks,
        "total_tokens": job.total_tokens,
        "duration": job.duration_display,
        "log_text": job.log_text,
        "error_message": job.error_message,
    })


@require_POST
def cancel_job_view(request, job_id):
    """Cancel a running job."""
    cancel_job(str(job_id))
    return redirect("job_detail", job_id=job_id)


@require_POST
def delete_job_view(request, job_id):
    """Delete a job and its chunks."""
    job = get_object_or_404(CrawlJob, pk=job_id)
    job.delete()
    return redirect("dashboard")


def job_chunks(request, job_id):
    """Browse chunks for a specific job."""
    from django.core.paginator import Paginator

    job = get_object_or_404(CrawlJob, pk=job_id)

    # Filters
    query = request.GET.get("q", "").strip()
    selected_source = request.GET.get("source", "").strip()
    page_num = request.GET.get("page", 1)

    chunks_qs = CrawlJobChunk.objects.filter(job=job)

    if query:
        chunks_qs = chunks_qs.filter(Q(text__icontains=query) | Q(document_title__icontains=query))

    if selected_source:
        chunks_qs = chunks_qs.filter(document_url=selected_source)

    total_chunks = chunks_qs.count()
    total_tokens = chunks_qs.aggregate(t=Sum("token_count"))["t"] or 0

    # Get unique source URLs for filter dropdown
    source_urls = (
        CrawlJobChunk.objects
        .filter(job=job)
        .values_list("document_url", flat=True)
        .distinct()
        .order_by("document_url")
    )

    paginator = Paginator(chunks_qs, 20)
    page_obj = paginator.get_page(page_num)

    context = {
        "job": job,
        "chunks": page_obj,
        "page_obj": page_obj,
        "query": query,
        "selected_source": selected_source,
        "source_urls": source_urls,
        "total_chunks": total_chunks,
        "total_tokens": total_tokens,
    }
    return render(request, "dashboard/job_chunks.html", context)


def job_logs(request, job_id):
    """View full logs for a job."""
    job = get_object_or_404(CrawlJob, pk=job_id)
    return render(request, "dashboard/job_logs.html", {"job": job, "log_lines": job.log_lines})
