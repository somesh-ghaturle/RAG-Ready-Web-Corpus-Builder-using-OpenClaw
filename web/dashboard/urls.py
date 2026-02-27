"""Dashboard URL configuration."""

from django.urls import path

from web.dashboard import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("jobs/new/", views.new_job, name="new_job"),
    path("jobs/<uuid:job_id>/", views.job_detail, name="job_detail"),
    path("jobs/<uuid:job_id>/status/", views.job_status_api, name="job_status_api"),
    path("jobs/<uuid:job_id>/cancel/", views.cancel_job_view, name="cancel_job"),
    path("jobs/<uuid:job_id>/delete/", views.delete_job_view, name="delete_job"),
    path("jobs/<uuid:job_id>/chunks/", views.job_chunks, name="job_chunks"),
    path("jobs/<uuid:job_id>/logs/", views.job_logs, name="job_logs"),
]
