"""REST API URL configuration."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from web.dashboard.api_views import CrawlJobViewSet

router = DefaultRouter()
router.register(r"jobs", CrawlJobViewSet, basename="api-jobs")

urlpatterns = [
    path("", include(router.urls)),
]
