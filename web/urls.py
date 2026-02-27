"""Root URL configuration."""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("web.dashboard.api_urls")),
    path("", include("web.dashboard.urls")),
]
