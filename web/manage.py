#!/usr/bin/env python
"""Django's command-line utility for the RAG Corpus Builder web UI."""
import os
import sys
from pathlib import Path


def main():
    """Run administrative tasks."""
    # Ensure the project root is on sys.path so 'web' package is importable
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
