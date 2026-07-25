"""Uvicorn entrypoint: ``uvicorn main:app``.

Building the app constructs the real providers from settings, so the provider
API keys must be present in the environment / .env before starting the server.
"""

from __future__ import annotations

from src.api.app import create_app

app = create_app()
