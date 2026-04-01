"""Granian ASGI targets for the load-balanced ComfyUI worker."""

from __future__ import annotations

from .api import app as main_app
from .health_api import health_app

__all__ = ["main_app", "health_app"]
