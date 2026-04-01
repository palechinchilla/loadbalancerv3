from __future__ import annotations


class ComfyServiceError(Exception):
    """Base service exception."""


class ComfyValidationError(ComfyServiceError):
    """Workflow validation failed."""


class ComfyUnavailableError(ComfyServiceError):
    """ComfyUI is unavailable."""
