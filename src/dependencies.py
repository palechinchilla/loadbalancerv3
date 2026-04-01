from __future__ import annotations

from .comfy_client import ComfyClient
from .service import GenerationService

comfy_client = ComfyClient()
generation_service = GenerationService(comfy_client)
