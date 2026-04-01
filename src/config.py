from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    comfy_host: str = os.getenv("COMFY_HOST", "127.0.0.1:8188")
    comfy_pid_file: str = os.getenv("COMFY_PID_FILE", "/tmp/comfyui.pid")
    comfy_api_available_interval_ms: int = int(
        os.getenv("COMFY_API_AVAILABLE_INTERVAL_MS", "50")
    )
    comfy_api_available_max_retries: int = int(
        os.getenv("COMFY_API_AVAILABLE_MAX_RETRIES", "0")
    )
    comfy_api_fallback_max_retries: int = int(
        os.getenv("COMFY_API_FALLBACK_MAX_RETRIES", "500")
    )
    websocket_reconnect_attempts: int = int(
        os.getenv("WEBSOCKET_RECONNECT_ATTEMPTS", "5")
    )
    websocket_reconnect_delay_s: int = int(
        os.getenv("WEBSOCKET_RECONNECT_DELAY_S", "3")
    )
    http_timeout_s: float = float(os.getenv("HTTP_TIMEOUT_S", "30"))
    http_connect_timeout_s: float = float(os.getenv("HTTP_CONNECT_TIMEOUT_S", "5"))
    port: int = int(os.getenv("PORT", "80"))
    port_health: int = int(os.getenv("PORT_HEALTH", os.getenv("PORT", "80")))
    log_level: str = os.getenv("LOG_LEVEL", "info")


settings = Settings()
