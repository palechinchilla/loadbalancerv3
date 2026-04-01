from __future__ import annotations

import os
from dataclasses import dataclass, field

from .config import settings


def get_comfyui_pid() -> int | None:
    try:
        with open(settings.comfy_pid_file, "r", encoding="utf-8") as pid_file:
            return int(pid_file.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def is_comfyui_process_alive() -> bool | None:
    pid = get_comfyui_pid()
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@dataclass(slots=True)
class WorkerState:
    ready: bool = False
    last_error: str | None = None
    request_count: int = 0
    last_prompt_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def mark_ready(self) -> None:
        self.ready = True
        self.last_error = None

    def mark_initializing(self, error: str | None = None) -> None:
        self.ready = False
        self.last_error = error


worker_state = WorkerState()
