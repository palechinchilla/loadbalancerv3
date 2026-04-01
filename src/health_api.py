from __future__ import annotations

from fastapi import FastAPI, Response, status

from .dependencies import comfy_client
from .state import worker_state

health_app = FastAPI(title="Runpod Health Server", version="1.0.0")


@health_app.get("/ping", response_model=None)
async def ping() -> Response | dict[str, str]:
    reachable = await comfy_client.server_reachable()
    if reachable:
        worker_state.mark_ready()
        return {"status": "healthy"}
    worker_state.mark_initializing("ComfyUI is still initializing")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
