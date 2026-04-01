from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, status

from .dependencies import comfy_client, generation_service
from .exceptions import ComfyUnavailableError, ComfyValidationError
from .models import GenerateRequest, GenerateResponse, HealthResponse
from .state import is_comfyui_process_alive, worker_state


@asynccontextmanager
async def lifespan(_: FastAPI):
    worker_state.mark_initializing()
    if await comfy_client.server_reachable():
        worker_state.mark_ready()
    yield
    await comfy_client.aclose()


app = FastAPI(
    title="Runpod Load-Balanced ComfyUI Worker",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    reachable = await comfy_client.server_reachable()
    if reachable:
        worker_state.mark_ready()
    else:
        worker_state.mark_initializing(worker_state.last_error)
    return HealthResponse(
        status="ready" if worker_state.ready else "initializing",
        comfyui_reachable=reachable,
        comfyui_process_alive=is_comfyui_process_alive(),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return await root()


@app.get("/ping", response_model=None)
async def ping() -> Response | dict[str, str]:
    reachable = await comfy_client.server_reachable()
    if reachable:
        worker_state.mark_ready()
        return {"status": "healthy"}
    worker_state.mark_initializing("ComfyUI is still initializing")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    try:
        response = await generation_service.generate(request)
    except ComfyValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ComfyUnavailableError as exc:
        worker_state.mark_initializing(str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not worker_state.ready:
        worker_state.mark_ready()
    return response
