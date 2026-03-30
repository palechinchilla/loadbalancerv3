import asyncio
import time
import uuid
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse

from worker_runtime import (
    ComfyWorker,
    GenerateRequest,
    SETTINGS,
    Trace,
    json_response,
    log,
    log_exception,
)


worker = ComfyWorker(SETTINGS)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    startup_task = asyncio.create_task(worker.wait_until_ready())
    app_instance.state.worker = worker
    try:
        yield
    finally:
        startup_task.cancel()
        with suppress(asyncio.CancelledError):
            await startup_task
        await worker.close()


app = FastAPI(title="ComfyUI Load Balancing Worker", lifespan=lifespan)


@app.get("/ping")
async def ping() -> Response:
    return Response(status_code=200, content="OK") if worker.ready else Response(status_code=204)


@app.post("/generate")
async def generate(request: GenerateRequest) -> Response:
    trace = Trace(str(uuid.uuid4()), warmup_state=worker.warmup_state)
    if not worker.ready:
        payload = {"error": "ComfyUI is not ready yet"}
        return json_response(payload, 503, trace, error=payload["error"])

    log("Received generate request", request_id=trace.request_id)
    worker.stamp(trace)
    trace.stages["receive"] = time.perf_counter_ns() - trace.started_ns
    try:
        async with asyncio.timeout(SETTINGS.processing_timeout_s):
            result = await worker.run(request, trace)
    except TimeoutError:
        payload = {"error": f"Workflow processing timed out after {SETTINGS.processing_timeout_s}s"}
        log(
            "Processing timed out after %ss",
            SETTINGS.processing_timeout_s,
            request_id=trace.request_id,
        )
        trace.warmup_state = worker.warmup_state
        return json_response(payload, 504, trace, error=payload["error"])
    except Exception as exc:
        log_exception("Unexpected error: %s", exc, request_id=trace.request_id)
        payload = {"error": f"An unexpected error occurred: {exc}"}
        trace.warmup_state = worker.warmup_state
        return json_response(payload, 500, trace, error=payload["error"])

    trace.warmup_state = worker.warmup_state
    if "error" in result:
        status_code = 422 if "validation" in result.get("error", "").lower() else 500
        return json_response(result, status_code, trace, error=result["error"])
    return json_response(result, 200, trace)


@app.post("/generate-stream")
async def generate_stream(request: GenerateRequest) -> Response:
    if not worker.ready:
        return json_response({"error": "ComfyUI is not ready yet"}, 503)
    request_id = str(uuid.uuid4())
    log("Received streaming generate request", request_id=request_id)
    return StreamingResponse(
        worker.stream(request, request_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
