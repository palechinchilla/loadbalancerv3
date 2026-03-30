import asyncio
import base64
import binascii
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
import websockets
from fastapi.responses import Response
from pydantic import BaseModel

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


APP_NAME = "worker-comfyui"
JSON_TYPE = "application/json"
STAGES = (
    "receive",
    "upload",
    "queue",
    "websocket_wait",
    "output_manifest",
    "image_fetch",
    "base64_encode",
    "response_serialize",
    "total",
)
POSTPROCESS = ("output_manifest", "image_fetch", "base64_encode", "response_serialize")
Manifest = dict[str, dict[str, Any]]


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def ms(duration_ns: int) -> float:
    return round(duration_ns / 1_000_000, 3)


def dumps(payload: Any) -> bytes:
    if orjson is not None:
        return orjson.dumps(payload)
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def loads(payload: bytes | str) -> Any:
    if orjson is not None:
        return orjson.loads(payload)
    return json.loads(payload)


def prefixed(message: str, request_id: str | None = None) -> str:
    return f"{APP_NAME} - [{request_id}] {message}" if request_id else f"{APP_NAME} - {message}"


class _SuppressPingAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return '"GET /ping HTTP/1.1"' not in record.getMessage()


LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").addFilter(_SuppressPingAccessFilter())


def log(
    message: str,
    *args: Any,
    request_id: str | None = None,
    level: int = logging.INFO,
) -> None:
    logger.log(level, prefixed(message, request_id), *args)


def log_exception(message: str, *args: Any, request_id: str | None = None) -> None:
    logger.exception(prefixed(message, request_id), *args)


def pid_alive(pid_file: str) -> bool | None:
    try:
        with open(pid_file, "r", encoding="utf-8") as handle:
            pid = int(handle.read().strip())
    except (FileNotFoundError, ValueError):
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def ws_event(raw_message: Any) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(raw_message, str):
        return None
    try:
        payload = loads(raw_message)
    except Exception:
        return None
    return payload.get("type"), payload.get("data") or {}


def record_output(outputs: Manifest, prompt_id: str, data: dict[str, Any]) -> None:
    if data.get("prompt_id") != prompt_id or not isinstance(data.get("output"), dict):
        return
    node_id = data.get("node")
    if node_id is None:
        return
    output = outputs.get(str(node_id), {}).copy()
    output.update({key: value for key, value in data["output"].items() if key != "images"})
    if "images" in data["output"]:
        output["images"] = data["output"]["images"]
    outputs[str(node_id)] = output


def manifest_has_images(outputs: Manifest) -> bool:
    return any(
        image.get("filename") and image.get("type") != "temp"
        for node in outputs.values()
        for image in node.get("images", [])
    )


def manifest_needs_history(outputs: Manifest, errors: list[str]) -> bool:
    if errors:
        return True
    saw_image = False
    for node in outputs.values():
        for image in node.get("images", []):
            if image.get("type") == "temp":
                continue
            saw_image = True
            if not image.get("filename") or image.get("type") is None:
                return True
    return not saw_image


def merge_outputs(primary: Manifest, fallback: dict[str, Any]) -> Manifest:
    merged = {
        str(node_id): dict(node_output)
        for node_id, node_output in fallback.items()
        if isinstance(node_output, dict)
    }
    for node_id, node_output in primary.items():
        merged.setdefault(str(node_id), {}).update(
            {key: value for key, value in node_output.items() if key != "images"}
        )
        if "images" in node_output:
            merged[str(node_id)]["images"] = node_output["images"]
    return merged


async def gather_limited(items: list[Any], limit: int, func) -> list[Any]:
    semaphore = asyncio.Semaphore(max(1, limit))

    async def run(item: Any) -> Any:
        async with semaphore:
            return await func(item)

    return await asyncio.gather(*(run(item) for item in items))


@dataclass(slots=True, frozen=True)
class Settings:
    comfy_host: str = "127.0.0.1:8188"
    comfy_pid_file: str = "/tmp/comfyui.pid"
    processing_timeout_s: int = 300
    api_interval_ms: int = 50
    api_max_retries: int = 0
    api_fallback_max_retries: int = 500
    ws_reconnect_attempts: int = 5
    ws_reconnect_delay_s: float = 3.0
    connect_timeout_s: float = 2.0
    read_timeout_s: float = 30.0
    write_timeout_s: float = 10.0
    pool_timeout_s: float = 5.0
    status_timeout_s: float = 5.0
    prompt_timeout_s: float = 30.0
    history_timeout_s: float = 15.0
    output_timeout_s: float = 30.0
    upload_timeout_s: float = 30.0
    max_connections: int = 64
    max_keepalive_connections: int = 32
    keepalive_expiry_s: float = 30.0
    output_fetch_concurrency: int = 8
    upload_concurrency: int = 4
    warmup_workflow_json: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            comfy_host=os.environ.get("COMFY_HOST", "127.0.0.1:8188"),
            comfy_pid_file=os.environ.get("COMFY_PID_FILE", "/tmp/comfyui.pid"),
            processing_timeout_s=env_int("PROCESSING_TIMEOUT_S", 300),
            api_interval_ms=env_int("COMFY_API_AVAILABLE_INTERVAL_MS", 50),
            api_max_retries=env_int("COMFY_API_AVAILABLE_MAX_RETRIES", 0),
            ws_reconnect_attempts=env_int("WEBSOCKET_RECONNECT_ATTEMPTS", 5),
            ws_reconnect_delay_s=env_float("WEBSOCKET_RECONNECT_DELAY_S", 3.0),
            connect_timeout_s=env_float("COMFY_HTTP_CONNECT_TIMEOUT_S", 2.0),
            read_timeout_s=env_float("COMFY_HTTP_READ_TIMEOUT_S", 30.0),
            write_timeout_s=env_float("COMFY_HTTP_WRITE_TIMEOUT_S", 10.0),
            pool_timeout_s=env_float("COMFY_HTTP_POOL_TIMEOUT_S", 5.0),
            status_timeout_s=env_float("COMFY_HTTP_STATUS_READ_TIMEOUT_S", 5.0),
            prompt_timeout_s=env_float("COMFY_HTTP_PROMPT_READ_TIMEOUT_S", 30.0),
            history_timeout_s=env_float("COMFY_HTTP_HISTORY_READ_TIMEOUT_S", 15.0),
            output_timeout_s=env_float("COMFY_HTTP_OUTPUT_READ_TIMEOUT_S", 30.0),
            upload_timeout_s=env_float("COMFY_HTTP_UPLOAD_READ_TIMEOUT_S", 30.0),
            max_connections=max(1, env_int("COMFY_HTTP_MAX_CONNECTIONS", 64)),
            max_keepalive_connections=max(
                1, env_int("COMFY_HTTP_MAX_KEEPALIVE_CONNECTIONS", 32)
            ),
            keepalive_expiry_s=env_float("COMFY_HTTP_KEEPALIVE_EXPIRY_S", 30.0),
            output_fetch_concurrency=max(1, env_int("COMFY_OUTPUT_FETCH_CONCURRENCY", 8)),
            upload_concurrency=max(1, env_int("COMFY_UPLOAD_CONCURRENCY", 4)),
            warmup_workflow_json=os.environ.get("WARMUP_WORKFLOW_JSON"),
        )

    @property
    def base_url(self) -> str:
        return f"http://{self.comfy_host}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.comfy_host}/ws"

    def timeout(self, *, read: float | None = None, write: float | None = None) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.connect_timeout_s,
            read=self.read_timeout_s if read is None else read,
            write=self.write_timeout_s if write is None else write,
            pool=self.pool_timeout_s,
        )


class ImageInput(BaseModel):
    model_config = {"extra": "ignore"}

    name: str
    image: str


class GenerateRequest(BaseModel):
    model_config = {"extra": "ignore"}

    workflow: dict[str, Any]
    images: list[ImageInput] | None = None
    comfy_org_api_key: str | None = None


@dataclass(slots=True)
class Trace:
    request_id: str
    started_ns: int = field(default_factory=time.perf_counter_ns)
    stages: dict[str, int] = field(default_factory=lambda: {stage: 0 for stage in STAGES})
    response_bytes: int = 0
    output_source: str = "unresolved"
    ready_age_ms: float | None = None
    warmup_state: str = "unknown"

    @contextmanager
    def span(self, stage: str):
        started_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            self.stages[stage] += time.perf_counter_ns() - started_ns

    def add(self, stage: str, duration_ns: int) -> None:
        self.stages[stage] += max(0, duration_ns)

    def finish(self) -> None:
        self.stages["total"] = time.perf_counter_ns() - self.started_ns

    def headers(self) -> dict[str, str]:
        postprocess_ns = sum(self.stages[stage] for stage in POSTPROCESS)
        headers = {
            "X-Worker-Request-Id": self.request_id,
            "X-Worker-Total-Ms": f"{ms(self.stages['total']):.3f}",
            "X-Worker-Queue-Ms": f"{ms(self.stages['queue']):.3f}",
            "X-Worker-Execute-Ms": f"{ms(self.stages['websocket_wait']):.3f}",
            "X-Worker-Postprocess-Ms": f"{ms(postprocess_ns):.3f}",
            "X-Worker-Response-Bytes": str(self.response_bytes),
            "X-Worker-Warmup-State": self.warmup_state,
        }
        if self.ready_age_ms is not None:
            headers["X-Worker-Ready-Age-Ms"] = f"{self.ready_age_ms:.3f}"
        return headers

    def summary(self, status_code: int, error: str | None = None) -> dict[str, Any]:
        payload = {
            "request_id": self.request_id,
            "status_code": status_code,
            "output_source": self.output_source,
            "response_bytes": self.response_bytes,
            "ready_age_ms": self.ready_age_ms,
            "warmup_state": self.warmup_state,
            "stages_ms": {stage: ms(self.stages[stage]) for stage in STAGES},
        }
        if error:
            payload["error"] = error
        return payload


def json_response(
    payload: dict[str, Any],
    status_code: int,
    trace: Trace | None = None,
    *,
    error: str | None = None,
) -> Response:
    if trace is None:
        return Response(content=dumps(payload), media_type=JSON_TYPE, status_code=status_code)
    with trace.span("response_serialize"):
        body = dumps(payload)
    trace.response_bytes = len(body)
    trace.finish()
    log("latency %s", dumps(trace.summary(status_code, error)).decode("utf-8", errors="replace"))
    return Response(
        content=body,
        media_type=JSON_TYPE,
        status_code=status_code,
        headers=trace.headers(),
    )


SETTINGS = Settings.from_env()


class ComfyWorker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ready = False
        self.api_reachable_at_ns: int | None = None
        self.warmup_started_at_ns: int | None = None
        self.warmup_finished_at_ns: int | None = None
        self.first_request_at_ns: int | None = None
        self.warmup_state = "not_started" if settings.warmup_workflow_json else "disabled"
        self.warmup_task: asyncio.Task[None] | None = None
        self.http = httpx.AsyncClient(
            base_url=settings.base_url,
            trust_env=False,
            http2=False,
            follow_redirects=False,
            limits=httpx.Limits(
                max_connections=settings.max_connections,
                max_keepalive_connections=settings.max_keepalive_connections,
                keepalive_expiry=settings.keepalive_expiry_s,
            ),
            timeout=settings.timeout(),
        )
        self.timeouts = {
            "status": settings.timeout(read=settings.status_timeout_s),
            "prompt": settings.timeout(
                read=settings.prompt_timeout_s, write=settings.prompt_timeout_s
            ),
            "history": settings.timeout(read=settings.history_timeout_s),
            "output": settings.timeout(read=settings.output_timeout_s),
            "upload": settings.timeout(
                read=settings.upload_timeout_s, write=settings.upload_timeout_s
            ),
        }

    @staticmethod
    def response_json(response: httpx.Response) -> Any:
        return loads(response.content)

    async def close(self) -> None:
        if self.warmup_task is not None:
            self.warmup_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self.warmup_task
        await self.http.aclose()

    async def is_reachable(self) -> bool:
        try:
            response = await self.http.get("/", timeout=self.timeouts["status"])
            return response.status_code == 200
        except httpx.RequestError:
            return False

    def ready_age(self, now_ns: int | None = None) -> float | None:
        if self.api_reachable_at_ns is None:
            return None
        return ms((time.perf_counter_ns() if now_ns is None else now_ns) - self.api_reachable_at_ns)

    def stamp(self, trace: Trace) -> None:
        now_ns = time.perf_counter_ns()
        if self.first_request_at_ns is None:
            self.first_request_at_ns = now_ns
            log(
                "First request arrived %.3fms after readiness.",
                self.ready_age(now_ns) or 0.0,
                level=logging.DEBUG,
            )
        trace.ready_age_ms = self.ready_age(now_ns)
        trace.warmup_state = self.warmup_state

    async def wait_until_ready(self) -> None:
        delay_s = max(1, self.settings.api_interval_ms) / 1000
        log_every = max(1, int(10_000 / max(1, self.settings.api_interval_ms)))
        url = f"{self.settings.base_url}/"
        attempts = 0
        log("Checking API server at %s...", url)
        while True:
            pid_state = pid_alive(self.settings.comfy_pid_file)
            if pid_state is False:
                log("ComfyUI process has exited. Server will not become reachable.")
                return
            if await self.is_reachable():
                self.ready = True
                self.api_reachable_at_ns = time.perf_counter_ns()
                log("API is reachable")
                log(
                    "Startup ready_ns=%s warmup_state=%s",
                    self.api_reachable_at_ns,
                    self.warmup_state,
                    level=logging.DEBUG,
                )
                self.start_warmup()
                return
            attempts += 1
            max_retries = (
                self.settings.api_max_retries or self.settings.api_fallback_max_retries
            )
            if pid_state is None and attempts >= max_retries:
                log(
                    "Failed to connect to server at %s after %s attempts (no PID file found).",
                    url,
                    max_retries,
                )
                return
            if attempts % log_every == 0:
                log(
                    "Still waiting for API server... (%ss elapsed, attempt %s)",
                    f"{attempts * delay_s:.0f}",
                    attempts,
                )
            await asyncio.sleep(delay_s)

    def start_warmup(self) -> None:
        if self.warmup_state != "not_started" or self.warmup_task is not None:
            return
        self.warmup_state = "running"
        self.warmup_started_at_ns = time.perf_counter_ns()
        self.warmup_task = asyncio.create_task(self.run_warmup())

    async def run_warmup(self) -> None:
        workflow_json = self.settings.warmup_workflow_json
        if not workflow_json:
            self.warmup_state = "disabled"
            return
        request_id = f"warmup-{uuid.uuid4()}"
        log("Warmup started.")
        try:
            workflow = loads(workflow_json)
            if not isinstance(workflow, dict):
                raise ValueError("WARMUP_WORKFLOW_JSON must decode to a JSON object.")
            client_id = str(uuid.uuid4())
            prompt_id = await self.queue_workflow(workflow, client_id)
            log("Warmup queued prompt %s", prompt_id, request_id=request_id, level=logging.DEBUG)
            async with asyncio.timeout(self.settings.processing_timeout_s):
                done, errors, _ = await self.monitor(prompt_id, client_id)
            if not done:
                raise RuntimeError("; ".join(errors) if errors else "warmup did not complete")
            self.warmup_state = "completed"
            self.warmup_finished_at_ns = time.perf_counter_ns()
            log(
                "Warmup finished in %.3fms.",
                ms(self.warmup_finished_at_ns - self.warmup_started_at_ns),
            )
        except asyncio.CancelledError:
            log("Warmup task cancelled.", level=logging.DEBUG)
            raise
        except Exception as exc:
            self.warmup_state = "failed"
            self.warmup_finished_at_ns = time.perf_counter_ns()
            log_exception("Warmup failed: %s", exc)
        finally:
            log(
                "Warmup state=%s started_at=%s finished_at=%s",
                self.warmup_state,
                self.warmup_started_at_ns,
                self.warmup_finished_at_ns,
                request_id=request_id,
                level=logging.DEBUG,
            )

    async def available_checkpoints(self) -> list[str]:
        try:
            response = await self.http.get("/object_info", timeout=self.timeouts["status"])
            response.raise_for_status()
            required = (
                self.response_json(response)
                .get("CheckpointLoaderSimple", {})
                .get("input", {})
                .get("required", {})
            )
            values = required.get("ckpt_name")
            return values[0] if values and isinstance(values[0], list) else []
        except Exception as exc:
            log("Warning: Could not fetch available models: %s", exc)
            return []

    async def validation_error(self, response: httpx.Response) -> str:
        raw = response.text
        try:
            data = self.response_json(response)
        except Exception:
            return f"ComfyUI validation failed (could not parse error response): {raw}"
        error_value = data.get("error")
        if isinstance(error_value, dict):
            message = error_value.get("message") or "Workflow validation failed"
        else:
            message = str(error_value or data.get("message") or "Workflow validation failed")
        node_errors: list[str] = []
        for node_id, node_error in (data.get("node_errors") or {}).items():
            if isinstance(node_error, dict):
                node_errors.extend(f"Node {node_id} ({key}): {value}" for key, value in node_error.items())
            else:
                node_errors.append(f"Node {node_id}: {node_error}")
        if data.get("type") == "prompt_outputs_failed_validation":
            message = data.get("message") or "Workflow validation failed"
        if node_errors:
            message = f"{message}:\n" + "\n".join(f"- {detail}" for detail in node_errors)
        elif raw and raw not in message:
            message = f"{message}. Raw response: {raw}"
        needs_models = data.get("type") == "prompt_outputs_failed_validation" or any(
            "not in list" in detail and "ckpt_name" in detail for detail in node_errors
        )
        if not needs_models:
            return message
        checkpoints = await self.available_checkpoints()
        hint = "\n\nThis usually means a required model or parameter is not available."
        if checkpoints:
            hint += f"\nAvailable checkpoint models: {', '.join(checkpoints)}"
        else:
            hint += "\nNo checkpoint models appear to be available. Please check your model installation."
        return message + hint

    async def upload_images(
        self, images: list[ImageInput], request_id: str | None = None
    ) -> list[str]:
        if not images:
            return []
        log("Uploading %s image(s)...", len(images), request_id=request_id)

        async def upload(image: ImageInput) -> str | None:
            try:
                raw = image.image.split(",", 1)[1] if "," in image.image else image.image
                blob = base64.b64decode(raw.strip(), validate=True)
                response = await self.http.post(
                    "/upload/image",
                    files={
                        "image": (image.name, blob, "image/png"),
                        "overwrite": (None, "true"),
                    },
                    timeout=self.timeouts["upload"],
                )
                response.raise_for_status()
                return None
            except (binascii.Error, ValueError) as exc:
                return f"Error decoding base64 for {image.name}: {exc}"
            except httpx.TimeoutException:
                return f"Timeout uploading {image.name}"
            except httpx.RequestError as exc:
                return f"Error uploading {image.name}: {exc}"
            except Exception as exc:
                return f"Unexpected error uploading {image.name}: {exc}"

        errors = [error for error in await gather_limited(images, self.settings.upload_concurrency, upload) if error]
        for error in errors:
            log(error, request_id=request_id)
        if not errors:
            log("image(s) upload complete", request_id=request_id)
        return errors

    async def queue_workflow(
        self,
        workflow: dict[str, Any],
        client_id: str,
        comfy_org_api_key: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {"prompt": workflow, "client_id": client_id}
        effective_key = comfy_org_api_key or os.environ.get("COMFY_ORG_API_KEY")
        if effective_key:
            payload["extra_data"] = {"api_key_comfy_org": effective_key}
        response = await self.http.post(
            "/prompt",
            content=dumps(payload),
            headers={"Content-Type": JSON_TYPE},
            timeout=self.timeouts["prompt"],
        )
        if response.status_code == 400:
            raise ValueError(await self.validation_error(response))
        response.raise_for_status()
        prompt_id = self.response_json(response).get("prompt_id")
        if not prompt_id:
            raise ValueError(f"Missing 'prompt_id' in queue response: {response.text}")
        return prompt_id

    async def history(self, prompt_id: str) -> dict[str, Any]:
        response = await self.http.get(f"/history/{prompt_id}", timeout=self.timeouts["history"])
        response.raise_for_status()
        return self.response_json(response)

    async def image_bytes(
        self, filename: str, subfolder: str, image_type: str | None
    ) -> bytes | None:
        try:
            response = await self.http.get(
                "/view",
                params={"filename": filename, "subfolder": subfolder, "type": image_type},
                timeout=self.timeouts["output"],
            )
            response.raise_for_status()
            return response.content
        except httpx.TimeoutException:
            log("Timeout fetching image data for %s", filename)
        except httpx.RequestError as exc:
            log("Error fetching image data for %s: %s", filename, exc)
        except Exception as exc:
            log("Unexpected error fetching image data for %s: %s", filename, exc)
        return None

    async def watch_prompt(
        self,
        prompt_id: str,
        client_id: str,
        *,
        emit_progress: bool,
        reconnect: bool,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        outputs: Manifest = {}
        attempts = 0
        while attempts <= self.settings.ws_reconnect_attempts:
            try:
                ws_url = f"{self.settings.ws_url}?clientId={client_id}"
                log("Connecting to websocket: %s", ws_url, level=logging.DEBUG)
                async with websockets.connect(ws_url, open_timeout=10) as websocket:
                    log("Websocket connected", level=logging.DEBUG)
                    async for raw_message in websocket:
                        event = ws_event(raw_message)
                        if event is None:
                            continue
                        kind, data = event
                        if kind == "executed":
                            record_output(outputs, prompt_id, data)
                            continue
                        if kind == "progress" and emit_progress:
                            yield "progress", {"step": data.get("value", 0), "total": data.get("max", 0)}
                            continue
                        if kind == "executing":
                            if data.get("node") is None and data.get("prompt_id") == prompt_id:
                                yield "complete", {"outputs": outputs}
                                return
                            if emit_progress and data.get("node"):
                                yield "status", {"message": f"Executing node {data['node']}"}
                            continue
                        if kind == "execution_error" and data.get("prompt_id") == prompt_id:
                            yield "error", {
                                "message": (
                                    f"Node Type: {data.get('node_type')}, "
                                    f"Node ID: {data.get('node_id')}, "
                                    f"Message: {data.get('exception_message')}"
                                ),
                                "outputs": outputs,
                            }
                            return
                return
            except websockets.ConnectionClosed as exc:
                if not reconnect:
                    yield "error", {"message": f"WebSocket error: {exc}", "outputs": outputs}
                    return
                attempts += 1
                if attempts > self.settings.ws_reconnect_attempts:
                    break
                log(
                    "Websocket closed unexpectedly: %s. Reconnecting (%s/%s)...",
                    exc,
                    attempts,
                    self.settings.ws_reconnect_attempts,
                    level=logging.DEBUG,
                )
                if not await self.is_reachable():
                    raise ConnectionError("ComfyUI HTTP unreachable during websocket reconnect")
                await asyncio.sleep(self.settings.ws_reconnect_delay_s)
            except Exception as exc:
                if emit_progress:
                    yield "error", {"message": f"WebSocket error: {exc}", "outputs": outputs}
                    return
                raise ConnectionError(f"WebSocket error: {exc}") from exc
        raise ConnectionError(
            f"Failed to reconnect websocket after {self.settings.ws_reconnect_attempts} attempts"
        )

    async def monitor(self, prompt_id: str, client_id: str) -> tuple[bool, list[str], Manifest]:
        async for event_type, payload in self.watch_prompt(
            prompt_id, client_id, emit_progress=False, reconnect=True
        ):
            if event_type == "complete":
                return True, [], payload.get("outputs", {})
            if event_type == "error":
                return (
                    False,
                    [f"Workflow execution error: {payload['message']}"],
                    payload.get("outputs", {}),
                )
        return False, [], {}

    async def resolve_manifest(
        self,
        prompt_id: str,
        websocket_outputs: Manifest,
        errors: list[str],
        request_id: str,
        trace: Trace | None = None,
    ) -> Manifest:
        context = trace.span("output_manifest") if trace is not None else nullcontext()
        with context:
            if not manifest_needs_history(websocket_outputs, errors):
                log(
                    "Using websocket output manifest without history fallback.",
                    request_id=request_id,
                    level=logging.DEBUG,
                )
                return websocket_outputs
            log(
                "Fetching history for prompt %s due to incomplete websocket manifest.",
                prompt_id,
                request_id=request_id,
                level=logging.DEBUG,
            )
            history = await self.history(prompt_id)
            if prompt_id not in history:
                raise KeyError(prompt_id)
            outputs = history.get(prompt_id, {}).get("outputs", {})
            return merge_outputs(websocket_outputs, outputs) if outputs else websocket_outputs

    async def collect_images(
        self, outputs: Manifest, request_id: str, trace: Trace | None = None
    ) -> tuple[list[dict[str, str]], list[str]]:
        errors: list[str] = []
        specs: list[tuple[str, str, str | None]] = []
        for node_id, node_output in outputs.items():
            for image in node_output.get("images", []):
                if image.get("type") == "temp":
                    continue
                filename = image.get("filename")
                if not filename:
                    errors.append(
                        f"Skipping image in node {node_id} due to missing filename: {image}"
                    )
                    continue
                specs.append((filename, image.get("subfolder", ""), image.get("type")))

        async def fetch(spec: tuple[str, str, str | None]) -> tuple[dict[str, str] | None, str | None]:
            filename, subfolder, image_type = spec
            fetch_started_ns = time.perf_counter_ns()
            blob = await self.image_bytes(filename, subfolder, image_type)
            if trace is not None:
                trace.add("image_fetch", time.perf_counter_ns() - fetch_started_ns)
            if not blob:
                return None, f"Failed to fetch image data for {filename} from /view endpoint."
            try:
                encode_started_ns = time.perf_counter_ns()
                data = base64.b64encode(blob).decode("ascii")
                if trace is not None:
                    trace.add("base64_encode", time.perf_counter_ns() - encode_started_ns)
                return {"filename": filename, "type": "base64", "data": data}, None
            except Exception as exc:
                return None, f"Error encoding {filename} to base64: {exc}"

        results = await gather_limited(
            specs, self.settings.output_fetch_concurrency, fetch
        )
        images = [image for image, error in results if image]
        errors.extend(error for image, error in results if not image and error)
        for error in errors:
            log(error, request_id=request_id)
        return images, errors

    async def run(self, request: GenerateRequest, trace: Trace) -> dict[str, Any]:
        if request.images:
            with trace.span("upload"):
                upload_errors = await self.upload_images(request.images, trace.request_id)
            if upload_errors:
                return {
                    "error": "Failed to upload one or more input images",
                    "details": upload_errors,
                }

        client_id = str(uuid.uuid4())
        try:
            with trace.span("queue"):
                prompt_id = await self.queue_workflow(
                    request.workflow, client_id, request.comfy_org_api_key
                )
            log("Queued workflow with ID: %s", prompt_id, request_id=trace.request_id)
        except httpx.RequestError as exc:
            log("Error queuing workflow: %s", exc, request_id=trace.request_id)
            return {"error": f"Error queuing workflow: {exc}"}
        except ValueError as exc:
            log("Workflow validation error: %s", exc, request_id=trace.request_id)
            return {"error": str(exc)}

        try:
            with trace.span("websocket_wait"):
                done, errors, websocket_outputs = await self.monitor(prompt_id, client_id)
        except Exception as exc:
            log_exception("WebSocket error: %s", exc, request_id=trace.request_id)
            return {"error": f"WebSocket communication error: {exc}"}
        if not done and not errors:
            return {
                "error": "Workflow monitoring loop exited without confirmation of completion or error."
            }

        used_history = manifest_needs_history(websocket_outputs, errors)
        try:
            outputs = await self.resolve_manifest(
                prompt_id, websocket_outputs, errors, trace.request_id, trace
            )
            trace.output_source = (
                "websocket"
                if not used_history
                else "websocket+history" if manifest_has_images(websocket_outputs) else "history"
            )
        except KeyError:
            message = f"Prompt ID {prompt_id} not found in history after execution."
            log(message, request_id=trace.request_id)
            if errors:
                errors.append(message)
                return {
                    "error": "Job processing failed, prompt ID not found in history.",
                    "details": errors,
                }
            return {"error": message}
        except Exception as exc:
            return {"error": f"Failed to fetch history: {exc}"}

        if not outputs and not errors:
            errors.append(f"No outputs found in history for prompt {prompt_id}.")
        images, image_errors = await self.collect_images(outputs, trace.request_id, trace)
        if image_errors:
            errors.extend(error for error in image_errors if error not in errors)
        if images:
            result: dict[str, Any] = {"images": images}
        elif errors:
            return {"error": "Job processing failed", "details": errors}
        else:
            result = {"status": "success_no_images", "images": []}
        if errors:
            result["errors"] = errors
        log(
            "Job completed. Returning %s image(s) via %s.",
            len(result.get("images", [])),
            trace.output_source,
            request_id=trace.request_id,
        )
        return result

    async def stream(self, request: GenerateRequest, request_id: str) -> AsyncIterator[str]:
        def sse(event: str, payload: dict[str, Any]) -> str:
            return f"event: {event}\ndata: {dumps(payload).decode('utf-8', errors='replace')}\n\n"

        if request.images:
            yield sse("status", {"message": "Uploading input images..."})
            upload_errors = await self.upload_images(request.images, request_id)
            if upload_errors:
                yield sse("error", {"message": "Failed to upload input images"})
                return

        yield sse("status", {"message": "Queuing workflow..."})
        client_id = str(uuid.uuid4())
        try:
            prompt_id = await self.queue_workflow(
                request.workflow, client_id, request.comfy_org_api_key
            )
        except (httpx.RequestError, ValueError) as exc:
            yield sse("error", {"message": str(exc)})
            return

        log("Queued workflow with ID: %s", prompt_id, request_id=request_id)
        yield sse("status", {"message": "Processing..."})
        websocket_outputs: Manifest = {}
        async for event_type, payload in self.watch_prompt(
            prompt_id, client_id, emit_progress=True, reconnect=False
        ):
            if event_type == "complete":
                websocket_outputs = payload.get("outputs", {})
                break
            if event_type == "error":
                yield sse("error", {"message": payload["message"]})
                return
            yield sse(event_type, payload)

        yield sse("status", {"message": "Fetching results..."})
        try:
            outputs = await self.resolve_manifest(prompt_id, websocket_outputs, [], request_id)
        except KeyError:
            yield sse("error", {"message": "Prompt not found in history"})
            return
        except Exception as exc:
            yield sse("error", {"message": f"Failed to fetch history: {exc}"})
            return

        images, errors = await self.collect_images(outputs, request_id)
        if not images and errors:
            yield sse("error", {"message": errors[0]})
            return
        log("Streaming complete. %s image(s).", len(images), request_id=request_id)
        yield sse("result", {"images": images})

__all__ = [
    "ComfyWorker",
    "GenerateRequest",
    "SETTINGS",
    "Trace",
    "json_response",
    "log",
    "log_exception",
]
