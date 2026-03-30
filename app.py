import asyncio
import base64
import binascii
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
import websockets
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

try:
    import orjson
except ImportError:  # pragma: no cover - runtime image installs orjson
    orjson = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")
logging.getLogger("httpx").setLevel(logging.WARNING)


class _SuppressPingAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return '"GET /ping HTTP/1.1"' not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_SuppressPingAccessFilter())


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    comfy_host: str = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
    comfy_pid_file: str = os.environ.get("COMFY_PID_FILE", "/tmp/comfyui.pid")
    processing_timeout_s: int = _env_int("PROCESSING_TIMEOUT_S", 300)
    port: int = _env_int("PORT", 80)
    api_interval_ms: int = _env_int("COMFY_API_AVAILABLE_INTERVAL_MS", 50)
    api_max_retries: int = _env_int("COMFY_API_AVAILABLE_MAX_RETRIES", 0)
    api_fallback_max_retries: int = 500
    ws_reconnect_attempts: int = _env_int("WEBSOCKET_RECONNECT_ATTEMPTS", 5)
    ws_reconnect_delay_s: int = _env_int("WEBSOCKET_RECONNECT_DELAY_S", 3)
    connect_timeout_s: float = _env_float("COMFY_HTTP_CONNECT_TIMEOUT_S", 2.0)
    read_timeout_s: float = _env_float("COMFY_HTTP_READ_TIMEOUT_S", 30.0)
    write_timeout_s: float = _env_float("COMFY_HTTP_WRITE_TIMEOUT_S", 10.0)
    pool_timeout_s: float = _env_float("COMFY_HTTP_POOL_TIMEOUT_S", 5.0)
    max_connections: int = max(1, _env_int("COMFY_HTTP_MAX_CONNECTIONS", 64))
    max_keepalive_connections: int = max(
        1, _env_int("COMFY_HTTP_MAX_KEEPALIVE_CONNECTIONS", 32)
    )
    keepalive_expiry_s: float = _env_float("COMFY_HTTP_KEEPALIVE_EXPIRY_S", 30.0)
    status_read_timeout_s: float = _env_float("COMFY_HTTP_STATUS_READ_TIMEOUT_S", 5.0)
    prompt_read_timeout_s: float = _env_float("COMFY_HTTP_PROMPT_READ_TIMEOUT_S", 30.0)
    prompt_write_timeout_s: float = _env_float(
        "COMFY_HTTP_PROMPT_WRITE_TIMEOUT_S", 30.0
    )
    history_read_timeout_s: float = _env_float(
        "COMFY_HTTP_HISTORY_READ_TIMEOUT_S", 15.0
    )
    output_read_timeout_s: float = _env_float("COMFY_HTTP_OUTPUT_READ_TIMEOUT_S", 30.0)
    upload_read_timeout_s: float = _env_float("COMFY_HTTP_UPLOAD_READ_TIMEOUT_S", 30.0)
    upload_write_timeout_s: float = _env_float(
        "COMFY_HTTP_UPLOAD_WRITE_TIMEOUT_S", 30.0
    )
    output_fetch_concurrency: int = max(1, _env_int("COMFY_OUTPUT_FETCH_CONCURRENCY", 8))

    @property
    def base_url(self) -> str:
        return f"http://{self.comfy_host}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.comfy_host}/ws"


SETTINGS = Settings()
LATENCY_STAGES = (
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
POSTPROCESS_STAGES = (
    "output_manifest",
    "image_fetch",
    "base64_encode",
    "response_serialize",
)
ImageManifest = dict[str, dict[str, Any]]


class ImageInput(BaseModel):
    name: str
    image: str


class GenerateRequest(BaseModel):
    workflow: dict[str, Any]
    images: list[ImageInput] | None = None
    comfy_org_api_key: str | None = None


@dataclass
class RequestTrace:
    request_id: str
    started_ns: int = field(default_factory=time.perf_counter_ns)
    stage_ns: dict[str, int] = field(
        default_factory=lambda: {stage: 0 for stage in LATENCY_STAGES}
    )
    response_bytes: int = 0
    output_source: str = "unresolved"

    @contextmanager
    def measure(self, stage: str):
        started_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            self.stage_ns[stage] += time.perf_counter_ns() - started_ns

    def add_ns(self, stage: str, duration_ns: int) -> None:
        self.stage_ns[stage] += max(0, duration_ns)

    def finish(self) -> None:
        self.stage_ns["total"] = time.perf_counter_ns() - self.started_ns

    def headers(self) -> dict[str, str]:
        postprocess_ns = sum(self.stage_ns[stage] for stage in POSTPROCESS_STAGES)
        return {
            "X-Worker-Request-Id": self.request_id,
            "X-Worker-Total-Ms": f"{_ns_to_ms(self.stage_ns['total']):.3f}",
            "X-Worker-Queue-Ms": f"{_ns_to_ms(self.stage_ns['queue']):.3f}",
            "X-Worker-Execute-Ms": f"{_ns_to_ms(self.stage_ns['websocket_wait']):.3f}",
            "X-Worker-Postprocess-Ms": f"{_ns_to_ms(postprocess_ns):.3f}",
            "X-Worker-Response-Bytes": str(self.response_bytes),
        }

    def summary(self, status_code: int, error: str | None = None) -> dict[str, Any]:
        payload = {
            "request_id": self.request_id,
            "status_code": status_code,
            "output_source": self.output_source,
            "response_bytes": self.response_bytes,
            "stages_ms": {
                stage: _ns_to_ms(self.stage_ns[stage]) for stage in LATENCY_STAGES
            },
        }
        if error:
            payload["error"] = error
        return payload


@dataclass
class ExecutionResult:
    done: bool
    errors: list[str] = field(default_factory=list)
    outputs: ImageManifest = field(default_factory=dict)


def _ns_to_ms(duration_ns: int) -> float:
    return round(duration_ns / 1_000_000, 3)


def _json_dumps(payload: Any) -> bytes:
    if orjson is not None:
        return orjson.dumps(payload)
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _json_loads(payload: str | bytes) -> Any:
    if orjson is not None:
        return orjson.loads(payload)
    return json.loads(payload)


def _response_json(
    payload: dict[str, Any],
    status_code: int,
    trace: RequestTrace | None = None,
    *,
    error: str | None = None,
) -> Response:
    if trace is None:
        return Response(
            content=_json_dumps(payload),
            media_type="application/json",
            status_code=status_code,
        )
    with trace.measure("response_serialize"):
        body = _json_dumps(payload)
    trace.response_bytes = len(body)
    trace.finish()
    logger.info(
        "worker-comfyui latency %s",
        _json_dumps(trace.summary(status_code, error)).decode("utf-8", errors="replace"),
    )
    return Response(
        content=body,
        media_type="application/json",
        status_code=status_code,
        headers=trace.headers(),
    )


def _pid_is_alive(pid_file: str) -> bool | None:
    try:
        with open(pid_file, "r") as handle:
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


def _record_ws_output(outputs: ImageManifest, prompt_id: str, data: dict[str, Any]) -> None:
    if data.get("prompt_id") != prompt_id or not isinstance(data.get("output"), dict):
        return
    node_id = data.get("node")
    if node_id is None:
        return
    existing = outputs.get(str(node_id), {}).copy()
    output = data["output"]
    if "images" in output:
        existing["images"] = output["images"]
    existing.update({k: v for k, v in output.items() if k != "images"})
    outputs[str(node_id)] = existing


def _manifest_has_images(outputs: ImageManifest) -> bool:
    return any(
        image.get("filename") and image.get("type") != "temp"
        for node in outputs.values()
        for image in node.get("images", [])
    )


def _manifest_needs_history(outputs: ImageManifest, errors: list[str]) -> bool:
    if errors or not _manifest_has_images(outputs):
        return True
    return any(
        image.get("type") != "temp"
        and (not image.get("filename") or image.get("type") is None)
        for node in outputs.values()
        for image in node.get("images", [])
    )


def _merge_outputs(primary: ImageManifest, fallback: dict[str, Any]) -> ImageManifest:
    merged = {
        str(node_id): dict(node_output)
        for node_id, node_output in fallback.items()
        if isinstance(node_output, dict)
    }
    for node_id, node_output in primary.items():
        merged.setdefault(str(node_id), {}).update(
            {k: v for k, v in node_output.items() if k != "images"}
        )
        if "images" in node_output:
            merged[str(node_id)]["images"] = node_output["images"]
    return merged

class ComfyWorker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ready = False
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
            timeout=self.timeout(),
        )

    def timeout(
        self, *, read_s: float | None = None, write_s: float | None = None
    ) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.settings.connect_timeout_s,
            read=self.settings.read_timeout_s if read_s is None else read_s,
            write=self.settings.write_timeout_s if write_s is None else write_s,
            pool=self.settings.pool_timeout_s,
        )

    @staticmethod
    def _json(response: httpx.Response) -> Any:
        return _json_loads(response.content)

    @staticmethod
    def _log(message: str, *args: Any) -> None:
        logger.info("worker-comfyui - " + message, *args)

    @staticmethod
    def _request_log(request_id: str, message: str, *args: Any) -> None:
        logger.info("worker-comfyui - [%s] " + message, request_id, *args)

    async def close(self) -> None:
        await self.http.aclose()

    async def server_status(self) -> dict[str, Any]:
        try:
            response = await self.http.get(
                "/", timeout=self.timeout(read_s=self.settings.status_read_timeout_s)
            )
            return {
                "reachable": response.status_code == 200,
                "status_code": response.status_code,
            }
        except httpx.RequestError as exc:
            return {"reachable": False, "error": str(exc)}

    async def wait_until_ready(self) -> None:
        delay_s = max(1, self.settings.api_interval_ms) / 1000
        log_every = max(1, int(10_000 / max(1, self.settings.api_interval_ms)))
        url = f"{self.settings.base_url}/"
        attempt = 0
        self._log("Checking API server at %s...", url)
        while True:
            if _pid_is_alive(self.settings.comfy_pid_file) is False:
                self._log("ComfyUI process has exited. Server will not become reachable.")
                return
            try:
                response = await self.http.get(
                    "/", timeout=self.timeout(read_s=self.settings.status_read_timeout_s)
                )
                if response.status_code == 200:
                    self.ready = True
                    self._log("API is reachable")
                    return
            except (httpx.TimeoutException, httpx.RequestError):
                pass
            attempt += 1
            fallback = (
                self.settings.api_max_retries
                if self.settings.api_max_retries > 0
                else self.settings.api_fallback_max_retries
            )
            if _pid_is_alive(self.settings.comfy_pid_file) is None and attempt >= fallback:
                self._log(
                    "Failed to connect to server at %s after %s attempts (no PID file found).",
                    url,
                    fallback,
                )
                return
            if attempt % log_every == 0:
                self._log(
                    "Still waiting for API server... (%ss elapsed, attempt %s)",
                    f"{attempt * delay_s:.0f}",
                    attempt,
                )
            await asyncio.sleep(delay_s)

    async def _available_checkpoints(self) -> list[str]:
        try:
            response = await self.http.get("/object_info", timeout=self.timeout(read_s=10.0))
            response.raise_for_status()
            required = (
                self._json(response)
                .get("CheckpointLoaderSimple", {})
                .get("input", {})
                .get("required", {})
            )
            values = required.get("ckpt_name")
            return values[0] if values and isinstance(values[0], list) else []
        except Exception as exc:
            self._log("Warning: Could not fetch available models: %s", exc)
            return []

    async def _validation_error(self, response: httpx.Response) -> str:
        raw = response.text
        try:
            data = self._json(response)
        except Exception:
            return f"ComfyUI validation failed (could not parse error response): {raw}"
        error_value = data.get("error")
        if isinstance(error_value, dict):
            message = error_value.get("message") or "Workflow validation failed"
            if error_value.get("type") == "prompt_outputs_failed_validation":
                message = "Workflow validation failed"
        else:
            message = str(error_value or data.get("message") or "Workflow validation failed")
        node_errors = []
        for node_id, node_error in (data.get("node_errors") or {}).items():
            if isinstance(node_error, dict):
                node_errors.extend(f"Node {node_id} ({k}): {v}" for k, v in node_error.items())
            else:
                node_errors.append(f"Node {node_id}: {node_error}")
        if data.get("type") == "prompt_outputs_failed_validation":
            message = data.get("message") or "Workflow validation failed"
        if node_errors:
            message = f"{message}:\n" + "\n".join(f"• {detail}" for detail in node_errors)
        elif raw and raw not in message:
            message = f"{message}. Raw response: {raw}"
        needs_models = data.get("type") == "prompt_outputs_failed_validation" or any(
            "not in list" in detail and "ckpt_name" in detail for detail in node_errors
        )
        if not needs_models:
            return message
        checkpoints = await self._available_checkpoints()
        hint = "\n\nThis usually means a required model or parameter is not available."
        if checkpoints:
            hint += f"\nAvailable checkpoint models: {', '.join(checkpoints)}"
        else:
            hint += "\nNo checkpoint models appear to be available. Please check your model installation."
        return message + hint

    async def upload_images(self, images: list[ImageInput]) -> list[str]:
        errors: list[str] = []
        if not images:
            return errors
        self._log("Uploading %s image(s)...", len(images))
        for image in images:
            try:
                raw = image.image.split(",", 1)[1] if "," in image.image else image.image
                blob = base64.b64decode(raw.strip(), validate=True)
                response = await self.http.post(
                    "/upload/image",
                    files={"image": (image.name, blob, "image/png"), "overwrite": (None, "true")},
                    timeout=self.timeout(
                        read_s=self.settings.upload_read_timeout_s,
                        write_s=self.settings.upload_write_timeout_s,
                    ),
                )
                response.raise_for_status()
            except (binascii.Error, ValueError) as exc:
                errors.append(f"Error decoding base64 for {image.name}: {exc}")
            except httpx.TimeoutException:
                errors.append(f"Timeout uploading {image.name}")
            except httpx.RequestError as exc:
                errors.append(f"Error uploading {image.name}: {exc}")
            except Exception as exc:
                errors.append(f"Unexpected error uploading {image.name}: {exc}")
        for error in errors:
            self._log(error)
        if not errors:
            self._log("image(s) upload complete")
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
            content=_json_dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=self.timeout(
                read_s=self.settings.prompt_read_timeout_s,
                write_s=self.settings.prompt_write_timeout_s,
            ),
        )
        if response.status_code == 400:
            raise ValueError(await self._validation_error(response))
        response.raise_for_status()
        prompt_id = self._json(response).get("prompt_id")
        if not prompt_id:
            raise ValueError(f"Missing 'prompt_id' in queue response: {response.text}")
        return prompt_id

    async def history(self, prompt_id: str) -> dict[str, Any]:
        response = await self.http.get(
            f"/history/{prompt_id}",
            timeout=self.timeout(read_s=self.settings.history_read_timeout_s),
        )
        response.raise_for_status()
        return self._json(response)

    async def image_bytes(
        self, filename: str, subfolder: str, image_type: str | None
    ) -> bytes | None:
        try:
            response = await self.http.get(
                "/view",
                params={"filename": filename, "subfolder": subfolder, "type": image_type},
                timeout=self.timeout(read_s=self.settings.output_read_timeout_s),
            )
            response.raise_for_status()
            return response.content
        except httpx.TimeoutException:
            self._log("Timeout fetching image data for %s", filename)
        except httpx.RequestError as exc:
            self._log("Error fetching image data for %s: %s", filename, exc)
        except Exception as exc:
            self._log("Unexpected error fetching image data for %s: %s", filename, exc)
        return None

    async def watch_prompt(
        self,
        prompt_id: str,
        client_id: str,
        *,
        emit_progress: bool,
        reconnect: bool,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        outputs: ImageManifest = {}
        attempts = 0
        while attempts <= self.settings.ws_reconnect_attempts:
            try:
                ws_url = f"{self.settings.ws_url}?clientId={client_id}"
                self._log("Connecting to websocket: %s", ws_url)
                async with websockets.connect(ws_url, open_timeout=10) as ws:
                    self._log("Websocket connected")
                    async for raw_message in ws:
                        if not isinstance(raw_message, str):
                            continue
                        try:
                            message = _json_loads(raw_message)
                        except Exception:
                            continue
                        kind = message.get("type")
                        data = message.get("data") or {}
                        if kind == "progress" and emit_progress:
                            yield "progress", {
                                "step": data.get("value", 0),
                                "total": data.get("max", 0),
                            }
                        elif kind == "executed":
                            _record_ws_output(outputs, prompt_id, data)
                        elif kind == "executing":
                            if data.get("node") is None and data.get("prompt_id") == prompt_id:
                                self._log("Execution finished for prompt %s", prompt_id)
                                yield "complete", {"outputs": outputs}
                                return
                            if emit_progress and data.get("node"):
                                yield "status", {"message": f"Executing node {data['node']}"}
                        elif kind == "execution_error" and data.get("prompt_id") == prompt_id:
                            details = (
                                f"Node Type: {data.get('node_type')}, "
                                f"Node ID: {data.get('node_id')}, "
                                f"Message: {data.get('exception_message')}"
                            )
                            yield "error", {"message": details, "outputs": outputs}
                            return
            except websockets.ConnectionClosed as exc:
                if not reconnect:
                    yield "error", {"message": f"WebSocket error: {exc}", "outputs": outputs}
                    return
                attempts += 1
                self._log(
                    "Websocket connection closed unexpectedly: %s. Attempting to reconnect (%s/%s)...",
                    exc,
                    attempts,
                    self.settings.ws_reconnect_attempts,
                )
                status = await self.server_status()
                if not status["reachable"]:
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

    async def monitor_execution(self, prompt_id: str, client_id: str) -> ExecutionResult:
        async for event_type, payload in self.watch_prompt(
            prompt_id, client_id, emit_progress=False, reconnect=True
        ):
            if event_type == "complete":
                return ExecutionResult(True, outputs=payload.get("outputs", {}))
            if event_type == "error":
                return ExecutionResult(
                    False,
                    [f"Workflow execution error: {payload['message']}"],
                    payload.get("outputs", {}),
                )
        return ExecutionResult(False)

    async def resolve_manifest(
        self,
        prompt_id: str,
        websocket_outputs: ImageManifest,
        request_id: str,
        errors: list[str],
        trace: RequestTrace | None = None,
    ) -> ImageManifest:
        with trace.measure("output_manifest") if trace is not None else nullcontext():
            if not _manifest_needs_history(websocket_outputs, errors):
                return websocket_outputs
            self._request_log(request_id, "Fetching history for prompt %s...", prompt_id)
            history = await self.history(prompt_id)
            if prompt_id not in history:
                raise KeyError(prompt_id)
            outputs = history.get(prompt_id, {}).get("outputs", {})
            return _merge_outputs(websocket_outputs, outputs) if outputs else websocket_outputs

    async def collect_images(
        self,
        outputs: dict[str, Any],
        request_id: str,
        trace: RequestTrace | None = None,
        errors: list[str] | None = None,
    ) -> list[dict[str, str]]:
        specs: list[tuple[str, str, str, str | None]] = []
        for node_id, node_output in outputs.items():
            for image in node_output.get("images", []):
                if image.get("type") == "temp":
                    continue
                if not image.get("filename"):
                    message = f"Skipping image in node {node_id} due to missing filename: {image}"
                    self._request_log(request_id, message)
                    if errors is not None:
                        errors.append(message)
                    continue
                specs.append(
                    (
                        image["filename"],
                        image.get("subfolder", ""),
                        image.get("type"),
                        str(node_id),
                    )
                )
            other_keys = [key for key in node_output if key != "images"]
            if other_keys:
                self._request_log(
                    request_id,
                    "WARNING: Node %s produced unhandled output keys: %s.",
                    node_id,
                    other_keys,
                )
        if not specs:
            return []
        semaphore = asyncio.Semaphore(self.settings.output_fetch_concurrency)

        async def fetch(
            spec: tuple[str, str, str | None, str]
        ) -> tuple[dict[str, str] | None, str | None]:
            filename, subfolder, image_type, _ = spec
            async with semaphore:
                fetch_started_ns = time.perf_counter_ns()
                image_bytes = await self.image_bytes(filename, subfolder, image_type)
            if trace is not None:
                trace.add_ns("image_fetch", time.perf_counter_ns() - fetch_started_ns)
            if not image_bytes:
                return None, f"Failed to fetch image data for {filename} from /view endpoint."
            try:
                encode_started_ns = time.perf_counter_ns()
                data = base64.b64encode(image_bytes).decode("ascii")
                if trace is not None:
                    trace.add_ns("base64_encode", time.perf_counter_ns() - encode_started_ns)
                return {"filename": filename, "type": "base64", "data": data}, None
            except Exception as exc:
                return None, f"Error encoding {filename} to base64: {exc}"

        results = await asyncio.gather(*(fetch(spec) for spec in specs))
        images: list[dict[str, str]] = []
        for image, error in results:
            if image:
                images.append(image)
            elif errors is not None and error:
                self._request_log(request_id, error)
                errors.append(error)
        return images

    async def process(
        self, request: GenerateRequest, request_id: str, trace: RequestTrace
    ) -> dict[str, Any]:
        if request.images:
            with trace.measure("upload"):
                upload_errors = await self.upload_images(request.images)
            if upload_errors:
                return {
                    "error": "Failed to upload one or more input images",
                    "details": upload_errors,
                }
        client_id = str(uuid.uuid4())
        try:
            with trace.measure("queue"):
                prompt_id = await self.queue_workflow(
                    request.workflow, client_id, request.comfy_org_api_key
                )
            self._request_log(request_id, "Queued workflow with ID: %s", prompt_id)
        except httpx.RequestError as exc:
            self._request_log(request_id, "Error queuing workflow: %s", exc)
            return {"error": f"Error queuing workflow: {exc}"}
        except ValueError as exc:
            self._request_log(request_id, "Workflow validation error: %s", exc)
            return {"error": str(exc)}
        try:
            with trace.measure("websocket_wait"):
                execution = await self.monitor_execution(prompt_id, client_id)
        except Exception as exc:
            logger.exception("worker-comfyui - [%s] WebSocket error: %s", request_id, exc)
            return {"error": f"WebSocket communication error: {exc}"}
        if not execution.done and not execution.errors:
            return {
                "error": "Workflow monitoring loop exited without confirmation of completion or error."
            }
        try:
            needs_history = _manifest_needs_history(execution.outputs, execution.errors)
            outputs = await self.resolve_manifest(
                prompt_id, execution.outputs, request_id, execution.errors, trace
            )
            trace.output_source = (
                "websocket"
                if not needs_history
                else "websocket+history" if _manifest_has_images(execution.outputs) else "history"
            )
        except KeyError:
            message = f"Prompt ID {prompt_id} not found in history after execution."
            self._request_log(request_id, message)
            if not execution.errors:
                return {"error": message}
            execution.errors.append(message)
            return {
                "error": "Job processing failed, prompt ID not found in history.",
                "details": execution.errors,
            }
        except Exception as exc:
            return {"error": f"Failed to fetch history: {exc}"}
        if not outputs:
            message = f"No outputs found in history for prompt {prompt_id}."
            self._request_log(request_id, message)
            if not execution.errors:
                execution.errors.append(message)
        images = await self.collect_images(outputs, request_id, trace, execution.errors)
        if images:
            result: dict[str, Any] = {"images": images}
        elif execution.errors:
            self._request_log(request_id, "Job failed with no output images.")
            return {"error": "Job processing failed", "details": execution.errors}
        else:
            self._request_log(
                request_id,
                "Job completed successfully, but the workflow produced no images.",
            )
            result = {"status": "success_no_images", "images": []}
        if execution.errors:
            result["errors"] = execution.errors
            self._request_log(
                request_id, "Job completed with errors/warnings: %s", execution.errors
            )
        self._request_log(
            request_id,
            "Job completed. Returning %s image(s) via %s.",
            len(result["images"]),
            trace.output_source,
        )
        return result

    async def stream(self, request: GenerateRequest, request_id: str) -> AsyncIterator[str]:
        def sse(event: str, payload: dict[str, Any]) -> str:
            data = _json_dumps(payload).decode("utf-8", errors="replace")
            return f"event: {event}\ndata: {data}\n\n"

        if request.images:
            yield sse("status", {"message": "Uploading input images..."})
            errors = await self.upload_images(request.images)
            if errors:
                yield sse("error", {"message": "Failed to upload input images"})
                return
        yield sse("status", {"message": "Queuing workflow..."})
        client_id = str(uuid.uuid4())
        try:
            prompt_id = await self.queue_workflow(
                request.workflow, client_id, request.comfy_org_api_key
            )
            self._request_log(request_id, "Queued workflow with ID: %s", prompt_id)
        except (httpx.RequestError, ValueError) as exc:
            yield sse("error", {"message": str(exc)})
            return
        yield sse("status", {"message": "Processing..."})
        websocket_outputs: ImageManifest = {}
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
            outputs = await self.resolve_manifest(
                prompt_id, websocket_outputs, request_id, [], None
            )
        except KeyError:
            yield sse("error", {"message": "Prompt not found in history"})
            return
        except Exception as exc:
            yield sse("error", {"message": f"Failed to fetch history: {exc}"})
            return
        images = await self.collect_images(outputs, request_id)
        self._request_log(request_id, "Streaming complete. %s image(s).", len(images))
        yield sse("result", {"images": images})


worker = ComfyWorker(SETTINGS)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    startup_task = asyncio.create_task(worker.wait_until_ready())
    app_instance.state.worker = worker
    try:
        yield
    finally:
        startup_task.cancel()
        try:
            await startup_task
        except asyncio.CancelledError:
            pass
        await worker.close()


app = FastAPI(title="ComfyUI Load Balancing Worker", lifespan=lifespan)


@app.get("/ping")
async def ping():
    return Response(status_code=200, content="OK") if worker.ready else Response(status_code=204)


@app.post("/generate")
async def generate(request: GenerateRequest):
    trace = RequestTrace(str(uuid.uuid4()))
    if not worker.ready:
        payload = {"error": "ComfyUI is not ready yet"}
        return _response_json(payload, 503, trace, error=payload["error"])
    ComfyWorker._request_log(trace.request_id, "Received generate request")
    trace.stage_ns["receive"] = time.perf_counter_ns() - trace.started_ns
    try:
        result = await asyncio.wait_for(
            worker.process(request, trace.request_id, trace),
            timeout=SETTINGS.processing_timeout_s,
        )
    except asyncio.TimeoutError:
        payload = {"error": f"Workflow processing timed out after {SETTINGS.processing_timeout_s}s"}
        ComfyWorker._request_log(
            trace.request_id, "Processing timed out after %ss", SETTINGS.processing_timeout_s
        )
        return _response_json(payload, 504, trace, error=payload["error"])
    except Exception as exc:
        logger.exception("worker-comfyui - [%s] Unexpected error: %s", trace.request_id, exc)
        payload = {"error": f"An unexpected error occurred: {exc}"}
        return _response_json(payload, 500, trace, error=payload["error"])
    if "error" in result:
        status_code = 422 if "validation" in result.get("error", "").lower() else 500
        return _response_json(result, status_code, trace, error=result.get("error"))
    return _response_json(result, 200, trace)


@app.post("/generate-stream")
async def generate_stream(request: GenerateRequest):
    if not worker.ready:
        return _response_json({"error": "ComfyUI is not ready yet"}, 503)
    request_id = str(uuid.uuid4())
    ComfyWorker._request_log(request_id, "Received streaming generate request")
    return StreamingResponse(
        worker.stream(request, request_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("worker-comfyui - Starting FastAPI server on port %s", SETTINGS.port)
    uvicorn.run(app, host="0.0.0.0", port=SETTINGS.port)
