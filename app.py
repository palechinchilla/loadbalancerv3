"""
FastAPI load-balancing worker for ComfyUI on RunPod.

Replaces the queue-based handler.py.  Exposes:
  GET  /ping      – health check (204 while booting, 200 when ready)
  POST /generate  – submit a ComfyUI workflow and receive output images
"""

import base64
import binascii
import json
import os
import uuid
import traceback
import logging
import asyncio
import time
from contextlib import asynccontextmanager, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any

import httpx
try:
    import orjson
except ImportError:  # pragma: no cover - runtime image installs orjson
    orjson = None
import websockets
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _SuppressPingAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return '"GET /ping HTTP/1.1"' not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_SuppressPingAccessFilter())

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Time to wait between API check attempts in milliseconds
COMFY_API_AVAILABLE_INTERVAL_MS = int(
    os.environ.get("COMFY_API_AVAILABLE_INTERVAL_MS", 50)
)
# Maximum number of API check attempts (0 = no limit, poll while process alive)
COMFY_API_AVAILABLE_MAX_RETRIES = int(
    os.environ.get("COMFY_API_AVAILABLE_MAX_RETRIES", 0)
)
# Fallback retry limit when PID file is unavailable and retries=0
COMFY_API_FALLBACK_MAX_RETRIES = 500
# PID file written by start.sh so we can detect if ComfyUI has crashed
COMFY_PID_FILE = "/tmp/comfyui.pid"

# WebSocket reconnection behaviour
WEBSOCKET_RECONNECT_ATTEMPTS = int(os.environ.get("WEBSOCKET_RECONNECT_ATTEMPTS", 5))
WEBSOCKET_RECONNECT_DELAY_S = int(os.environ.get("WEBSOCKET_RECONNECT_DELAY_S", 3))

# Host where ComfyUI is running
COMFY_HOST = "127.0.0.1:8188"
COMFY_BASE_URL = f"http://{COMFY_HOST}"

# Per-request processing timeout (RunPod allows 5.5 min max)
PROCESSING_TIMEOUT_S = int(os.environ.get("PROCESSING_TIMEOUT_S", 300))

# Port for the FastAPI server
PORT = int(os.environ.get("PORT", 80))

# Shared HTTP client tuning for low-latency local ComfyUI traffic
COMFY_HTTP_CONNECT_TIMEOUT_S = float(os.environ.get("COMFY_HTTP_CONNECT_TIMEOUT_S", 2.0))
COMFY_HTTP_READ_TIMEOUT_S = float(os.environ.get("COMFY_HTTP_READ_TIMEOUT_S", 30.0))
COMFY_HTTP_WRITE_TIMEOUT_S = float(os.environ.get("COMFY_HTTP_WRITE_TIMEOUT_S", 10.0))
COMFY_HTTP_POOL_TIMEOUT_S = float(os.environ.get("COMFY_HTTP_POOL_TIMEOUT_S", 5.0))
COMFY_HTTP_MAX_CONNECTIONS = max(1, int(os.environ.get("COMFY_HTTP_MAX_CONNECTIONS", 64)))
COMFY_HTTP_MAX_KEEPALIVE_CONNECTIONS = max(
    1, int(os.environ.get("COMFY_HTTP_MAX_KEEPALIVE_CONNECTIONS", 32))
)
COMFY_HTTP_KEEPALIVE_EXPIRY_S = float(
    os.environ.get("COMFY_HTTP_KEEPALIVE_EXPIRY_S", 30.0)
)
COMFY_OUTPUT_FETCH_CONCURRENCY = max(
    1, int(os.environ.get("COMFY_OUTPUT_FETCH_CONCURRENCY", 8))
)
COMFY_HTTP_STATUS_READ_TIMEOUT_S = float(
    os.environ.get("COMFY_HTTP_STATUS_READ_TIMEOUT_S", 5.0)
)
COMFY_HTTP_PROMPT_READ_TIMEOUT_S = float(
    os.environ.get("COMFY_HTTP_PROMPT_READ_TIMEOUT_S", 30.0)
)
COMFY_HTTP_PROMPT_WRITE_TIMEOUT_S = float(
    os.environ.get("COMFY_HTTP_PROMPT_WRITE_TIMEOUT_S", 30.0)
)
COMFY_HTTP_HISTORY_READ_TIMEOUT_S = float(
    os.environ.get("COMFY_HTTP_HISTORY_READ_TIMEOUT_S", 15.0)
)
COMFY_HTTP_OUTPUT_READ_TIMEOUT_S = float(
    os.environ.get("COMFY_HTTP_OUTPUT_READ_TIMEOUT_S", 30.0)
)
COMFY_HTTP_UPLOAD_READ_TIMEOUT_S = float(
    os.environ.get("COMFY_HTTP_UPLOAD_READ_TIMEOUT_S", 30.0)
)
COMFY_HTTP_UPLOAD_WRITE_TIMEOUT_S = float(
    os.environ.get("COMFY_HTTP_UPLOAD_WRITE_TIMEOUT_S", 30.0)
)

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
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            self.stage_ns[stage] += time.perf_counter_ns() - start_ns

    def add_ns(self, stage: str, duration_ns: int) -> None:
        self.stage_ns[stage] += max(0, duration_ns)

    def mark_total(self) -> None:
        self.stage_ns["total"] = time.perf_counter_ns() - self.started_ns


@dataclass
class MonitorResult:
    execution_done: bool
    errors: list[str]
    outputs: dict[str, dict[str, Any]]


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class ImageInput(BaseModel):
    name: str
    image: str


class GenerateRequest(BaseModel):
    workflow: dict
    images: list[ImageInput] | None = None
    comfy_org_api_key: str | None = None


# ---------------------------------------------------------------------------
# ComfyUI helper functions
# ---------------------------------------------------------------------------

def _http_timeout(*, read_s: float | None = None, write_s: float | None = None) -> httpx.Timeout:
    return httpx.Timeout(
        connect=COMFY_HTTP_CONNECT_TIMEOUT_S,
        read=COMFY_HTTP_READ_TIMEOUT_S if read_s is None else read_s,
        write=COMFY_HTTP_WRITE_TIMEOUT_S if write_s is None else write_s,
        pool=COMFY_HTTP_POOL_TIMEOUT_S,
    )


def _build_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=COMFY_BASE_URL,
        trust_env=False,
        http2=False,
        follow_redirects=False,
        limits=httpx.Limits(
            max_connections=COMFY_HTTP_MAX_CONNECTIONS,
            max_keepalive_connections=COMFY_HTTP_MAX_KEEPALIVE_CONNECTIONS,
            keepalive_expiry=COMFY_HTTP_KEEPALIVE_EXPIRY_S,
        ),
        timeout=_http_timeout(),
    )


def _ns_to_ms(duration_ns: int) -> float:
    return round(duration_ns / 1_000_000, 3)


def _serialize_json_bytes(payload: dict[str, Any]) -> bytes:
    if orjson is not None:
        return orjson.dumps(payload)
    return json.dumps(
        payload, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _load_json_bytes(payload: bytes) -> Any:
    if orjson is not None:
        return orjson.loads(payload)
    return json.loads(payload)


def _load_response_json(response: httpx.Response) -> Any:
    return _load_json_bytes(response.content)


def _trace_headers(trace: RequestTrace) -> dict[str, str]:
    postprocess_ns = max(
        trace.stage_ns["total"]
        - trace.stage_ns["receive"]
        - trace.stage_ns["upload"]
        - trace.stage_ns["queue"]
        - trace.stage_ns["websocket_wait"],
        0,
    )
    return {
        "X-Worker-Request-Id": trace.request_id,
        "X-Worker-Total-Ms": f"{_ns_to_ms(trace.stage_ns['total']):.3f}",
        "X-Worker-Queue-Ms": f"{_ns_to_ms(trace.stage_ns['queue']):.3f}",
        "X-Worker-Execute-Ms": f"{_ns_to_ms(trace.stage_ns['websocket_wait']):.3f}",
        "X-Worker-Postprocess-Ms": f"{_ns_to_ms(postprocess_ns):.3f}",
        "X-Worker-Response-Bytes": str(trace.response_bytes),
    }


def _log_request_summary(
    trace: RequestTrace, status_code: int, error: str | None = None
) -> None:
    summary = {
        "request_id": trace.request_id,
        "status_code": status_code,
        "output_source": trace.output_source,
        "response_bytes": trace.response_bytes,
        "stages_ms": {
            stage: _ns_to_ms(trace.stage_ns[stage]) for stage in LATENCY_STAGES
        },
    }
    if error:
        summary["error"] = error
    logger.info("worker-comfyui latency %s", json.dumps(summary, separators=(",", ":")))


async def _comfy_server_status(http_client: httpx.AsyncClient) -> dict[str, Any]:
    """Return a dictionary with basic reachability info for the ComfyUI HTTP server."""
    try:
        resp = await http_client.get(
            "/", timeout=_http_timeout(read_s=COMFY_HTTP_STATUS_READ_TIMEOUT_S)
        )
        return {
            "reachable": resp.status_code == 200,
            "status_code": resp.status_code,
        }
    except httpx.RequestError as exc:
        return {"reachable": False, "error": str(exc)}


def _get_comfyui_pid():
    """Read the ComfyUI process PID from the PID file written by start.sh."""
    try:
        with open(COMFY_PID_FILE, "r") as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def _is_comfyui_process_alive():
    """Check whether the ComfyUI process is still running.

    Returns True if alive, False if dead, None if PID file not found.
    """
    pid = _get_comfyui_pid()
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists but we can't signal it



async def upload_images(http_client: httpx.AsyncClient, images: list[dict[str, str]]) -> dict[str, Any]:
    """
    Upload a list of base64 encoded images to the ComfyUI server using the /upload/image endpoint.
    """
    if not images:
        return {"status": "success", "message": "No images to upload", "details": []}

    responses = []
    upload_errors = []

    print(f"worker-comfyui - Uploading {len(images)} image(s)...")

    for image in images:
        try:
            name = image["name"]
            image_data_uri = image["image"]

            if "," in image_data_uri:
                base64_data = image_data_uri.split(",", 1)[1]
            else:
                base64_data = image_data_uri

            blob = base64.b64decode(base64_data.strip(), validate=True)

            files = {
                "image": (name, blob, "image/png"),
                "overwrite": (None, "true"),
            }

            response = await http_client.post(
                "/upload/image",
                files=files,
                timeout=_http_timeout(
                    read_s=COMFY_HTTP_UPLOAD_READ_TIMEOUT_S,
                    write_s=COMFY_HTTP_UPLOAD_WRITE_TIMEOUT_S,
                ),
            )
            response.raise_for_status()

            responses.append(f"Successfully uploaded {name}")
            print(f"worker-comfyui - Successfully uploaded {name}")

        except (binascii.Error, ValueError) as e:
            error_msg = f"Error decoding base64 for {image.get('name', 'unknown')}: {e}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except httpx.TimeoutException:
            error_msg = f"Timeout uploading {image.get('name', 'unknown')}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except httpx.RequestError as e:
            error_msg = f"Error uploading {image.get('name', 'unknown')}: {e}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except Exception as e:
            error_msg = (
                f"Unexpected error uploading {image.get('name', 'unknown')}: {e}"
            )
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)

    if upload_errors:
        print(f"worker-comfyui - image(s) upload finished with errors")
        return {
            "status": "error",
            "message": "Some images failed to upload",
            "details": upload_errors,
        }

    print(f"worker-comfyui - image(s) upload complete")
    return {
        "status": "success",
        "message": "All images uploaded successfully",
        "details": responses,
    }


async def get_available_models(http_client: httpx.AsyncClient) -> dict[str, Any]:
    """Get list of available models from ComfyUI."""
    try:
        response = await http_client.get("/object_info", timeout=_http_timeout(read_s=10.0))
        response.raise_for_status()
        object_info = _load_response_json(response)

        available_models = {}
        if "CheckpointLoaderSimple" in object_info:
            checkpoint_info = object_info["CheckpointLoaderSimple"]
            if "input" in checkpoint_info and "required" in checkpoint_info["input"]:
                ckpt_options = checkpoint_info["input"]["required"].get("ckpt_name")
                if ckpt_options and len(ckpt_options) > 0:
                    available_models["checkpoints"] = (
                        ckpt_options[0] if isinstance(ckpt_options[0], list) else []
                    )

        return available_models
    except Exception as e:
        print(f"worker-comfyui - Warning: Could not fetch available models: {e}")
        return {}


async def queue_workflow(
    http_client: httpx.AsyncClient,
    workflow: dict[str, Any],
    client_id: str,
    comfy_org_api_key: str | None = None,
) -> dict[str, Any]:
    """
    Queue a workflow to be processed by ComfyUI.

    Raises:
        ValueError: If the workflow validation fails with detailed error information.
    """
    payload = {"prompt": workflow, "client_id": client_id}

    key_from_env = os.environ.get("COMFY_ORG_API_KEY")
    effective_key = comfy_org_api_key if comfy_org_api_key else key_from_env
    if effective_key:
        payload["extra_data"] = {"api_key_comfy_org": effective_key}

    response = await http_client.post(
        "/prompt",
        content=_serialize_json_bytes(payload),
        headers={"Content-Type": "application/json"},
        timeout=_http_timeout(
            read_s=COMFY_HTTP_PROMPT_READ_TIMEOUT_S,
            write_s=COMFY_HTTP_PROMPT_WRITE_TIMEOUT_S,
        ),
    )

    if response.status_code == 400:
        print(f"worker-comfyui - ComfyUI returned 400. Response body: {response.text}")
        try:
            error_data = _load_response_json(response)
            print(f"worker-comfyui - Parsed error data: {error_data}")

            error_message = "Workflow validation failed"
            error_details = []

            if "error" in error_data:
                error_info = error_data["error"]
                if isinstance(error_info, dict):
                    error_message = error_info.get("message", error_message)
                    if error_info.get("type") == "prompt_outputs_failed_validation":
                        error_message = "Workflow validation failed"
                else:
                    error_message = str(error_info)

            if "node_errors" in error_data:
                for node_id, node_error in error_data["node_errors"].items():
                    if isinstance(node_error, dict):
                        for error_type, error_msg in node_error.items():
                            error_details.append(
                                f"Node {node_id} ({error_type}): {error_msg}"
                            )
                    else:
                        error_details.append(f"Node {node_id}: {node_error}")

            if error_data.get("type") == "prompt_outputs_failed_validation":
                error_message = error_data.get("message", "Workflow validation failed")
                available_models = await get_available_models(http_client)
                if available_models.get("checkpoints"):
                    error_message += f"\n\nThis usually means a required model or parameter is not available."
                    error_message += f"\nAvailable checkpoint models: {', '.join(available_models['checkpoints'])}"
                else:
                    error_message += "\n\nThis usually means a required model or parameter is not available."
                    error_message += "\nNo checkpoint models appear to be available. Please check your model installation."
                raise ValueError(error_message)

            if error_details:
                detailed_message = f"{error_message}:\n" + "\n".join(
                    f"• {detail}" for detail in error_details
                )

                if any(
                    "not in list" in detail and "ckpt_name" in detail
                    for detail in error_details
                ):
                    available_models = await get_available_models(http_client)
                    if available_models.get("checkpoints"):
                        detailed_message += f"\n\nAvailable checkpoint models: {', '.join(available_models['checkpoints'])}"
                    else:
                        detailed_message += "\n\nNo checkpoint models appear to be available. Please check your model installation."

                raise ValueError(detailed_message)
            else:
                raise ValueError(f"{error_message}. Raw response: {response.text}")

        except (json.JSONDecodeError, KeyError):
            raise ValueError(
                f"ComfyUI validation failed (could not parse error response): {response.text}"
            )

    response.raise_for_status()
    return _load_response_json(response)


async def get_history(http_client: httpx.AsyncClient, prompt_id: str) -> dict[str, Any]:
    """Retrieve the history of a given prompt using its ID."""
    response = await http_client.get(
        f"/history/{prompt_id}",
        timeout=_http_timeout(read_s=COMFY_HTTP_HISTORY_READ_TIMEOUT_S),
    )
    response.raise_for_status()
    return _load_response_json(response)


async def get_image_data(
    http_client: httpx.AsyncClient,
    filename: str,
    subfolder: str,
    image_type: str | None,
) -> bytes | None:
    """Fetch image bytes from the ComfyUI /view endpoint."""
    data = {"filename": filename, "subfolder": subfolder, "type": image_type}
    try:
        response = await http_client.get(
            "/view",
            params=data,
            timeout=_http_timeout(read_s=COMFY_HTTP_OUTPUT_READ_TIMEOUT_S),
        )
        response.raise_for_status()
        return response.content
    except httpx.TimeoutException:
        print(f"worker-comfyui - Timeout fetching image data for {filename}")
        return None
    except httpx.RequestError as e:
        print(f"worker-comfyui - Error fetching image data for {filename}: {e}")
        return None
    except Exception as e:
        print(
            f"worker-comfyui - Unexpected error fetching image data for {filename}: {e}"
        )
        return None


def _record_ws_output(
    outputs: dict[str, dict[str, Any]], prompt_id: str, data: dict[str, Any]
) -> None:
    if data.get("prompt_id") != prompt_id:
        return

    node_id = data.get("node")
    node_output = data.get("output")
    if node_id is None or not isinstance(node_output, dict):
        return

    normalized_node_id = str(node_id)
    existing = outputs.get(normalized_node_id, {}).copy()
    if "images" in node_output:
        existing["images"] = node_output["images"]
    for key, value in node_output.items():
        if key != "images":
            existing[key] = value
    outputs[normalized_node_id] = existing


def _manifest_has_usable_images(outputs: dict[str, dict[str, Any]]) -> bool:
    for node_output in outputs.values():
        for image_info in node_output.get("images", []):
            if image_info.get("filename") and image_info.get("type") != "temp":
                return True
    return False


def _manifest_is_incomplete(outputs: dict[str, dict[str, Any]]) -> bool:
    for node_output in outputs.values():
        for image_info in node_output.get("images", []):
            if image_info.get("type") == "temp":
                continue
            if not image_info.get("filename") or image_info.get("type") is None:
                return True
    return False


def _should_fetch_history(
    websocket_outputs: dict[str, dict[str, Any]], errors: list[str]
) -> bool:
    return bool(errors) or not _manifest_has_usable_images(websocket_outputs) or _manifest_is_incomplete(websocket_outputs)


def _merge_output_manifests(
    websocket_outputs: dict[str, dict[str, Any]],
    history_outputs: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for node_id, node_output in history_outputs.items():
        if isinstance(node_output, dict):
            merged[str(node_id)] = dict(node_output)

    for node_id, node_output in websocket_outputs.items():
        if not isinstance(node_output, dict):
            continue
        existing = merged.get(str(node_id), {}).copy()
        if "images" in node_output:
            existing["images"] = node_output["images"]
        for key, value in node_output.items():
            if key != "images":
                existing[key] = value
        merged[str(node_id)] = existing

    return merged


def _collect_output_fetch_specs(
    outputs: dict[str, Any], request_id: str, errors: list[str] | None = None
) -> list[tuple[str, str, str, str | None]]:
    specs: list[tuple[str, str, str, str | None]] = []

    for node_id, node_output in outputs.items():
        if "images" in node_output:
            for image_info in node_output["images"]:
                filename = image_info.get("filename")
                subfolder = image_info.get("subfolder", "")
                img_type = image_info.get("type")

                if img_type == "temp":
                    continue

                if not filename:
                    warn_msg = (
                        f"Skipping image in node {node_id} due to missing filename: {image_info}"
                    )
                    print(f"worker-comfyui - [{request_id}] {warn_msg}")
                    if errors is not None:
                        errors.append(warn_msg)
                    continue

                specs.append((node_id, filename, subfolder, img_type))

        other_keys = [k for k in node_output.keys() if k != "images"]
        if other_keys:
            warn_msg = f"Node {node_id} produced unhandled output keys: {other_keys}."
            print(f"worker-comfyui - [{request_id}] WARNING: {warn_msg}")
            print(
                f"worker-comfyui - [{request_id}] --> If this output is useful, please consider opening an issue on GitHub to discuss adding support."
            )

    return specs


async def _collect_output_images(
    http_client: httpx.AsyncClient,
    outputs: dict[str, Any],
    request_id: str,
    trace: RequestTrace | None = None,
    errors: list[str] | None = None,
) -> list[dict[str, str]]:
    specs = _collect_output_fetch_specs(outputs, request_id, errors)
    if not specs:
        return []

    semaphore = asyncio.Semaphore(COMFY_OUTPUT_FETCH_CONCURRENCY)

    async def fetch_one(
        spec: tuple[str, str, str, str | None]
    ) -> tuple[dict[str, str] | None, str | None]:
        _, filename, subfolder, img_type = spec
        async with semaphore:
            fetch_started_ns = time.perf_counter_ns()
            image_bytes = await get_image_data(
                http_client, filename, subfolder, img_type
            )
        if trace is not None:
            trace.add_ns("image_fetch", time.perf_counter_ns() - fetch_started_ns)

        if not image_bytes:
            return None, f"Failed to fetch image data for {filename} from /view endpoint."

        try:
            encode_started_ns = time.perf_counter_ns()
            base64_image = base64.b64encode(image_bytes).decode("ascii")
            if trace is not None:
                trace.add_ns("base64_encode", time.perf_counter_ns() - encode_started_ns)
            return (
                {
                    "filename": filename,
                    "type": "base64",
                    "data": base64_image,
                },
                None,
            )
        except Exception as e:
            return None, f"Error encoding {filename} to base64: {e}"

    results = await asyncio.gather(*(fetch_one(spec) for spec in specs))
    output_data: list[dict[str, str]] = []

    for image_payload, error_msg in results:
        if image_payload is not None:
            output_data.append(image_payload)
        elif errors is not None and error_msg:
            print(f"worker-comfyui - [{request_id}] {error_msg}")
            errors.append(error_msg)

    return output_data


# ---------------------------------------------------------------------------
# Async WebSocket monitoring
# ---------------------------------------------------------------------------

async def _monitor_execution(
    http_client: httpx.AsyncClient, prompt_id: str, client_id: str
) -> MonitorResult:
    """
    Connect to ComfyUI WebSocket and monitor workflow execution.

    Returns:
        MonitorResult with completion state, errors, and websocket output metadata.
    """
    ws_url = f"ws://{COMFY_HOST}/ws?clientId={client_id}"
    errors = []
    outputs: dict[str, dict[str, Any]] = {}
    reconnect_count = 0

    while reconnect_count <= WEBSOCKET_RECONNECT_ATTEMPTS:
        try:
            print(f"worker-comfyui - Connecting to websocket: {ws_url}")
            async with websockets.connect(ws_url, open_timeout=10) as ws:
                print(f"worker-comfyui - Websocket connected")
                async for raw_message in ws:
                    if not isinstance(raw_message, str):
                        continue

                    try:
                        message = json.loads(raw_message)
                    except json.JSONDecodeError:
                        print(f"worker-comfyui - Received invalid JSON message via websocket.")
                        continue

                    msg_type = message.get("type")

                    if msg_type == "executed":
                        data = message.get("data", {})
                        _record_ws_output(outputs, prompt_id, data)

                    elif msg_type == "executing":
                        data = message.get("data", {})
                        if (
                            data.get("node") is None
                            and data.get("prompt_id") == prompt_id
                        ):
                            print(
                                f"worker-comfyui - Execution finished for prompt {prompt_id}"
                            )
                            return MonitorResult(True, errors, outputs)

                    elif msg_type == "execution_error":
                        data = message.get("data", {})
                        if data.get("prompt_id") == prompt_id:
                            error_details = (
                                f"Node Type: {data.get('node_type')}, "
                                f"Node ID: {data.get('node_id')}, "
                                f"Message: {data.get('exception_message')}"
                            )
                            print(
                                f"worker-comfyui - Execution error received: {error_details}"
                            )
                            errors.append(f"Workflow execution error: {error_details}")
                            return MonitorResult(False, errors, outputs)

        except websockets.ConnectionClosed as closed_err:
            reconnect_count += 1
            print(
                f"worker-comfyui - Websocket connection closed unexpectedly: {closed_err}. "
                f"Attempting to reconnect ({reconnect_count}/{WEBSOCKET_RECONNECT_ATTEMPTS})..."
            )

            srv_status = await _comfy_server_status(http_client)
            if not srv_status["reachable"]:
                print(
                    f"worker-comfyui - ComfyUI HTTP unreachable – aborting websocket reconnect: "
                    f"{srv_status.get('error', 'status ' + str(srv_status.get('status_code')))}"
                )
                raise ConnectionError("ComfyUI HTTP unreachable during websocket reconnect")

            print(
                f"worker-comfyui - ComfyUI HTTP reachable (status {srv_status.get('status_code')}), "
                f"waiting {WEBSOCKET_RECONNECT_DELAY_S}s before retry..."
            )
            await asyncio.sleep(WEBSOCKET_RECONNECT_DELAY_S)

        except Exception as e:
            raise ConnectionError(f"WebSocket error: {e}")

    raise ConnectionError(
        f"Failed to reconnect websocket after {WEBSOCKET_RECONNECT_ATTEMPTS} attempts"
    )


async def _monitor_execution_streaming(prompt_id, client_id):
    """
    Async generator version of _monitor_execution.

    Yields (event_type, data_dict) tuples for SSE streaming:
      - ("progress", {"step": int, "total": int})
      - ("status", {"message": str})
      - ("error", {"message": str})
      - ("_complete", {"outputs": dict[str, dict]})

    Returns normally when execution completes.
    """
    ws_url = f"ws://{COMFY_HOST}/ws?clientId={client_id}"
    outputs: dict[str, dict[str, Any]] = {}

    try:
        async with websockets.connect(ws_url, open_timeout=10) as ws:
            async for raw_message in ws:
                if not isinstance(raw_message, str):
                    continue
                try:
                    message = json.loads(raw_message)
                except json.JSONDecodeError:
                    continue

                msg_type = message.get("type")
                data = message.get("data", {})

                if msg_type == "progress":
                    yield ("progress", {
                        "step": data.get("value", 0),
                        "total": data.get("max", 0),
                    })
                elif msg_type == "executed":
                    _record_ws_output(outputs, prompt_id, data)
                elif msg_type == "executing":
                    if (
                        data.get("node") is None
                        and data.get("prompt_id") == prompt_id
                    ):
                        yield ("_complete", {"outputs": outputs})
                        return  # execution finished
                    elif data.get("node"):
                        yield ("status", {
                            "message": f"Executing node {data['node']}",
                        })
                elif msg_type == "execution_error":
                    if data.get("prompt_id") == prompt_id:
                        yield ("error", {
                            "message": data.get("exception_message", "Execution error"),
                        })
                        return
    except Exception as e:
        yield ("error", {"message": f"WebSocket error: {e}"})


async def _resolve_output_manifest(
    http_client: httpx.AsyncClient,
    prompt_id: str,
    websocket_outputs: dict[str, dict[str, Any]],
    request_id: str,
    errors: list[str],
    trace: RequestTrace | None = None,
) -> dict[str, dict[str, Any]]:
    with trace.measure("output_manifest") if trace is not None else nullcontext():
        outputs = websocket_outputs
        history_outputs: dict[str, Any] = {}

        if _should_fetch_history(websocket_outputs, errors):
            print(f"worker-comfyui - [{request_id}] Fetching history for prompt {prompt_id}...")
            history = await get_history(http_client, prompt_id)
            if prompt_id not in history:
                raise KeyError(prompt_id)
            history_outputs = history.get(prompt_id, {}).get("outputs", {})

        if history_outputs:
            outputs = _merge_output_manifests(websocket_outputs, history_outputs)

        return outputs


def _build_json_http_response(
    status_code: int,
    payload: dict[str, Any],
    trace: RequestTrace,
    *,
    error: str | None = None,
) -> Response:
    with trace.measure("response_serialize"):
        body = _serialize_json_bytes(payload)
    trace.response_bytes = len(body)
    trace.mark_total()
    headers = _trace_headers(trace)
    _log_request_summary(trace, status_code, error=error)
    return Response(
        content=body,
        media_type="application/json",
        status_code=status_code,
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Core workflow processing (async)
# ---------------------------------------------------------------------------

async def process_workflow(
    http_client: httpx.AsyncClient,
    request_data: GenerateRequest,
    request_id: str,
    trace: RequestTrace,
) -> dict[str, Any]:
    """Process a ComfyUI workflow from start to finish."""
    workflow = request_data.workflow
    input_images = request_data.images
    comfy_org_api_key = request_data.comfy_org_api_key

    # Upload input images if provided
    if input_images:
        images_dicts = [img.model_dump() for img in input_images]
        with trace.measure("upload"):
            upload_result = await upload_images(http_client, images_dicts)
        if upload_result["status"] == "error":
            return {
                "error": "Failed to upload one or more input images",
                "details": upload_result["details"],
            }

    # Generate a unique client_id for this request's WebSocket session
    client_id = str(uuid.uuid4())

    # Queue the workflow on ComfyUI
    try:
        with trace.measure("queue"):
            queued_workflow = await queue_workflow(
                http_client, workflow, client_id, comfy_org_api_key
            )
        prompt_id = queued_workflow.get("prompt_id")
        if not prompt_id:
            raise ValueError(
                f"Missing 'prompt_id' in queue response: {queued_workflow}"
            )
        print(f"worker-comfyui - [{request_id}] Queued workflow with ID: {prompt_id}")
    except httpx.RequestError as e:
        print(f"worker-comfyui - [{request_id}] Error queuing workflow: {e}")
        return {"error": f"Error queuing workflow: {e}"}
    except ValueError as e:
        print(f"worker-comfyui - [{request_id}] Workflow validation error: {e}")
        return {"error": str(e)}

    # Monitor execution via WebSocket
    try:
        with trace.measure("websocket_wait"):
            monitor_result = await _monitor_execution(http_client, prompt_id, client_id)
    except (ConnectionError, Exception) as e:
        print(f"worker-comfyui - [{request_id}] WebSocket error: {e}")
        print(traceback.format_exc())
        return {"error": f"WebSocket communication error: {e}"}

    execution_done = monitor_result.execution_done
    errors = monitor_result.errors

    if not execution_done and not errors:
        return {
            "error": "Workflow monitoring loop exited without confirmation of completion or error."
        }

    try:
        history_fallback_used = _should_fetch_history(monitor_result.outputs, errors)
        outputs = await _resolve_output_manifest(
            http_client,
            prompt_id,
            monitor_result.outputs,
            request_id,
            errors,
            trace=trace,
        )
        if history_fallback_used:
            trace.output_source = (
                "websocket+history"
                if _manifest_has_usable_images(monitor_result.outputs)
                else "history"
            )
        else:
            trace.output_source = "websocket"
    except KeyError:
        error_msg = f"Prompt ID {prompt_id} not found in history after execution."
        print(f"worker-comfyui - [{request_id}] {error_msg}")
        if not errors:
            return {"error": error_msg}
        errors.append(error_msg)
        return {
            "error": "Job processing failed, prompt ID not found in history.",
            "details": errors,
        }
    except Exception as e:
        return {"error": f"Failed to fetch history: {e}"}

    if not outputs:
        warning_msg = f"No outputs found in history for prompt {prompt_id}."
        print(f"worker-comfyui - [{request_id}] {warning_msg}")
        if not errors:
            errors.append(warning_msg)

    # Process output images
    output_data = await _collect_output_images(
        http_client, outputs, request_id, trace=trace, errors=errors
    )

    # Build final result
    final_result = {}

    if output_data:
        final_result["images"] = output_data

    if errors:
        final_result["errors"] = errors
        print(f"worker-comfyui - [{request_id}] Job completed with errors/warnings: {errors}")

    if not output_data and errors:
        print(f"worker-comfyui - [{request_id}] Job failed with no output images.")
        return {
            "error": "Job processing failed",
            "details": errors,
        }
    elif not output_data and not errors:
        print(
            f"worker-comfyui - [{request_id}] Job completed successfully, but the workflow produced no images."
        )
        final_result["status"] = "success_no_images"
        final_result["images"] = []

    print(
        f"worker-comfyui - [{request_id}] Job completed. Returning {len(output_data)} image(s) via {trace.output_source}."
    )
    return final_result


# ---------------------------------------------------------------------------
# Startup: wait for ComfyUI to become ready
# ---------------------------------------------------------------------------

async def _wait_for_comfyui(app_instance):
    """
    Background task that polls ComfyUI until it responds with HTTP 200.
    Sets app.state.comfyui_ready = True once ready.
    """
    url = f"http://{COMFY_HOST}/"
    delay = max(1, COMFY_API_AVAILABLE_INTERVAL_MS) / 1000  # convert ms to seconds
    log_every = max(1, int(10_000 / max(1, COMFY_API_AVAILABLE_INTERVAL_MS)))
    attempt = 0

    print(f"worker-comfyui - Checking API server at {url}...")
    http_client: httpx.AsyncClient = app_instance.state.http_client

    while True:
        process_status = _is_comfyui_process_alive()
        if process_status is False:
            print(
                "worker-comfyui - ComfyUI process has exited. "
                "Server will not become reachable."
            )
            return  # stay unhealthy

        try:
            resp = await http_client.get(
                "/", timeout=_http_timeout(read_s=COMFY_HTTP_STATUS_READ_TIMEOUT_S)
            )
            if resp.status_code == 200:
                print(f"worker-comfyui - API is reachable")
                app_instance.state.comfyui_ready = True
                return
        except httpx.TimeoutException:
            pass
        except httpx.RequestError:
            pass

        attempt += 1

        fallback = (
            COMFY_API_AVAILABLE_MAX_RETRIES
            if COMFY_API_AVAILABLE_MAX_RETRIES > 0
            else COMFY_API_FALLBACK_MAX_RETRIES
        )
        if process_status is None and attempt >= fallback:
            print(
                f"worker-comfyui - Failed to connect to server at {url} "
                f"after {fallback} attempts (no PID file found)."
            )
            return  # stay unhealthy

        if attempt % log_every == 0:
            elapsed_s = attempt * delay
            print(
                f"worker-comfyui - Still waiting for API server... "
                f"({elapsed_s:.0f}s elapsed, attempt {attempt})"
            )

        await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Manage application startup and shutdown."""
    app_instance.state.comfyui_ready = False
    app_instance.state.http_client = _build_http_client()
    startup_task = asyncio.create_task(_wait_for_comfyui(app_instance))
    try:
        yield
    finally:
        startup_task.cancel()
        try:
            await startup_task
        except asyncio.CancelledError:
            pass
        await app_instance.state.http_client.aclose()


app = FastAPI(title="ComfyUI Load Balancing Worker", lifespan=lifespan)


@app.get("/ping")
async def ping():
    """Health check endpoint required by RunPod load balancer."""
    if not app.state.comfyui_ready:
        return Response(status_code=204)  # 204 = initializing
    return Response(status_code=200, content="OK")


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Submit a ComfyUI workflow and receive output images."""
    request_id = str(uuid.uuid4())
    trace = RequestTrace(request_id)

    if not app.state.comfyui_ready:
        payload = {"error": "ComfyUI is not ready yet"}
        return _build_json_http_response(
            status_code=503,
            payload=payload,
            trace=trace,
            error=payload["error"],
        )

    print(f"worker-comfyui - [{request_id}] Received generate request")
    trace.stage_ns["receive"] = time.perf_counter_ns() - trace.started_ns

    try:
        result = await asyncio.wait_for(
            process_workflow(app.state.http_client, request, request_id, trace),
            timeout=PROCESSING_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        print(f"worker-comfyui - [{request_id}] Processing timed out after {PROCESSING_TIMEOUT_S}s")
        payload = {"error": f"Workflow processing timed out after {PROCESSING_TIMEOUT_S}s"}
        return _build_json_http_response(
            status_code=504,
            payload=payload,
            trace=trace,
            error=payload["error"],
        )
    except Exception as e:
        print(f"worker-comfyui - [{request_id}] Unexpected error: {e}")
        print(traceback.format_exc())
        payload = {"error": f"An unexpected error occurred: {e}"}
        return _build_json_http_response(
            status_code=500,
            payload=payload,
            trace=trace,
            error=payload["error"],
        )

    if "error" in result:
        status_code = 422 if "validation" in result.get("error", "").lower() else 500
        return _build_json_http_response(
            status_code=status_code,
            payload=result,
            trace=trace,
            error=result.get("error"),
        )

    return _build_json_http_response(status_code=200, payload=result, trace=trace)


# ---------------------------------------------------------------------------
# SSE streaming endpoint
# ---------------------------------------------------------------------------

async def _stream_workflow(
    http_client: httpx.AsyncClient, request_data: GenerateRequest, request_id: str
):
    """Async generator that yields SSE-formatted strings with progress events."""

    def sse(event, data):
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    workflow = request_data.workflow
    input_images = request_data.images
    comfy_org_api_key = request_data.comfy_org_api_key

    # Upload input images if provided
    if input_images:
        yield sse("status", {"message": "Uploading input images..."})
        images_dicts = [img.model_dump() for img in input_images]
        upload_result = await upload_images(http_client, images_dicts)
        if upload_result["status"] == "error":
            yield sse("error", {"message": "Failed to upload input images"})
            return

    # Queue the workflow
    yield sse("status", {"message": "Queuing workflow..."})
    client_id = str(uuid.uuid4())

    try:
        queued = await queue_workflow(
            http_client, workflow, client_id, comfy_org_api_key
        )
        prompt_id = queued.get("prompt_id")
        if not prompt_id:
            yield sse("error", {"message": "Missing prompt_id in queue response"})
            return
        print(f"worker-comfyui - [{request_id}] Queued workflow with ID: {prompt_id}")
    except (httpx.RequestError, ValueError) as e:
        yield sse("error", {"message": str(e)})
        return

    # Monitor execution, forwarding progress events
    yield sse("status", {"message": "Processing..."})
    websocket_outputs: dict[str, dict[str, Any]] = {}

    async for event_type, event_data in _monitor_execution_streaming(prompt_id, client_id):
        if event_type == "_complete":
            websocket_outputs = event_data.get("outputs", {})
            continue
        yield sse(event_type, event_data)
        if event_type == "error":
            return

    # Fetch results
    yield sse("status", {"message": "Fetching results..."})

    try:
        outputs = await _resolve_output_manifest(
            http_client,
            prompt_id,
            websocket_outputs,
            request_id,
            [],
        )
    except KeyError:
        yield sse("error", {"message": "Prompt not found in history"})
        return
    except Exception as e:
        yield sse("error", {"message": f"Failed to fetch history: {e}"})
        return
    output_data = await _collect_output_images(http_client, outputs, request_id)

    result = {"images": output_data}
    print(f"worker-comfyui - [{request_id}] Streaming complete. {len(output_data)} image(s).")
    yield sse("result", result)


@app.post("/generate-stream")
async def generate_stream(request: GenerateRequest):
    """Submit a ComfyUI workflow and receive SSE progress events + output images."""
    if not app.state.comfyui_ready:
        return JSONResponse(
            status_code=503,
            content={"error": "ComfyUI is not ready yet"},
        )

    request_id = str(uuid.uuid4())
    print(f"worker-comfyui - [{request_id}] Received streaming generate request")

    return StreamingResponse(
        _stream_workflow(app.state.http_client, request, request_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Entry point (for direct execution / uvicorn)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print(f"worker-comfyui - Starting FastAPI server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
