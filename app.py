"""
FastAPI load-balancing worker for ComfyUI on RunPod.

Replaces the queue-based handler.py.  Exposes:
  GET  /ping      – health check (204 while booting, 200 when ready)
  POST /generate  – submit a ComfyUI workflow and receive output images
"""

import json
import urllib.parse
import os
import requests
import base64
from io import BytesIO
import uuid
import traceback
import logging
import asyncio
from contextlib import asynccontextmanager

import websockets
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Per-request processing timeout (RunPod allows 5.5 min max)
PROCESSING_TIMEOUT_S = int(os.environ.get("PROCESSING_TIMEOUT_S", 300))

# Port for the FastAPI server
PORT = int(os.environ.get("PORT", 80))


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

def _comfy_server_status():
    """Return a dictionary with basic reachability info for the ComfyUI HTTP server."""
    try:
        resp = requests.get(f"http://{COMFY_HOST}/", timeout=5)
        return {
            "reachable": resp.status_code == 200,
            "status_code": resp.status_code,
        }
    except Exception as exc:
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



def upload_images(images):
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

            blob = base64.b64decode(base64_data)

            files = {
                "image": (name, BytesIO(blob), "image/png"),
                "overwrite": (None, "true"),
            }

            response = requests.post(
                f"http://{COMFY_HOST}/upload/image", files=files, timeout=30
            )
            response.raise_for_status()

            responses.append(f"Successfully uploaded {name}")
            print(f"worker-comfyui - Successfully uploaded {name}")

        except base64.binascii.Error as e:
            error_msg = f"Error decoding base64 for {image.get('name', 'unknown')}: {e}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except requests.Timeout:
            error_msg = f"Timeout uploading {image.get('name', 'unknown')}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except requests.RequestException as e:
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


def get_available_models():
    """Get list of available models from ComfyUI."""
    try:
        response = requests.get(f"http://{COMFY_HOST}/object_info", timeout=10)
        response.raise_for_status()
        object_info = response.json()

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


def queue_workflow(workflow, client_id, comfy_org_api_key=None):
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
    data = json.dumps(payload).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    response = requests.post(
        f"http://{COMFY_HOST}/prompt", data=data, headers=headers, timeout=30
    )

    if response.status_code == 400:
        print(f"worker-comfyui - ComfyUI returned 400. Response body: {response.text}")
        try:
            error_data = response.json()
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
                available_models = get_available_models()
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
                    available_models = get_available_models()
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
    return response.json()


def get_history(prompt_id):
    """Retrieve the history of a given prompt using its ID."""
    response = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def get_image_data(filename, subfolder, image_type):
    """Fetch image bytes from the ComfyUI /view endpoint."""
    print(
        f"worker-comfyui - Fetching image data: type={image_type}, subfolder={subfolder}, filename={filename}"
    )
    data = {"filename": filename, "subfolder": subfolder, "type": image_type}
    url_values = urllib.parse.urlencode(data)
    try:
        response = requests.get(f"http://{COMFY_HOST}/view?{url_values}", timeout=60)
        response.raise_for_status()
        print(f"worker-comfyui - Successfully fetched image data for {filename}")
        return response.content
    except requests.Timeout:
        print(f"worker-comfyui - Timeout fetching image data for {filename}")
        return None
    except requests.RequestException as e:
        print(f"worker-comfyui - Error fetching image data for {filename}: {e}")
        return None
    except Exception as e:
        print(
            f"worker-comfyui - Unexpected error fetching image data for {filename}: {e}"
        )
        return None


# ---------------------------------------------------------------------------
# Async WebSocket monitoring
# ---------------------------------------------------------------------------

async def _monitor_execution(prompt_id, client_id):
    """
    Connect to ComfyUI WebSocket and monitor workflow execution.

    Returns:
        tuple: (execution_done: bool, errors: list[str])
    """
    ws_url = f"ws://{COMFY_HOST}/ws?clientId={client_id}"
    errors = []
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

                    if msg_type == "status":
                        status_data = message.get("data", {}).get("status", {})
                        print(
                            f"worker-comfyui - Status update: {status_data.get('exec_info', {}).get('queue_remaining', 'N/A')} items remaining in queue"
                        )

                    elif msg_type == "executing":
                        data = message.get("data", {})
                        if (
                            data.get("node") is None
                            and data.get("prompt_id") == prompt_id
                        ):
                            print(
                                f"worker-comfyui - Execution finished for prompt {prompt_id}"
                            )
                            return True, errors

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
                            return False, errors

        except websockets.ConnectionClosed as closed_err:
            reconnect_count += 1
            print(
                f"worker-comfyui - Websocket connection closed unexpectedly: {closed_err}. "
                f"Attempting to reconnect ({reconnect_count}/{WEBSOCKET_RECONNECT_ATTEMPTS})..."
            )

            srv_status = _comfy_server_status()
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

    Returns normally when execution completes.
    """
    ws_url = f"ws://{COMFY_HOST}/ws?clientId={client_id}"

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
                elif msg_type == "executing":
                    if (
                        data.get("node") is None
                        and data.get("prompt_id") == prompt_id
                    ):
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


# ---------------------------------------------------------------------------
# Core workflow processing (async)
# ---------------------------------------------------------------------------

async def process_workflow(request_data, request_id):
    """Process a ComfyUI workflow from start to finish."""
    workflow = request_data.workflow
    input_images = request_data.images
    comfy_org_api_key = request_data.comfy_org_api_key

    # Upload input images if provided
    if input_images:
        images_dicts = [img.model_dump() for img in input_images]
        upload_result = await asyncio.to_thread(upload_images, images_dicts)
        if upload_result["status"] == "error":
            return {
                "error": "Failed to upload one or more input images",
                "details": upload_result["details"],
            }

    # Generate a unique client_id for this request's WebSocket session
    client_id = str(uuid.uuid4())

    # Queue the workflow on ComfyUI
    try:
        queued_workflow = await asyncio.to_thread(
            queue_workflow, workflow, client_id, comfy_org_api_key
        )
        prompt_id = queued_workflow.get("prompt_id")
        if not prompt_id:
            raise ValueError(
                f"Missing 'prompt_id' in queue response: {queued_workflow}"
            )
        print(f"worker-comfyui - [{request_id}] Queued workflow with ID: {prompt_id}")
    except requests.RequestException as e:
        print(f"worker-comfyui - [{request_id}] Error queuing workflow: {e}")
        return {"error": f"Error queuing workflow: {e}"}
    except ValueError as e:
        print(f"worker-comfyui - [{request_id}] Workflow validation error: {e}")
        return {"error": str(e)}

    # Monitor execution via WebSocket
    try:
        execution_done, errors = await _monitor_execution(prompt_id, client_id)
    except (ConnectionError, Exception) as e:
        print(f"worker-comfyui - [{request_id}] WebSocket error: {e}")
        print(traceback.format_exc())
        return {"error": f"WebSocket communication error: {e}"}

    if not execution_done and not errors:
        return {
            "error": "Workflow monitoring loop exited without confirmation of completion or error."
        }

    # Fetch history (even if there were execution errors, some outputs might exist)
    try:
        print(f"worker-comfyui - [{request_id}] Fetching history for prompt {prompt_id}...")
        history = await asyncio.to_thread(get_history, prompt_id)
    except Exception as e:
        return {"error": f"Failed to fetch history: {e}"}

    if prompt_id not in history:
        error_msg = f"Prompt ID {prompt_id} not found in history after execution."
        print(f"worker-comfyui - [{request_id}] {error_msg}")
        if not errors:
            return {"error": error_msg}
        else:
            errors.append(error_msg)
            return {
                "error": "Job processing failed, prompt ID not found in history.",
                "details": errors,
            }

    prompt_history = history.get(prompt_id, {})
    outputs = prompt_history.get("outputs", {})

    if not outputs:
        warning_msg = f"No outputs found in history for prompt {prompt_id}."
        print(f"worker-comfyui - [{request_id}] {warning_msg}")
        if not errors:
            errors.append(warning_msg)

    # Process output images
    output_data = []
    print(f"worker-comfyui - [{request_id}] Processing {len(outputs)} output nodes...")

    for node_id, node_output in outputs.items():
        if "images" in node_output:
            print(
                f"worker-comfyui - [{request_id}] Node {node_id} contains {len(node_output['images'])} image(s)"
            )
            for image_info in node_output["images"]:
                filename = image_info.get("filename")
                subfolder = image_info.get("subfolder", "")
                img_type = image_info.get("type")

                if img_type == "temp":
                    print(
                        f"worker-comfyui - [{request_id}] Skipping image {filename} because type is 'temp'"
                    )
                    continue

                if not filename:
                    warn_msg = f"Skipping image in node {node_id} due to missing filename: {image_info}"
                    print(f"worker-comfyui - [{request_id}] {warn_msg}")
                    errors.append(warn_msg)
                    continue

                image_bytes = await asyncio.to_thread(
                    get_image_data, filename, subfolder, img_type
                )

                if image_bytes:
                    try:
                        base64_image = base64.b64encode(image_bytes).decode("utf-8")
                        output_data.append(
                            {
                                "filename": filename,
                                "type": "base64",
                                "data": base64_image,
                            }
                        )
                        print(f"worker-comfyui - [{request_id}] Encoded {filename} as base64")
                    except Exception as e:
                        error_msg = f"Error encoding {filename} to base64: {e}"
                        print(f"worker-comfyui - [{request_id}] {error_msg}")
                        errors.append(error_msg)
                else:
                    error_msg = f"Failed to fetch image data for {filename} from /view endpoint."
                    errors.append(error_msg)

        # Check for other output types
        other_keys = [k for k in node_output.keys() if k != "images"]
        if other_keys:
            warn_msg = (
                f"Node {node_id} produced unhandled output keys: {other_keys}."
            )
            print(f"worker-comfyui - [{request_id}] WARNING: {warn_msg}")
            print(
                f"worker-comfyui - [{request_id}] --> If this output is useful, please consider opening an issue on GitHub to discuss adding support."
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

    print(f"worker-comfyui - [{request_id}] Job completed. Returning {len(output_data)} image(s).")
    return final_result


# ---------------------------------------------------------------------------
# Startup: wait for ComfyUI to become ready
# ---------------------------------------------------------------------------

def _build_prewarm_workflow():
    """Build a minimal 64x64, 1-step workflow to force-load all models into VRAM."""
    return {
        "9": {
            "inputs": {"filename_prefix": "prewarm", "images": ["65", 0]},
            "class_type": "SaveImage",
        },
        "62": {
            "inputs": {
                "clip_name": "qwen_3_4b.safetensors",
                "type": "lumina2",
                "device": "default",
            },
            "class_type": "CLIPLoader",
        },
        "63": {
            "inputs": {"vae_name": "ae.safetensors"},
            "class_type": "VAELoader",
        },
        "64": {
            "inputs": {"conditioning": ["67", 0]},
            "class_type": "ConditioningZeroOut",
        },
        "65": {
            "inputs": {"samples": ["70", 0], "vae": ["63", 0]},
            "class_type": "VAEDecode",
        },
        "67": {
            "inputs": {"text": "warmup", "clip": ["62", 0]},
            "class_type": "CLIPTextEncode",
        },
        "68": {
            "inputs": {"width": 64, "height": 64, "batch_size": 1},
            "class_type": "EmptySD3LatentImage",
        },
        "69": {
            "inputs": {"shift": 3, "model": ["74", 0]},
            "class_type": "ModelSamplingAuraFlow",
        },
        "70": {
            "inputs": {
                "seed": 42,
                "steps": 1,
                "cfg": 1,
                "sampler_name": "res_multistep",
                "scheduler": "simple",
                "denoise": 1,
                "model": ["69", 0],
                "positive": ["67", 0],
                "negative": ["64", 0],
                "latent_image": ["68", 0],
            },
            "class_type": "KSampler",
        },
        "74": {
            "inputs": {
                "model_name": "z_image_turbo_bf16.safetensors",
                "weight_dtype": "bf16",
                "compute_dtype": "fp16",
                "patch_cublaslinear": True,
                "sage_attention": "sageattn_qk_int8_pv_fp8_cuda++",
                "enable_fp16_accumulation": True,
            },
            "class_type": "DiffusionModelLoaderKJ",
        },
    }


async def _prewarm_models():
    """Run a tiny dummy workflow to force all models into VRAM."""
    import time

    t0 = time.monotonic()
    print("worker-comfyui - Pre-warming models (64x64, 1 step)...")

    workflow = _build_prewarm_workflow()
    client_id = str(uuid.uuid4())

    try:
        queued = await asyncio.to_thread(queue_workflow, workflow, client_id)
        prompt_id = queued.get("prompt_id")
        if not prompt_id:
            print("worker-comfyui - Pre-warm: missing prompt_id, skipping")
            return False

        execution_done, errors = await _monitor_execution(prompt_id, client_id)

        elapsed = time.monotonic() - t0
        if execution_done:
            print(f"worker-comfyui - Pre-warm complete in {elapsed:.1f}s — models are hot")
            return True
        else:
            print(f"worker-comfyui - Pre-warm finished with errors after {elapsed:.1f}s: {errors}")
            return False

    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"worker-comfyui - Pre-warm failed after {elapsed:.1f}s: {e}")
        return False


# Whether to pre-warm models at startup (disable with PREWARM_MODELS=0)
PREWARM_MODELS = os.environ.get("PREWARM_MODELS", "1") != "0"


async def _wait_for_comfyui(app_instance):
    """
    Background task that polls ComfyUI until it responds with HTTP 200,
    optionally pre-warms models, then sets app.state.comfyui_ready = True.
    """
    url = f"http://{COMFY_HOST}/"
    delay = max(1, COMFY_API_AVAILABLE_INTERVAL_MS) / 1000  # convert ms to seconds
    log_every = max(1, int(10_000 / max(1, COMFY_API_AVAILABLE_INTERVAL_MS)))
    attempt = 0

    print(f"worker-comfyui - Checking API server at {url}...")

    while True:
        process_status = _is_comfyui_process_alive()
        if process_status is False:
            print(
                "worker-comfyui - ComfyUI process has exited. "
                "Server will not become reachable."
            )
            return  # stay unhealthy

        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                print(f"worker-comfyui - API is reachable")
                break
        except requests.Timeout:
            pass
        except requests.RequestException:
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

    # Pre-warm models so first real request hits hot VRAM
    if PREWARM_MODELS:
        await _prewarm_models()
    else:
        print("worker-comfyui - Model pre-warming disabled (PREWARM_MODELS=0)")

    app_instance.state.comfyui_ready = True


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Manage application startup and shutdown."""
    app_instance.state.comfyui_ready = False
    startup_task = asyncio.create_task(_wait_for_comfyui(app_instance))
    yield
    startup_task.cancel()
    try:
        await startup_task
    except asyncio.CancelledError:
        pass


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
    if not app.state.comfyui_ready:
        return JSONResponse(
            status_code=503,
            content={"error": "ComfyUI is not ready yet"},
        )

    request_id = str(uuid.uuid4())
    print(f"worker-comfyui - [{request_id}] Received generate request")

    try:
        result = await asyncio.wait_for(
            process_workflow(request, request_id),
            timeout=PROCESSING_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        print(f"worker-comfyui - [{request_id}] Processing timed out after {PROCESSING_TIMEOUT_S}s")
        return JSONResponse(
            status_code=504,
            content={"error": f"Workflow processing timed out after {PROCESSING_TIMEOUT_S}s"},
        )
    except Exception as e:
        print(f"worker-comfyui - [{request_id}] Unexpected error: {e}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"An unexpected error occurred: {e}"},
        )

    if "error" in result:
        status_code = 422 if "validation" in result.get("error", "").lower() else 500
        return JSONResponse(status_code=status_code, content=result)

    return JSONResponse(status_code=200, content=result)


# ---------------------------------------------------------------------------
# SSE streaming endpoint
# ---------------------------------------------------------------------------

async def _stream_workflow(request_data, request_id):
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
        upload_result = await asyncio.to_thread(upload_images, images_dicts)
        if upload_result["status"] == "error":
            yield sse("error", {"message": "Failed to upload input images"})
            return

    # Queue the workflow
    yield sse("status", {"message": "Queuing workflow..."})
    client_id = str(uuid.uuid4())

    try:
        queued = await asyncio.to_thread(
            queue_workflow, workflow, client_id, comfy_org_api_key
        )
        prompt_id = queued.get("prompt_id")
        if not prompt_id:
            yield sse("error", {"message": "Missing prompt_id in queue response"})
            return
        print(f"worker-comfyui - [{request_id}] Queued workflow with ID: {prompt_id}")
    except (requests.RequestException, ValueError) as e:
        yield sse("error", {"message": str(e)})
        return

    # Monitor execution, forwarding progress events
    yield sse("status", {"message": "Processing..."})

    async for event_type, event_data in _monitor_execution_streaming(prompt_id, client_id):
        yield sse(event_type, event_data)
        if event_type == "error":
            return

    # Fetch results
    yield sse("status", {"message": "Fetching results..."})

    try:
        history = await asyncio.to_thread(get_history, prompt_id)
    except Exception as e:
        yield sse("error", {"message": f"Failed to fetch history: {e}"})
        return

    if prompt_id not in history:
        yield sse("error", {"message": "Prompt not found in history"})
        return

    outputs = history[prompt_id].get("outputs", {})
    output_data = []

    for node_id, node_output in outputs.items():
        if "images" not in node_output:
            continue
        for image_info in node_output["images"]:
            filename = image_info.get("filename")
            subfolder = image_info.get("subfolder", "")
            img_type = image_info.get("type")

            if img_type == "temp" or not filename:
                continue

            image_bytes = await asyncio.to_thread(
                get_image_data, filename, subfolder, img_type
            )
            if not image_bytes:
                continue

            b64 = base64.b64encode(image_bytes).decode("utf-8")
            output_data.append(
                {"filename": filename, "type": "base64", "data": b64}
            )

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
        _stream_workflow(request, request_id),
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
