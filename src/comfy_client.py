from __future__ import annotations

import asyncio
import base64
import json
import logging
import socket
import time
from io import BytesIO
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

import httpx
import websocket

from .config import settings
from .exceptions import ComfyUnavailableError, ComfyValidationError
from .state import is_comfyui_process_alive

logger = logging.getLogger(__name__)


class ComfyClient:
    def __init__(self) -> None:
        timeout = httpx.Timeout(
            settings.http_timeout_s,
            connect=settings.http_connect_timeout_s,
            read=settings.http_timeout_s,
            write=settings.http_timeout_s,
            pool=settings.http_timeout_s,
        )
        self._client = httpx.AsyncClient(
            base_url=f"http://{settings.comfy_host}",
            timeout=timeout,
            http2=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def server_reachable(self) -> bool:
        try:
            response = await self._client.get("/")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def wait_until_reachable(self) -> bool:
        delay_s = max(0.001, settings.comfy_api_available_interval_ms / 1000)
        retries = settings.comfy_api_available_max_retries
        max_attempts = (
            retries
            if retries > 0
            else settings.comfy_api_fallback_max_retries
            if is_comfyui_process_alive() is None
            else None
        )

        attempt = 0
        while True:
            if is_comfyui_process_alive() is False:
                return False
            if await self.server_reachable():
                return True
            attempt += 1
            if max_attempts is not None and attempt >= max_attempts:
                return False
            await asyncio.sleep(delay_s)

    async def upload_images(self, images: list[dict[str, str]] | None) -> list[str]:
        if not images:
            return []

        upload_errors: list[str] = []
        for image in images:
            name = image["name"]
            data_uri = image["image"]
            base64_data = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
            try:
                blob = base64.b64decode(base64_data)
            except Exception as exc:
                upload_errors.append(f"Error decoding base64 for {name}: {exc}")
                continue

            files = {
                "image": (name, BytesIO(blob), "image/png"),
                "overwrite": (None, "true"),
            }
            try:
                response = await self._client.post("/upload/image", files=files)
                response.raise_for_status()
            except httpx.TimeoutException:
                upload_errors.append(f"Timeout uploading {name}")
            except httpx.HTTPError as exc:
                upload_errors.append(f"Error uploading {name}: {exc}")

        return upload_errors

    async def get_available_models(self) -> dict[str, list[str]]:
        try:
            response = await self._client.get("/object_info")
            response.raise_for_status()
            object_info = response.json()
        except Exception as exc:
            logger.warning("Could not fetch available models: %s", exc)
            return {}

        available_models: dict[str, list[str]] = {}
        loader_info = object_info.get("CheckpointLoaderSimple", {})
        required = loader_info.get("input", {}).get("required", {})
        ckpt_options = required.get("ckpt_name")
        if ckpt_options and len(ckpt_options) > 0:
            available_models["checkpoints"] = (
                ckpt_options[0] if isinstance(ckpt_options[0], list) else []
            )
        return available_models

    async def queue_workflow(
        self, workflow: dict[str, Any], client_id: str, comfy_org_api_key: str | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"prompt": workflow, "client_id": client_id}
        if comfy_org_api_key:
            payload["extra_data"] = {"api_key_comfy_org": comfy_org_api_key}

        response = await self._client.post("/prompt", json=payload)
        if response.status_code == 400:
            raise await self._build_validation_error(response)
        response.raise_for_status()
        return response.json()

    async def _build_validation_error(
        self, response: httpx.Response
    ) -> ComfyValidationError:
        try:
            error_data = response.json()
        except json.JSONDecodeError:
            return ComfyValidationError(
                f"ComfyUI validation failed: {response.text or response.status_code}"
            )

        error_message = "Workflow validation failed"
        details: list[str] = []

        error_info = error_data.get("error")
        if isinstance(error_info, dict):
            error_message = error_info.get("message", error_message)
            if error_info.get("type") == "prompt_outputs_failed_validation":
                error_message = "Workflow validation failed"
        elif error_info:
            error_message = str(error_info)

        node_errors = error_data.get("node_errors", {})
        for node_id, node_error in node_errors.items():
            if isinstance(node_error, dict):
                for error_type, error_msg in node_error.items():
                    details.append(f"Node {node_id} ({error_type}): {error_msg}")
            else:
                details.append(f"Node {node_id}: {node_error}")

        if error_data.get("type") == "prompt_outputs_failed_validation":
            available = await self.get_available_models()
            if available.get("checkpoints"):
                error_message += (
                    "\n\nThis usually means a required model or parameter is not available."
                    f"\nAvailable checkpoint models: {', '.join(available['checkpoints'])}"
                )
            else:
                error_message += (
                    "\n\nThis usually means a required model or parameter is not available."
                    "\nNo checkpoint models appear to be available. Please check your model installation."
                )
            return ComfyValidationError(error_message)

        if details:
            return ComfyValidationError(
                f"{error_message}:\n" + "\n".join(f"- {detail}" for detail in details)
            )
        return ComfyValidationError(f"{error_message}. Raw response: {response.text}")

    async def get_history(self, prompt_id: str) -> dict[str, Any]:
        response = await self._client.get(f"/history/{prompt_id}")
        response.raise_for_status()
        return response.json()

    async def get_image_data(
        self, filename: str, subfolder: str, image_type: str
    ) -> bytes | None:
        params = urlencode(
            {"filename": filename, "subfolder": subfolder, "type": image_type}
        )
        try:
            response = await self._client.get(f"/view?{params}")
            response.raise_for_status()
            return response.content
        except httpx.HTTPError as exc:
            logger.error("Failed to fetch image %s: %s", filename, exc)
            return None

    async def wait_for_prompt_completion(self, prompt_id: str, client_id: str) -> list[str]:
        errors: list[str] = []
        ws_url = f"ws://{settings.comfy_host}/ws?clientId={client_id}"
        await asyncio.to_thread(self._listen_for_completion, ws_url, prompt_id, errors)
        return errors

    def _listen_for_completion(
        self, ws_url: str, prompt_id: str, errors: list[str]
    ) -> None:
        ws: websocket.WebSocket | None = None
        try:
            ws = websocket.WebSocket()
            ws.settimeout(10)
            ws.connect(ws_url, timeout=10)
            while True:
                try:
                    out = ws.recv()
                    if not isinstance(out, str):
                        continue
                    message = json.loads(out)
                    if message.get("type") == "executing":
                        data = message.get("data", {})
                        if data.get("node") is None and data.get("prompt_id") == prompt_id:
                            return
                    if message.get("type") == "execution_error":
                        data = message.get("data", {})
                        if data.get("prompt_id") == prompt_id:
                            errors.append(
                                "Workflow execution error: "
                                f"Node Type: {data.get('node_type')}, "
                                f"Node ID: {data.get('node_id')}, "
                                f"Message: {data.get('exception_message')}"
                            )
                            return
                except websocket.WebSocketTimeoutException:
                    if is_comfyui_process_alive() is False:
                        raise ComfyUnavailableError("ComfyUI process exited during execution")
                    continue
                except websocket.WebSocketConnectionClosedException as exc:
                    ws = self._reconnect_websocket(ws_url, exc)
        except websocket.WebSocketException as exc:
            raise ComfyUnavailableError(f"WebSocket communication error: {exc}") from exc
        finally:
            if ws and ws.connected:
                ws.close()

    def _reconnect_websocket(
        self, ws_url: str, initial_error: Exception
    ) -> websocket.WebSocket:
        logger.warning("Websocket closed unexpectedly: %s", initial_error)
        last_error = initial_error
        for _ in range(settings.websocket_reconnect_attempts):
            if is_comfyui_process_alive() is False:
                raise ComfyUnavailableError(
                    "ComfyUI process exited during websocket reconnect"
                )
            try:
                new_ws = websocket.WebSocket()
                new_ws.settimeout(10)
                new_ws.connect(ws_url, timeout=10)
                return new_ws
            except (
                websocket.WebSocketException,
                ConnectionRefusedError,
                socket.timeout,
                OSError,
            ) as exc:
                last_error = exc
                time.sleep(settings.websocket_reconnect_delay_s)
        raise ComfyUnavailableError(
            f"Connection closed and failed to reconnect. Last error: {last_error}"
        )

    @staticmethod
    def make_request_id() -> str:
        return str(uuid4())
