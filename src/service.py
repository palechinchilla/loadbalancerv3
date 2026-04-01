from __future__ import annotations

import base64
import os
import time

from .comfy_client import ComfyClient
from .exceptions import ComfyUnavailableError, ComfyValidationError
from .models import GenerateRequest, GenerateResponse, GeneratedImage
from .state import worker_state


class GenerationService:
    def __init__(self, comfy_client: ComfyClient) -> None:
        self.comfy_client = comfy_client

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        if not await self.comfy_client.wait_until_reachable():
            raise ComfyUnavailableError("ComfyUI server is not reachable.")

        request_id = self.comfy_client.make_request_id()
        started_at = time.perf_counter()

        upload_errors = await self.comfy_client.upload_images(
            [image.model_dump() for image in request.images] if request.images else None
        )
        if upload_errors:
            raise ComfyValidationError(
                "Failed to upload one or more input images:\n" + "\n".join(upload_errors)
            )

        client_id = self.comfy_client.make_request_id()
        queued_workflow = await self.comfy_client.queue_workflow(
            request.workflow,
            client_id,
            request.comfy_org_api_key or os.getenv("COMFY_ORG_API_KEY"),
        )
        prompt_id = queued_workflow.get("prompt_id")
        if not prompt_id:
            raise ComfyValidationError(
                f"Missing 'prompt_id' in queue response: {queued_workflow}"
            )

        worker_state.request_count += 1
        worker_state.last_prompt_id = str(prompt_id)

        errors = await self.comfy_client.wait_for_prompt_completion(str(prompt_id), client_id)
        history = await self.comfy_client.get_history(str(prompt_id))
        prompt_history = history.get(str(prompt_id))
        if not prompt_history:
            raise ComfyValidationError(
                f"Prompt ID {prompt_id} not found in history after execution."
            )

        generated_images, fetch_errors = await self._collect_images(
            prompt_history.get("outputs", {})
        )
        errors.extend(fetch_errors)
        duration_ms = int((time.perf_counter() - started_at) * 1000)

        return GenerateResponse(
            request_id=request_id,
            images=generated_images,
            errors=errors or None,
            duration_ms=duration_ms,
        )

    async def _collect_images(self, outputs: dict) -> tuple[list[GeneratedImage], list[str]]:
        output_data: list[GeneratedImage] = []
        errors: list[str] = []

        for node_output in outputs.values():
            for image_info in node_output.get("images", []):
                filename = image_info.get("filename")
                subfolder = image_info.get("subfolder", "")
                image_type = image_info.get("type")

                if image_type == "temp":
                    continue
                if not filename:
                    errors.append(f"Skipping image due to missing filename: {image_info}")
                    continue

                image_bytes = await self.comfy_client.get_image_data(
                    filename,
                    subfolder,
                    image_type,
                )
                if not image_bytes:
                    errors.append(
                        f"Failed to fetch image data for {filename} from /view endpoint."
                    )
                    continue

                output_data.append(
                    GeneratedImage(
                        filename=filename,
                        type="base64",
                        data=base64.b64encode(image_bytes).decode("utf-8"),
                    )
                )

        return output_data, errors
