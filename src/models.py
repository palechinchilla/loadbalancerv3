from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InputImage(BaseModel):
    name: str = Field(min_length=1)
    image: str = Field(min_length=1)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow: dict[str, Any]
    images: list[InputImage] | None = None
    comfy_org_api_key: str | None = None

    @field_validator("workflow", mode="before")
    @classmethod
    def validate_workflow_payload(cls, value: Any) -> Any:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSON format in workflow") from exc
            if not isinstance(parsed, dict):
                raise ValueError("workflow must decode to an object")
            return parsed
        if not isinstance(value, dict):
            raise ValueError("workflow must be an object")
        return value


class GeneratedImage(BaseModel):
    filename: str
    type: Literal["base64"]
    data: str


class GenerateResponse(BaseModel):
    request_id: str
    images: list[GeneratedImage] = Field(default_factory=list)
    errors: list[str] | None = None
    duration_ms: int | None = None


class HealthResponse(BaseModel):
    status: str
    comfyui_reachable: bool
    comfyui_process_alive: bool | None
