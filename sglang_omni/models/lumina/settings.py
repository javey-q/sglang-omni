# SPDX-License-Identifier: Apache-2.0
"""Client controls and wire types shared by the controller, preprocessing, and engine."""

from __future__ import annotations

from typing import Literal, TypedDict

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from sglang_omni.pipeline.umm import UMMFinishReason

IMAGE_EDGE_FACTOR = 32


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImagePart(TypedDict):
    type: Literal["image"]
    image: Image.Image
    resized_height: int
    resized_width: int


class ChatMessage(TypedDict):
    """A canonical turn: system content stays a string, other turns hold ordered parts."""

    role: Literal["system", "user", "assistant"]
    content: str | list[TextPart | ImagePart]


class EngineInputs(TypedDict):
    messages: list[ChatMessage]
    generation: dict[str, int | float | None]


class TextSegment(TypedDict):
    kind: Literal["text"]
    text: str


class ImageSegment(TypedDict):
    kind: Literal["image"]
    mime_type: str
    size_bytes: int
    sha256: str
    url: str
    height: int
    width: int


class EventResult(TypedDict):
    """One engine dispatch; finish_reason stays None while the turn continues."""

    output: TextSegment | ImageSegment | None
    finish_reason: UMMFinishReason | None


class LuminaGenerationParams(BaseModel):
    """stage_params.orchestrator of a Lumina request."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    height: int = Field(default=256, ge=32, le=1024)
    width: int = Field(default=256, ge=32, le=1024)
    input_long_edge: int = Field(default=256, ge=32, le=1024)
    max_events: int = Field(default=8, ge=1)
    max_image_events: int = Field(default=4, ge=0)
    max_new_tokens: int = Field(default=256, ge=1)
    text_denoising_steps: int = Field(default=4, ge=1)
    image_steps: int = Field(default=8, ge=1)
    g_text: float = Field(default=4.0, ge=0.0)
    g_image: float = Field(default=1.0, ge=0.0)
    image_temperature: float = Field(default=1.0, gt=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    seed: int | None = Field(default=None, ge=0)

    @field_validator("height", "width", "input_long_edge")
    @classmethod
    def edge_aligned(cls, edge: int) -> int:
        if edge % IMAGE_EDGE_FACTOR:
            raise ValueError(f"must be a multiple of {IMAGE_EDGE_FACTOR}")
        else:
            return edge
