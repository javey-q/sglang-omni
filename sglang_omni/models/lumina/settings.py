# SPDX-License-Identifier: Apache-2.0
"""Client controls and wire types shared by the controller, preprocessing, and engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypedDict

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sglang_omni.pipeline.umm import UMMFinishReason

IMAGE_EDGE_FACTOR = 32
# The largest edge Lumina draws, so also the largest a generated history image keeps.
MAX_IMAGE_EDGE = 1024
# A running event cannot be stopped on the serial engine, so these bound how long one request holds it.
MAX_IMAGE_STEPS = 256
MAX_NEW_TOKENS = 4096

# interleaved answers in text and images; t2i and i2i draw exactly one image; mmu answers in one text.
LuminaTask = Literal["interleaved", "t2i", "i2i", "mmu"]
MaskSchedule = Literal["linear", "cosine", "power"]
# threshold_floor reveals every code above the threshold, and at least the scheduled count.
ConfidencePolicy = Literal["threshold_floor", "scheduled_topk"]
# The trained T2I and I2I protocols: greedy codes, T2I with text CFG only and many steps.
IMAGE_TASK_DEFAULTS: dict[str, dict[str, int | float]] = {
    "t2i": {
        "image_steps": 64,
        "g_text": 4.0,
        "g_image": 4.0,
        "image_temperature": 0.0,
        "top_p": 1.0,
    },
    "i2i": {
        "image_steps": 8,
        "g_text": 4.0,
        "g_image": 1.0,
        "image_temperature": 0.0,
        "top_p": 1.0,
    },
}


def image_part_source(part: Mapping[str, object]) -> object:
    """The inline image, path, or URL of a client's image or image_url part, still unchecked."""
    source = part.get("image") if part.get("type") == "image" else part.get("image_url")
    return source.get("url") if isinstance(source, Mapping) else source


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
    generation: dict[str, int | float | str | None]


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

    task: LuminaTask = "interleaved"
    height: int = Field(default=256, ge=32, le=MAX_IMAGE_EDGE)
    width: int = Field(default=256, ge=32, le=MAX_IMAGE_EDGE)
    input_long_edge: int = Field(default=256, ge=32, le=MAX_IMAGE_EDGE)
    max_events: int = Field(default=8, ge=1)
    max_image_events: int = Field(default=4, ge=0)
    max_new_tokens: int = Field(default=256, ge=1, le=MAX_NEW_TOKENS)
    # An mmu answer may open with this token, so the model reasons before it answers.
    assistant_prefill: Literal["<think>"] | None = None
    text_denoising_steps: int = Field(default=4, ge=1)
    image_steps: int = Field(default=8, ge=1, le=MAX_IMAGE_STEPS)
    g_text: float = Field(default=4.0, ge=0.0)
    g_image: float = Field(default=1.0, ge=0.0)
    # Zero takes the most likely code.
    image_temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    mask_schedule: MaskSchedule = "cosine"
    mask_schedule_power: float = Field(default=2.0, gt=0.0)
    confidence_policy: ConfidencePolicy = "threshold_floor"
    seed: int | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def task_defaults(cls, data: object) -> object:
        """Fields a T2I or I2I client leaves out take that task's protocol values."""
        # A task that is not a string is left for field validation, which rejects it.
        if (
            isinstance(data, dict)
            and isinstance(data.get("task"), str)
            and data["task"] in IMAGE_TASK_DEFAULTS
        ):
            return {**IMAGE_TASK_DEFAULTS[data["task"]], **data}
        else:
            return data

    @model_validator(mode="after")
    def prefill_opens_an_understanding_answer(self) -> LuminaGenerationParams:
        """Only an mmu answer has a fixed opening a prefill can take."""
        if self.assistant_prefill is not None and self.task != "mmu":
            raise ValueError("assistant_prefill needs task mmu")
        else:
            return self

    @field_validator("height", "width", "input_long_edge")
    @classmethod
    def edge_aligned(cls, edge: int) -> int:
        if edge % IMAGE_EDGE_FACTOR:
            raise ValueError(f"must be a multiple of {IMAGE_EDGE_FACTOR}")
        else:
            return edge
