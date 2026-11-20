# SPDX-License-Identifier: Apache-2.0
"""Turn a client's chat messages into the canonical messages the canvas reads."""

from __future__ import annotations

import math
from collections.abc import Mapping

from PIL import Image, ImageOps

from sglang_omni.models.lumina.settings import (
    IMAGE_EDGE_FACTOR,
    MAX_IMAGE_EDGE,
    ChatMessage,
    EngineInputs,
    ImagePart,
    LuminaGenerationParams,
    TextPart,
    image_part_source,
)
from sglang_omni.pipeline.umm import UMMRequestError
from sglang_omni.preprocessing.image import ensure_image_list_async
from sglang_omni.proto import StagePayload

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
# An understanding image spans 4 to 16384 tokens here; the processor then applies the model's own pixel budget.
NATIVE_MIN_PIXELS = 4 * IMAGE_EDGE_FACTOR**2
NATIVE_MAX_PIXELS = 16384 * IMAGE_EDGE_FACTOR**2
NATIVE_MAX_ASPECT_RATIO = 200


def smart_resize_shape(height: int, width: int, *, long_edge: int) -> tuple[int, int]:
    """Fix the long edge and round the short edge to the nearest aligned size."""
    factor = IMAGE_EDGE_FACTOR
    scale = long_edge / max(height, width)
    if height >= width:
        short_edge = max(factor, math.floor(width * scale / factor + 0.5) * factor)
        return long_edge, min(short_edge, long_edge)
    else:
        short_edge = max(factor, math.floor(height * scale / factor + 0.5) * factor)
        return min(short_edge, long_edge), long_edge


def native_resize_shape(height: int, width: int) -> tuple[int, int]:
    """Round both edges to the factor; an area outside the native range is scaled back into it."""
    factor = IMAGE_EDGE_FACTOR
    if max(height, width) / min(height, width) > NATIVE_MAX_ASPECT_RATIO:
        raise UMMRequestError(
            f"Lumina reads images up to an aspect ratio of {NATIVE_MAX_ASPECT_RATIO}, got {height}x{width}"
        )
    else:
        pass
    aligned_height = max(factor, round(height / factor) * factor)
    aligned_width = max(factor, round(width / factor) * factor)
    if aligned_height * aligned_width > NATIVE_MAX_PIXELS:
        shrink = math.sqrt(height * width / NATIVE_MAX_PIXELS)
        return (
            math.floor(height / shrink / factor) * factor,
            math.floor(width / shrink / factor) * factor,
        )
    elif aligned_height * aligned_width < NATIVE_MIN_PIXELS:
        growth = math.sqrt(NATIVE_MIN_PIXELS / (height * width))
        return (
            math.ceil(height * growth / factor) * factor,
            math.ceil(width * growth / factor) * factor,
        )
    else:
        return aligned_height, aligned_width


def native_image(image: Image.Image) -> ImagePart:
    """An image to understand keeps its own size, up to alignment and the native range."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    height, width = native_resize_shape(image.height, image.width)
    image = image.resize((width, height), resample=Image.Resampling.BICUBIC)
    return {
        "type": "image",
        "image": image,
        "resized_height": height,
        "resized_width": width,
    }


def canonical_image(image: Image.Image, *, role: str, long_edge: int) -> ImagePart:
    """User images are resized; generated history keeps its aligned size so it stays exact, up to the largest edge Lumina draws."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    is_aligned = not (
        image.height % IMAGE_EDGE_FACTOR or image.width % IMAGE_EDGE_FACTOR
    )
    is_generated = role == "assistant" and is_aligned
    if is_generated and max(image.height, image.width) <= MAX_IMAGE_EDGE:
        height, width = image.height, image.width
    else:
        height, width = smart_resize_shape(
            image.height,
            image.width,
            long_edge=MAX_IMAGE_EDGE if is_generated else long_edge,
        )
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
    return {
        "type": "image",
        "image": image,
        "resized_height": height,
        "resized_width": width,
    }


async def load_image(part: Mapping[str, object]) -> Image.Image:
    """The image of a chat part, from an inline image, a path, or a URL."""
    source = image_part_source(part)
    images = await ensure_image_list_async(source)
    image = images[0]
    if not isinstance(image, Image.Image):
        raise ValueError(f"Lumina could not load an image from {type(source).__name__}")
    else:
        return image


async def canonical_content(
    content: str | list[str | Mapping[str, object]],
    *,
    role: str,
    params: LuminaGenerationParams,
) -> list[TextPart | ImagePart]:
    """Parts of a turn whose shape the controller already validated."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    else:
        pass
    parts: list[TextPart | ImagePart] = []
    for part in content:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
        elif part["type"] == "text":
            text = part["text"]
            assert isinstance(text, str)
            parts.append({"type": "text", "text": text})
        else:
            image = await load_image(part)
            parts.append(
                native_image(image)
                if params.task == "mmu"
                else canonical_image(image, role=role, long_edge=params.input_long_edge)
            )
    return parts


async def preprocess(payload: StagePayload) -> StagePayload:
    """Load and size every image here, so no download ever blocks the GPU stage."""
    inputs = payload.request.inputs
    messages = inputs["messages"] if isinstance(inputs, dict) else inputs
    params = LuminaGenerationParams.model_validate(
        payload.request.metadata["lumina_generation"]
    )
    # The output route replaces the system body, so a client's system text changes nothing.
    canonical: list[ChatMessage] = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    ]
    for message in messages:
        role = message["role"]
        if role != "system":
            content = await canonical_content(
                message["content"], role=role, params=params
            )
            canonical.append({"role": role, "content": content})
        else:
            pass
    if params.task in ("t2i", "i2i"):
        # The trained T2I and I2I turn, whatever order the client sent: the source image, then the prompt as one text.
        parts = canonical[-1]["content"]
        prompt = "".join(part["text"] for part in parts if part["type"] == "text")
        canonical[-1]["content"] = [
            *(part for part in parts if part["type"] == "image"),
            {"type": "text", "text": prompt},
        ]
    else:
        pass
    inputs_for_engine: EngineInputs = {
        "messages": canonical,
        "generation": params.model_dump(),
    }
    payload.data = inputs_for_engine
    return payload
