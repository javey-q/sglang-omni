# SPDX-License-Identifier: Apache-2.0
"""Turn a client's chat messages into the canonical messages the canvas reads."""

from __future__ import annotations

import math
from collections.abc import Mapping

from PIL import Image, ImageOps

from sglang_omni.models.lumina.settings import (
    IMAGE_EDGE_FACTOR,
    ChatMessage,
    EngineInputs,
    ImagePart,
    LuminaGenerationParams,
    TextPart,
)
from sglang_omni.preprocessing.image import ensure_image_list_async
from sglang_omni.proto import StagePayload

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


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


def canonical_image(image: Image.Image, *, role: str, long_edge: int) -> ImagePart:
    """User images are resized; generated history keeps its aligned size so it stays exact."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    is_aligned = not (
        image.height % IMAGE_EDGE_FACTOR or image.width % IMAGE_EDGE_FACTOR
    )
    if role == "assistant" and is_aligned:
        height, width = image.height, image.width
    else:
        height, width = smart_resize_shape(
            image.height, image.width, long_edge=long_edge
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
    source = part["image"] if part["type"] == "image" else part["image_url"]
    if isinstance(source, Mapping):
        source = source["url"]
    else:
        pass
    if not isinstance(source, (str, Image.Image)):
        raise ValueError("Lumina image parts need an image, a path, or a URL")
    else:
        pass
    images = await ensure_image_list_async(source)
    image = images[0]
    if not isinstance(image, Image.Image):
        raise ValueError(f"Lumina could not load an image from {type(source).__name__}")
    else:
        return image


async def canonical_content(
    content: object, *, role: str, long_edge: int
) -> list[TextPart | ImagePart]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    elif not isinstance(content, list):
        raise ValueError(f"Lumina {role} content must be text or a list of parts")
    else:
        pass
    parts: list[TextPart | ImagePart] = []
    for part in content:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
        elif not isinstance(part, dict):
            raise ValueError("Lumina content parts must be text or image parts")
        elif part.get("type") == "text":
            parts.append({"type": "text", "text": part["text"]})
        elif part.get("type") in ("image", "image_url"):
            image = await load_image(part)
            parts.append(canonical_image(image, role=role, long_edge=long_edge))
        else:
            raise ValueError(f"Lumina does not accept {part.get('type')!r} parts")
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
                message["content"], role=role, long_edge=params.input_long_edge
            )
            canonical.append({"role": role, "content": content})
        else:
            pass
    inputs_for_engine: EngineInputs = {
        "messages": canonical,
        "generation": params.model_dump(),
    }
    payload.data = inputs_for_engine
    return payload
