# SPDX-License-Identifier: Apache-2.0
"""Lumina on the shared UMM controller: prepare once, then one dispatch per event."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from PIL import Image
from pydantic import ValidationError

from sglang_omni.models.lumina.settings import (
    IMAGE_TASK_DEFAULTS,
    LuminaGenerationParams,
    LuminaTask,
    image_part_source,
)
from sglang_omni.pipeline.umm import (
    CloseReason,
    UMMContext,
    UMMDispatch,
    UMMOutput,
    UMMRequestError,
    UMMStep,
)
from sglang_omni.proto import OmniRequest

PREPARE = "prepare"
EVENT = "event"
# Request params, set only when the client sent them, that mean the same Lumina control: the seed, and max_tokens or max_completion_tokens.
TOP_LEVEL_CONTROLS = ("seed", "max_new_tokens")


def validate_content(role: str, content: object) -> None:
    """Text, or a list of text and image parts, in the shapes the preprocessing stage reads."""
    if isinstance(content, str):
        return
    elif not isinstance(content, list):
        raise UMMRequestError(f"Lumina {role} content must be text or a list of parts")
    else:
        pass
    for part in content:
        if isinstance(part, str):
            pass
        elif not isinstance(part, dict):
            raise UMMRequestError("Lumina content parts must be text or image parts")
        elif part.get("type") == "text":
            if not isinstance(part.get("text"), str):
                raise UMMRequestError("A Lumina text part needs its text")
            else:
                pass
        elif part.get("type") in ("image", "image_url"):
            if not isinstance(image_part_source(part), (str, Image.Image)):
                raise UMMRequestError(
                    "A Lumina image part needs an image, a path, or a URL"
                )
            else:
                pass
        else:
            raise UMMRequestError(f"Lumina does not accept {part.get('type')!r} parts")


def validate_messages(inputs: object) -> None:
    """Reject requests the model cannot serve before they reach a stage."""
    if isinstance(inputs, dict):
        for key in ("images", "audios", "videos"):
            if inputs.get(key):
                raise UMMRequestError(
                    f"Lumina takes {key} as chat content parts, not as a request field"
                )
            else:
                pass
        messages = inputs.get("messages")
    else:
        messages = inputs
    if not isinstance(messages, list) or not messages:
        raise UMMRequestError("Lumina requires chat messages")
    else:
        pass
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in (
            "system",
            "user",
            "assistant",
        ):
            raise UMMRequestError(
                "Lumina messages need a system, user, or assistant role"
            )
        elif message["role"] != "system":
            # A client's system turn is replaced by the output route, so its content is never read.
            validate_content(message["role"], message.get("content"))
        else:
            pass
    if messages[-1]["role"] != "user":
        raise UMMRequestError("Lumina conversations must end in a user message")
    elif any(message["role"] == "system" for message in messages[1:]):
        raise UMMRequestError("Lumina takes a system message only as the first turn")
    else:
        pass


def validate_image_task(inputs: object, task: LuminaTask) -> None:
    """T2I and I2I serve one user turn, with no source image and exactly one."""
    messages = inputs.get("messages") if isinstance(inputs, dict) else inputs
    turns = [message for message in messages if message["role"] != "system"]
    if len(turns) != 1:
        raise UMMRequestError(f"Lumina {task} takes exactly one user turn")
    else:
        pass
    content = turns[0]["content"]
    images = (
        0
        if isinstance(content, str)
        else sum(
            isinstance(part, dict) and part.get("type") in ("image", "image_url")
            for part in content
        )
    )
    expected = 1 if task == "i2i" else 0
    if images != expected:
        raise UMMRequestError(
            f"Lumina {task} takes {expected} input images, the request has {images}"
        )
    else:
        pass


def generation_params(request: OmniRequest) -> LuminaGenerationParams:
    """The client's Lumina controls from stage_params.orchestrator, which top-level fields fill in."""
    stage_params = request.params.get("stage_params") or {}
    options = (
        stage_params.get("orchestrator", {}) if isinstance(stage_params, dict) else None
    )
    if not isinstance(options, dict):
        raise UMMRequestError("stage_params.orchestrator must be a mapping")
    else:
        pass
    merged = dict(options)
    for name in TOP_LEVEL_CONTROLS:
        value = request.params.get(name)
        if value is None:
            pass
        elif name in options and options[name] != value:
            raise UMMRequestError(
                f"The top-level {name} {value} disagrees with "
                f"stage_params.orchestrator.{name} {options[name]}"
            )
        else:
            merged[name] = value
    try:
        return LuminaGenerationParams.model_validate(merged)
    except ValidationError as error:
        raise UMMRequestError(f"Lumina generation controls: {error}") from error


def event_dispatch(context: UMMContext, inputs: Mapping[str, object]) -> UMMDispatch:
    # Only the first event carries the images: the engine keeps the session after it.
    request = OmniRequest(
        dict(inputs), context.request.params, context.request.metadata
    )
    return UMMDispatch(EVENT, request)


class LuminaUMMAdapter:
    """Keeps no per-request state: the engine owns the session, the controller the outputs."""

    def __init__(self, model_path: str) -> None:
        # note (qiujiawei): the controller's adapter contract passes the model path; this adapter reads no model files.
        self.model_path = model_path

    def start(self, context: UMMContext) -> UMMStep:
        validate_messages(context.request.inputs)
        params = generation_params(context.request)
        if params.task in IMAGE_TASK_DEFAULTS:
            validate_image_task(context.request.inputs, params.task)
        else:
            pass
        # Preparation, then at most one dispatch per allowed decision.
        needed = params.max_events + 2
        if needed > context.remaining_dispatches:
            raise UMMRequestError(
                f"max_events={params.max_events} needs {needed} phases, more than the "
                f"{context.remaining_dispatches} this deployment allows. Lower "
                "max_events or raise stages.orchestrator.factory.max_dispatches"
            )
        else:
            pass
        metadata = {
            **context.request.metadata,
            "lumina_generation": params.model_dump(),
        }
        return UMMStep(PREPARE, replace(context.request, metadata=metadata))

    def advance(self, context: UMMContext, phase: str, result: object) -> UMMStep:
        if not isinstance(result, dict):
            raise ValueError(f"Lumina phase {phase!r} returned no mapping")
        elif phase == PREPARE:
            return UMMStep(dispatches=(event_dispatch(context, result),))
        elif phase != EVENT:
            raise ValueError(f"Lumina adapter received unexpected phase {phase!r}")
        else:
            pass
        output = result["output"]
        if output is None or output.get("text") == "":
            outputs = ()
        elif output["kind"] == "text":
            outputs = (UMMOutput("text", output["text"]),)
        else:
            outputs = (UMMOutput("image", output),)
        finish_reason = result["finish_reason"]
        if finish_reason is None:
            return UMMStep(outputs=outputs, dispatches=(event_dispatch(context, {}),))
        else:
            return UMMStep(outputs=outputs, finish_reason=finish_reason)

    def usage(self, context: UMMContext) -> int:
        return 0

    def close(self, context: UMMContext, reason: CloseReason) -> None:
        pass
