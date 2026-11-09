# SPDX-License-Identifier: Apache-2.0
"""Stage factories of the Lumina pipeline."""

from __future__ import annotations

import base64
import hashlib
import io
import logging

from sglang_omni.models.lumina.canvas import prepare_interleaved_prefix
from sglang_omni.models.lumina.generation import (
    EndEvent,
    GenerationSession,
    ImageEvent,
    IncompleteInterleavedGenerationError,
    IncompleteTextGenerationError,
)
from sglang_omni.models.lumina.modeling import (
    DecoderOptions,
    LuminaRuntime,
    load_lumina_runtime,
)
from sglang_omni.models.lumina.preprocessor import preprocess
from sglang_omni.models.lumina.settings import (
    EngineInputs,
    EventResult,
    ImageSegment,
    LuminaGenerationParams,
    TextSegment,
)
from sglang_omni.pipeline.umm import UMMFinishReason
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.device import resolve_concrete_device

logger = logging.getLogger(__name__)


def create_preprocessing_executor() -> SimpleScheduler[StagePayload, StagePayload]:
    return SimpleScheduler(preprocess)


def start_session(runtime: LuminaRuntime, inputs: EngineInputs) -> GenerationSession:
    """Build the canvas of a new turn; every history image is encoded here."""
    params = LuminaGenerationParams.model_validate(inputs["generation"])
    prefix = prepare_interleaved_prefix(runtime, inputs["messages"])
    return GenerationSession(runtime, prefix, params=params)


def image_output(event: ImageEvent) -> ImageSegment:
    """One inline image segment, lossless so a later turn re-encodes the same pixels."""
    buffer = io.BytesIO()
    event.image.save(buffer, format="PNG")
    png = buffer.getvalue()
    encoded = base64.b64encode(png).decode("ascii")
    return {
        "kind": "image",
        "mime_type": "image/png",
        "size_bytes": len(png),
        "sha256": hashlib.sha256(png).hexdigest(),
        "url": f"data:image/png;base64,{encoded}",
        "height": event.height,
        "width": event.width,
    }


def create_engine_executor(
    model_path: str,
    *,
    compile_decoder: bool,
    code_lm_head: bool,
    fused_rope: bool,
    fused_rmsnorm: bool,
    cfg_batch: bool,
    device: str | None = None,
    gpu_id: int | None = None,
) -> SimpleScheduler[StagePayload, StagePayload]:
    """One dispatch runs one event, so concurrent requests alternate between events."""
    runtime = load_lumina_runtime(
        model_path,
        resolve_concrete_device(device, gpu_id),
        compile_decoder=compile_decoder,
        options=DecoderOptions(
            code_lm_head=code_lm_head,
            fused_rope=fused_rope,
            fused_rmsnorm=fused_rmsnorm,
            cfg_batch=cfg_batch,
        ),
    )
    tokenizer = runtime.processor.tokenizer
    sessions: dict[str, GenerationSession] = {}

    def decode_text(token_ids: list[int]) -> TextSegment:
        text = tokenizer.decode(token_ids, skip_special_tokens=False).strip()
        return {"kind": "text", "text": text}

    def run_event(payload: StagePayload) -> StagePayload:
        session = sessions.get(payload.request_id)
        if session is None:
            session = start_session(runtime, payload.data)
            sessions[payload.request_id] = session
        else:
            pass
        output: TextSegment | ImageSegment | None = None
        finish_reason: UMMFinishReason | None = None
        try:
            event = session.step()
            if isinstance(event, EndEvent):
                finish_reason = "stop"
            else:
                output = (
                    image_output(event)
                    if isinstance(event, ImageEvent)
                    else decode_text(event.token_ids)
                )
                # The last allowed decision produced an event, so the turn stays open.
                finish_reason = "length" if session.is_exhausted else None
        except IncompleteTextGenerationError as error:
            logger.info(f"Lumina text truncated for {payload.request_id}: {error}")
            output, finish_reason = decode_text(error.token_ids), "length"
        except IncompleteInterleavedGenerationError as error:
            logger.info(f"Lumina turn truncated for {payload.request_id}: {error}")
            finish_reason = "length"
        except Exception:
            sessions.pop(payload.request_id, None)
            raise
        if finish_reason is not None:
            sessions.pop(payload.request_id, None)
        else:
            pass
        result: EventResult = {"output": output, "finish_reason": finish_reason}
        payload.data = result
        return payload

    def forget(request_id: str) -> None:
        if sessions.pop(request_id, None) is not None:
            logger.info(
                f"Lumina session released for {request_id}, {len(sessions)} active"
            )
        else:
            pass

    return SimpleScheduler(run_event, abort_callback=forget)
