# SPDX-License-Identifier: Apache-2.0
"""The Lumina canvas: tokens, positions, and metadata of one conversation."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.lumina.contracts import (
    GENERATED_GRID_FACTOR,
    TEXT_BLOCK_SIZE,
    CFGBranch,
    CleanEventKind,
    ConditionProvenance,
    TokenRegistry,
    TokenRole,
    VisualSpanKind,
    chat_header_token_patterns,
    serialize_role_header,
)
from sglang_omni.models.lumina.metadata import (
    TokenMetadata,
    canonical_metadata,
    compact_unconditional_position_ids,
    constant_metadata,
    grid_mrope_position_ids,
    next_mrope_position,
    scalar_mrope_position_ids,
)
from sglang_omni.models.lumina.modeling import LuminaRuntime
from sglang_omni.models.lumina.serialization import (
    append_text_end_and_align,
    canonicalize_message_text_inputs,
    find_subsequence,
    insert_output_route_after_system_header,
    message_parts,
    process_messages,
    rewrite_proxy_assistant_headers,
)
from sglang_omni.models.lumina.settings import IMAGE_EDGE_FACTOR, ChatMessage

CLEAN_EVENT_ROLES = {
    CleanEventKind.STRUCTURE: TokenRole.CLEAN_STRUCTURE,
    CleanEventKind.TEXT: TokenRole.CLEAN_TEXT,
    CleanEventKind.TEXT_START: TokenRole.TEXT_START_PREFIX,
    CleanEventKind.IMAGE_START: TokenRole.VISION_START_PREFIX,
}
CLEAN_EVENT_PROVENANCE = {
    CleanEventKind.STRUCTURE: ConditionProvenance.SAFE,
    CleanEventKind.TEXT: ConditionProvenance.TEXT,
    CleanEventKind.TEXT_START: ConditionProvenance.TEXT,
    CleanEventKind.IMAGE_START: ConditionProvenance.IMAGE,
}


@dataclass(frozen=True, kw_only=True)
class GenerationPrefix:
    """Everything committed so far, in canonical order, with batch size one."""

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    metadata: TokenMetadata
    registry: TokenRegistry
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    source_insight_codes: torch.Tensor | None
    source_insight_grid_hw: torch.Tensor | None


def assistant_aligned_tokens(
    raw_ids: torch.Tensor,
    messages: list[ChatMessage],
    *,
    tokenizer: PreTrainedTokenizerBase,
    registry: TokenRegistry,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrap assistant text into blocks and order every token of the history by event."""
    token_values = raw_ids.detach().cpu().tolist()
    device = raw_ids.device
    token_parts: list[torch.Tensor] = []
    event_parts: list[torch.Tensor] = []
    next_event = 0

    def append_plain(part: torch.Tensor) -> None:
        nonlocal next_event
        token_parts.append(part)
        event_parts.append(
            torch.arange(
                next_event, next_event + part.numel(), dtype=torch.int32, device=device
            )
        )
        next_event += part.numel()

    cursor = 0
    for message_index, message in enumerate(messages):
        _, content_start = find_subsequence(
            token_values,
            serialize_role_header(tokenizer, message["role"], registry),
            cursor,
            len(token_values),
            label=f"the header of message {message_index}",
        )
        turn_end, _ = find_subsequence(
            token_values,
            [registry.im_end_token_id],
            content_start,
            len(token_values),
            label=f"the end of message {message_index}",
        )
        part_cursor = content_start
        for part in message_parts(message) if message["role"] == "assistant" else []:
            if part["type"] == "text":
                encoded = tokenizer.encode(part["text"], add_special_tokens=False)
                if not encoded:
                    continue
                else:
                    pass
                start, end = find_subsequence(
                    token_values,
                    encoded,
                    part_cursor,
                    turn_end,
                    label=f"a text part of message {message_index}",
                )
                append_plain(raw_ids[cursor:start])
                text_start = torch.tensor(
                    [registry.text_start_token_id], dtype=torch.long, device=device
                )
                token_parts.append(text_start)
                event_parts.append(
                    torch.tensor([next_event], dtype=torch.int32, device=device)
                )
                next_event += 1
                aligned = append_text_end_and_align(
                    raw_ids[start:end], registry=registry
                )
                for block in aligned.split(TEXT_BLOCK_SIZE):
                    token_parts.append(block)
                    event_parts.append(
                        torch.full(
                            (TEXT_BLOCK_SIZE,),
                            next_event,
                            dtype=torch.int32,
                            device=device,
                        )
                    )
                    next_event += 1
            else:
                start, _ = find_subsequence(
                    token_values,
                    [registry.vision_start_token_id],
                    part_cursor,
                    turn_end,
                    label=f"an image of message {message_index}",
                )
                _, end = find_subsequence(
                    token_values,
                    [registry.vision_end_token_id],
                    start + 1,
                    turn_end,
                    label=f"an image end of message {message_index}",
                )
                append_plain(raw_ids[cursor:start])
                image_tokens = raw_ids[start:end]
                image_events = torch.full(
                    (image_tokens.numel(),),
                    next_event + 1,
                    dtype=torch.int32,
                    device=device,
                )
                image_events[0] = next_event
                token_parts.append(image_tokens)
                event_parts.append(image_events)
                next_event += 2
            part_cursor = end
            cursor = end
        append_plain(raw_ids[cursor : turn_end + 1])
        cursor = turn_end + 1
    append_plain(raw_ids[cursor:])
    return torch.cat(token_parts), torch.cat(event_parts)


def prepare_interleaved_prefix(
    runtime: LuminaRuntime, messages: list[ChatMessage]
) -> GenerationPrefix:
    """Serialize a multi-turn history, encode its images, and open the next response.

    The processor cannot render assistant image parts, so those turns pass through it as user turns and their headers are restored exactly afterwards.
    """
    registry = runtime.model.registry
    tokenizer = runtime.processor.tokenizer
    device = runtime.device
    proxy_messages: list[ChatMessage] = [
        (
            {"role": "user", "content": message["content"]}
            if message["role"] == "assistant"
            and any(part["type"] == "image" for part in message_parts(message))
            else message
        )
        for message in messages
    ]
    processed = process_messages(runtime.processor, proxy_messages, device)
    raw_ids = rewrite_proxy_assistant_headers(
        processed.input_ids[0],
        proxy_messages,
        messages,
        tokenizer=tokenizer,
        registry=registry,
    )
    raw_ids = insert_output_route_after_system_header(
        raw_ids, tokenizer=tokenizer, registry=registry
    )
    raw_ids = canonicalize_message_text_inputs(
        raw_ids, messages, tokenizer=tokenizer, registry=registry
    )
    token_ids, event_orders = assistant_aligned_tokens(
        raw_ids, messages, tokenizer=tokenizer, registry=registry
    )
    position_ids, _ = runtime.model.base_model.model.get_rope_index(
        input_ids=token_ids.unsqueeze(0),
        image_grid_thw=processed.image_grid_thw,
        attention_mask=torch.ones(
            (1, token_ids.numel()), dtype=torch.long, device=device
        ),
    )
    codes: list[int] = []
    grids: list[list[int]] = []
    for message in messages:
        for part in message_parts(message):
            if part["type"] == "image":
                height, width = part["resized_height"], part["resized_width"]
                codes.extend(runtime.encode_image(part["image"], height, width))
                grids.append(
                    [height // GENERATED_GRID_FACTOR, width // GENERATED_GRID_FACTOR]
                )
            else:
                pass
    return GenerationPrefix(
        input_ids=token_ids.unsqueeze(0),
        position_ids=position_ids,
        metadata=canonical_metadata(
            token_ids,
            event_orders=event_orders,
            registry=registry,
            header_patterns=chat_header_token_patterns(tokenizer, registry),
        ),
        registry=registry,
        pixel_values=processed.pixel_values,
        image_grid_thw=processed.image_grid_thw,
        source_insight_codes=(
            torch.tensor(codes, dtype=torch.long, device=device) if codes else None
        ),
        source_insight_grid_hw=(
            torch.tensor(grids, dtype=torch.long, device=device) if grids else None
        ),
    )


def append_optional_tensor(
    existing: torch.Tensor | None, suffix: torch.Tensor
) -> torch.Tensor:
    return suffix if existing is None else torch.cat((existing, suffix), dim=0)


def append_native_image_reprefill(
    runtime: LuminaRuntime,
    previous: GenerationPrefix,
    image: Image.Image,
    raw_codes: list[int],
    *,
    height: int,
    width: int,
) -> GenerationPrefix:
    """Close one generated image so later events read it exactly like a source image."""
    registry = previous.registry
    device = previous.input_ids.device
    assert int(previous.input_ids[0, -1]) == registry.vision_start_token_id
    visual_batch = runtime.processor.image_processor(
        images=[image], do_resize=False, return_tensors="pt"
    )
    payload_length = (height // IMAGE_EDGE_FACTOR) * (width // IMAGE_EDGE_FACTOR)
    suffix_ids = torch.tensor(
        [
            [
                *([registry.image_pad_token_id] * payload_length),
                registry.vision_end_token_id,
            ]
        ],
        dtype=previous.input_ids.dtype,
        device=device,
    )
    input_ids = torch.cat((previous.input_ids, suffix_ids), dim=1)
    image_grid_thw = append_optional_tensor(
        previous.image_grid_thw, visual_batch["image_grid_thw"].to(device)
    )
    position_ids, _ = runtime.model.base_model.model.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=torch.ones_like(input_ids),
    )
    event = int(previous.metadata.event_orders[0, -1].item()) + 1
    visual_span = int(previous.metadata.visual_span_ids.max().item()) + 1
    previous_grids = previous.source_insight_grid_hw
    detail_image = 0 if previous_grids is None else int(previous_grids.shape[0])
    payload_metadata = constant_metadata(
        payload_length,
        role=TokenRole.NATIVE_IMAGE,
        event=event,
        device=device,
        visual_span=visual_span,
        visual_kind=VisualSpanKind.NATIVE_VIT,
        provenance=ConditionProvenance.IMAGE,
        detail_image=detail_image,
    )
    end_metadata = constant_metadata(
        1,
        role=TokenRole.CLEAN_STRUCTURE,
        event=event,
        device=device,
        visual_span=visual_span,
        visual_kind=VisualSpanKind.NATIVE_VIT,
        provenance=ConditionProvenance.IMAGE,
    )
    grid = [[height // GENERATED_GRID_FACTOR, width // GENERATED_GRID_FACTOR]]
    return replace(
        previous,
        input_ids=input_ids,
        position_ids=position_ids,
        metadata=previous.metadata.extended(payload_metadata).extended(end_metadata),
        pixel_values=append_optional_tensor(
            previous.pixel_values, visual_batch["pixel_values"].to(device)
        ),
        image_grid_thw=image_grid_thw,
        source_insight_codes=append_optional_tensor(
            previous.source_insight_codes,
            torch.tensor(raw_codes, dtype=torch.long, device=device),
        ),
        source_insight_grid_hw=append_optional_tensor(
            previous_grids, torch.tensor(grid, dtype=torch.long, device=device)
        ),
    )


def append_clean_event(
    prefix: GenerationPrefix, token_ids: torch.Tensor, *, kind: CleanEventKind
) -> GenerationPrefix:
    """Commit one finished event: its tokens take sequential scalar positions."""
    device = prefix.input_ids.device
    length = token_ids.numel()
    metadata = prefix.metadata
    suffix = constant_metadata(
        length,
        role=CLEAN_EVENT_ROLES[kind],
        event=int(metadata.event_orders.max().item()) + 1,
        device=device,
        text_block=(
            int(metadata.text_block_ids.max().item()) + 1
            if kind == CleanEventKind.TEXT
            else -1
        ),
        provenance=CLEAN_EVENT_PROVENANCE[kind],
    )
    positions = scalar_mrope_position_ids(
        next_mrope_position(prefix.position_ids), length, device
    )
    return replace(
        prefix,
        input_ids=torch.cat((prefix.input_ids, token_ids.reshape(1, -1)), dim=1),
        position_ids=torch.cat((prefix.position_ids, positions), dim=-1),
        metadata=metadata.extended(suffix),
    )


def event_batch(
    prefix: GenerationPrefix,
    current_ids: torch.Tensor,
    *,
    role: TokenRole,
    branch: CFGBranch,
    height: int | None = None,
    width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, TokenMetadata]:
    """Tokens, positions, and metadata of the prefix followed by one target event."""
    device = current_ids.device
    length = current_ids.shape[1]
    event = int(prefix.metadata.event_orders.max().item()) + 1
    start = next_mrope_position(prefix.position_ids)
    if role == TokenRole.IMAGE_TARGET:
        code_positions = grid_mrope_position_ids(
            height=height, width=width, start=start, device=device
        )
        end_position = torch.full(
            (3, 1),
            next_mrope_position(code_positions),
            dtype=torch.long,
            device=device,
        )
        positions = torch.cat((code_positions, end_position), dim=1).unsqueeze(1)
        end_token = torch.tensor(
            [[prefix.registry.vision_end_token_id]],
            dtype=current_ids.dtype,
            device=device,
        )
        event_ids = torch.cat((current_ids, end_token), dim=1)
        suffix = constant_metadata(
            length,
            role=role,
            event=event,
            device=device,
            visual_span=event,
            visual_kind=VisualSpanKind.TARGET_INSIGHT_CODES,
        ).extended(
            constant_metadata(
                1,
                role=TokenRole.CLEAN_STRUCTURE,
                event=event,
                device=device,
                visual_span=event,
                visual_kind=VisualSpanKind.TARGET_INSIGHT_CODES,
            )
        )
    else:
        positions = scalar_mrope_position_ids(start, length, device)
        event_ids = current_ids
        suffix = constant_metadata(
            length,
            role=role,
            event=event,
            device=device,
            text_block=event if role == TokenRole.TEXT_TARGET else -1,
        )
    metadata = prefix.metadata.extended(suffix)
    input_ids = torch.cat((prefix.input_ids, event_ids), dim=1)
    position_ids = torch.cat((prefix.position_ids, positions), dim=-1)
    if branch == CFGBranch.UNCONDITIONAL:
        position_ids[:, 0] = compact_unconditional_position_ids(
            position_ids[:, 0], metadata
        )
    else:
        pass
    return input_ids, position_ids, metadata
