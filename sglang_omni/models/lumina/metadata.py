# SPDX-License-Identifier: Apache-2.0
"""Per-token metadata of a canvas and the MRoPE positions of what follows it."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch

from sglang_omni.models.lumina.contracts import (
    GENERATED_GRID_FACTOR,
    TEXT_BLOCK_SIZE,
    ConditionProvenance,
    TokenRegistry,
    TokenRole,
    VisualSpanKind,
)
from sglang_omni.models.lumina.serialization import subsequence_offsets


class TokenMetadata(NamedTuple):
    """One [1, tokens] column per attribute the attention program reads."""

    token_roles: torch.Tensor
    event_orders: torch.Tensor
    text_block_ids: torch.Tensor
    visual_span_ids: torch.Tensor
    visual_span_kinds: torch.Tensor
    condition_provenance: torch.Tensor
    detail_image_ids: torch.Tensor

    def select(self, indices: torch.Tensor) -> TokenMetadata:
        return TokenMetadata(*(column.index_select(1, indices) for column in self))

    def extended(self, suffix: TokenMetadata) -> TokenMetadata:
        return TokenMetadata(
            *(torch.cat(columns, dim=1) for columns in zip(self, suffix, strict=True))
        )


def constant_metadata(
    length: int,
    *,
    role: TokenRole,
    event: int,
    device: torch.device,
    text_block: int = -1,
    visual_span: int = -1,
    visual_kind: VisualSpanKind = VisualSpanKind.NONE,
    provenance: ConditionProvenance = ConditionProvenance.SAFE,
    detail_image: int = -1,
) -> TokenMetadata:
    """Columns of a run of tokens that share every attribute."""
    return TokenMetadata(
        token_roles=torch.full((1, length), int(role), dtype=torch.int8, device=device),
        event_orders=torch.full((1, length), event, dtype=torch.int32, device=device),
        text_block_ids=torch.full(
            (1, length), text_block, dtype=torch.int32, device=device
        ),
        visual_span_ids=torch.full(
            (1, length), visual_span, dtype=torch.int32, device=device
        ),
        visual_span_kinds=torch.full(
            (1, length), int(visual_kind), dtype=torch.int8, device=device
        ),
        condition_provenance=torch.full(
            (1, length), int(provenance), dtype=torch.int8, device=device
        ),
        detail_image_ids=torch.full(
            (1, length), detail_image, dtype=torch.int32, device=device
        ),
    )


def text_boundary_spans(
    token_ids: torch.Tensor, *, registry: TokenRegistry
) -> list[tuple[int, int]]:
    """Half-open spans from each text_start through the ALIGN after its text_end."""
    token_values = token_ids.detach().cpu().tolist()
    spans: list[tuple[int, int]] = []
    start: int | None = None
    index = 0
    while index < len(token_values):
        value = token_values[index]
        if value == registry.text_start_token_id:
            if start is not None:
                raise ValueError("Nested canonical text spans are invalid")
            else:
                pass
            start = index
        elif value == registry.text_end_token_id:
            if start is None:
                raise ValueError("Canonical text_end has no matching text_start")
            else:
                pass
            stop = index + 1 + ((start - index) % TEXT_BLOCK_SIZE)
            padding = token_values[index + 1 : stop]
            if stop > len(token_values) or set(padding) - {registry.align_token_id}:
                raise ValueError("Canonical text_end has an invalid ALIGN suffix")
            elif (
                stop < len(token_values)
                and token_values[stop] == registry.align_token_id
            ):
                raise ValueError("Canonical text span contains excess ALIGN tokens")
            else:
                pass
            spans.append((start, stop))
            start = None
            index = stop - 1
        else:
            pass
        index += 1
    if start is not None:
        raise ValueError("Canonical text span is not closed")
    else:
        pass
    return spans


def vision_boundary_spans(
    token_ids: torch.Tensor, *, registry: TokenRegistry
) -> list[tuple[int, int]]:
    """Indices of each vision_start and its vision_end."""
    token_values = token_ids.detach().cpu().tolist()
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(token_values):
        if value == registry.vision_start_token_id:
            if start is not None:
                raise ValueError("Nested canonical vision spans are invalid")
            else:
                pass
            start = index
        elif value == registry.vision_end_token_id:
            if start is None:
                raise ValueError("Canonical vision_end has no matching vision_start")
            elif index == start + 1:
                raise ValueError("Canonical vision payload cannot be empty")
            else:
                pass
            spans.append((start, index))
            start = None
        else:
            pass
    if start is not None:
        raise ValueError("Canonical vision span is not closed")
    else:
        pass
    return spans


def canonical_metadata(
    token_ids: torch.Tensor,
    *,
    event_orders: torch.Tensor,
    registry: TokenRegistry,
    header_patterns: Sequence[Sequence[int]],
) -> TokenMetadata:
    """Label a finished conversation: structure is SAFE, text blocks and images are spans."""
    length = token_ids.numel()
    device = token_ids.device
    roles = torch.full(
        (length,), int(TokenRole.CLEAN_TEXT), dtype=torch.int8, device=device
    )
    events = event_orders.clone()
    text_blocks = torch.full((length,), -1, dtype=torch.int32, device=device)
    visual_spans = torch.full((length,), -1, dtype=torch.int32, device=device)
    visual_kinds = torch.full(
        (length,), int(VisualSpanKind.NONE), dtype=torch.int8, device=device
    )
    provenance = torch.full(
        (length,), int(ConditionProvenance.TEXT), dtype=torch.int8, device=device
    )
    detail_images = torch.full((length,), -1, dtype=torch.int32, device=device)
    for token_id in (
        registry.im_start_token_id,
        registry.im_end_token_id,
        registry.vision_start_token_id,
        registry.vision_end_token_id,
        *registry.route_ids,
    ):
        selected = token_ids == token_id
        roles[selected] = int(TokenRole.CLEAN_STRUCTURE)
        provenance[selected] = int(ConditionProvenance.SAFE)
    token_values = token_ids.detach().cpu().tolist()
    for pattern in header_patterns:
        for offset in subsequence_offsets(token_values, pattern):
            header = slice(offset, offset + len(pattern))
            roles[header] = int(TokenRole.CLEAN_STRUCTURE)
            provenance[header] = int(ConditionProvenance.SAFE)
    next_text_block = 0
    for start, stop in text_boundary_spans(token_ids, registry=registry):
        start_event = int(events[start].item())
        roles[start] = int(TokenRole.TEXT_START_PREFIX)
        provenance[start] = int(ConditionProvenance.TEXT)
        for block_index, offset in enumerate(range(start + 1, stop, TEXT_BLOCK_SIZE)):
            block = slice(offset, offset + TEXT_BLOCK_SIZE)
            events[block] = start_event + 1 + block_index
            text_blocks[block] = next_text_block
            roles[block] = int(TokenRole.CLEAN_TEXT)
            provenance[block] = int(ConditionProvenance.TEXT)
            next_text_block += 1
    spans = vision_boundary_spans(token_ids, registry=registry)
    for image_index, (start, end) in enumerate(spans):
        payload = slice(start + 1, end)
        if not bool((token_ids[payload] == registry.image_pad_token_id).all().item()):
            raise ValueError("Canonical vision payload must contain only image pads")
        else:
            pass
        visual_event = slice(start + 1, end + 1)
        events[visual_event] = int(events[start].item()) + 1
        roles[start] = int(TokenRole.VISION_START_PREFIX)
        roles[payload] = int(TokenRole.NATIVE_IMAGE)
        roles[end] = int(TokenRole.CLEAN_STRUCTURE)
        visual_spans[visual_event] = image_index
        visual_kinds[visual_event] = int(VisualSpanKind.NATIVE_VIT)
        provenance[start : end + 1] = int(ConditionProvenance.IMAGE)
        detail_images[payload] = image_index
    return TokenMetadata(
        token_roles=roles.unsqueeze(0),
        event_orders=events.unsqueeze(0),
        text_block_ids=text_blocks.unsqueeze(0),
        visual_span_ids=visual_spans.unsqueeze(0),
        visual_span_kinds=visual_kinds.unsqueeze(0),
        condition_provenance=provenance.unsqueeze(0),
        detail_image_ids=detail_images.unsqueeze(0),
    )


def next_mrope_position(position_ids: torch.Tensor) -> int:
    return int(position_ids.max().item()) + 1


def scalar_mrope_position_ids(
    start: int, length: int, device: torch.device
) -> torch.Tensor:
    """[3, 1, length] positions of sequential tokens, equal on all three axes."""
    scalar = torch.arange(start, start + length, device=device)
    return scalar.view(1, 1, -1).expand(3, 1, -1)


def grid_mrope_position_ids(
    *, height: int, width: int, start: int, device: torch.device
) -> torch.Tensor:
    """[3, codes] positions of raster-ordered codes, one per 16x16 pixels."""
    token_height = height // GENERATED_GRID_FACTOR
    token_width = width // GENERATED_GRID_FACTOR
    rows = torch.arange(token_height, device=device).view(-1, 1).expand(-1, token_width)
    columns = (
        torch.arange(token_width, device=device).view(1, -1).expand(token_height, -1)
    )
    temporal = torch.zeros_like(rows)
    grid = torch.stack([temporal.flatten(), rows.flatten(), columns.flatten()])
    return grid.to(torch.long) + start


def compact_unconditional_position_ids(
    position_ids: torch.Tensor, metadata: TokenMetadata
) -> torch.Tensor:
    """Rebase [3, tokens] positions so an image event reads SAFE tokens consecutively.

    The image start follows the retained SAFE tokens of earlier events, and the H/16 grid and vision_end move right behind it. Dropped tokens keep stale positions because the unconditional mask isolates them.
    """
    token_roles = metadata.token_roles[0]
    event_orders = metadata.event_orders[0]
    result = position_ids.clone()
    target_visual = metadata.visual_span_kinds[0] == int(
        VisualSpanKind.TARGET_INSIGHT_CODES
    )
    safe_prefix = (
        metadata.condition_provenance[0] == int(ConditionProvenance.SAFE)
    ) & ~target_visual
    safe_indices = torch.nonzero(safe_prefix, as_tuple=False).flatten()
    compact = torch.arange(
        safe_indices.numel(), dtype=torch.long, device=position_ids.device
    )
    result[:, safe_indices] = compact.unsqueeze(0).expand(3, -1)

    code_indices = torch.nonzero(
        token_roles == int(TokenRole.IMAGE_TARGET), as_tuple=False
    ).flatten()
    target_event = int(event_orders[code_indices[0]].item())
    starts = torch.nonzero(
        (token_roles == int(TokenRole.VISION_START_PREFIX))
        & (event_orders + 1 == target_event),
        as_tuple=False,
    ).flatten()
    assert starts.numel() == 1
    start_position = int((safe_prefix & (event_orders < target_event)).sum().item())
    result[:, starts[0]] = start_position
    target_indices = torch.nonzero(target_visual, as_tuple=False).flatten()
    old_start = int(position_ids[:, code_indices].min().item())
    result[:, target_indices] = (
        position_ids[:, target_indices] - old_start + start_position + 1
    )
    return result
