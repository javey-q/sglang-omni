# SPDX-License-Identifier: Apache-2.0
"""Attention program of Lumina: per-token metadata in, boolean masks out."""

from __future__ import annotations

import torch

from sglang_omni.models.lumina.contracts import (
    CFGBranch,
    ConditionProvenance,
    TokenRole,
    VisualSpanKind,
)
from sglang_omni.models.lumina.metadata import TokenMetadata


def kept_for_branch(provenance: torch.Tensor, branch: CFGBranch) -> torch.Tensor:
    """Tokens a CFG branch conditions on: I drops text, U keeps only SAFE."""
    if branch == CFGBranch.FULL:
        return torch.ones_like(provenance, dtype=torch.bool)
    elif branch == CFGBranch.IMAGE_ONLY:
        return provenance != int(ConditionProvenance.TEXT)
    else:
        return provenance == int(ConditionProvenance.SAFE)


def event_attention_mask(
    query: TokenMetadata,
    key: TokenMetadata,
    query_key_indices: torch.Tensor,
    *,
    branch: CFGBranch,
) -> torch.Tensor:
    """The [1, 1, queries, keys] mask: causal across events, bidirectional within one."""
    query_axis = (slice(None), slice(None), None)
    key_axis = (slice(None), None, slice(None))
    query_role = query.token_roles[query_axis]
    query_event = query.event_orders[query_axis]
    query_text_block = query.text_block_ids[query_axis]
    query_visual_span = query.visual_span_ids[query_axis]
    query_visual_kind = query.visual_span_kinds[query_axis]
    key_event = key.event_orders[key_axis]
    key_visual_kind = key.visual_span_kinds[key_axis]
    query_kept = kept_for_branch(query.condition_provenance, branch)[query_axis]
    key_kept = kept_for_branch(key.condition_provenance, branch)[key_axis]

    # In the unconditional branch an image's own start token is isolated so it cannot carry dropped conditions, yet the codes that follow must still read it.
    current_image_start = (
        (query_role == int(TokenRole.IMAGE_TARGET))
        & (query_visual_kind == int(VisualSpanKind.TARGET_INSIGHT_CODES))
        & (key.token_roles[key_axis] == int(TokenRole.VISION_START_PREFIX))
        & (key.condition_provenance[key_axis] == int(ConditionProvenance.IMAGE))
        & (key_event + 1 == query_event)
    )
    same_event = key_event == query_event
    same_text_block = (query_text_block >= 0) & (
        query_text_block == key.text_block_ids[key_axis]
    )
    same_visual_span = (query_visual_span >= 0) & (
        query_visual_span == key.visual_span_ids[key_axis]
    )
    target_visual = (
        same_visual_span
        & (query_visual_kind == int(VisualSpanKind.TARGET_INSIGHT_CODES))
        & (key_visual_kind == int(VisualSpanKind.TARGET_INSIGHT_CODES))
    )
    native_visual = (
        same_visual_span
        & same_event
        & (query_visual_kind == int(VisualSpanKind.NATIVE_VIT))
        & (key_visual_kind == int(VisualSpanKind.NATIVE_VIT))
    )
    structural_internal = same_event & ~same_text_block & ~same_visual_span
    visible = (
        query_kept
        & (key_kept | current_image_start)
        & (
            (key_event < query_event)
            | same_text_block
            | target_visual
            | native_visual
            | structural_internal
        )
    )
    # A dropped token keeps a self edge so its attention row is never empty.
    key_indices = torch.arange(key.token_roles.shape[1], device=query_role.device)
    dropped_self = ~query_kept & (query_key_indices.unsqueeze(-1) == key_indices)
    return (visible | dropped_self).unsqueeze(1)
