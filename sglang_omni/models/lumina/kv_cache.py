# SPDX-License-Identifier: Apache-2.0
"""Per-branch KV caches. Target K/V never persist; only clean commits grow them."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from sglang_omni.models.lumina.attention import event_attention_mask, kept_for_branch
from sglang_omni.models.lumina.canvas import GenerationPrefix, event_batch
from sglang_omni.models.lumina.contracts import CFGBranch, TokenRole
from sglang_omni.models.lumina.metadata import TokenMetadata
from sglang_omni.models.lumina.modeling import LuminaRuntime
from sglang_omni.models.lumina.static_memory import StaticMemoryCache


@dataclass
class BranchKVCache:
    """One branch's compact canonical-prefix KV and the metadata of its slots."""

    past_key_values: DynamicCache
    metadata: TokenMetadata
    canonical_indices: torch.Tensor
    attention_masks: dict[TokenRole, torch.Tensor] = field(default_factory=dict)
    detail_source_indices: torch.Tensor | None = None
    detail_values_by_layer: tuple[torch.Tensor | None, ...] | None = None

    @property
    def length(self) -> int:
        return self.canonical_indices.numel()


class GenerationKVCache:
    """Batch-size-one branch cache manager of one generation session."""

    def __init__(self, runtime: LuminaRuntime) -> None:
        self.runtime = runtime
        self.branches: dict[CFGBranch, BranchKVCache] = {}
        self.static_memory = StaticMemoryCache(runtime)

    def attach_branch_detail(
        self, prefix: GenerationPrefix, branch: CFGBranch, cache: BranchKVCache
    ) -> None:
        """Point a branch at the Detail values of the source tokens it physically keeps."""
        memory = (
            None
            if branch == CFGBranch.UNCONDITIONAL
            else self.static_memory.ensure_detail(prefix)
        )
        if memory is None:
            return
        else:
            pass
        canonical_to_cache = {
            canonical: index
            for index, canonical in enumerate(cache.canonical_indices.tolist())
        }
        source_cache_indices = [
            canonical_to_cache[canonical]
            for canonical in memory.source_canonical_indices.tolist()
        ]
        # Only the unconditional branch drops image tokens, and it reads no Detail.
        cache.detail_source_indices = torch.tensor(
            source_cache_indices,
            dtype=torch.long,
            device=cache.canonical_indices.device,
        )
        cache.detail_values_by_layer = memory.values_by_layer

    def ensure_branch(
        self,
        prefix: GenerationPrefix,
        *,
        branch: CFGBranch,
        event_position_ids: torch.Tensor,
        event_metadata: TokenMetadata,
    ) -> BranchKVCache:
        """Materialize one compact branch prefix using upcoming-event positions."""
        existing = self.branches.get(branch)
        if existing is not None:
            return existing
        else:
            pass
        prefix_length = prefix.input_ids.shape[1]
        indices = torch.nonzero(
            kept_for_branch(
                event_metadata.condition_provenance[0, :prefix_length], branch
            ),
            as_tuple=False,
        ).flatten()
        input_ids = prefix.input_ids.index_select(1, indices)
        metadata = event_metadata.select(indices)
        decoder_inputs = self.static_memory.decoder_inputs(
            prefix, input_ids, indices, branch
        )
        cache_positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        outputs = self.runtime.model.base_model.language_model(
            input_ids=None,
            position_ids=event_position_ids[:, :, :prefix_length].index_select(
                2, indices
            ),
            attention_mask=event_attention_mask(
                metadata, metadata, cache_positions.unsqueeze(0), branch=branch
            ),
            past_key_values=DynamicCache(config=self.runtime.model.base_model.config),
            use_cache=True,
            cache_position=cache_positions,
            inputs_embeds=decoder_inputs.inputs_embeds,
            visual_pos_masks=decoder_inputs.visual_pos_masks,
            deepstack_visual_embeds=decoder_inputs.deepstack_visual_embeds,
        )
        cache = BranchKVCache(
            past_key_values=outputs.past_key_values,
            metadata=metadata,
            canonical_indices=indices,
        )
        self.attach_branch_detail(prefix, branch, cache)
        self.branches[branch] = cache
        return cache

    def forward_event(
        self,
        prefix: GenerationPrefix,
        current_ids: torch.Tensor,
        *,
        role: TokenRole,
        branch: CFGBranch = CFGBranch.FULL,
        height: int | None = None,
        width: int | None = None,
    ) -> torch.Tensor:
        """Run only the current block or image against one compact prefix cache."""
        input_ids, position_ids, metadata = event_batch(
            prefix, current_ids, role=role, branch=branch, height=height, width=width
        )
        prefix_length = prefix.input_ids.shape[1]
        cache = self.ensure_branch(
            prefix,
            branch=branch,
            event_position_ids=position_ids,
            event_metadata=metadata,
        )
        event_indices = torch.arange(
            prefix_length, input_ids.shape[1], device=input_ids.device
        )
        if branch == CFGBranch.UNCONDITIONAL and role == TokenRole.IMAGE_TARGET:
            target_event = int(metadata.event_orders[0, prefix_length].item())
            starts = torch.nonzero(
                (
                    metadata.token_roles[0, :prefix_length]
                    == int(TokenRole.VISION_START_PREFIX)
                )
                & (metadata.event_orders[0, :prefix_length] + 1 == target_event),
                as_tuple=False,
            ).flatten()
            query_indices = torch.cat((starts, event_indices))
        else:
            query_indices = event_indices
        query_ids = input_ids.index_select(1, query_indices)
        query_metadata = metadata.select(query_indices)
        query_key_indices = torch.arange(
            cache.length, cache.length + query_ids.shape[1], device=query_ids.device
        ).view(1, -1)
        attention_mask = cache.attention_masks.get(role)
        if attention_mask is None:
            attention_mask = event_attention_mask(
                query_metadata,
                cache.metadata.extended(query_metadata),
                query_key_indices,
                branch=branch,
            )
            cache.attention_masks[role] = attention_mask
        else:
            pass
        is_detail_read = (
            role == TokenRole.IMAGE_TARGET and cache.detail_values_by_layer is not None
        )
        target_indices = torch.arange(
            prefix_length, prefix_length + current_ids.numel(), device=input_ids.device
        )
        matches = query_indices.unsqueeze(1) == target_indices.unsqueeze(0)
        output_indices = matches.to(torch.int64).argmax(dim=0)
        outputs = self.runtime.model.base_model(
            input_ids=query_ids,
            position_ids=position_ids.index_select(2, query_indices),
            attention_mask=attention_mask,
            past_key_values=cache.past_key_values,
            use_cache=False,
            cache_position=query_key_indices[0],
            # note (qiujiawei): gathering rows before the head keeps the full-prefix GEMM shape; a shorter one changes BF16 logits.
            logits_to_keep=output_indices.view(1, -1),
            cached_detail_values=(
                cache.detail_values_by_layer if is_detail_read else None
            ),
            cached_detail_source_indices=(
                cache.detail_source_indices if is_detail_read else None
            ),
            cached_detail_active=(
                query_metadata.token_roles == int(TokenRole.IMAGE_TARGET)
                if is_detail_read
                else None
            ),
        )
        return outputs.logits

    def append_clean_branch(
        self,
        previous: GenerationPrefix,
        current: GenerationPrefix,
        branch: CFGBranch,
        cache: BranchKVCache,
    ) -> None:
        # The next event differs even when this branch keeps nothing of the committed one.
        cache.attention_masks = {}
        canonical_start = previous.input_ids.shape[1]
        canonical_stop = current.input_ids.shape[1]
        canonical_suffix = torch.arange(
            canonical_start, canonical_stop, device=current.input_ids.device
        )
        canonical_suffix = canonical_suffix[
            kept_for_branch(
                current.metadata.condition_provenance[
                    0, canonical_start:canonical_stop
                ],
                branch,
            )
        ]
        if canonical_suffix.numel() == 0:
            return
        else:
            pass
        old_length = cache.length
        new_length = old_length + canonical_suffix.numel()
        query_ids = current.input_ids.index_select(1, canonical_suffix)
        cache_positions = torch.arange(old_length, new_length, device=query_ids.device)
        query_positions = current.position_ids.index_select(2, canonical_suffix)
        if branch == CFGBranch.UNCONDITIONAL:
            query_positions = cache_positions.view(1, 1, -1).expand(3, 1, -1)
        else:
            pass
        query_metadata = current.metadata.select(canonical_suffix)
        key_metadata = cache.metadata.extended(query_metadata)
        decoder_inputs = self.static_memory.decoder_inputs(
            current, query_ids, canonical_suffix, branch
        )
        outputs = self.runtime.model.base_model.language_model(
            input_ids=None,
            position_ids=query_positions,
            attention_mask=event_attention_mask(
                query_metadata,
                key_metadata,
                cache_positions.view(1, -1),
                branch=branch,
            ),
            past_key_values=cache.past_key_values,
            use_cache=True,
            cache_position=cache_positions,
            inputs_embeds=decoder_inputs.inputs_embeds,
            visual_pos_masks=decoder_inputs.visual_pos_masks,
            deepstack_visual_embeds=decoder_inputs.deepstack_visual_embeds,
        )
        cache.past_key_values = outputs.past_key_values
        cache.metadata = key_metadata
        cache.canonical_indices = torch.cat((cache.canonical_indices, canonical_suffix))

    def commit_clean_prefix_change(
        self, previous: GenerationPrefix, current: GenerationPrefix
    ) -> None:
        """Append one canonical clean suffix to every materialized branch."""
        for branch, cache in self.branches.items():
            self.append_clean_branch(previous, current, branch, cache)

    def commit_native_image_reprefill(
        self, previous: GenerationPrefix, current: GenerationPrefix
    ) -> None:
        """Encode one new image once, commit it, and read it from Detail afterwards."""
        self.static_memory.commit_image(previous, current)
        self.commit_clean_prefix_change(previous, current)
        for branch, cache in self.branches.items():
            self.attach_branch_detail(current, branch, cache)
