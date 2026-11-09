# SPDX-License-Identifier: Apache-2.0
"""Per-branch KV caches. Target K/V never persist; only clean commits grow them."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from sglang_omni.models.lumina.attention import event_attention_mask, kept_for_branch
from sglang_omni.models.lumina.canvas import GenerationPrefix, event_batch
from sglang_omni.models.lumina.contracts import CFGBranch, TokenRole
from sglang_omni.models.lumina.metadata import TokenMetadata
from sglang_omni.models.lumina.modeling import LuminaRuntime
from sglang_omni.models.lumina.static_memory import StaticMemoryCache
from sglang_omni.models.lumina.target_inputs import (
    ImageBatch,
    TargetInputs,
    stack_image_branches,
)


@dataclass
class BranchKVCache:
    """One branch's compact canonical-prefix KV and the metadata of its slots."""

    past_keys: list[torch.Tensor]
    past_values: list[torch.Tensor]
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
        self.image_batches: dict[tuple[CFGBranch, ...], ImageBatch] = {}

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
        outputs = self.runtime.model.decode(
            inputs_embeds=decoder_inputs.inputs_embeds,
            position_ids=event_position_ids[:, :, :prefix_length].index_select(
                2, indices
            ),
            attention_mask=event_attention_mask(
                metadata, metadata, cache_positions.unsqueeze(0), branch=branch
            ),
            visual_pos_masks=decoder_inputs.visual_pos_masks,
            deepstack_visual_embeds=decoder_inputs.deepstack_visual_embeds,
        )
        cache = BranchKVCache(
            past_keys=outputs.keys,
            past_values=outputs.values,
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
        query_ids, inputs = self.target_inputs(
            prefix, current_ids, role=role, branch=branch, height=height, width=width
        )
        return self.target_logits(query_ids, inputs)

    def target_inputs(
        self,
        prefix: GenerationPrefix,
        current_ids: torch.Tensor,
        *,
        role: TokenRole,
        branch: CFGBranch,
        height: int | None,
        width: int | None,
    ) -> tuple[torch.Tensor, TargetInputs]:
        """Query ids and the rest of one branch's target forward."""
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
        inputs = TargetInputs(
            position_ids=position_ids.index_select(2, query_indices),
            attention_mask=attention_mask,
            past_keys=cache.past_keys,
            past_values=cache.past_values,
            detail_values=(cache.detail_values_by_layer if is_detail_read else None),
            detail_source_indices=(
                cache.detail_source_indices if is_detail_read else None
            ),
            detail_active=(
                query_metadata.token_roles == int(TokenRole.IMAGE_TARGET)
                if is_detail_read
                else None
            ),
            output_indices=matches.to(torch.int64).argmax(dim=0),
            code_columns=self.image_code_columns(role),
        )
        return query_ids, inputs

    def forward_image_branches(
        self,
        prefix: GenerationPrefix,
        current_ids: torch.Tensor,
        *,
        branches: tuple[CFGBranch, ...],
        height: int,
        width: int,
    ) -> dict[CFGBranch, torch.Tensor]:
        """Image-target logits of every CFG branch, batched into one forward with cfg_batch."""
        if not self.runtime.model.options.cfg_batch or len(branches) == 1:
            return {
                branch: self.forward_event(
                    prefix,
                    current_ids,
                    role=TokenRole.IMAGE_TARGET,
                    branch=branch,
                    height=height,
                    width=width,
                )
                for branch in branches
            }
        else:
            pass
        batch = self.image_batches.get(branches)
        if batch is None:
            batch = stack_image_branches(
                branches,
                [
                    self.target_inputs(
                        prefix,
                        current_ids,
                        role=TokenRole.IMAGE_TARGET,
                        branch=branch,
                        height=height,
                        width=width,
                    )
                    for branch in branches
                ],
            )
            self.image_batches[branches] = batch
        else:
            pass
        # Positions, masks, and the prefix are fixed until the event commits; only canvas ids change.
        batch.query_ids.index_copy_(
            1, batch.inputs.output_indices, current_ids.expand(len(branches), -1)
        )
        logits = self.target_logits(batch.query_ids, batch.inputs)
        return {branch: logits[row : row + 1] for row, branch in enumerate(branches)}

    def target_logits(
        self, query_ids: torch.Tensor, inputs: TargetInputs
    ) -> torch.Tensor:
        outputs = self.runtime.model.decode(
            inputs_embeds=self.runtime.model.get_input_embeddings()(query_ids),
            position_ids=inputs.position_ids,
            attention_mask=inputs.attention_mask,
            past_keys=inputs.past_keys,
            past_values=inputs.past_values,
            detail_values=inputs.detail_values,
            detail_source_indices=inputs.detail_source_indices,
            detail_active=inputs.detail_active,
        )
        # note (qiujiawei): selecting rows before the head keeps the reference's GEMM shape; another shape changes BF16 logits.
        return self.runtime.model.token_logits(
            outputs.hidden_states,
            inputs.output_indices,
            code_columns=inputs.code_columns,
        )

    def image_code_columns(self, role: TokenRole) -> bool:
        """Whether logits of this role cover only the image codes."""
        return (
            role == TokenRole.IMAGE_TARGET and self.runtime.model.options.code_lm_head
        )

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
        outputs = self.runtime.model.decode(
            inputs_embeds=decoder_inputs.inputs_embeds,
            position_ids=query_positions,
            attention_mask=event_attention_mask(
                query_metadata,
                key_metadata,
                cache_positions.view(1, -1),
                branch=branch,
            ),
            past_keys=cache.past_keys,
            past_values=cache.past_values,
            visual_pos_masks=decoder_inputs.visual_pos_masks,
            deepstack_visual_embeds=decoder_inputs.deepstack_visual_embeds,
        )
        cache.past_keys = outputs.keys
        cache.past_values = outputs.values
        cache.metadata = key_metadata
        cache.canonical_indices = torch.cat((cache.canonical_indices, canonical_suffix))

    def commit_clean_prefix_change(
        self, previous: GenerationPrefix, current: GenerationPrefix
    ) -> None:
        """Append one canonical clean suffix to every materialized branch."""
        # Stacked batches copy the old prefix; the next image event stacks them again.
        self.image_batches = {}
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
