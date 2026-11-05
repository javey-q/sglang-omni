# SPDX-License-Identifier: Apache-2.0
"""ViT features and Detail values of a session's images, shared by its CFG branches."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang_omni.models.lumina.canvas import GenerationPrefix
from sglang_omni.models.lumina.contracts import CFGBranch
from sglang_omni.models.lumina.detail import StaticDetailMemory
from sglang_omni.models.lumina.modeling import LuminaRuntime


@dataclass(frozen=True)
class StaticVisualMemory:
    """Native ViT and deepstack features of a prefix's images, in token order."""

    pixel_values: torch.Tensor
    canonical_indices: torch.Tensor
    embeddings: torch.Tensor
    deepstack: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class DecoderInputs:
    """Embeddings of a decoder call, with visual features spliced in."""

    inputs_embeds: torch.Tensor
    visual_pos_masks: torch.Tensor | None = None
    deepstack_visual_embeds: list[torch.Tensor] | None = None


class StaticMemoryCache:
    """Encodes each image once per session, however many branches read it."""

    def __init__(self, runtime: LuminaRuntime) -> None:
        self.runtime = runtime
        self.visual: StaticVisualMemory | None = None
        self.detail: StaticDetailMemory | None = None

    def ensure_visual(self, prefix: GenerationPrefix) -> StaticVisualMemory | None:
        pixels = prefix.pixel_values
        if pixels is None:
            return None
        elif self.visual is not None and self.visual.pixel_values is pixels:
            return self.visual
        else:
            pass
        image_parts, deepstack = self.runtime.model.base_model.get_image_features(
            pixels, prefix.image_grid_thw
        )
        self.visual = StaticVisualMemory(
            pixel_values=pixels,
            canonical_indices=torch.nonzero(
                prefix.input_ids[0] == prefix.registry.image_pad_token_id,
                as_tuple=False,
            ).flatten(),
            embeddings=torch.cat(tuple(image_parts), dim=0),
            deepstack=tuple(deepstack),
        )
        return self.visual

    def decoder_inputs(
        self,
        prefix: GenerationPrefix,
        input_ids: torch.Tensor,
        canonical_indices: torch.Tensor,
        branch: CFGBranch,
    ) -> DecoderInputs:
        """Embed a branch's tokens, placing ViT features on its native image tokens."""
        inputs_embeds = self.runtime.model.get_input_embeddings()(input_ids)
        memory = (
            None if branch == CFGBranch.UNCONDITIONAL else self.ensure_visual(prefix)
        )
        if memory is None:
            return DecoderInputs(inputs_embeds)
        else:
            pass
        canonical_to_memory = {
            canonical: index
            for index, canonical in enumerate(memory.canonical_indices.tolist())
        }
        branch_positions: list[int] = []
        memory_positions: list[int] = []
        for branch_index, canonical in enumerate(canonical_indices.tolist()):
            memory_index = canonical_to_memory.get(canonical)
            if memory_index is not None:
                branch_positions.append(branch_index)
                memory_positions.append(memory_index)
            else:
                pass
        if not branch_positions:
            return DecoderInputs(inputs_embeds)
        else:
            pass
        branch_index_tensor = torch.tensor(
            branch_positions, dtype=torch.long, device=input_ids.device
        )
        memory_index_tensor = torch.tensor(
            memory_positions, dtype=torch.long, device=memory.embeddings.device
        )
        selected = memory.embeddings.index_select(0, memory_index_tensor).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[0, branch_index_tensor] = selected
        visual_pos_masks = torch.zeros_like(input_ids, dtype=torch.bool)
        visual_pos_masks[0, branch_index_tensor] = True
        return DecoderInputs(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=[
                value.index_select(0, memory_index_tensor).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                for value in memory.deepstack
            ],
        )

    def ensure_detail(self, prefix: GenerationPrefix) -> StaticDetailMemory | None:
        if prefix.source_insight_codes is None:
            return None
        elif self.detail is not None:
            return self.detail
        else:
            pass
        self.detail = self.runtime.model.build_static_detail_memory(
            input_ids=prefix.input_ids,
            source_insight_codes=prefix.source_insight_codes,
            source_insight_grid_hw=prefix.source_insight_grid_hw,
            detail_image_ids=prefix.metadata.detail_image_ids,
        )
        return self.detail

    def commit_image(
        self, previous: GenerationPrefix, current: GenerationPrefix
    ) -> None:
        """Encode the newly committed image alone and extend the shared features."""
        previous_memory = self.ensure_visual(previous)
        old_prefix_length = previous.input_ids.shape[1]
        new_grid = current.image_grid_thw[-1:]
        pixels = current.pixel_values
        new_pixels = pixels[pixels.shape[0] - int(new_grid.prod().item()) :]
        image_parts, deepstack = self.runtime.model.base_model.get_image_features(
            new_pixels, new_grid
        )
        embedding = self.runtime.model.get_input_embeddings().weight
        new_embeddings = torch.cat(tuple(image_parts), dim=0).to(
            embedding.device, embedding.dtype
        )
        new_deepstack = tuple(
            value.to(embedding.device, embedding.dtype) for value in deepstack
        )
        image_mask = (
            current.input_ids[0, old_prefix_length:]
            == previous.registry.image_pad_token_id
        )
        new_indices = (
            torch.nonzero(image_mask, as_tuple=False).flatten() + old_prefix_length
        )
        if previous_memory is None:
            self.visual = StaticVisualMemory(
                pixels, new_indices, new_embeddings, new_deepstack
            )
        else:
            self.visual = StaticVisualMemory(
                pixels,
                torch.cat((previous_memory.canonical_indices, new_indices)),
                torch.cat((previous_memory.embeddings, new_embeddings)),
                tuple(
                    torch.cat((old_value, new_value))
                    for old_value, new_value in zip(
                        previous_memory.deepstack, new_deepstack, strict=True
                    )
                ),
            )
        self.detail = None
