# SPDX-License-Identifier: Apache-2.0
"""Interleaved event loop of Lumina: singleton, text blocks, image canvases."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image

from sglang_omni.models.lumina.canvas import (
    GenerationPrefix,
    append_clean_event,
    append_native_image_reprefill,
)
from sglang_omni.models.lumina.contracts import (
    TEXT_BLOCK_SIZE,
    CleanEventKind,
    OutputRole,
    TokenRegistry,
    TokenRole,
)
from sglang_omni.models.lumina.image_codes import generate_image_codes
from sglang_omni.models.lumina.kv_cache import GenerationKVCache
from sglang_omni.models.lumina.modeling import LuminaRuntime
from sglang_omni.models.lumina.serialization import canonicalize_generated_text_block
from sglang_omni.models.lumina.settings import LuminaGenerationParams


@dataclass(frozen=True)
class TextEvent:
    token_ids: list[int]


@dataclass(frozen=True)
class ImageEvent:
    raw_ids: list[int]
    height: int
    width: int
    image: Image.Image


@dataclass(frozen=True)
class EndEvent:
    """The model closed its turn."""


class IncompleteTextGenerationError(RuntimeError):
    """Text generation exhausted its budget before text_end; keeps the partial text."""

    def __init__(self, message: str, *, token_ids: list[int]) -> None:
        super().__init__(message)
        self.token_ids = token_ids


class IncompleteInterleavedGenerationError(RuntimeError):
    """Interleaved generation exhausted its event budget before im_end."""


def greedy_tokens(
    logits: torch.Tensor,
    *,
    registry: TokenRegistry,
    role: OutputRole,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = registry.apply_role_mask(logits, role).float()
    flat = flat.reshape(-1, flat.shape[-1])
    sampled = flat.argmax(-1, keepdim=True)
    probabilities = torch.softmax(flat, -1).gather(-1, sampled)
    shape = logits.shape[:-1]
    return sampled.view(shape), probabilities.view(shape)


def mask_premature_text_align(
    logits: torch.Tensor,
    current: torch.Tensor,
    *,
    registry: TokenRegistry,
) -> torch.Tensor:
    """Allow ALIGN only after a text_end already committed earlier in this block."""
    seen_end = current == registry.text_end_token_id
    end_before = (seen_end.cumsum(dim=-1) - seen_end.to(torch.long)) > 0
    result = logits.clone()
    result[..., registry.align_token_id] = result[
        ..., registry.align_token_id
    ].masked_fill(~end_before, -torch.inf)
    return result


def generate_text_blocks(
    kv_cache: GenerationKVCache,
    prefix: GenerationPrefix,
    *,
    params: LuminaGenerationParams,
    opening_token_ids: list[int],
    should_commit_closing_block: bool,
) -> tuple[list[int], GenerationPrefix]:
    """Denoise and commit q=4 blocks until text_end; returns the visible tokens.

    The opening tokens start the first block and stay visible. A turn that ends with this text reads nothing after its closing block, so it can leave that block uncommitted.
    """
    registry = prefix.registry
    device = prefix.input_ids.device
    assert len(opening_token_ids) < TEXT_BLOCK_SIZE
    visible: list[int] = []
    steps = params.text_denoising_steps
    block_count = (params.max_new_tokens + TEXT_BLOCK_SIZE - 1) // TEXT_BLOCK_SIZE
    for block_index in range(block_count):
        current = torch.full(
            (1, TEXT_BLOCK_SIZE),
            registry.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        if block_index == 0:
            current[0, : len(opening_token_ids)] = torch.tensor(
                opening_token_ids, dtype=torch.long, device=device
            )
        else:
            pass
        for step in range(steps):
            active = current == registry.mask_token_id
            if not bool(active.any().item()):
                break
            else:
                pass
            logits = kv_cache.forward_event(prefix, current, role=TokenRole.TEXT_TARGET)
            logits = mask_premature_text_align(logits, current, registry=registry)
            sampled, probability = greedy_tokens(
                logits, registry=registry, role=OutputRole.TEXT_BLOCK
            )
            remaining = int(active.sum().item())
            transfer_count = max(1, (remaining + steps - step - 1) // (steps - step))
            confidence = torch.where(
                active, probability, torch.full_like(probability, -torch.inf)
            )
            transfer = confidence > params.confidence_threshold
            if int(transfer.sum().item()) < transfer_count:
                transfer.zero_()
                transfer.scatter_(
                    1,
                    confidence.topk(min(transfer_count, remaining), dim=-1).indices,
                    True,
                )
            else:
                pass
            current = torch.where(transfer & active, sampled, current)
        assert not bool((current == registry.mask_token_id).any().item())
        raw_block = current[0].detach()
        block = raw_block.tolist()
        canonical_block, end = canonicalize_generated_text_block(
            raw_block, registry=registry
        )
        visible.extend(block if end is None else block[:end])
        previous = prefix
        prefix = append_clean_event(previous, canonical_block, kind=CleanEventKind.TEXT)
        if end is None or should_commit_closing_block:
            kv_cache.commit_clean_prefix_change(previous, prefix)
        else:
            pass
        if end is not None:
            return visible, prefix
        else:
            pass
    raise IncompleteTextGenerationError(
        "Text generation reached max_new_tokens without <|text_end|>",
        token_ids=visible,
    )


class GenerationSession:
    """One assistant turn: each step decides and fully commits one event."""

    def __init__(
        self,
        runtime: LuminaRuntime,
        prefix: GenerationPrefix,
        *,
        params: LuminaGenerationParams,
    ) -> None:
        self.runtime = runtime
        self.prefix = prefix
        self.params = params
        self.kv_cache = GenerationKVCache(runtime)
        self.generator = torch.Generator(device=runtime.device)
        if params.seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(params.seed)
        self.decision_count = 0
        self.image_event_count = 0

    @property
    def is_exhausted(self) -> bool:
        """Every allowed decision was made, so step would only report truncation."""
        return self.decision_count > self.params.max_events

    def append_decision(self, token_id: int, *, kind: CleanEventKind) -> None:
        """Commit the singleton's token as a clean event on every materialized branch."""
        previous = self.prefix
        token = torch.tensor([token_id], device=previous.input_ids.device)
        self.prefix = append_clean_event(previous, token, kind=kind)
        self.kv_cache.commit_clean_prefix_change(previous, self.prefix)

    @torch.inference_mode()
    def step(self) -> TextEvent | ImageEvent | EndEvent:
        if self.params.task == "mmu":
            return self.task_text_step()
        elif self.params.task != "interleaved":
            return self.draw_image()
        elif self.is_exhausted:
            raise IncompleteInterleavedGenerationError(
                "Interleaved generation exceeded max_events without <|im_end|>"
            )
        else:
            pass
        self.decision_count += 1
        registry = self.prefix.registry
        logits = self.kv_cache.forward_event(
            self.prefix,
            torch.full(
                (1, 1),
                registry.mask_token_id,
                dtype=torch.long,
                device=self.prefix.input_ids.device,
            ),
            role=TokenRole.SINGLETON_TARGET,
        )
        token, _ = greedy_tokens(
            logits, registry=registry, role=OutputRole.INTERLEAVED_SINGLETON
        )
        token_id = int(token.item())
        if token_id == registry.im_end_token_id:
            self.append_decision(token_id, kind=CleanEventKind.STRUCTURE)
            return EndEvent()
        elif token_id == registry.text_start_token_id:
            self.append_decision(token_id, kind=CleanEventKind.TEXT_START)
            token_ids, self.prefix = generate_text_blocks(
                self.kv_cache,
                self.prefix,
                params=self.params,
                opening_token_ids=[],
                should_commit_closing_block=True,
            )
            return TextEvent(token_ids)
        else:
            self.append_decision(token_id, kind=CleanEventKind.IMAGE_START)
            return self.generate_image_event()

    def generate_image_event(self) -> ImageEvent:
        if self.image_event_count >= self.params.max_image_events:
            raise IncompleteInterleavedGenerationError(
                f"Lumina turn exceeded max_image_events={self.params.max_image_events}"
            )
        else:
            pass
        self.image_event_count += 1
        event = self.draw_image()
        current = append_native_image_reprefill(
            self.runtime,
            self.prefix,
            event.image,
            event.raw_ids,
            height=event.height,
            width=event.width,
        )
        self.kv_cache.commit_native_image_reprefill(self.prefix, current)
        self.prefix = current
        return event

    def task_text_step(self) -> TextEvent:
        """An MMU turn: its prefix already opens the one answer, and nothing reads it afterwards."""
        registry = self.prefix.registry
        token_ids, self.prefix = generate_text_blocks(
            self.kv_cache,
            self.prefix,
            params=self.params,
            opening_token_ids=(
                []
                if self.params.assistant_prefill is None
                else [registry.think_start_token_id]
            ),
            should_commit_closing_block=False,
        )
        return TextEvent(token_ids)

    def draw_image(self) -> ImageEvent:
        """Reveal the opened canvas and decode it; a T2I or I2I prefix already opens its one image."""
        height, width = self.params.height, self.params.width
        raw_ids = generate_image_codes(
            self.kv_cache,
            self.prefix,
            height=height,
            width=width,
            params=self.params,
            generator=self.generator,
        )
        image = self.runtime.decode_codes(raw_ids, height, width)
        return ImageEvent(raw_ids, height, width, image)
