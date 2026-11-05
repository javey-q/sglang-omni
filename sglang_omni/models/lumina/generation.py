# SPDX-License-Identifier: Apache-2.0
"""Interleaved event loop of Lumina: singleton, text blocks, image canvases."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from PIL import Image

from sglang_omni.models.lumina.canvas import (
    GenerationPrefix,
    append_clean_event,
    append_native_image_reprefill,
)
from sglang_omni.models.lumina.contracts import (
    GENERATED_GRID_FACTOR,
    TEXT_BLOCK_SIZE,
    CFGBranch,
    CleanEventKind,
    OutputRole,
    TokenRegistry,
    TokenRole,
)
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


def cfg_branch_coefficients(g_text: float, g_image: float) -> dict[CFGBranch, float]:
    """Expanded coefficients of F + g_text(F-I) + g_image(I-U)."""
    return {
        CFGBranch.FULL: 1.0 + g_text,
        CFGBranch.IMAGE_ONLY: g_image - g_text,
        CFGBranch.UNCONDITIONAL: -g_image,
    }


def required_cfg_branches(g_text: float, g_image: float) -> tuple[CFGBranch, ...]:
    coefficients = cfg_branch_coefficients(g_text, g_image)
    return tuple(branch for branch, value in coefficients.items() if abs(value) > 0.0)


def combine_cfg_logits(
    logits: dict[CFGBranch, torch.Tensor],
    *,
    g_text: float,
    g_image: float,
) -> torch.Tensor:
    coefficients = cfg_branch_coefficients(g_text, g_image)
    required = required_cfg_branches(g_text, g_image)
    result = logits[required[0]] * coefficients[required[0]]
    for branch in required[1:]:
        result = result + logits[branch] * coefficients[branch]
    return result


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


def sample_image_tokens(
    logits: torch.Tensor,
    *,
    registry: TokenRegistry,
    temperature: float,
    top_p: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = registry.apply_role_mask(logits, OutputRole.IMAGE_SPAN).float()
    scaled = flat.reshape(-1, flat.shape[-1]) / temperature
    ordered, indices = scaled.sort(descending=True)
    cumulative = ordered.softmax(-1).cumsum(-1)
    reject = cumulative > top_p
    reject[:, 1:] = reject[:, :-1].clone()
    reject[:, 0] = False
    reject = torch.zeros_like(reject).scatter(-1, indices, reject)
    scaled = scaled.masked_fill(reject, -torch.inf)
    distribution = scaled.softmax(-1)
    sampled = torch.multinomial(distribution, 1, generator=generator)
    probabilities = distribution.gather(-1, sampled)
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


def scheduled_image_transfer_count(
    remaining: int, *, step: int, total_steps: int
) -> int:
    """How many masked codes a cosine schedule reveals at this step, at least."""
    steps_left = total_steps - step
    if steps_left == 1:
        return remaining
    else:
        pass
    current_survival = math.cos(math.pi * (step / total_steps) / 2.0)
    next_survival = math.cos(math.pi * ((step + 1) / total_steps) / 2.0)
    scheduled = math.ceil(remaining * (1.0 - next_survival / current_survival))
    return min(remaining, max(1, scheduled))


def generate_image_codes(
    kv_cache: GenerationKVCache,
    prefix: GenerationPrefix,
    *,
    height: int,
    width: int,
    params: LuminaGenerationParams,
    generator: torch.Generator,
) -> list[int]:
    """Reveal one canvas under three-branch CFG and return its raw InsightTok codes."""
    registry = prefix.registry
    current = torch.full(
        (1, (height // GENERATED_GRID_FACTOR) * (width // GENERATED_GRID_FACTOR)),
        registry.mask_token_id,
        dtype=torch.long,
        device=prefix.input_ids.device,
    )
    required = required_cfg_branches(params.g_text, params.g_image)
    for step in range(params.image_steps):
        active = current == registry.mask_token_id
        if not bool(active.any().item()):
            break
        else:
            pass
        branch_logits = {
            branch: kv_cache.forward_event(
                prefix,
                current,
                role=TokenRole.IMAGE_TARGET,
                branch=branch,
                height=height,
                width=width,
            )
            for branch in required
        }
        logits = combine_cfg_logits(
            branch_logits, g_text=params.g_text, g_image=params.g_image
        )
        sampled, probability = sample_image_tokens(
            logits,
            registry=registry,
            temperature=params.image_temperature,
            top_p=params.top_p,
            generator=generator,
        )
        scheduled = scheduled_image_transfer_count(
            int(active.sum().item()), step=step, total_steps=params.image_steps
        )
        confidence = torch.where(
            active, probability, torch.full_like(probability, -torch.inf)
        )
        transfer = (confidence > params.confidence_threshold) & active
        if int(transfer.sum().item()) < scheduled:
            transfer = torch.zeros_like(active)
            transfer.scatter_(1, confidence.topk(scheduled, dim=-1).indices, True)
        else:
            pass
        current = torch.where(transfer & active, sampled, current)
    assert not bool((current == registry.mask_token_id).any().item())
    return [value - registry.code_start_id for value in current[0].tolist()]


def generate_text_blocks(
    kv_cache: GenerationKVCache,
    prefix: GenerationPrefix,
    *,
    params: LuminaGenerationParams,
) -> tuple[list[int], GenerationPrefix]:
    """Denoise and commit q=4 blocks until text_end; returns the visible tokens."""
    registry = prefix.registry
    device = prefix.input_ids.device
    visible: list[int] = []
    steps = params.text_denoising_steps
    for _ in range((params.max_new_tokens + TEXT_BLOCK_SIZE - 1) // TEXT_BLOCK_SIZE):
        current = torch.full(
            (1, TEXT_BLOCK_SIZE),
            registry.mask_token_id,
            dtype=torch.long,
            device=device,
        )
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
        kv_cache.commit_clean_prefix_change(previous, prefix)
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
        if self.is_exhausted:
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
                self.kv_cache, self.prefix, params=self.params
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
        current = append_native_image_reprefill(
            self.runtime, self.prefix, image, raw_ids, height=height, width=width
        )
        self.kv_cache.commit_native_image_reprefill(self.prefix, current)
        self.prefix = current
        return ImageEvent(raw_ids, height, width, image)
