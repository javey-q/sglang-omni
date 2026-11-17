# SPDX-License-Identifier: Apache-2.0
"""Reveal one image canvas under CFG, with the trained schedules and confidence policies."""

from __future__ import annotations

import math

import torch

from sglang_omni.models.lumina.canvas import GenerationPrefix
from sglang_omni.models.lumina.contracts import (
    GENERATED_GRID_FACTOR,
    CFGBranch,
    OutputRole,
    TokenRegistry,
    TokenRole,
)
from sglang_omni.models.lumina.kv_cache import GenerationKVCache
from sglang_omni.models.lumina.settings import LuminaGenerationParams, MaskSchedule


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


def sample_image_tokens(
    logits: torch.Tensor,
    *,
    registry: TokenRegistry,
    temperature: float,
    top_p: float,
    generator: torch.Generator,
    code_columns: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample token ids; code-column logits hold exactly the codes the role mask allows."""
    if code_columns:
        flat = logits.float()
        offset = registry.code_start_id
    else:
        flat = registry.apply_role_mask(logits, OutputRole.IMAGE_SPAN).float()
        offset = 0
    flat = flat.reshape(-1, flat.shape[-1])
    if temperature == 0.0:
        sampled = flat.argmax(-1, keepdim=True)
        probabilities = torch.softmax(flat, -1).gather(-1, sampled)
    else:
        scaled = flat / temperature
        if top_p < 1.0:
            ordered, indices = scaled.sort(descending=True)
            cumulative = ordered.softmax(-1).cumsum(-1)
            reject = cumulative > top_p
            reject[:, 1:] = reject[:, :-1].clone()
            reject[:, 0] = False
            reject = torch.zeros_like(reject).scatter(-1, indices, reject)
            scaled = scaled.masked_fill(reject, -torch.inf)
        else:
            pass
        distribution = scaled.softmax(-1)
        sampled = torch.multinomial(distribution, 1, generator=generator)
        probabilities = distribution.gather(-1, sampled)
    shape = logits.shape[:-1]
    return (sampled + offset).view(shape), probabilities.view(shape)


def mask_survival(progress: float, *, schedule: MaskSchedule, power: float) -> float:
    """Share of the canvas still masked after this fraction of the steps."""
    if schedule == "linear":
        return 1.0 - progress
    elif schedule == "cosine":
        return math.cos(math.pi * progress / 2.0)
    else:
        return 1.0 - progress**power


def scheduled_image_transfer_count(
    remaining: int, *, step: int, total_steps: int, params: LuminaGenerationParams
) -> int:
    """How many masked codes the schedule reveals at this step, at least."""
    steps_left = total_steps - step
    if params.mask_schedule == "linear":
        # Integer steps, so linear keeps its trajectory exactly rather than through a ratio.
        return min(remaining, (remaining + steps_left - 1) // steps_left)
    elif steps_left == 1:
        return remaining
    else:
        pass
    current_survival, next_survival = (
        mask_survival(
            position / total_steps,
            schedule=params.mask_schedule,
            power=params.mask_schedule_power,
        )
        for position in (step, step + 1)
    )
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
        branch_logits = kv_cache.forward_image_branches(
            prefix, current, branches=required, height=height, width=width
        )
        logits = combine_cfg_logits(
            branch_logits, g_text=params.g_text, g_image=params.g_image
        )
        sampled, probability = sample_image_tokens(
            logits,
            registry=registry,
            temperature=params.image_temperature,
            top_p=params.top_p,
            generator=generator,
            code_columns=kv_cache.image_code_columns(TokenRole.IMAGE_TARGET),
        )
        scheduled = scheduled_image_transfer_count(
            int(active.sum().item()),
            step=step,
            total_steps=params.image_steps,
            params=params,
        )
        confidence = torch.where(
            active, probability, torch.full_like(probability, -torch.inf)
        )
        transfer = (confidence > params.confidence_threshold) & active
        if (
            params.confidence_policy == "scheduled_topk"
            or int(transfer.sum().item()) < scheduled
        ):
            transfer = torch.zeros_like(active)
            transfer.scatter_(1, confidence.topk(scheduled, dim=-1).indices, True)
        else:
            pass
        current = torch.where(transfer & active, sampled, current)
    assert not bool((current == registry.mask_token_id).any().item())
    return [value - registry.code_start_id for value in current[0].tolist()]
