# SPDX-License-Identifier: Apache-2.0
"""Token registry, roles, and output vocabularies of Lumina."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

import torch
from transformers import PreTrainedTokenizerBase

TEXT_BLOCK_SIZE = 4
GENERATED_GRID_FACTOR = 16


class VisualSpanKind(IntEnum):
    NONE = 0
    NATIVE_VIT = 1
    TARGET_INSIGHT_CODES = 2


class TokenRole(IntEnum):
    CLEAN_TEXT = 1
    CLEAN_STRUCTURE = 2
    NATIVE_IMAGE = 3
    TEXT_TARGET = 4
    IMAGE_TARGET = 5
    SINGLETON_TARGET = 6
    VISION_START_PREFIX = 8
    TEXT_START_PREFIX = 9


class ConditionProvenance(IntEnum):
    SAFE = 0
    TEXT = 1
    IMAGE = 2


class CFGBranch(IntEnum):
    FULL = 0
    IMAGE_ONLY = 1
    UNCONDITIONAL = 2


class OutputRole(StrEnum):
    TEXT_BLOCK = "text_block"
    IMAGE_SPAN = "image_span"
    INTERLEAVED_SINGLETON = "interleaved_singleton"


class CleanEventKind(StrEnum):
    """What a token committed to the canvas is, which fixes its role and provenance."""

    STRUCTURE = "structure"
    TEXT = "text"
    TEXT_START = "text_start"
    IMAGE_START = "image_start"


@dataclass(frozen=True)
class TokenRegistry:
    """Direct token ids of the Lumina vocabulary."""

    align_token_id: int = 151662
    think_start_token_id: int = 151667
    think_end_token_id: int = 151668
    text_start_token_id: int = 151669
    text_end_token_id: int = 151670
    mask_token_id: int = 151671
    out_image_token_id: int = 151672
    out_interleaved_token_id: int = 151673
    out_text_token_id: int = 151674
    endoftext_token_id: int = 151643
    im_start_token_id: int = 151644
    im_end_token_id: int = 151645
    object_ref_start_token_id: int = 151646
    object_ref_end_token_id: int = 151647
    box_start_token_id: int = 151648
    box_end_token_id: int = 151649
    quad_start_token_id: int = 151650
    quad_end_token_id: int = 151651
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_pad_token_id: int = 151654
    image_pad_token_id: int = 151655
    video_pad_token_id: int = 151656
    tool_call_start_token_id: int = 151657
    tool_call_end_token_id: int = 151658
    fim_prefix_token_id: int = 151659
    fim_middle_token_id: int = 151660
    fim_suffix_token_id: int = 151661
    repo_name_token_id: int = 151663
    file_sep_token_id: int = 151664
    tool_response_start_token_id: int = 151665
    tool_response_end_token_id: int = 151666
    native_tokenizer_vocab_size: int = 151669
    code_start_id: int = 151936
    codebook_size: int = 16384
    # Three retired rows follow the codebook, so the logits are wider than the codes.
    new_vocab_size: int = 168323

    @property
    def code_end_id(self) -> int:
        return self.code_start_id + self.codebook_size - 1

    @property
    def route_ids(self) -> tuple[int, int, int]:
        return (
            self.out_text_token_id,
            self.out_image_token_id,
            self.out_interleaved_token_id,
        )

    @property
    def native_non_text_control_ids(self) -> tuple[int, ...]:
        return (
            self.endoftext_token_id,
            self.object_ref_start_token_id,
            self.object_ref_end_token_id,
            self.box_start_token_id,
            self.box_end_token_id,
            self.quad_start_token_id,
            self.quad_end_token_id,
            self.vision_pad_token_id,
            self.tool_call_start_token_id,
            self.tool_call_end_token_id,
            self.fim_prefix_token_id,
            self.fim_middle_token_id,
            self.fim_suffix_token_id,
            self.repo_name_token_id,
            self.file_sep_token_id,
            self.tool_response_start_token_id,
            self.tool_response_end_token_id,
        )

    def role_vocabulary_mask(
        self,
        role: OutputRole,
        *,
        vocab_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the hard vocabulary mask of one generation role."""
        allowed = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        if role == OutputRole.IMAGE_SPAN:
            allowed[self.code_start_id : self.code_end_id + 1] = True
        elif role == OutputRole.INTERLEAVED_SINGLETON:
            allowed[
                torch.tensor(
                    [
                        self.text_start_token_id,
                        self.vision_start_token_id,
                        self.im_end_token_id,
                    ],
                    dtype=torch.long,
                    device=device,
                )
            ] = True
        else:
            allowed[: self.native_tokenizer_vocab_size] = True
            forbidden = {
                *self.native_non_text_control_ids,
                self.im_start_token_id,
                self.im_end_token_id,
                self.vision_start_token_id,
                self.vision_end_token_id,
                self.image_pad_token_id,
                self.video_pad_token_id,
                self.text_start_token_id,
                self.mask_token_id,
                *self.route_ids,
            }
            allowed[sorted(forbidden)] = False
            allowed[self.align_token_id] = True
            allowed[self.think_start_token_id] = True
            allowed[self.think_end_token_id] = True
            allowed[self.text_end_token_id] = True
        return allowed

    def apply_role_mask(self, logits: torch.Tensor, role: OutputRole) -> torch.Tensor:
        allowed = self.role_vocabulary_mask(
            role, vocab_size=logits.shape[-1], device=logits.device
        )
        return logits.masked_fill(~allowed, -torch.inf)


def serialize_role_header(
    tokenizer: PreTrainedTokenizerBase, role: str, registry: TokenRegistry
) -> list[int]:
    body = tokenizer.encode(f"{role}\n", add_special_tokens=False)
    return [registry.im_start_token_id, *body]


def chat_header_token_patterns(
    tokenizer: PreTrainedTokenizerBase,
    registry: TokenRegistry,
) -> tuple[tuple[int, ...], ...]:
    """Return the exact token sequences of the system, user, and assistant headers."""
    return tuple(
        tuple(serialize_role_header(tokenizer, role, registry))
        for role in ("system", "user", "assistant")
    )
