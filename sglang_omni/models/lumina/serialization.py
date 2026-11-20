# SPDX-License-Identifier: Apache-2.0
"""Chat messages to raw canvas tokens: template, output route, and text wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase, ProcessorMixin

from sglang_omni.models.lumina.contracts import (
    TEXT_BLOCK_SIZE,
    TokenRegistry,
    serialize_role_header,
)
from sglang_omni.models.lumina.settings import ChatMessage, ImagePart, TextPart


@dataclass(frozen=True)
class ProcessedMessages:
    input_ids: torch.Tensor
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None


def message_parts(message: ChatMessage) -> list[TextPart | ImagePart]:
    content = message["content"]
    return [] if isinstance(content, str) else content


def find_subsequence(
    token_values: list[int],
    pattern: Sequence[int],
    start: int,
    stop: int,
    *,
    label: str,
) -> tuple[int, int]:
    """First half-open match of pattern inside token_values[start:stop]."""
    width = len(pattern)
    needle = list(pattern)
    for offset in range(start, stop - width + 1):
        if token_values[offset : offset + width] == needle:
            return offset, offset + width
        else:
            pass
    raise ValueError(f"Could not align {label} in the serialized conversation")


def subsequence_offsets(token_values: list[int], pattern: Sequence[int]) -> list[int]:
    width = len(pattern)
    needle = list(pattern)
    return [
        offset
        for offset in range(len(token_values) - width + 1)
        if token_values[offset : offset + width] == needle
    ]


def native_images(messages: list[ChatMessage]) -> list[Image.Image] | None:
    """Images of the messages in order, each already at its explicit H/32 size."""
    images = []
    for message in messages:
        for part in message_parts(message):
            if part["type"] == "image":
                assert part["image"].size == (
                    part["resized_width"],
                    part["resized_height"],
                )
                images.append(part["image"].convert("RGB"))
            else:
                pass
    return images or None


def process_messages(
    processor: ProcessorMixin,
    messages: list[ChatMessage],
    device: torch.device,
    *,
    should_fit_pixel_budget: bool,
) -> ProcessedMessages:
    """Tokenize and patchify. Images keep their size unless the processor is to fit them into its own pixel budget."""
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    batch = processor(
        text=[text],
        images=native_images(messages),
        videos=None,
        padding=False,
        return_tensors="pt",
        do_resize=should_fit_pixel_budget,
    )
    pixel_values = batch.get("pixel_values")
    image_grid_thw = batch.get("image_grid_thw")
    return ProcessedMessages(
        input_ids=batch["input_ids"].to(device),
        pixel_values=None if pixel_values is None else pixel_values.to(device),
        image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(device),
    )


def separate_text_parts(
    raw_ids: torch.Tensor,
    messages: list[ChatMessage],
    *,
    tokenizer: PreTrainedTokenizerBase,
    registry: TokenRegistry,
) -> torch.Tensor:
    """Tokenize each turn's header and every text part on its own, as the canvas wraps them.

    The template renders a header newline and the text parts up to the next image as one string, so BPE can merge tokens across them. Each such run is checked against that string and then retokenized part by part.
    """
    token_values = raw_ids.detach().cpu().tolist()
    separated: list[int] = []
    cursor = 0
    for message_index, message in enumerate(messages):
        content = message["content"]
        # The texts between consecutive images of the turn; the first run opens with the header.
        text_runs: list[list[str]] = [[f"{message['role']}\n"]]
        if isinstance(content, str):
            text_runs[0].append(content)
        else:
            for part in content:
                if part["type"] == "text":
                    text_runs[-1].append(part["text"])
                else:
                    text_runs.append([])
        _, run_start = find_subsequence(
            token_values,
            [registry.im_start_token_id],
            cursor,
            len(token_values),
            label=f"the start of message {message_index}",
        )
        separated.extend(token_values[cursor:run_start])
        for run_index, texts in enumerate(text_runs):
            is_last_run = run_index == len(text_runs) - 1
            closing_token_id = (
                registry.im_end_token_id
                if is_last_run
                else registry.vision_start_token_id
            )
            run_end, _ = find_subsequence(
                token_values,
                [closing_token_id],
                run_start,
                len(token_values),
                label=f"the end of a text run of message {message_index}",
            )
            rendered = tokenizer.encode("".join(texts), add_special_tokens=False)
            if token_values[run_start:run_end] != rendered:
                raise ValueError(
                    f"Could not align the text of message {message_index} in the serialized conversation"
                )
            else:
                pass
            for text in texts:
                separated.extend(tokenizer.encode(text, add_special_tokens=False))
            if is_last_run:
                next_start = run_end + 1
            else:
                _, next_start = find_subsequence(
                    token_values,
                    [registry.vision_end_token_id],
                    run_end,
                    len(token_values),
                    label=f"an image end of message {message_index}",
                )
            separated.extend(token_values[run_end:next_start])
            run_start = next_start
        cursor = run_start
    separated.extend(token_values[cursor:])
    return torch.tensor(separated, dtype=raw_ids.dtype, device=raw_ids.device)


def rewrite_proxy_assistant_headers(
    raw_ids: torch.Tensor,
    proxy_messages: list[ChatMessage],
    messages: list[ChatMessage],
    *,
    tokenizer: PreTrainedTokenizerBase,
    registry: TokenRegistry,
) -> torch.Tensor:
    """Restore assistant headers that were proxied as user turns for their images."""
    token_values = raw_ids.detach().cpu().tolist()
    cursor = 0
    replacements: list[tuple[int, int, list[int]]] = []
    for proxy, original in zip(proxy_messages, messages, strict=True):
        start, end = find_subsequence(
            token_values,
            serialize_role_header(tokenizer, proxy["role"], registry),
            cursor,
            len(token_values),
            label="a proxied header",
        )
        turn_end, _ = find_subsequence(
            token_values,
            [registry.im_end_token_id],
            end,
            len(token_values),
            label="a proxied turn end",
        )
        if proxy["role"] != original["role"]:
            header = serialize_role_header(tokenizer, original["role"], registry)
            replacements.append((start, end, header))
        else:
            pass
        cursor = turn_end + 1
    result = raw_ids
    for start, end, header in reversed(replacements):
        payload = torch.tensor(header, dtype=result.dtype, device=result.device)
        result = torch.cat((result[:start], payload, result[end:]))
    return result


def insert_output_route_after_system_header(
    raw_ids: torch.Tensor,
    *,
    tokenizer: PreTrainedTokenizerBase,
    registry: TokenRegistry,
    route_token_id: int,
) -> torch.Tensor:
    """Replace the system body with the output-route token of the task."""
    header = tokenizer.encode("<|im_start|>system\n", add_special_tokens=False)
    token_values = raw_ids.detach().cpu().tolist()
    offsets = subsequence_offsets(token_values, header)
    if len(offsets) != 1:
        raise ValueError(
            f"Output routing needs exactly one system header, found {len(offsets)}"
        )
    elif any(route in token_values for route in registry.route_ids):
        raise ValueError("An output-route token is already present in the prompt")
    else:
        pass
    content_start = offsets[0] + len(header)
    content_end, _ = find_subsequence(
        token_values,
        [registry.im_end_token_id],
        content_start,
        len(token_values),
        label="the system turn end",
    )
    route = torch.tensor(
        [route_token_id],
        dtype=raw_ids.dtype,
        device=raw_ids.device,
    )
    return torch.cat((raw_ids[:content_start], route, raw_ids[content_end:]))


def append_text_end_and_align(
    text_ids: torch.Tensor, *, registry: TokenRegistry
) -> torch.Tensor:
    """Close a text payload and pad it to whole blocks with ALIGN."""
    padding = (-(text_ids.numel() + 1)) % TEXT_BLOCK_SIZE
    suffix = torch.full(
        (1 + padding,),
        registry.align_token_id,
        dtype=text_ids.dtype,
        device=text_ids.device,
    )
    suffix[0] = registry.text_end_token_id
    return torch.cat((text_ids, suffix))


def canonicalize_generated_text_block(
    block_ids: torch.Tensor, *, registry: TokenRegistry
) -> tuple[torch.Tensor, int | None]:
    """Replace the sampled tail after the first text_end with canonical ALIGN."""
    matches = torch.nonzero(
        block_ids == registry.text_end_token_id, as_tuple=False
    ).flatten()
    if not matches.numel():
        return block_ids.clone(), None
    else:
        pass
    end = int(matches[0].item())
    canonical = block_ids.clone()
    canonical[end + 1 :] = registry.align_token_id
    return canonical, end


def canonicalize_message_text_inputs(
    raw_ids: torch.Tensor,
    messages: list[ChatMessage],
    *,
    tokenizer: PreTrainedTokenizerBase,
    registry: TokenRegistry,
) -> torch.Tensor:
    """Wrap every user text part in text_start, text_end, and ALIGN; other turns stay."""
    token_values = raw_ids.detach().cpu().tolist()
    replacements: list[tuple[int, int, torch.Tensor]] = []
    cursor = 0
    forbidden = {
        registry.text_start_token_id,
        registry.text_end_token_id,
        registry.align_token_id,
    }
    for message_index, message in enumerate(messages):
        _, content_start = find_subsequence(
            token_values,
            serialize_role_header(tokenizer, message["role"], registry),
            cursor,
            len(token_values),
            label=f"{message['role']} header {message_index}",
        )
        turn_end, _ = find_subsequence(
            token_values,
            [registry.im_end_token_id],
            content_start,
            len(token_values),
            label=f"the end of message {message_index}",
        )
        cursor = turn_end + 1
        if message["role"] != "user":
            continue
        else:
            pass
        part_cursor = content_start
        for part in message_parts(message):
            encoded = (
                tokenizer.encode(part["text"], add_special_tokens=False)
                if part["type"] == "text"
                else []
            )
            if not encoded:
                continue
            elif forbidden.intersection(encoded):
                raise ValueError("User text cannot contain Lumina text wrappers")
            else:
                pass
            start, end = find_subsequence(
                token_values,
                encoded,
                part_cursor,
                turn_end,
                label=f"a text part of message {message_index}",
            )
            text_start = torch.tensor(
                [registry.text_start_token_id],
                dtype=raw_ids.dtype,
                device=raw_ids.device,
            )
            wrapped = torch.cat(
                (
                    text_start,
                    append_text_end_and_align(raw_ids[start:end], registry=registry),
                )
            )
            replacements.append((start, end, wrapped))
            part_cursor = end
    result = raw_ids
    for start, end, wrapped in reversed(replacements):
        result = torch.cat((result[:start], wrapped, result[end:]))
    return result
