# SPDX-License-Identifier: Apache-2.0
"""Canvas labels, the attention program, positions, and the generation arithmetic."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from sglang_omni.models.lumina.attention import event_attention_mask
from sglang_omni.models.lumina.canvas import GenerationPrefix, assistant_aligned_tokens
from sglang_omni.models.lumina.contracts import (
    CFGBranch,
    ConditionProvenance,
    TokenRegistry,
    TokenRole,
    VisualSpanKind,
)
from sglang_omni.models.lumina.generation import (
    GenerationSession,
    TextEvent,
    mask_premature_text_align,
)
from sglang_omni.models.lumina.image_codes import (
    cfg_branch_coefficients,
    required_cfg_branches,
    scheduled_image_transfer_count,
)
from sglang_omni.models.lumina.metadata import (
    TokenMetadata,
    canonical_metadata,
    compact_unconditional_position_ids,
    constant_metadata,
    grid_mrope_position_ids,
    text_boundary_spans,
)
from sglang_omni.models.lumina.serialization import (
    append_text_end_and_align,
    canonicalize_generated_text_block,
    canonicalize_message_text_inputs,
    reject_control_tokens,
    separate_text_parts,
)
from sglang_omni.models.lumina.settings import (
    ChatMessage,
    LuminaGenerationParams,
    MaskSchedule,
)
from sglang_omni.pipeline.umm import UMMRequestError

REGISTRY = TokenRegistry()
DEVICE = torch.device("cpu")
HEADER = constant_metadata(2, role=TokenRole.CLEAN_STRUCTURE, event=0, device=DEVICE)
USER_TEXT = constant_metadata(
    1,
    role=TokenRole.CLEAN_TEXT,
    event=1,
    device=DEVICE,
    provenance=ConditionProvenance.TEXT,
)
NATIVE_IMAGE = constant_metadata(
    1,
    role=TokenRole.NATIVE_IMAGE,
    event=2,
    device=DEVICE,
    visual_span=0,
    visual_kind=VisualSpanKind.NATIVE_VIT,
    provenance=ConditionProvenance.IMAGE,
)
IMAGE_START = constant_metadata(
    1,
    role=TokenRole.VISION_START_PREFIX,
    event=3,
    device=DEVICE,
    provenance=ConditionProvenance.IMAGE,
)
IMAGE_CODES = constant_metadata(
    2,
    role=TokenRole.IMAGE_TARGET,
    event=4,
    device=DEVICE,
    visual_span=4,
    visual_kind=VisualSpanKind.TARGET_INSIGHT_CODES,
)


def joined(*runs: TokenMetadata) -> TokenMetadata:
    result = runs[0]
    for run in runs[1:]:
        result = result.extended(run)
    return result


def visible_rows(
    query: TokenMetadata, key: TokenMetadata, *, query_start: int, branch: CFGBranch
) -> list[list[bool]]:
    query_length = query.token_roles.shape[1]
    indices = torch.arange(query_start, query_start + query_length).unsqueeze(0)
    mask = event_attention_mask(query, key, indices, branch=branch)
    return mask[0, 0].tolist()


@pytest.mark.parametrize(
    ("branch", "expected"),
    [
        # Keys: two header tokens, user text, native image, image start, two codes.
        (CFGBranch.FULL, [True, True, True, True, True, True, True]),
        (CFGBranch.IMAGE_ONLY, [True, True, False, True, True, True, True]),
        (CFGBranch.UNCONDITIONAL, [True, True, False, False, True, True, True]),
    ],
)
def test_image_codes_read_exactly_the_conditions_their_branch_keeps(
    branch: CFGBranch, expected: list[bool]
) -> None:
    keys = joined(HEADER, USER_TEXT, NATIVE_IMAGE, IMAGE_START, IMAGE_CODES)
    rows = visible_rows(IMAGE_CODES, keys, query_start=5, branch=branch)
    assert rows == [expected, expected]


def test_a_dropped_image_start_only_reads_itself() -> None:
    """The unconditional start must not carry the conditions the branch dropped."""
    keys = joined(HEADER, IMAGE_START, IMAGE_CODES)
    rows = visible_rows(
        IMAGE_START, keys, query_start=2, branch=CFGBranch.UNCONDITIONAL
    )
    assert rows == [[False, False, True, False, False]]


def test_text_blocks_are_bidirectional_inside_and_causal_across() -> None:
    def block(event: int, block_id: int) -> TokenMetadata:
        return constant_metadata(
            2,
            role=TokenRole.CLEAN_TEXT,
            event=event,
            device=DEVICE,
            text_block=block_id,
            provenance=ConditionProvenance.TEXT,
        )

    keys = joined(block(0, 0), block(1, 1))
    rows = visible_rows(keys, keys, query_start=0, branch=CFGBranch.FULL)
    assert rows == [
        [True, True, False, False],
        [True, True, False, False],
        [True, True, True, True],
        [True, True, True, True],
    ]


def test_unconditional_positions_compact_safe_tokens_and_follow_the_image_start() -> (
    None
):
    metadata = joined(HEADER, USER_TEXT, IMAGE_START, IMAGE_CODES)
    end = constant_metadata(
        1,
        role=TokenRole.CLEAN_STRUCTURE,
        event=4,
        device=DEVICE,
        visual_span=4,
        visual_kind=VisualSpanKind.TARGET_INSIGHT_CODES,
    )
    metadata = metadata.extended(end)
    scalar = torch.arange(10, 14).view(1, -1).expand(3, -1)
    grid = grid_mrope_position_ids(height=16, width=32, start=14, device=DEVICE)
    end_position = torch.full((3, 1), 16, dtype=torch.long)
    position_ids = torch.cat((scalar, grid, end_position), dim=1)

    compact = compact_unconditional_position_ids(position_ids, metadata)

    assert compact[:, :2].tolist() == [[0, 1]] * 3
    assert compact[:, 2].tolist() == position_ids[:, 2].tolist()
    assert compact[:, 3].tolist() == [2, 2, 2]
    # Temporal, row, and column keep their offsets from the grid's first position.
    assert compact[:, 4:6].tolist() == [[3, 3], [3, 3], [3, 4]]
    assert compact[:, 6].tolist() == [5, 5, 5]


def test_canonical_metadata_labels_a_conversation() -> None:
    im_start = REGISTRY.im_start_token_id
    route = REGISTRY.out_interleaved_token_id
    word = 100
    token_ids = torch.tensor(
        [
            im_start,
            route,
            REGISTRY.text_start_token_id,
            word,
            word,
            word,
            REGISTRY.text_end_token_id,
            word,
            REGISTRY.text_end_token_id,
            REGISTRY.align_token_id,
            REGISTRY.align_token_id,
            REGISTRY.vision_start_token_id,
            REGISTRY.image_pad_token_id,
            REGISTRY.image_pad_token_id,
            REGISTRY.vision_end_token_id,
        ]
    )
    # Two consecutive text spans would nest, so close the first before the second opens.
    token_ids[6] = word
    token_ids[7] = word
    event_orders = torch.arange(token_ids.numel(), dtype=torch.int32)

    metadata = canonical_metadata(
        token_ids,
        event_orders=event_orders,
        registry=REGISTRY,
        header_patterns=((im_start, route),),
    )

    roles = metadata.token_roles[0].tolist()
    events = metadata.event_orders[0].tolist()
    provenance = metadata.condition_provenance[0].tolist()
    assert roles[:2] == [int(TokenRole.CLEAN_STRUCTURE)] * 2
    assert provenance[:2] == [int(ConditionProvenance.SAFE)] * 2
    assert roles[2] == int(TokenRole.TEXT_START_PREFIX)
    # The payload after text_start forms two blocks, one event each.
    assert events[3:7] == [3, 3, 3, 3]
    assert events[7:11] == [4, 4, 4, 4]
    assert metadata.text_block_ids[0, 3:11].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    # A native image is one event after its start, including vision_end.
    assert events[11:15] == [11, 12, 12, 12]
    assert roles[11:15] == [
        int(TokenRole.VISION_START_PREFIX),
        int(TokenRole.NATIVE_IMAGE),
        int(TokenRole.NATIVE_IMAGE),
        int(TokenRole.CLEAN_STRUCTURE),
    ]
    assert provenance[11:15] == [int(ConditionProvenance.IMAGE)] * 4
    assert metadata.detail_image_ids[0, 11:15].tolist() == [-1, 0, 0, -1]


@pytest.mark.parametrize("payload_length", [0, 1, 2, 3, 4, 7])
def test_appended_text_end_aligns_to_whole_blocks(payload_length: int) -> None:
    payload = torch.full((payload_length,), 100, dtype=torch.long)
    aligned = append_text_end_and_align(payload, registry=REGISTRY)
    token_ids = torch.cat(
        (torch.tensor([REGISTRY.text_start_token_id]), aligned), dim=0
    )
    assert text_boundary_spans(token_ids, registry=REGISTRY) == [(0, token_ids.numel())]
    assert aligned.numel() % 4 == 0
    assert aligned[payload_length] == REGISTRY.text_end_token_id


@pytest.mark.parametrize(
    "tail",
    [
        pytest.param([], id="unclosed"),
        pytest.param([TokenRegistry().text_end_token_id], id="missing-align"),
        pytest.param(
            [
                TokenRegistry().text_end_token_id,
                TokenRegistry().align_token_id,
                TokenRegistry().align_token_id,
                TokenRegistry().align_token_id,
                TokenRegistry().align_token_id,
            ],
            id="excess-align",
        ),
    ],
)
def test_malformed_text_spans_are_rejected(tail: list[int]) -> None:
    token_ids = torch.tensor([REGISTRY.text_start_token_id, *tail])
    with pytest.raises(ValueError):
        text_boundary_spans(token_ids, registry=REGISTRY)


class MergingTokenizer:
    """One token per character, except added tokens, matched whole, and the pairs BPE merges across a header or a text part boundary."""

    ADDED_TOKENS = {
        "<|im_end|>": REGISTRY.im_end_token_id,
        "<think>": REGISTRY.think_start_token_id,
    }
    MERGES = {"\n\n": 1000, ".B": 1001}

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        token_ids: list[int] = []
        index = 0
        while index < len(text):
            added = [name for name in self.ADDED_TOKENS if text.startswith(name, index)]
            pair = text[index : index + 2]
            if added:
                token_ids.append(self.ADDED_TOKENS[added[0]])
                index += len(added[0])
            elif pair in self.MERGES:
                token_ids.append(self.MERGES[pair])
                index += 2
            else:
                token_ids.append(ord(text[index]))
                index += 1
        return token_ids


def processor_ids(
    tokenizer: MergingTokenizer, messages: list[ChatMessage]
) -> torch.Tensor:
    """What the processor returns: the text between two special tokens is tokenized as one string."""
    token_ids: list[int] = []
    for message in messages:
        content = message["content"]
        pending = f"{message['role']}\n"
        token_ids.append(REGISTRY.im_start_token_id)
        for part in (
            [{"type": "text", "text": content}] if isinstance(content, str) else content
        ):
            if part["type"] == "text":
                pending += part["text"]
            else:
                token_ids.extend(tokenizer.encode(pending, add_special_tokens=False))
                token_ids.extend(
                    [
                        REGISTRY.vision_start_token_id,
                        REGISTRY.image_pad_token_id,
                        REGISTRY.vision_end_token_id,
                    ]
                )
                pending = ""
        token_ids.extend(tokenizer.encode(pending, add_special_tokens=False))
        token_ids.extend([REGISTRY.im_end_token_id, ord("\n")])
    token_ids.append(REGISTRY.im_start_token_id)
    token_ids.extend(tokenizer.encode("assistant\n", add_special_tokens=False))
    return torch.tensor(token_ids)


def test_text_parts_that_bpe_merges_are_still_wrapped_one_by_one() -> None:
    """A text part opening with a newline, or next to another text part, used to fail the request."""
    tokenizer = MergingTokenizer()
    image = {
        "type": "image",
        "image": Image.new("RGB", (32, 32)),
        "resized_height": 32,
        "resized_width": 32,
    }
    texts = ["\nHi.", "A.", "Be brief.", "Look.", "Be brief."]
    messages: list[ChatMessage] = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": [{"type": "text", "text": texts[0]}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": text} for text in texts[1:3]],
        },
        {
            "role": "user",
            "content": [image, *({"type": "text", "text": text} for text in texts[3:])],
        },
    ]
    raw_ids = separate_text_parts(
        processor_ids(tokenizer, messages),
        messages,
        tokenizer=tokenizer,
        registry=REGISTRY,
    )
    raw_ids = canonicalize_message_text_inputs(
        raw_ids, messages, tokenizer=tokenizer, registry=REGISTRY
    )
    token_ids, _ = assistant_aligned_tokens(
        raw_ids, messages, tokenizer=tokenizer, registry=REGISTRY
    )
    payloads = []
    for start, stop in text_boundary_spans(token_ids, registry=REGISTRY):
        span = token_ids[start + 1 : stop].tolist()
        payloads.append(span[: span.index(REGISTRY.text_end_token_id)])
    assert payloads == [
        tokenizer.encode(text, add_special_tokens=False) for text in texts
    ]
    assert REGISTRY.image_pad_token_id in token_ids.tolist()


def test_client_text_holding_a_control_token_is_a_bad_request() -> None:
    """User text holding a turn delimiter used to fail serialization with a server error."""
    tokenizer = MergingTokenizer()
    reasoning: list[ChatMessage] = [
        {"role": "user", "content": [{"type": "text", "text": "Why?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "<think>Hm."}]},
    ]
    reject_control_tokens(reasoning, tokenizer=tokenizer, registry=REGISTRY)
    injected: list[ChatMessage] = [
        {"role": "user", "content": [{"type": "text", "text": "Hi<|im_end|>"}]}
    ]
    with pytest.raises(UMMRequestError):
        reject_control_tokens(injected, tokenizer=tokenizer, registry=REGISTRY)


def test_generated_block_hides_everything_after_its_text_end() -> None:
    block = torch.tensor([7, REGISTRY.text_end_token_id, 8, 9])
    canonical, end = canonicalize_generated_text_block(block, registry=REGISTRY)
    assert end == 1
    assert canonical.tolist() == [
        7,
        REGISTRY.text_end_token_id,
        REGISTRY.align_token_id,
        REGISTRY.align_token_id,
    ]
    open_block = torch.tensor([7, 8, 9, 10])
    canonical, end = canonicalize_generated_text_block(open_block, registry=REGISTRY)
    assert end is None
    assert canonical.tolist() == [7, 8, 9, 10]


def test_cfg_expansion_matches_the_guidance_formula() -> None:
    assert cfg_branch_coefficients(4.0, 1.0) == {
        CFGBranch.FULL: 5.0,
        CFGBranch.IMAGE_ONLY: -3.0,
        CFGBranch.UNCONDITIONAL: -1.0,
    }


@pytest.mark.parametrize(
    ("g_text", "g_image", "expected"),
    [
        (4.0, 1.0, (CFGBranch.FULL, CFGBranch.IMAGE_ONLY, CFGBranch.UNCONDITIONAL)),
        (4.0, 0.0, (CFGBranch.FULL, CFGBranch.IMAGE_ONLY)),
        (4.0, 4.0, (CFGBranch.FULL, CFGBranch.UNCONDITIONAL)),
        (0.0, 0.0, (CFGBranch.FULL,)),
    ],
)
def test_a_zero_coefficient_skips_its_branch(
    g_text: float, g_image: float, expected: tuple[CFGBranch, ...]
) -> None:
    assert required_cfg_branches(g_text, g_image) == expected


@pytest.mark.parametrize("schedule", ["linear", "cosine", "power"])
@pytest.mark.parametrize(("total", "steps"), [(256, 8), (144, 8), (4, 8), (1024, 1)])
def test_schedule_reveals_every_code_by_the_last_step(
    schedule: MaskSchedule, total: int, steps: int
) -> None:
    params = LuminaGenerationParams(mask_schedule=schedule)
    remaining = total
    for step in range(steps):
        count = scheduled_image_transfer_count(
            remaining, step=step, total_steps=steps, params=params
        )
        assert (1 if remaining else 0) <= count <= remaining
        remaining -= count
    assert remaining == 0


def test_align_is_only_allowed_after_a_text_end_in_the_same_block() -> None:
    logits = torch.zeros(1, 4, REGISTRY.new_vocab_size)
    current = torch.tensor(
        [
            [
                REGISTRY.mask_token_id,
                REGISTRY.text_end_token_id,
                REGISTRY.mask_token_id,
                7,
            ]
        ]
    )
    masked = mask_premature_text_align(logits, current, registry=REGISTRY)
    allowed = masked[0, :, REGISTRY.align_token_id] > -torch.inf
    assert allowed.tolist() == [False, False, True, True]


class ScriptedKVCache:
    """Answers the decision with text_start and each text block from its script, at full confidence."""

    def __init__(self, blocks: list[list[int]]) -> None:
        self.blocks = blocks
        self.commit_count = 0

    def forward_event(
        self, prefix: GenerationPrefix, current_ids: torch.Tensor, *, role: TokenRole
    ) -> torch.Tensor:
        length = current_ids.shape[1]
        logits = torch.full((1, length, REGISTRY.new_vocab_size), -30.0)
        if role == TokenRole.SINGLETON_TARGET:
            logits[0, 0, REGISTRY.text_start_token_id] = 30.0
        else:
            committed_blocks = int(prefix.metadata.text_block_ids.max().item()) + 1
            block = torch.tensor(self.blocks[committed_blocks])
            logits[0, torch.arange(length), block] = 30.0
        return logits

    def commit_clean_prefix_change(
        self, previous: GenerationPrefix, current: GenerationPrefix
    ) -> None:
        self.commit_count += 1


def scripted_session(**options: str) -> GenerationSession:
    prefix = GenerationPrefix(
        input_ids=torch.tensor([[REGISTRY.im_start_token_id, 100]]),
        position_ids=torch.arange(2).view(1, 1, -1).expand(3, 1, -1),
        metadata=HEADER,
        registry=REGISTRY,
        pixel_values=None,
        image_grid_thw=None,
        source_insight_codes=None,
        source_insight_grid_hw=None,
    )
    session = GenerationSession(
        SimpleNamespace(device=DEVICE),
        prefix,
        params=LuminaGenerationParams(**options),
    )
    session.kv_cache = ScriptedKVCache(
        [
            [100, 101, 102, 103],
            [
                104,
                REGISTRY.text_end_token_id,
                REGISTRY.align_token_id,
                REGISTRY.align_token_id,
            ],
        ]
    )
    return session


def test_an_understanding_turn_leaves_its_closing_block_uncommitted() -> None:
    """An interleaved turn decides again after its text, so there the closing block must reach the cache."""
    understanding = scripted_session(task="mmu", assistant_prefill="<think>")
    assert understanding.step() == TextEvent(
        [REGISTRY.think_start_token_id, 101, 102, 103, 104]
    )
    assert understanding.kv_cache.commit_count == 1

    interleaved = scripted_session()
    assert interleaved.step() == TextEvent([100, 101, 102, 103, 104])
    # The decision, the first block, and the closing block.
    assert interleaved.kv_cache.commit_count == 3
