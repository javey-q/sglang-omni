# SPDX-License-Identifier: Apache-2.0
"""The Lumina decoder matches the vendored decoder, and its speedups keep its results exactly or within rounding."""

from types import SimpleNamespace

import pytest
import torch
from torch._dynamo.utils import counters
from transformers import DynamicCache

from sglang_omni.models.lumina.contracts import CFGBranch, TokenRegistry
from sglang_omni.models.lumina.generation import sample_image_tokens
from sglang_omni.models.lumina.kv_cache import GenerationKVCache
from sglang_omni.models.lumina.modeling import DecoderOptions, LuminaModel
from sglang_omni.models.lumina.target_inputs import TargetInputs, stack_image_branches
from sglang_omni.vendor.bard_vl.configuration_bard_vl import BardVLConfig
from sglang_omni.vendor.bard_vl.modeling_bard_vl import BardVLForConditionalGeneration

HIDDEN_SIZE = 64
VOCAB_SIZE = 32
LAYERS = 3
DETAIL_SOURCES = 2


def tiny_model(
    device: str = "cpu",
    *,
    options: DecoderOptions = DecoderOptions(),
    head_dim: int = 16,
    dtype: torch.dtype = torch.bfloat16,
) -> LuminaModel:
    torch.manual_seed(0)
    config = BardVLConfig(
        text_config={
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": 128,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": head_dim,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "num_heads": 2,
            "out_hidden_size": HIDDEN_SIZE,
            "deepstack_visual_indexes": [0],
        },
    )
    base_model = BardVLForConditionalGeneration(config)
    # LuminaModel keeps detached weight views, so the dtype is fixed before it is built.
    base_model = base_model.to(device=device, dtype=dtype).eval()
    return (
        LuminaModel(
            base_model,
            registry=TokenRegistry(),
            detail_start_layer=0,
            options=options,
        )
        .to(dtype)
        .eval()
    )


@pytest.fixture(scope="module")
def model() -> LuminaModel:
    return tiny_model()


def embeddings(length: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(1, length, HIDDEN_SIZE, generator=generator)
    return values.to(torch.bfloat16)


def scalar_positions(start: int, length: int) -> torch.Tensor:
    return torch.arange(start, start + length).view(1, 1, -1).expand(3, 1, -1)


def causal_mask(queries: int, keys: int) -> torch.Tensor:
    return torch.tril(
        torch.ones(queries, keys, dtype=torch.bool), diagonal=keys - queries
    )[None, None]


def assert_cache_equal(
    cache: DynamicCache, keys: list[torch.Tensor], values: list[torch.Tensor]
) -> None:
    for layer, key, value in zip(cache.layers, keys, values, strict=True):
        assert torch.equal(layer.keys, key)
        assert torch.equal(layer.values, value)


@torch.inference_mode()
def test_prefill_target_and_commit_match_the_vendored_decoder(
    model: LuminaModel,
) -> None:
    decoder = model.base_model.language_model
    cache = DynamicCache(config=model.base_model.config)
    prefix, target, commit = 6, 3, 2

    vendored_prefill = decoder(
        inputs_embeds=embeddings(prefix, seed=1),
        position_ids=scalar_positions(0, prefix),
        attention_mask=causal_mask(prefix, prefix),
        past_key_values=cache,
        use_cache=True,
    )
    prefill = model.decode(
        inputs_embeds=embeddings(prefix, seed=1),
        position_ids=scalar_positions(0, prefix),
        attention_mask=causal_mask(prefix, prefix),
    )
    assert_cache_equal(cache, prefill.keys, prefill.values)

    every_row = torch.arange(prefix)
    logits = model.token_logits(prefill.hidden_states, every_row, code_columns=False)
    vendored_logits = model.base_model.lm_head(vendored_prefill.last_hidden_state)
    assert torch.equal(logits, vendored_logits)

    vendored_target = decoder(
        inputs_embeds=embeddings(target, seed=2),
        position_ids=scalar_positions(prefix, target),
        attention_mask=causal_mask(target, prefix + target),
        past_key_values=cache,
        use_cache=False,
    )
    read = model.decode(
        inputs_embeds=embeddings(target, seed=2),
        position_ids=scalar_positions(prefix, target),
        attention_mask=causal_mask(target, prefix + target),
        past_keys=prefill.keys,
        past_values=prefill.values,
    )
    logits = model.token_logits(
        read.hidden_states, torch.arange(target), code_columns=False
    )
    vendored_logits = model.base_model.lm_head(vendored_target.last_hidden_state)
    assert torch.equal(logits, vendored_logits)
    # Reading the target must leave the committed prefix alone.
    assert cache.layers[0].keys.shape[-2] == prefix

    decoder(
        inputs_embeds=embeddings(commit, seed=3),
        position_ids=scalar_positions(prefix, commit),
        attention_mask=causal_mask(commit, prefix + commit),
        past_key_values=cache,
        use_cache=True,
    )
    committed = model.decode(
        inputs_embeds=embeddings(commit, seed=3),
        position_ids=scalar_positions(prefix, commit),
        attention_mask=causal_mask(commit, prefix + commit),
        past_keys=prefill.keys,
        past_values=prefill.values,
    )
    assert_cache_equal(cache, committed.keys, committed.values)


@torch.inference_mode()
def test_deepstack_features_match_the_vendored_decoder(model: LuminaModel) -> None:
    length = 5
    visual_pos_masks = torch.tensor([[False, True, True, False, False]])
    deepstack = [embeddings(2, seed=4)[0]]
    vendored = model.base_model.language_model(
        inputs_embeds=embeddings(length, seed=5),
        position_ids=scalar_positions(0, length),
        attention_mask=causal_mask(length, length),
        past_key_values=DynamicCache(config=model.base_model.config),
        use_cache=True,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack,
    )
    output = model.decode(
        inputs_embeds=embeddings(length, seed=5),
        position_ids=scalar_positions(0, length),
        attention_mask=causal_mask(length, length),
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack,
    )
    logits = model.token_logits(
        output.hidden_states, torch.arange(length), code_columns=False
    )
    assert torch.equal(logits, model.base_model.lm_head(vendored.last_hidden_state))


@torch.inference_mode()
def test_detail_reaches_only_the_tokens_marked_active(model: LuminaModel) -> None:
    prefix, target = 4, 2
    prefill = model.decode(
        inputs_embeds=embeddings(prefix, seed=6),
        position_ids=scalar_positions(0, prefix),
        attention_mask=causal_mask(prefix, prefix),
    )
    # V of the source tokens, one [batch, key_value_heads, sources, head_dim] per layer.
    source_values = tuple(
        torch.ones_like(value[:, :, :2, :], dtype=torch.bfloat16)
        for value in prefill.values
    )

    def read(active: list[bool] | None) -> torch.Tensor:
        return model.decode(
            inputs_embeds=embeddings(target, seed=7),
            position_ids=scalar_positions(prefix, target),
            attention_mask=causal_mask(target, prefix + target),
            past_keys=prefill.keys,
            past_values=prefill.values,
            detail_values=None if active is None else source_values,
            detail_source_indices=None if active is None else torch.tensor([0, 1]),
            detail_active=None if active is None else torch.tensor([active]),
        ).hidden_states

    plain = read(None)
    assert torch.equal(read([False, False]), plain)
    partly = read([True, False])
    assert not torch.equal(partly[:, 0], plain[:, 0])


@pytest.mark.accelerator
@torch.inference_mode()
def test_warmed_compiled_decoder_agrees_with_eager_and_never_recompiles() -> None:
    if not torch.cuda.is_available():
        pytest.skip("the compiled decoder needs CUDA")
    else:
        pass
    device = torch.device("cuda")
    compiled = tiny_model("cuda")
    eager = tiny_model("cuda")
    compiled.enable_compiled_decoder()
    graphs = counters["stats"]["unique_graphs"]

    def run(model: LuminaModel, length: int, *, prefix: int) -> torch.Tensor:
        prefill = model.decode(
            inputs_embeds=embeddings(prefix, seed=8).to(device),
            position_ids=scalar_positions(0, prefix).to(device),
            attention_mask=causal_mask(prefix, prefix).to(device),
        )
        target = model.decode(
            inputs_embeds=embeddings(length, seed=9).to(device),
            position_ids=scalar_positions(prefix, length).to(device),
            attention_mask=causal_mask(length, prefix + length).to(device),
            past_keys=prefill.keys,
            past_values=prefill.values,
        )
        return target.hidden_states.float()

    for length, prefix in ((1, 70), (4, 70), (4, 300), (129, 70), (129, 300)):
        difference = run(compiled, length, prefix=prefix) - run(
            eager, length, prefix=prefix
        )
        reference = run(eager, length, prefix=prefix)
        assert difference.norm() <= 1e-2 * reference.norm()
    assert counters["stats"]["unique_graphs"] == graphs


def test_code_columns_sample_the_codes_the_role_mask_allows() -> None:
    registry = TokenRegistry()
    full = torch.randn(
        (1, 3, registry.new_vocab_size), generator=torch.Generator().manual_seed(4)
    )
    codes = full[..., registry.code_start_id : registry.code_end_id + 1]

    def sample(
        logits: torch.Tensor, *, code_columns: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # A tiny top_p keeps only the best code, so both paths must agree exactly.
        return sample_image_tokens(
            logits,
            registry=registry,
            temperature=1.0,
            top_p=1e-9,
            generator=torch.Generator().manual_seed(5),
            code_columns=code_columns,
        )

    masked_ids, masked_probability = sample(full, code_columns=False)
    code_ids, code_probability = sample(codes, code_columns=True)
    assert torch.equal(masked_ids, code_ids)
    assert torch.equal(code_ids, codes.argmax(-1) + registry.code_start_id)
    assert torch.equal(masked_probability, code_probability)


def branch_inputs(
    model: LuminaModel,
    *,
    prefix: int,
    query: int,
    canvas_rows: list[int],
    detail_values: tuple[torch.Tensor, ...] | None,
    seed: int,
) -> tuple[torch.Tensor, TargetInputs]:
    prefill = model.decode(
        inputs_embeds=embeddings(prefix, seed=seed).to(torch.float32),
        position_ids=scalar_positions(0, prefix),
        attention_mask=causal_mask(prefix, prefix),
    )
    query_ids = torch.randint(
        0, 32, (1, query), generator=torch.Generator().manual_seed(seed)
    )
    mask = torch.ones((1, 1, query, prefix + query), dtype=torch.bool)
    active = torch.zeros((1, query), dtype=torch.bool)
    active[0, canvas_rows] = True
    return query_ids, TargetInputs(
        position_ids=scalar_positions(prefix, query),
        attention_mask=mask,
        past_keys=prefill.keys,
        past_values=prefill.values,
        detail_values=detail_values,
        detail_source_indices=None if detail_values is None else torch.tensor([0, 2]),
        detail_active=None if detail_values is None else active,
        output_indices=torch.tensor(canvas_rows),
        code_columns=False,
    )


@torch.inference_mode()
def test_batched_branches_match_their_separate_forwards() -> None:
    model = tiny_model(dtype=torch.float32)
    cache = SimpleNamespace(runtime=SimpleNamespace(model=model))
    generator = torch.Generator().manual_seed(6)
    detail_values = tuple(
        torch.randn((1, 2, DETAIL_SOURCES, 16), generator=generator) for _ in range(3)
    )
    # Like F and U: different prefix lengths, and U's query has one extra leading row.
    rows = [
        branch_inputs(
            model,
            prefix=7,
            query=3,
            canvas_rows=[0, 1],
            detail_values=detail_values,
            seed=1,
        ),
        branch_inputs(
            model, prefix=5, query=4, canvas_rows=[1, 2], detail_values=None, seed=2
        ),
    ]
    separate = [
        GenerationKVCache.target_logits(cache, ids, inputs) for ids, inputs in rows
    ]
    batch = stack_image_branches((CFGBranch.FULL, CFGBranch.UNCONDITIONAL), rows)
    batched = GenerationKVCache.target_logits(cache, batch.query_ids, batch.inputs)
    for row, logits in enumerate(separate):
        torch.testing.assert_close(batched[row : row + 1], logits, rtol=1e-4, atol=1e-5)


def decoded(model: LuminaModel, device: torch.device) -> torch.Tensor:
    prefill = model.decode(
        inputs_embeds=embeddings(9, seed=10).to(device),
        position_ids=scalar_positions(0, 9).to(device),
        attention_mask=causal_mask(9, 9).to(device),
    )
    return model.decode(
        inputs_embeds=embeddings(4, seed=11).to(device),
        position_ids=scalar_positions(9, 4).to(device),
        attention_mask=causal_mask(4, 13).to(device),
        past_keys=prefill.keys,
        past_values=prefill.values,
    ).hidden_states


@pytest.mark.accelerator
@torch.inference_mode()
def test_fused_rope_is_bitwise_equal_and_fused_rmsnorm_close() -> None:
    if not torch.cuda.is_available():
        pytest.skip("the fused kernels need CUDA")
    else:
        pass
    device = torch.device("cuda")
    # The RMSNorm kernel needs head_dim to be a multiple of 32.
    eager = decoded(tiny_model("cuda", head_dim=32), device)
    rope = decoded(
        tiny_model("cuda", head_dim=32, options=DecoderOptions(fused_rope=True)), device
    )
    assert torch.equal(rope, eager)
    norm = decoded(
        tiny_model("cuda", head_dim=32, options=DecoderOptions(fused_rmsnorm=True)),
        device,
    )
    assert (norm.float() - eager.float()).norm() <= 1e-2 * eager.float().norm()
