# SPDX-License-Identifier: Apache-2.0
"""The Lumina decoder function must stay bitwise equal to the vendored decoder."""

import pytest
import torch
from torch._dynamo.utils import counters
from transformers import DynamicCache

from sglang_omni.models.lumina.contracts import TokenRegistry
from sglang_omni.models.lumina.modeling import LuminaModel
from sglang_omni.vendor.bard_vl.configuration_bard_vl import BardVLConfig
from sglang_omni.vendor.bard_vl.modeling_bard_vl import BardVLForConditionalGeneration

HIDDEN_SIZE = 64
VOCAB_SIZE = 32
LAYERS = 3


def tiny_model(device: str = "cpu") -> LuminaModel:
    torch.manual_seed(0)
    config = BardVLConfig(
        text_config={
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": 128,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
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
    base_model = base_model.to(device=device, dtype=torch.bfloat16).eval()
    return (
        LuminaModel(base_model, registry=TokenRegistry(), detail_start_layer=0)
        .to(torch.bfloat16)
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
    logits = model.token_logits(prefill.hidden_states, every_row)
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
    logits = model.token_logits(read.hidden_states, torch.arange(target))
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
    logits = model.token_logits(output.hidden_states, torch.arange(length))
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
