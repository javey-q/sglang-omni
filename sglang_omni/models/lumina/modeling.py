# SPDX-License-Identifier: Apache-2.0
"""Lumina model: decoder layers with a Detail residual, plus its image codec."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from transformers import AutoProcessor, ProcessorMixin

from sglang_omni.models.lumina.contracts import GENERATED_GRID_FACTOR, TokenRegistry
from sglang_omni.models.lumina.decoder import (
    DetailRead,
    LayerConstants,
    LayerWeights,
    decoder_layer_forward,
    layer_rms_norm,
)
from sglang_omni.models.lumina.detail import (
    AdaptiveDetailResampler,
    DetailLayer,
    StaticDetailMemory,
)
from sglang_omni.vendor.bard_vl.modeling_bard_vl import BardVLForConditionalGeneration
from sglang_omni.vendor.insight_tok.insight_tok import InsightTok

INSIGHT_TOK_CODEBOOK_SIZE = 16384
INSIGHT_TOK_CODE_CHANNELS = 256
SUPPORTED_INPUT_IMAGE_ATTENTION = "full_within_image"
# Real proportions: compiled graphs also guard on how queries compare with keys.
WARM_UP_PREFIX_LENGTHS = (64, 512)
WARM_UP_TEXT_LENGTHS = (1, 4)
WARM_UP_IMAGE_LENGTHS = (65, 193)


@dataclass(frozen=True, kw_only=True)
class DecoderOptions:
    """Speedups of the plain decoder; with all of them off it stays bitwise equal to the vendored one."""

    # Image logits over the code columns only: the role mask allows nothing else, so only rounding changes.
    code_lm_head: bool = False
    fused_rope: bool = False
    fused_rmsnorm: bool = False
    # All CFG branches of an image step in one padded forward.
    cfg_batch: bool = False


@dataclass(frozen=True)
class DecoderOutput:
    hidden_states: torch.Tensor
    keys: list[torch.Tensor]
    values: list[torch.Tensor]


class LuminaModel(nn.Module):
    """The vendored decoder with event attention masks and a Detail side memory."""

    def __init__(
        self,
        base_model: BardVLForConditionalGeneration,
        *,
        registry: TokenRegistry,
        detail_start_layer: int,
        options: DecoderOptions,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.registry = registry
        self.options = options
        embedding = base_model.get_input_embeddings().weight
        self.detail_resampler = AdaptiveDetailResampler(embedding.shape[1])
        self.detail_resampler.to(device=embedding.device, dtype=embedding.dtype)
        text_config = base_model.config.text_config
        for layer_index, layer in enumerate(self.decoder_layers):
            layer.detail_layer_index = layer_index
            if layer_index >= detail_start_layer:
                layer.add_module(
                    "detail",
                    DetailLayer(text_config).to(
                        device=embedding.device, dtype=embedding.dtype
                    ),
                )
            else:
                layer.add_module("detail", None)
        self.layer_weights = [
            LayerWeights.from_layer(layer) for layer in self.decoder_layers
        ]
        self.layer_forward = decoder_layer_forward
        attention = self.decoder_layers[0].self_attn
        self.layer_constants = LayerConstants(
            head_dim=attention.head_dim,
            key_value_groups=attention.num_key_value_groups,
            attention_scale=attention.scaling,
            norm_epsilon=self.decoder_layers[0].input_layernorm.variance_epsilon,
            fused_rope=options.fused_rope,
            fused_rmsnorm=options.fused_rmsnorm,
        )
        head = base_model.lm_head.weight
        self.code_head_weight = head.detach()[
            registry.code_start_id : registry.code_end_id + 1
        ]

    @property
    def decoder_layers(self) -> nn.ModuleList:
        return self.base_model.model.language_model.layers

    def get_input_embeddings(self) -> nn.Embedding:
        return self.base_model.get_input_embeddings()

    def decode(
        self,
        *,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_keys: list[torch.Tensor] | None = None,
        past_values: list[torch.Tensor] | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        detail_values: tuple[torch.Tensor | None, ...] | None = None,
        detail_source_indices: torch.Tensor | None = None,
        detail_active: torch.Tensor | None = None,
    ) -> DecoderOutput:
        """Run the stack; each layer's K and V come back with its past tokens first."""
        cos, sin = self.base_model.language_model.rotary_emb(
            inputs_embeds, position_ids
        )
        hidden_states = inputs_embeds
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for layer_index, weights in enumerate(self.layer_weights):
            source_values = (
                None if detail_values is None else detail_values[layer_index]
            )
            hidden_states, key, value = self.layer_forward(
                weights,
                self.layer_constants,
                hidden_states,
                cos,
                sin,
                attention_mask,
                None if past_keys is None else past_keys[layer_index],
                None if past_values is None else past_values[layer_index],
                (
                    None
                    if source_values is None
                    else DetailRead(
                        self.decoder_layers[layer_index].detail,
                        source_values,
                        detail_source_indices,
                        detail_active,
                    )
                ),
            )
            keys.append(key)
            values.append(value)
            if deepstack_visual_embeds is not None and layer_index < len(
                deepstack_visual_embeds
            ):
                hidden_states[visual_pos_masks, :] = (
                    hidden_states[visual_pos_masks, :]
                    + deepstack_visual_embeds[layer_index]
                )
            else:
                pass
        return DecoderOutput(hidden_states, keys, values)

    @torch.inference_mode()
    def enable_compiled_decoder(self) -> None:
        """Compile the layer function and every path through it before any request.

        Compiling here also makes a broken CUDA toolchain fail at startup rather than on the first request.
        """
        # note (qiujiawei): weights are inputs of the layer function, so all layers share one graph per path.
        inductor_config.emulate_precision_casts = True
        # Autotuning times reduction block sizes, and the size changes rounding between fresh caches. Dynamo resets inductor_config.deterministic after every frame, so it cannot pin them.
        inductor_config.test_configs.force_filter_reduction_configs = True
        self.layer_forward = torch.compile(decoder_layer_forward, dynamic=True)
        weight = self.get_input_embeddings().weight

        def run(
            length: int,
            past: DecoderOutput | None,
            detail_values: tuple[torch.Tensor | None, ...] | None = None,
            detail_source_indices: torch.Tensor | None = None,
        ) -> DecoderOutput:
            past_length = 0 if past is None else past.keys[0].shape[-2]
            positions = torch.arange(past_length, past_length + length).to(
                weight.device
            )
            return self.decode(
                inputs_embeds=weight.new_zeros((1, length, weight.shape[1])),
                position_ids=positions.view(1, 1, -1).expand(3, 1, -1),
                attention_mask=torch.ones(
                    (1, 1, length, past_length + length),
                    dtype=torch.bool,
                    device=weight.device,
                ),
                past_keys=None if past is None else past.keys,
                past_values=None if past is None else past.values,
                detail_values=detail_values,
                detail_source_indices=detail_source_indices,
                detail_active=(
                    None
                    if detail_values is None
                    else torch.ones((1, length), dtype=torch.bool, device=weight.device)
                ),
            )

        for prefix_length in WARM_UP_PREFIX_LENGTHS:
            prefill = run(prefix_length, None)
            for length in (*WARM_UP_TEXT_LENGTHS, *WARM_UP_IMAGE_LENGTHS):
                run(length, prefill)
            source_indices = torch.arange(prefix_length // 2, device=weight.device)
            detail_values = tuple(
                None if layer.detail is None else value.index_select(2, source_indices)
                for layer, value in zip(
                    self.decoder_layers, prefill.values, strict=True
                )
            )
            for length in WARM_UP_IMAGE_LENGTHS:
                run(length, prefill, detail_values, source_indices)

    def token_logits(
        self,
        hidden_states: torch.Tensor,
        output_indices: torch.Tensor,
        *,
        code_columns: bool,
    ) -> torch.Tensor:
        """Project the chosen rows; the head's input shape decides BF16 rounding.

        With code_columns the logits cover only the image codes, column 0 being the first code.
        """
        normalized = layer_rms_norm(
            self.layer_constants,
            hidden_states,
            self.base_model.language_model.norm.weight,
        )
        selected = normalized.index_select(1, output_indices)
        weight = (
            self.code_head_weight if code_columns else self.base_model.lm_head.weight
        )
        return F.linear(selected, weight)

    def detail_base(
        self,
        *,
        input_ids: torch.Tensor,
        source_insight_codes: torch.Tensor,
        source_insight_grid_hw: torch.Tensor,
        detail_image_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Resampled code embeddings at each image's native payload positions."""
        embeddings = self.get_input_embeddings()(
            source_insight_codes + self.registry.code_start_id
        )
        dense = embeddings.new_zeros((*input_ids.shape, embeddings.shape[-1]))
        cursor = 0
        for image_id, (height, width) in enumerate(source_insight_grid_hw.tolist()):
            count = height * width
            positions = torch.nonzero(detail_image_ids == image_id, as_tuple=False)
            target_height, target_width = height // 2, width // 2
            assert positions.shape[0] == target_height * target_width
            dense[positions[:, 0], positions[:, 1]] = (
                self.detail_resampler.forward_grid(
                    embeddings[cursor : cursor + count],
                    source_height=height,
                    source_width=width,
                    target_height=target_height,
                    target_width=target_width,
                )
            )
            cursor += count
        return dense

    def build_static_detail_memory(
        self,
        *,
        input_ids: torch.Tensor,
        source_insight_codes: torch.Tensor,
        source_insight_grid_hw: torch.Tensor,
        detail_image_ids: torch.Tensor,
    ) -> StaticDetailMemory:
        """Build branch-shareable Detail V of every committed native image."""
        detail_base = self.detail_base(
            input_ids=input_ids,
            source_insight_codes=source_insight_codes,
            source_insight_grid_hw=source_insight_grid_hw,
            detail_image_ids=detail_image_ids,
        )
        source_indices = torch.nonzero(
            detail_image_ids[0] >= 0, as_tuple=False
        ).flatten()
        return StaticDetailMemory(
            source_canonical_indices=source_indices,
            values_by_layer=tuple(
                (
                    None
                    if layer.detail is None
                    else layer.detail.project_values(detail_base).index_select(
                        2, source_indices
                    )
                )
                for layer in self.decoder_layers
            ),
        )


@dataclass
class LuminaRuntime:
    """The loaded model, its processor, and the image codec of one engine process."""

    model: LuminaModel
    processor: ProcessorMixin
    insight_tok: InsightTok

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def encode_image(self, image: Image.Image, height: int, width: int) -> list[int]:
        """Encode pixels to raw InsightTok codes, one per 16x16 pixels."""
        resized = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).mul(2.0).sub(1.0)
        _, _, (_, _, indices) = self.insight_tok.encode(
            pixels.to(device=self.device, dtype=torch.float32)
        )
        return indices.reshape(-1).tolist()

    @torch.inference_mode()
    def decode_codes(self, raw_ids: list[int], height: int, width: int) -> Image.Image:
        """Decode raw InsightTok codes to an RGB image."""
        indices = torch.tensor(raw_ids, dtype=torch.long, device=self.device)
        pixels = self.insight_tok.decode_code(
            indices,
            shape=(
                1,
                INSIGHT_TOK_CODE_CHANNELS,
                height // GENERATED_GRID_FACTOR,
                width // GENERATED_GRID_FACTOR,
            ),
            channel_first=True,
        )
        pixels = pixels.float().cpu().clamp(-1.0, 1.0)
        pixels = pixels.add(1.0).mul(127.5).round().byte()
        return Image.fromarray(pixels.squeeze(0).permute(1, 2, 0).numpy())


def load_lumina_runtime(
    model_path: str,
    device: torch.device,
    *,
    compile_decoder: bool,
    options: DecoderOptions,
) -> LuminaRuntime:
    """Load one exported Lumina directory and pin the process-wide numerics it needs."""
    if compile_decoder and (options.fused_rope or options.fused_rmsnorm):
        raise ValueError(
            "compile_decoder fuses RoPE and RMSNorm itself; turn off fused_rope and fused_rmsnorm"
        )
    else:
        pass
    model_directory = Path(model_path)
    # note (qiujiawei): torch's cuDNN SDPA is not reproducible here; multi-image turns differ even within one process.
    torch.backends.cuda.enable_cudnn_sdp(False)
    # BF16 GEMM reduction would compound across decoder layers on short events.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    config = json.loads((model_directory / "config.json").read_text(encoding="utf-8"))
    lumina_config = config["lumina"]
    if lumina_config["input_image_attention"] != SUPPORTED_INPUT_IMAGE_ATTENTION:
        raise ValueError(
            f"Lumina serves only {SUPPORTED_INPUT_IMAGE_ATTENTION} input image attention"
        )
    else:
        pass
    base_model = BardVLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, _attn_implementation="sdpa"
    ).to(device)
    model = LuminaModel(
        base_model,
        registry=TokenRegistry(),
        detail_start_layer=lumina_config["detail_start_layer"],
        options=options,
    ).to(device)
    detail_state = load_file(model_directory / "detail.safetensors", device=str(device))
    loaded = model.load_state_dict(detail_state, strict=False)
    # Base-model tensors are loaded already, so only Detail names may be reported here.
    missing_detail = [
        name
        for name in loaded.missing_keys
        if ".detail." in name or name.startswith("detail_resampler")
    ]
    if loaded.unexpected_keys or missing_detail:
        raise ValueError(
            f"Detail weights do not match: missing {missing_detail[:8]!r}, "
            f"unexpected {loaded.unexpected_keys[:8]!r}"
        )
    else:
        pass
    model.eval()
    if compile_decoder:
        model.enable_compiled_decoder()
    else:
        pass
    insight_tok = InsightTok(
        dd_config_key="default",
        codebook_size=INSIGHT_TOK_CODEBOOK_SIZE,
        codebook_embed_dim=INSIGHT_TOK_CODE_CHANNELS,
        commit_loss_beta=0.02,
        codebook_l2_norm=True,
        codebook_show_usage=True,
        quant_disable_autocast=False,
    )
    insight_tok.load_state_dict(load_file(model_directory / "insight_tok.safetensors"))
    insight_tok.eval().to(device=device, dtype=torch.float32)
    return LuminaRuntime(
        model=model,
        processor=AutoProcessor.from_pretrained(model_path),
        insight_tok=insight_tok,
    )
