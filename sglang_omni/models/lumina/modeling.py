# SPDX-License-Identifier: Apache-2.0
"""Lumina model: decoder layers with a Detail residual, plus its image codec."""

from __future__ import annotations

import json
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from transformers import AutoProcessor, DynamicCache, ProcessorMixin

from sglang_omni.models.lumina.contracts import GENERATED_GRID_FACTOR, TokenRegistry
from sglang_omni.models.lumina.detail import (
    AdaptiveDetailResampler,
    DetailLayer,
    PostNormQueryCapture,
    StaticDetailMemory,
)
from sglang_omni.vendor.bard_vl.modeling_bard_vl import (
    BardVLForConditionalGeneration,
    BardVLTextDecoderLayer,
)
from sglang_omni.vendor.insight_tok.insight_tok import InsightTok

INSIGHT_TOK_CODEBOOK_SIZE = 16384
INSIGHT_TOK_CODE_CHANNELS = 256
SUPPORTED_INPUT_IMAGE_ATTENTION = "full_within_image"


def decoder_forward(
    self: BardVLTextDecoderLayer,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values: DynamicCache | None = None,
    use_cache: bool | None = False,
    cache_position: torch.Tensor | None = None,
    cached_detail_values: tuple[torch.Tensor | None, ...] | None = None,
    cached_detail_source_indices: torch.Tensor | None = None,
    cached_detail_active: torch.Tensor | None = None,
) -> torch.Tensor:
    """The decoder layer with the Detail residual after self attention."""
    is_detail_active = self.detail is not None and cached_detail_values is not None
    if is_detail_active:
        self.detail_query_capture.begin()
    else:
        pass
    attended, _ = self.self_attn(
        hidden_states=self.input_layernorm(hidden_states),
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )
    hidden_semantic = hidden_states + attended
    if is_detail_active:
        layer_index = self.detail_layer_index
        source_key = past_key_values.layers[layer_index].keys
        source_key = source_key.index_select(
            -2, cached_detail_source_indices.to(source_key.device)
        )
        hidden_semantic = hidden_semantic + self.detail(
            query_post_rope=self.detail_query_capture.finish(position_embeddings),
            source_key_post_rope=source_key,
            source_values=cached_detail_values[layer_index].to(
                source_key.device, source_key.dtype
            ),
            hidden_semantic=hidden_semantic,
            active=cached_detail_active,
        )
    else:
        pass
    output = self.mlp(self.post_attention_layernorm(hidden_semantic))
    return hidden_semantic + output


class LuminaModel(nn.Module):
    """The vendored decoder with event attention masks and a Detail side memory."""

    def __init__(
        self,
        base_model: BardVLForConditionalGeneration,
        *,
        registry: TokenRegistry,
        detail_start_layer: int,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.registry = registry
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
                layer.detail_query_capture = PostNormQueryCapture(layer.self_attn)
            else:
                layer.add_module("detail", None)
            layer.forward = types.MethodType(decoder_forward, layer)

    @property
    def decoder_layers(self) -> nn.ModuleList:
        return self.base_model.model.language_model.layers

    def get_input_embeddings(self) -> nn.Embedding:
        return self.base_model.get_input_embeddings()

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


def load_lumina_runtime(model_path: str, device: torch.device) -> LuminaRuntime:
    """Load one exported Lumina directory and pin the process-wide numerics it needs."""
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
