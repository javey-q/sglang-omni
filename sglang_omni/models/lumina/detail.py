# SPDX-License-Identifier: Apache-2.0
"""Detail cross attention: image-target queries read source images' InsightTok codes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.vendor.bard_vl.configuration_bard_vl import BardVLTextConfig
from sglang_omni.vendor.bard_vl.modeling_bard_vl import BardVLTextAttention


@dataclass(frozen=True)
class StaticDetailMemory:
    """Branch-independent Detail V aligned to canonical source-image positions."""

    source_canonical_indices: torch.Tensor
    values_by_layer: tuple[torch.Tensor | None, ...]


class AdaptiveDetailResampler(nn.Module):
    """Embed and resample each H/16 detail grid to its H/32 ViT grid."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.input_projection = nn.Linear(hidden_size, hidden_size)
        self.scale_projection = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward_grid(
        self,
        embeddings: torch.Tensor,
        *,
        source_height: int,
        source_width: int,
        target_height: int,
        target_width: int,
    ) -> torch.Tensor:
        grid = (
            embeddings.view(source_height, source_width, self.hidden_size)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        if target_height <= source_height and target_width <= source_width:
            resized = F.interpolate(
                grid.float(), size=(target_height, target_width), mode="area"
            )
        else:
            resized = F.interpolate(
                grid.float(),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        resized = (
            resized.to(embeddings.dtype)
            .squeeze(0)
            .permute(1, 2, 0)
            .reshape(-1, self.hidden_size)
        )
        projected = self.input_projection(self.norm(resized))
        scale = embeddings.new_tensor(
            [
                math.log(source_height / target_height),
                math.log(source_width / target_width),
            ]
        ).view(1, 2)
        return projected + self.scale_projection(scale).to(projected.dtype)


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_mrope(
    query: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Apply the decoder's MRoPE broadcast to normalized Q projections."""
    cos, sin = position_embeddings
    return (query * cos.unsqueeze(1)) + (rotate_half(query) * sin.unsqueeze(1))


class PostNormQueryCapture:
    """Call-scoped capture of the normalized Q of the main attention."""

    def __init__(self, attention: BardVLTextAttention) -> None:
        self.is_active = False
        self.query: torch.Tensor | None = None
        attention.q_norm.register_forward_hook(self.capture_query)

    def capture_query(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if self.is_active:
            self.query = output
        else:
            pass

    def begin(self) -> None:
        self.is_active = True

    def finish(
        self, position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        assert self.query is not None
        query = self.query.transpose(1, 2)
        self.is_active = False
        self.query = None
        return apply_mrope(query, position_embeddings)


class DetailLayer(nn.Module):
    """Post-MRoPE read of a source image's Detail values, as a gated residual."""

    def __init__(self, config: BardVLTextConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.detail_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.detail_v_proj = nn.Linear(
            hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.detail_o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, hidden_size, bias=True
        )
        self.gate_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.detail_gate = nn.Linear(hidden_size, 1, bias=True)

    def project_values(self, detail_base: torch.Tensor) -> torch.Tensor:
        """Project static source Detail values once per set of committed images."""
        batch_size, sequence_length, _ = detail_base.shape
        values = self.detail_v_proj(self.detail_norm(detail_base))
        return values.view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self,
        *,
        query_post_rope: torch.Tensor,
        source_key_post_rope: torch.Tensor,
        source_values: torch.Tensor,
        hidden_semantic: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        attended = F.scaled_dot_product_attention(
            query_post_rope,
            source_key_post_rope,
            source_values,
            scale=self.scaling,
            enable_gqa=query_post_rope.shape[1] != source_key_post_rope.shape[1],
        )
        attended = (
            attended.transpose(1, 2)
            .reshape(query_post_rope.shape[0], query_post_rope.shape[2], -1)
            .contiguous()
        )
        gate = torch.sigmoid(self.detail_gate(self.gate_norm(hidden_semantic)))
        residual = gate * self.detail_o_proj(attended)
        return residual * active.unsqueeze(-1).to(residual.dtype)
