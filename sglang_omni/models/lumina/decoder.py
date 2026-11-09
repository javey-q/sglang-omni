# SPDX-License-Identifier: Apache-2.0
"""One decoder layer as a function of explicit tensors.

It runs the vendored layer's operators in the same order. The decoder's cost is host time rather than GPU time, so skipping module and cache dispatch is what pays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from sglang.kernels.ops.diffusion.rope.rope_rotate_half_bitexact import (
    fused_rope_rotate_half_bitexact,
)
from sglang.kernels.ops.layernorm.rmsnorm_hf import rmsnorm_hf

from sglang_omni.models.lumina.detail import DetailLayer
from sglang_omni.vendor.bard_vl.modeling_bard_vl import BardVLTextDecoderLayer


class LayerWeights(NamedTuple):
    input_norm: torch.Tensor
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    o_proj: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    post_norm: torch.Tensor
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor

    @classmethod
    def from_layer(cls, layer: BardVLTextDecoderLayer) -> LayerWeights:
        attention = layer.self_attn
        projections = (
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
            attention.o_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        )
        assert all(projection.bias is None for projection in projections)
        return cls(
            input_norm=layer.input_layernorm.weight.detach(),
            q_proj=attention.q_proj.weight.detach(),
            k_proj=attention.k_proj.weight.detach(),
            v_proj=attention.v_proj.weight.detach(),
            o_proj=attention.o_proj.weight.detach(),
            q_norm=attention.q_norm.weight.detach(),
            k_norm=attention.k_norm.weight.detach(),
            post_norm=layer.post_attention_layernorm.weight.detach(),
            gate_proj=layer.mlp.gate_proj.weight.detach(),
            up_proj=layer.mlp.up_proj.weight.detach(),
            down_proj=layer.mlp.down_proj.weight.detach(),
        )


@dataclass(frozen=True)
class LayerConstants:
    """Values every layer of the decoder shares."""

    head_dim: int
    key_value_groups: int
    attention_scale: float
    norm_epsilon: float
    # One Triton kernel for each rotary embedding, bit-exact with the eager chain.
    fused_rope: bool = False
    # One kernel for each RMSNorm; its FP32 sum order differs from the eager mean.
    fused_rmsnorm: bool = False


class DetailRead(NamedTuple):
    """What one layer needs to add the Detail residual to its image-target tokens."""

    layer: DetailLayer
    source_values: torch.Tensor
    source_indices: torch.Tensor
    active: torch.Tensor


def rms_norm(
    hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    as_float = hidden_states.to(torch.float32)
    variance = as_float.pow(2).mean(-1, keepdim=True)
    normalized = as_float * torch.rsqrt(variance + epsilon)
    return weight * normalized.to(hidden_states.dtype)


def fused_rms_norm(
    hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    shape = hidden_states.shape
    return rmsnorm_hf(hidden_states.reshape(-1, shape[-1]), weight, epsilon).view(shape)


def layer_rms_norm(
    constants: LayerConstants, hidden_states: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    if constants.fused_rmsnorm:
        return fused_rms_norm(hidden_states, weight, constants.norm_epsilon)
    else:
        return rms_norm(hidden_states, weight, constants.norm_epsilon)


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(
    constants: LayerConstants,
    heads: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Rotate [batch, length, heads, head_dim] activations; returns [batch, heads, length, head_dim]."""
    if constants.fused_rope:
        table_shape = (-1, cos.shape[-1])
        return fused_rope_rotate_half_bitexact(
            heads, cos.reshape(table_shape), sin.reshape(table_shape)
        ).transpose(1, 2)
    else:
        heads = heads.transpose(1, 2)
        head_cos = cos.unsqueeze(1)
        head_sin = sin.unsqueeze(1)
        return (heads * head_cos) + (rotate_half(heads) * head_sin)


def decoder_layer_forward(
    weights: LayerWeights,
    constants: LayerConstants,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key: torch.Tensor | None,
    past_value: torch.Tensor | None,
    detail_read: DetailRead | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the layer's hidden states and its full K and V, past tokens first."""
    batch_size, length = hidden_states.shape[:2]
    head_shape = (batch_size, length, -1, constants.head_dim)
    normalized = layer_rms_norm(constants, hidden_states, weights.input_norm)
    query = apply_rope(
        constants,
        layer_rms_norm(
            constants,
            F.linear(normalized, weights.q_proj).view(head_shape),
            weights.q_norm,
        ),
        cos,
        sin,
    )
    key = apply_rope(
        constants,
        layer_rms_norm(
            constants,
            F.linear(normalized, weights.k_proj).view(head_shape),
            weights.k_norm,
        ),
        cos,
        sin,
    )
    value = F.linear(normalized, weights.v_proj).view(head_shape).transpose(1, 2)
    if past_key is not None:
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)
    else:
        # Compiled graphs guard on strides, so a first K and V must already look concatenated.
        key = key.contiguous()
        value = value.contiguous()

    # A boolean mask rules out the grouped SDPA path, so K and V repeat per query head.
    groups = constants.key_value_groups
    key_heads, key_length = key.shape[1:3]
    repeated_shape = (batch_size, key_heads * groups, key_length, constants.head_dim)
    expanded_shape = (batch_size, key_heads, groups, key_length, constants.head_dim)
    attended = F.scaled_dot_product_attention(
        query,
        key[:, :, None, :, :].expand(expanded_shape).reshape(repeated_shape),
        value[:, :, None, :, :].expand(expanded_shape).reshape(repeated_shape),
        attn_mask=attention_mask,
        dropout_p=0.0,
        scale=constants.attention_scale,
        is_causal=False,
    )
    attended = attended.transpose(1, 2).contiguous().reshape(batch_size, length, -1)
    hidden_semantic = hidden_states + F.linear(attended.contiguous(), weights.o_proj)

    if detail_read is not None:
        indices = detail_read.source_indices
        if indices.dim() == 1:
            source_key = past_key.index_select(-2, indices)
        else:
            # A batch of branches reads each row's own source positions.
            gather_shape = (-1, past_key.shape[1], -1, past_key.shape[-1])
            source_key = past_key.gather(
                2, indices[:, None, :, None].expand(gather_shape)
            )
        hidden_semantic = hidden_semantic + detail_read.layer(
            query_post_rope=query,
            source_key_post_rope=source_key,
            source_values=detail_read.source_values.to(
                source_key.device, source_key.dtype
            ),
            hidden_semantic=hidden_semantic,
            active=detail_read.active,
        )
    else:
        pass

    post_normalized = layer_rms_norm(constants, hidden_semantic, weights.post_norm)
    gated = F.silu(F.linear(post_normalized, weights.gate_proj))
    mlp_output = F.linear(
        gated * F.linear(post_normalized, weights.up_proj), weights.down_proj
    )
    return hidden_semantic + mlp_output, key, value
