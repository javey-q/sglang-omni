# SPDX-License-Identifier: Apache-2.0
"""Inputs of one target forward, alone or stacked across the CFG branches of an image step."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sglang_omni.models.lumina.contracts import CFGBranch


@dataclass(frozen=True)
class TargetInputs:
    """Everything of one target forward except its query token ids."""

    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_keys: list[torch.Tensor]
    past_values: list[torch.Tensor]
    detail_values: tuple[torch.Tensor | None, ...] | None
    detail_source_indices: torch.Tensor | None
    detail_active: torch.Tensor | None
    output_indices: torch.Tensor
    code_columns: bool


@dataclass(frozen=True)
class ImageBatch:
    """The CFG branches of one image event stacked into one padded forward."""

    branches: tuple[CFGBranch, ...]
    query_ids: torch.Tensor
    inputs: TargetInputs


def stack_image_branches(
    branches: tuple[CFGBranch, ...],
    rows: list[tuple[torch.Tensor, TargetInputs]],
) -> ImageBatch:
    """Left-pad queries so canvases align, right-pad prefixes, and mask every pad out.

    A padded query row sees only itself, so its softmax stays finite; its output is never read.
    """
    query_width = max(query_ids.shape[1] for query_ids, _ in rows)
    key_width = max(inputs.past_keys[0].shape[-2] for _, inputs in rows)
    pads = [query_width - query_ids.shape[1] for query_ids, _ in rows]
    output_indices = rows[0][1].output_indices + pads[0]
    device = output_indices.device
    mask = torch.zeros(
        (len(rows), 1, query_width, key_width + query_width),
        dtype=torch.bool,
        device=device,
    )
    query_ids, positions, actives, keys, values = [], [], [], [], []
    for row, ((ids, inputs), pad) in enumerate(zip(rows, pads, strict=True)):
        if not torch.equal(inputs.output_indices + pad, output_indices):
            raise ValueError("CFG branches must place the canvas on the same rows")
        else:
            pass
        past_length = inputs.past_keys[0].shape[-2]
        mask[row, 0, pad:, :past_length] = inputs.attention_mask[0, 0, :, :past_length]
        mask[row, 0, pad:, key_width + pad :] = inputs.attention_mask[
            0, 0, :, past_length:
        ]
        pad_rows = torch.arange(pad, device=device)
        mask[row, 0, pad_rows, key_width + pad_rows] = True
        query_ids.append(F.pad(ids, (pad, 0), value=int(ids[0, -1])))
        positions.append(F.pad(inputs.position_ids, (pad, 0)))
        active = (
            torch.zeros_like(ids, dtype=torch.bool)
            if inputs.detail_active is None
            else inputs.detail_active
        )
        actives.append(F.pad(active, (pad, 0), value=False))
        keys.append(inputs.past_keys)
        values.append(inputs.past_values)
    reads = [inputs for _, inputs in rows if inputs.detail_values is not None]
    detail_values = None if not reads else reads[0].detail_values
    detail_source_indices = (
        None
        if not reads
        else torch.stack(
            [
                (
                    reads[0].detail_source_indices
                    if inputs.detail_source_indices is None
                    else inputs.detail_source_indices
                )
                for _, inputs in rows
            ]
        )
    )

    def stacked(per_row: list[list[torch.Tensor]]) -> list[torch.Tensor]:
        return [
            torch.cat(
                [
                    F.pad(layer, (0, 0, 0, key_width - layer.shape[-2]))
                    for layer in layers
                ]
            )
            for layers in zip(*per_row, strict=True)
        ]

    return ImageBatch(
        branches=branches,
        query_ids=torch.cat(query_ids),
        inputs=TargetInputs(
            position_ids=torch.cat(positions, dim=1),
            attention_mask=mask,
            past_keys=stacked(keys),
            past_values=stacked(values),
            detail_values=(
                None
                if detail_values is None
                else tuple(
                    (
                        None
                        if value is None
                        else value.expand(len(rows), -1, -1, -1).contiguous()
                    )
                    for value in detail_values
                )
            ),
            detail_source_indices=detail_source_indices,
            detail_active=None if not reads else torch.cat(actives),
            output_indices=output_indices,
            code_columns=rows[0][1].code_columns,
        ),
    )
