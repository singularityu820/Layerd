from __future__ import annotations

import torch
from torch import Tensor


def snapped_layered_depth(depth: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Fill missing layer-k depth with the previous snapped layer where possible."""
    squeeze = depth.ndim == 3
    if squeeze:
        depth = depth.unsqueeze(0)
        valid = valid.unsqueeze(0)

    snapped = depth.clone()
    snapped_valid = valid.bool().clone()

    for layer_idx in range(1, snapped.shape[1]):
        missing = ~snapped_valid[:, layer_idx] & snapped_valid[:, layer_idx - 1]
        snapped[:, layer_idx] = torch.where(
            missing, snapped[:, layer_idx - 1], snapped[:, layer_idx]
        )
        snapped_valid[:, layer_idx] = snapped_valid[:, layer_idx] | missing

    if squeeze:
        snapped = snapped.squeeze(0)
        snapped_valid = snapped_valid.squeeze(0)
    return snapped, snapped_valid


def silog_loss(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor | None = None,
    variance_focus: float = 0.85,
    eps: float = 1e-6,
    scale: float = 10.0,
) -> Tensor:
    """Scale-invariant logarithmic loss used by many metric depth models."""
    prediction = prediction.clamp_min(eps)
    target = target.clamp_min(eps)
    log_diff = torch.log(prediction) - torch.log(target)

    if valid is not None:
        valid = valid.bool()
        if valid.ndim == log_diff.ndim - 1:
            valid = valid.unsqueeze(1)
        log_diff = log_diff[valid.expand_as(log_diff)]

    if log_diff.numel() == 0:
        return prediction.sum() * 0.0

    mse = torch.mean(log_diff.square())
    bias = torch.mean(log_diff).square()
    return scale * torch.sqrt(torch.clamp(mse - variance_focus * bias, min=eps))


def masked_mse(prediction: Tensor, target: Tensor, valid: Tensor | None = None) -> Tensor:
    diff = prediction - target
    if valid is None:
        return diff.square().mean()
    if valid.ndim == diff.ndim - 1:
        valid = valid.unsqueeze(1)
    valid = valid.bool().expand_as(diff)
    if valid.sum() == 0:
        return prediction.sum() * 0.0
    return diff[valid].square().mean()

