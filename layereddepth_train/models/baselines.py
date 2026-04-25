from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .backbones import build_depth_core


def _layer_id_map(layer_index: Tensor, height: int, width: int, num_layers: int) -> Tensor:
    values = layer_index.float()
    if num_layers > 1:
        values = values / float(num_layers - 1)
    return values[:, None, None, None].expand(-1, 1, height, width)


def gather_layer(depth_stack: Tensor, layer_index: Tensor) -> Tensor:
    index = layer_index[:, None, None, None].expand(-1, 1, *depth_stack.shape[-2:])
    return torch.gather(depth_stack, dim=1, index=index)


class MultiHeadBaseline(nn.Module):
    """Paper baseline: RGB input, all layer depths in one forward pass."""

    def __init__(
        self,
        num_layers: int,
        core_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.core = build_depth_core(3, num_layers, core_config, positive_output=True)

    def forward(self, image: Tensor, layer_index: Tensor | None = None) -> Tensor:
        depth_stack = self.core(image)
        if layer_index is None:
            return depth_stack
        return gather_layer(depth_stack, layer_index)

    @torch.no_grad()
    def predict_all_layers(self, image: Tensor) -> Tensor:
        return self.forward(image)


class LayerIndexConcatBaseline(nn.Module):
    """Paper baseline: RGB plus layer-index channel predicts one layer."""

    def __init__(
        self,
        num_layers: int,
        core_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.core = build_depth_core(4, 1, core_config, positive_output=True)

    def forward(self, image: Tensor, layer_index: Tensor) -> Tensor:
        b, _, h, w = image.shape
        layer_map = _layer_id_map(layer_index, h, w, self.num_layers).to(image.device)
        return self.core(torch.cat([image, layer_map], dim=1))

    @torch.no_grad()
    def predict_all_layers(self, image: Tensor) -> Tensor:
        outputs = []
        for idx in range(self.num_layers):
            layer_index = torch.full((image.shape[0],), idx, device=image.device, dtype=torch.long)
            outputs.append(self.forward(image, layer_index))
        return torch.cat(outputs, dim=1)


class RecurrentBaseline(nn.Module):
    """Paper baseline: RGB plus previous-layer depth predicts current layer."""

    def __init__(
        self,
        num_layers: int,
        core_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.core = build_depth_core(4, 1, core_config, positive_output=True)

    def forward(self, image: Tensor, previous_depth: Tensor) -> Tensor:
        if previous_depth.ndim == 3:
            previous_depth = previous_depth.unsqueeze(1)
        return self.core(torch.cat([image, previous_depth], dim=1))

    @torch.no_grad()
    def predict_all_layers(self, image: Tensor) -> Tensor:
        previous = torch.zeros(
            image.shape[0], 1, image.shape[-2], image.shape[-1], device=image.device
        )
        outputs = []
        for _ in range(self.num_layers):
            previous = self.forward(image, previous)
            outputs.append(previous)
        return torch.cat(outputs, dim=1)


def build_paper_baseline(config: Mapping[str, Any]) -> nn.Module:
    num_layers = int(config.get("num_layers", 7))
    architecture = str(config.get("architecture", "index_concat")).lower()
    core_config = config.get("core", None)

    if architecture in {"multi_head", "multihead"}:
        return MultiHeadBaseline(num_layers=num_layers, core_config=core_config)
    if architecture in {"index_concat", "layer_index_concat"}:
        return LayerIndexConcatBaseline(num_layers=num_layers, core_config=core_config)
    if architecture == "recurrent":
        return RecurrentBaseline(num_layers=num_layers, core_config=core_config)
    raise ValueError(f"Unknown paper baseline architecture: {architecture}")

