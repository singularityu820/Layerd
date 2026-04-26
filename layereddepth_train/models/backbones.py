from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class TinyDepthCore(nn.Module):
    """Small U-Net used for smoke tests and ablations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 48,
        min_depth: float = 1e-3,
        positive_output: bool = True,
    ) -> None:
        super().__init__()
        self.min_depth = min_depth
        self.positive_output = positive_output

        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.mid = ConvBlock(base_channels * 4, base_channels * 4)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        mid = self.mid(F.avg_pool2d(e3, 2))

        d3 = F.interpolate(mid, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        y = self.head(d1)
        if self.positive_output:
            y = F.softplus(y) + self.min_depth
        return y


class NewCRFsAdapter(nn.Module):
    """Adapter around the official NeWCRFs implementation.

    The paper uses NeWCRFs as the backbone. This adapter keeps that dependency
    optional so the rest of the training code can still be developed without it.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_depth: float = 10.0,
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        kwargs = dict(kwargs or {})
        kwargs.setdefault("version", "large07")
        kwargs.setdefault("inv_depth", False)
        kwargs.setdefault("max_depth", max_depth)
        kwargs.setdefault("pretrained", None)

        try:
            _ensure_local_newcrfs_on_path()
            from newcrfs.networks.NewCRFDepth import NewCRFDepth
        except ImportError as exc:
            raise ImportError(
                "NeWCRFs is not importable. Make sure third_party/NeWCRFs exists "
                "and install the NeWCRFs dependencies from requirements.txt. "
                f"Original import error: {exc}"
            ) from exc

        self.model = NewCRFDepth(**kwargs)
        if in_channels != 3:
            _patch_first_rgb_conv(self.model, in_channels)
        if out_channels != 1:
            _patch_depth_head(self.model, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        y = self.model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        if y.ndim == 3:
            y = y.unsqueeze(1)
        return y


def _ensure_local_newcrfs_on_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    local_repo = project_root / "third_party" / "NeWCRFs"
    if local_repo.exists():
        repo_path = str(local_repo)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)


def _patch_first_rgb_conv(module: nn.Module, in_channels: int) -> None:
    for parent in module.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Conv2d) and child.in_channels == 3:
                replacement = nn.Conv2d(
                    in_channels,
                    child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias is not None,
                    padding_mode=child.padding_mode,
                )
                with torch.no_grad():
                    replacement.weight[:, :3].copy_(child.weight)
                    if in_channels > 3:
                        mean_weight = child.weight.mean(dim=1, keepdim=True)
                        replacement.weight[:, 3:].copy_(mean_weight.expand(-1, in_channels - 3, -1, -1))
                    if child.bias is not None:
                        replacement.bias.copy_(child.bias)
                setattr(parent, name, replacement)
                return
    raise RuntimeError("Could not find the first RGB Conv2d to adapt for NeWCRFs.")


def _patch_depth_head(module: nn.Module, out_channels: int) -> None:
    head = getattr(module, "disp_head1", None)
    conv = getattr(head, "conv1", None)
    if not isinstance(conv, nn.Conv2d):
        raise RuntimeError("Could not find NeWCRFs disp_head1.conv1 for multi-layer output.")

    replacement = nn.Conv2d(
        conv.in_channels,
        out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        replacement.weight.copy_(conv.weight.repeat(out_channels, 1, 1, 1))
        if conv.bias is not None:
            replacement.bias.copy_(conv.bias.repeat(out_channels))
    head.conv1 = replacement


def build_depth_core(
    in_channels: int,
    out_channels: int,
    config: Mapping[str, Any] | None = None,
    positive_output: bool = True,
) -> nn.Module:
    config = dict(config or {})
    name = str(config.pop("name", "tiny")).lower()
    if name == "tiny":
        return TinyDepthCore(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=int(config.pop("base_channels", 48)),
            min_depth=float(config.pop("min_depth", 1e-3)),
            positive_output=positive_output,
        )
    if name == "newcrfs":
        return NewCRFsAdapter(
            in_channels=in_channels,
            out_channels=out_channels,
            max_depth=float(config.pop("max_depth", 10.0)),
            kwargs=config.pop("kwargs", None),
        )
    raise ValueError(f"Unknown depth core: {name}")
