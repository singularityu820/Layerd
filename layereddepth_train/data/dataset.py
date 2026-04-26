from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from layereddepth_train.losses import snapped_layered_depth


def _resolve(path: str | Path, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def _load_image(path: Path, image_size: tuple[int, int] | None) -> Tensor:
    image = Image.open(path).convert("RGB")
    if image_size is not None:
        image = image.resize((image_size[1], image_size[0]), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _load_array(
    path: Path,
    image_size: tuple[int, int] | None,
    scale: float = 1.0,
    is_mask: bool = False,
) -> Tensor:
    if path.suffix.lower() == ".npy":
        array = np.load(path).astype(np.float32)
    else:
        image = Image.open(path)
        if image_size is not None:
            resample = Image.NEAREST if is_mask else Image.BILINEAR
            image = image.resize((image_size[1], image_size[0]), resample)
        array = np.asarray(image, dtype=np.float32)

    if array.ndim == 3:
        array = array[..., 0]
    if image_size is not None and path.suffix.lower() == ".npy":
        image = Image.fromarray(array)
        resample = Image.NEAREST if is_mask else Image.BILINEAR
        image = image.resize((image_size[1], image_size[0]), resample)
        array = np.asarray(image, dtype=np.float32)

    if is_mask:
        array = (array > 0).astype(np.float32)
    else:
        array = array / scale
    return torch.from_numpy(array).float()


class ManifestLayeredDepthDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        manifest_path: str | Path,
        root: str | Path | None = None,
        num_layers: int = 7,
        image_size: tuple[int, int] | None = None,
        depth_scale: float = 1.0,
        use_snapped_depth: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = Path(root) if root is not None else self.manifest_path.parent
        self.num_layers = num_layers
        self.image_size = image_size
        self.depth_scale = depth_scale
        self.use_snapped_depth = use_snapped_depth

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                "LayeredDepth manifest not found: "
                f"{self.manifest_path}\n"
                "For a local smoke test, run:\n"
                "  python -m layereddepth_train.train --config configs/tiny_smoke.yaml\n"
                "For real training, place your dataset under data/layereddepth_syn "
                "or update data.root in the config, then create manifests/train.jsonl "
                "and manifests/val.jsonl."
            )

        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.samples: list[dict[str, Any]] = [
                json.loads(line) for line in handle if line.strip()
            ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = self.samples[index]
        image = _load_image(_resolve(sample["image"], self.root), self.image_size)

        depth_paths = sample["depth_layers"][: self.num_layers]
        depth_layers = [
            _load_array(_resolve(path, self.root), self.image_size, self.depth_scale)
            for path in depth_paths
        ]
        while len(depth_layers) < self.num_layers:
            depth_layers.append(torch.zeros_like(depth_layers[0]))
        depth = torch.stack(depth_layers, dim=0)

        mask_paths = sample.get("valid_masks")
        if mask_paths is None:
            valid = depth > 0
        else:
            masks = [
                _load_array(_resolve(path, self.root), self.image_size, is_mask=True)
                for path in mask_paths[: self.num_layers]
            ]
            while len(masks) < self.num_layers:
                masks.append(torch.zeros_like(masks[0]))
            valid = torch.stack(masks, dim=0).bool()

        snapped, snapped_valid = snapped_layered_depth(depth, valid)
        return {
            "image": image,
            "depth": snapped if self.use_snapped_depth else depth,
            "valid": snapped_valid if self.use_snapped_depth else valid,
            "raw_depth": depth,
            "raw_valid": valid,
        }
